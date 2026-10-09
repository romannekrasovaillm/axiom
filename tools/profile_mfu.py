#!/usr/bin/env python3
"""MFU кампании, стадия 2 — прибор атрибуции времени шага (jax.profiler.trace).

Зачем
-----
Стадия 1 (``tools/mfu_bf16_protocol.py``) исчерпала рычаги dtype/flash/ckpt/
allocator/batch: лучшие 2.54 % (dense124m-b4, fp32) / 0.66 % (l3full-b1) MFU,
и все они измерены на train-шаге без лоадера.  Версия узкого места — хост-
диспетчеризация и memory-bound микрооперации графа (SM 96 %, ~1 FLOP/байт).
Этот прибор отвечает на вопрос «куда уходят ~97 % времени шага» **точными
числами трейса**, а не прозой: он исполняет ту же ногу ``net/train_loop.train``
(``net/*`` не трогается) под ``jax.profiler.trace`` и агрегирует
``traceEvents``.

Что делает
----------
1. Разогрев: ``--warmup`` шагов без трейса (прогрев jit/аллокатора; кэш
   компиляции XLA включается, чтобы измеряемое окно не состояло из компиляции).
2. Замер: ``--steps`` шагов под ``jax.profiler.trace`` → каталог ``--trace-dir``.
3. Разбор: ``traceEvents`` (Chrome/Perfetto JSON) → top-25 XLA-операций по
   суммарному времени, доли, эвристическая группировка по имени (attention /
   einsum / scan / collective / reduce / elementwise / copy / other) и **gap** —
   время окна трейса вне исполненных операций (хост-диспетчеризация).
4. Запись ``evidence/mfu-profile/profile-<name>.json`` + печать таблицы в stdout.

Честность чисел (fail-closed)
-----------------------------
Группировка по имени — **эвристика, а не вердикт** (поле ``caveats``): имена
операций разбираются по каноническому префиксу HLO (``einsum.42`` → ``einsum``).
Если данных нет — чисел не выдумываем: пустой/битый каталог трейса даёт
``ProfileError`` и статус ``TRACE-ERROR`` (без top ops), а отсутствие GPU —
``EMPTY-PENDING`` с готовым планом (как в стадии 1).  Прогоны на стенде
выполняет архитектор (AD-7/C-040: перед запуском взять лок на
``~/gb10-shared/.locks``).

Префлайт ADR-041 подключён: ``jax_preflight.ensure_mem_fraction()`` на импорте
модуля (лимит XLA до создания рантайма — этот прибор и есть nsys-путь инцидента
08.10), ``gate_or_exit()`` — в прогонном пути ``main`` до исполнения клетки.
Разбор уже собранного трейса (``--parse-trace``) — проверяющий путь без GPU/jax:
гейта не несёт.

Запуск::

    python3 tools/profile_mfu.py --selftest
    python3 tools/profile_mfu.py --config net/config-dense124m.json --batch 4 \\
        --seq 8192 --steps 6 --warmup 2 --name dense124m-b4 \\
        --out evidence/mfu-profile/profile-dense124m-b4.json \\
        --trace-dir evidence/mfu-profile/trace-dense124m-b4
    # разобрать уже собранный трейс (без GPU/jax):
    python3 tools/profile_mfu.py --parse-trace evidence/mfu-profile/trace-dense124m-b4 \\
        --name dense124m-b4 --out /tmp/profile.json

Пара nsys (сводки по CUDA-ядрам и API): ``tools/profile_mfu_nsys.sh``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

CASE_DIR = Path(__file__).resolve().parent.parent

# The model package lives at the case root; a cell may be launched from any cwd
# (the driver pins ``cwd``, but ``sys.path[0]`` is this script's directory, not
# the cwd), so make the import explicit here.
if str(CASE_DIR) not in sys.path:
    sys.path.insert(0, str(CASE_DIR))

# ADR-041: дисциплина памяти JAX — префлайт ДО создания рантайма.  Лимит
# ``XLA_PYTHON_CLIENT_MEM_FRACTION`` выставляется здесь (до первого ``import
# jax`` внутри ``run_cell`` и до пробы ``gpu_available``, которая наследует тот
# же ``os.environ``), а fail-closed гейт стенда — в прогонном пути ``main``.
_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import jax_preflight  # noqa: E402

jax_preflight.ensure_mem_fraction()

#: Схема отчёта прибора.
REPORT_SCHEMA = "axiom-mfu-profile/1"

#: Версия раскладки отчёта (растёт при изменении полей).
REPORT_VERSION = 1

#: Дефолты окна: короткий прогон, разогрев отдельно (jit/аллокатор).
DEFAULT_STEPS = 6
DEFAULT_WARMUP = 2

#: Сколько операций максимум попадает в отчёт (top-N по суммарному времени).
TOP_N = 25

#: Клетки кампании: (имя, конфиг, батч, seq) — те же, что в ``mfu_bf16_protocol``.
PROFILE_CELLS: tuple[tuple[str, str, int, int], ...] = (
    ("dense124m-b1", "net/config-dense124m.json", 1, 8192),
    ("dense124m-b4", "net/config-dense124m.json", 4, 8192),
    ("l3full-b1", "net/config.json", 1, 8192),
)

#: Режимы dtype/flash (env-гейты фиксируются ДО импорта net/*; см. стадию 1).
MODES: tuple[tuple[str, dict[str, str]], ...] = (
    ("fp32", {"AXIOM_COMPUTE_DTYPE": "fp32", "AXIOM_MLA_DENSE_FLASH": "0"}),
    ("bf16", {"AXIOM_COMPUTE_DTYPE": "bf16", "AXIOM_MLA_DENSE_FLASH": "0"}),
    ("bf16+flash", {"AXIOM_COMPUTE_DTYPE": "bf16", "AXIOM_MLA_DENSE_FLASH": "1"}),
)

#: Эвристика группировки: (группа, ключи в имени).  Порядок = приоритет
#: (``reduce-scatter`` обязан попасть в ``collective`` раньше ``reduce``).
#: Канонический префикс HLO отделяется точкой (``einsum.42``), по нему матчим
#: в первую очередь; полное имя — как запасной матч.
OP_GROUP_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "attention",
        ("attention", "flash-attention", "scaled-dot", "softmax", "sdpa", "mla"),
    ),
    (
        "collective",
        (
            "all-reduce", "allreduce", "reduce-scatter", "reducescatter",
            "all-gather", "allgather", "all-to-all", "alltoall",
            "collective-permute", "collective", "psum", "pmean", "pmax",
            "send", "recv",
        ),
    ),
    ("scan", ("scan", "while-loop", "while", "conditional")),
    ("einsum", ("einsum", "dot", "convolution", "conv", "gemm", "matmul")),
    ("reduce", ("reduce", "argmax", "argmin", "top-k", "topk", "sort")),
    (
        "elementwise",
        (
            "add", "multiply", "mul", "subtract", "divide", "tanh",
            "exponential", "exp", "logistic", "log", "sine", "cosine",
            "negate", "abs", "maximum", "minimum", "compare", "select",
            "clamp", "convert", "rsqrt", "sqrt", "power", "erf", "gelu",
            "relu", "silu", "swish",
        ),
    ),
    (
        "copy",
        (
            "copy", "transpose", "reshape", "slice", "dynamic-slice",
            "dynamic-update-slice", "gather", "scatter", "concatenate",
            "concat", "pad", "reverse", "broadcast", "iota", "bitcast",
            "tuple", "get-tuple-element", "fusion", "custom-call", "call",
            "parameter", "constant", "async",
        ),
    ),
)

#: Времена traceEvents — в микросекундах.
TRACE_TIME_UNIT = 1e-6

#: Объяснение эвристик — уезжает в отчёт как поле ``caveats``.
CAVEATS: tuple[str, ...] = (
    "группировка операций по имени — эвристика (канонический префикс HLO), не вердикт",
    "op_total суммирует ДЫРКИ операций как есть (вложенность может учитываться дважды); "
    "gap считается по объединению интервалов и потому двойного учёта не несёт",
    "окно трейса покрывает только замер (--steps шагов); разогрев --warmup вне окна",
    "gap — верхняя оценка хост-диспетчеризации: включает всё, что не классифицировано как операция",
    "сводки CUDA-ядер/API — отдельный прибор tools/profile_mfu_nsys.sh (nsys), не этот файл",
)


class ProfileError(RuntimeError):
    """Вход или след профиля непригодны — числа не выносятся (fail-closed)."""


# --------------------------------------------------------------------------- #
# Разбор traceEvents (чистые функции; без jax — тестируются на мок-фикстуре)
# --------------------------------------------------------------------------- #


def classify_op(name: str) -> str:
    """Эвристическая группа операции по имени (см. ``OP_GROUP_RULES``).

    Сначала матч по каноническому префиксу HLO (до первой точки), затем по
    полному имени.  Незнакомое имя — ``"other"`` (число не теряется: оно уезжает
    в ``unclassified_seconds``).  Это эвристика, а не вердикт: имя операции —
    свободный текст, и группировка честно помечена полем ``caveats``.
    """
    lowered = str(name).lower()
    head = lowered.split(".", 1)[0]
    for group, keys in OP_GROUP_RULES:
        for key in keys:
            if key in head or key in lowered:
                return group
    return "other"


def is_op(name: str) -> bool:
    """Классифицирована ли операция хоть в какую-то группу (``other`` — нет)."""
    return classify_op(name) != "other"


def iter_trace_files(trace_dir: Path):
    """Все кандидаты-файлы трейса (``*.json`` / ``*.json.gz``) рекурсивно."""
    root = Path(trace_dir)
    if not root.exists():
        raise ProfileError(f"каталог трейса не найден: {root}")
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        name = path.name.lower()
        if name.endswith(".json") or name.endswith(".json.gz"):
            yield path


def _read_json(path: Path) -> Any:
    if path.name.lower().endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    return json.loads(path.read_text(encoding="utf-8"))


def _events_from_payload(payload: Any) -> list:
    """``traceEvents`` из Chrome/Perfetto-полезной нагрузки (или сам список)."""
    if isinstance(payload, dict):
        events = payload.get("traceEvents")
        return events if isinstance(events, list) else []
    return payload if isinstance(payload, list) else []


def load_trace_events(trace_dir: Path) -> tuple[list[dict], dict[str, Any]]:
    """Прочитать ``traceEvents`` из каталога трейса — и только они.

    Возвращает ``(events, meta)``.  Если ни один файл не несёт ``traceEvents``
    (каталог пуст, все файлы битые, или это только ``.xplane.pb``/сводки) — это
    ``ProfileError``: чисел нет, и выдумывать их нечем.  Если ``traceEvents``
    лежат более чем в одном файле (JAX кладёт копию рядом с перфетто-ссылкой),
    берётся **один** источник — файл с наибольшим числом событий; остальные
    попадают в ``meta.duplicate_trace_files`` (а не суммируются: иначе доли
    удвоились бы).
    """
    root = Path(trace_dir)
    seen: list[str] = []
    broken: list[dict[str, str]] = []
    candidates: list[tuple[Path, list]] = []
    for path in iter_trace_files(root):
        seen.append(str(path))
        try:
            payload = _read_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            broken.append({"file": str(path), "error": str(exc)})
            continue
        events = _events_from_payload(payload)
        if events:
            candidates.append((path, events))

    if not candidates:
        raise ProfileError(
            f"в каталоге трейса нет traceEvents: {root} "
            f"(файлов просмотрено {len(seen)}, битых {len(broken)})"
        )

    file_used, events = max(candidates, key=lambda pair: len(pair[1]))
    meta = {
        "files_seen": seen,
        "file_used": str(file_used),
        "event_count": len(events),
        "broken_files": broken,
        "duplicate_trace_files": [str(p) for p, _ in candidates if p != file_used],
    }
    return events, meta


def complete_events(events: list[Any]) -> list[dict[str, Any]]:
    """Нормализовать X-события (``ph == "X"``) в секунды ``{name, cat, ts, dur}``.

    Только длительностные события: у ``B``/``E``/``M``/``i`` нет ``dur`` и их
    вклад в агрегат был бы некорректен.
    """
    out: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict) or event.get("ph") != "X":
            continue
        name = event.get("name")
        ts = event.get("ts")
        dur = event.get("dur")
        if not isinstance(name, str):
            continue
        if not isinstance(ts, (int, float)) or isinstance(ts, bool):
            continue
        if not isinstance(dur, (int, float)) or isinstance(dur, bool):
            continue
        if float(dur) < 0:
            continue
        out.append(
            {
                "name": name,
                "cat": event.get("cat"),
                "ts": float(ts) * TRACE_TIME_UNIT,
                "dur": float(dur) * TRACE_TIME_UNIT,
            }
        )
    return out


def _union_seconds(intervals: list[tuple[float, float]]) -> float:
    """Суммарная длина объединения интервалов (вложенность не удваивается)."""
    if not intervals:
        return 0.0
    total = 0.0
    cur_start, cur_end = intervals[0]
    for start, end in sorted(intervals)[1:]:
        if start > cur_end:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    total += cur_end - cur_start
    return total


def aggregate_ops(
    events: list[dict[str, Any]], *, top_n: int = TOP_N
) -> dict[str, Any]:
    """Top-N операций по суммарному времени + групповой агрегат.

    ``share`` считается от ``op_total_seconds`` (сумма длительностей
    классифицированных операций), поэтому доли top-N суммируются к 1.  Доля от
    окна трейса — отдельное поле ``share_of_window`` (gap виден рядом и не
    размывается).
    """
    by_name: dict[str, dict[str, float]] = defaultdict(lambda: {"total": 0.0, "count": 0.0})
    by_group: dict[str, dict[str, float]] = defaultdict(lambda: {"total": 0.0, "count": 0.0})
    unclassified_total = 0.0
    unclassified_count = 0
    for event in events:
        group = classify_op(event["name"])
        if group == "other":
            unclassified_total += event["dur"]
            unclassified_count += 1
            continue
        by_name[event["name"]]["total"] += event["dur"]
        by_name[event["name"]]["count"] += 1
        by_group[group]["total"] += event["dur"]
        by_group[group]["count"] += 1

    op_total = sum(entry["total"] for entry in by_name.values())
    window = trace_window(events)
    window_seconds = window["window_seconds"]

    def _share(value: float) -> Optional[float]:
        return (value / op_total) if op_total > 0 else None

    top_ops = [
        {
            "name": name,
            "group": classify_op(name),
            "total_seconds": entry["total"],
            "share": _share(entry["total"]),
            "share_of_window": (
                entry["total"] / window_seconds if window_seconds else None
            ),
            "count": int(entry["count"]),
            "mean_seconds": entry["total"] / entry["count"] if entry["count"] else None,
        }
        for name, entry in sorted(
            by_name.items(), key=lambda kv: kv[1]["total"], reverse=True
        )[:top_n]
    ]
    groups = [
        {
            "group": group,
            "total_seconds": entry["total"],
            "share": _share(entry["total"]),
            "count": int(entry["count"]),
        }
        for group, entry in sorted(
            by_group.items(), key=lambda kv: kv[1]["total"], reverse=True
        )
    ]
    return {
        "top_ops": top_ops,
        "groups": groups,
        "op_total_seconds": op_total,
        "op_event_count": sum(int(e["count"]) for e in by_group.values()),
        "unclassified_seconds": unclassified_total,
        "unclassified_count": unclassified_count,
    }


def trace_window(events: list[dict[str, Any]]) -> dict[str, Optional[float]]:
    """Окно трейса: ``[min ts, max ts+dur]`` по всем X-событиям (секунды)."""
    if not events:
        return {"start_seconds": None, "end_seconds": None, "window_seconds": None}
    start = min(event["ts"] for event in events)
    end = max(event["ts"] + event["dur"] for event in events)
    return {
        "start_seconds": start,
        "end_seconds": end,
        "window_seconds": end - start,
    }


def compute_gap(events: list[dict[str, Any]]) -> dict[str, Optional[float]]:
    """Доля времени окна ВНЕ операций (хост-диспетчеризация).

    ``gap = окно − объединение интервалов операций``.  Классифицированный
    элемент — это операция; всё остальное (в том числе ``other``) в покрытие не
    входит, поэтому gap — именно «время вне операций», а не «неучтённое».  Это
    верхняя оценка: она включает и хост-ожидание, и не-операционные регионы.
    """
    window = trace_window(events)
    window_seconds = window["window_seconds"]
    if window_seconds is None:
        return {**window, "covered_seconds": None, "gap_seconds": None, "gap_share": None}
    op_events = [event for event in events if is_op(event["name"])]
    covered = _union_seconds(
        [(event["ts"], event["ts"] + event["dur"]) for event in op_events]
    )
    covered = min(covered, window_seconds)
    gap = max(0.0, window_seconds - covered)
    return {
        "window_seconds": window_seconds,
        "covered_seconds": covered,
        "gap_seconds": gap,
        "gap_share": (gap / window_seconds) if window_seconds > 0 else None,
        "op_events_covering": len(op_events),
    }


def analyze_trace(trace_dir: Path) -> dict[str, Any]:
    """Полный разбор каталога трейса (fail-closed на пустом/битом трейсе)."""
    raw_events, meta = load_trace_events(trace_dir)
    events = complete_events(raw_events)
    if not events:
        raise ProfileError(
            f"в traceEvents нет ни одного длительностного X-события: {trace_dir} "
            f"(событий в файле {len(raw_events)})"
        )
    analysis = aggregate_ops(events)
    return {
        "trace": {**meta, "complete_event_count": len(events)},
        "gap": compute_gap(events),
        "top_ops": analysis["top_ops"],
        "groups": analysis["groups"],
        "op_total_seconds": analysis["op_total_seconds"],
        "op_event_count": analysis["op_event_count"],
        "unclassified_seconds": analysis["unclassified_seconds"],
        "unclassified_count": analysis["unclassified_count"],
    }


# --------------------------------------------------------------------------- #
# Прогон клетки: train_loop.train под jax.profiler.trace
# --------------------------------------------------------------------------- #


def synthetic_batches(batch: int, seq: int, steps: int, vocab: int, seed: int = 0):
    """Детерминированные батчи ``(B, T)`` id-токенов (как в стадии 1).

    Данные синтетические осознанно: прибор меряет ФОРМУ шага, а стоимость
    упакованного лоадера в атрибуцию не входит (и в стадии 1 объявлена вне
    знаменателя MFU).
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    for _ in range(max(0, int(steps))):
        yield rng.integers(0, vocab, size=(batch, seq), dtype=np.int32)


def _enable_compile_cache() -> dict[str, Any]:
    """Включить персистентный кэш компиляции XLA (окно трейса без компиляции).

    Разогрев (``--warmup``) компилирует граф первым; замер — второй прогон
    тех же форм.  Кэш по HLO даёт попадание на втором прогоне, и в окно трейса
    не попадает первая компиляция.  Без кэша окно честно включает компиляцию —
    это видно в ``caveats``/``compile_cache`` отчёта, а не скрыто.
    """
    import tempfile

    cache_dir = Path(tempfile.gettempdir()) / "axiom-profile-mfu-xla-cache"
    try:
        import jax

        cache_dir.mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(cache_dir))
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        return {"enabled": True, "dir": str(cache_dir)}
    except Exception as exc:  # pragma: no cover - зависит от версии jax
        return {"enabled": False, "reason": f"{type(exc).__name__}: {exc}", "dir": str(cache_dir)}


def _trace_context(trace_dir: Path):
    """Контекст-менеджер ``jax.profiler.trace`` (совместим со старыми версиями)."""
    import jax

    try:
        return jax.profiler.trace(str(trace_dir), create_perfetto_link=False)
    except TypeError:  # pragma: no cover - старые версии без kwarg
        return jax.profiler.trace(str(trace_dir))


def run_cell(
    config: str,
    batch: int,
    seq: int,
    *,
    name: str,
    mode: str = "fp32",
    steps: int = DEFAULT_STEPS,
    warmup: int = DEFAULT_WARMUP,
    trace_dir: Path,
    no_trace: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    """Исполнить клетку штатной ногой ``net/train_loop.train`` под трейсом.

    ``net/*`` не трогается: прибор вызывает ровно ``train_loop.train``.
    Разогрев (``warmup`` шагов) исполняется ДО трейса, замер (``steps`` шагов) —
    под ``jax.profiler.trace``.  Возвращает метаданные прогона (без разбора
    трейса — его делает вызывающий, чтобы ``--parse-trace`` не требовал GPU).
    """
    gates = dict(MODES).get(mode)
    if gates is None:
        raise ProfileError(f"неизвестный режим: {mode!r} (есть {[m for m, _ in MODES]})")

    # env-гейты фиксируются ДО импорта net/* (compute_dtype читает их на импорте).
    os.environ.update(gates)
    os.environ.pop("JAX_PLATFORMS", None)

    import jax

    from net import train_loop as tl
    from net.config import load_config

    cfg = load_config(CASE_DIR / config)
    compile_cache = _enable_compile_cache()

    def make_config(run_steps: int) -> "tl.TrainConfig":
        return tl.TrainConfig(
            steps=run_steps,
            total_steps=max(1, run_steps),
            seed=seed,
            micro_batch=batch,
            grad_checkpointing=bool(getattr(cfg, "grad_ckpt_policy", "none") != "none"),
            param_dtype="float32",
            metrics_path=None,  # профиль не пишет метрики: журнал не нужен
            log_every=0,
        )

    budget = tl.Budget(
        run_ref=f"mfu-profile-{name}-{mode}",
        path=Path(trace_dir),
        present=True,
        budget_method="прибор профиля MFU (стадия 2): смета не расходуется",
        stop_rule=f"{warmup}+{steps} шагов прибора",
    )

    started = time.time()
    if warmup > 0:
        tl.train(
            cfg,
            synthetic_batches(batch, seq, warmup, cfg.vocab_size, seed=seed),
            train_config=make_config(warmup),
            budget=budget,
        )

    if no_trace:
        result = tl.train(
            cfg,
            synthetic_batches(batch, seq, steps, cfg.vocab_size, seed=seed),
            train_config=make_config(steps),
            budget=budget,
        )
        return {
            "device": [repr(d) for d in jax.devices()],
            "backend": jax.default_backend(),
            "jax_version": jax.__version__,
            "compile_cache": compile_cache,
            "steps_done": result.steps_done,
            "wall_seconds": time.time() - started,
            "trace": None,
        }

    trace_dir = Path(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    with _trace_context(trace_dir):
        result = tl.train(
            cfg,
            synthetic_batches(batch, seq, steps, cfg.vocab_size, seed=seed),
            train_config=make_config(steps),
            budget=budget,
        )
    return {
        "device": [repr(d) for d in jax.devices()],
        "backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "compile_cache": compile_cache,
        "steps_done": result.steps_done,
        "wall_seconds": time.time() - started,
        "trace_dir": str(trace_dir),
    }


# --------------------------------------------------------------------------- #
# Отчёт
# --------------------------------------------------------------------------- #


def cell_meta(config: str, batch: int, seq: int, *, name: str, mode: str,
              steps: int, warmup: int) -> dict[str, Any]:
    return {
        "name": name,
        "config": config,
        "batch": int(batch),
        "seq": int(seq),
        "mode": mode,
    }


def build_report(
    *,
    cell: dict[str, Any],
    steps: int,
    warmup: int,
    status: str,
    analysis: Optional[dict[str, Any]] = None,
    run_meta: Optional[dict[str, Any]] = None,
    note: Optional[str] = None,
    error: Optional[str] = None,
) -> dict[str, Any]:
    """Собрать отчёт прибора.  Без ``analysis`` чисел профиля нет — и не будет."""
    run_meta = run_meta or {}
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "version": REPORT_VERSION,
        "status": status,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "cell": cell,
        "steps": int(steps),
        "warmup": int(warmup),
        "jax_version": run_meta.get("jax_version"),
        "device": run_meta.get("device", []),
        "backend": run_meta.get("backend"),
        "compile_cache": run_meta.get("compile_cache"),
        "runner": "net/train_loop.train (та же нога, что в претрейне; net/* не меняется)",
        "trace_dir": run_meta.get("trace_dir"),
        "caveats": list(CAVEATS),
    }
    if analysis is not None:
        report.update(
            {
                "trace": analysis["trace"],
                "gap": analysis["gap"],
                "top_ops": analysis["top_ops"],
                "groups": analysis["groups"],
                "op_total_seconds": analysis["op_total_seconds"],
                "op_event_count": analysis["op_event_count"],
                "unclassified_seconds": analysis["unclassified_seconds"],
                "unclassified_count": analysis["unclassified_count"],
            }
        )
    else:
        # Никаких выдуманных чисел: без разбора полей профиля нет вовсе.
        report["gap"] = None
        report["top_ops"] = []
        report["groups"] = []
    if note:
        report["note"] = note
    if error:
        report["error"] = error
    return report


def write_report(report: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


def format_table(report: dict[str, Any]) -> str:
    """Человекочитаемая таблица top-операций и gap (в stdout)."""
    lines: list[str] = []
    cell = report.get("cell", {})
    lines.append(
        f"[profile-mfu] {report.get('status')} · {cell.get('name')} · "
        f"{cell.get('config')} b{cell.get('batch')} seq{cell.get('seq')} "
        f"{cell.get('mode')} · шаги {report.get('steps')} (+{report.get('warmup')} разогрев)"
    )
    gap = report.get("gap") or {}
    if gap.get("window_seconds") is not None:
        lines.append(
            f"  окно {gap['window_seconds'] * 1e3:.1f} мс · операции "
            f"{gap['covered_seconds'] * 1e3:.1f} мс · gap "
            f"{gap['gap_seconds'] * 1e3:.1f} мс ({_pct(gap.get('gap_share'))})"
        )
    top_ops = report.get("top_ops") or []
    if top_ops:
        lines.append(f"  {'операция':<44} {'группа':<12} {'время, мс':>10} {'доля':>7}")
        for op in top_ops:
            lines.append(
                f"  {op['name'][:44]:<44} {op['group']:<12} "
                f"{op['total_seconds'] * 1e3:>10.2f} {_pct(op.get('share')):>7}"
            )
    elif report.get("status") == "EMPTY-PENDING":
        lines.append("  (чисел нет: прогон не выполнен)")
    return "\n".join(lines)


def _pct(value: Optional[float]) -> str:
    return f"{value * 100:.1f}%" if value is not None else "—"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def plan() -> list[dict[str, Any]]:
    """План клеток прибора (те же три формы, что в стадии 1)."""
    return [
        {"name": name, "config": config, "batch": batch, "seq": seq}
        for name, config, batch, seq in PROFILE_CELLS
    ]


def gpu_available() -> bool:
    """Есть ли исполнимый GPU (без падения при отсутствии) — зонд стадии 1."""
    env = dict(os.environ)
    env.pop("JAX_PLATFORMS", None)
    probe = (
        "import jax\n"
        "reps = [repr(d).lower() for d in jax.devices()]\n"
        "try:\n"
        "    backend = jax.default_backend()\n"
        "except Exception:\n"
        "    backend = ''\n"
        "hit = backend == 'gpu' or any('gpu' in r or 'cuda' in r for r in reps)\n"
        "print('gpu' if hit else 'cpu')\n"
    )
    try:
        out = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, timeout=300, env=env
        )
    except Exception:
        return False
    return out.returncode == 0 and out.stdout.strip().endswith("gpu")


def selftest() -> int:
    checks: list[tuple[str, bool]] = []

    # --- классификация операций (эвристика) ---
    checks.append(("einsum.42 → einsum", classify_op("einsum.42") == "einsum"))
    checks.append(("all-reduce.1 → collective", classify_op("all-reduce.1") == "collective"))
    checks.append(("reduce-scatter.0 → collective (не reduce)", classify_op("reduce-scatter.0") == "collective"))
    checks.append(("reduce.3 → reduce", classify_op("reduce.3") == "reduce"))
    checks.append(("copy.7 → copy", classify_op("copy.7") == "copy"))
    checks.append(("fusion.9 → copy", classify_op("fusion.9") == "copy"))
    checks.append(("jit-step → other", classify_op("jit-step") == "other"))
    checks.append(("is_op(einsum)=True", is_op("einsum.1") is True))
    checks.append(("is_op(jit-step)=False", is_op("jit-step") is False))

    # --- агрегация на синтетике (детерминированные длительности) ---
    # ts/dur в микросекундах: einsum 2×10, reduce 1×5, copy 1×15 → op_total 40.
    sample = complete_events(
        [
            {"ph": "X", "name": "einsum.1", "ts": 0, "dur": 10},
            {"ph": "X", "name": "einsum.1", "ts": 10, "dur": 10},
            {"ph": "X", "name": "reduce.2", "ts": 20, "dur": 5},
            {"ph": "X", "name": "copy.3", "ts": 25, "dur": 15},
            {"ph": "X", "name": "jit-step", "ts": 0, "dur": 100},  # other, в op не идёт
            {"ph": "B", "name": "einsum.1", "ts": 0},  # не X → мимо
        ]
    )
    checks.append(("X-события нормализованы (5 из 6)", len(sample) == 5))
    analysis = aggregate_ops(sample)
    checks.append(("op_total = 40 мкс", abs(analysis["op_total_seconds"] - 40e-6) < 1e-12))
    checks.append(("top-1 — einsum.1, 20 мкс", analysis["top_ops"][0]["name"] == "einsum.1"
                   and abs(analysis["top_ops"][0]["total_seconds"] - 20e-6) < 1e-12))
    checks.append(("доля einsum = 0.5", abs(analysis["top_ops"][0]["share"] - 0.5) < 1e-12))
    checks.append(("count einsum = 2", analysis["top_ops"][0]["count"] == 2))
    checks.append(("unclassified = jit-step 100 мкс",
                   abs(analysis["unclassified_seconds"] - 100e-6) < 1e-12
                   and analysis["unclassified_count"] == 1))

    # --- gap: окно 100 мкс (jit-step), покрытие операций 40 мкс → gap 60 мкс ---
    gap = compute_gap(sample)
    checks.append(("окно трейса = 100 мкс", abs(gap["window_seconds"] - 100e-6) < 1e-12))
    checks.append(("покрытие операций = 40 мкс", abs(gap["covered_seconds"] - 40e-6) < 1e-12))
    checks.append(("gap = 60 мкс (0.6)", abs(gap["gap_seconds"] - 60e-6) < 1e-12
                   and abs(gap["gap_share"] - 0.6) < 1e-12))

    # --- вложенность не удваивается объединением интервалов ---
    nested = complete_events(
        [
            {"ph": "X", "name": "einsum.1", "ts": 0, "dur": 100},
            {"ph": "X", "name": "reduce.2", "ts": 10, "dur": 20},  # внутри einsum
        ]
    )
    checks.append(("union не удваивает вложенность", abs(compute_gap(nested)["covered_seconds"] - 100e-6) < 1e-12))

    # --- fail-closed: нет данных — нет чисел ---
    import tempfile

    empty = Path(tempfile.mkdtemp()) / "trace-less"
    empty.mkdir()
    (empty / "summary.json").write_text(json.dumps({"other": 1}), encoding="utf-8")
    raised = False
    try:
        load_trace_events(empty)
    except ProfileError:
        raised = True
    checks.append(("пустой трейс (без traceEvents) → ProfileError", raised))

    broken = Path(tempfile.mkdtemp())
    (broken / "trace.json").write_text("{не json", encoding="utf-8")
    raised = False
    try:
        load_trace_events(broken)
    except ProfileError:
        raised = True
    checks.append(("битый трейс → ProfileError", raised))

    # --- дедупликация источников: берём файл с наибольшим числом событий ---
    dup = Path(tempfile.mkdtemp())
    (dup / "a.json").write_text(json.dumps({"traceEvents": [{"ph": "X", "name": "copy.1", "ts": 0, "dur": 1}]}), encoding="utf-8")
    (dup / "b.json").write_text(
        json.dumps({"traceEvents": [
            {"ph": "X", "name": "einsum.1", "ts": 0, "dur": 2},
            {"ph": "X", "name": "reduce.1", "ts": 2, "dur": 3},
        ]}),
        encoding="utf-8",
    )
    events, meta = load_trace_events(dup)
    checks.append(("берётся файл с наибольшим traceEvents (2)", len(events) == 2
                   and meta["file_used"].endswith("b.json")
                   and len(meta["duplicate_trace_files"]) == 1))

    # --- отчёт без анализа не несёт чисел ---
    report = build_report(
        cell=cell_meta("net/config-dense124m.json", 4, 8192, name="dense124m-b4", mode="fp32", steps=6, warmup=2),
        steps=6, warmup=2, status="EMPTY-PENDING",
    )
    checks.append(("EMPTY-PENDING без чисел", report["gap"] is None and report["top_ops"] == []))
    checks.append(("caveats несут пометку 'эвристика'", any("эвристика" in c for c in report["caveats"])))

    # --- план клеток ---
    checks.append(("план = 3 клетки кампании", len(plan()) == 3))

    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'} {name}")
    if failed:
        print(f"FAIL: {len(failed)} из {len(checks)}", file=sys.stderr)
        return 1
    print(f"PASS: {len(checks)} проверок")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="net/config-dense124m.json")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--mode", choices=[m for m, _ in MODES], default="fp32")
    parser.add_argument("--name", default="dense124m-b1")
    parser.add_argument("--out", default=None,
                        help="путь отчёта (дефолт evidence/mfu-profile/profile-<name>.json)")
    parser.add_argument("--trace-dir", default=None,
                        help="каталог jax.profiler.trace (дефолт evidence/mfu-profile/trace-<name>)")
    parser.add_argument("--no-trace", action="store_true",
                        help="не включать jax.profiler.trace (прогон под внешним профилировщиком, nsys)")
    parser.add_argument("--parse-trace", default=None,
                        help="разобрать уже собранный каталог трейса (без GPU/jax)")
    parser.add_argument("--plan", action="store_true",
                        help="напечатать план клеток (TSV) и выйти")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args(argv)

    if args.selftest:
        return selftest()

    if args.plan:
        for cell in plan():
            print(f"{cell['name']}\t{cell['config']}\t{cell['batch']}\t{cell['seq']}")
        return 0

    out_path = Path(args.out) if args.out else (
        CASE_DIR / "evidence" / "mfu-profile" / f"profile-{args.name}.json"
    )
    trace_dir = Path(args.trace_dir) if args.trace_dir else (
        CASE_DIR / "evidence" / "mfu-profile" / f"trace-{args.name}"
    )
    cell = cell_meta(args.config, args.batch, args.seq, name=args.name,
                     mode=args.mode, steps=args.steps, warmup=args.warmup)

    # --- разбор уже собранного трейса: GPU/jax не нужны -----------------------
    if args.parse_trace is not None:
        try:
            analysis = analyze_trace(Path(args.parse_trace))
        except ProfileError as exc:
            report = build_report(cell=cell, steps=args.steps, warmup=args.warmup,
                                  status="TRACE-ERROR", error=str(exc))
            write_report(report, out_path)
            print(f"[profile-mfu] TRACE-ERROR: {exc} → {out_path}", file=sys.stderr)
            return 2
        report = build_report(cell=cell, steps=args.steps, warmup=args.warmup,
                              status="COMPLETE", analysis=analysis,
                              run_meta={"trace_dir": str(args.parse_trace)})
        write_report(report, out_path)
        print(format_table(report))
        print(f"[profile-mfu] COMPLETE → {out_path}")
        return 0

    # --- прогон ---------------------------------------------------------------
    if not gpu_available() and not args.allow_cpu:
        report = build_report(cell=cell, steps=args.steps, warmup=args.warmup,
                              status="EMPTY-PENDING")
        report["note"] = (
            "GPU недоступен: клетка не исполнена. Отчёт несёт план (cell/steps/warmup); "
            "прогон выполняет архитектор на стенде GB10 (AD-7/C-040: лок на "
            "~/gb10-shared/.locks)."
        )
        write_report(report, out_path)
        print(f"[profile-mfu] EMPTY-PENDING (GPU нет) → {out_path}")
        return 0

    jax_preflight.gate_or_exit()  # ADR-041: состояние стенда до реального прогона
    try:
        run_meta = run_cell(
            args.config, args.batch, args.seq, name=args.name, mode=args.mode,
            steps=args.steps, warmup=args.warmup, trace_dir=trace_dir,
            no_trace=args.no_trace,
        )
    except ProfileError as exc:
        report = build_report(cell=cell, steps=args.steps, warmup=args.warmup,
                              status="TRACE-ERROR", error=str(exc))
        write_report(report, out_path)
        print(f"[profile-mfu] TRACE-ERROR: {exc} → {out_path}", file=sys.stderr)
        return 2

    if args.no_trace:
        report = build_report(cell=cell, steps=args.steps, warmup=args.warmup,
                              status="COMPLETE", run_meta=run_meta,
                              note="jax.profiler.trace отключён (--no-trace): прибор работает "
                                   "под внешним профилировщиком (tools/profile_mfu_nsys.sh).")
        write_report(report, out_path)
        print(format_table(report))
        print(f"[profile-mfu] COMPLETE (без трейса) → {out_path}")
        return 0

    try:
        analysis = analyze_trace(trace_dir)
    except ProfileError as exc:
        report = build_report(cell=cell, steps=args.steps, warmup=args.warmup,
                              status="TRACE-ERROR", run_meta=run_meta, error=str(exc))
        write_report(report, out_path)
        print(f"[profile-mfu] TRACE-ERROR: {exc} → {out_path}", file=sys.stderr)
        return 2

    report = build_report(cell=cell, steps=args.steps, warmup=args.warmup,
                          status="COMPLETE", analysis=analysis, run_meta=run_meta)
    write_report(report, out_path)
    print(format_table(report))
    print(f"[profile-mfu] COMPLETE → {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
