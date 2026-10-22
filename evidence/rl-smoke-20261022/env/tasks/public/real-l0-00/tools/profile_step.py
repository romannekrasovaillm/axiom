#!/usr/bin/env python3
"""Прибор профиля train-шага под форму ``chunked_cc`` — раскладка времени по компонентам.

Зачем
-----
ADR-047 п. 3 запрещает оптимизировать *предполагаемое*: форма ``chunked_cc``
ускорила шаг l3-full с 82.7–85.3 с до 31.5–33.0 с (260.2 ток/с против 99.0), а
вклад KDA оценён «18 слоёв × ~0.345 с ≈ 6 с из ~31.5 с, то есть ~20 %». Оценка
получена умножением замера одного слоя на число слоёв — это не раскладка шага, а
экстраполяция. Прибор отвечает на вопрос «куда уходят остальные ~80 %» **числами
самого шага**: он исполняет ту же ногу ``net/train_loop.train`` под
``jax.profiler.trace`` и агрегирует ``traceEvents``.

Что делает
----------
1. Разогрев: ``--warmup`` шагов без трейса (jit/аллокатор; компиляция уходит за
   пределы измеряемого окна — это видно в отчёте, а не подразумевается).
2. Замер: ``--steps`` шагов под ``jax.profiler.trace`` → каталог ``--trace-dir``.
3. Разбор (без jax, тестируется на фикстурах):
   * top-25 XLA-операций по суммарному device-времени;
   * **раскладка по категориям** — KDA / MLA-внимание / MoE-FFN / LM-голова /
     optimizer-apply / collective / GEMM-einsum / elementwise / memcpy-memset /
     прочие операции, плюс **gap** (время окна вне операций);
   * **по шагам** — доля первого шага (компиляция) отдельным полем;
   * сверка раскладки с длительностью шага и явные находки.
4. Запись ``evidence/kda-rewrite/step-profile.json`` + читаемая таблица в stdout.

Два уровня раскладки (честность чисел, а не одна цифра)
------------------------------------------------------
Требование «сумма категорий сходится с длительностью шага» механизировано на
**двух** уровнях, потому что это разные утверждения:

* **внутренняя сходимость** — ``сумма(категории) + перекрытие + gap == окно``.
  Каждая операция относится ровно к одной категории, а интервалы категории
  считаются **объединением**, а не суммой длительностей: вложенные и
  перекрывающиеся интервалы не удваиваются. Величина перекрытия выводится
  отдельным полем, а не растворяется в категориях.
* **внешняя сходимость (порог 95 %)** — ``окно трейса ÷ длительность шага`` по
  стенным часам (медиана ``step_seconds`` из ``metrics.jsonl``). Порог осмыслен
  именно здесь: он отвечает, покрывает ли трейс весь шаг, а не только его
  device-часть. Провал — находка ``step_not_covered``.

``gap`` — это **время вне операций** (хост-диспетчеризация, ожидание), а не
«прочее»: при доле > 40 % окна выводится отдельная находка ``gap_dominates``.

Происхождение категории (эвристика названа эвристикой)
-----------------------------------------------------
Имена XLA-операций не несут семантики слоя: ``fusion.123`` не говорит, KDA это
или MoE. Поэтому у каждой категории считается **происхождение** — доля секунд,
отнесённых структурным свидетельством (``by_module`` — из ``args`` события,
``by_region`` — из позиции в графе, ``by_shape`` — из подписей форм) против
доли, отнесённой **эвристикой по имени** (``by_name``). Отчёт печатает
``structural_share``: при низком значении семантическая раскладка — гипотеза, и
это видно в поле, а не в прозе. Опорная нога (``phase_legs``) меряет KDA / MLA /
MoE / CE / шаг оптимизатора **изолированными проходами** и служит перекрёстной
проверкой, но её сумма НЕ выдаётся за время шага (проходы не интерливаются).

Запуск::

    python3 tools/profile_step.py --selftest
    python3 tools/profile_step.py --plan
    python3 tools/profile_step.py \\
        --config net/config.json --impl chunked_cc --batch 1 --seq 8192 \\
        --steps 4 --warmup 2 --name l3full-chunked-cc \\
        --out evidence/kda-rewrite/step-profile.json \\
        --trace-dir evidence/kda-rewrite/trace-l3full-chunked-cc
    # разобрать уже собранный трейс (без GPU/jax):
    python3 tools/profile_step.py --parse-trace <каталог> --out /tmp/step-profile.json

Прогон заодно пишет ``metrics.jsonl`` рядом с каталогом трейса (там же, где
``train_loop`` пишет KPI): это носитель стенного времени шага и полей ``sec_*``
опорной ноги.  Без него внешняя сходимость не считается — порог не выдумывается.

Без CUDA-устройства прибор **fail-closed**: статус ``EMPTY-PENDING`` с планом и
причиной, числа не имитируются. ``--allow-cpu`` честно исполняет малую геометрию
на CPU и помечает отчёт ``SMOKE-CPU`` (провод прибора, не число).

Границы прибора
---------------
* ``net/*`` не трогается: прибор вызывает ровно ``train_loop.train``; режим
  ``chunked_cc`` приходит переопределением конфига **в памяти**
  (``dataclasses.replace``), дефолт ``kda_impl`` в ``net/config.json`` не меняется.
* ``scan_layers=true``: 24 слоя l3-full исполняются одним ``lax.scan``, поэтому
  «слой → время» из трейса **не выводится**. Отсюда опорная нога: семантическая
  раскладка в категориях — эвристика по именам, а не разметка скан-тела.
* Данные синтетические: прибор меряет **форму** шага; стоимость упакованного
  лоадера в раскладку не входит (та же граница, что в стадии 2-бис ``profile_mfu``).
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ``net`` и ``tools`` живут в корне репозитория; прибор запускают из любого cwd.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "tools"))

#: Схема отчёта прибора.
REPORT_SCHEMA = "axiom-step-profile/1"

#: Версия раскладки отчёта (растёт при изменении полей).
REPORT_VERSION = 1

#: Канонические пути дельты ADR-047.
DEFAULT_OUT = _REPO_ROOT / "evidence" / "kda-rewrite" / "step-profile.json"
DEFAULT_TRACE_DIR = _REPO_ROOT / "evidence" / "kda-rewrite" / "trace-l3full-chunked-cc"

#: Дефолты окна (ADR-047 п. 6: 4 шага, T=8192, B=1, grad-checkpointing).
DEFAULT_STEPS = 4
DEFAULT_WARMUP = 2

#: Профилируемая клетка: конфиг l3-full, форма ``chunked_cc``.
DEFAULT_CONFIG = "net/config.json"
DEFAULT_IMPL = "chunked_cc"
DEFAULT_BATCH = 1
DEFAULT_SEQ = 8192

#: Сколько операций попадает в отчёт (top-N по суммарному device-времени).
TOP_N = 25

#: Порог внешней сходимости: окно трейса против длительности шага.
RECONCILE_THRESHOLD = 0.95

#: Порог находки «gap доминирует» (доля окна вне операций).
GAP_FINDING_THRESHOLD = 0.40

#: Порог находки «имя-эвристика доминирует» (структурных свидетельств мало).
STRUCTURAL_FINDING_THRESHOLD = 0.50

#: Порог находки «нераспознанных операций много».
UNCLASSIFIED_FINDING_THRESHOLD = 0.15

#: Времена traceEvents — в микросекундах.
TRACE_TIME_UNIT = 1e-6

#: Допуск сравнения интервалов, с (трассы округляют микросекунды).
_EPS = 1e-12

#: Категории раскладки в порядке вывода.  Первые пять — семантические
#: (запрошены постановкой как «KDA/MLA/MoE/LM-голова/optimizer»), остальные —
#: механические. ``gemm_einsum`` и ``other_ops`` выходят за буквальный список
#: постановки, но без них раскладка не полна (крупные GEMM и незнакомые имена
#: иначе растворились бы в elementwise), поэтому они заявлены явно.
CATEGORY_ORDER: tuple[str, ...] = (
    "kda",
    "mla_attention",
    "moe_ffn",
    "lm_head",
    "optimizer_apply",
    "collective",
    "gemm_einsum",
    "elementwise",
    "memcpy_memset",
    "other_ops",
)

#: Правила классификации: ``(категория, ключи, происхождение)``, порядок = приоритет.
#:
#: Матч идёт (а) по ``args.hlo_module``/``args.module`` события, если он есть —
#: это структурное свидетельство ``by_module``; (б) по каноническому префиксу
#: HLO-имени (до первой точки: ``einsum.42`` → ``einsum``) и полному имени — это
#: эвристика ``by_name``. Коллективы и memcpy стоят ВЫШЕ elementwise не случайно:
#: ``reduce-scatter`` обязан попасть в ``collective`` раньше, чем в ``reduce``.
CATEGORY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "optimizer_apply",
        (
            "optimizer", "optax", "adam", "adamw", "muon", "scale_by", "scale-",
            "bias_correction", "bias-correction", "apply_update", "apply-update",
            "update_moment", "moment", "weight_decay", "weight-decay", "sgd",
            "learning_rate", "learning-rate", "master", "ema",
        ),
    ),
    (
        "lm_head",
        (
            "cross_entropy", "cross-entropy", "log_softmax", "log-softmax",
            "logits", "ce_chunk", "ce-chunk", "chunked_ce", "ntp_head", "head",
            "embedding", "token_embed", "vocab", "mtp", "nll", "softmax_cross",
        ),
    ),
    (
        "moe_ffn",
        (
            "moe", "expert", "router", "routing", "top_k", "top-k", "topk",
            "dispatch", "combine", "capacity", "situ", "glu", "swiglu", "geglu",
            "ffn", "mlp", "intermediate", "gate_up", "gate-up", "down_proj",
            "qb_", "load_balance", "load-balance", "z_loss",
        ),
    ),
    (
        "kda",
        (
            "kda", "delta_rule", "delta-rule", "delta_step", "wyut", "cc_tile",
            "cc-tile", "cc_scores", "cc-scores", "associative", "decay", "decay_ratio",
            "decay-ratio", "short_conv", "short-conv", "shortconv", "beta",
            "forget", "chunk_state", "chunk-state", "state_pass", "state-pass",
            "triangular", "tril", "triu",
        ),
    ),
    (
        "mla_attention",
        (
            "mla", "attention", "flash", "sdpa", "fa_", "fa-", "rope", "rotary",
            "indexer", "index_head", "index-head", "latent", "kv_pool", "kv-pool",
            "block_merge", "block-merge", "swa", "window", "sliding", "causal",
            "mask", "pool",
        ),
    ),
    (
        "collective",
        (
            "all-reduce", "allreduce", "reduce-scatter", "reducescatter",
            "all-gather", "allgather", "all-to-all", "alltoall",
            "collective-permute", "collective_permute", "collective", "psum",
            "pmean", "pmax", "pmin", "send", "recv", "shard",
        ),
    ),
    (
        "memcpy_memset",
        (
            "memcpy", "memcpy-d2d", "memcpy_d2d", "memcpy-d2h", "memcpy-h2d",
            "memset", "async-done", "async_done", "async-start", "async_start",
            "copy-done", "copy_done", "copy-start", "copy_start", "dma",
        ),
    ),
    (
        "gemm_einsum",
        ("einsum", "dot", "convolution", "conv", "gemm", "matmul", "cublas",
         "cusolver", "triton"),
    ),
    (
        "elementwise",
        (
            "add", "multiply", "mul", "subtract", "sub", "divide", "div",
            "tanh", "exponential", "exp", "logistic", "log", "sine", "cosine",
            "negate", "abs", "maximum", "minimum", "max", "min", "compare",
            "select", "where", "clamp", "convert", "rsqrt", "sqrt", "power",
            "erf", "gelu", "relu", "silu", "swish", "sigmoid", "rsqrt",
            "reduce", "argmax", "argmin", "sort", "cumsum", "cumprod",
            "broadcast", "iota", "bitcast", "reshape", "transpose", "slice",
            "gather", "scatter", "concatenate", "concat", "pad", "reverse",
            "copy", "fusion", "custom-call", "custom_call", "call", "tuple",
            "get-tuple-element", "parameter", "constant", "while", "scan",
            "conditional", "reduce-window", "reduce_window", "dynamic-slice",
            "dynamic-update-slice", "dynamic_slice", "dynamic_update_slice",
            "clz", "popcnt", "shift", "and", "or", "xor", "not", "real",
            "imag", "complex", "fft", "rng", "random",
        ),
    ),
)

#: Объяснение эвристик — уезжает в отчёт полем ``caveats``.
CAVEATS: tuple[str, ...] = (
    "раскладка по категориям частично эвристична (имена XLA-операций не несут "
    "семантики слоя); доля структурных свидетельств выведена полем "
    "structural_share — при низком значении семантические категории гипотеза, "
    "а не разметка",
    "scan_layers=true: 24 слоя l3-full исполняются одним lax.scan, поэтому "
    "«слой → секунды» из трейса не выводится; категории kda/mla/moe получены "
    "по именам и опорной ноге, не по разметке скан-тела",
    "интервалы категории считаются ОБЪЕДИНЕНИЕМ, а не суммой длительностей: "
    "вложенность и перекрытие не удваиваются; величина перекрытия выведена "
    "полем overlap_seconds",
    "gap — время окна ВНЕ операций (хост-диспетчеризация, ожидание, device-простой), "
    "а не «прочее»: это верхняя оценка неучтённого",
    "внешняя сходимость (порог 95 %) считается против стенного времени шага из "
    "metrics.jsonl; если metrics не собраны, поле пустое — порог не выдумывается",
    "опорные ноги phase_legs (kda/mla/moe/ce/backopt) — дополнительные "
    "декомпозированные проходы, они НЕ интерливаются как в forward, поэтому их "
    "сумма не равна времени шага и не выдаётся за него",
    "опорные ноги компилируют 4 дополнительные jit-функции: их компиляция "
    "(хостовое время) ложится в ПЕРВЫЙ измеренный шаг — это видно в полях "
    "first_step_share/compile_share; device-раскладка ею не затронута, а внешняя "
    "сходимость считается по МЕДИАНЕ шагов и потому устойчива; --no-phase-legs "
    "снимает эффект",
    "данные синтетические: прибор меряет форму шага, стоимость упакованного "
    "лоадера в раскладку не входит",
)

#: Регексп-маркер шага по умолчанию: имена jitted-функций шага в хостовой части трейса.
DEFAULT_STEP_MARKER = r"(?i)(value_and_grad|loss_fn|grad_fn|train_step|jit_step|step_fn)"

#: Категории ``cat``, которыми трейсер помечает ХОСТОВЫЕ события (кадры Python,
#: диспетчеризация jit, вызовы CUDA API).  Их время — не device-время: считать их
#: операциями значило бы удвоить шаг хост-кадрами (проверено на макете: ``jit_loss_fn``
#: накрывает все device-операции и «съедает» таблицу top-25).
HOST_CATS: frozenset[str] = frozenset(
    {
        "python_function", "python", "jit", "user_annotation", "trace",
        "gpu_user_annotation", "xla modules", "cuda_runtime", "cuda_api",
        "cuda", "host", "cpu_op", "annotation", "gpu_memcpy",
    }
)

#: Подстроки ``cat``, однозначно указывающие на device-событие (XLA-операция/ядро).
DEVICE_CAT_HINTS: tuple[str, ...] = ("xla op", "xla_ops", "xla_ops", "kernel", "device")

#: Канонический вид имени HLO-инструкции: ``fusion.123``, ``memcpy-d2d.1``, ``while.0``.
#: Имена хостовых кадров так не выглядят (``jit_loss_fn``, ``train``) — по этому
#: признаку device-операции отличаются от хост-кадров даже без ``cat``.
HLO_INSTRUCTION_RE = re.compile(r"^[A-Za-z_][\w\-]*\.\d+$")

#: Порог находки «слишком много событий неопознанного происхождения».
UNKNOWN_FINDING_THRESHOLD = 0.05


class ProfileError(RuntimeError):
    """Вход или след профиля непригодны — числа не выносятся (fail-closed)."""


# --------------------------------------------------------------------------- #
# Классификация и разбор traceEvents (чистые функции; без jax)
# --------------------------------------------------------------------------- #


def _canonical_head(name: str) -> str:
    """Канонический префикс HLO-имени (``einsum.42`` → ``einsum``)."""
    return str(name).lower().split(".", 1)[0]


def classify_category(name: str, *, module: str | None = None) -> tuple[str, str]:
    """Категория операции и её происхождение.

    Возвращает ``(category, provenance)``.  Порядок матча: ``module`` (структурное
    свидетельство ``by_module``) → канонический префикс имени → полное имя
    (эвристика ``by_name``).  Незнакомая операция — ``("other_ops", "by_name")``:
    число не теряется, оно уезжает в категорию ``other_ops``.

    Это эвристика, а не вердикт: имя операции — свободный текст, поэтому
    происхождение возвращается рядом с категорией, чтобы вызывающий мог посчитать
    ``structural_share``, а не поверить раскладке на слово.
    """
    if module:
        lowered_module = str(module).lower()
        for category, keys in CATEGORY_RULES:
            for key in keys:
                if key in lowered_module:
                    return category, "by_module"

    lowered = str(name).lower()
    head = _canonical_head(lowered)
    for category, keys in CATEGORY_RULES:
        for key in keys:
            if key in head or key in lowered:
                return category, "by_name"
    return "other_ops", "by_name"


def is_classified(name: str, *, module: str | None = None) -> bool:
    """Отнесена ли операция к содержательной категории (``other_ops`` — нет)."""
    return classify_category(name, module=module)[0] != "other_ops"


def iter_trace_files(trace_dir: Path):
    """Кандидаты-файлы трейса (``*.json`` / ``*.json.gz``) рекурсивно."""
    root = Path(trace_dir)
    if not root.exists():
        raise ProfileError(f"каталог трейса не найден: {root}")
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name.lower().endswith((".json", ".json.gz")):
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

    Если ни один файл не несёт ``traceEvents`` (каталог пуст, файлы битые, или
    это только ``.xplane.pb``/сводки) — ``ProfileError``: чисел нет, и выдумывать
    их нечем.  Если ``traceEvents`` лежат более чем в одном файле (JAX кладёт
    копию рядом с перфетто-ссылкой), берётся **один** источник — файл с
    наибольшим числом событий; остальные попадают в ``meta.duplicate_trace_files``
    (а не суммируются: иначе доли удвоились бы).
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
            broken.append({"file": str(path), "error": f"{type(exc).__name__}: {exc}"})
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


def _module_of(event: dict) -> Optional[str]:
    """Структурное свидетельство из ``args`` события, если оно там есть.

    JAX/XLA кладут в ``args`` разное в зависимости от версии: ``hlo_module``,
    ``module``, ``tf_op``, ``producer``.  Берётся первое непустое строковое.
    Отсутствие — законный случай: тогда категория решается эвристикой по имени.
    """
    args = event.get("args")
    if not isinstance(args, dict):
        return None
    for key in ("hlo_module", "hloModule", "module", "producer", "tf_op"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def complete_events(events: list[Any]) -> list[dict[str, Any]]:
    """Нормализовать X-события (``ph == "X"``) в секунды ``{name, cat, module, ts, dur}``.

    Только длительностные события: у ``B``/``E``/``M``/``i`` нет ``dur``, и их
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
                "module": _module_of(event),
                "pid": event.get("pid"),
                "tid": event.get("tid"),
                "ts": float(ts) * TRACE_TIME_UNIT,
                "dur": float(dur) * TRACE_TIME_UNIT,
            }
        )
    return out


def _track_of(event: dict[str, Any]) -> tuple[Any, Any]:
    """Дорожка события ``(pid, tid)`` — граница, в пределах которой возможно вложение.

    Вложение проверяется ТОЛЬКО внутри одной дорожки: операции разных потоков и
    устройств могут накладываться по времени, и геометрическая вложенность между
    ними была бы ложной — крупная операция молча пропала бы из раскладки.  Трассы
    без ``pid``/``tid`` (узкие фикстуры) дают одну дорожку ``(None, None)``.
    """
    return (event.get("pid"), event.get("tid"))


def classify_origin(event: dict[str, Any]) -> str:
    """Происхождение X-события: ``"device"`` (операция на устройстве), ``"host"``
    (кадр Python/jit/CUDA API) или ``"unknown"``.

    Прибор раскладывает **device-время**: хостовые кадры накрывают устройство
    целиком и в таблице top-25 затопили бы настоящие ядра.  Признаки, по порядку:

    1. ``cat`` из :data:`HOST_CATS` → host;
    2. ``args`` несёт ``hlo_op``/``hlo_module`` → device;
    3. имя канонического вида HLO-инструкции (:data:`HLO_INSTRUCTION_RE`) → device;
    4. ``cat`` содержит подсказку устройства → device;
    5. иначе ``unknown`` — событие не выбрасывается молча: оно считается и
       выводится, а его доля выше порога даёт находку ``unknown_origin_high``
       (значит, таблицы признаков надо расширить под конкретную версию трейсера).
    """
    cat = str(event.get("cat") or "").strip().lower()
    if cat in HOST_CATS:
        return "host"
    args = event.get("args")
    if isinstance(args, dict) and any(
        key in args for key in ("hlo_op", "hlo_module", "hlo_op_name", "hlo_instruction")
    ):
        return "device"
    if HLO_INSTRUCTION_RE.match(str(event.get("name") or "")):
        return "device"
    if any(hint in cat for hint in DEVICE_CAT_HINTS):
        return "device"
    return "unknown"


def split_by_origin(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Разделить X-события по происхождению (ключи ``device``/``host``/``unknown``)."""
    buckets: dict[str, list[dict[str, Any]]] = {"device": [], "host": [], "unknown": []}
    for event in events:
        buckets[classify_origin(event)].append(event)
    return buckets


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


def mark_leaves(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Пометить события-листья: те, что не содержат внутри себя других событий.

    Раскладку строим по листьям, иначе вложенные интервалы (модуль содержит
    инструкции) учлись бы дважды.  Событие считается ребёнком, если оно вложено
    в другое **на той же дорожке** (``pid``/``tid``): наложение интервалов разных
    потоков — это перекрытие, а не вложение, и принимать его за вложение значило
    бы молча потерять крупную операцию.

    Сортировка ``(ts, -dur)`` гарантирует, что контейнер открывается раньше
    содержимого, поэтому достаточно одного прохода со стеком открытых
    интервалов — O(n log n) на трассах в сотни тысяч событий.
    """
    ordered = sorted(events, key=lambda event: (event["ts"], -event["dur"]))
    marked: list[dict[str, Any]] = []
    stacks: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for event in ordered:
        track = _track_of(event)
        stack = stacks[track]
        while stack and stack[-1]["ts"] + stack[-1]["dur"] <= event["ts"] + _EPS:
            stack.pop()
        entry = dict(event)
        entry["has_child"] = False
        parent = stack[-1] if stack else None
        # Совпадение границ (та же ts и длительность) — это ДУБЛЬ события, а не
        # вложенность: иначе один из дублей пропал бы из листьев, и его секунды
        # потерялись бы молча.
        identical = bool(
            parent is not None
            and event["ts"] == parent["ts"]
            and event["dur"] == parent["dur"]
        )
        entry["inside_parent"] = bool(
            parent is not None
            and not identical
            and event["ts"] + event["dur"] <= parent["ts"] + parent["dur"] + _EPS
        )
        if entry["inside_parent"]:
            parent["has_child"] = True
        stack.append(entry)
        marked.append(entry)
    return marked


def leaf_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Листья трейса: события без вложенных событий (см. :func:`mark_leaves`)."""
    return [event for event in mark_leaves(events) if not event["has_child"]]


def container_summary(marked: list[dict[str, Any]]) -> dict[str, Any]:
    """Контейнеры (события с детьми) — видимы полем, а не выброшены молча.

    Контейнеры не входят в раскладку (иначе вложенность учлась бы дважды), но и
    исчезать из отчёта не должны: ``container_seconds`` — объединение их
    интервалов, и по нему видно, сколько времени «покрыто родителями».
    """
    containers = [event for event in marked if event["has_child"]]
    return {
        "container_events": len(containers),
        "container_names": sorted({event["name"] for event in containers})[:TOP_N],
        "container_seconds": _union_seconds(
            [(event["ts"], event["ts"] + event["dur"]) for event in containers]
        ),
    }


def aggregate_categories(events: list[dict[str, Any]], leaves: list[dict[str, Any]]) -> dict[str, Any]:
    """Раскладка листьев по категориям с происхождением каждой.

    Секунды категории — длина **объединения** её интервалов, поэтому вложенность
    и перекрытие внутри категории не удваиваются.  Сумма категорий может быть
    меньше покрытия операциями: разность выводится полем ``overlap_seconds``
    (перекрытие РАЗНЫХ категорий), а не растворяется в «прочее».
    """
    by_category: dict[str, list[tuple[float, float]]] = defaultdict(list)
    provenance: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, int] = defaultdict(int)
    for event in leaves:
        category, origin = classify_category(event["name"], module=event.get("module"))
        by_category[category].append((event["ts"], event["ts"] + event["dur"]))
        provenance[category][origin] += event["dur"]
        counts[category] += 1

    window = trace_window(events)
    window_seconds = window["window_seconds"] or 0.0
    covered = _union_seconds(
        [(event["ts"], event["ts"] + event["dur"]) for event in leaves]
    )
    covered = min(covered, window_seconds) if window_seconds else 0.0

    rows: list[dict[str, Any]] = []
    for category in CATEGORY_ORDER:
        intervals = by_category.get(category)
        if not intervals:
            continue
        seconds = _union_seconds(intervals)
        origin_seconds = provenance[category]
        origin_total = sum(origin_seconds.values()) or 1.0
        rows.append(
            {
                "category": category,
                "seconds": seconds,
                "share_of_window": (seconds / window_seconds) if window_seconds else None,
                "share_of_covered": (seconds / covered) if covered else None,
                "count": counts[category],
                "provenance": {key: origin_seconds[key] / origin_total for key in sorted(origin_seconds)},
            }
        )
    # Категории вне заявленного списка (расширение правил без правки константы).
    extra = [name for name in sorted(by_category) if name not in CATEGORY_ORDER]
    for category in extra:
        seconds = _union_seconds(by_category[category])
        rows.append(
            {
                "category": category,
                "seconds": seconds,
                "share_of_window": (seconds / window_seconds) if window_seconds else None,
                "share_of_covered": (seconds / covered) if covered else None,
                "count": counts[category],
                "provenance": {"by_name": 1.0},
            }
        )

    category_total = sum(row["seconds"] for row in rows)
    # Перекрытие категорий — диагностика, а не часть тождества: если листья разных
    # категорий накладываются (разные потоки/устройства), сумма категорий БОЛЬШЕ
    # покрытия.  Тождество раскладки держится на покрытии (``covered + gap == окно``),
    # а эта величина говорит, насколько категории «двойного счёта» несут.
    overlap = max(0.0, category_total - covered)
    structural = 0.0
    named = 0.0
    for row in rows:
        for origin, share in row["provenance"].items():
            if origin in ("by_module", "by_region", "by_shape"):
                structural += row["seconds"] * share
            else:
                named += row["seconds"] * share
    structural_base = structural + named
    other = next((row for row in rows if row["category"] == "other_ops"), None)
    return {
        "categories": rows,
        "categories_seconds": category_total,
        "covered_seconds": covered,
        "overlap_seconds": overlap,
        "structural_share": (structural / structural_base) if structural_base else None,
        "unclassified_share_of_covered": (
            (other["seconds"] / covered) if (other and covered) else 0.0
        ),
    }


def aggregate_ops(events: list[dict[str, Any]], *, top_n: int = TOP_N) -> dict[str, Any]:
    """Top-N **операций** по суммарному времени.

    Список строится по ЛИСТЬЯМ (см. :func:`leaf_events`): контейнеры
    (``jit_loss_fn``, ``while``-тело скан-цикла) — не операции, и их длительность
    затопила бы таблицу, спрятав настоящие ядра.  Вызывающий передаёт листья.
    """
    by_name: dict[str, dict[str, Any]] = defaultdict(lambda: {"total": 0.0, "count": 0, "category": None})
    for event in events:
        category, _ = classify_category(event["name"], module=event.get("module"))
        entry = by_name[event["name"]]
        entry["total"] += event["dur"]
        entry["count"] += 1
        entry["category"] = category

    window = trace_window(events)
    window_seconds = window["window_seconds"] or 0.0
    top_ops = [
        {
            "name": name,
            "category": entry["category"],
            "total_seconds": entry["total"],
            "count": int(entry["count"]),
            "mean_seconds": entry["total"] / entry["count"] if entry["count"] else None,
            "share_of_window": (entry["total"] / window_seconds) if window_seconds else None,
        }
        for name, entry in sorted(by_name.items(), key=lambda kv: kv[1]["total"], reverse=True)[
            : max(0, int(top_n))
        ]
    ]
    return {"top_ops": top_ops, "distinct_ops": len(by_name)}


def compute_gap(
    events: list[dict[str, Any]], *, leaves: Optional[list[dict[str, Any]]] = None
) -> dict[str, Optional[float]]:
    """Доля времени окна ВНЕ операций (хост-диспетчеризация, device-простой).

    ``gap = окно − объединение интервалов листьев``.  Классифицированный элемент —
    операция; всё остальное (в том числе ``other_ops``) в покрытие входит, потому
    что это исполненная операция.  Gap — именно «время вне операций», и он
    выводится всегда, а не прячется в «прочее».  ``leaves`` можно передать, чтобы
    не пересчитывать пометку листьев на большой трассе.
    """
    window = trace_window(events)
    window_seconds = window["window_seconds"]
    if window_seconds is None:
        return {
            **window,
            "covered_seconds": None,
            "gap_seconds": None,
            "gap_share": None,
            "leaf_events": 0,
        }
    if leaves is None:
        leaves = leaf_events(events)
    covered = _union_seconds([(event["ts"], event["ts"] + event["dur"]) for event in leaves])
    covered = min(covered, window_seconds)
    gap = max(0.0, window_seconds - covered)
    return {
        **window,
        "covered_seconds": covered,
        "gap_seconds": gap,
        "gap_share": (gap / window_seconds) if window_seconds > 0 else None,
        "leaf_events": len(leaves),
    }


def detect_steps(
    events: list[dict[str, Any]],
    *,
    device_events: Optional[list[dict[str, Any]]] = None,
    expected: int,
    marker: str = DEFAULT_STEP_MARKER,
) -> dict[str, Any]:
    """Выделить окна шагов по маркерам в трейсе (доска — top-25 по устройству).

    Маркер — имя jitted-функции шага (``value_and_grad``/``loss_fn``/…); в
    хостовой части трейса JAX кладёт по такому событию на шаг, и оно накрывает
    device-работу шага.  Маркеры ищутся по ВСЕМ событиям (они хостовые), а
    покрытие внутри окна считается по **device-операциям**.  Если число найденных
    окон не совпало с ожидаемым — границы **не выдумываются**: ``resolved=False``
    и находка ``step_boundaries_unresolved``.
    """
    pattern = re.compile(marker)
    hits = [
        event
        for event in events
        if pattern.search(str(event["name"]))
        and event["dur"] > 0
    ]
    if not hits:
        return {
            "resolved": False,
            "reason": "в трейсе нет событий-маркеров шага",
            "marker": marker,
            "matches": 0,
            "windows": [],
        }
    if len(hits) != int(expected):
        return {
            "resolved": False,
            "reason": f"маркеров {len(hits)}, ожидалось {int(expected)}",
            "marker": marker,
            "matches": len(hits),
            "windows": [],
        }
    hits.sort(key=lambda event: event["ts"])
    pool = device_events if device_events is not None else events
    windows: list[dict[str, Any]] = []
    for index, hit in enumerate(hits):
        start = hit["ts"]
        end = hit["ts"] + hit["dur"]
        inside = [
            event
            for event in pool
            if event is not hit and event["ts"] >= start - _EPS and event["ts"] + event["dur"] <= end + _EPS
        ]
        covered = _union_seconds([(e["ts"], e["ts"] + e["dur"]) for e in leaf_events(inside)])
        windows.append(
            {
                "index": index,
                "start_seconds": start,
                "end_seconds": end,
                "window_seconds": end - start,
                "marked_seconds": hit["dur"],
                "device_ops_seconds": covered,
                "device_seconds": covered,
                "op_events": len(inside),
            }
        )
    return {
        "resolved": True,
        "marker": marker,
        "matches": len(hits),
        "windows": windows,
    }


def per_step_summary(
    step_windows: dict[str, Any], metrics_steps: list[dict[str, Any]] | None
) -> dict[str, Any]:
    """Сводка по шагам: доли первого шага и разброс (device и стенное время).

    Доля первого шага считается **отдельным полем** (``first_step_share``), а не
    растворяется в среднем: после разогрева компиляция ожидается за пределами
    окна, и первый измеренный шаг — первое место, где её остаток был бы виден.
    """
    device = [row["device_seconds"] for row in step_windows.get("windows", [])] if step_windows.get("resolved") else []
    wall = [
        float(row["step_seconds"])
        for row in (metrics_steps or [])
        if isinstance(row.get("step_seconds"), (int, float)) and not isinstance(row.get("step_seconds"), bool)
    ]
    total_device = sum(device) if device else None
    total_wall = sum(wall) if wall else None

    def _share_first(values: list[float]) -> Optional[float]:
        if len(values) < 2:
            return None
        total = sum(values)
        return (values[0] / total) if total > 0 else None

    out: dict[str, Any] = {
        "device_step_seconds": device or None,
        "device_first_step_share": _share_first(device),
        "wall_step_seconds": wall or None,
        "wall_first_step_share": _share_first(wall),
        "wall_median_step_seconds": None,
        "compile_share": None,
    }
    if wall:
        ordered = sorted(wall)
        middle = len(ordered) // 2
        median = (
            ordered[middle]
            if len(ordered) % 2
            else (ordered[middle - 1] + ordered[middle]) / 2.0
        )
        out["wall_median_step_seconds"] = median
        if median > 0 and len(wall) >= 2:
            # Доля первого шага СВЕРХ медианы остальных — оценка остатка компиляции
            # после разогрева. Отрицательная разность — не «доля», а 0.0.
            rest = [value for value in wall[1:]]
            rest_median = sorted(rest)[len(rest) // 2] if rest else median
            if rest_median > 0:
                out["compile_share"] = max(0.0, (wall[0] - rest_median) / rest_median)
    return out


def reconcile(
    *,
    window_seconds: Optional[float],
    categories_seconds: Optional[float],
    covered_seconds: Optional[float],
    overlap_seconds: Optional[float],
    gap_seconds: Optional[float],
    wall_step_seconds: Optional[float],
    steps: int,
    threshold: float = RECONCILE_THRESHOLD,
) -> dict[str, Any]:
    """Сверка раскладки: внутренняя (точная) и внешняя (порог 95 %).

    **Внутренняя**: ``покрытие операциями + gap == окно`` — тождество раскладки,
    оно обязано держаться точно; расхождение означает дефект прибора, а не
    свойство шага.  Сверяется **покрытие**, а не сумма категорий: если листья
    разных категорий перекрываются (несколько потоков/устройств), сумма категорий
    законно больше покрытия, и это несёт поле ``overlap_seconds``.

    **Внешняя**: ``окно ÷ (длительность шага × число шагов)`` — покрывает ли трейс
    весь шаг.  Именно здесь осмыслен порог 95 %: без стенного времени шага он не
    выдумывается (``basis = "none"``, ``external_share = None``).
    """
    internal_residual: Optional[float] = None
    if all(value is not None for value in (window_seconds, covered_seconds, gap_seconds)):
        internal_residual = (covered_seconds + gap_seconds) - window_seconds

    categories_plus_gap: Optional[float] = None
    if all(
        value is not None
        for value in (window_seconds, categories_seconds, gap_seconds)
    ):
        categories_plus_gap = categories_seconds + gap_seconds

    step_total = None
    external_share = None
    basis = "none"
    if wall_step_seconds is not None and int(steps) > 0:
        step_total = float(wall_step_seconds) * int(steps)
        if step_total > 0 and window_seconds is not None:
            external_share = window_seconds / step_total
            basis = "wall_clock_steps"
    return {
        "threshold": threshold,
        "internal_residual_seconds": internal_residual,
        "internal_ok": (internal_residual is not None and abs(internal_residual) <= 1e-6),
        "categories_seconds": categories_seconds,
        "categories_plus_gap_seconds": categories_plus_gap,
        "overlap_seconds": overlap_seconds,
        "step_total_seconds": step_total,
        "external_share": external_share,
        "external_ok": (external_share is not None and external_share >= threshold),
        "basis": basis,
    }


def build_findings(
    *,
    gap: dict[str, Any],
    categories: dict[str, Any],
    reconciliation: dict[str, Any],
    step_windows: dict[str, Any],
    origins: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Находки прибора — то, что нельзя молча спрятать в «прочее»."""
    findings: list[dict[str, Any]] = []
    unknown_share = (origins or {}).get("unknown_share_of_window")
    if unknown_share is not None and unknown_share > UNKNOWN_FINDING_THRESHOLD:
        findings.append(
            {
                "code": "unknown_origin_high",
                "severity": "medium",
                "message": (
                    f"событий неопознанного происхождения {unknown_share:.1%} окна > "
                    f"{UNKNOWN_FINDING_THRESHOLD:.0%}: признаки происхождения "
                    "(HOST_CATS/DEVICE_CAT_HINTS/HLO_INSTRUCTION_RE) не покрывают "
                    "этот трейсер — их секунды в раскладку НЕ вошли"
                ),
            }
        )
    gap_share = gap.get("gap_share")
    if gap_share is not None and gap_share > GAP_FINDING_THRESHOLD:
        findings.append(
            {
                "code": "gap_dominates",
                "severity": "high",
                "message": (
                    f"доля окна вне операций {gap_share:.1%} > "
                    f"{GAP_FINDING_THRESHOLD:.0%}: шаг упирается в хост-диспетчеризацию "
                    "и/или device-простой, а не в арифметику ядер"
                ),
            }
        )
    structural = categories.get("structural_share")
    if structural is not None and structural < STRUCTURAL_FINDING_THRESHOLD:
        findings.append(
            {
                "code": "attribution_name_only",
                "severity": "medium",
                "message": (
                    f"структурных свидетельств раскладки {structural:.1%} < "
                    f"{STRUCTURAL_FINDING_THRESHOLD:.0%}: семантические категории "
                    "(kda/mla/moe/lm_head) опираются на имена операций — гипотеза, "
                    "перекрёстная проверка обязательна (phase_legs)"
                ),
            }
        )
    unclassified = categories.get("unclassified_share_of_covered")
    if unclassified is not None and unclassified > UNCLASSIFIED_FINDING_THRESHOLD:
        findings.append(
            {
                "code": "unclassified_high",
                "severity": "medium",
                "message": (
                    f"нераспознанных операций {unclassified:.1%} покрытия > "
                    f"{UNCLASSIFIED_FINDING_THRESHOLD:.0%}: правила категорий неполны"
                ),
            }
        )
    if reconciliation.get("basis") != "none" and reconciliation.get("external_ok") is False:
        findings.append(
            {
                "code": "step_not_covered",
                "severity": "high",
                "message": (
                    f"окно трейса покрывает {reconciliation['external_share']:.1%} "
                    f"стенного времени шагов < {reconciliation['threshold']:.0%}: "
                    "часть шага не попала в трейс (или шаг длительность включает "
                    "хост-ожидание вне окна профилирования)"
                ),
            }
        )
    if reconciliation.get("internal_ok") is False:
        findings.append(
            {
                "code": "partition_residual",
                "severity": "critical",
                "message": (
                    f"раскладка не сходится с окном: остаток "
                    f"{reconciliation['internal_residual_seconds']:.6f} с — "
                    "дефект прибора, а не свойство шага"
                ),
            }
        )
    if step_windows.get("resolved") is False:
        findings.append(
            {
                "code": "step_boundaries_unresolved",
                "severity": "medium",
                "message": (
                    f"границы шагов не выделены ({step_windows.get('reason')}): "
                    "per-step раскладка device-времени не выводится, "
                    "стенное время по шагам берётся из metrics.jsonl"
                ),
            }
        )
    return findings


def origin_summary(buckets: dict[str, list[dict[str, Any]]], window_seconds: Optional[float]) -> dict[str, Any]:
    """Сколько событий и секунд в каждом происхождении — видно, а не выброшено."""
    unknown_seconds = _union_seconds(
        [(event["ts"], event["ts"] + event["dur"]) for event in buckets["unknown"]]
    )
    host_seconds = _union_seconds(
        [(event["ts"], event["ts"] + event["dur"]) for event in buckets["host"]]
    )
    return {
        "device_events": len(buckets["device"]),
        "host_events": len(buckets["host"]),
        "unknown_events": len(buckets["unknown"]),
        "unknown_seconds": unknown_seconds,
        "host_seconds": host_seconds,
        "unknown_share_of_window": (
            (unknown_seconds / window_seconds) if window_seconds else None
        ),
    }


def analyze_trace(trace_dir: Path, *, expected_steps: int, step_marker: str = DEFAULT_STEP_MARKER) -> dict[str, Any]:
    """Полный разбор каталога трейса (fail-closed на пустом/битом трейсе).

    Раскладка строится по **device-событиям**: хостовые кадры (``jit``,
    ``python_function``) накрывают устройство целиком и в раскладке не участвуют —
    их число и секунды выводятся отдельно, а не растворяются в категориях.
    """
    raw_events, meta = load_trace_events(trace_dir)
    events = complete_events(raw_events)
    if not events:
        raise ProfileError(
            f"в traceEvents нет ни одного длительностного X-события: {trace_dir} "
            f"(событий в файле {len(raw_events)})"
        )
    buckets = split_by_origin(events)
    device = buckets["device"]
    if not device:
        raise ProfileError(
            f"в traceEvents нет device-операций: {trace_dir} (событий {len(events)}, "
            f"хостовых {len(buckets['host'])}, неопознанных {len(buckets['unknown'])}) — "
            "раскладывать нечего; при нестандартном трейсере расширьте таблицы "
            "признаков происхождения (HOST_CATS/DEVICE_CAT_HINTS/HLO_INSTRUCTION_RE)"
        )
    marked = mark_leaves(device)
    leaves = [event for event in marked if not event["has_child"]]
    categories = aggregate_categories(device, leaves)
    ops = aggregate_ops(leaves)
    gap = compute_gap(device, leaves=leaves)
    origins = origin_summary(buckets, (trace_window(device) or {}).get("window_seconds"))
    step_windows = detect_steps(
        events, device_events=device, expected=expected_steps, marker=step_marker
    )
    return {
        "trace": {
            **meta,
            "complete_event_count": len(events),
            "device_event_count": len(device),
            "leaf_event_count": len(leaves),
        },
        "origins": origins,
        "containers": container_summary(marked),
        "window": trace_window(device),
        "gap": gap,
        "categories": categories,
        "top_ops": ops["top_ops"],
        "distinct_ops": ops["distinct_ops"],
        "steps": step_windows,
    }


# --------------------------------------------------------------------------- #
# Опорная нога: метрики шага из metrics.jsonl
# --------------------------------------------------------------------------- #


def phase_legs_from_metrics(metrics_path: Path | None) -> dict[str, Any]:
    """Прочитать ``step_seconds`` и поля ``sec_*`` из ``metrics.jsonl``.

    Файл — штатный носитель KPI; прибор его **читает**, а не пишет.  Отсутствие
    файла — не ошибка: опорная нога не выдумывается (``available=False``).
    """
    if metrics_path is None:
        return {"available": False, "reason": "путь метрик не задан"}
    path = Path(metrics_path)
    if not path.is_file():
        return {"available": False, "reason": f"файл метрик не найден: {path}"}

    records: list[dict[str, Any]] = []
    broken = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            broken += 1
            continue
        if isinstance(payload, dict):
            records.append(payload)
    if not records:
        return {"available": False, "reason": f"в метриках нет записей: {path}", "broken_lines": broken}

    leg_fields = ("sec_kda", "sec_mla", "sec_moe", "sec_ce", "sec_backopt")
    legs: dict[str, Optional[float]] = {name: None for name in leg_fields}
    profiled = 0
    for record in records:
        if not record.get("phase_profile"):
            continue
        profiled += 1
        for name in leg_fields:
            value = record.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                legs[name] = float(value) if legs[name] is None else max(legs[name], float(value))
    return {
        "available": True,
        "path": str(path),
        "records": len(records),
        "broken_lines": broken,
        "profiled_records": profiled,
        "legs_seconds": legs,
        "caveat": (
            "опорные ноги — дополнительные декомпозированные проходы (kda/mla/moe/ce/"
            "шаг оптимизатора) на тех же входах; они не интерливаются как в forward, "
            "поэтому их сумма НЕ равна времени шага"
        ),
        "steps": records,
    }


# --------------------------------------------------------------------------- #
# Прогон: net/train_loop.train под jax.profiler.trace
# --------------------------------------------------------------------------- #


def synthetic_batches(batch: int, seq: int, steps: int, vocab: int, seed: int = 0):
    """Детерминированные батчи ``(B, T)`` id-токенов.

    Данные синтетические осознанно: прибор меряет ФОРМУ шага, а стоимость
    упакованного лоадера в раскладку не входит (та же граница, что у стадии 2-бис).
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    for _ in range(max(0, int(steps))):
        yield rng.integers(0, vocab, size=(batch, seq), dtype=np.int32)


def _enable_compile_cache() -> dict[str, Any]:
    """Персистентный кэш компиляции XLA: компиляция не попадает в окно замера.

    Разогрев (``--warmup``) компилирует граф первым; замер — второй прогон тех же
    форм.  Без кэша окно честно включает компиляцию — это видно в отчёте
    (``compile_cache``), а не скрыто.
    """
    import tempfile

    cache_dir = Path(tempfile.gettempdir()) / "axiom-step-profile-xla-cache"
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


def build_model_config(config_path: str | Path, impl: str):
    """Конфиг модели с переопределённой формой KDA — **в памяти**.

    ``ModelConfig`` объявлен ``frozen=True``, поэтому переопределение идёт
    ``dataclasses.replace``: дефолт ``kda_impl`` в ``net/config.json`` не
    меняется, файл не трогается.  Конфиг валидируется штатным ``validate_config``.
    """
    import dataclasses

    from net.config import load_config, validate_config

    cfg = load_config(Path(config_path))
    cfg = dataclasses.replace(cfg, kda_impl=str(impl))
    validate_config(cfg)
    return cfg


def run_window(
    config: str,
    *,
    impl: str,
    batch: int,
    seq: int,
    steps: int,
    warmup: int,
    trace_dir: Path,
    metrics_path: Path | None,
    phase_profile: bool,
    grad_checkpointing: bool = True,
    seed: int = 0,
    name: str = "l3full",
    ns_steps: int = 5,
    legacy_muon_all_2d: bool = False,
) -> dict[str, Any]:
    """Исполнить окно замера штатной ногой ``net/train_loop.train`` под трейсом.

    ADR-041: лимит памяти XLA выставляется **до** ``import jax`` (см.
    :func:`_preflight_memory`), а совмещённый стенд GB10 проверяется гейтом.
    ``net/*`` не трогается: прибор вызывает ровно ``train_loop.train``.
    """
    os.environ.pop("JAX_PLATFORMS", None)

    import jax

    from net import train_loop as tl

    cfg = build_model_config(config, impl)
    compile_cache = _enable_compile_cache()

    def make_config(run_steps: int) -> Any:
        return tl.TrainConfig(
            steps=run_steps,
            total_steps=max(1, run_steps),
            seed=seed,
            micro_batch=batch,
            grad_checkpointing=bool(grad_checkpointing),
            param_dtype="float32",
            metrics_path=metrics_path,
            log_every=0,
            kpi_every=1,  # опорная нога — на каждом шаге окна
            phase_profile=bool(phase_profile),
            ns_steps=int(ns_steps),
            legacy_muon_all_2d=bool(legacy_muon_all_2d),
        )

    budget = tl.Budget(
        run_ref=f"step-profile-{name}",
        path=Path(trace_dir),
        present=True,
        budget_method="прибор профиля шага (ADR-047 п. 3): смета не расходуется",
        stop_rule=f"{warmup}+{steps} шагов прибора",
    )

    def leg(run_steps: int):
        return tl.train(
            cfg,
            synthetic_batches(batch, seq, run_steps, cfg.vocab_size, seed=seed),
            train_config=make_config(run_steps),
            budget=budget,
        )

    started = time.time()
    warmup_seconds = 0.0
    if warmup > 0:
        # Разогрев — ВНЕ окна трейса: jit/аллокатор/компиляция не измеряются.
        warmup_started = time.perf_counter()
        leg(warmup)
        warmup_seconds = time.perf_counter() - warmup_started

    trace_dir = Path(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    measured_started = time.perf_counter()
    with _trace_context(trace_dir):
        result = leg(steps)
    measured_seconds = time.perf_counter() - measured_started

    return {
        "device": [repr(device) for device in jax.devices()],
        "backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "compile_cache": compile_cache,
        "kda_impl": str(cfg.kda_impl),
        "steps_done": result.steps_done,
        "losses": [float(value) for value in result.losses],
        "warmup_seconds": warmup_seconds,
        "measured_wall_seconds": measured_seconds,
        "wall_seconds": time.time() - started,
        "trace_dir": str(trace_dir),
        "metrics_path": str(metrics_path) if metrics_path else None,
    }


# --------------------------------------------------------------------------- #
# Зонд устройства и префлайт (fail-closed)
# --------------------------------------------------------------------------- #


def gpu_available() -> bool:
    """Есть ли исполнимый GPU (без падения при отсутствии) — зонд стадии 2-бис."""
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
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )
    except Exception:
        return False
    return out.returncode == 0 and out.stdout.strip().endswith("gpu")


def _preflight_memory() -> None:
    """ADR-041: лимит памяти XLA — ДО импорта jax (маркер ``ensure_mem_fraction()``).

    Инцидент 08.10: прогон без лимита взял дефолтный резерв ~75 % устройства, и
    совмещённый стенд ушёл в global OOM.  Вызов обязан стоять до ``import jax``.
    """
    import jax_preflight

    jax_preflight.ensure_mem_fraction()


# --------------------------------------------------------------------------- #
# Отчёт
# --------------------------------------------------------------------------- #


def cell_meta(config: str, batch: int, seq: int, *, impl: str, name: str, steps: int, warmup: int,
              ns_steps: int = 5, legacy_muon_all_2d: bool = False) -> dict[str, Any]:
    return {
        "name": name,
        "config": config,
        "impl": impl,
        "batch": int(batch),
        "seq": int(seq),
        "steps": int(steps),
        "warmup": int(warmup),
        # ADR-048: режим оптимизатора — часть условий клетки, иначе «до/после»
        # по ``sec_backopt`` неотличимо от смены чего-то ещё.
        "ns_steps": int(ns_steps),
        "legacy_muon_all_2d": bool(legacy_muon_all_2d),
    }


def plan() -> list[dict[str, Any]]:
    """План прибора: одна клетка — l3-full под ``chunked_cc`` (ADR-047 п. 6)."""
    return [
        {
            "name": "l3full-chunked-cc",
            "config": DEFAULT_CONFIG,
            "impl": DEFAULT_IMPL,
            "batch": DEFAULT_BATCH,
            "seq": DEFAULT_SEQ,
            "steps": DEFAULT_STEPS,
            "warmup": DEFAULT_WARMUP,
            "command": (
                "python3 tools/profile_step.py --config net/config.json "
                f"--impl {DEFAULT_IMPL} --batch {DEFAULT_BATCH} --seq {DEFAULT_SEQ} "
                f"--steps {DEFAULT_STEPS} --warmup {DEFAULT_WARMUP} "
                "--out evidence/kda-rewrite/step-profile.json"
            ),
        }
    ]


def build_report(
    *,
    cell: dict[str, Any],
    status: str,
    analysis: Optional[dict[str, Any]] = None,
    run_meta: Optional[dict[str, Any]] = None,
    phase_legs: Optional[dict[str, Any]] = None,
    note: Optional[str] = None,
    error: Optional[str] = None,
) -> dict[str, Any]:
    """Собрать отчёт.  Без ``analysis`` чисел профиля нет — и не будет."""
    run_meta = run_meta or {}
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "version": REPORT_VERSION,
        "instrument": "tools/profile_step.py",
        "adr": "ADR-047",
        "status": status,
        "created": datetime.now(timezone.utc).isoformat(),
        "cell": cell,
        "runner": "net/train_loop.train (та же нога, что в претрейне; net/* не меняется)",
        "jax_version": run_meta.get("jax_version"),
        "device": run_meta.get("device", []),
        "backend": run_meta.get("backend"),
        "compile_cache": run_meta.get("compile_cache"),
        "trace_dir": run_meta.get("trace_dir"),
        "warmup_seconds": run_meta.get("warmup_seconds"),
        "measured_wall_seconds": run_meta.get("measured_wall_seconds"),
        "kda_impl": run_meta.get("kda_impl", cell.get("impl")),
        "losses": run_meta.get("losses"),
        "categories": [],
        "top_ops": [],
        "steps": None,
        "gap": None,
        "containers": None,
        "origins": None,
        "reconciliation": None,
        "findings": [],
        "phase_legs": phase_legs,
        "caveats": list(CAVEATS),
    }
    if analysis is not None:
        metrics_steps = (phase_legs or {}).get("steps") if isinstance(phase_legs, dict) else None
        steps_summary = per_step_summary(analysis["steps"], metrics_steps)
        gap = analysis["gap"]
        categories = analysis["categories"]
        reconciliation = reconcile(
            window_seconds=(analysis["window"] or {}).get("window_seconds"),
            categories_seconds=categories.get("categories_seconds"),
            covered_seconds=categories.get("covered_seconds"),
            overlap_seconds=categories.get("overlap_seconds"),
            gap_seconds=gap.get("gap_seconds"),
            wall_step_seconds=steps_summary.get("wall_median_step_seconds"),
            steps=int(cell.get("steps") or 0),
        )
        report.update(
            {
                "trace": analysis["trace"],
                "window": analysis["window"],
                "origins": analysis.get("origins"),
                "containers": analysis.get("containers"),
                "distinct_ops": analysis.get("distinct_ops"),
                "categories": categories["categories"],
                "categories_seconds": categories["categories_seconds"],
                "covered_seconds": categories["covered_seconds"],
                "overlap_seconds": categories["overlap_seconds"],
                "structural_share": categories["structural_share"],
                "top_ops": analysis["top_ops"],
                "gap": gap,
                "steps": {**analysis["steps"], **steps_summary},
                "reconciliation": reconciliation,
                "findings": build_findings(
                    gap=gap,
                    categories=categories,
                    reconciliation=reconciliation,
                    step_windows=analysis["steps"],
                    origins=analysis.get("origins"),
                ),
            }
        )
    if note:
        report["note"] = note
    if error:
        report["error"] = error
    return report


def write_report(report: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


def _pct(value: Optional[float]) -> str:
    return f"{value * 100:.1f}%" if value is not None else "—"


def _ms(value: Optional[float]) -> str:
    return f"{value * 1e3:.2f}" if value is not None else "—"


def format_table(report: dict[str, Any]) -> str:
    """Человекочитаемая таблица: категории, top-операции, шаги, находки."""
    lines: list[str] = []
    cell = report.get("cell") or {}
    lines.append(
        f"[step-profile] {report.get('status')} · {cell.get('name')} · "
        f"{cell.get('config')} impl={cell.get('impl')} b{cell.get('batch')} "
        f"seq{cell.get('seq')} · шаги {cell.get('steps')} (+{cell.get('warmup')} разогрев)"
    )
    window = report.get("window") or {}
    if window.get("window_seconds") is not None:
        lines.append(
            f"  окно {_ms(window['window_seconds'])} мс · покрыто операциями "
            f"{_ms(report.get('covered_seconds'))} мс · перекрытие "
            f"{_ms(report.get('overlap_seconds'))} мс · gap "
            f"{_ms((report.get('gap') or {}).get('gap_seconds'))} мс "
            f"({_pct((report.get('gap') or {}).get('gap_share'))})"
        )
    reconciliation = report.get("reconciliation")
    if isinstance(reconciliation, dict) and reconciliation.get("basis") != "none":
        lines.append(
            f"  сходимость: внутренняя остаток "
            f"{(reconciliation.get('internal_residual_seconds') or 0.0):.9f} с "
            f"({'ok' if reconciliation.get('internal_ok') else 'BAD'}) · внешняя "
            f"{_pct(reconciliation.get('external_share'))} против порога "
            f"{_pct(reconciliation.get('threshold'))} "
            f"({'ok' if reconciliation.get('external_ok') else 'FAIL'})"
        )
    if report.get("structural_share") is not None:
        lines.append(
            f"  структурных свидетельств раскладки: {_pct(report.get('structural_share'))}"
        )
    origins = report.get("origins") or {}
    if origins:
        lines.append(
            f"  событий: device {origins.get('device_events')} · host "
            f"{origins.get('host_events')} (вне раскладки) · неопознанных "
            f"{origins.get('unknown_events')} ({_pct(origins.get('unknown_share_of_window'))} окна)"
        )
    containers = report.get("containers") or {}
    if containers.get("container_events"):
        lines.append(
            f"  контейнеров (не операции, в раскладку не входят): "
            f"{containers['container_events']} · {_ms(containers.get('container_seconds'))} мс · "
            f"{', '.join(containers.get('container_names') or [])}"
        )

    categories = report.get("categories") or []
    if categories:
        lines.append(f"  {'категория':<18} {'секунды':>9} {'доля окна':>10} {'событий':>8}  происхождение")
        for row in categories:
            origin = ", ".join(
                f"{key.replace('by_', '')}={value:.2f}" for key, value in (row.get("provenance") or {}).items()
            )
            lines.append(
                f"  {row['category']:<18} {row['seconds']:>9.3f} "
                f"{_pct(row.get('share_of_window')):>10} {row['count']:>8}  {origin}"
            )

    top_ops = report.get("top_ops") or []
    if top_ops:
        lines.append(f"  {'операция':<40} {'категория':<15} {'секунды':>9} {'шт':>6}")
        for op in top_ops:
            lines.append(
                f"  {op['name'][:40]:<40} {op['category']:<15} "
                f"{op['total_seconds']:>9.3f} {op['count']:>6}"
            )

    steps = report.get("steps") or {}
    if steps.get("wall_step_seconds"):
        lines.append(
            f"  шаги (стенные, с): {[round(v, 3) for v in steps['wall_step_seconds']]} · "
            f"медиана {_ms(steps.get('wall_median_step_seconds'))} мс · доля первого "
            f"{_pct(steps.get('wall_first_step_share'))} · остаток компиляции "
            f"{_pct(steps.get('compile_share'))}"
        )
    if steps.get("device_step_seconds"):
        lines.append(
            f"  шаги (device, с): {[round(v, 3) for v in steps['device_step_seconds']]} · "
            f"доля первого {_pct(steps.get('device_first_step_share'))}"
        )

    legs = report.get("phase_legs") or {}
    if isinstance(legs, dict) and legs.get("available"):
        rendered = ", ".join(
            f"{name}={value:.3f}" if value is not None else f"{name}=—"
            for name, value in (legs.get("legs_seconds") or {}).items()
        )
        lines.append(f"  опорные ноги (с, НЕ сумма шага): {rendered}")

    findings = report.get("findings") or []
    if findings:
        lines.append("  находки:")
        for finding in findings:
            lines.append(f"    [{finding['severity']}] {finding['code']}: {finding['message']}")
    elif report.get("status") != "EMPTY-PENDING":
        lines.append("  находки: нет")

    if report.get("status") == "EMPTY-PENDING":
        lines.append("  (чисел нет: прогон не выполнен — см. note)")
    if report.get("error"):
        lines.append(f"  ошибка: {report['error']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Самопроверка (без jax и без GPU)
# --------------------------------------------------------------------------- #


def selftest() -> int:
    checks: list[tuple[str, bool]] = []

    # --- классификация: семантика и происхождение ---
    checks.append(("einsum.42 → gemm_einsum", classify_category("einsum.42")[0] == "gemm_einsum"))
    checks.append(
        ("reduce-scatter.0 → collective (не elementwise)", classify_category("reduce-scatter.0")[0] == "collective")
    )
    checks.append(("memcpy-d2d.1 → memcpy_memset", classify_category("memcpy-d2d.1")[0] == "memcpy_memset"))
    checks.append(("kda_chunk.3 → kda", classify_category("kda_chunk.3")[0] == "kda"))
    checks.append(("moe_dispatch.2 → moe_ffn", classify_category("moe_dispatch.2")[0] == "moe_ffn"))
    checks.append(("mla_flash.5 → mla_attention", classify_category("mla_flash.5")[0] == "mla_attention"))
    checks.append(
        ("cross_entropy.7 → lm_head", classify_category("cross_entropy.7")[0] == "lm_head")
    )
    checks.append(
        ("adam_apply.1 → optimizer_apply", classify_category("adam_apply.1")[0] == "optimizer_apply")
    )
    checks.append(
        ("fusion.9 → elementwise", classify_category("fusion.9")[0] == "elementwise")
    )
    checks.append(
        ("незнакомое имя → other_ops",
         classify_category("mystery-op.1")[0] == "other_ops")
    )
    checks.append(
        ("module kda → by_module",
         classify_category("fusion.1", module="kda_block") == ("kda", "by_module"))
    )
    checks.append(
        ("без module → by_name",
         classify_category("moe_router.1") == ("moe_ffn", "by_name"))
    )

    # --- происхождение события: хост-кадры вне раскладки device-времени ---
    checks.append(
        ("cat python_function → host", classify_origin({"name": "train", "cat": "python_function"}) == "host")
    )
    checks.append(
        ("cat jit → host", classify_origin({"name": "jit_loss_fn", "cat": "jit"}) == "host")
    )
    checks.append(
        ("имя HLO-инструкции → device",
         classify_origin({"name": "fusion.12"}) == "device")
    )
    checks.append(
        ("args.hlo_op → device",
         classify_origin({"name": "opaque", "args": {"hlo_op": "dot"}}) == "device")
    )
    checks.append(
        ("cat XLA Ops → device",
         classify_origin({"name": "kernel_x", "cat": "XLA Ops"}) == "device")
    )
    checks.append(
        ("хост-имя без признаков → unknown (не выдумываем)",
         classify_origin({"name": "jit_loss_fn"}) == "unknown")
    )
    buckets = split_by_origin(
        complete_events(
            [
                {"ph": "X", "name": "jit_loss_fn", "ts": 0, "dur": 100, "cat": "jit"},
                {"ph": "X", "name": "fusion.0", "ts": 10, "dur": 10},
                {"ph": "X", "name": "mystery", "ts": 30, "dur": 5},
            ]
        )
    )
    checks.append(("раскладка по происхождению: 1 device / 1 host / 1 unknown",
                   len(buckets["device"]) == 1 and len(buckets["host"]) == 1
                   and len(buckets["unknown"]) == 1))
    origins = origin_summary(buckets, 100e-6)
    checks.append(("секунды неопознанных выведены (5 мкс)",
                   abs(origins["unknown_seconds"] - 5e-6) < 1e-12))
    unknown_finding = build_findings(
        gap={"gap_share": 0.1},
        categories={"structural_share": 0.9, "unclassified_share_of_covered": 0.0},
        reconciliation={"basis": "none", "internal_ok": True},
        step_windows={"resolved": True},
        origins={"unknown_share_of_window": 0.9},
    )
    checks.append(("находка unknown_origin_high",
                   [finding["code"] for finding in unknown_finding] == ["unknown_origin_high"]))

    # --- нормализация событий ---
    sample = complete_events(
        [
            {"ph": "X", "name": "einsum.1", "ts": 0, "dur": 10},
            {"ph": "X", "name": "einsum.1", "ts": 10, "dur": 10},
            {"ph": "X", "name": "reduce.2", "ts": 20, "dur": 5},
            {"ph": "X", "name": "memcpy-d2d.3", "ts": 25, "dur": 15},
            {"ph": "B", "name": "einsum.1", "ts": 0},
            {"ph": "X", "name": "bad", "ts": 0, "dur": -1},
        ]
    )
    checks.append(("X-события нормализованы (4 из 6)", len(sample) == 4))
    window = trace_window(sample)
    checks.append(("окно = 40 мкс", abs(window["window_seconds"] - 40e-6) < 1e-12))

    # --- вложенность: листья против контейнера ---
    nested = complete_events(
        [
            {"ph": "X", "name": "while.0", "ts": 0, "dur": 100},
            {"ph": "X", "name": "einsum.1", "ts": 10, "dur": 20},
            {"ph": "X", "name": "reduce.2", "ts": 40, "dur": 10},
        ]
    )
    leaves = leaf_events(nested)
    checks.append(("листьев 2 из 3 (контейнер отброшен)", len(leaves) == 2))
    gap = compute_gap(nested)
    checks.append(("покрытие = 30 мкс (вложенность не удвоена)",
                   abs(gap["covered_seconds"] - 30e-6) < 1e-12))
    checks.append(("gap = 70 мкс (0.7)", abs(gap["gap_share"] - 0.7) < 1e-12))

    # --- раскладка: покрытие + gap == окно, сумма категорий == покрытие ---
    categories = aggregate_categories(nested, leaves)
    checks.append(("сумма категорий = покрытие (перекрытия нет)",
                   abs(categories["categories_seconds"] - gap["covered_seconds"]) < 1e-12))
    checks.append(("перекрытие = 0", categories["overlap_seconds"] < 1e-12))
    categories_by_name = {row["category"]: row for row in categories["categories"]}
    checks.append(("einsum ушёл в gemm_einsum",
                   "gemm_einsum" in categories_by_name
                   and abs(categories_by_name["gemm_einsum"]["seconds"] - 20e-6) < 1e-12))
    checks.append(("reduce.2 ушёл в elementwise (не gemm)",
                   "elementwise" in categories_by_name
                   and abs(categories_by_name["elementwise"]["seconds"] - 10e-6) < 1e-12))

    # --- перекрытие РАЗНЫХ категорий: покрытие < суммы категорий ---
    # Листья наложены, но НЕ вложены (второй выходит за границу первого):
    # union = 25 мкс, сумма длительностей = 35 мкс → перекрытие 10 мкс.
    overlapping = complete_events(
        [
            {"ph": "X", "name": "einsum.1", "ts": 0, "dur": 20},
            {"ph": "X", "name": "memcpy-d2d.1", "ts": 10, "dur": 15},
        ]
    )
    overlapping_leaves = leaf_events(overlapping)
    checks.append(("наложенные невложенные события — оба листья", len(overlapping_leaves) == 2))
    overlap_categories = aggregate_categories(overlapping, overlapping_leaves)
    overlap_gap = compute_gap(overlapping)
    checks.append(("перекрытие категорий выведено (10 мкс)",
                   abs(overlap_categories["overlap_seconds"] - 10e-6) < 1e-12))
    checks.append(("сумма категорий (35) больше покрытия (25)",
                   overlap_categories["categories_seconds"] > overlap_categories["covered_seconds"]))
    checks.append(("внутренняя сходимость при перекрытии: покрытие + gap == окно",
                   abs((overlap_categories["covered_seconds"] + overlap_gap["gap_seconds"]) - 25e-6) < 1e-12))

    # --- top-операции строятся по листьям: контейнер не затопляет таблицу ---
    ops = aggregate_ops(leaves)
    op_names = {row["name"] for row in ops["top_ops"]}
    checks.append(("контейнер while.0 не попал в top-операции", "while.0" not in op_names))
    checks.append(("настоящая операция в top-операциях", "einsum.1" in op_names))
    checks.append(("top-1 — самая тяжёлая ОПЕРАЦИЯ (einsum.1, 20 мкс)",
                   ops["top_ops"][0]["name"] == "einsum.1"
                   and abs(ops["top_ops"][0]["total_seconds"] - 20e-6) < 1e-12))

    # --- вложение только внутри дорожки: наложение разных потоков — не вложение ---
    # Крупная операция (KDA) на дорожке 1 и мелкие на дорожке 2 накладываются по
    # времени; без учёта дорожки KDA стала бы «контейнером» и молча исчезла.
    cross_track = complete_events(
        [
            {"ph": "X", "name": "kda_cc_scores.1", "ts": 0, "dur": 100, "pid": 1, "tid": 1},
            {"ph": "X", "name": "fusion.0", "ts": 10, "dur": 5, "pid": 2, "tid": 1},
            {"ph": "X", "name": "fusion.1", "ts": 40, "dur": 5, "pid": 2, "tid": 1},
        ]
    )
    cross_leaves = leaf_events(cross_track)
    checks.append(("наложение разных дорожек → все три листья (KDA не потеряна)",
                   len(cross_leaves) == 3
                   and any(event["name"] == "kda_cc_scores.1" for event in cross_leaves)))
    cross_categories = {row["category"]: row for row in aggregate_categories(cross_track, cross_leaves)["categories"]}
    checks.append(("KDA на своей дорожке осталась в категории kda",
                   abs(cross_categories["kda"]["seconds"] - 100e-6) < 1e-12))

    # --- контейнеры видны полем, а не выброшены молча ---
    containers = container_summary(mark_leaves(nested))
    checks.append(("контейнер посчитан (while.0)", containers["container_events"] == 1
                   and "while.0" in containers["container_names"]))
    checks.append(("секунды контейнера выведены (100 мкс)",
                   abs(containers["container_seconds"] - 100e-6) < 1e-12))

    # --- дубль события (та же ts и длительность) не теряется из листьев ---
    duplicates = complete_events(
        [
            {"ph": "X", "name": "einsum.1", "ts": 0, "dur": 20},
            {"ph": "X", "name": "einsum.1", "ts": 0, "dur": 20},
        ]
    )
    checks.append(("дубль не съедает соседа", len(leaf_events(duplicates)) == 2))

    # --- сверка: внутренняя точная, внешняя против стенного времени ---
    reconciliation = reconcile(
        window_seconds=40e-6,
        categories_seconds=30e-6,
        covered_seconds=30e-6,
        overlap_seconds=0.0,
        gap_seconds=10e-6,
        wall_step_seconds=11e-6,
        steps=4,
    )
    checks.append(("внутренняя сходимость ok", reconciliation["internal_ok"] is True))
    checks.append(("внешняя сходимость 40/(11*4) = 0.909 < 0.95 → FAIL",
                   reconciliation["external_ok"] is False
                   and abs(reconciliation["external_share"] - 40e-6 / 44e-6) < 1e-12))
    reconciliation_ok = reconcile(
        window_seconds=40e-6,
        categories_seconds=30e-6,
        covered_seconds=30e-6,
        overlap_seconds=0.0,
        gap_seconds=10e-6,
        wall_step_seconds=10.5e-6,
        steps=4,
    )
    checks.append(("внешняя сходимость 40/42 = 0.952 ≥ 0.95 → ok",
                   reconciliation_ok["external_ok"] is True))
    no_metrics = reconcile(
        window_seconds=40e-6,
        categories_seconds=30e-6,
        covered_seconds=30e-6,
        overlap_seconds=0.0,
        gap_seconds=10e-6,
        wall_step_seconds=None,
        steps=4,
    )
    checks.append(("без стенного времени порог не выдумывается (basis=none)",
                   no_metrics["basis"] == "none" and no_metrics["external_share"] is None))

    # --- находки ---
    findings = build_findings(
        gap={"gap_share": 0.8},
        categories={"structural_share": 0.1, "unclassified_share_of_covered": 0.3},
        reconciliation={**reconciliation, "basis": "wall_clock_steps"},
        step_windows={"resolved": False, "reason": "нет маркеров"},
    )
    codes = {finding["code"] for finding in findings}
    checks.append(("находка gap_dominates при gap>40 %", "gap_dominates" in codes))
    checks.append(("находка attribution_name_only", "attribution_name_only" in codes))
    checks.append(("находка unclassified_high", "unclassified_high" in codes))
    checks.append(("находка step_not_covered", "step_not_covered" in codes))
    checks.append(("находка step_boundaries_unresolved", "step_boundaries_unresolved" in codes))
    quiet = build_findings(
        gap={"gap_share": 0.1},
        categories={"structural_share": 0.9, "unclassified_share_of_covered": 0.01},
        reconciliation=reconciliation_ok,
        step_windows={"resolved": True},
    )
    checks.append(("чистый профиль → находок нет", quiet == []))

    # --- партиция расходится: находка critical ---
    bad = build_findings(
        gap={"gap_share": 0.1},
        categories={"structural_share": 0.9, "unclassified_share_of_covered": 0.0},
        reconciliation={
            "internal_ok": False,
            "internal_residual_seconds": 1e-3,
            "basis": "none",
            "external_ok": None,
            "external_share": None,
            "threshold": RECONCILE_THRESHOLD,
        },
        step_windows={"resolved": True},
    )
    checks.append(("раскладка врёт → critical partition_residual",
                   [f["code"] for f in bad] == ["partition_residual"]
                   and bad[0]["severity"] == "critical"))

    # --- выделение шагов ---
    step_events = complete_events(
        [
            {"ph": "X", "name": "jit_loss_fn", "ts": 0, "dur": 1000},
            {"ph": "X", "name": "jit_loss_fn", "ts": 1000, "dur": 900},
            {"ph": "X", "name": "einsum.1", "ts": 100, "dur": 50},
            {"ph": "X", "name": "einsum.1", "ts": 1100, "dur": 40},
        ]
    )
    detected = detect_steps(step_events, expected=2)
    checks.append(("шагов выделено 2", detected["resolved"] is True and len(detected["windows"]) == 2))
    checks.append(("первое окно длиннее второго (компиляция)",
                   detected["windows"][0]["device_seconds"] > detected["windows"][1]["device_seconds"]))
    unresolved = detect_steps(step_events, expected=4)
    checks.append(("число маркеров ≠ ожидаемому → не выдумываем границы",
                   unresolved["resolved"] is False and unresolved["matches"] == 2))
    no_marker = detect_steps(complete_events([{"ph": "X", "name": "einsum.1", "ts": 0, "dur": 1}]), expected=1)
    checks.append(("нет маркеров → resolved=False", no_marker["resolved"] is False))

    # --- сводка по шагам: доля первого отдельным полем ---
    summary = per_step_summary(
        detected,
        [{"step_seconds": 33.0}, {"step_seconds": 31.0}, {"step_seconds": 31.0}],
    )
    checks.append(("доля первого шага отдельным полем",
                   summary["wall_first_step_share"] is not None
                   and summary["wall_first_step_share"] > 1 / 3))
    checks.append(("медиана стенного времени = 31.0",
                   abs(summary["wall_median_step_seconds"] - 31.0) < 1e-12))
    checks.append(("остаток компиляции первого шага выведен",
                   abs(summary["compile_share"] - (33.0 - 31.0) / 31.0) < 1e-12))

    # --- опорные ноги из метрик ---
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    metrics = tmp / "metrics.jsonl"
    metrics.write_text(
        "\n".join(
            [
                json.dumps({"step": 1, "step_seconds": 33.0, "phase_profile": True,
                            "sec_kda": 6.0, "sec_mla": 2.0, "sec_moe": 9.0,
                            "sec_ce": 1.0, "sec_backopt": 8.0}),
                "не json",
                json.dumps({"step": 2, "step_seconds": 31.0, "phase_profile": True,
                            "sec_kda": 5.5, "sec_mla": 2.1, "sec_moe": 9.2,
                            "sec_ce": 1.1, "sec_backopt": 8.3}),
            ]
        ),
        encoding="utf-8",
    )
    legs = phase_legs_from_metrics(metrics)
    checks.append(("опорные ноги прочитаны", legs["available"] is True and legs["profiled_records"] == 2))
    checks.append(("sec_kda берётся максимумом по записям (6.0)",
                   abs(legs["legs_seconds"]["sec_kda"] - 6.0) < 1e-12))
    checks.append(("битая строка метрик посчитана, а не проглочена", legs["broken_lines"] == 1))
    checks.append(("оговорка о ногах присутствует", "НЕ равна" in legs["caveat"]))
    missing_legs = phase_legs_from_metrics(tmp / "nope.jsonl")
    checks.append(("нет файла метрик → available=False", missing_legs["available"] is False))

    # --- fail-closed: нет данных — нет чисел ---
    empty = tmp / "trace-less"
    empty.mkdir()
    (empty / "summary.json").write_text(json.dumps({"other": 1}), encoding="utf-8")
    raised = False
    try:
        load_trace_events(empty)
    except ProfileError:
        raised = True
    checks.append(("пустой трейс (без traceEvents) → ProfileError", raised))

    broken = tmp / "trace-broken"
    broken.mkdir()
    (broken / "trace.json").write_text("{не json", encoding="utf-8")
    raised = False
    try:
        load_trace_events(broken)
    except ProfileError:
        raised = True
    checks.append(("битый трейс → ProfileError", raised))

    no_x = tmp / "trace-no-x"
    no_x.mkdir()
    (no_x / "trace.json").write_text(
        json.dumps({"traceEvents": [{"ph": "M", "name": "meta"}]}), encoding="utf-8"
    )
    raised = False
    try:
        analyze_trace(no_x, expected_steps=1)
    except ProfileError:
        raised = True
    checks.append(("трейс без X-событий → ProfileError", raised))

    # --- дедупликация источников: берём файл с наибольшим числом событий ---
    dup = tmp / "trace-dup"
    dup.mkdir()
    (dup / "a.json").write_text(
        json.dumps({"traceEvents": [{"ph": "X", "name": "copy.1", "ts": 0, "dur": 1}]}),
        encoding="utf-8",
    )
    (dup / "b.json").write_text(
        json.dumps({"traceEvents": [
            {"ph": "X", "name": "einsum.1", "ts": 0, "dur": 2},
            {"ph": "X", "name": "reduce.1", "ts": 2, "dur": 3},
        ]}),
        encoding="utf-8",
    )
    events, meta = load_trace_events(dup)
    checks.append(("берётся файл с наибольшим traceEvents (2)",
                   len(events) == 2 and meta["file_used"].endswith("b.json")
                   and len(meta["duplicate_trace_files"]) == 1))

    # --- gz-источник читается ---
    gz_dir = tmp / "trace-gz"
    gz_dir.mkdir()
    with gzip.open(gz_dir / "trace.json.gz", "wt", encoding="utf-8") as fh:
        json.dump({"traceEvents": [{"ph": "X", "name": "einsum.1", "ts": 0, "dur": 5}]}, fh)
    gz_events, _ = load_trace_events(gz_dir)
    checks.append(("gz-трейс читается", len(gz_events) == 1))

    # --- отчёт без анализа не несёт чисел ---
    report = build_report(
        cell=cell_meta(DEFAULT_CONFIG, 1, 8192, impl=DEFAULT_IMPL, name="l3full", steps=4, warmup=2),
        status="EMPTY-PENDING",
    )
    checks.append(("EMPTY-PENDING без чисел",
                   report["gap"] is None and report["top_ops"] == [] and report["categories"] == []))
    checks.append(("caveats несут пометку про эвристику имён",
                   any("эвристич" in caveat for caveat in report["caveats"])))
    checks.append(("отчёт помечен схемой", report["schema"] == REPORT_SCHEMA))

    # --- ADR-041: маркер префлайта и его вызов на ПРОГОННОМ пути до import jax ---
    source = (_REPO_ROOT / "tools" / "profile_step.py").read_text(encoding="utf-8")
    marker = "jax_preflight.ensure_mem_fraction()"
    checks.append(("ADR-041: ensure_mem_fraction() вызывается в файле", marker in source))
    body = source[source.index("def main(") :]
    checks.append(
        ("ADR-041: на прогонном пути префлайт вызван ДО run_window",
         body.index("_preflight_memory()") < body.index("run_window("))
    )

    # --- план клетки ---
    checks.append(("план = одна клетка l3-full (chunked_cc)",
                   len(plan()) == 1 and plan()[0]["impl"] == "chunked_cc"))

    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'} {name}")
    if failed:
        print(f"FAIL: {len(failed)} из {len(checks)}", file=sys.stderr)
        return 1
    print(f"PASS: {len(checks)} проверок")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _empty_pending_note() -> str:
    return (
        "GPU недоступен: клетка не исполнена. Отчёт несёт план и причину; прогон "
        "выполняет архитектор на стенде GB10 (ADR-041: лимит памяти XLA + гейт "
        "совмещённого стенда; AD-7/C-040: лок на ~/gb10-shared/.locks). "
        "Числа не имитируются — для проверки провода прибора на CPU есть --allow-cpu."
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--impl", default=DEFAULT_IMPL,
                        choices=["chunked", "wyut", "chunked_cc"],
                        help="форма KDA (переопределяется в памяти; дефолт конфига не меняется)")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--seq", type=int, default=DEFAULT_SEQ)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--name", default="l3full-chunked-cc")
    parser.add_argument("--out", default=None, help=f"путь отчёта (дефолт {DEFAULT_OUT})")
    parser.add_argument("--trace-dir", default=None, help="каталог jax.profiler.trace")
    parser.add_argument("--metrics", default=None,
                        help="metrics.jsonl профильного прогона (опорные ноги; дефолт — рядом с трейсом)")
    parser.add_argument("--no-phase-legs", action="store_true",
                        help="не включать опорную ногу (phase_profile) — короче прогон")
    parser.add_argument("--ns-steps", type=int, default=5,
                        help="ADR-048: число Newton-Schulz итераций Muon (дефолт 5)")
    parser.add_argument("--legacy-muon-all-2d", dest="legacy_muon_all_2d",
                        action="store_true", default=False,
                        help="ADR-048: прежняя классификация («любой ndim==2 -> Muon») — "
                             "для замера sec_backopt «до/после»")
    parser.add_argument("--step-marker", default=DEFAULT_STEP_MARKER,
                        help="регексп маркера шага для per-step раскладки device-времени")
    parser.add_argument("--parse-trace", default=None,
                        help="разобрать уже собранный каталог трейса (без GPU/jax)")
    parser.add_argument("--plan", action="store_true", help="напечатать план клетки и выйти")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="честный CPU-прогон малой геометрии (SMOKE-CPU), не число")
    args = parser.parse_args(argv)

    if args.selftest:
        return selftest()

    if args.plan:
        cell = plan()[0]
        print(f"{cell['name']}\t{cell['config']}\t{cell['impl']}\t{cell['batch']}\t{cell['seq']}\t"
              f"{cell['steps']}\t{cell['warmup']}")
        print(f"# {cell['command']}")
        return 0

    out_path = Path(args.out) if args.out else DEFAULT_OUT
    trace_dir = Path(args.trace_dir) if args.trace_dir else DEFAULT_TRACE_DIR
    metrics_path = (
        Path(args.metrics) if args.metrics
        else (Path(args.trace_dir).parent / "metrics.jsonl" if args.trace_dir else None)
    )
    cell = cell_meta(
        args.config, args.batch, args.seq,
        impl=args.impl, name=args.name, steps=args.steps, warmup=args.warmup,
        ns_steps=args.ns_steps, legacy_muon_all_2d=bool(args.legacy_muon_all_2d),
    )

    # --- разбор уже собранного трейса: GPU/jax не нужны -----------------------
    if args.parse_trace is not None:
        try:
            analysis = analyze_trace(Path(args.parse_trace), expected_steps=args.steps,
                                     step_marker=args.step_marker)
        except ProfileError as exc:
            report = build_report(cell=cell, status="TRACE-ERROR", error=str(exc),
                                  phase_legs=phase_legs_from_metrics(metrics_path))
            write_report(report, out_path)
            print(f"[step-profile] TRACE-ERROR: {exc} → {out_path}", file=sys.stderr)
            return 2
        report = build_report(
            cell=cell, status="COMPLETE", analysis=analysis,
            run_meta={"trace_dir": str(args.parse_trace)},
            phase_legs=phase_legs_from_metrics(metrics_path),
        )
        write_report(report, out_path)
        print(format_table(report))
        print(f"[step-profile] COMPLETE → {out_path}")
        return 0

    # --- прогон ---------------------------------------------------------------
    if not gpu_available() and not args.allow_cpu:
        report = build_report(cell=cell, status="EMPTY-PENDING",
                              phase_legs=phase_legs_from_metrics(metrics_path),
                              note=_empty_pending_note())
        write_report(report, out_path)
        print(format_table(report))
        print(f"[step-profile] EMPTY-PENDING (GPU нет) → {out_path}")
        return 0

    # ADR-041: лимит памяти — ДО import jax; гейт совмещённого стенда — на прогоне.
    _preflight_memory()
    import jax_preflight

    jax_preflight.gate_or_exit()

    status = "SMOKE-CPU" if args.allow_cpu else "COMPLETE"
    metrics_path = metrics_path or (Path(trace_dir).parent / "metrics.jsonl")
    try:
        run_meta = run_window(
            args.config, impl=args.impl, batch=args.batch, seq=args.seq,
            steps=args.steps, warmup=args.warmup, trace_dir=trace_dir,
            metrics_path=metrics_path, phase_profile=not args.no_phase_legs,
            name=args.name, ns_steps=args.ns_steps,
            legacy_muon_all_2d=bool(args.legacy_muon_all_2d),
        )
    except Exception as exc:  # fail-closed: прогон не состоялся — чисел не будет
        report = build_report(
            cell=cell, status="TRACE-ERROR",
            error=f"{type(exc).__name__}: {exc}",
            phase_legs=phase_legs_from_metrics(metrics_path),
            note="прогон клетки не состоялся; отчёт несёт только причину",
        )
        write_report(report, out_path)
        print(f"[step-profile] TRACE-ERROR: {type(exc).__name__}: {exc} → {out_path}",
              file=sys.stderr)
        return 2

    try:
        analysis = analyze_trace(trace_dir, expected_steps=args.steps, step_marker=args.step_marker)
    except ProfileError as exc:
        report = build_report(cell=cell, status="TRACE-ERROR", run_meta=run_meta, error=str(exc),
                              phase_legs=phase_legs_from_metrics(metrics_path))
        write_report(report, out_path)
        print(f"[step-profile] TRACE-ERROR: {exc} → {out_path}", file=sys.stderr)
        return 2

    report = build_report(
        cell=cell, status=status, analysis=analysis, run_meta=run_meta,
        phase_legs=phase_legs_from_metrics(metrics_path),
    )
    write_report(report, out_path)
    print(format_table(report))
    print(f"[step-profile] {status} → {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
