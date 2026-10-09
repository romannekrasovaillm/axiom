#!/usr/bin/env python3
"""MFU-протокол BF16-кампании (фаза 1, Опция A) — прибор замера, не прогон.

Что мерится
-----------
Десять шагов оптимизатора штатной ногой ``net/train_loop.train`` (тот же
``jax.jit(value_and_grad(loss_fn))``, что в претрейне, — прибор не пересказывает
путь, а исполняет его), по клеткам «конфигурация × режим dtype»:

    dense-124m  b1  seq 8192
    dense-124m  b4  seq 8192
    l3-full     b1  seq 8192

    режимы: fp32 (базис) | bf16 | bf16+flash

``bf16`` включает гейт ``AXIOM_COMPUTE_DTYPE=bf16`` (bf16-операнды GEMM с
fp32-аккумулятором, ``net/compute_dtype.py``); ``flash`` — гейт
``AXIOM_MLA_DENSE_FLASH=1`` (fused attention плотного оракула, ``net/mla.py``).

Медиана хвоста
--------------
jit-разгон оседает к шагу 3–4, поэтому в вердикт идёт медиана по шагам с
``step >= --warmup`` (по умолчанию 4-й..10-й), а первые ``--warmup`` шагов
остаются в отчёте отдельным полем ``warmup_tok_s`` — разгон виден, но не
определяет число. Медиана, а не среднее: шаг оптимизатора на GB10 плавает
(соседи по железу, термотроттлинг), среднее такое плавание впитывает.

Знаменатель MFU
---------------
Пик берётся **только** из пиннутого носителя ``evidence/kpi-pins.json``
(``mfu_reference.measured_denominator``: bf16 98.2 TFLOPS, fp32 45.2 — измеренный
cuBLAS-пик, коммит 42b7408). Файл читается и никогда не пишется: подмена
знаменателя под результат — именно то, ради чего пин существует. Если пина нет
или в нём нет нужного dtype — MFU не выдумывается, поле остаётся ``null``.

Журнал
------
Схема журнала метрик не меняется: каждая клетка пишет обычный
``pretrain-metrics/v1`` (``net/train_loop.MetricsWriter``, одна строка на шаг) в
``<out>/journal/<config>__<mode>.jsonl``. Отчёт ссылается на файлы журнала, а не
пересказывает их.

Стенд
-----
GPU-прогоны выполняет архитектор на стенде GB10 (AD-7/C-040: одна модельная
нагрузка за раз — перед запуском возьмите лок ``~/gb10-shared/.locks``). Если
GPU нет, прибор НЕ исполняет клетки: он пишет отчёт со статусом
``EMPTY-PENDING`` (плюс готовый план клеток) и выходит с кодом 0 — «прогон ещё
не сделан» не должно выглядеть как «прогон прошёл» и не должно выглядеть как
поломка. ``--allow-cpu`` снимает этот запрет для отладки самого прибора; такие
числа помечаются ``device: cpu`` и в вердикт по MFU не годятся.

Каждая клетка исполняется в СВОЁМ процессе (``--cell``): env-гейт фиксируется на
весь процесс, и компиляция одной клетки не может переиспользовать граф другой.

Запуск::

    python3 tools/mfu_bf16_protocol.py --selftest
    python3 tools/mfu_bf16_protocol.py --out evidence/mfu-bf16/mfu-report.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any, Optional

CASE_DIR = Path(__file__).resolve().parent.parent

# The model package lives at the case root; a cell may be launched from any cwd
# (the driver pins ``cwd`` for the subprocess, but ``sys.path[0]`` is this
# script's directory, not the cwd), so make the import explicit here.
# ``tools/`` держит префлайт памяти: скриптом этот каталог уже ``sys.path[0]``,
# а импортирующий тест приносит свой список путей — добавляем явно.
for _path in (str(CASE_DIR), str(Path(__file__).resolve().parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# ADR-041: дисциплина памяти JAX — префлайт ДО import jax (лимит XLA).
import jax_preflight  # noqa: E402

jax_preflight.ensure_mem_fraction()

#: Схема отчёта прибора.
REPORT_SCHEMA = "axiom-mfu-bf16-report/1"

#: Пиннутый носитель знаменателя MFU (read-only, ADR-032/форма gemm-peak).
KPI_PINS = CASE_DIR / "evidence" / "kpi-pins.json"

#: Клетки протокола: (имя, конфиг, батч, seq).
CONFIGS: tuple[tuple[str, str, int, int], ...] = (
    ("dense124m-b1", "net/config-dense124m.json", 1, 8192),
    ("dense124m-b4", "net/config-dense124m.json", 4, 8192),
    ("l3full-b1", "net/config.json", 1, 8192),
)

#: Режимы: (имя, env-гейты). fp32 — базис, с ним же сравниваются остальные.
MODES: tuple[tuple[str, dict[str, str]], ...] = (
    ("fp32", {"AXIOM_COMPUTE_DTYPE": "fp32", "AXIOM_MLA_DENSE_FLASH": "0"}),
    ("bf16", {"AXIOM_COMPUTE_DTYPE": "bf16", "AXIOM_MLA_DENSE_FLASH": "0"}),
    ("bf16+flash", {"AXIOM_COMPUTE_DTYPE": "bf16", "AXIOM_MLA_DENSE_FLASH": "1"}),
)

#: Сколько шагов гоняем и сколько первых выбрасываем как jit-разгон.
STEPS = 10
WARMUP = 3

#: Допуск согласованности повторных замеров одной клетки (шум стенда, ±3 %
#: как в сертификации знаменателя 42b7408).
REPEAT_TOLERANCE = 0.05


class ProtocolError(RuntimeError):
    """Вход или окружение непригодны — вердикт не выносится (fail-closed)."""


# --------------------------------------------------------------------------- #
# Знаменатель MFU: только из пиннутого носителя
# --------------------------------------------------------------------------- #


def load_peak_tflops(pins_path: Path = KPI_PINS) -> dict[str, Any]:
    """Пики по dtype из ``evidence/kpi-pins.json`` (read-only).

    Возвращает ``{"bf16": 98.2, "fp32": 45.2, "source": ...}``; отсутствие
    пина — ``{"bf16": None, "fp32": None, ...}``, а не выдуманное число.
    """
    out: dict[str, Any] = {"bf16": None, "fp32": None, "source": None}
    try:
        payload = json.loads(Path(pins_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        out["source"] = f"пин недоступен: {exc}"
        return out
    ref = (payload.get("mfu_reference") or {}).get("measured_denominator") or {}
    out["bf16"] = ref.get("denominator_bf16_best")
    fp32 = ref.get("fp32_tflops") or []
    out["fp32"] = max(fp32) if fp32 else None
    out["source"] = ref.get("source_log") or (payload.get("mfu_reference") or {}).get("added")
    return out


def peak_for_mode(mode: str, peaks: dict[str, Any]) -> Optional[float]:
    return peaks.get("bf16") if mode.startswith("bf16") else peaks.get("fp32")


# --------------------------------------------------------------------------- #
# Медиана хвоста
# --------------------------------------------------------------------------- #


def tail_median(values: list[float], warmup: int = WARMUP) -> Optional[float]:
    """Медиана значений начиная с индекса ``warmup`` (jit-разгон отброшен).

    Короткий ряд (все записи — разгон) даёт ``None``: на разгоне вердикт не
    выносится, а не «выносится по худшему».
    """
    tail = [float(v) for v in values[warmup:] if v is not None and v > 0]
    return median(tail) if tail else None


def warmup_median(values: list[float], warmup: int = WARMUP) -> Optional[float]:
    head = [float(v) for v in values[:warmup] if v is not None and v > 0]
    return median(head) if head else None


# --------------------------------------------------------------------------- #
# Одна клетка: 10 шагов штатной ногой train_loop
# --------------------------------------------------------------------------- #


def synthetic_batches(batch: int, seq: int, steps: int, vocab: int, seed: int = 0):
    """Детерминированные батчи ``(B, T)`` id-токенов.

    Данные синтетические осознанно: прибор измеряет ОСЬ dtype, а не стоимость
    упакованного лоадера, и синтетика делает клетки сравнимыми между собой при
    любом состоянии датасета на стенде. Стоимость даталоадера объявлена
    не входящей в измеряемую величину (см. ``data_source`` в отчёте).
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    for _ in range(steps):
        yield rng.integers(0, vocab, size=(batch, seq), dtype=np.int32)


def run_cell(config_path: str, batch: int, seq: int, mode: str, out_dir: Path,
             steps: int = STEPS) -> dict[str, Any]:
    """Исполнить одну клетку в текущем процессе (env уже выставлен драйвером)."""
    import jax

    from net import train_loop as tl
    from net.config import load_config

    cfg = load_config(CASE_DIR / config_path)
    peaks = load_peak_tflops()
    peak = peak_for_mode(mode, peaks)

    journal_dir = Path(out_dir) / "journal"
    journal_dir.mkdir(parents=True, exist_ok=True)
    journal_path = journal_dir / f"{Path(config_path).stem}__{mode.replace('+', '_')}.jsonl"
    if journal_path.exists():
        journal_path.unlink()

    train_config = tl.TrainConfig(
        steps=steps,
        total_steps=steps,
        seed=0,
        micro_batch=batch,
        grad_checkpointing=bool(getattr(cfg, "grad_ckpt_policy", "none") != "none"),
        param_dtype="float32",
        peak_tflops=peak,
        peak_tflops_source=str(peaks.get("source") or ""),
        metrics_path=journal_path,
        log_every=0,
    )
    budget = tl.Budget(
        run_ref=f"mfu-bf16-{Path(config_path).stem}-{mode}",
        path=Path(out_dir),
        present=True,
        budget_method="протокол MFU (фаза 1 BF16-кампании): смета не расходуется",
        stop_rule="10 шагов прибора",
    )

    started = time.time()
    result = tl.train(
        cfg,
        synthetic_batches(batch, seq, steps, cfg.vocab_size),
        train_config=train_config,
        budget=budget,
    )
    wall = time.time() - started

    rows = [
        json.loads(line)
        for line in journal_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tok_s = [r.get("tokens_per_sec") for r in rows]
    tflops = [r.get("tflops_achieved") for r in rows]
    median_tok_s = tail_median(tok_s, WARMUP)
    return {
        "config": config_path,
        "batch": batch,
        "seq": seq,
        "mode": mode,
        "steps": steps,
        "warmup_steps": WARMUP,
        "device": [repr(d) for d in jax.devices()],
        "backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "matmul_precision": jax.config.jax_default_matmul_precision,
        "param_dtype": train_config.param_dtype,
        "grad_checkpointing": train_config.grad_checkpointing,
        "peak_tflops": peak,
        "tok_s_median_tail": median_tok_s,
        "tok_s_warmup": warmup_median(tok_s, WARMUP),
        "tflops_median_tail": tail_median(tflops, WARMUP),
        "mfu": (
            tail_median(tflops, WARMUP) / peak
            if tail_median(tflops, WARMUP) is not None and peak
            else None
        ),
        "loss_first": rows[0].get("loss") if rows else None,
        "loss_last": rows[-1].get("loss") if rows else None,
        "wall_seconds": wall,
        "journal": str(Path(journal_path).relative_to(CASE_DIR))
        if journal_path.is_relative_to(CASE_DIR)
        else str(journal_path),
        "journal_schema": rows[0].get("schema") if rows else None,
        "steps_done": result.steps_done,
        "stop_reason": result.stop_reason,
    }


# --------------------------------------------------------------------------- #
# Драйвер: план + подпроцессы
# --------------------------------------------------------------------------- #


def plan() -> list[dict[str, Any]]:
    return [
        {"config": cfg, "name": name, "batch": b, "seq": s, "mode": mode}
        for name, cfg, b, s in CONFIGS
        for mode, _env in MODES
    ]


def gpu_available() -> bool:
    """Есть ли исполнимый GPU у этого интерпретатора (без падения при отсутствии).

    Вариант repr не должен решать судьбу зонда: на стенде GB10 JAX называет
    устройство ``CudaDevice(id=0)`` — подстроки ``gpu`` в таком repr нет, и
    прежний зонд давал ложное «GPU нет» при живом GPU. Поэтому принимаем обе
    формы (``gpu``/``cuda``) и сверяемся с бэкендом, который JAX реально выбрал
    (``default_backend() == 'gpu'``).
    """
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


def spawn_cell(cell: dict[str, Any], out_dir: Path, steps: int) -> dict[str, Any]:
    """Клетка в отдельном процессе: env-гейт фиксирован на весь процесс."""
    env = dict(os.environ)
    env.pop("JAX_PLATFORMS", None)
    env.update(dict(MODES)[cell["mode"]])
    tmp = Path(out_dir) / f".cell-{cell['name']}__{cell['mode'].replace('+', '_')}.json"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--cell",
        cell["config"],
        str(cell["batch"]),
        str(cell["seq"]),
        cell["mode"],
        "--steps",
        str(steps),
        "--cell-out",
        str(tmp),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(CASE_DIR))
    if proc.returncode != 0 or not tmp.exists():
        return {
            "config": cell["config"],
            "batch": cell["batch"],
            "seq": cell["seq"],
            "mode": cell["mode"],
            "error": (proc.stderr or proc.stdout or "").strip().splitlines()[-1:]
            or ["неизвестная ошибка клетки"],
            "exit_code": proc.returncode,
        }
    payload = json.loads(tmp.read_text(encoding="utf-8"))
    tmp.unlink(missing_ok=True)
    return payload


def build_report(cells: list[dict[str, Any]], status: str, peaks: dict[str, Any],
                 device: list[str] | None = None) -> dict[str, Any]:
    """Собрать отчёт: план, клетки, сравнение режимов внутри каждой конфигурации."""
    comparisons = []
    for name, cfg, b, s in CONFIGS:
        row: dict[str, Any] = {"config": cfg, "name": name, "batch": b, "seq": s, "modes": {}}
        for cell in cells:
            if cell.get("name") == name and "error" not in cell:
                m = cell.get("mode")
                if m is None:  # клетка, пришедшая из --cell, не несёт name
                    continue
                row["modes"][m] = {
                    "tok_s": cell.get("tok_s_median_tail"),
                    "mfu": cell.get("mfu"),
                    "peak_tflops": cell.get("peak_tflops"),
                }
        base = row["modes"].get("fp32", {}).get("tok_s")
        for m, entry in row["modes"].items():
            entry["speedup_vs_fp32"] = (
                (entry["tok_s"] / base) if base and entry.get("tok_s") else None
            )
        comparisons.append(row)
    return {
        "schema": REPORT_SCHEMA,
        "status": status,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": device or [],
        "data_source": "synthetic (детерминированные id-токены; стоимость упакованного лоадера вне измеряемой величины)",
        "protocol": {
            "steps": STEPS,
            "warmup_steps": WARMUP,
            "metric": "медиана хвоста (шаги >= warmup) ток/с и MFU; разгон — отдельным полем",
            "modes": {name: env for name, env in MODES},
            "configs": [{"name": n, "config": c, "batch": b, "seq": s} for n, c, b, s in CONFIGS],
            "runner": "net/train_loop.train (та же нога, что в претрейне)",
            "journal_schema": "pretrain-metrics/v1 (не изменяется)",
        },
        "denominator": peaks,
        "comparisons": comparisons,
        "cells": cells,
    }


def write_report(report: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def selftest() -> int:
    checks: list[tuple[str, bool]] = []

    peaks = load_peak_tflops()
    checks.append(("пин знаменателя прочитан (bf16)", peaks.get("bf16") == 98.2))
    checks.append(("пин знаменателя прочитан (fp32)", peaks.get("fp32") == 45.2))
    checks.append(("пин отсутствует — MFU не выдуман", load_peak_tflops(Path("/nonexistent.json"))["bf16"] is None))

    checks.append(("медиана хвоста отбрасывает разгон", tail_median([1.0, 2.0, 3.0, 10.0, 12.0, 11.0], 3) == 11.0))
    checks.append(("короткий ряд — вердикта нет", tail_median([1.0, 2.0, 3.0], 3) is None))
    checks.append(("нули/None не портят медиану", tail_median([1.0, 1.0, 1.0, None, 0.0, 5.0], 3) == 5.0))
    checks.append(("разгон измеряется отдельно", warmup_median([1.0, 2.0, 3.0, 9.0], 3) == 2.0))

    checks.append(("режим bf16 берёт bf16-пик", peak_for_mode("bf16", peaks) == 98.2))
    checks.append(("режим bf16+flash берёт bf16-пик", peak_for_mode("bf16+flash", peaks) == 98.2))
    checks.append(("режим fp32 берёт fp32-пик", peak_for_mode("fp32", peaks) == 45.2))

    p = plan()
    checks.append(("план: 3 конфигурации × 3 режима", len(p) == 9))
    checks.append(("базис fp32 присутствует в каждом наборе", sum(c["mode"] == "fp32" for c in p) == 3))

    # Отчёт без прогона: статус EMPTY-PENDING, план на месте, числа не выдуманы.
    empty = build_report([], "EMPTY-PENDING", peaks)
    checks.append(("пустой отчёт — EMPTY-PENDING", empty["status"] == "EMPTY-PENDING"))
    checks.append(("пустой отчёт не содержит tok_s", all(
        e.get("tok_s") is None
        for row in empty["comparisons"] for e in row["modes"].values()
    )))
    checks.append(("конфигурации в отчёте = конфигурации плана", len(empty["comparisons"]) == len(CONFIGS)))

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
    parser.add_argument("--out", default=str(CASE_DIR / "evidence" / "mfu-bf16" / "mfu-report.json"))
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="разрешить прогон без GPU (отладка прибора; числа помечаются cpu)",
    )
    parser.add_argument("--cell", nargs=4, metavar=("CONFIG", "BATCH", "SEQ", "MODE"),
                        help="внутренний режим: исполнить одну клетку в этом процессе")
    parser.add_argument("--cell-out", default=None)
    args = parser.parse_args(argv)

    if args.selftest:
        return selftest()

    if args.cell is not None:
        config, batch, seq, mode = args.cell
        out_dir = Path(args.cell_out).parent if args.cell_out else Path(args.out).parent
        payload = run_cell(config, int(batch), int(seq), mode, out_dir, steps=args.steps)
        if args.cell_out:
            Path(args.cell_out).write_text(
                json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
            )
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=1))
        return 0

    out_path = Path(args.out)
    peaks = load_peak_tflops()

    if not gpu_available() and not args.allow_cpu:
        report = build_report([], "EMPTY-PENDING", peaks)
        report["note"] = (
            "GPU недоступен: клетки не исполнены. Отчёт несёт готовый план "
            "(protocol.configs × protocol.modes); прогон выполняет архитектор на "
            "стенде GB10 (AD-7/C-040: лок на ~/gb10-shared/.locks перед запуском)."
        )
        write_report(report, out_path)
        print(f"[mfu-bf16] EMPTY-PENDING (GPU нет) → {out_path}")
        return 0

    cells: list[dict[str, Any]] = []
    for cell in plan():
        print(f"[mfu-bf16] клетка {cell['name']} · {cell['mode']} …", flush=True)
        try:
            cells.append({**spawn_cell(cell, out_path.parent, args.steps), "name": cell["name"]})
        except ProtocolError as exc:
            cells.append({**cell, "error": [str(exc)]})
    failures = [c for c in cells if "error" in c]
    status = "COMPLETE" if not failures else ("PARTIAL" if len(failures) < len(cells) else "FAILED")
    report = build_report(cells, status, peaks)
    report["failures"] = failures
    write_report(report, out_path)
    print(f"[mfu-bf16] {status} → {out_path}")
    return 0 if status == "COMPLETE" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
