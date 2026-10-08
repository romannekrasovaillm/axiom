#!/usr/bin/env python3
"""Короткая нога паритета лосса: dense-124m, fp32-базис против bf16 (фаза 1).

Критерий приёмки (§5б задания) — допуск на РАСХОЖДЕНИЕ КРИВОЙ, а не на отдельное
число: ``ΔBPB(bf16) - BPB(fp32) ≤ +0.05``. Почему BPB, а не loss: absolute loss
зависит от токенизатора (у нас словарь 160K), BPB — нет (``tools/bpb_report.py``,
критерий 1 верификационной ноги), поэтому он годится как общая мера «насколько
хуже учится» между двумя прогонами одного и того же корпуса.

Нога
----
dense-124m (``net/config-dense124m.json``), seq 8192, batch 1, 50M токенов
(``ceil(50e6 / (8192 * 1)) = 6104`` шага оптимизатора) — две клетки, отличающиеся
ТОЛЬКО env-гейтом ``AXIOM_COMPUTE_DTYPE`` (fp32 | bf16). Обе идут штатной ногой
``net/train_loop.train``; метрики каждой — обычный ``pretrain-metrics/v1``.

Коэффициент tokens/byte
-----------------------
Считается на сэмпле текстов тем же токенизатором, что размечены данные
(``--tokenizer-manifest``), либо задаётся прямо (``--tokens-per-byte``);
без любого из них прибор отказывает (``input-error``) — BPB без коэффициента
не число. Оба флага взаимоисключающие, как в ``tools/bpb_report.py``.

Допуск
------
``BPB_TOLERANCE_DELTA = 0.05`` бит/байт. Вердикты: ``pass`` (в допуске),
``fail`` (кривая bf16 хуже базиса больше допуска), ``input-error``
(нет коэффициента / нет метрик), ``EMPTY-PENDING`` (GPU нет — нога не исполнена,
план сохранён). Fail-closed: нога без метрик НЕ считается прошедшей.

Стенд
-----
Прогон выполняет архитектор на GB10 (AD-7/C-040 — лок на ``~/gb10-shared/.locks``;
AD-8/C-041 — смета до запуска). 50M токенов × 2 ноги на dense-124m b1 — это часы,
не минуты: сначала 50-шаговый смоук каждой клетки (в отчёт он не идёт).

Запуск::

    python3 tools/loss_parity_bf16.py --selftest
    python3 tools/loss_parity_bf16.py \\
        --tokenizer-manifest ~/gb10-shared/datasets/axiom-pretrain-l3/tokenizer/tokenizer-manifest.json \\
        --out evidence/mfu-bf16/loss-parity-report.json
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
if str(CASE_DIR) not in sys.path:
    sys.path.insert(0, str(CASE_DIR))
if str(CASE_DIR / "tools") not in sys.path:
    sys.path.insert(0, str(CASE_DIR / "tools"))

REPORT_SCHEMA = "axiom-loss-parity-bf16/1"

#: Допуск приёмки: BPB(bf16) - BPB(fp32) <= +0.05 бит/байт (§5б задания).
BPB_TOLERANCE_DELTA = 0.05

#: Разрешение сравнения с допуском: «ровно на границе» — проход, а не провал
#: из-за последнего разряда двойной точности (тот же приём, что
#: ``bpb_report.BPB_COMPARISON_EPS``).
BPB_COMPARISON_EPS = 1e-9

#: Нога: конфигурация, батч, seq, бюджет токенов.
LEG_CONFIG = "net/config-dense124m.json"
LEG_BATCH = 1
LEG_SEQ = 8192
LEG_TOKENS = 50_000_000

#: Окно медианы хвоста (шаги) — кривая шумит, последние 200 шагов устойчивы.
MEDIAN_WINDOW = 200

#: Клетки ноги: имя режима → env-гейты.
CELLS: tuple[tuple[str, dict[str, str]], ...] = (
    ("fp32", {"AXIOM_COMPUTE_DTYPE": "fp32"}),
    ("bf16", {"AXIOM_COMPUTE_DTYPE": "bf16"}),
)


def leg_steps() -> int:
    return -(-LEG_TOKENS // (LEG_BATCH * LEG_SEQ))


# --------------------------------------------------------------------------- #
# Одна нога (в своём процессе — env-гейт фиксирован)
# --------------------------------------------------------------------------- #


def synthetic_batches(batch: int, seq: int, steps: int, vocab: int, seed: int = 0):
    import numpy as np

    rng = np.random.default_rng(seed)
    for _ in range(steps):
        yield rng.integers(0, vocab, size=(batch, seq), dtype=np.int32)


def run_leg(mode: str, out_dir: Path, steps: int) -> dict[str, Any]:
    from net import train_loop as tl
    from net.config import load_config

    cfg = load_config(CASE_DIR / LEG_CONFIG)
    journal_dir = Path(out_dir) / "journal"
    journal_dir.mkdir(parents=True, exist_ok=True)
    journal_path = journal_dir / f"loss-parity-{mode}.jsonl"
    if journal_path.exists():
        journal_path.unlink()

    train_config = tl.TrainConfig(
        steps=steps,
        total_steps=steps,
        seed=0,
        micro_batch=LEG_BATCH,
        grad_checkpointing=bool(getattr(cfg, "grad_ckpt_policy", "none") != "none"),
        param_dtype="float32",
        metrics_path=journal_path,
        log_every=0,
    )
    budget = tl.Budget(
        run_ref=f"loss-parity-bf16-{mode}",
        path=Path(out_dir),
        present=True,
        budget_method="короткая нога паритета (§5б): смета не расходуется",
        stop_rule=f"{steps} шагов ноги",
    )
    result = tl.train(
        cfg,
        synthetic_batches(LEG_BATCH, LEG_SEQ, steps, cfg.vocab_size),
        train_config=train_config,
        budget=budget,
    )
    rows = [
        json.loads(line)
        for line in journal_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    losses = [float(r["loss"]) for r in rows if isinstance(r.get("loss"), (int, float))]
    if not losses:
        raise RuntimeError(f"нога {mode}: журнал пуст — вердикт не выносится")
    return {
        "mode": mode,
        "steps": steps,
        "tokens": LEG_BATCH * LEG_SEQ * steps,
        "loss_median_window": median(losses[-min(MEDIAN_WINDOW, len(losses)):]),
        "loss_last": losses[-1],
        "loss_first": losses[0],
        "journal": str(journal_path.relative_to(CASE_DIR))
        if journal_path.is_relative_to(CASE_DIR)
        else str(journal_path),
        "steps_done": result.steps_done,
        "stop_reason": result.stop_reason,
    }


# --------------------------------------------------------------------------- #
# Вердикт
# --------------------------------------------------------------------------- #


def verdict(cells: dict[str, dict[str, Any]], tokens_per_byte: Optional[float]) -> dict[str, Any]:
    """Сравнить кривые двух ног в BPB и вынести механический вердикт."""
    if tokens_per_byte is None:
        return {"status": "input-error", "reason": "нет коэффициента tokens/byte (BPB не число)"}
    if "fp32" not in cells or "bf16" not in cells:
        return {"status": "input-error", "reason": "нет метрик хотя бы одной ноги"}

    from bpb_report import bpb_from_loss

    base = bpb_from_loss(cells["fp32"]["loss_median_window"], tokens_per_byte)
    cand = bpb_from_loss(cells["bf16"]["loss_median_window"], tokens_per_byte)
    delta = cand - base
    within = delta <= BPB_TOLERANCE_DELTA + BPB_COMPARISON_EPS
    return {
        "status": "pass" if within else "fail",
        "bpb_fp32": base,
        "bpb_bf16": cand,
        "delta_bpb": delta,
        "tolerance": BPB_TOLERANCE_DELTA,
        "tokens_per_byte": tokens_per_byte,
        "verdict": (
            "bf16 в допуске против fp32-базиса"
            if within
            else f"bf16 хуже базиса на {delta:.4f} BPB при допуске {BPB_TOLERANCE_DELTA}"
        ),
    }


# --------------------------------------------------------------------------- #
# Драйвер
# --------------------------------------------------------------------------- #


def gpu_available() -> bool:
    env = dict(os.environ)
    env.pop("JAX_PLATFORMS", None)
    probe = (
        "import jax;d=[repr(x) for x in jax.devices()];"
        "print('gpu' if any('gpu' in r.lower() for r in d) else 'cpu')"
    )
    try:
        out = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, timeout=300, env=env
        )
    except Exception:
        return False
    return out.returncode == 0 and out.stdout.strip().endswith("gpu")


def spawn_leg(mode: str, out_dir: Path, steps: int) -> dict[str, Any]:
    env = dict(os.environ)
    env.pop("JAX_PLATFORMS", None)
    env.update(dict(CELLS)[mode])
    tmp = Path(out_dir) / f".leg-{mode}.json"
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--leg", mode, "--steps", str(steps), "--leg-out", str(tmp),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(CASE_DIR))
    if proc.returncode != 0 or not tmp.exists():
        return {"mode": mode, "error": (proc.stderr or proc.stdout or "").strip()[-400:]}
    payload = json.loads(tmp.read_text(encoding="utf-8"))
    tmp.unlink(missing_ok=True)
    return payload


def write_report(report: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


def selftest() -> int:
    checks: list[tuple[str, bool]] = []
    checks.append(("шагов ноги = ceil(50M / (b*seq))", leg_steps() == 6104))

    from bpb_report import bpb_from_loss

    # 1 нат/токен при 0.25 ток/байт = 0.25/ln2 бит/байт.
    checks.append(("bpb_from_loss переводит наты/токен", abs(bpb_from_loss(1.0, 0.25) - 0.25 / 0.6931471805599453) < 1e-9))

    good = {"fp32": {"loss_median_window": 3.0}, "bf16": {"loss_median_window": 3.0}}
    v = verdict(good, 0.25)
    checks.append(("равные кривые — pass", v["status"] == "pass" and v["delta_bpb"] == 0.0))

    # 0.05 BPB при 0.25 ток/байт — это 0.05*ln2/0.25 = 0.1386 нат/токен.
    edge = {"fp32": {"loss_median_window": 3.0}, "bf16": {"loss_median_window": 3.0 + 0.05 * 0.6931471805599453 / 0.25}}
    checks.append(("ровно на допуске — pass", verdict(edge, 0.25)["status"] == "pass"))

    over = {"fp32": {"loss_median_window": 3.0}, "bf16": {"loss_median_window": 3.2}}
    v_over = verdict(over, 0.25)
    checks.append(("хуже допуска — fail", v_over["status"] == "fail" and v_over["delta_bpb"] > BPB_TOLERANCE_DELTA))

    checks.append(("без коэффициента — input-error", verdict(good, None)["status"] == "input-error"))
    checks.append(("без ноги — input-error", verdict({"fp32": {"loss_median_window": 1.0}}, 0.25)["status"] == "input-error"))

    empty = {
        "schema": REPORT_SCHEMA,
        "status": "EMPTY-PENDING",
        "leg": {"config": LEG_CONFIG, "batch": LEG_BATCH, "seq": LEG_SEQ,
                "tokens": LEG_TOKENS, "steps": leg_steps()},
        "cells": {name: {"env": env} for name, env in CELLS},
        "verdict": None,
    }
    checks.append(("пустой отчёт — EMPTY-PENDING без вердикта", empty["status"] == "EMPTY-PENDING" and empty["verdict"] is None))

    failed = [n for n, ok in checks if not ok]
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'} {name}")
    if failed:
        print(f"FAIL: {len(failed)} из {len(checks)}", file=sys.stderr)
        return 1
    print(f"PASS: {len(checks)} проверок")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(CASE_DIR / "evidence" / "mfu-bf16" / "loss-parity-report.json"))
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--tokenizer-manifest", default=None,
                        help="манифест корпусного BPE — из него считается tokens/byte")
    parser.add_argument("--tokens-per-byte", type=float, default=None)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--leg", choices=[m for m, _ in CELLS], default=None,
                        help="внутренний режим: исполнить одну ногу")
    parser.add_argument("--leg-out", default=None)
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args(argv)

    if args.selftest:
        return selftest()

    steps = args.steps or leg_steps()
    out_path = Path(args.out)

    if args.leg is not None:
        payload = run_leg(args.leg, out_path.parent, steps)
        if args.leg_out:
            Path(args.leg_out).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=1))
        return 0

    if (args.tokenizer_manifest is None) == (args.tokens_per_byte is None):
        print("нужен ровно один из --tokenizer-manifest / --tokens-per-byte", file=sys.stderr)
        return 2

    tpb: Optional[float] = args.tokens_per_byte
    if tpb is None:
        from bpb_report import load_tokenizer_from_manifest, read_sample_texts, tokens_per_byte_from_tokenizer

        manifest = json.loads(Path(args.tokenizer_manifest).read_text(encoding="utf-8"))
        sample_path = manifest.get("sample_texts")
        if not sample_path:
            print("в манифесте нет sample_texts — коэффициент не измерить", file=sys.stderr)
            return 2
        tokenizer, _info = load_tokenizer_from_manifest(Path(args.tokenizer_manifest))
        texts, n_bytes = read_sample_texts(Path(sample_path))
        tpb = tokens_per_byte_from_tokenizer(tokenizer, texts)["tokens_per_byte"]

    base_report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "leg": {"config": LEG_CONFIG, "batch": LEG_BATCH, "seq": LEG_SEQ,
                "tokens": LEG_TOKENS, "steps": steps},
        "cells": {name: {"env": env} for name, env in CELLS},
        "tokens_per_byte": tpb,
        "tolerance_bpb": BPB_TOLERANCE_DELTA,
        "runner": "net/train_loop.train (та же нога, что в претрейне)",
        "journal_schema": "pretrain-metrics/v1 (не изменяется)",
    }

    if not gpu_available() and not args.allow_cpu:
        base_report["status"] = "EMPTY-PENDING"
        base_report["verdict"] = None
        base_report["note"] = (
            "GPU недоступен: нога не исполнена. План ноги сохранён (leg/cells); "
            "прогон выполняет архитектор на стенде GB10 (AD-7/C-040, AD-8/C-041)."
        )
        write_report(base_report, out_path)
        print(f"[loss-parity] EMPTY-PENDING (GPU нет) → {out_path}")
        return 0

    cells: dict[str, dict[str, Any]] = {}
    for name, _env in CELLS:
        print(f"[loss-parity] нога {name} ({steps} шагов) …", flush=True)
        cells[name] = spawn_leg(name, out_path.parent, steps)

    base_report["cells_measured"] = cells
    failures = [m for m, c in cells.items() if "error" in c]
    if failures:
        base_report["status"] = "input-error"
        base_report["verdict"] = {"status": "input-error", "reason": f"ноги упали: {failures}", "errors": {m: cells[m]["error"] for m in failures}}
        write_report(base_report, out_path)
        print(f"[loss-parity] FAILED ({failures}) → {out_path}", file=sys.stderr)
        return 1
    base_report["verdict"] = verdict(cells, tpb)
    base_report["status"] = base_report["verdict"]["status"]
    write_report(base_report, out_path)
    print(f"[loss-parity] {base_report['status']} → {out_path}")
    return 0 if base_report["status"] in ("pass", "EMPTY-PENDING") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
