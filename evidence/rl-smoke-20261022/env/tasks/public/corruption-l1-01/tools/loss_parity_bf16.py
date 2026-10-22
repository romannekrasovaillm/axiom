#!/usr/bin/env python3
"""Короткая нога паритета лосса: fp32-базис против bf16 (BF16-кампания, фаза 1).

Критерий приёмки (§5б задания) — допуск на РАСХОЖДЕНИЕ КРИВОЙ, а не на отдельное
число: ``ΔBPB(bf16) - BPB(fp32) ≤ +0.05``. Почему BPB, а не loss: absolute loss
зависит от токенизатора (у нас словарь 160K), BPB — нет (``tools/bpb_report.py``,
критерий 1 верификационной ноги), поэтому он годится как общая мера «насколько
хуже учится» между двумя прогонами одного и того же корпуса.

Нога
----
Нога **параметризована**, а не захардкожена: ``--config`` (путь к
``net/config-*.json``, дефолт ``net/config-dense124m.json``), ``--total-tokens``
(дефолт 50M), ``--shard-root``/``--tokens-root`` и ``--streams``.  Число шагов
выводится из объёма: ``steps = ceil(total_tokens / (batch * seq))`` (batch 1,
seq 8192) — у ноги две клетки, отличающиеся ТОЛЬКО env-гейтом
``AXIOM_COMPUTE_DTYPE`` (fp32 | bf16), и обе идут штатной ногой
``net/train_loop.train``; метрики каждой — обычный ``pretrain-metrics/v1``.

Данные
------
При заданных ``--shard-root``/``--tokens-root`` train кормится **тем же
упакованным лоадером**, что и ``tools/pretrain_run.py``
(``net.train_loop.PackedTokenLoader``, корпус tokens-v2).  Поток по умолчанию —
``W`` (``--streams`` переопределяет); лоадер не шаффлит упакованные записи,
поэтому обе ноги (fp32/bf16) собираются одним сидом и читают **одну и ту же**
последовательность батчей — парный дизайн: расхождение кривых принадлежит
dtype-гейту, а не данным.  Симлинки на ``~/gb10-shared`` не разворачиваются в
копии (C-032) — путь читается на месте.

Fail-closed по evidence
-----------------------
Без флагов корпуса данные синтетические (случайные id), и вердикта паритета они
не несут: статус обязан быть ``input-error`` («синтетические данные не несут
вердикта паритета»), НИКОГДА ``pass``.  Синтетический прогон допустим только как
смоук-проверка механики.  Перед ногой исполняется 50-шаговый смоук каждой клетки;
в отчёт он не идёт (verification-leg §Предостережения 1).

Коэффициент tokens/byte
-----------------------
Считается на сэмпле текстов тем же токенизатором, что размечены данные
(``--tokenizer-manifest``), либо задаётся прямо (``--tokens-per-byte``);
без любого из них прибор отказывает (``input-error``) — BPB без коэффициента
не число.  Оба флага взаимоисключающие, как в ``tools/bpb_report.py``.

Допуск
------
``BPB_TOLERANCE_DELTA = 0.05`` бит/байт. Вердикты: ``pass`` (в допуске),
``fail`` (кривая bf16 хуже базиса больше допуска), ``input-error``
(синтетика / нет коэффициента / нет метрик), ``EMPTY-PENDING`` (GPU нет — нога
не исполнена, план сохранён). Fail-closed: нога без метрик НЕ считается прошедшей.

Стенд
-----
Прогон выполняет архитектор на GB10 (AD-7/C-040 — лок на ``~/gb10-shared/.locks``;
AD-8/C-041 — смета до запуска).  Нога «l3full, 5M токенов, batch 1, seq 8192» —
611 шагов; сначала 50-шаговый смоук каждой клетки, чтобы оценить с/шаг.

Запуск::

    python3 tools/loss_parity_bf16.py --selftest
    python3 tools/loss_parity_bf16.py \\
        --config net/config.json --total-tokens 5e6 \\
        --tokens-root ~/gb10-shared/datasets/axiom-pretrain-l3/tokens-v2 \\
        --tokenizer-manifest ~/gb10-shared/datasets/axiom-pretrain-l3/tokenizer/tokenizer-manifest.json \\
        --out evidence/mfu-bf16/loss-parity-report.json
"""

from __future__ import annotations

import argparse
import json
import math
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

# ADR-041: дисциплина памяти JAX — префлайт ДО import jax (лимит XLA).
import jax_preflight  # noqa: E402

jax_preflight.ensure_mem_fraction()

REPORT_SCHEMA = "axiom-loss-parity-bf16/1"

#: Допуск приёмки: BPB(bf16) - BPB(fp32) <= +0.05 бит/байт (§5б задания).
BPB_TOLERANCE_DELTA = 0.05

#: Разрешение сравнения с допуском: «ровно на границе» — проход, а не провал
#: из-за последнего разряда двойной точности (тот же приём, что
#: ``bpb_report.BPB_COMPARISON_EPS``).
BPB_COMPARISON_EPS = 1e-9

#: Дефолты ноги — переопределяются CLI (``--config`` / ``--total-tokens`` /
#: ``--streams``).  Объём и конфиг НЕ захардкожены: шаги вычисляются, поэтому
#: нога «l3full, 5M токенов» — это ``--config net/config.json --total-tokens 5e6``.
DEFAULT_CONFIG = "net/config-dense124m.json"
#: Совместимость: `LEG_CONFIG` = дефолт ``--config``.
LEG_CONFIG = DEFAULT_CONFIG
LEG_BATCH = 1
LEG_SEQ = 8192
LEG_TOKENS = 50_000_000  # дефолт --total-tokens
#: Дефолт --streams: поток W (как в verification-leg: --data …/tokens-v2/W).
#: Смешивать потоки ноге паритета не нужно — один поток строго детерминирован.
DEFAULT_STREAMS = "W"

#: Один сид на обе ноги (fp32/bf16) — инициализация и обход корпуса парные.
PARITY_SEED = 0

#: Окно медианы хвоста (шаги) — кривая шумит, последние 200 шагов устойчивы.
MEDIAN_WINDOW = 200

#: Смоук механики перед ногой: 50 шагов на клетку, в отчёт не идёт
#: (verification-leg §Предостережения 1 — сначала смоук, потом часы ноги).
SMOKE_STEPS = 50

#: Причина отказа вердикта на синтетических данных (fail-closed по evidence).
SYNTHETIC_NO_VERDICT = "синтетические данные не несут вердикта паритета"

#: Клетки ноги: имя режима → env-гейты.
CELLS: tuple[tuple[str, dict[str, str]], ...] = (
    ("fp32", {"AXIOM_COMPUTE_DTYPE": "fp32"}),
    ("bf16", {"AXIOM_COMPUTE_DTYPE": "bf16"}),
)


def leg_steps(
    total_tokens: float = LEG_TOKENS, batch: int = LEG_BATCH, seq: int = LEG_SEQ
) -> int:
    """Шаги ноги: ``ceil(total_tokens / (batch * seq))`` — вычисление, не константа."""
    if batch < 1 or seq < 1:
        raise ValueError("batch и seq должны быть >= 1")
    if total_tokens <= 0:
        raise ValueError("total_tokens должен быть > 0")
    return int(math.ceil(total_tokens / (batch * seq)))


def parse_streams(raw: str) -> tuple[str, ...]:
    """Потоки корпуса через запятую → кортеж имён (пустой список — ошибка)."""
    streams = tuple(name.strip() for name in str(raw).split(",") if name.strip())
    if not streams:
        raise ValueError("пустой список потоков")
    return streams


def default_mix(streams: tuple[str, ...]) -> dict[str, float]:
    """Веса микса: набор W+C — пропорция ADR-021, иначе равные доли."""
    if set(streams) == {"W", "C"}:
        return {"W": 0.85, "C": 0.15}
    return {name: 1.0 for name in streams}


def resolve_tokens_root(
    shard_root: Optional[str], tokens_root: Optional[str]
) -> Optional[Path]:
    """Корень упакованного корпуса: явный ``--tokens-root`` сильнее ``--shard-root/tokens``.

    Симлинки НЕ разворачиваются (C-032): путь берётся как объявлен, копий данных
    не создаётся — читаем на месте.  ``None`` (ни одного флага) — синтетика.
    """
    if tokens_root:
        return Path(tokens_root)
    if shard_root:
        return Path(shard_root) / "tokens"
    return None


# --------------------------------------------------------------------------- #
# План ноги (детерминирован по построению)
# --------------------------------------------------------------------------- #


def build_plan(
    *,
    config: str,
    batch: int,
    seq: int,
    total_tokens: float,
    streams: tuple[str, ...],
    corpus: Optional[str],
) -> dict[str, Any]:
    """План ноги и её клеток — чистый факт параметров.

    ``corpus`` — путь к упакованному корпусу (``tokens/``) либо ``None``:
    ``None`` означает синтетику, и вердикт паритета по такой ноге не выносится
    (fail-closed).  Клетки отличаются ТОЛЬКО env-гейтом ``AXIOM_COMPUTE_DTYPE`` —
    иначе нога не парная.
    """
    streams = tuple(streams)
    return {
        "config": str(config),
        "batch": int(batch),
        "seq": int(seq),
        "total_tokens": int(total_tokens),
        "steps": leg_steps(total_tokens, batch, seq),
        "streams": list(streams),
        "seed": PARITY_SEED,
        "corpus": str(corpus) if corpus else None,
        "data": "corpus" if corpus else "synthetic",
        "cells": [{"mode": mode, "env": dict(env)} for mode, env in CELLS],
    }


# --------------------------------------------------------------------------- #
# Батчи ноги: реальный корпус или синтетика
# --------------------------------------------------------------------------- #


def synthetic_batches(batch: int, seq: int, steps: int, vocab: int, seed: int = PARITY_SEED):
    import numpy as np

    rng = np.random.default_rng(seed)
    for _ in range(steps):
        yield rng.integers(0, vocab, size=(batch, seq), dtype=np.int32)


def corpus_batches(
    tokens_root: Path | str,
    streams: tuple[str, ...],
    seq_len: int,
    batch: int,
):
    """Тот же упакованный лоадер, что у ``tools/pretrain_run.py`` (packed-путь).

    ``PackedTokenLoader`` не шаффлит упакованные записи: порядок шардов —
    манифест, выбор потока — детерминированный дефицит микса.  Значит, две сборки
    на одном ``tokens_root`` дают побайтово одинаковый поток — парный дизайн ноги.
    """
    from net import train_loop as tl

    streams = tuple(streams)
    return tl.PackedTokenLoader(
        tokens_root=tokens_root,
        streams=streams,
        seq_len=int(seq_len),
        batch_size=int(batch),
        mix=default_mix(streams),
    )


def build_batches(
    cfg,
    *,
    steps: int,
    batch: int,
    seq: int,
    tokens_root: Optional[Path],
    streams: tuple[str, ...],
) -> tuple[Any, str]:
    """Батчи ноги и вид источника: ``corpus`` (packed) или ``synthetic``."""
    if tokens_root is None:
        return synthetic_batches(batch, seq, steps, cfg.vocab_size), "synthetic"
    return corpus_batches(tokens_root, streams, seq, batch), "corpus"


# --------------------------------------------------------------------------- #
# Одна нога (в своём процессе — env-гейт фиксирован)
# --------------------------------------------------------------------------- #


def run_leg(
    mode: str,
    out_dir: Path,
    steps: int,
    *,
    config: Optional[str] = None,
    tokens_root: Optional[str] = None,
    streams: Optional[tuple[str, ...]] = None,
    batch: Optional[int] = None,
    seq: Optional[int] = None,
) -> dict[str, Any]:
    from net import train_loop as tl
    from net.config import load_config

    cfg_path = Path(config if config is not None else LEG_CONFIG)
    if not cfg_path.is_absolute():
        cfg_path = CASE_DIR / cfg_path
    cfg = load_config(cfg_path)

    batch_v = int(batch if batch is not None else LEG_BATCH)
    seq_v = int(seq if seq is not None else LEG_SEQ)
    streams_v = tuple(streams) if streams else parse_streams(DEFAULT_STREAMS)
    tokens_root_v = Path(tokens_root) if tokens_root is not None else None

    journal_dir = Path(out_dir) / "journal"
    journal_dir.mkdir(parents=True, exist_ok=True)
    journal_path = journal_dir / f"loss-parity-{mode}.jsonl"
    if journal_path.exists():
        journal_path.unlink()

    train_config = tl.TrainConfig(
        steps=steps,
        total_steps=steps,
        seed=PARITY_SEED,
        micro_batch=batch_v,
        grad_checkpointing=bool(getattr(cfg, "grad_ckpt_policy", "none") != "none"),
        param_dtype="float32",
        metrics_path=journal_path,
        log_every=0,
        data_kind="packed" if tokens_root_v is not None else "raw",
    )
    budget = tl.Budget(
        run_ref=f"loss-parity-bf16-{mode}",
        path=Path(out_dir),
        present=True,
        budget_method="короткая нога паритета (§5б): смета не расходуется",
        stop_rule=f"{steps} шагов ноги",
    )
    batches, data_kind = build_batches(
        cfg, steps=steps, batch=batch_v, seq=seq_v, tokens_root=tokens_root_v, streams=streams_v
    )
    result = tl.train(cfg, iter(batches), train_config=train_config, budget=budget)
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
        "config": str(config if config is not None else LEG_CONFIG),
        "data_kind": data_kind,
        "seed": PARITY_SEED,
        "steps": steps,
        "tokens": batch_v * seq_v * steps,
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


def verdict(
    cells: dict[str, dict[str, Any]],
    tokens_per_byte: Optional[float],
    *,
    real_corpus: bool = False,
) -> dict[str, Any]:
    """Сравнить кривые двух ног в BPB и вынести механический вердикт.

    Fail-closed по evidence: без **объявленного** корпуса вердикта паритета нет —
    синтетика (случайные id) не доказывает паритет обучения на реальном корпусе,
    и ``pass`` на ней был бы фальсификацией.  Поэтому ``real_corpus`` по умолчанию
    ``False``: забыть объявить корпус безопаснее, чем выдать вердикт по синтетике.
    """
    if not real_corpus:
        return {"status": "input-error", "reason": SYNTHETIC_NO_VERDICT}
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
    """Есть ли исполнимый GPU у этого интерпретатора (без падения при отсутствии).

    Вариант repr не должен решать судьбу зонда: на стенде GB10 JAX называет
    устройство ``CudaDevice(id=0)`` — подстроки ``gpu`` в таком repr нет, и
    прежний зонд давал ложное «GPU нет» при живом GPU. Поэтому принимаем обе
    формы (``gpu``/``cuda``) и сверяемся с бэкендом, который JAX реально выбрал
    (``default_backend() == 'gpu'``). Тот же зонд, что в ``mfu_bf16_protocol.py``.
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


def spawn_leg(
    mode: str,
    out_dir: Path,
    steps: int,
    *,
    config: Optional[str] = None,
    tokens_root: Optional[Path | str] = None,
    streams: Optional[tuple[str, ...]] = None,
) -> dict[str, Any]:
    env = dict(os.environ)
    env.pop("JAX_PLATFORMS", None)
    env.update(dict(CELLS)[mode])
    tmp = Path(out_dir) / f".leg-{mode}.json"
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--leg", mode, "--steps", str(steps), "--leg-out", str(tmp),
        "--config", str(config if config is not None else LEG_CONFIG),
    ]
    if streams:
        cmd += ["--streams", ",".join(streams)]
    if tokens_root is not None:
        cmd += ["--tokens-root", str(tokens_root)]
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
    checks.append(
        ("шаги l3full 5M (b1, seq8192) = 611", leg_steps(5_000_000, 1, 8192) == 611)
    )

    from bpb_report import bpb_from_loss

    # 1 нат/токен при 0.25 ток/байт = 0.25/ln2 бит/байт.
    checks.append(("bpb_from_loss переводит наты/токен", abs(bpb_from_loss(1.0, 0.25) - 0.25 / 0.6931471805599453) < 1e-9))

    good = {"fp32": {"loss_median_window": 3.0}, "bf16": {"loss_median_window": 3.0}}
    v = verdict(good, 0.25, real_corpus=True)
    checks.append(("равные кривые — pass", v["status"] == "pass" and v["delta_bpb"] == 0.0))

    # 0.05 BPB при 0.25 ток/байт — это 0.05*ln2/0.25 = 0.1386 нат/токен.
    edge = {"fp32": {"loss_median_window": 3.0}, "bf16": {"loss_median_window": 3.0 + 0.05 * 0.6931471805599453 / 0.25}}
    checks.append(("ровно на допуске — pass", verdict(edge, 0.25, real_corpus=True)["status"] == "pass"))

    over = {"fp32": {"loss_median_window": 3.0}, "bf16": {"loss_median_window": 3.2}}
    v_over = verdict(over, 0.25, real_corpus=True)
    checks.append(("хуже допуска — fail", v_over["status"] == "fail" and v_over["delta_bpb"] > BPB_TOLERANCE_DELTA))

    checks.append(("без коэффициента — input-error", verdict(good, None, real_corpus=True)["status"] == "input-error"))
    checks.append(("без ноги — input-error", verdict({"fp32": {"loss_median_window": 1.0}}, 0.25, real_corpus=True)["status"] == "input-error"))

    # Fail-closed: синтетика (корпус не объявлен) вердикта паритета не несёт.
    synthetic = verdict(good, 0.25)
    checks.append(
        (
            "fail-closed: синтетика — input-error, никогда pass",
            synthetic["status"] == "input-error" and synthetic["reason"] == SYNTHETIC_NO_VERDICT,
        )
    )

    # План ноги детерминирован, клетки парные (отличаются только dtype-гейтом).
    plan_kwargs = dict(
        config="net/config.json", batch=1, seq=8192, total_tokens=5_000_000,
        streams=("W", "C"), corpus="corpus",
    )
    plan_a, plan_b = build_plan(**plan_kwargs), build_plan(**plan_kwargs)
    checks.append(
        (
            "план ноги детерминирован, клетки парные",
            plan_a == plan_b
            and [c["mode"] for c in plan_a["cells"]] == ["fp32", "bf16"]
            and {k for c in plan_a["cells"] for k in c["env"]} == {"AXIOM_COMPUTE_DTYPE"},
        )
    )

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
    # --- параметры ноги (не захардкожены: конфиг, объём, источник данных) ---
    parser.add_argument("--config", default=LEG_CONFIG,
                        help=f"путь к конфигу модели (дефолт {LEG_CONFIG})")
    parser.add_argument("--total-tokens", type=float, default=LEG_TOKENS,
                        help=f"бюджет токенов ноги (дефолт {LEG_TOKENS}); steps вычисляются")
    parser.add_argument("--shard-root", default=None,
                        help="корень шард-набора; упакованный корпус берётся из <root>/tokens")
    parser.add_argument("--tokens-root", default=None,
                        help="корень упакованного корпуса tokens-v2 (потоки — --streams); "
                             "сильнее --shard-root")
    parser.add_argument("--streams", default=DEFAULT_STREAMS,
                        help=f"потоки корпуса через запятую (дефолт {DEFAULT_STREAMS})")
    parser.add_argument("--no-smoke", dest="smoke", action="store_false", default=True,
                        help="не гонять 50-шаговый смоук механики перед ногой")
    parser.add_argument("--leg", choices=[m for m, _ in CELLS], default=None,
                        help="внутренний режим: исполнить одну ногу")
    parser.add_argument("--leg-out", default=None)
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args(argv)

    if args.selftest:
        return selftest()

    try:
        streams = parse_streams(args.streams)
    except ValueError as exc:
        print(f"--streams: {exc}", file=sys.stderr)
        return 2
    tokens_root = resolve_tokens_root(args.shard_root, args.tokens_root)
    steps = args.steps or leg_steps(args.total_tokens, LEG_BATCH, LEG_SEQ)
    out_path = Path(args.out)

    if args.leg is not None:
        payload = run_leg(
            args.leg, out_path.parent, steps,
            config=args.config, tokens_root=tokens_root, streams=streams,
        )
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

    plan = build_plan(
        config=args.config, batch=LEG_BATCH, seq=LEG_SEQ,
        total_tokens=args.total_tokens, streams=streams, corpus=tokens_root,
    )
    base_report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "leg": {
            "config": plan["config"], "batch": plan["batch"], "seq": plan["seq"],
            "tokens": plan["total_tokens"], "steps": plan["steps"],
            "streams": plan["streams"], "seed": plan["seed"],
        },
        "data": {"kind": plan["data"], "corpus": plan["corpus"]},
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

    # --- смоук механики: 50 шагов на клетку, в отчёт не идёт ------------------
    if args.smoke:
        for name, _env in CELLS:
            print(f"[loss-parity] смоук {name} ({SMOKE_STEPS} шагов, в отчёт не идёт) …", flush=True)
            smoke = spawn_leg(
                name, out_path.parent, SMOKE_STEPS,
                config=args.config, tokens_root=tokens_root, streams=streams,
            )
            if "error" in smoke:
                base_report["status"] = "input-error"
                base_report["verdict"] = {
                    "status": "input-error",
                    "reason": f"смоук ноги {name} упал: {smoke['error']}",
                }
                write_report(base_report, out_path)
                print(f"[loss-parity] смоук {name} упал → {out_path}", file=sys.stderr)
                return 1

    # --- fail-closed: синтетика вердикта паритета не несёт --------------------
    if plan["data"] != "corpus":
        base_report["status"] = "input-error"
        base_report["verdict"] = {"status": "input-error", "reason": SYNTHETIC_NO_VERDICT}
        base_report["note"] = (
            "Флаги корпуса не заданы (--shard-root/--tokens-root): данные "
            "синтетические, прогон допустим только как смоук механики. Для "
            "вердикта паритета укажите упакованный корпус tokens-v2."
        )
        write_report(base_report, out_path)
        print(f"[loss-parity] input-error (синтетика: {SYNTHETIC_NO_VERDICT}) → {out_path}", file=sys.stderr)
        return 1

    cells: dict[str, dict[str, Any]] = {}
    for name, _env in CELLS:
        print(f"[loss-parity] нога {name} ({steps} шагов) …", flush=True)
        cells[name] = spawn_leg(
            name, out_path.parent, steps,
            config=args.config, tokens_root=tokens_root, streams=streams,
        )

    base_report["cells_measured"] = cells
    failures = [m for m, c in cells.items() if "error" in c]
    if failures:
        base_report["status"] = "input-error"
        base_report["verdict"] = {"status": "input-error", "reason": f"ноги упали: {failures}", "errors": {m: cells[m]["error"] for m in failures}}
        write_report(base_report, out_path)
        print(f"[loss-parity] FAILED ({failures}) → {out_path}", file=sys.stderr)
        return 1
    base_report["verdict"] = verdict(cells, tpb, real_corpus=True)
    base_report["status"] = base_report["verdict"]["status"]
    write_report(base_report, out_path)
    print(f"[loss-parity] {base_report['status']} → {out_path}")
    return 0 if base_report["status"] in ("pass", "EMPTY-PENDING") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
