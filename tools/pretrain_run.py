#!/usr/bin/env python3
"""Претрейн-луп L3 (V-4): прогон по шардам W/C + машинный след стадии.

CLI к ``net/train_loop.py``: собирает потоковый даталоадер по шард-файлам
``~/gb10-shared/datasets/axiom-pretrain-l3/{W,C}`` (ADR-021: микс ~85/15 по
токенам), гоняет цикл на скелете L3 (``net/model.py`` + ``net/optimizer.py``,
bf16-параметры при fp32-мастере, grad-checkpointing для ``l3-full``), пишет
метрики jsonl, Orbax-чекпойнты с ``tree_hash`` и курсор resume, и останавливает
прогон по смете AD-8.

Что прогон делает по шагам:

1. **Смета до запуска** (AD-8).  Читается ``evidence/budget/<run-ref>.json``
   (или ``--budget-file``); отсутствие сметы — блокирующий отказ старта, обход
   только явным ``--budget-limit-usd`` для смоука, и этот обход виден в журнале.
2. **Токенизатор.**  Пиннутый канонический BPE (``net/config.json:tokenizer_hash``,
   ADR-4): подмена токенизатора молча запрещена.  Размер словаря модели выводится
   из фактически испускаемых id, а не берётся из пресета — иначе модель читала бы
   embedding по чужим индексам.
3. **Данные.**  Манифесты шардов читаются как контракт (файлы, sha256, records);
   их пиннутые хеши уезжают в журнал — это evidence AD-4 без пересчёта 26 ГБ.
4. **Цикл.**  ``net.train_loop.train``: WSD-расписание (warmup — stable — decay
   последние 5%), стоп-правило по смете, метрики и чекпойнты.
5. **След.**  ``<out>/journal.json`` + ``<out>/metrics.jsonl``: пути ТОЛЬКО
   относительные от корня репозитория либо ``~/…`` (ADR-014 п. 8).

Запуск (локальный GPU; см. net/README.md про LD_LIBRARY_PATH):

    export LD_LIBRARY_PATH=$(ls -d ~/venv-axiom/lib/python3.11/site-packages/nvidia/*/lib | tr '\\n' ':')
    ~/venv-axiom/bin/python tools/pretrain_run.py \\
        --model-preset small --steps 30 --batch-size 1 --seq-len 8192 \\
        --run-ref pretrain-smoke --budget-file /tmp/pretrain-smoke-budget.json

Выход: ``0`` — шаги исполнены; ``1`` — прогон не исполнен (отказ сметы/данных,
ошибка) — журнал при этом пишется честно, с фактическим статусом и причиной.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

CASE_DIR = Path(__file__).resolve().parent.parent
for _path in (str(CASE_DIR), str(CASE_DIR / "tools")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_sft_smoke as sft_stage  # noqa: E402 — источник общих хелперов стадий

#: Схема журнала прогона.
JOURNAL_SCHEMA = "pretrain-run-journal/v1"

#: Подкаталог шард-наборов на каноническом диске (ADR-021).
DEFAULT_SHARD_SUBDIR = "datasets/axiom-pretrain-l3"

#: Подкаталог прогонов на каноническом диске (вне git: журналы и чекпойнты).
DEFAULT_RUN_SUBDIR = "runs/pretrain-l3"

#: Пресеты, для которых grad-checkpointing обязателен (урок OOM 956 ГиБ).
GRAD_CHECKPOINT_REQUIRED = (sft_stage.L3_FULL_PRESET,)


class StageRefused(RuntimeError):
    """Прогон не исполнен; причина — в журнале."""


#: Абсолютный путь в строке журнала: от корня ФС, минимум с одним сегментом-
#: каталогом.  Не срабатывает на ``~/…``, относительных путях и URL.
ATOMIC_PATH = re.compile(r"(?<![\w:/~])/(?:[\w.-]+/)+[\w.-]*")


# ---------------------------------------------------------------------------
# Аргументы
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Претрейн-луп L3 (V-4): потоковый микс W/C, WSD, стоп-правило AD-8",
    )
    parser.add_argument("--run-ref", default="pretrain-l3", help="имя прогона (смета и журнал)")
    parser.add_argument("--shard-root", default=None,
                        help=f"каталог шард-наборов (по умолчанию ~/gb10-shared/{DEFAULT_SHARD_SUBDIR})")
    parser.add_argument("--out", default=None,
                        help=f"каталог журнала (по умолчанию ~/gb10-shared/{DEFAULT_RUN_SUBDIR}/<run-ref>)")
    parser.add_argument("--streams", default="W,C", help="потоки шардов через запятую")
    parser.add_argument("--mix", default=None,
                        help="веса микса, напр. W=0.85,C=0.15 (по умолчанию — ADR-021)")
    parser.add_argument("--model-preset", choices=sft_stage.SMOKE_PRESETS + (sft_stage.L3_FULL_PRESET,),
                        default="small")
    parser.add_argument("--steps", type=int, default=30, help="шагов в этой ноге")
    parser.add_argument("--total-steps", type=int, default=None,
                        help="горизонт расписания (по умолчанию = --steps)")
    parser.add_argument("--seq-len", type=int, default=8192, help="длина упаковки (T)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--shuffle-window", type=int, default=10000,
                        help="окно детерминированного шаффла (документов)")
    parser.add_argument("--max-doc-tokens", type=int, default=None,
                        help="обрезка документа в токенах (по умолчанию — без обрезки)")
    parser.add_argument("--lr", type=float, default=1e-2, help="пиковый LR")
    parser.add_argument("--schedule", choices=("wsd", "cosine"), default="wsd")
    parser.add_argument("--warmup-ratio", type=float, default=0.01)
    parser.add_argument("--decay-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--param-dtype", choices=("bfloat16", "float32"), default="bfloat16",
                        help="bfloat16 — bf16-параметры при fp32-мастере (прод), float32 — parity")
    parser.add_argument("--grad-checkpointing", dest="grad_checkpointing",
                        action="store_true", default=None,
                        help="remat графа (обязателен для l3-full)")
    parser.add_argument("--no-grad-checkpointing", dest="grad_checkpointing",
                        action="store_false")
    parser.add_argument("--grad-checkpointing-policy", choices=("full", "selective"), default="full")
    parser.add_argument("--checkpoint-every", type=int, default=0,
                        help="шагов между чекпойнтами (0 — выключено)")
    parser.add_argument("--keep-last", type=int, default=2, help="сколько чекпойнтов хранить")
    parser.add_argument("--ckpt-dir", default=None, help="каталог чекпойнтов (по умолчанию <out>/checkpoints)")
    parser.add_argument("--resume", action="store_true", help="продолжить с последнего чекпойнта")
    parser.add_argument("--metrics", default=None, help="файл метрик jsonl (по умолчанию <out>/metrics.jsonl)")
    parser.add_argument("--budget-file", default=None,
                        help="файл сметы (по умолчанию evidence/budget/<run-ref>.json)")
    parser.add_argument("--budget-limit-usd", type=float, default=None,
                        help="явный лимит смоука, если сметы нет (AD-8: факт, а не подмена)")
    parser.add_argument("--usd-per-gpu-hour", type=float, default=None,
                        help="ставка аренды: без неё правило по USD не оценивается")
    parser.add_argument("--peak-tflops", type=float, default=None,
                        help="объявленный пик железа для MFU; не объявлен — MFU не выдумывается")
    parser.add_argument("--peak-tflops-source", default="",
                        help="откуда взято число пика (для журнала)")
    parser.add_argument("--log-every", type=int, default=0, help="печатать каждый N-й шаг")
    parser.add_argument("--json", action="store_true", help="печать журнала в stdout")
    return parser.parse_args(list(argv) if argv is not None else None)


# ---------------------------------------------------------------------------
# Сборка прогона
# ---------------------------------------------------------------------------


def resolve_paths(args: argparse.Namespace) -> dict[str, Any]:
    """Пути прогона: шарды и результаты — на каноническом диске (C-032/C-033)."""
    shared = sft_stage.shared_root()
    shard_root = Path(args.shard_root) if args.shard_root else shared / DEFAULT_SHARD_SUBDIR
    out_dir = Path(args.out) if args.out else shared / DEFAULT_RUN_SUBDIR / args.run_ref
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else out_dir / "checkpoints"
    metrics_path = Path(args.metrics) if args.metrics else out_dir / "metrics.jsonl"
    budget_path = (
        Path(args.budget_file)
        if args.budget_file
        else CASE_DIR / "evidence" / "budget" / f"{args.run_ref}.json"
    )
    return {
        "shard_root": shard_root,
        "out_dir": out_dir,
        "ckpt_dir": ckpt_dir,
        "metrics_path": metrics_path,
        "budget_path": budget_path,
    }


def parse_mix(raw: Optional[str], streams: tuple[str, ...]) -> dict[str, float]:
    """Веса микса: ``W=0.85,C=0.15`` либо дефолт ADR-021 для набора W+C."""
    if raw:
        mix: dict[str, float] = {}
        for item in raw.split(","):
            name, _, value = item.partition("=")
            if not _:
                raise StageRefused(f"вес микса без значения: {item!r} (ожидается W=0.85)")
            mix[name.strip()] = float(value)
        missing = [name for name in streams if name not in mix]
        if missing:
            raise StageRefused(f"в миксе нет весов для потоков {missing}")
        return mix
    if set(streams) == {"W", "C"}:
        return {"W": 0.85, "C": 0.15}
    return {name: 1.0 for name in streams}


def backend_block(tokenizer, cfg, repo_root: Optional[Path]) -> dict[str, Any]:
    """Пиннинг бэкенда и точности в журнал (ADR-010/ADR-013).

    Заголовок прогона обязан нести политику точности, устройства и режим
    детерминизма: без этой строки результат не доказательство (ADR-011), а
    расхождение CPU/GPU на одном коммите уже случалось (6e-08 против 3.37e-04).
    """
    conftest = sft_stage._load_acceptance_conftest()
    block = sft_stage._backend_block(conftest)
    import jax

    block["xla_flags"] = os.environ.get("XLA_FLAGS", "")
    block["tokenizer_hash"] = tokenizer.vocab_hash()
    block["vocab_size"] = int(cfg.vocab_size)
    block["note"] = (
        "детерминизм XLA включается гейтовым профилем (NET_GATE_PROFILE=1, "
        "ADR-013): без него побитовая воспроизводимость между процессами на GPU "
        "не гарантируется — расхождение на 1e-4 после нескольких шагов это "
        "выбор ядер, а не ошибка resume"
    )
    return block


def build_tokenizer_and_config(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    """Пиннутый токенизатор + конфиг скелета с покрывающим словарём.

    Словарь модели выводится из фактически испускаемых id канонического BPE
    (как в стадии SFT): пресет — масштаб, а не словарь, и молча подставить
    туда чужой размер значило бы читать embedding по неверным индексам.
    """
    tokenizer, tokenizer_hash = sft_stage.canonical_tokenizer()
    max_id = max(tokenizer._merge_id.values()) if tokenizer._merge_id else 0
    max_emitted = max(3 + 256 - 1, max_id)
    vocab_size = 1 << max(10, int(max_emitted).bit_length())
    cfg = sft_stage.build_model_config(vocab_size, args.model_preset, qat_weights=False)
    info = {
        "hash": tokenizer_hash,
        "source": "net/tokenizer.py canonical (net/config.json:tokenizer_hash)",
        "vocab_size": tokenizer.vocab_size,
        "merges": len(tokenizer.merges),
        "max_emitted_id": int(max_emitted),
        "model_vocab_size": int(vocab_size),
        "note": (
            "модельный vocab покрывает все испускаемые id канонического "
            "токенизатора; полный BPE 160K на корпусе — масштаб вне скелета"
        ),
    }
    return tokenizer, cfg, info


def resume_cursor(manager) -> Any:
    """Курсор данных из чекпойнта — часть resume, без которой прогон «поехал».

    Чекпойнт хранит две вещи: веса и позицию потока.  Восстановить только веса
    значит продолжить обучение на данных с начала корпуса: лосс выглядит
    правдоподобно, а прогон воспроизводит уже пройденные документы.  Поэтому
    загрузчик создаётся уже с курсором, а не «догоняет» его позже.
    """
    from net.train_loop import MixCursor

    if manager is None:
        return None
    latest = manager.latest()
    if latest is None:
        return None
    payload = latest.get("cursor") or {}
    if not payload.get("schema"):
        return None
    return MixCursor.from_json(payload)


def peak_rss_mb() -> Optional[float]:
    """Пик RSS процесса — свидетельство потоковости чтения корпуса.

    ``ru_maxrss`` на Linux измеряется в килобайтах; значение за весь процесс,
    а не за шаг, поэтому в журнал идёт как верхняя граница живого окна
    (окно шаффла + недобранные токены), а не как «память даталоадера».
    """
    try:
        import resource

        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    except Exception:  # pragma: no cover — платформа без resource
        return None


def describe_shards(shard_set, repo_root: Optional[Path]) -> dict[str, Any]:
    """Пиннутые хеши шард-набора: evidence AD-4 без пересчёта гигабайтов."""
    return {
        "name": shard_set.name,
        "manifest": sft_stage.repo_rel(shard_set.manifest_path, repo_root),
        "source": shard_set.source,
        "shards": len(shard_set.entries),
        "records": shard_set.total_records,
        "approx_tokens": shard_set.total_tokens,
        "pinned": [
            {"file": entry.file, "sha256": entry.sha256, "records": entry.records}
            for entry in shard_set.entries
        ],
    }


def metrics_summary(rows: list[dict], path: Path, repo_root: Optional[Path]) -> dict[str, Any]:
    """Свод метрик прогона: лосс, ток/с, MFU — медианы, а не один удачный шаг."""
    if not rows:
        return {"path": sft_stage.repo_rel(path, repo_root), "rows": 0}
    tps = [row["tokens_per_sec"] for row in rows if row.get("tokens_per_sec")]
    mfu_values = [row["mfu"] for row in rows if row.get("mfu") is not None]
    tail = rows[len(rows) // 2 :] or rows
    return {
        "path": sft_stage.repo_rel(path, repo_root),
        "rows": len(rows),
        "loss_first": rows[0]["loss"],
        "loss_last": rows[-1]["loss"],
        "loss_median_tail": statistics.median(row["loss"] for row in tail),
        "tokens_per_sec_median": statistics.median(tps) if tps else None,
        "mfu_median": statistics.median(mfu_values) if mfu_values else None,
        "mfu_params_only": True,
        "mfu_note": (
            "MFU считается по параметрической части (6·N_active·tokens); вклад "
            "внимания (квадратичный по T) не входит — число неполное по построению"
        ),
    }


# ---------------------------------------------------------------------------
# Прогон
# ---------------------------------------------------------------------------


def execute(args: argparse.Namespace) -> tuple[dict[str, Any], bool]:
    """Исполнить прогон; вернуть (журнал, успех).  Журнал пишется всегда."""
    from net import train_loop as tl

    started = time.time()
    repo_root = sft_stage.detect_repo_root()
    paths = resolve_paths(args)
    streams = tuple(name.strip() for name in args.streams.split(",") if name.strip())
    out_dir = paths["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    journal: dict[str, Any] = {
        "schema": JOURNAL_SCHEMA,
        "stage": "pretrain",
        "status": "absent",
        "scale": "smoke",
        "run_ref": args.run_ref,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "paths": {
            "shard_root": sft_stage.repo_rel(paths["shard_root"], repo_root),
            "out_dir": sft_stage.repo_rel(out_dir, repo_root),
            "checkpoints": sft_stage.repo_rel(paths["ckpt_dir"], repo_root),
            "metrics": sft_stage.repo_rel(paths["metrics_path"], repo_root),
        },
    }

    try:
        tokenizer, cfg, tokenizer_info = build_tokenizer_and_config(args)
    except Exception as exc:  # токенизатор не собрался — прогон не начат
        journal["refusal"] = f"токенизатор/конфиг: {exc}"
        _write_journal(journal, out_dir)
        print(f"[pretrain] ОТКАЗ: {exc}", file=sys.stderr, flush=True)
        return journal, False

    journal["tokenizer"] = tokenizer_info
    journal["model"] = {
        "preset": args.model_preset,
        "vocab_size": int(cfg.vocab_size),
        "hidden": int(cfg.hidden),
        "num_layers": int(cfg.num_layers),
        "param_dtype": args.param_dtype,
        "chunk_size": args.chunk_size,
    }

    journal["backend"] = backend_block(tokenizer, cfg, repo_root)
    if not journal["backend"].get("determinism_pinned"):
        print(
            "[pretrain] ВНИМАНИЕ: детерминизм XLA не запиннен (NET_GATE_PROFILE=1) — "
            "побитовая воспроизводимость между процессами на GPU не гарантируется "
            "(ADR-013); запись в журнале это фиксирует",
            file=sys.stderr,
            flush=True,
        )

    # --- смета (AD-8) -----------------------------------------------------
    budget = tl.load_budget(paths["budget_path"])
    try:
        budget = tl.require_budget(budget, explicit_limit_usd=args.budget_limit_usd)
    except tl.PretrainBudgetError as exc:
        journal["budget"] = {
            "estimate_present": budget.present,
            "estimate_path": sft_stage.repo_rel(paths["budget_path"], repo_root),
            "verdict": str(exc),
        }
        journal["refusal"] = str(exc)
        _write_journal(journal, out_dir)
        print(f"[pretrain] ОТКАЗ сметы: {exc}", file=sys.stderr, flush=True)
        return journal, False

    # --- данные -----------------------------------------------------------
    mix = parse_mix(args.mix, streams)
    ckpt_manager = tl.CheckpointManager(paths["ckpt_dir"], keep_last=args.keep_last)
    if args.resume and ckpt_manager.latest() is None:
        reason = f"--resume: в {sft_stage.repo_rel(paths['ckpt_dir'], repo_root)} нет чекпойнтов"
        journal["refusal"] = reason
        _write_journal(journal, out_dir)
        print(f"[pretrain] ОТКАЗ: {reason}", file=sys.stderr, flush=True)
        return journal, False

    data_cursor = resume_cursor(ckpt_manager) if args.resume else None
    if args.resume:
        journal["resume"] = {
            "from_step": ckpt_manager.latest()["step"],
            "cursor_restored": data_cursor is not None,
        }
    try:
        loader = tl.PretrainMixLoader(
            shard_root=paths["shard_root"],
            streams=streams,
            encode=tokenizer.encode,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            seed=args.seed,
            mix=mix,
            shuffle_window=args.shuffle_window,
            max_doc_tokens=args.max_doc_tokens,
            cursor=data_cursor,
        )
    except tl.PretrainDataError as exc:
        journal["refusal"] = f"данные: {exc}"
        _write_journal(journal, out_dir)
        print(f"[pretrain] ОТКАЗ данных: {exc}", file=sys.stderr, flush=True)
        return journal, False

    journal["data"] = {
        "shard_root": sft_stage.repo_rel(paths["shard_root"], repo_root),
        "streams": [describe_shards(loader._streams[name].shard_set, repo_root) for name in streams],
        "mix_declared": mix,
        "shuffle_window": args.shuffle_window,
        "streaming": True,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
    }

    # --- grad-checkpointing: для l3-full обязателен ------------------------
    grad_checkpointing = args.grad_checkpointing
    if grad_checkpointing is None:
        grad_checkpointing = args.model_preset in GRAD_CHECKPOINT_REQUIRED
    if args.model_preset in GRAD_CHECKPOINT_REQUIRED and not grad_checkpointing:
        reason = (
            "grad-checkpointing обязателен для пресета l3-full: без remat графа "
            "прогон упирается в память активаций (урок OOM 956 ГиБ). "
            "Явное --no-grad-checkpointing для l3-full запрещено."
        )
        journal["refusal"] = reason
        _write_journal(journal, out_dir)
        print(f"[pretrain] ОТКАЗ: {reason}", file=sys.stderr, flush=True)
        return journal, False

    # --- цикл -------------------------------------------------------------
    manager = ckpt_manager
    train_config = tl.TrainConfig(
        steps=args.steps,
        total_steps=args.total_steps,
        lr=args.lr,
        schedule=args.schedule,
        warmup_ratio=args.warmup_ratio,
        decay_ratio=args.decay_ratio,
        seed=args.seed,
        chunk_size=args.chunk_size,
        grad_checkpointing=bool(grad_checkpointing),
        grad_checkpointing_policy=args.grad_checkpointing_policy,
        param_dtype=args.param_dtype,
        checkpoint_every=args.checkpoint_every,
        keep_last=args.keep_last,
        ckpt_dir=paths["ckpt_dir"],
        metrics_path=paths["metrics_path"],
        peak_tflops=args.peak_tflops,
        peak_tflops_source=args.peak_tflops_source,
        usd_per_gpu_hour=args.usd_per_gpu_hour,
        log_every=args.log_every,
    )
    journal["loop"] = {
        "steps_requested": args.steps,
        "total_steps": args.total_steps or args.steps,
        "schedule": args.schedule,
        "lr": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "decay_ratio": args.decay_ratio,
        "grad_checkpointing": bool(grad_checkpointing),
        "grad_checkpointing_policy": args.grad_checkpointing_policy,
        "param_dtype": args.param_dtype,
        "resume": bool(args.resume),
    }
    journal["gpu"] = {
        "peak_tflops": args.peak_tflops,
        "peak_tflops_source": args.peak_tflops_source,
        "peak_declared_not_measured": args.peak_tflops is not None,
    }
    print(
        f"[pretrain] {args.model_preset}: vocab={cfg.vocab_size}, "
        f"T={args.seq_len}, B={args.batch_size}, шагов {args.steps}, "
        f"grad-checkpointing={bool(grad_checkpointing)}, "
        f"токенизатор={tokenizer_info['hash'][:16]}…",
        flush=True,
    )

    try:
        result = tl.train(
            cfg,
            iter(loader),
            train_config=train_config,
            budget=budget,
            resume_from=manager if args.resume else None,
            loader=loader,
        )
    except Exception as exc:  # прогон не состоялся — причина в журнал
        journal["refusal"] = f"цикл: {type(exc).__name__}: {exc}"
        _write_journal(journal, out_dir)
        print(f"[pretrain] ОШИБКА цикла: {exc}", file=sys.stderr, flush=True)
        return journal, False

    stats = loader.stats()
    journal.update(
        {
            "status": "executed" if result.steps_done > 0 else "absent",
            "scale": "smoke" if args.steps <= 200 else "run",
            "steps_done": result.steps_done,
            "start_step": result.start_step,
            "tokens_seen": result.tokens_seen,
            "loss_first": result.losses[0] if result.losses else None,
            "loss_last": result.losses[-1] if result.losses else None,
            "lr_first": result.lr_history[0] if result.lr_history else None,
            "lr_last": result.lr_history[-1] if result.lr_history else None,
            "tree_hash": result.tree_hash,
            "tree_hash_master": result.master_tree_hash,
            "tree_hash_note": (
                "tree_hash — обслуживаемые веса (params); tree_hash_master — "
                "fp32-мастер, он же лежит в чекпойнте. При param_dtype=float32 "
                "веса и мастер совпадают и хеши равны"
            ),
            "timings": result.timings,
            "wall_clock_s": round(time.time() - started, 3),
            "data_consumed": {
                "documents_read": stats["documents_read"],
                "documents_dropped_empty": stats["documents_dropped_empty"],
                "documents_truncated": stats["documents_truncated"],
                "tokens_total": stats["tokens_total"],
                "token_share": stats["token_share"],
                "pending_tokens": stats["pending_tokens"],
                "peak_rss_mb": peak_rss_mb(),
                "streams": stats["streams"],
            },
            "budget": result.budget_report,
            "checkpoint": (
                {**result.checkpoint, "path": sft_stage.repo_rel(Path(result.checkpoint["path"]), repo_root)}
                if result.checkpoint
                else None
            ),
            "stop": {
                "reason": result.stop_reason,
                "stopped_by_budget": result.stopped_by_budget,
            },
        }
    )
    journal["metrics_summary"] = metrics_summary(
        tl.MetricsWriter.read(paths["metrics_path"]), paths["metrics_path"], repo_root
    )
    journal["notes"] = [
        "mfu_params_only: MFU по параметрической части, вклад внимания не входит",
        "mfu: доля от ОБЪЯВЛЕННОГО пика; на смоуке с крошечной моделью число мало "
        "по построению и не является замером эффективности железа",
        "token_share: фактическая доля потоков в израсходованном окне корпуса",
    ]
    _write_journal(journal, out_dir)

    print(
        f"[pretrain] шагов {result.steps_done}, токенов {result.tokens_seen}, "
        f"loss {journal['loss_first']} → {journal['loss_last']}, "
        f"tree_hash={result.tree_hash[:16]}…, остановка: {result.stop_reason or 'по плану'}",
        flush=True,
    )
    return journal, result.steps_done > 0 and not _hard_failure(result)


def _hard_failure(result) -> bool:
    """Отказом считается прогон без единого шага либо оборванный по данным."""
    if result.steps_done == 0:
        return True
    reason = result.stop_reason or ""
    return reason.startswith("data exhausted") or reason.startswith("остановка без причины")


def portable_paths(value: Any, repo_root: Optional[Path]) -> Any:
    """Сделать строки журнала портируемыми: только относительные/``~/…`` пути.

    Сообщения об отказах приходят из модуля и несут пути файлов (смета, шард,
    манифест).  Абсолютный путь сборочной машины в артефакте — дефект приёмки
    (ADR-014 п. 8: манифест с путями worktree стал непортируемым и утёк личными
    путями), поэтому путь нормализуется здесь, а не «как-нибудь потом».
    """
    if isinstance(value, str):
        return portable_text(value, repo_root)
    if isinstance(value, dict):
        return {key: portable_paths(item, repo_root) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable_paths(item, repo_root) for item in value]
    return value


def portable_text(text: str, repo_root: Optional[Path]) -> str:
    """Одна строка: корень репозитория → относительный путь, дом → ``~``."""
    if repo_root is not None:
        root = str(repo_root)
        text = text.replace(root + "/", "").replace(root, ".")
    home = str(Path.home())
    return text.replace(home + "/", "~/").replace(home, "~")


def absolute_path_leaks(journal: dict[str, Any]) -> list[str]:
    """Оставшиеся абсолютные пути в журнале — сигнал, а не молчание.

    Ловится и «домашний» след сборочной машины, и любой абсолютный путь
    (``/tmp/…``, ``/var/…``) в значениях строк: допустимы только относительные
    пути от корня репозитория и форма ``~/…`` (ADR-014 п. 8).  Шаблон не
    срабатывает на URL (``https://…``) и на относительных путях — проверка
    ищет путь, начинающийся с корня файловой системы.
    """
    leaks: list[str] = []
    needle = str(Path.home())

    def walk(node: Any, where: str) -> None:
        if isinstance(node, dict):
            for key, item in node.items():
                walk(item, f"{where}.{key}")
        elif isinstance(node, (list, tuple)):
            for index, item in enumerate(node):
                walk(item, f"{where}[{index}]")
        elif isinstance(node, str):
            if needle in node or ATOMIC_PATH.search(node):
                leaks.append(f"{where}: {node[:120]}")

    walk(journal, "journal")
    return leaks


def _write_journal(journal: dict[str, Any], out_dir: Path) -> Path:
    """Записать журнал, сняв абсолютные пути; утечка после снятия — дефект."""
    repo_root = sft_stage.detect_repo_root()
    journal = portable_paths(journal, repo_root)
    leaks = absolute_path_leaks(journal)
    if leaks:
        journal["path_leaks"] = leaks
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "journal.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(journal, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)
    return path


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    journal, ok = execute(args)
    if args.json:
        print(json.dumps(journal, ensure_ascii=False, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
