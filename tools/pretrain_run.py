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
3. **Данные.**  По умолчанию — претокенизированные ``tokens/{W,C,Q}/*.bin``
   (``PackedTokenLoader``), если манифесты на месте; иначе — сырые
   ``manifest-*.jsonl.zst`` (``PretrainMixLoader``, fallback).  Оба пути читают
   манифесты как контракт (файлы, sha256, records), их пиннутые хеши уезжают в
   журнал — это evidence AD-4 без пересчёта 26 ГБ.  Для packed vocab модели
   выводится из манифеста (id в ``.bin``), а не из диапазона id канонического BPE.
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
import hashlib
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

# ADR-041: дисциплина памяти JAX — префлайт ДО import jax (лимит XLA + гейт стенда).
import jax_preflight  # noqa: E402

jax_preflight.ensure_mem_fraction()

import run_sft_smoke as sft_stage  # noqa: E402 — источник общих хелперов стадий

#: Схема журнала прогона.
JOURNAL_SCHEMA = "pretrain-run-journal/v1"

#: Подкаталог шард-наборов на каноническом диске (ADR-021).
DEFAULT_SHARD_SUBDIR = "datasets/axiom-pretrain-l3"

#: Подкаталог прогонов на каноническом диске (вне git: журналы и чекпойнты).
DEFAULT_RUN_SUBDIR = "runs/pretrain-l3"

#: Корпусной BPE (``tools/bpe_train.py``) в корне шард-набора: артефакт и
#: манифест с его sha256.  Тот же файл, чей хеш пиннится ``net/config.json``
#: (AD-4) и которым размечены ``tokens/*.bin``.
CORPUS_TOKENIZER_SUBDIR = "tokenizer"
CORPUS_TOKENIZER_FILE = "tokenizer.model"
CORPUS_TOKENIZER_MANIFEST = "tokenizer-manifest.json"

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
    parser.add_argument("--tokens-root", default=None,
                        help="каталог претокенизированных .bin (tokens/{W,C}/manifest-*.json); "
                             "по умолчанию <shard-root>/tokens")
    parser.add_argument("--packed", dest="packed", action="store_true", default=None,
                        help="читать готовые токены PackedTokenLoader-ом; по умолчанию — "
                             "включено, если манифесты tokens/ на месте")
    parser.add_argument("--no-packed", dest="packed", action="store_false",
                        help="форсировать raw-манифесты jsonl (fallback)")
    parser.add_argument("--out", default=None,
                        help=f"каталог журнала (по умолчанию ~/gb10-shared/{DEFAULT_RUN_SUBDIR}/<run-ref>)")
    parser.add_argument("--journal", default=None,
                        help="файл журнала (по умолчанию <out>/journal.json); на аренде "
                             "задаётся явно — дефолтный путь C-032 там отсутствует (R1)")
    parser.add_argument("--streams", default="W,C", help="потоки шардов через запятую")
    parser.add_argument("--mix", default=None,
                        help="веса микса, напр. W=0.85,C=0.15 (по умолчанию — ADR-021)")
    parser.add_argument("--decay-stream", default=None,
                        help="decay-шард (ADR-021: Q) на последних --decay-ratio шагах; "
                             "по умолчанию фаза выключена (только микс W/C)")
    parser.add_argument("--decay-shard-root", default=None,
                        help="каталог decay-шарда (по умолчанию --shard-root)")
    parser.add_argument("--decay-seed", type=int, default=None,
                        help="сид шаффла decay-шарда (по умолчанию --seed)")
    parser.add_argument("--model-preset", choices=sft_stage.SMOKE_PRESETS + (sft_stage.L3_FULL_PRESET,),
                        default="small")
    parser.add_argument("--config-path", type=Path, default=None,
                        help="явный путь конфига модели (ревизия VERIFICATION-LEG, "
                             "SPEC §Раннер): имеет приоритет над --model-preset. "
                             "Подмена net/config.json запрещена (пин скелета), поэтому "
                             "конфиг ноги A/B задаётся путём, а пресеты small/tiny/"
                             "l3-full остаются нетронутыми. По умолчанию — прежнее "
                             "поведение (выбор по пресету)")
    parser.add_argument("--steps", type=int, default=30, help="шагов в этой ноге")
    parser.add_argument("--total-steps", type=int, default=None,
                        help="горизонт расписания (по умолчанию = --steps)")
    parser.add_argument("--seq-len", type=int, default=8192, help="длина упаковки (T)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--micro-batch",
        type=int,
        default=None,
        help=(
            "размер микробатча (последовательностей на один forward), ADR-031 "
            "delta B; по умолчанию равен --batch-size (текущее поведение). "
            "Меньший микробатч — память, добирается накоплением --accum-tokens"
        ),
    )
    parser.add_argument(
        "--accum-tokens",
        type=int,
        default=0,
        help=(
            "накапливать градиенты до ~T токенов на один шаг оптимизатора "
            "(ADR-031: батч 256K накоплением); 0 = выключено (один микробатч "
            "на шаг). Число микробатчей = ceil(T/(micro-batch*seq-len))"
        ),
    )
    parser.add_argument("--shuffle-window", type=int, default=10000,
                        help="окно детерминированного шаффла (документов)")
    parser.add_argument("--max-doc-tokens", type=int, default=None,
                        help="обрезка документа в токенах (по умолчанию — без обрезки)")
    parser.add_argument("--lr", type=float, default=1e-2, help="пиковый LR")
    parser.add_argument("--schedule", choices=("wsd", "cosine"), default="wsd")
    parser.add_argument("--warmup-ratio", type=float, default=0.01)
    parser.add_argument("--decay-ratio", type=float, default=None,
                        help="доля шагов decay (по умолчанию 0.05); явное значение "
                             "запрещает авто-уменьшение под объём Q (H1)")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--ns-steps", type=int, default=5,
                        help="ADR-048: число Newton-Schulz итераций Muon (дефолт 5 — "
                             "прежнее значение; снижение — решение по замеру)")
    parser.add_argument("--legacy-muon-all-2d", dest="legacy_muon_all_2d",
                        action="store_true", default=False,
                        help="ADR-048: прежняя классификация («любой ndim==2 -> Muon», "
                             "embeddings/LM head включительно) — для сравнения «до/после»")
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
    parser.add_argument("--ckpt-every-min", type=float, default=None,
                        help="минут между чекпойнтами (риск-каденс аренды, runbook §3: 30)")
    parser.add_argument("--stop-file", default=None,
                        help="файл-стоп килл-свитча (по умолчанию env STOP_FILE или путь "
                             "из курсора чекпойнта); существование останавливает прогон")
    parser.add_argument("--stop-check-every", type=int, default=1,
                        help="как часто проверять --stop-file (в шагах)")
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
    decay_shard_root = (
        Path(args.decay_shard_root) if args.decay_shard_root else shard_root
    )
    out_dir = Path(args.out) if args.out else shared / DEFAULT_RUN_SUBDIR / args.run_ref
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else out_dir / "checkpoints"
    metrics_path = Path(args.metrics) if args.metrics else out_dir / "metrics.jsonl"
    # К2: претокенизированные бины лежат рядом с сырыми шардами (runbook §1:
    # ``rm -a tokens/ ... /root/data/tokens/`` при ``--shard-root /root/data``).
    tokens_root = Path(args.tokens_root) if args.tokens_root else shard_root / "tokens"
    journal_path = Path(args.journal) if args.journal else out_dir / "journal.json"
    budget_path = (
        Path(args.budget_file)
        if args.budget_file
        else CASE_DIR / "evidence" / "budget" / f"{args.run_ref}.json"
    )
    return {
        "shard_root": shard_root,
        "decay_shard_root": decay_shard_root,
        "tokens_root": tokens_root,
        "out_dir": out_dir,
        "ckpt_dir": ckpt_dir,
        "metrics_path": metrics_path,
        "journal_path": journal_path,
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


class CorpusTokenizer:
    """Корпусной BPE (``tools/bpe_train.py``) для путей претрейна.

    Обёртка над ``tokenizers.Tokenizer`` (Rust) — **тем же** словарём, которым
    размечены ``tokens/*.bin`` (``tools/pretokenize.py``).  Сырой путь обязан
    кодировать текст им, а не скелетной заглушкой ``net/tokenizer.py``: id
    заглушки лежат в другом словаре, поэтому модель читала бы embedding по
    неверным индексам (наблюдался NaN лосса на packed-миксе).  ``vocab_hash``
    возвращает sha256 файла-артефакта — то же значение, что пин
    ``net/config.json:tokenizer_hash`` и ``tokenizer_hash`` манифеста (AD-4),
    поэтому сверка с пином идёт по префиксу.
    """

    def __init__(self, tokenizer: Any, *, vocab_hash: str, vocab_size: int, merges: int):
        self._tokenizer = tokenizer
        self._vocab_hash = str(vocab_hash)
        self.vocab_size = int(vocab_size)
        #: Число мерджей (int, в отличие от списка у ``BPETokenizer``) — из
        #: манифеста, для журнала; минус паддинг словаря под 160K.
        self.merges = int(merges)

    def encode(self, text: str) -> list[int]:
        return [int(token) for token in self._tokenizer.encode(text).ids]

    def vocab_hash(self) -> str:
        return self._vocab_hash


def _file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    """sha256 файла (потоково) — то же значение, что ``tokenizer_hash`` (AD-4)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def load_corpus_tokenizer(
    shard_root: Path, *, pinned: str
) -> Optional[CorpusTokenizer]:
    """Корпусной BPE из ``<shard-root>/tokenizer``; ``None`` — файла нет.

    Артефакт ``tokenizer.model`` (HF ``tokenizers`` json) + манифест рядом.
    Сверка fail-closed: хеш файла обязан совпасть с ``tokenizer_hash`` манифеста
    и с пином ``net/config.json`` (ADR-4), иначе подмена токенизатора молча
    запрещена — id поехали бы по чужому словарю, а лосс выглядел бы нормально.
    Отсутствие артефакта — не ошибка: стадия честно откатывается на скелетную
    заглушку с её собственным гейтом (поведение прежних смоуков).
    """
    directory = Path(shard_root) / CORPUS_TOKENIZER_SUBDIR
    artifact = directory / CORPUS_TOKENIZER_FILE
    if not artifact.is_file():
        return None
    actual = _file_sha256(artifact)
    manifest: dict[str, Any] = {}
    manifest_path = directory / CORPUS_TOKENIZER_MANIFEST
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise StageRefused(
                f"манифест корпусного токенизатора не разбирается: {manifest_path}: {exc}"
            ) from exc
        declared = str(manifest.get("tokenizer_hash") or "")
        if declared and declared != actual:
            raise StageRefused(
                f"tokenizer_hash артефакта {actual[:16]}… ≠ манифеста "
                f"{declared[:16]}… — файл изменён после упаковки (ADR-4)"
            )
    if not sft_stage.tokenizer_hash_matches(actual, pinned):
        raise StageRefused(
            f"корпусной токенизатор {actual[:16]}… не совпал с пином "
            f"net/config.json {str(pinned)[:16]}…: подмена токенизатора молча "
            "запрещена (ADR-4) — пересоберите tokenizer/ каноническим BPE или "
            "уберите его из корня шард-набора"
        )
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:  # pragma: no cover — окружение без библиотеки
        raise StageRefused(
            "нет библиотеки tokenizers для загрузки корпусного BPE: pip install tokenizers"
        ) from exc
    try:
        tokenizer = Tokenizer.from_file(str(artifact))
    except Exception as exc:  # нечитаемый артефакт — отказ, а не молчаливый откат
        raise StageRefused(
            f"корпусной токенизатор не читается ({type(exc).__name__}): {exc}"
        ) from exc
    vocab_size = int(manifest.get("vocab_size") or tokenizer.get_vocab_size())
    merges = int(manifest.get("merges") or max(0, vocab_size - 4 - 256))
    return CorpusTokenizer(
        tokenizer, vocab_hash=actual, vocab_size=vocab_size, merges=merges
    )


def build_tokenizer_and_config(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    """Пиннутый токенизатор + конфиг скелета с покрывающим словарём.

    Приоритет — корпусной BPE ``<shard-root>/tokenizer`` (``tools/bpe_train.py``):
    он же пиннится ``net/config.json`` и им размечены данные.  Найден и сходится
    с пином — сырой путь кодирует текст тем же словарём, что packed-путь; найден,
    но не сходится — блокирующий отказ.  Файла нет — прежнее поведение:
    скелетная заглушка ``net/tokenizer.py`` с собственным гейтом.  Словарь модели
    выводится из фактически испускаемых id загруженного токенизатора (пресет —
    масштаб, а не словарь): иначе embedding читался бы по неверным индексам.
    """
    # Пин читается из конфига ПРОГОНА: с явным --config-path это его файл
    # (иначе сверяли бы данные против чужого конфига), без него — прежний
    # net/config.json.  Вызов без аргумента сохранён как отдельная ветка:
    # подмена ``config_tokenizer_pin`` нулевой лямбдой в тестах остаётся рабочей.
    config_path = getattr(args, "config_path", None)
    pinned = (
        sft_stage.config_tokenizer_pin(config_path)
        if config_path is not None
        else sft_stage.config_tokenizer_pin()
    )
    paths = resolve_paths(args)
    streams = tuple(name.strip() for name in getattr(args, "streams", "W,C").split(",") if name.strip())
    # Packed-путь берёт токенизатор из манифестов tokens/ (``_packed_tokenizer_pin``)
    # и этот оверрайд не использует — не трогаем его и не подменяем ему словарь.
    packed = resolve_packed(args, paths["tokens_root"], streams)
    corpus = None if packed else load_corpus_tokenizer(paths["shard_root"], pinned=pinned)
    if corpus is not None:
        tokenizer = corpus
        tokenizer_hash = tokenizer.vocab_hash()
        max_id = max(0, int(tokenizer.vocab_size) - 1)
        source = (
            "corpus BPE tools/bpe_train.py (<shard-root>/tokenizer) — тот же "
            "словарь, которым размечены tokens/*.bin"
        )
    else:
        tokenizer, tokenizer_hash = sft_stage.canonical_tokenizer()
        max_id = max(tokenizer._merge_id.values()) if tokenizer._merge_id else 0
        source = (
            "net/tokenizer.py canonical — синтетическая заглушка скелета "
            "(не корпусной BPE 160K); <shard-root>/tokenizer отсутствует"
        )
    max_emitted = max(3 + 256 - 1, max_id)
    vocab_size = 1 << max(10, int(max_emitted).bit_length())
    cfg = sft_stage.build_model_config(
        vocab_size, args.model_preset, qat_weights=False,
        config_path=getattr(args, "config_path", None),
    )
    merges = tokenizer.merges if isinstance(tokenizer.merges, int) else len(tokenizer.merges)
    info = {
        "hash": tokenizer_hash,
        "source": source,
        "vocab_size": tokenizer.vocab_size,
        "merges": int(merges),
        "max_emitted_id": int(max_emitted),
        "model_vocab_size": int(vocab_size),
        "config_pin": pinned or None,
        "matches_config_pin": sft_stage.tokenizer_hash_matches(tokenizer_hash, pinned),
        "note": (
            "модельный vocab покрывает все испускаемые id загруженного "
            "токенизатора (макс. id → ближайшая степень двойки): для корпусного "
            "BPE 160K это 262144, тогда как packed-путь берёт vocab точным из "
            "манифеста tokens/ (160000) — расхождение путей осознанное и видно "
            "в журнале"
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


def describe_packed_shards(shard_set, repo_root: Optional[Path]) -> dict[str, Any]:
    """Пиннутые хеши претокенизированного набора (К2): records/slots/stream_tokens."""
    return {
        "name": shard_set.name,
        "kind": "packed",
        "manifest": sft_stage.repo_rel(shard_set.manifest_path, repo_root),
        "source": shard_set.source,
        "shards": len(shard_set.entries),
        "records": shard_set.total_records,
        "slots": shard_set.total_slots,
        "stream_tokens": shard_set.total_stream_tokens,
        "seq_len": shard_set.seq_len,
        "tokenizer_hash": shard_set.tokenizer_hash,
        "pinned": [
            {
                "file": entry.file,
                "source": entry.source,
                "source_sha256": entry.source_sha256,
                "records": entry.records,
                "sha256": entry.sha256,
                "tokenizer_hash": shard_set.tokenizer_hash,
            }
            for entry in shard_set.entries
        ],
    }


def describe_loader(loader, streams, repo_root: Optional[Path]) -> list[dict[str, Any]]:
    """Описание шард-наборов активного пути (packed или raw)."""
    from net import train_loop as tl

    if isinstance(loader, tl.PackedTokenLoader):
        return [describe_packed_shards(loader._sets[name], repo_root) for name in streams]
    return [describe_shards(loader._streams[name].shard_set, repo_root) for name in streams]


def describe_stream(loader, name: str, repo_root: Optional[Path]) -> dict[str, Any]:
    """Описание одного потока (для decay-шарда Q) на активном пути."""
    from net import train_loop as tl

    if isinstance(loader, tl.PackedTokenLoader):
        return describe_packed_shards(loader._sets[name], repo_root)
    if name == loader.decay_stream and loader._q_stream is not None:
        return describe_shards(loader._q_stream.shard_set, repo_root)
    return describe_shards(loader._streams[name].shard_set, repo_root)


def data_consumed_block(stats: dict, data_kind: str) -> dict[str, Any]:
    """Секция ``data_consumed`` журнала: у packed и raw разные счётчики."""
    common = {
        "kind": data_kind,
        "token_share": stats.get("token_share"),
        "phase": stats.get("phase"),
        "decay_start": stats.get("decay_start"),
        "peak_rss_mb": peak_rss_mb(),
        "streams": stats.get("streams"),
    }
    if data_kind == "packed":
        common.update(
            {
                "records_total": stats.get("records_total"),
                "slots_total": stats.get("slots_total"),
                "dropped_tail_records": stats.get("dropped_tail_records"),
            }
        )
        return common
    common.update(
        {
            "documents_read": stats.get("documents_read"),
            "documents_dropped_empty": stats.get("documents_dropped_empty"),
            "documents_truncated": stats.get("documents_truncated"),
            "tokens_total": stats.get("tokens_total"),
            "pending_tokens": stats.get("pending_tokens"),
        }
    )
    return common


def _packed_tokenizer_pin(
    tl, tokens_root: Path, streams: tuple[str, ...], decay_stream: Optional[str]
) -> dict[str, Any]:
    """Токенизатор packed-данных: хеш и словарь из манифестов tokens/ (К2).

    Все потоки (включая decay-шард) обязаны быть собраны **одним** токенизатором:
    разошедшиеся хеши означают, что id разных потоков лежат в разных словарях, и
    модель читала бы их по общим индексам — это отказ контракта данных, а не
    предупреждение.  ``vocab_size`` берётся максимальным по потокам: он и
    покрывает все фактические id.
    """
    names = list(streams) + ([decay_stream] if decay_stream else [])
    hashes: set[str] = set()
    vocab = 0
    for name in names:
        allowed = tuple(streams) if name in streams else (name,)
        shard_set = tl.load_packed_shard_set(
            tl.packed_manifest_path(tokens_root, name), allowed=allowed
        )
        hashes.add(shard_set.tokenizer_hash)
        vocab = max(vocab, int((shard_set.source or {}).get("vocab_size") or 0))
    if len(hashes) != 1:
        raise tl.PretrainDataError(
            "потоки packed-данных собраны разными токенизаторами: "
            + ", ".join(sorted(h[:16] for h in hashes))
        )
    if vocab < 3:
        raise tl.PretrainDataError("манифест tokens не объявляет vocab_size токенизатора")
    return {"hash": hashes.pop(), "vocab_size": vocab, "streams": names}


def canonical_hash_for_packed(
    manifest_hash: str, *, build_hash: str = "", config_pin: str | None = None
) -> dict[str, Any]:
    """Канон токенизатора packed-пути (ADR-004, амендмент от 2026-10-05, п. 2).

    Пин ``net/config.json:tokenizer_hash`` — полный sha256 корпусного BPE v2,
    которым обязаны быть размечены ``tokens/*.bin``.  Если пин объявлен и
    непуст, каноном выступает **он**, и ``matches_canonical`` считается против
    него (``pin_manifest == config_pin``); заглушка скелета (``build_hash``)
    каноном не объявляется никогда.  Пина нет/пуст — прежнее поведение: канон —
    токенизатор сборки, сверка с ним.
    """
    manifest = str(manifest_hash or "").strip()
    pin = config_pin if config_pin is not None else sft_stage.config_tokenizer_pin()
    pin = str(pin or "").strip()
    if pin:
        return {
            "canonical_hash": pin,
            "matches_canonical": manifest == pin,
            "canonical_source": "net/config.json",
        }
    build = str(build_hash or "").strip()
    return {
        "canonical_hash": build,
        "matches_canonical": manifest == build,
        "canonical_source": "build_tokenizer",
    }


def packed_tokenizer_info(
    tokenizer_info: dict[str, Any],
    pin: dict[str, Any],
    *,
    model_vocab_size: int,
    config_pin: str | None = None,
) -> tuple[dict[str, Any], str | None]:
    """Tokenizer-блок packed-пути: канон — пин ``net/config.json``, не заглушка.

    Модель обязана покрыть фактические id в ``.bin`` (механика К2):
    ``model_vocab_size`` берётся из манифеста tokens/.  ``matches_canonical=false``
    не блокирует packed-путь — это расхождение контракта данных, громко
    зафиксированное в журнале; блокирует его отдельная сверка пресета l3-full по
    хешу манифеста.  Возвращает (журнальный блок, текст предупреждения или None):
    предупреждение — только при реальном расхождении данных с каноном.
    """
    canon = canonical_hash_for_packed(
        str(pin["hash"]),
        build_hash=str(tokenizer_info.get("hash") or ""),
        config_pin=config_pin,
    )
    declared = canon["canonical_source"] == "net/config.json"
    canonical_name = (
        "пин net/config.json"
        if declared
        else "токенизатор сборки (пин net/config.json не объявлен)"
    )
    info = {
        "hash": str(pin["hash"]),
        "source": "packed manifest tokens/ (токенизатор, которым собраны .bin)",
        "vocab_size": int(pin["vocab_size"]),
        "model_vocab_size": int(model_vocab_size),
        "canonical_hash": canon["canonical_hash"],
        "canonical_source": canon["canonical_source"],
        "matches_canonical": canon["matches_canonical"],
        "streams": pin["streams"],
        "note": (
            "model_vocab_size выведен из манифеста tokens/ — покрывает фактические "
            "id в .bin, а не диапазон id канонического BPE (иначе embedding по чужим "
            "индексам). matches_canonical сверяет данные с каноном ("
            + canonical_name
            + "): расхождение — контракт данных, не гигиена"
        ),
    }
    warning = None
    if not canon["matches_canonical"]:
        warning = (
            "[pretrain] ВНИМАНИЕ: токенизатор packed-данных "
            f"{str(pin['hash'])[:16]}… ≠ канонического {canon['canonical_hash'][:16]}… "
            f"({canonical_name}) — прогон идёт на словаре корпуса, расхождение "
            "зафиксировано в журнале (tokenizer.matches_canonical=false)"
        )
    return info, warning


def resolve_packed(args: argparse.Namespace, tokens_root: Path, streams: tuple[str, ...]) -> bool:
    """К2: включён ли packed-путь.  Явный флаг сильнее авто-детекта.

    Авто-детект (``--packed`` не задан): packed включается, если для **всех**
    потоков микса есть манифест ``tokens/<S>/manifest-<s>.json``.  Отсутствие
    манифеста — не ошибка, а повод пойти raw-путём (fallback): сырые шарды лежат
    там же.  Битый манифест при этом не прячется — его отвергнет конструктор.
    """
    from net import train_loop as tl

    if args.packed is not None:
        return bool(args.packed)
    return all(tl.packed_manifest_path(tokens_root, name).is_file() for name in streams)


def _pinned_run(run: dict[str, Any]) -> dict[str, Any]:
    """Срез запинненных параметров прогона для журнала resume (H4)."""
    keys = ("seed", "total_steps", "warmup_ratio", "decay_ratio", "param_dtype", "data_kind")
    return {key: run.get(key) for key in keys if key in run}


def validate_resume_pins(
    run: dict[str, Any],
    *,
    seed: int,
    warmup_ratio: float,
    decay_ratio: float,
    data_kind: str,
    enabled: bool,
) -> list[str]:
    """H4: сверить argv с запинненными в курсоре параметрами; список расхождений.

    Resume обязан продолжать **ту же** траекторию: другой seed — другой шаффл и
    другая инициализация, другая доля decay — другая граница фаз, другой источник —
    чужая позиция потока.  Молча продолжить с новыми параметрами значило бы выдать
    иную траекторию за продолжение прежней, поэтому расхождение — отказ с перечнем
    (а не warning: предупреждение в логе не защищает от подмены).
    """
    if not enabled or not run:
        return []
    mismatches: list[str] = []

    def check(key: str, expected: Any) -> None:
        if key not in run or run[key] is None:
            return
        actual = run[key]
        if isinstance(expected, float):
            try:
                ok = abs(float(actual) - expected) <= 1e-12
            except (TypeError, ValueError):
                ok = False
        else:
            ok = actual == expected
        if not ok:
            mismatches.append(f"{key}: курсор {actual!r} ≠ argv {expected!r}")

    check("seed", int(seed))
    check("warmup_ratio", float(warmup_ratio))
    check("decay_ratio", float(decay_ratio))
    check("data_kind", data_kind)
    return mismatches


def _decay_available_tokens(
    tl, paths: dict[str, Any], args: argparse.Namespace, streams, packed: bool, repo_root
) -> Optional[int]:
    """Объём decay-шарда Q (в токенах потока) для проверки окна H1; None — неизвестен."""
    try:
        if packed:
            shard_set = tl.load_packed_shard_set(
                tl.packed_manifest_path(paths["tokens_root"], args.decay_stream),
                allowed=(args.decay_stream,),
            )
            return int(shard_set.total_stream_tokens)
        manifest = (
            paths["decay_shard_root"]
            / args.decay_stream
            / f"manifest-{args.decay_stream.lower()}.json"
        )
        return int(tl.load_shard_set(manifest, allowed=(args.decay_stream,)).total_tokens)
    except tl.PretrainDataError:
        return None


def resolve_stop_file(args: argparse.Namespace, cursor_run: dict[str, Any]) -> Optional[Path]:
    """К5: путь stop-файла — argv → env ``STOP_FILE`` → курсор предыдущей ноги."""
    if args.stop_file:
        return Path(args.stop_file)
    env = os.environ.get("STOP_FILE")
    if env:
        return Path(env)
    pinned = cursor_run.get("stop_file")
    return Path(pinned) if pinned else None


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
    from net.model import loss_impl as _loss_impl

    started = time.time()
    repo_root = sft_stage.detect_repo_root()
    paths = resolve_paths(args)
    streams = tuple(name.strip() for name in args.streams.split(",") if name.strip())
    out_dir = paths["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    journal_path = paths["journal_path"]
    # К2 решается до сборки модели: vocab модели обязан покрыть id из .bin (см. ниже).
    packed = resolve_packed(args, paths["tokens_root"], streams)

    # H1: явный --decay-ratio запрещает авто-уменьшение под объём Q; дефолт (0.05)
    # может быть уменьшен, если decay-окно шире шарда (иначе Q исчерпается на финише).
    decay_ratio = args.decay_ratio if args.decay_ratio is not None else 0.05
    decay_ratio_explicit = args.decay_ratio is not None

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
            "tokens_root": sft_stage.repo_rel(paths["tokens_root"], repo_root),
            "out_dir": sft_stage.repo_rel(out_dir, repo_root),
            "checkpoints": sft_stage.repo_rel(paths["ckpt_dir"], repo_root),
            "metrics": sft_stage.repo_rel(paths["metrics_path"], repo_root),
            "journal": sft_stage.repo_rel(journal_path, repo_root),
        },
    }

    try:
        tokenizer, cfg, tokenizer_info = build_tokenizer_and_config(args)
    except Exception as exc:  # токенизатор не собрался — прогон не начат
        journal["refusal"] = f"токенизатор/конфиг: {exc}"
        _write_journal(journal, out_dir, journal_path=journal_path)
        print(f"[pretrain] ОТКАЗ: {exc}", file=sys.stderr, flush=True)
        return journal, False

    # К2: packed-данные собраны **токенизатором корпуса**, чей словарь может быть
    # шире диапазона id канонического BPE.  Модель обязана покрыть фактические id
    # в .bin (иначе embedding читается по чужим индексам — наблюдался NaN лосса).
    # Поэтому при packed vocab берётся из манифеста tokens/, а расхождение с пином
    # net/config.json фиксируется громко, а не прячется.
    if packed:
        try:
            pin = _packed_tokenizer_pin(tl, paths["tokens_root"], streams, args.decay_stream)
        except tl.PretrainDataError as exc:
            journal["refusal"] = f"данные: {exc}"
            _write_journal(journal, out_dir, journal_path=journal_path)
            print(f"[pretrain] ОТКАЗ данных: {exc}", file=sys.stderr, flush=True)
            return journal, False
        # Канон — пин net/config.json (ADR-004 амендмент п.2), а не заглушка
        # скелета, которую build_tokenizer_and_config даёт на стенде без
        # корпусного токенизатора.  Пина нет — прежнее поведение.
        cfg = sft_stage.build_model_config(
            pin["vocab_size"], args.model_preset, qat_weights=False,
            config_path=getattr(args, "config_path", None),
        )
        tokenizer_info, canonical_warning = packed_tokenizer_info(
            tokenizer_info, pin, model_vocab_size=int(cfg.vocab_size)
        )
        if canonical_warning:
            print(canonical_warning, file=sys.stderr, flush=True)

    journal["tokenizer"] = tokenizer_info
    journal["model"] = {
        "preset": args.model_preset,
        "vocab_size": int(cfg.vocab_size),
        "hidden": int(cfg.hidden),
        "num_layers": int(cfg.num_layers),
        "param_dtype": args.param_dtype,
        "chunk_size": args.chunk_size,
    }
    # Явный конфиг (--config-path, VERIFICATION-LEG): источник ноги A/B обязан
    # быть виден и запинен в журнале — путь относительный (ADR-014 п. 8) + sha256.
    if getattr(args, "config_path", None) is not None:
        journal["model"]["config_source"] = sft_stage.repo_rel(args.config_path, repo_root)
        journal["model"]["config_sha256"] = hashlib.sha256(
            Path(args.config_path).read_bytes()
        ).hexdigest()

    journal["backend"] = backend_block(tokenizer, cfg, repo_root)
    # Хеш токенизатора в блоке бэкенда обязан называть тот, чем размечены данные.
    journal["backend"]["tokenizer_hash"] = tokenizer_info["hash"]
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
        _write_journal(journal, out_dir, journal_path=journal_path)
        print(f"[pretrain] ОТКАЗ сметы: {exc}", file=sys.stderr, flush=True)
        return journal, False

    # --- grad-checkpointing: для l3-full обязателен ------------------------
    # Порядок отказов: бюджет → grad-checkpointing → seed.  Раньше этот отказ
    # стоял после сборки даталоадера, и отсутствие шардов маскировало его
    # («данные: …»).  Проверка не зависит от данных, поэтому идёт сразу за сметой.
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
        _write_journal(journal, out_dir, journal_path=journal_path)
        print(f"[pretrain] ОТКАЗ: {reason}", file=sys.stderr, flush=True)
        return journal, False

    # --- данные -----------------------------------------------------------
    mix = parse_mix(args.mix, streams)
    ckpt_manager = tl.CheckpointManager(paths["ckpt_dir"], keep_last=args.keep_last)
    if args.resume and ckpt_manager.latest() is None:
        reason = f"--resume: в {sft_stage.repo_rel(paths['ckpt_dir'], repo_root)} нет чекпойнтов"
        journal["refusal"] = reason
        _write_journal(journal, out_dir, journal_path=journal_path)
        print(f"[pretrain] ОТКАЗ: {reason}", file=sys.stderr, flush=True)
        return journal, False

    data_cursor = resume_cursor(ckpt_manager) if args.resume else None
    cursor_run = ((ckpt_manager.latest() or {}).get("cursor") or {}).get("run") or {}
    if args.resume:
        journal["resume"] = {
            "from_step": ckpt_manager.latest()["step"],
            "cursor_restored": data_cursor is not None,
            "run_pinned": _pinned_run(cursor_run),
        }

    # --- К2: выбор пути данных (packed основной, raw fallback) -------------
    if packed and args.decay_stream:
        q_manifest = tl.packed_manifest_path(paths["tokens_root"], args.decay_stream)
        if not q_manifest.is_file():
            reason = (
                f"packed-путь: нет манифеста decay-шарда {args.decay_stream} "
                f"({sft_stage.repo_rel(q_manifest, repo_root)}) — decay-фаза не разложена "
                "в .bin; пересоберите tokens или запустите с --no-packed"
            )
            journal["refusal"] = reason
            _write_journal(journal, out_dir, journal_path=journal_path)
            print(f"[pretrain] ОТКАЗ данных: {reason}", file=sys.stderr, flush=True)
            return journal, False
    data_kind = "packed" if packed else "raw"

    # Граница decay-фазы (ADR-021): та же формула, что у WSD-LR (decay_start_step).
    # На resume горизонт берётся из курсора прогона, если --total-steps не задан
    # явно: иначе вторая нога считала бы границу от длины ноги — та же ошибка,
    # что и с LR по обрезанному горизонту.
    horizon = args.total_steps or args.steps
    if args.resume and args.total_steps is None:
        horizon = int(cursor_run.get("total_steps") or horizon)

    # --- ADR-031 delta B: forward micro-batch vs accumulated whole batch ----
    # ``loader_batch`` is the *forward* width (the memory knob): the loader
    # yields that many sequences per item.  ``planned_batch`` is the equivalent
    # per-optimizer-step width used by planning (H1 decay window, journal): with
    # accumulation on, a step consumes ~``accum_tokens`` tokens, i.e.
    # ``ceil(accum_tokens / seq_len)`` sequences.
    if args.micro_batch is not None and int(args.micro_batch) < 1:
        print(
            f"[pretrain] ОТКАЗ: --micro-batch должен быть >= 1, "
            f"получено {args.micro_batch}",
            file=sys.stderr,
            flush=True,
        )
        return journal, False
    if int(args.accum_tokens) < 0:
        print(
            f"[pretrain] ОТКАЗ: --accum-tokens должен быть >= 0 (0 = выключено), "
            f"получено {args.accum_tokens}",
            file=sys.stderr,
            flush=True,
        )
        return journal, False
    loader_batch = int(args.micro_batch) if args.micro_batch else int(args.batch_size)
    planned_batch = int(args.batch_size)
    if int(args.accum_tokens) > 0:
        seq = max(1, int(args.seq_len))
        planned_batch = max(1, -(-int(args.accum_tokens) // seq))

    # --- H4: resume пиннит параметры прогона -------------------------------
    mismatch = validate_resume_pins(
        cursor_run,
        seed=args.seed,
        warmup_ratio=args.warmup_ratio,
        decay_ratio=decay_ratio,
        data_kind=data_kind,
        enabled=bool(args.resume),
    )
    if mismatch:
        reason = (
            "--resume: параметры не совпали с запинненными в курсоре — "
            + "; ".join(mismatch)
            + ". Продолжать другим сидом/долей/источником значило бы выдать другую "
            "траекторию за ту же (H4)"
        )
        journal["refusal"] = reason
        _write_journal(journal, out_dir, journal_path=journal_path)
        print(f"[pretrain] ОТКАЗ resume: {reason}", file=sys.stderr, flush=True)
        return journal, False

    # --- l3-full: токенизатор данных обязан совпасть с пином конфига -------
    # Пин net/config.json описывает корпусной BPE 160K.  Для малого пресета
    # легитимна синтетическая заглушка (её хеш объявлен в журнале), но
    # производственный пресет l3-full обязан читать данные, размеченные
    # каноническим BPE: чужие id молча читались бы по неверным индексам.
    # Сверка идёт последней из «дешёвых» отказов (после бюджета/grad/seed):
    # это контракт данных, а не стартовая гигиена, и она не должна маскировать
    # отказы сметы или resume.
    # Явный --config-path делает эту сверку обязательной и для ноги A/B
    # (VERIFICATION-LEG): она идёт по производственному конфигу, поэтому данные
    # обязаны быть размечены ровно тем токенизатором, который в нём запинен;
    # пин читается из того же файла, а не из net/config.json.
    config_path = getattr(args, "config_path", None)
    if args.model_preset == sft_stage.L3_FULL_PRESET or config_path is not None:
        pinned = (
            sft_stage.config_tokenizer_pin(config_path)
            if config_path is not None
            else sft_stage.config_tokenizer_pin()
        )
        actual_hash = str(tokenizer_info.get("hash") or "")
        if not sft_stage.tokenizer_hash_matches(actual_hash, pinned):
            pin_source = (
                "net/config.json"
                if config_path is None
                else sft_stage.repo_rel(config_path, repo_root)
            )
            label = args.model_preset if config_path is None else f"--config-path {pin_source}"
            reason = (
                f"токенизатор {label} {actual_hash[:16]}… не совпал с пином "
                f"{pin_source} {pinned[:16]}…: данные размечены не каноническим "
                "BPE (ADR-4) — пересоберите tokens/ каноническим токенизатором"
            )
            journal["refusal"] = reason
            _write_journal(journal, out_dir, journal_path=journal_path)
            print(f"[pretrain] ОТКАЗ данных: {reason}", file=sys.stderr, flush=True)
            return journal, False

    # --- H1: decay-окно против объёма Q ------------------------------------
    decay_plan = None
    if args.decay_stream:
        available_q = _decay_available_tokens(tl, paths, args, streams, packed, repo_root)
        decay_plan = tl.decay_window_plan(
            total_steps=horizon,
            decay_ratio=decay_ratio,
            batch_size=planned_batch,
            seq_len=args.seq_len,
            available_tokens=available_q,
        )
        if decay_plan.adjusted:
            if decay_ratio_explicit:
                reason = (
                    f"явный --decay-ratio {decay_ratio} не влезает в Q: {decay_plan.reason}. "
                    "Уменьшите --decay-ratio или возьмите шард Q больше (H1)"
                )
                journal["refusal"] = reason
                _write_journal(journal, out_dir, journal_path=journal_path)
                print(f"[pretrain] ОТКАЗ данных: {reason}", file=sys.stderr, flush=True)
                return journal, False
            decay_ratio = decay_plan.decay_ratio
            print(f"[pretrain] H1: {decay_plan.reason}", file=sys.stderr, flush=True)
    decay_start = (
        tl.decay_start_step(
            horizon, warmup_ratio=args.warmup_ratio, decay_ratio=decay_ratio
        )
        if args.decay_stream
        else None
    )

    # --- К5: файл-стоп килл-свитча (argv → env STOP_FILE → курсор) ---------
    stop_file = resolve_stop_file(args, cursor_run)

    try:
        if packed:
            loader = tl.PackedTokenLoader(
                tokens_root=paths["tokens_root"],
                streams=streams,
                seq_len=args.seq_len,
                batch_size=loader_batch,
                mix=mix,
                cursor=data_cursor,
                decay_stream=args.decay_stream,
                decay_start=decay_start,
                decay_shard_root=paths["tokens_root"] if args.decay_stream else None,
            )
        else:
            loader = tl.PretrainMixLoader(
                shard_root=paths["shard_root"],
                streams=streams,
                encode=tokenizer.encode,
                seq_len=args.seq_len,
                batch_size=loader_batch,
                seed=args.seed,
                mix=mix,
                shuffle_window=args.shuffle_window,
                max_doc_tokens=args.max_doc_tokens,
                cursor=data_cursor,
                decay_stream=args.decay_stream,
                decay_start=decay_start,
                decay_shard_root=paths["decay_shard_root"] if args.decay_stream else None,
                decay_seed=args.decay_seed,
            )
    except tl.PretrainDataError as exc:
        journal["refusal"] = f"данные: {exc}"
        _write_journal(journal, out_dir, journal_path=journal_path)
        print(f"[pretrain] ОТКАЗ данных: {exc}", file=sys.stderr, flush=True)
        return journal, False

    journal["data"] = {
        "kind": data_kind,
        "shard_root": sft_stage.repo_rel(paths["shard_root"], repo_root),
        "tokens_root": sft_stage.repo_rel(paths["tokens_root"], repo_root),
        "streams": describe_loader(loader, streams, repo_root),
        "mix_declared": mix,
        "shuffle_window": args.shuffle_window if not packed else None,
        "streaming": True,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "loader_batch": loader_batch,
        "micro_batch": args.micro_batch,
        "accum_tokens": args.accum_tokens,
        "planned_batch": planned_batch,
        "decay": (
            {
                "stream": args.decay_stream,
                "start_step": decay_start,
                "horizon_steps": horizon,
                "ratio": decay_ratio,
                "ratio_explicit": decay_ratio_explicit,
                "window_plan": (
                    {
                        "adjusted": decay_plan.adjusted,
                        "decay_steps": decay_plan.decay_steps,
                        "window_tokens": decay_plan.window_tokens,
                        "available_tokens": decay_plan.available_tokens,
                        "reason": decay_plan.reason,
                    }
                    if decay_plan is not None
                    else None
                ),
                "shard_root": sft_stage.repo_rel(
                    paths["decay_shard_root"], repo_root
                ),
                "seed": args.decay_seed if args.decay_seed is not None else args.seed,
                "shards": describe_stream(loader, args.decay_stream, repo_root),
            }
            if args.decay_stream
            else None
        ),
    }

    # --- цикл -------------------------------------------------------------
    manager = ckpt_manager
    train_config = tl.TrainConfig(
        steps=args.steps,
        total_steps=args.total_steps,
        lr=args.lr,
        schedule=args.schedule,
        warmup_ratio=args.warmup_ratio,
        decay_ratio=decay_ratio,
        seed=args.seed,
        chunk_size=args.chunk_size,
        grad_checkpointing=bool(grad_checkpointing),
        grad_checkpointing_policy=args.grad_checkpointing_policy,
        param_dtype=args.param_dtype,
        checkpoint_every=args.checkpoint_every,
        checkpoint_every_min=args.ckpt_every_min,
        keep_last=args.keep_last,
        ckpt_dir=paths["ckpt_dir"],
        metrics_path=paths["metrics_path"],
        peak_tflops=args.peak_tflops,
        peak_tflops_source=args.peak_tflops_source,
        usd_per_gpu_hour=args.usd_per_gpu_hour,
        log_every=args.log_every,
        stop_file=stop_file,
        stop_check_every=args.stop_check_every,
        data_kind=data_kind,
        micro_batch=loader_batch,
        accum_tokens=args.accum_tokens,
        ns_steps=args.ns_steps,
        legacy_muon_all_2d=bool(args.legacy_muon_all_2d),
    )
    journal["loop"] = {
        "steps_requested": args.steps,
        "total_steps": horizon,
        "schedule": args.schedule,
        "lr": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "decay_ratio": decay_ratio,
        "decay_ratio_requested": args.decay_ratio,
        "decay_ratio_explicit": decay_ratio_explicit,
        "decay_stream": args.decay_stream,
        "decay_start": decay_start,
        "grad_checkpointing": bool(grad_checkpointing),
        "grad_checkpointing_policy": args.grad_checkpointing_policy,
        "grad_ckpt_policy": getattr(cfg, "grad_ckpt_policy", None),
        "ce_chunk_tokens": getattr(cfg, "ce_chunk_tokens", None),
        "loss_impl": _loss_impl(cfg),
        "param_dtype": args.param_dtype,
        "checkpoint_every": args.checkpoint_every,
        "checkpoint_every_min": args.ckpt_every_min,
        "stop_file": sft_stage.repo_rel(stop_file, repo_root) if stop_file else None,
        "resume": bool(args.resume),
        "data_kind": data_kind,
        "micro_batch": loader_batch,
        "accum_tokens": args.accum_tokens,
        "planned_batch": planned_batch,
        "tokens_per_step": (
            int(args.accum_tokens) if int(args.accum_tokens) > 0
            else loader_batch * int(args.seq_len)
        ),
        # ADR-048: режим оптимизатора — часть условий прогона: без него
        # сравнение «до/после» по метрикам неотличимо от смены чего-то ещё.
        "ns_steps": args.ns_steps,
        "legacy_muon_all_2d": bool(args.legacy_muon_all_2d),
    }
    journal["gpu"] = {
        "peak_tflops": args.peak_tflops,
        "peak_tflops_source": args.peak_tflops_source,
        "peak_declared_not_measured": args.peak_tflops is not None,
    }
    print(
        f"[pretrain] {args.model_preset}: vocab={cfg.vocab_size}, "
        f"T={args.seq_len}, B={args.batch_size} (micro={loader_batch}, "
        f"accum={args.accum_tokens or 'off'}), шагов {args.steps}, "
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
        _write_journal(journal, out_dir, journal_path=journal_path)
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
            "data_consumed": data_consumed_block(stats, data_kind),
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
    # H5: чтение метрик — в try: битая/оборванная преемпшном строка не должна
    # обнулять уже сформированный журнал (он важнее сводки).
    try:
        journal["metrics_summary"] = metrics_summary(
            tl.MetricsWriter.read(paths["metrics_path"]), paths["metrics_path"], repo_root
        )
    except Exception as exc:  # сводка вторична — сам журнал не теряем
        journal["metrics_summary"] = {"error": f"{type(exc).__name__}: {exc}"}
    journal["notes"] = [
        "mfu_params_only: MFU по параметрической части (6·N_active·tokens), вклад "
        "внимания не входит; N для MFU — active_param_count, не N_total",
        "mfu: доля от ОБЪЯВЛЕННОГО пика; на смоуке с крошечной моделью число мало "
        "по построению и не является замером эффективности железа (на H800 "
        "калибровка ~35% — ожидаемый порядок, не гарантия)",
        "token_share: фактическая доля потоков в израсходованном окне корпуса; "
        "для packed — по слотам записей",
    ]
    _write_journal(journal, out_dir, journal_path=journal_path)

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


def _write_journal(
    journal: dict[str, Any], out_dir: Path, *, journal_path: Path | None = None
) -> Path:
    """Записать журнал, сняв абсолютные пути; утечка после снятия — дефект."""
    repo_root = sft_stage.detect_repo_root()
    journal = portable_paths(journal, repo_root)
    leaks = absolute_path_leaks(journal)
    if leaks:
        journal["path_leaks"] = leaks
    path = Path(journal_path) if journal_path is not None else out_dir / "journal.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(journal, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)
    return path


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    jax_preflight.gate_or_exit()  # ADR-041: состояние стенда до реального прогона
    journal, ok = execute(args)
    if args.json:
        print(json.dumps(journal, ensure_ascii=False, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
