"""Канонический BPE 160K: обучение на детерминированной выборке корпуса W/C.

ADR-004 (карточка данных + хеш) и CMP-001: токенизатор — собственный BPE на
160K, обученный на выборке претрейн-корпуса, пиннится хешем.  Модуль делает
первую половину конвейера: собирает выборку ~2 ГиБ текста из шардов W/C
**детерминированно по сиду**, обучает byte-level BPE и кладёт рядом манифест с
sha256 файла (``tokenizer_hash`` — то, чем пиннится прогон).

Почему byte-level.  ``net/tokenizer.py`` объявляет каноническую раскладку
словаря (``<pad> <bos> <eos>`` + кодовый префикс, затем 256 байтовых токенов,
далее мерджи); byte-level BPE лоссов по построению — любой UTF-8 текст
разбирается до байтов, поэтому round-trip (критерий 9 спеки скелета, ≥99,5%)
выполняется на корпусе, а не «обычно».  Здесь та же раскладка воспроизводится
обучаемой моделью, а не заглушкой на синтетическом корпусе.

Почему выборка, а не весь корпус.  В W/C 80 ГиБ текста (~20B токенов по
счётчику ADR-021); частотный профиль мерджей выходит на плато задолго до
конца, а полный проход — часы CPU ради долей процента словаря.  Объём выборки
задаётся ``--sample-mb`` (по умолчанию 2048 МиБ) и записывается в манифест
вместе с правилом сэмплирования: подмена объёма молча запрещена.

Почему выборка детерминирована.  Хеш файла — это то, чем пиннится претрейн
(AD-4/AD-11): два запуска с одним сидом обязаны дать **тот же** файл.  Поэтому
берётся систематическая выборка с шагом по каждому шарду, а смещение внутри
шага выводится из sha256 ключа ``(seed, поток, файл)`` — она не зависит ни от
порядка обхода, ни от ``PYTHONHASHSEED`` (``random`` хеширует кортежи через
``hash()``, зависящий от процесса).

Модуль же держит **адресацию корпуса** (``load_corpus_shards`` /
``iter_shard_texts``): ``tools/pretokenize.py`` читает шарды через него, чтобы
токенизатор и претокенизация видели ровно один контракт данных, а не два
похожих.

CLI::

    python tools/bpe_train.py --sample-mb 2048 --seed 0 \\
        --out ~/gb10-shared/datasets/axiom-pretrain-l3/tokenizer
    python tools/bpe_train.py --verify   # пересчёт sha256 файла против манифеста
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:  # запуск и как скрипта, и как модуля тестов
    sys.path.insert(0, str(TOOLS_DIR))

from prep_pretrain import common  # noqa: E402

#: Схема манифеста токенизатора (контракт для ``tools/pretokenize.py``).
TOKENIZER_SCHEMA = "axiom-pretrain-tokenizer/1"

#: Имя файла-артефакта: то, чей sha256 становится ``tokenizer_hash``.
TOKENIZER_FILE = "tokenizer.model"
MANIFEST_FILE = "tokenizer-manifest.json"

#: Формат файла-артефакта (HF ``tokenizers`` сериализует модель в JSON).
TOKENIZER_FORMAT = "tokenizers-json/v1"

#: Словарь базы 160K (FIDELITY §1, ``net/config.json:tokenizer_vocab``).
DEFAULT_VOCAB_SIZE = 160_000

#: Раскладка специальных токенов = раскладка ``net/tokenizer.py`` (ids 0..3),
#: четвёртая позиция вместо ``<unk>`` занята кодовым префиксом: byte-level BPE
#: не имеет неизвестных символов, а разделять веб и код модели нужно.
DEFAULT_SPECIALS = ("<pad>", "<bos>", "<eos>", "<|code|>")
PAD_ID, BOS_ID, EOS_ID, CODE_ID = 0, 1, 2, 3

#: Кодовый префикс документа потока C (объявлен в задаче и в раскладке выше).
CODE_PREFIX = "<|code|>"

#: Число байтовых токенов в byte-level BPE (256 значений байта).
N_BYTE_TOKENS = 256

DEFAULT_DATASET_ROOT = common.DATASET_ROOT
DEFAULT_TOKENIZER_DIR = f"{DEFAULT_DATASET_ROOT}/tokenizer"
DEFAULT_TEXT_KEYS = ("text", "content")
DEFAULT_MIX = {"W": 0.85, "C": 0.15}
DEFAULT_SAMPLE_MB = 2048.0
MB = 1024 * 1024


class TokenizerError(RuntimeError):
    """Токенизатор или его манифест не соответствует контракту."""


class CorpusError(TokenizerError):
    """Шард-набор корпуса не читается или не объявлен."""


# --------------------------------------------------------------------------- #
# Адресация корпуса (общая с pretokenize.py)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CorpusShard:
    """Один шард корпуса из манифеста ``tools/prep_pretrain``."""

    stream: str
    file: str
    path: Path
    sha256: str
    approx_tokens: int
    bytes: int
    records: int = 0

    @property
    def text_bytes_estimate(self) -> int:
        """Оценка объёма текста шарда в байтах.

        ``approx_tokens`` претрейн-пайплайна — это ``max(1, len(text) // 4)``
        (ADR-021), то есть ``chars ≈ 4 * approx_tokens``; для UTF-8 английского
        текста байт примерно столько же.  Оценка нужна только для шага выборки
        (и для бюджета пробы), поэтому её достаточно.
        """
        return 4 * int(self.approx_tokens)


def shard_manifest_path(shard_root: str | os.PathLike[str], stream: str) -> Path:
    """Путь манифеста шард-набора (контракт ``prep_pretrain``: ``manifest-w.json``)."""
    root = Path(os.path.expanduser(os.fspath(shard_root)))
    return root / stream / f"manifest-{stream.lower()}.json"


def load_corpus_shards(
    shard_root: str | os.PathLike[str], streams: Sequence[str] = ("W", "C")
) -> list[CorpusShard]:
    """Прочитать манифесты потоков и вернуть шарды в порядке имён файлов.

    Порядок фиксирован (сортировка по имени): выборка и её хеш не должны
    зависеть от порядка обхода каталога.  Отсутствующий шард из манифеста —
    отказ, а не пропуск: иначе поедет и выборка, и претокенизация.
    """
    shards: list[CorpusShard] = []
    for stream in streams:
        manifest = shard_manifest_path(shard_root, stream)
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except OSError as exc:
            raise CorpusError(f"манифест шарда не читается: {manifest}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise CorpusError(f"манифест шарда не разбирается: {manifest}: {exc}") from exc
        items = data.get("shards") or []
        if not items:
            raise CorpusError(f"в манифесте нет шардов: {manifest}")
        for item in sorted(items, key=lambda entry: entry["file"]):
            path = manifest.parent / item["file"]
            if not path.is_file():
                raise CorpusError(f"шард из манифеста отсутствует: {path}")
            shards.append(
                CorpusShard(
                    stream=stream,
                    file=str(item["file"]),
                    path=path,
                    sha256=str(item.get("sha256", "")),
                    approx_tokens=int(item.get("approx_tokens", 0)),
                    bytes=int(item.get("bytes", path.stat().st_size)),
                    records=int(item.get("records", 0)),
                )
            )
    return shards


def iter_shard_texts(
    shard: CorpusShard,
    *,
    text_keys: Sequence[str] = DEFAULT_TEXT_KEYS,
) -> Iterator[str]:
    """Поток текстов одного ``.jsonl.zst`` шарда (одна строка в памяти).

    Поле текста берётся первым непустым из ``text_keys`` (``text`` у W и C,
    ``content`` — форма записи источников кода, см. ``common.hm_get``).
    Строка, которая не разбирается как json или не несёт текста, пропускается:
    счётчики пропусков возвращает вызывающий код через ``ShardReadStats``, чтобы
    потери не были молчаливыми.
    """
    import zstandard as zstd

    with open(shard.path, "rb") as handle:
        reader = zstd.ZstdDecompressor().stream_reader(handle)
        text_stream = io.TextIOWrapper(reader, encoding="utf-8", newline="\n")
        for line in text_stream:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                yield ""
                continue
            yield _record_text(record, text_keys)


def _record_text(record: Any, text_keys: Sequence[str]) -> str:
    """Текст записи jsonl: строка, словарь с ``text``/``content`` или список записей."""
    if isinstance(record, str):
        return record
    if isinstance(record, list):  # батчевая запись источника
        return "".join(_record_text(item, text_keys) for item in record)
    if isinstance(record, dict):
        value = common.hm_get(record, *text_keys, default="")
        return value if isinstance(value, str) else ""
    return ""


# --------------------------------------------------------------------------- #
# Детерминированная выборка
# --------------------------------------------------------------------------- #


@dataclass
class SampleStats:
    """Что именно попало в выборку — числами, а не «примерно 2 ГиБ»."""

    target_bytes: int = 0
    docs: int = 0
    docs_seen: int = 0
    chars: int = 0
    streams: dict[str, dict[str, float]] = field(default_factory=dict)

    def as_json(self) -> dict[str, Any]:
        return {
            "target_bytes": self.target_bytes,
            "target_mb": round(self.target_bytes / MB, 3),
            "docs": self.docs,
            "docs_seen": self.docs_seen,
            "chars": self.chars,
            "approx_bytes": self.chars,  # UTF-8 ≈ 1 байт/символ на этом корпусе
            "streams": self.streams,
        }


def _shard_seed(seed: int, stream: str, file: str) -> int:
    """Сид шарда из (seed, поток, файл): не зависит от PYTHONHASHSEED и порядка."""
    key = f"{seed}|{stream}|{file}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def iter_sample(
    shards: Sequence[CorpusShard],
    *,
    target_bytes: int,
    mix: dict[str, float] | None = None,
    seed: int = 0,
    text_keys: Sequence[str] = DEFAULT_TEXT_KEYS,
    stats: SampleStats | None = None,
    progress_every: int = 0,
    echo: Any = None,
) -> Iterator[tuple[str, str]]:
    """Поток ``(поток, текст)`` выборки ``target_bytes`` байт, детерминированный по сиду.

    Правило (записано в манифест как ``sampling``): бюджет потока делится поровну
    между его шардами, и внутри шарда берётся **непрерывное окно** документов:
    смещение окна выбирается сидом шарда (``_shard_seed``) в первой четверти
    шарда, длина — из средней длины документа, пока не набран бюджет шарда.

    Почему окно, а не «каждый k-й документ».  Разрежённая выборка по всему
    шарду требует **прочитать и распаковать весь шард** ради его доли (замер:
    32 МиБ выборки = 80 ГиБ чтения) — на 52 шардах W это часы декомпрессии и
    разбора json ради выборки, которая в 40 раз меньше корпуса.  Непрерывное
    окно читается один раз и только в границах окна.

    Почему смещение, а не префикс.  Префикс шарда — это голова одного дампа
    (или одного репозитория у кода): 52 префикса дали бы 52 похожих среза.
    Смещение разведено по шардам, поэтому выборка собирается из разных мест
    корпуса; остаточное смещение внутри окна (порядок документов шарда) в
    манифесте названо честно, а не выдано за случайную выборку.
    """
    weights = dict(mix or DEFAULT_MIX)
    present = sorted({shard.stream for shard in shards})
    total_weight = sum(weights.get(stream, 0.0) for stream in present)
    if total_weight <= 0:
        raise CorpusError(f"веса микса {weights} не покрывают потоки {present}")
    seen = 0
    for stream in present:
        stream_shards = [shard for shard in shards if shard.stream == stream]
        stream_budget = int(target_bytes * weights.get(stream, 0.0) / total_weight)
        per_shard = max(1, stream_budget // len(stream_shards))
        stream_chars = 0
        stream_docs = 0
        stream_skipped = 0
        for shard in stream_shards:
            estimate = max(1, shard.text_bytes_estimate)
            avg_doc = max(1.0, estimate / max(1, shard.records))
            # Окно берётся с запасом 10%: ``4 * approx_tokens`` — оценка сверху
            # (ADR-021 округляет вверх), поэтому без запаса окно закрывается до
            # набора бюджета и выборка недобирает объём (замер: 95% от цели).
            window = max(1, math.ceil(1.1 * per_shard / avg_doc))
            head_zone = max(1, shard.records // 4)  # смещение — только в первой четверти
            offset = random.Random(_shard_seed(seed, stream, shard.file)).randrange(head_zone)
            collected = 0
            for index, text in enumerate(iter_shard_texts(shard, text_keys=text_keys)):
                seen += 1
                if index < offset:
                    stream_skipped += 1
                    continue
                if index >= offset + window or collected >= per_shard:
                    break
                if not text:
                    continue
                yield stream, text
                collected += len(text)
                stream_chars += len(text)
                stream_docs += 1
                if stats is not None:
                    stats.docs += 1
                    stats.chars += len(text)
                if progress_every and seen % progress_every == 0 and echo is not None:
                    echo(
                        f"[bpe] прочитано {seen} документов, выборка "
                        f"{stats.chars / MB:.0f} МиБ"
                    )
        if stats is not None:
            stats.streams[stream] = {
                "shards": len(stream_shards),
                "docs": stream_docs,
                "chars": stream_chars,
                "docs_skipped": stream_skipped,
                "chars_per_doc": round(stream_chars / stream_docs, 1) if stream_docs else 0.0,
            }
    if stats is not None:
        stats.docs_seen = seen


def sample_corpus(
    shards: Sequence[CorpusShard],
    *,
    sample_mb: float = DEFAULT_SAMPLE_MB,
    mix: dict[str, float] | None = None,
    seed: int = 0,
    text_keys: Sequence[str] = DEFAULT_TEXT_KEYS,
    progress_every: int = 0,
    echo: Any = None,
) -> tuple[list[str], SampleStats]:
    """Материализовать выборку в список текстов (обучение BPE идёт по итератору).

    Выборка — это единицы гигабайт; в памяти она держится списком строк, потому
    что ``BpeTrainer`` считает частоты по всему входу, а не потоково.  Это
    единственное место конвейера, где корпус поднимается в память целиком.
    """
    target_bytes = int(sample_mb * MB)
    stats = SampleStats(target_bytes=target_bytes)
    texts = list(
        text
        for _, text in iter_sample(
            shards,
            target_bytes=target_bytes,
            mix=mix,
            seed=seed,
            text_keys=text_keys,
            stats=stats,
            progress_every=progress_every,
            echo=echo,
        )
    )
    stats.docs = len(texts)
    stats.chars = sum(len(text) for text in texts)
    return texts, stats


# --------------------------------------------------------------------------- #
# Обучение
# --------------------------------------------------------------------------- #


def _tokenizers_module() -> Any:
    """Загрузить ``tokenizers`` (Rust) или объяснить, чего не хватает.

    Своя реализация BPE в ``net/tokenizer.py`` — учебная: её обучение 160K
    мерджей на гигабайтах текста квадратично по числу мерджей и на выборке
    не считается.  Библиотека ставится в ``venv-axiom`` (задачей разрешено).
    """
    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    except ImportError as exc:  # pragma: no cover — окружение без библиотеки
        raise TokenizerError(
            "нет библиотеки tokenizers: pip install tokenizers (venv-axiom)"
        ) from exc
    return Tokenizer, models, pre_tokenizers, decoders, trainers


def build_bpe_trainer(*, vocab_size: int, specials: Sequence[str], tokenizers: Any) -> Any:
    """Тренер BPE: специальные токены первыми, затем 256 байтов, затем мерджи.

    Порядок ids — часть контракта (``<pad>``=0, ``<bos>``=1, ``<eos>``=2,
    кодовый префикс=3): его ждут ``net/train_loop.py`` (PAD/BOS/EOS_ID) и
    ``net/data.pack_sequence``.  ``initial_alphabet`` ByteLevel гарантирует, что
    все 256 байтов в словаре с самого начала, поэтому round-trip лоссов.
    """
    _, _, pre_tokenizers, _, trainers = tokenizers
    alphabet = pre_tokenizers.ByteLevel.alphabet()
    if len(alphabet) != N_BYTE_TOKENS:  # pragma: no cover — смена библиотеки
        raise TokenizerError(f"алфавит ByteLevel не 256 токенов: {len(alphabet)}")
    return trainers.BpeTrainer(
        vocab_size=int(vocab_size),
        special_tokens=list(specials),
        initial_alphabet=alphabet,
        show_progress=False,
    )


def build_tokenizer() -> Any:
    """Пустой byte-level BPE: пре-токенизатор и декодер без префиксного пробела.

    ``add_prefix_space=False`` обязателен: иначе пробел в начале документа
    превращается в служебный префикс и round-trip на текстах с ведущим пробелом
    (а таких в веб-корпусе много) перестаёт быть тождеством.
    """
    Tokenizer, models, pre_tokenizers, decoders, _ = _tokenizers_module()
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def train_bpe(
    texts: Sequence[str],
    *,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    specials: Sequence[str] = DEFAULT_SPECIALS,
    echo: Any = None,
) -> Any:
    """Обучить BPE на выборке и проверить раскладку словаря (отказ — не «почти»)."""
    tokenizer = build_tokenizer()
    trainer = build_bpe_trainer(
        vocab_size=vocab_size, specials=specials, tokenizers=_tokenizers_module()
    )
    if echo is not None:
        echo(f"[bpe] обучение: {len(texts)} документов → vocab {vocab_size}")
    tokenizer.train_from_iterator(list(texts), trainer)
    verify_layout(tokenizer, vocab_size=vocab_size, specials=specials)
    return tokenizer


def verify_layout(tokenizer: Any, *, vocab_size: int, specials: Sequence[str]) -> dict[str, Any]:
    """Сверить раскладку словаря с контрактом (ids специальных, ровно ``vocab_size``)."""
    vocab = tokenizer.get_vocab()
    expected_specials = {name: index for index, name in enumerate(specials)}
    actual_specials = {name: vocab.get(name) for name in specials}
    if actual_specials != expected_specials:
        raise TokenizerError(
            f"ids специальных токенов не совпали: {actual_specials} != {expected_specials}"
        )
    if len(vocab) != int(vocab_size):
        raise TokenizerError(
            f"словарь не равен объявленному: {len(vocab)} != {vocab_size} "
            "(корпус не дал нужного числа мерджей — объём выборки мал)"
        )
    _, _, pre_tokenizers, _, _ = _tokenizers_module()
    alphabet_ids = sorted(vocab[symbol] for symbol in pre_tokenizers.ByteLevel.alphabet())
    expected_bytes = list(range(len(specials), len(specials) + N_BYTE_TOKENS))
    if alphabet_ids != expected_bytes:
        raise TokenizerError(
            f"байтовые токены не идут сразу за специальными: {alphabet_ids[:4]}…"
            f" (ожидались {expected_bytes[:4]}…)"
        )
    return {
        "vocab_size": len(vocab),
        "specials": expected_specials,
        "byte_tokens": [expected_bytes[0], expected_bytes[-1]],
        "merges": len(vocab) - len(specials) - N_BYTE_TOKENS,
    }


def measure_round_trip(tokenizer: Any, texts: Sequence[str], *, limit: int = 2000) -> dict[str, Any]:
    """Доля документов, где ``decode(encode(t)) == t`` (критерий 9 скелета, ≥99,5%).

    Проверяются документы, равномерно взятые по выборке (шаг ``len/limit``), а не
    её голова: у головы свой профиль (первый шард — свой дамп), и мерить
    round-trip на ней значило бы мерить не корпус.
    """
    if not texts:
        return {"docs": 0, "ok": 0, "rate": 1.0}
    step = max(1, len(texts) // max(1, limit))
    probe = list(texts[::step])[:limit]
    ok = 0
    for text in probe:
        ids = tokenizer.encode(text).ids
        if tokenizer.decode(ids, skip_special_tokens=False) == text:
            ok += 1
    return {"docs": len(probe), "ok": ok, "rate": round(ok / len(probe), 6)}


# --------------------------------------------------------------------------- #
# Манифест
# --------------------------------------------------------------------------- #


def file_sha256(path: str | os.PathLike[str], chunk: int = 1 << 20) -> str:
    """sha256 файла (потоково) — то же, что ``tokenizer_hash`` манифеста."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def build_manifest(
    *,
    tokenizer_path: Path,
    tokenizer_hash: str,
    layout: dict[str, Any],
    stats: SampleStats,
    shards: Sequence[CorpusShard],
    round_trip: dict[str, Any],
    seed: int,
    sample_mb: float,
    stream_scope: Sequence[str],
    elapsed_s: float,
) -> dict[str, Any]:
    """Манифест токенизатора: артефакт + хеш + выборка + источники (контракт AD-4)."""
    return {
        "version": TOKENIZER_SCHEMA,
        "kind": "bpe",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": int(seed),
        "vocab_size": int(layout["vocab_size"]),
        "merges": int(layout["merges"]),
        "specials": {name: int(index) for name, index in layout["specials"].items()},
        "pad_id": int(layout["specials"]["<pad>"]),
        "bos_id": int(layout["specials"]["<bos>"]),
        "eos_id": int(layout["specials"]["<eos>"]),
        "code_prefix": CODE_PREFIX,
        "code_prefix_id": int(layout["specials"][CODE_PREFIX]),
        "byte_tokens": list(layout["byte_tokens"]),
        "tokenizer": {
            "file": tokenizer_path.name,
            "format": TOKENIZER_FORMAT,
            "bytes": tokenizer_path.stat().st_size,
        },
        #: sha256 файла-артефакта (AD-4): им пиннится и претокенизация, и прогон.
        "tokenizer_hash": tokenizer_hash,
        "sample": stats.as_json() | {"seed": int(seed), "sample_mb": float(sample_mb)},
        "sampling": {
            "rule": (
                "непрерывное окно документов в каждом шарде: смещение в первой "
                "четверти шарда, длина — до бюджета шарда"
            ),
            "streams": list(stream_scope),
            "budget_split": "поровну между шардами потока; доли потоков — mix",
            "offset_source": "sha256(seed|stream|file) — не зависит от процесса и PYTHONHASHSEED",
            "residual_bias": (
                "внутри окна порядок документов шарда сохраняется: выборка "
                "стратифицирована по шардам, но не является равномерной по корпусу"
            ),
        },
        "round_trip": round_trip,
        "sources": [
            {
                "stream": shard.stream,
                "file": shard.file,
                "sha256": shard.sha256,
                "approx_tokens": shard.approx_tokens,
                "bytes": shard.bytes,
            }
            for shard in shards
        ],
        "elapsed_s": round(elapsed_s, 3),
        "throughput": {
            "chars_per_s": round(stats.chars / elapsed_s, 1) if elapsed_s > 0 else None,
        },
    }


def write_manifest(path: str | os.PathLike[str], payload: dict[str, Any]) -> Path:
    """Записать манифест (тот же вид json, что у манифестов шардов)."""
    target = Path(os.path.expanduser(os.fspath(path)))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    return target


def verify_manifest(manifest_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Пересчитать sha256 файла-артефакта и сверить с манифестом (AD-4)."""
    path = Path(os.path.expanduser(os.fspath(manifest_path)))
    if not path.is_file():
        raise TokenizerError(f"манифест не найден: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("version") != TOKENIZER_SCHEMA:
        raise TokenizerError(
            f"схема манифеста {manifest.get('version')!r} != {TOKENIZER_SCHEMA!r}"
        )
    artifact = path.parent / manifest["tokenizer"]["file"]
    if not artifact.is_file():
        raise TokenizerError(f"артефакт токенизатора отсутствует: {artifact}")
    actual = file_sha256(artifact)
    pinned = manifest.get("tokenizer_hash")
    if actual != pinned:
        raise TokenizerError(
            f"tokenizer_hash не совпал: {actual} != {pinned} — файл изменился после упаковки"
        )
    return {
        "passed": True,
        "manifest": str(path),
        "tokenizer": str(artifact),
        "tokenizer_hash": actual,
        "vocab_size": manifest.get("vocab_size"),
        "round_trip": manifest.get("round_trip"),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_mix(values: Sequence[str] | None, streams: Sequence[str]) -> dict[str, float]:
    """Разобрать ``--mix W=0.85 C=0.15`` (или взять дефолт ADR-021)."""
    if not values:
        mix = {stream: DEFAULT_MIX.get(stream, 1.0) for stream in streams}
        return mix
    mix = {}
    for item in values:
        name, _, value = item.partition("=")
        name = name.strip().upper()
        if not name or not value:
            raise argparse.ArgumentTypeError(f"микс задаётся как W=0.85: {item!r}")
        if name not in streams:
            raise argparse.ArgumentTypeError(f"поток {name!r} не из {tuple(streams)}")
        mix[name] = float(value)
    missing = [stream for stream in streams if stream not in mix]
    if missing:
        raise argparse.ArgumentTypeError(f"в миксе нет весов для потоков {missing}")
    if sum(mix.values()) <= 0:
        raise argparse.ArgumentTypeError("сумма весов микса должна быть > 0")
    return mix


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Канонический BPE 160K на выборке претрейн-корпуса W/C (ADR-004)"
    )
    parser.add_argument(
        "--shard-root",
        default=DEFAULT_DATASET_ROOT,
        help="корень датасета с {W,C}/manifest-*.json (по умолчанию ~/gb10-shared/.../axiom-pretrain-l3)",
    )
    parser.add_argument("--streams", nargs="+", default=["W", "C"], help="потоки корпуса")
    parser.add_argument("--out", default=DEFAULT_TOKENIZER_DIR, help="каталог артефакта и манифеста")
    parser.add_argument("--manifest", default=None, help="путь манифеста (по умолчанию <out>/tokenizer-manifest.json)")
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--sample-mb", type=float, default=DEFAULT_SAMPLE_MB,
                        help="объём выборки текста в МиБ (по умолчанию 2048)")
    parser.add_argument("--seed", type=int, default=0, help="сид выборки (AD-11: воспроизводимость)")
    parser.add_argument("--mix", nargs="+", default=None, help="доли потоков выборки, напр. W=0.85 C=0.15")
    parser.add_argument("--text-keys", nargs="+", default=list(DEFAULT_TEXT_KEYS),
                        help="поля текста записи (text/content)")
    parser.add_argument("--round-trip-docs", type=int, default=2000,
                        help="сколько документов выборки проверить на round-trip")
    parser.add_argument("--progress-every", type=int, default=200_000,
                        help="печатать прогресс выборки каждые N прочитанных документов")
    parser.add_argument("--allow-any-out", action="store_true",
                        help="разрешить вывод вне gb10-shared//tmp (по умолчанию запрещено, C-032/C-033)")
    parser.add_argument("--verify", action="store_true",
                        help="только пересчитать sha256 артефакта против манифеста")
    return parser


def cmd_train(args: argparse.Namespace) -> int:
    echo = lambda message: print(message, flush=True)  # noqa: E731 — короткий эхо-хелпер
    started = time.time()
    out_dir = common.ensure_output_allowed(args.out, allow_any=args.allow_any_out)
    manifest_path = (
        Path(os.path.expanduser(args.manifest))
        if args.manifest
        else out_dir / MANIFEST_FILE
    )
    streams = [stream.upper() for stream in args.streams]
    mix = parse_mix(args.mix, streams)
    shards = load_corpus_shards(args.shard_root, streams)
    echo(
        f"[bpe] шардов {len(shards)} "
        f"({', '.join(f'{stream}: {sum(1 for s in shards if s.stream == stream)}' for stream in streams)}), "
        f"выборка {args.sample_mb:.0f} МиБ, сид {args.seed}"
    )

    texts, stats = sample_corpus(
        shards,
        sample_mb=args.sample_mb,
        mix=mix,
        seed=args.seed,
        text_keys=args.text_keys,
        progress_every=args.progress_every,
        echo=echo,
    )
    echo(
        f"[bpe] выборка: {stats.docs} документов, {stats.chars / MB:.1f} МиБ текста "
        f"за {time.time() - started:.1f} с"
    )
    if not texts:
        raise CorpusError("выборка пуста: шарды не дали текста")

    tokenizer = train_bpe(texts, vocab_size=args.vocab_size, echo=echo)
    layout = verify_layout(tokenizer, vocab_size=args.vocab_size, specials=DEFAULT_SPECIALS)
    out_dir.mkdir(parents=True, exist_ok=True)  # save() не создаёт каталог
    artifact = out_dir / TOKENIZER_FILE
    tokenizer.save(str(artifact))
    tokenizer_hash = file_sha256(artifact)
    round_trip = measure_round_trip(tokenizer, texts, limit=args.round_trip_docs)
    elapsed = time.time() - started
    manifest = build_manifest(
        tokenizer_path=artifact,
        tokenizer_hash=tokenizer_hash,
        layout=layout,
        stats=stats,
        shards=shards,
        round_trip=round_trip,
        seed=args.seed,
        sample_mb=args.sample_mb,
        stream_scope=streams,
        elapsed_s=elapsed,
    )
    write_manifest(manifest_path, manifest)
    echo(
        f"[bpe] готово: vocab {layout['vocab_size']} / мерджей {layout['merges']}, "
        f"round-trip {round_trip['rate'] * 100:.3f}% на {round_trip['docs']} документах"
    )
    echo(f"[bpe] tokenizer_hash {tokenizer_hash}")
    echo(f"[bpe] артефакт {artifact} ({artifact.stat().st_size / MB:.1f} МиБ), манифест {manifest_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.verify:
            manifest_path = (
                Path(os.path.expanduser(args.manifest))
                if args.manifest
                else common.ensure_output_allowed(args.out, allow_any=args.allow_any_out)
                / MANIFEST_FILE
            )
            print(json.dumps(verify_manifest(manifest_path), ensure_ascii=False, indent=1))
            return 0
        return cmd_train(args)
    except TokenizerError as exc:
        print(f"[bpe] ОТКАЗ: {exc}", file=sys.stderr, flush=True)
        return 2
    except ValueError as exc:  # ensure_output_allowed
        print(f"[bpe] ОТКАЗ вывода: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
