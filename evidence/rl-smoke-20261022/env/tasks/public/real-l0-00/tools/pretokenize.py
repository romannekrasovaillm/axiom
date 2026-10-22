"""Претокенизация корпуса W/C в упакованные бинарники (ADR-004, ADR-021).

Токенизация — самый дорогой невозвратный труд подготовки данных (ADR-004:
«costly»), поэтому она делается **один раз** локально и кладётся на gb10-shared
рядом с шардами: претрейн на аренде читает готовые id-потоки, а не пересчитывает
BPE на каждой эпохе.

Формат (контракт с ``net/train_loop.py``, читатель — ``PackedTokenLoader``)::

    tokens/{W,C}/<shard>.bin   записи ровно по seq_len (T=8192) uint32 LE
    запись = [BOS] + поток[i*(T-1) : (i+1)*(T-1)]
    поток  = конкатенация документов: ([CODE_PREFIX] для C) + encode(text) + [EOS]

Почему запись начинается с BOS и «съедает» один токен потока: даталоадер
``net/train_loop.py`` отдаёт в луп строки ``(B, T)``, у которых позиция 0 — BOS
(``net.data.pack_sequence``).  Здесь это же свойство даётся **файлом**, поэтому
адаптеру чтения не нужна арифметика границ: запись = строка модели.  Цена —
один служебный токен на 8192 (0,012%), и она записана в манифест, а не спрятана.

Почему EOS-разделитель документов, а не «просто склейка»: склейка без границ
учит модель продолжать чужой документ как свой, а границу восстановить потом
нечем.  EOS делает стык документов наблюдаемым и позволяет тесту проверить, что
стыки — это именно стыки: между двумя EOS лежит ровно один документ.

Почему длинный документ режется **без фальшивого EOS** (H3): документ, не
влезший в запись целиком, продолжается со следующей записи (``push_stream``
переносит остаток в новый буфер), а EOS ставится только на его настоящем конце.
Старый путь упаковки (``net.data.pack_sequence``: ``tokens[:T-2] + [eos]``)
рождал EOS посреди документа на каждом разрезе окна — модель училась обрывать
текст на произвольной границе.  Здесь разрез записи служебный: он не добавляет
токенов и не меняет поток, поэтому «между двумя EOS ровно один документ»
держится и на документах длиннее ``T-1``, и читатель ``PackedTokenLoader`` может
отдавать записи как есть, без арифметики границ.

Почему work-pool по шардам: шард — единица независимой работы (свой файл, свой
sha256), поэтому параллелизм не требует ни блока, ни seek по сжатому потоку.
Каждый процесс читает свой шард потоково; память — батч документов и буфер
одной записи.

CLI::

    python tools/pretokenize.py --limit-mb 200 --out /tmp/axiom-tokens-probe --workers 8
    python tools/pretokenize.py --workers 8            # боевой прогон W+C
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Sequence

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:  # запуск и как скрипта, и как модуля тестов
    sys.path.insert(0, str(TOOLS_DIR))

from bpe_train import (  # noqa: E402
    BOS_ID,
    CODE_ID,
    CODE_PREFIX,
    DEFAULT_DATASET_ROOT,
    DEFAULT_TEXT_KEYS,
    EOS_ID,
    MANIFEST_FILE as TOKENIZER_MANIFEST_FILE,
    PAD_ID,
    TOKENIZER_FILE,
    CorpusShard,
    CorpusError,
    TokenizerError,
    file_sha256,
    iter_shard_texts,
    load_corpus_shards,
)
from prep_pretrain import common  # noqa: E402

#: Схема манифеста претокенизированного корпуса (контракт читателя).
TOKENS_SCHEMA = "axiom-pretrain-tokens/1"

#: Раскладка записи .bin — строка, которую читатель сверяет с кодом.
RECORD_LAYOUT = "bos + (seq_len-1) токенов потока"
STREAM_LAYOUT = "документ: [code_prefix для C] + encode(text) + eos"
DTYPE = "uint32"
BYTE_ORDER = "little"

DEFAULT_SEQ_LEN = 8192
DEFAULT_TOKENIZER_DIR = f"{DEFAULT_DATASET_ROOT}/tokenizer"
DEFAULT_TOKENS_DIR = f"{DEFAULT_DATASET_ROOT}/tokens"
DEFAULT_WORKERS = min(8, os.cpu_count() or 1)
MB = 1024 * 1024


class PretokenizeError(RuntimeError):
    """Претокенизация не соответствует контракту данных."""


# --------------------------------------------------------------------------- #
# Токенизатор (пиннинг и раскладка)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TokenizerPin:
    """Запинненный канонический токенизатор: файл, хеш, ids раскладки."""

    path: Path
    tokenizer_hash: str
    vocab_size: int
    code_prefix: str
    code_prefix_id: int
    eos_id: int
    bos_id: int
    pad_id: int
    manifest_path: Path | None = None

    def as_json(self) -> dict[str, Any]:
        return {
            "file": self.path.name,
            "path": str(self.path),
            "sha256": self.tokenizer_hash,
            "tokenizer_hash": self.tokenizer_hash,
            "vocab_size": self.vocab_size,
            "code_prefix": self.code_prefix,
            "code_prefix_id": self.code_prefix_id,
            "eos_id": self.eos_id,
            "bos_id": self.bos_id,
            "pad_id": self.pad_id,
        }


def load_tokenizer_manifest(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Прочитать манифест токенизатора (``tools/bpe_train.py``) или отказать."""
    manifest_path = Path(os.path.expanduser(os.fspath(path)))
    if not manifest_path.is_file():
        raise PretokenizeError(
            f"манифест токенизатора не найден: {manifest_path} — "
            "сначала tools/bpe_train.py (или укажите --tokenizer)"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PretokenizeError(f"манифест токенизатора не разбирается: {manifest_path}") from exc
    if manifest.get("version") != "axiom-pretrain-tokenizer/1":
        raise PretokenizeError(
            f"схема манифеста токенизатора {manifest.get('version')!r} не поддерживается"
        )
    return manifest


def load_tokenizer_tokenizer(pin: TokenizerPin) -> Any:
    """Загрузить сам BPE из файла-артефакта (в воркере — свой экземпляр)."""
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:  # pragma: no cover — окружение без библиотеки
        raise PretokenizeError(
            "нет библиотеки tokenizers: pip install tokenizers (venv-axiom)"
        ) from exc
    tokenizer = Tokenizer.from_file(str(pin.path))
    vocab = tokenizer.get_vocab()
    if vocab.get(pin.code_prefix) != pin.code_prefix_id:
        raise PretokenizeError(
            f"в токенизаторе нет кодового префикса {pin.code_prefix!r} на id "
            f"{pin.code_prefix_id} — корпус C не пометить"
        )
    return tokenizer


def resolve_tokenizer_pin(
    *,
    tokenizer_dir: str | os.PathLike[str] = DEFAULT_TOKENIZER_DIR,
    tokenizer_path: str | os.PathLike[str] | None = None,
    expect_hash: str | None = None,
) -> TokenizerPin:
    """Вывести пин токенизатора из артефакта и манифеста, сверив sha256.

    Хеш файла — то, чем запиннен прогон (AD-4).  Сверка обязательна: без неё
    претокенизация молча уехала бы по другому словарю, а id-потоки остались бы
    «валидными» — расхождение вылезло бы только на лоссе.
    """
    directory = Path(os.path.expanduser(os.fspath(tokenizer_dir)))
    path = Path(os.path.expanduser(os.fspath(tokenizer_path))) if tokenizer_path else directory / TOKENIZER_FILE
    if not path.is_file():
        raise PretokenizeError(f"файл токенизатора не найден: {path}")
    manifest_path = None
    manifest: dict[str, Any] = {}
    candidate = directory / TOKENIZER_MANIFEST_FILE
    if candidate.is_file():
        manifest_path = candidate
        manifest = load_tokenizer_manifest(candidate)
    actual = file_sha256(path)
    pinned = manifest.get("tokenizer_hash")
    if pinned and pinned != actual:
        raise PretokenizeError(
            f"tokenizer_hash {actual} != запинненного {pinned} — токенизатор изменён"
        )
    if expect_hash and actual != expect_hash:
        raise PretokenizeError(
            f"tokenizer_hash {actual} != ожидаемого {expect_hash} (--expect-tokenizer-hash)"
        )
    return TokenizerPin(
        path=path,
        tokenizer_hash=actual,
        vocab_size=int(manifest.get("vocab_size") or _vocab_size(path)),
        code_prefix=str(manifest.get("code_prefix") or CODE_PREFIX),
        code_prefix_id=int(manifest.get("code_prefix_id", CODE_ID)),
        eos_id=int(manifest.get("eos_id", EOS_ID)),
        bos_id=int(manifest.get("bos_id", BOS_ID)),
        pad_id=int(manifest.get("pad_id", PAD_ID)),
        manifest_path=manifest_path,
    )


def _vocab_size(path: Path) -> int:
    """Размер словаря прямо из файла-артефакта (страховка без манифеста)."""
    from tokenizers import Tokenizer

    return int(Tokenizer.from_file(str(path)).get_vocab_size())


# --------------------------------------------------------------------------- #
# Упаковка одного шарда
# --------------------------------------------------------------------------- #


@dataclass
class ShardTokenEntry:
    """Запись шарда в манифесте: счётчики, хеши и скорость по факту."""

    file: str
    source: str
    source_sha256: str
    limit_bytes: int = 0
    records: int = 0
    tokens: int = 0          # слоты записей = records * seq_len (то, что съест луп)
    stream_tokens: int = 0   # токены потока: документы + EOS + кодовые префиксы
    pad_tokens: int = 0
    eos_tokens: int = 0
    code_prefix_tokens: int = 0
    documents: int = 0
    documents_truncated: int = 0
    skipped_documents: int = 0
    chars: int = 0
    bytes: int = 0
    sha256: str = ""
    elapsed_s: float = 0.0
    tokens_per_sec: float = 0.0
    skipped: bool = False

    def as_json(self) -> dict[str, Any]:
        return asdict(self)


def encode_batch(tokenizer: Any, texts: Sequence[str]) -> list[list[int]]:
    """Пакетное кодирование (Rust-путь: поштучный ``encode`` на 20B токенов не тянет)."""
    return [encoding.ids for encoding in tokenizer.encode_batch(list(texts))]


def _stream_tokens(
    ids: Sequence[int],
    *,
    code_prefix_id: int | None,
    eos_id: int,
) -> np.ndarray:
    """Токены документа в потоке: ``[code_prefix] + ids + [eos]`` (uint32)."""
    head = 1 + (1 if code_prefix_id is not None else 0)
    out = np.empty(len(ids) + head, dtype=np.uint32)
    offset = 0
    if code_prefix_id is not None:
        out[0] = code_prefix_id
        offset = 1
    out[offset : offset + len(ids)] = np.asarray(ids, dtype=np.uint32)
    out[-1] = eos_id
    return out


#: Задача воркера (только простые типы — передаётся через pickle).
def tokenize_shard(job: dict[str, Any]) -> dict[str, Any]:
    """Претокенизировать один шард в ``.bin`` и вернуть запись манифеста.

    Воркер сам читает свой шард, свой токенизатор и пишет свой файл: границы
    работы — границы файла, поэтому общий кэш или блокировка не нужны.
    """
    seq_len = int(job["seq_len"])
    pin = TokenizerPin(
        path=Path(job["tokenizer_path"]),
        tokenizer_hash=str(job["tokenizer_hash"]),
        vocab_size=int(job["vocab_size"]),
        code_prefix=str(job["code_prefix"]),
        code_prefix_id=int(job["code_prefix_id"]),
        eos_id=int(job["eos_id"]),
        bos_id=int(job["bos_id"]),
        pad_id=int(job["pad_id"]),
    )
    tokenizer = load_tokenizer_tokenizer(pin)
    shard = CorpusShard(
        stream=str(job["stream"]),
        file=str(job["source"]),
        path=Path(job["source_path"]),
        sha256=str(job["source_sha256"]),
        approx_tokens=int(job["approx_tokens"]),
        bytes=int(job["bytes"]),
    )
    out_path = Path(job["out_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = out_path.with_suffix(out_path.suffix + ".part")
    limit_bytes = int(job.get("limit_bytes") or 0)
    max_doc_tokens = job.get("max_doc_tokens")
    text_keys = tuple(job.get("text_keys") or DEFAULT_TEXT_KEYS)
    batch_chars = int(job.get("batch_chars") or 2 * MB)
    batch_docs = int(job.get("batch_docs") or 1024)
    code_prefix_id = pin.code_prefix_id if str(job["stream"]).upper() == "C" else None

    entry = ShardTokenEntry(
        file=out_path.name,
        source=shard.file,
        source_sha256=shard.sha256,
        limit_bytes=limit_bytes,
    )
    started = time.time()
    digest = hashlib.sha256()
    buffer = np.empty(seq_len - 1, dtype=np.uint32)
    filled = 0
    rows: list[np.ndarray] = []
    pending_texts: list[str] = []
    pending_chars = 0

    def flush_records(handle: Any) -> None:
        """Сбросить накопленные записи одним куском (I/O по NFS — пачками)."""
        nonlocal rows
        if not rows:
            return
        block = np.stack(rows, axis=0)
        payload = block.tobytes(order="C")
        handle.write(payload)
        digest.update(payload)
        entry.records += block.shape[0]
        rows = []

    def push_stream(tokens: np.ndarray) -> None:
        """Докласть токены потока в буфер записи, отдавая полные записи наружу."""
        nonlocal filled
        start = 0
        total = len(tokens)
        while start < total:
            room = (seq_len - 1) - filled
            take = min(room, total - start)
            buffer[filled : filled + take] = tokens[start : start + take]
            filled += take
            start += take
            if filled == seq_len - 1:
                row = np.empty(seq_len, dtype=np.uint32)
                row[0] = pin.bos_id
                row[1:] = buffer
                rows.append(row)
                filled = 0
                if len(rows) >= 512:
                    flush_records(handle)

    with open(part_path, "wb") as handle:
        for text in iter_shard_texts(shard, text_keys=text_keys):
            if not text:
                entry.skipped_documents += 1
                continue
            pending_texts.append(text)
            pending_chars += len(text)
            if len(pending_texts) < batch_docs and pending_chars < batch_chars:
                continue
            for ids in encode_batch(tokenizer, pending_texts):
                if max_doc_tokens is not None and len(ids) > int(max_doc_tokens):
                    entry.documents_truncated += 1
                    ids = ids[: int(max_doc_tokens)]
                entry.documents += 1
                entry.eos_tokens += 1
                if code_prefix_id is not None:
                    entry.code_prefix_tokens += 1
                push_stream(_stream_tokens(ids, code_prefix_id=code_prefix_id, eos_id=pin.eos_id))
            entry.chars += pending_chars
            pending_texts = []
            pending_chars = 0
            if limit_bytes and entry.chars >= limit_bytes:
                break
        if pending_texts:
            for ids in encode_batch(tokenizer, pending_texts):
                if max_doc_tokens is not None and len(ids) > int(max_doc_tokens):
                    entry.documents_truncated += 1
                    ids = ids[: int(max_doc_tokens)]
                entry.documents += 1
                entry.eos_tokens += 1
                if code_prefix_id is not None:
                    entry.code_prefix_tokens += 1
                push_stream(_stream_tokens(ids, code_prefix_id=code_prefix_id, eos_id=pin.eos_id))
            entry.chars += pending_chars
        # Хвост: последняя запись добивается PAD (и начинается с BOS, как все).
        if filled:
            row = np.empty(seq_len, dtype=np.uint32)
            row[0] = pin.bos_id
            row[1 : 1 + filled] = buffer[:filled]
            row[1 + filled :] = pin.pad_id
            entry.pad_tokens += int(seq_len - 1 - filled)
            rows.append(row)
        flush_records(handle)
        entry.bytes = handle.tell()
    # Перенос .part → финальное имя: усечённый файл не считается готовым шардом.
    os.replace(part_path, out_path)
    entry.sha256 = digest.hexdigest()
    entry.tokens = entry.records * seq_len
    entry.stream_tokens = entry.tokens - entry.pad_tokens - entry.records  # минус BOS-слоты
    entry.elapsed_s = round(time.time() - started, 3)
    entry.tokens_per_sec = (
        round(entry.stream_tokens / entry.elapsed_s, 1) if entry.elapsed_s > 0 else 0.0
    )
    return entry.as_json()


# --------------------------------------------------------------------------- #
# Манифест прогона
# --------------------------------------------------------------------------- #


@dataclass
class TokensManifest:
    """``manifest-{w,c}.json`` каталога ``tokens/``: шарды + пины + скорость."""

    path: Path
    stream: str
    seq_len: int
    pin: TokenizerPin
    limit_mb: float | None = None
    workers: int = 1
    started: float = field(default_factory=time.time)
    shards: list[dict[str, Any]] = field(default_factory=list)

    def entry_for(self, source: str) -> dict[str, Any] | None:
        for entry in self.shards:
            if entry["source"] == source:
                return entry
        return None

    def add(self, entry: dict[str, Any]) -> None:
        self.shards = [item for item in self.shards if item["source"] != entry["source"]]
        self.shards.append(entry)
        self.shards.sort(key=lambda item: item["source"])
        self.save()

    def totals(self) -> dict[str, Any]:
        return {
            "shards": len(self.shards),
            "records": sum(int(entry["records"]) for entry in self.shards),
            "tokens": sum(int(entry["tokens"]) for entry in self.shards),
            "stream_tokens": sum(int(entry["stream_tokens"]) for entry in self.shards),
            "pad_tokens": sum(int(entry["pad_tokens"]) for entry in self.shards),
            "documents": sum(int(entry["documents"]) for entry in self.shards),
            "chars": sum(int(entry["chars"]) for entry in self.shards),
            "bytes": sum(int(entry["bytes"]) for entry in self.shards),
            "elapsed_s": round(sum(float(entry["elapsed_s"]) for entry in self.shards), 3),
        }

    def save(self) -> Path:
        totals = self.totals()
        elapsed = time.time() - self.started
        payload = {
            "version": TOKENS_SCHEMA,
            "kind": "packed-uint32",
            "shard": self.stream,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seq_len": self.seq_len,
            "dtype": DTYPE,
            "byte_order": BYTE_ORDER,
            "record_layout": RECORD_LAYOUT,
            "stream_layout": STREAM_LAYOUT,
            "bos_id": self.pin.bos_id,
            "eos_id": self.pin.eos_id,
            "pad_id": self.pin.pad_id,
            "code_prefix": self.pin.code_prefix,
            "code_prefix_id": self.pin.code_prefix_id,
            "tokenizer": self.pin.as_json(),
            "tokenizer_hash": self.pin.tokenizer_hash,
            "stream": self.stream,
            "streams": None,
            "limit_mb": self.limit_mb,
            "workers": self.workers,
            "shards": self.shards,
            "totals": totals,
            "throughput": {
                "tokens_per_sec": round(totals["stream_tokens"] / elapsed, 1) if elapsed > 0 else None,
                "records_per_sec": round(totals["records"] / elapsed, 1) if elapsed > 0 else None,
                "chars_per_sec": round(totals["chars"] / elapsed, 1) if elapsed > 0 else None,
                "elapsed_s": round(elapsed, 3),
            },
            "sources": [
                {
                    "file": entry["source"],
                    "sha256": entry["source_sha256"],
                    "bin": entry["file"],
                    "bin_sha256": entry["sha256"],
                }
                for entry in self.shards
            ],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        return self.path


def load_tokens_manifest(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Прочитать манифест претокенизированного шард-набора (читатель train_loop)."""
    manifest_path = Path(os.path.expanduser(os.fspath(path)))
    if not manifest_path.is_file():
        raise PretokenizeError(f"манифест tokens не найден: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data.get("version") != TOKENS_SCHEMA:
        raise PretokenizeError(
            f"схема манифеста tokens {data.get('version')!r} != {TOKENS_SCHEMA!r}"
        )
    return data


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def shard_job(
    shard: CorpusShard,
    *,
    args: argparse.Namespace,
    out_dir: Path,
    pin: TokenizerPin,
    limit_bytes: int = 0,
) -> dict[str, Any]:
    """Описание работы воркера по шарду (простые типы — через pickle)."""
    return {
        "stream": shard.stream,
        "source": shard.file,
        "source_path": str(shard.path),
        "source_sha256": shard.sha256,
        "approx_tokens": shard.approx_tokens,
        "bytes": shard.bytes,
        "out_path": str(out_dir / f"{Path(shard.file).name.split('.')[0]}.bin"),
        "seq_len": args.seq_len,
        "limit_bytes": int(limit_bytes),
        "max_doc_tokens": args.max_doc_tokens,
        "text_keys": list(args.text_keys),
        "batch_chars": int(args.batch_mb * MB),
        "batch_docs": args.batch_docs,
        "tokenizer_path": str(pin.path),
        "tokenizer_hash": pin.tokenizer_hash,
        "vocab_size": pin.vocab_size,
        "code_prefix": pin.code_prefix,
        "code_prefix_id": pin.code_prefix_id,
        "eos_id": pin.eos_id,
        "bos_id": pin.bos_id,
        "pad_id": pin.pad_id,
    }


def _run_shard_job(payload: tuple[int, dict[str, Any]]) -> tuple[int, dict[str, Any]]:
    """Обёртка воркера: результат вместе с номером задачи.

    Прогон трёхчасовой, поэтому результаты принимаются по мере готовности
    (``imap_unordered``), а не в порядке очереди: с ``imap`` манифест ждал бы
    самого медленного шарда раунда, и обрыв прогона терял бы уже готовые
    ``.bin`` — на resume они пересчитывались бы заново.
    """
    index, job = payload
    return index, tokenize_shard(job)


def shard_is_done(entry: dict[str, Any] | None, job: dict[str, Any]) -> bool:
    """Готов ли шард: запись манифеста есть, файл на месте и хеши совпали.

    Проверка по хешу, а не по факту существования файла: оборванный прогон
    оставляет ``.bin`` (пишется через ``.part``, но перезапуск мог упасть между
    rename и записью манифеста), а неверный шард в манифесте — это тихо
    испорченный корпус.
    """
    if not entry or not entry.get("sha256"):
        return False
    path = Path(job["out_path"])
    if not path.is_file():
        return False
    if entry.get("source_sha256") != job["source_sha256"]:
        return False
    if int(entry.get("limit_bytes", 0)) != int(job.get("limit_bytes", 0)):
        return False  # проба с другим пределом — другой шард, не «уже готов»
    if entry.get("sha256") != file_sha256(path):
        return False
    return True


def cmd_pretokenize(args: argparse.Namespace) -> int:
    echo = lambda message: print(message, flush=True)  # noqa: E731
    started = time.time()
    out_root = common.ensure_output_allowed(args.out, allow_any=args.allow_any_out)
    streams = [stream.upper() for stream in args.streams]
    pin = resolve_tokenizer_pin(
        tokenizer_dir=args.tokenizer_dir,
        tokenizer_path=args.tokenizer,
        expect_hash=args.expect_tokenizer_hash,
    )
    echo(
        f"[tokens] токенизатор {pin.path} vocab {pin.vocab_size} "
        f"hash {pin.tokenizer_hash[:16]}… (code_prefix id {pin.code_prefix_id})"
    )
    shards = load_corpus_shards(args.shard_root, streams)
    manifests: dict[str, TokensManifest] = {}
    jobs: list[tuple[CorpusShard, dict[str, Any], TokensManifest]] = []
    #: Бюджет пробы (``--limit-mb``) отсчитывается на поток по оценке объёма
    #: шардов (``approx_tokens * 4``, ADR-021) и раздаётся шардам по порядку: так
    #: проба меряет скорость на настоящих шардах обоих потоков, а не на одном.
    #: Бюджет целочисленный: вычитание оценки из float оставляло остаток 0,76 Б,
    #: бюджет «не кончался», и проба продолжала читать поток без ограничения
    #: (``limit_bytes=0`` у воркера означает «без предела»).
    budget = {stream: (int(args.limit_mb * MB) if args.limit_mb else None) for stream in streams}
    for shard in shards:
        manifest = manifests.get(shard.stream)
        if manifest is None:
            manifest = TokensManifest(
                path=out_root / shard.stream / f"manifest-{shard.stream.lower()}.json",
                stream=shard.stream,
                seq_len=args.seq_len,
                pin=pin,
                limit_mb=args.limit_mb,
                workers=args.workers,
            )
            if manifest.path.is_file():  # resume: подхватить прошлые шарды
                previous = load_tokens_manifest(manifest.path)
                manifest.shards = list(previous.get("shards") or [])
                manifest.started = started
            manifests[shard.stream] = manifest
        limit_bytes = 0  # 0 у воркера = «без предела» (боевой прогон, не проба)
        if budget[shard.stream] is not None:
            if budget[shard.stream] <= 0:  # бюджет потока исчерпан — остальные шарды не берём
                continue
            limit_bytes = min(int(budget[shard.stream]), max(1, shard.text_bytes_estimate))
            budget[shard.stream] -= limit_bytes
        job = shard_job(
            shard, args=args, out_dir=out_root / shard.stream, pin=pin, limit_bytes=limit_bytes
        )
        if not args.restart and shard_is_done(manifest.entry_for(shard.file), job):
            echo(f"[tokens] {shard.stream}/{shard.file}: уже готов (пропуск)")
            continue
        jobs.append((shard, job, manifest))
    if not jobs:
        echo("[tokens] нечего делать: все шарды готовы")
        for manifest in manifests.values():
            manifest.save()
        return 0

    echo(f"[tokens] шардов к обработке: {len(jobs)} (воркеров {args.workers})")
    done = 0
    pending = {index: (shard, manifest) for index, (shard, _job, manifest) in enumerate(jobs)}
    try:
        with Pool(processes=max(1, args.workers)) as pool:
            for index, result in pool.imap_unordered(
                _run_shard_job, list(enumerate(job for _, job, _ in jobs)), chunksize=1
            ):
                done += 1
                shard, manifest = pending.pop(index)
                manifest.add(result)  # манифест пишется по мере готовности шардов
                totals = manifest.totals()
                echo(
                    f"[tokens] {shard.stream}/{result['file']}: {result['records']} записей, "
                    f"{result['stream_tokens'] / 1e6:.1f}M токенов, {result['documents']} документов, "
                    f"{result['tokens_per_sec'] / 1e6:.2f}M ток/с, {result['elapsed_s']:.1f} с "
                    f"(итого {totals['stream_tokens'] / 1e9:.2f}B, {done}/{len(jobs)})"
                )
    except Exception as exc:
        # Отказ воркера обязан быть назван вместе с числом готовых шардов: сам по
        # себе traceback не говорит, какой ``.bin`` остался недописанным (он лежит
        # как ``.part`` и не считается готовым — перезапуск продолжит с него).
        raise PretokenizeError(
            f"воркер упал на шарде (готово {done} из {len(jobs)}): {exc!r} — "
            "перезапуск продолжит с неготовых шардов"
        ) from exc
    for manifest in manifests.values():
        path = manifest.save()
        totals = manifest.totals()
        throughput = json.loads(path.read_text(encoding="utf-8"))["throughput"]
        echo(
            f"[tokens] {manifest.stream}: {totals['shards']} шардов, "
            f"{totals['stream_tokens'] / 1e9:.3f}B токенов, {totals['bytes'] / 1e9:.1f} ГБ, "
            f"ток/с {throughput['tokens_per_sec'] / 1e6:.2f}M, манифест {path}"
        )
    echo(f"[tokens] готово за {time.time() - started:.1f} с")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Претокенизация W/C в упакованные .bin (uint32, T=8192)"
    )
    parser.add_argument("--shard-root", default=DEFAULT_DATASET_ROOT,
                        help="корень датасета с {W,C}/manifest-*.json")
    parser.add_argument("--streams", nargs="+", default=["W", "C"], help="потоки корпуса")
    parser.add_argument("--out", default=DEFAULT_TOKENS_DIR, help="каталог tokens/{W,C}/*.bin")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN, help="длина записи T")
    parser.add_argument("--tokenizer-dir", default=DEFAULT_TOKENIZER_DIR,
                        help="каталог tokenizer.model + tokenizer-manifest.json")
    parser.add_argument("--tokenizer", default=None, help="явный путь файла токенизатора")
    parser.add_argument("--expect-tokenizer-hash", default=None,
                        help="ожидаемый sha256 токенизатора (если файла манифеста нет)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="процессов параллельно")
    parser.add_argument("--limit-mb", type=float, default=None,
                        help="предел текста N МиБ на каждый поток (проба скорости)")
    parser.add_argument("--max-doc-tokens", type=int, default=None,
                        help="усечь документ длиннее N токенов (по умолчанию без усечения)")
    parser.add_argument("--batch-mb", type=float, default=2.0, help="текста на батч кодирования, МиБ")
    parser.add_argument("--batch-docs", type=int, default=1024, help="документов на батч кодирования")
    parser.add_argument("--text-keys", nargs="+", default=list(DEFAULT_TEXT_KEYS),
                        help="поля текста записи (text/content)")
    parser.add_argument("--restart", action="store_true", help="перезаписать готовые шарды")
    parser.add_argument("--allow-any-out", action="store_true",
                        help="разрешить вывод вне gb10-shared//tmp (C-032/C-033)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return cmd_pretokenize(args)
    except (PretokenizeError, TokenizerError, CorpusError) as exc:
        print(f"[tokens] ОТКАЗ: {exc}", file=sys.stderr, flush=True)
        return 2
    except ValueError as exc:  # ensure_output_allowed
        print(f"[tokens] ОТКАЗ вывода: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
