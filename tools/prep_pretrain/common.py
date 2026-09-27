"""Общие примитивы шардирования претрейн-датасета L3 (ADR-021).

Модуль держит всё, что одинаково для шардов W (веб) и C (код):

* ``ShardWriter`` — запись jsonl-шардов с ротацией по размеру сжатого файла,
  потоковый sha256 каждого финального шарда, запись через ``.part`` + rename;
* ``Manifest`` — список готовых шардов ``{file, bytes, sha256, approx_tokens,
  records}`` и курсор resume по числу прочитанных записей источника;
* ``BoundedHashSet`` — точный документный дедуп с ограничением памяти (LRU);
* ``approx_tokens`` — счёт по ADR-021: ``max(1, len(text) // 4)``;
* источники записи: локальный jsonl (тесты, офлайн-фикстуры) и потоковый
  HuggingFace ``datasets`` (боевой режим, без материализации датасета).

Данные пишутся только в ``~/gb10-shared/datasets/axiom-pretrain-l3/`` и во
временные каталоги (C-032/C-033); ``ensure_output_allowed`` это стережёт.
"""

from __future__ import annotations

import glob as glob_mod
import gzip
import hashlib
import json
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

PIPELINE_VERSION = "axiom-pretrain-l3/1"

#: Канонический корень датасета (симлинки в репо, сами данные — на gb10-shared).
DATASET_ROOT = "~/gb10-shared/datasets/axiom-pretrain-l3"

#: Куда разрешено писать выход пайплайна. Всё остальное CLI отклоняет.
ALLOWED_OUTPUT_ROOTS = ("~/gb10-shared", "/tmp", "/var/tmp")

#: Прокси-схемы, которые httpx (huggingface_hub >= 1.x) не разбирает без socksio.
SOCKS_PROXY_SCHEMES = ("socks://", "socks4://", "socks5://")

DEFAULT_CODEC = "zstd"
DEFAULT_SHARD_BYTES = 500 * 1024 * 1024
DEFAULT_DEDUP_WINDOW = 5_000_000
DEFAULT_MANIFEST_EVERY = 20_000


# --------------------------------------------------------------------------- #
# Окружение
# --------------------------------------------------------------------------- #


def sanitize_proxy_env(env: dict[str, str] | None = None) -> list[str]:
    """Убрать ``ALL_PROXY``/``all_proxy`` с socks-схемой; вернуть список правок.

    ``httpx`` (на нём стоит huggingface_hub 1.x) не умеет socks без пакета
    ``socksio`` и падает ещё до запроса: ``Unknown scheme for proxy URL
    URL('socks://127.0.0.1:7890/')``. Переменные с http(s)-схемой не трогаем:
    они рабочие и нужны там, где прямой доступ закрыт.
    """
    target = os.environ if env is None else env
    changed: list[str] = []
    for name in ("ALL_PROXY", "all_proxy"):
        value = target.get(name)
        if value and value.lower().startswith(SOCKS_PROXY_SCHEMES):
            target.pop(name, None)
            changed.append(name)
    return changed


def ensure_output_allowed(path: str | os.PathLike[str], allow_any: bool = False) -> Path:
    """Разрешить вывод только внутри ``ALLOWED_OUTPUT_ROOTS`` (иначе ValueError).

    Запрет — часть контракта аренды: сырьё и крупные данные не расползаются по
    рабочим каталогам (C-032/C-033).
    """
    resolved = Path(os.path.expanduser(os.fspath(path))).resolve()
    if allow_any:
        return resolved
    for root in ALLOWED_OUTPUT_ROOTS:
        root_path = Path(os.path.expanduser(root)).resolve()
        if resolved == root_path or root_path in resolved.parents:
            return resolved
    raise ValueError(
        f"вывод вне разрешённых корней {ALLOWED_OUTPUT_ROOTS}: {resolved} "
        f"(обход — только явным --allow-any-out)"
    )


# --------------------------------------------------------------------------- #
# Счёт токенов и дедуп
# --------------------------------------------------------------------------- #


def approx_tokens(text: str) -> int:
    """Оценка числа токенов по ADR-021: ``max(1, len(text) // 4)`` (символы/4)."""
    return max(1, len(text) // 4)


def text_hash(text: str) -> str:
    """sha256 текста документа (нормализация: UTF-8, без обрезки пробелов)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class BoundedHashSet:
    """Точный дедуп с ограничением памяти: LRU последних ``capacity`` хешей.

    Источники (FineWeb-Edu, The Stack) уже дедуплицированы — этот фильтр
    страховочный, поэтому память важнее полноты истории: вытесненный хеш
    забывается и документ может пройти повторно (это записано в отчёте числом
    ``dedup_window``).
    """

    __slots__ = ("_seen", "_capacity", "hits")

    def __init__(self, capacity: int = DEFAULT_DEDUP_WINDOW) -> None:
        if capacity <= 0:
            raise ValueError("capacity должен быть > 0")
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._capacity = capacity
        self.hits = 0

    @property
    def size(self) -> int:
        return len(self._seen)

    @property
    def capacity(self) -> int:
        return self._capacity

    def duplicate(self, digest: str) -> bool:
        """True — если хеш уже встречался; иначе запомнить его (с вытеснением)."""
        if digest in self._seen:
            self._seen.move_to_end(digest)
            self.hits += 1
            return True
        self._seen[digest] = None
        if len(self._seen) > self._capacity:
            self._seen.popitem(last=False)
        return False


# --------------------------------------------------------------------------- #
# Запись шардов
# --------------------------------------------------------------------------- #


class HashingWriter:
    """Файловая обёртка: считает sha256 и байты ровно того, что легло в файл.

    Хеш считается по байтам на диске, поэтому он же — sha256 финального
    шард-файла (проверяется ``verify_manifest`` и тестом T-p1).
    """

    __slots__ = ("_raw", "_hash", "_bytes")

    def __init__(self, raw: Any) -> None:
        self._raw = raw
        self._hash = hashlib.sha256()
        self._bytes = 0

    def write(self, data: bytes) -> int:
        written = self._raw.write(data)
        self._hash.update(data)
        self._bytes += len(data)
        return written

    def flush(self) -> None:
        self._raw.flush()

    def tell(self) -> int:
        return self._bytes

    @property
    def sha256(self) -> str:
        return self._hash.hexdigest()

    @property
    def bytes(self) -> int:
        return self._bytes


CODEC_EXTENSIONS = {"zstd": ".jsonl.zst", "gzip": ".jsonl.gz", "none": ".jsonl"}


def _open_compressor(sink: HashingWriter, codec: str, level: int) -> Any:
    if codec == "zstd":
        import zstandard  # локальный импорт: тесты на gzip не требуют пакета

        return zstandard.ZstdCompressor(level=level).stream_writer(sink, closefd=False)
    if codec == "gzip":
        return gzip.GzipFile(fileobj=sink, mode="wb", compresslevel=level)
    if codec == "none":
        return sink
    raise ValueError(f"неизвестный кодек: {codec}")


class ShardWriter:
    """Пишет jsonl-шарды, ротируя файл по размеру сжатого вывода.

    Файл открывается как ``<prefix>-<NNNNN>.jsonl.zst.part`` и переименовывается
    в финальное имя только после успешного закрытия: прерванный прогон не
    оставляет в манифесте «полу-шард», а хвост ``.part`` на resume удаляется.
    """

    def __init__(
        self,
        out_dir: str | os.PathLike[str],
        prefix: str,
        shard_bytes: int = DEFAULT_SHARD_BYTES,
        codec: str = DEFAULT_CODEC,
        level: int = 3,
        start_index: int = 0,
        existing_shards: Sequence[dict] | None = None,
        flush_every: int | None = None,
        on_shard: Callable[[dict], None] | None = None,
    ) -> None:
        if codec not in CODEC_EXTENSIONS:
            raise ValueError(f"неизвестный кодек: {codec}")
        self.out_dir = Path(os.path.expanduser(os.fspath(out_dir)))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.shard_bytes = shard_bytes
        self.codec = codec
        self.level = level
        self.on_shard = on_shard
        self.index = start_index
        self.shards: list[dict] = list(existing_shards or [])
        # zstd копит вывод во внутреннем буфере, поэтому «сколько байт уже в
        # файле» видно только после сброса блока. Сбрасываем каждые
        # flush_every байт несжатого jsonl — иначе ротация по размеру
        # срабатывает с задержкой в буфер целиком.
        self.flush_every = flush_every or max(1 << 20, shard_bytes // 8)
        self._raw: Any = None
        self._hashing: HashingWriter | None = None
        self._sink: Any = None
        self._part: Path | None = None
        self._records = 0
        self._tokens = 0
        self._uncompressed = 0
        self._uncompressed_at_flush = 0

    # -- состояние текущего шарда ------------------------------------------- #

    @property
    def open(self) -> bool:
        return self._raw is not None

    @property
    def current_bytes(self) -> int:
        return self._hashing.bytes if self._hashing is not None else 0

    @property
    def current_records(self) -> int:
        return self._records

    @property
    def current_tokens(self) -> int:
        return self._tokens

    @property
    def total_bytes(self) -> int:
        """Байты всех закрытых шардов плюс текущего открытого."""
        return sum(s["bytes"] for s in self.shards) + self.current_bytes

    @property
    def total_tokens(self) -> int:
        return sum(s["approx_tokens"] for s in self.shards) + self._tokens

    def _shard_name(self, index: int) -> str:
        return f"{self.prefix}-{index:05d}{CODEC_EXTENSIONS[self.codec]}"

    def _open(self) -> None:
        name = self._shard_name(self.index)
        self._part = self.out_dir / f"{name}.part"
        self._raw = self._part.open("wb")
        self._hashing = HashingWriter(self._raw)
        self._sink = _open_compressor(self._hashing, self.codec, self.level)
        self._records = 0
        self._tokens = 0
        self._uncompressed = 0
        self._uncompressed_at_flush = 0

    def add(self, record: dict, tokens: int) -> dict | None:
        """Записать документ; вернуть закрытый шард, если сработала ротация."""
        if self._raw is None:
            self._open()
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        self._sink.write(line.encode("utf-8"))
        self._records += 1
        self._tokens += tokens
        self._uncompressed += len(line.encode("utf-8"))
        if (
            self.current_bytes < self.shard_bytes
            and self._uncompressed - self._uncompressed_at_flush >= self.flush_every
        ):
            self._sink.flush()
            self._uncompressed_at_flush = self._uncompressed
        if self.current_bytes >= self.shard_bytes:
            return self._close_shard()
        return None

    def _close_shard(self) -> dict | None:
        if self._raw is None:
            return None
        self._sink.flush()
        if self.codec == "zstd":
            self._sink.close()
        elif self.codec == "gzip":
            self._sink.close()
        self._raw.flush()
        self._raw.close()
        hashing = self._hashing
        part = self._part
        assert hashing is not None and part is not None
        final = part.with_suffix("")
        os.replace(part, final)
        entry = {
            "file": final.name,
            "bytes": hashing.bytes,
            "sha256": hashing.sha256,
            "approx_tokens": self._tokens,
            "records": self._records,
        }
        self.shards.append(entry)
        self._raw = self._hashing = self._sink = self._part = None
        self._records = self._tokens = 0
        self.index += 1
        if self.on_shard is not None:
            self.on_shard(entry)
        return entry

    def close(self) -> dict | None:
        """Закрыть текущий (неполный) шард; пустой шард не создаётся."""
        if self._raw is None:
            return None
        if self._records == 0:
            self._sink = None
            self._raw.close()
            if self._part is not None:
                self._part.unlink(missing_ok=True)
            self._raw = self._hashing = self._part = None
            return None
        return self._close_shard()

    def drop_partial(self) -> None:
        """Убрать незавершённый ``.part`` (вызывается на старте resume)."""
        if self._part is not None:
            return
        for stale in self.out_dir.glob(f"{self.prefix}-*.part"):
            stale.unlink(missing_ok=True)


def verify_manifest(manifest: dict, out_dir: str | os.PathLike[str]) -> dict:
    """Пересчитать sha256/размер шардов из манифеста по файлам на диске."""
    root = Path(os.path.expanduser(os.fspath(out_dir)))
    ok, bad = 0, []
    for entry in manifest.get("shards", []):
        path = root / entry["file"]
        if not path.exists():
            bad.append({"file": entry["file"], "problem": "missing"})
            continue
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        if digest.hexdigest() != entry.get("sha256") or size != entry.get("bytes"):
            bad.append(
                {
                    "file": entry["file"],
                    "problem": "mismatch",
                    "expected_sha256": entry.get("sha256"),
                    "actual_sha256": digest.hexdigest(),
                    "expected_bytes": entry.get("bytes"),
                    "actual_bytes": size,
                }
            )
            continue
        ok += 1
    return {"checked": len(manifest.get("shards", [])), "ok": ok, "bad": bad}


# --------------------------------------------------------------------------- #
# Манифест
# --------------------------------------------------------------------------- #


class Manifest:
    """``manifest-w.json`` / ``manifest-c.json``: шарды + курсор resume.

    Курсор — число записей источника, уже прочитанных в завершённых шардах: на
    resume источник проматывается на ``source_records`` и обработка продолжается
    с того же места (см. ``iter_source``).
    """

    def __init__(self, path: str | os.PathLike[str], shard: str, **defaults: Any) -> None:
        self.path = Path(os.path.expanduser(os.fspath(path)))
        self.data: dict[str, Any] = {
            "version": PIPELINE_VERSION,
            "shard": shard,
            "shards": [],
            "source_records": 0,
            "totals": {"shards": 0, "bytes": 0, "approx_tokens": 0, "records": 0},
        }
        self.data.update(defaults)
        self.saved_at = 0

    @classmethod
    def load(cls, path: str | os.PathLike[str], shard: str, **defaults: Any) -> "Manifest":
        manifest = cls(path, shard, **defaults)
        if manifest.path.exists():
            loaded = json.loads(manifest.path.read_text(encoding="utf-8"))
            manifest.data.update(loaded)
        return manifest

    @property
    def shards(self) -> list[dict]:
        return self.data["shards"]

    @property
    def source_records(self) -> int:
        return int(self.data.get("source_records", 0))

    @property
    def totals(self) -> dict:
        totals = {
            "shards": len(self.shards),
            "bytes": sum(s["bytes"] for s in self.shards),
            "approx_tokens": sum(s["approx_tokens"] for s in self.shards),
            "records": sum(s["records"] for s in self.shards),
        }
        self.data["totals"] = totals
        return totals

    def add_shard(self, entry: dict, source_records: int) -> None:
        self.shards.append(entry)
        self.data["source_records"] = source_records
        self.totals
        self.save()

    def set_cursor(self, source_records: int, counters: dict | None = None) -> None:
        self.data["source_records"] = source_records
        if counters:
            self.data["counters"] = dict(counters)
        self.totals

    def save(self, force: bool = True) -> None:
        now = time.time()
        if not force and now - self.saved_at < 5:
            return
        self.saved_at = now
        self.data["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)


# --------------------------------------------------------------------------- #
# Источники записей
# --------------------------------------------------------------------------- #


def iter_local_jsonl(glob: Sequence[str] | str, **_: Any) -> Iterator[dict]:
    """Поток записей из локальных jsonl-файлов (фикстуры, офлайн-источники).

    Файлы читаются построчно, в память поднимается одна строка; битые строки
    пропускаются с записью в ``stats`` вызывающего кода не делается — источник
    доверенный (фикстуры теста).
    """
    paths: list[str] = []
    for pattern in [glob] if isinstance(glob, str) else list(glob):
        expanded = os.path.expanduser(pattern)
        matched = sorted(glob_mod.glob(expanded, recursive=True))
        if not matched and Path(expanded).is_file():
            matched = [expanded]
        paths.extend(matched)
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def iter_hf_stream(
    repo: str,
    config: str | None = None,
    split: str = "train",
    **_: Any,
) -> Iterator[dict]:
    """Поток записей из датасета на HuggingFace (``datasets`` streaming).

    Датасет не материализуется: ``load_dataset(..., streaming=True)`` отдаёт
    итератор, под капотом — загрузка parquet-файлов по мере чтения.
    """
    sanitize_proxy_env()
    from datasets import load_dataset  # импорт ленивый: офлайн-путь его не требует

    dataset = load_dataset(repo, config, split=split, streaming=True)
    return iter(dataset)


def iter_source(
    source: Callable[..., Iterator[dict]],
    skip_records: int = 0,
    **kwargs: Any,
) -> Iterator[dict]:
    """Итератор источника с проматыванием ``skip_records`` (resume).

    Проматывание — линейное чтение, а не случайный доступ: у ``datasets`` в
    streaming-режиме другого способа встать на позицию нет.
    """
    records = source(**kwargs)
    for position, record in enumerate(records):
        if position < skip_records:
            continue
        yield record


#: Реестр итераторов источника: имя → (функция, имена аргументов спецификации).
SOURCE_ITERATORS: dict[str, tuple[Callable[..., Iterator[dict]], tuple[str, ...]]] = {
    "hf": (iter_hf_stream, ("repo", "config", "split")),
    "local": (iter_local_jsonl, ("glob",)),
}


def parse_source_spec(spec: str) -> dict:
    """``hf:<repo>[:<config>]`` или ``local:<glob>`` → спецификация источника.

    Спецификация JSON-сериализуема и попадает в манифест и отчёт: по отчёту
    видно, что именно читалось, а не что предполагалось прочитать.
    """
    if spec.startswith("hf:"):
        parts = spec[3:].split(":")
        if len(parts) not in (1, 2) or not parts[0]:
            raise ValueError(f"ожидалось hf:<repo>[:<config>], получено {spec!r}")
        return {
            "kind": "hf",
            "repo": parts[0],
            "config": parts[1] if len(parts) == 2 and parts[1] else None,
            "split": "train",
        }
    if spec.startswith("local:"):
        glob = spec[6:]
        if not glob:
            raise ValueError(f"пустой glob в {spec!r}")
        return {"kind": "local", "glob": glob}
    raise ValueError(f"неизвестная схема источника: {spec!r} (ждали hf:… или local:…)")


def source_iterator(
    spec: dict, iterators: dict | None = None
) -> tuple[Callable[..., Iterator[dict]], dict]:
    """Итератор и его именованные аргументы по спецификации источника."""
    registry = iterators or SOURCE_ITERATORS
    kind = spec.get("kind")
    if kind not in registry:
        raise ValueError(f"неизвестный вид источника: {kind!r}")
    func, names = registry[kind]
    return func, {name: spec[name] for name in names if name in spec}


def hm_get(record: dict, *names: str, default: Any = None) -> Any:
    """Достать первое непустое поле из ``record`` или его ``meta`` словаря."""
    for name in names:
        if name in record and record[name] not in (None, ""):
            return record[name]
    meta = record.get("meta")
    if isinstance(meta, dict):
        for name in names:
            if name in meta and meta[name] not in (None, ""):
                return meta[name]
    return default


# --------------------------------------------------------------------------- #
# Общий прогон шарда
# --------------------------------------------------------------------------- #


def run_shard(
    *,
    shard: str,
    out_dir: str | os.PathLike[str],
    target_tokens: int,
    source_spec: dict,
    normalize: Callable[[dict], tuple[str, dict] | None],
    manifest_path: str | os.PathLike[str],
    report_path: str | os.PathLike[str] | None = None,
    source_iterators: dict | None = None,
    shard_bytes: int = DEFAULT_SHARD_BYTES,
    codec: str = DEFAULT_CODEC,
    level: int = 3,
    flush_every: int | None = None,
    dedup_window: int = DEFAULT_DEDUP_WINDOW,
    skip_records: int = 0,
    restart: bool = False,
    max_records: int | None = None,
    max_output_bytes: int | None = None,
    manifest_every: int = DEFAULT_MANIFEST_EVERY,
    manifest_extra: dict | None = None,
    progress: bool = False,
    allow_any_out: bool = False,
) -> dict:
    """Один шард датасета: источник → нормализация → дедуп → jsonl-шарды.

    Возвращает числовой отчёт и пишет его в ``report_path``. Ни содержимого
    документов, ни их заголовков в отчёте нет — только счётчики, размеры, хеши
    и скорости.

    Остановка прогона: достигнут ``target_tokens`` (по записанным записям),
    ``max_records`` или ``max_output_bytes``; отдельный исход — источник
    кончился раньше цели (``source_exhausted=true`` в отчёте).
    """
    started = time.time()
    out_dir = ensure_output_allowed(out_dir, allow_any=allow_any_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = ensure_output_allowed(manifest_path, allow_any=allow_any_out)

    manifest = Manifest.load(
        manifest_path,
        shard,
        source=source_spec,
        target_tokens=target_tokens,
        codec=codec,
        dedup_window=dedup_window,
        shard_bytes=shard_bytes,
    )
    if restart:
        # --restart: курсор и список шардов обнуляются, старые файлы перезапишутся.
        manifest = Manifest(
            manifest_path,
            shard,
            source=source_spec,
            target_tokens=target_tokens,
            codec=codec,
            dedup_window=dedup_window,
            shard_bytes=shard_bytes,
        )
    if manifest_extra:
        manifest.data.update(manifest_extra)

    resume_from = skip_records if skip_records else manifest.source_records
    writer = ShardWriter(
        out_dir,
        prefix=shard,
        shard_bytes=shard_bytes,
        codec=codec,
        level=level,
        start_index=len(manifest.shards),
        existing_shards=manifest.shards,
        flush_every=flush_every,
        on_shard=lambda entry: None,
    )
    writer.drop_partial()

    dedup = BoundedHashSet(dedup_window)
    counters: dict[str, int] = {
        "records_read": 0,
        "records_kept": 0,
        "dropped_dedup": 0,
        "approx_tokens": 0,
        "source_text_chars": 0,
    }

    def counters_snapshot() -> dict:
        """Счётчики плюс причины отбраковки, накопленные нормализатором."""
        merged: dict[str, Any] = dict(counters)
        stats = getattr(normalize, "stats", None)
        if stats:
            merged["dropped_by_rule"] = dict(stats)
        return merged

    func, kwargs = source_iterator(source_spec, source_iterators)
    source_records = resume_from
    stop_reason = "source_exhausted"
    try:
        for record in iter_source(func, skip_records=resume_from, **kwargs):
            if max_records is not None and counters["records_read"] >= max_records:
                stop_reason = "max_records"
                break
            if max_output_bytes is not None and writer.total_bytes >= max_output_bytes:
                stop_reason = "max_output_bytes"
                break
            if target_tokens and writer.total_tokens >= target_tokens:
                stop_reason = "target_tokens"
                break
            counters["records_read"] += 1
            # source_records растёт только после успешной нормализации: запись,
            # на которой прогон оборвался, не считается прочитанной и будет
            # перечитана на resume (иначе она теряется — off-by-one).
            normalized = normalize(record)
            source_records += 1
            if normalized is None:
                continue
            text, meta = normalized
            counters["source_text_chars"] += len(text)
            if dedup.duplicate(text_hash(text)):
                counters["dropped_dedup"] += 1
                continue
            tokens = approx_tokens(text)
            entry = writer.add({"text": text, "meta": meta}, tokens)
            counters["records_kept"] += 1
            counters["approx_tokens"] += tokens
            if entry is not None:
                manifest.add_shard(entry, source_records)
            elif counters["records_read"] % manifest_every == 0:
                manifest.set_cursor(source_records, counters)
                manifest.save(force=False)
            if progress and counters["records_read"] % 10_000 == 0:
                print(
                    f"  [{shard}] записей {counters['records_read']}, "
                    f"в шард {counters['records_kept']}, "
                    f"{human_bytes(writer.total_bytes)}, "
                    f"{counters['approx_tokens'] / 1e9:.3f}B токенов",
                    flush=True,
                )
    finally:
        final = writer.close()
        if final is not None:
            manifest.add_shard(final, source_records)
        manifest.set_cursor(source_records, counters)
        manifest.data["stop_reason"] = stop_reason
        manifest.data["counters"] = counters_snapshot()
        manifest.totals
        manifest.save()

    seconds = elapsed(started)
    totals = manifest.totals
    text_chars = counters["source_text_chars"]
    char_rate = rate_per_second(text_chars, seconds)
    report = {
        "pipeline": PIPELINE_VERSION,
        "shard": shard,
        "source": source_spec,
        "out_dir": str(out_dir),
        "manifest": str(manifest_path),
        "codec": codec,
        "shard_bytes_limit": shard_bytes,
        "flush_every": flush_every or max(1 << 20, shard_bytes // 8),
        "dedup_window": dedup_window,
        "target_tokens": target_tokens,
        "stop_reason": stop_reason,
        "source_records_consumed": source_records,
        "source_records_skipped_on_resume": resume_from,
        "counters": counters_snapshot(),
        "shards": manifest.shards,
        "totals": totals,
        "seconds": round(seconds, 3),
        "output_mb_per_s": rate_per_second(totals["bytes"] / (1024 * 1024), seconds),
        "records_per_s": rate_per_second(counters["records_read"], seconds),
        "tokens_per_s": rate_per_second(totals["approx_tokens"], seconds),
        "source_text_mb": round(text_chars / (1024 * 1024), 3),
        "source_text_mb_per_s": rate_per_second(text_chars / (1024 * 1024), seconds),
        # Оценка длительности полной загрузки по темпу чтения источника:
        # по ADR-021 токенов ≈ символов/4, значит на target_tokens нужно 4× символов.
        "eta_hours_to_target": (
            round((target_tokens * 4) / char_rate / 3600, 3) if char_rate > 0 else None
        ),
        "max_rss_mb": max_rss_mb(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if report_path is not None:
        report["report"] = str(write_report(report_path, report))
    return report


# --------------------------------------------------------------------------- #
# Мелкие утилиты
# --------------------------------------------------------------------------- #


def elapsed(started: float) -> float:
    return max(0.0, time.time() - started)


def rate_per_second(count: float, seconds: float) -> float:
    return round(count / seconds, 3) if seconds > 0 else 0.0


def json_line_count(path: str | os.PathLike[str]) -> int:
    """Число строк в файле (для проверок; сам файл в память не поднимается)."""
    total = 0
    with open(os.path.expanduser(os.fspath(path)), "rb") as handle:
        for _ in handle:
            total += 1
    return total


def write_report(path: str | os.PathLike[str], payload: dict) -> Path:
    target = Path(os.path.expanduser(os.fspath(path)))
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, target)
    return target


def human_bytes(value: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def max_rss_mb() -> float:
    """Пиковый RSS процесса в МиБ — свидетельство потоковости прогона."""
    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(rss / 1024.0, 1)  # Linux отдаёт КиБ
