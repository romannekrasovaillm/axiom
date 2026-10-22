"""Ступень 4 пайплайна axiom-domain-ds-v1: точные дубли, near-dup (MinHash) и контейнмент.

* **Точный дубль** — совпал sha256 нормализованного текста эпизода
  (регистр и пробелы не различаются, метаданные и id — не участвуют).
* **Near-dup** — 5-граммные шинглы → MinHash на 64 подписи → бандинг 16×4;
  оценка Jaccard по совпавшим подписям ≥ ``JACCARD_THRESHOLD`` (0.8) — дубль.
* **Контейнмент** (``containment_dedup``) — содержание записи целиком лежит
  внутри другой записи: карточка K собрана из дистиллята D. Документный Jaccard
  такой пары мал (дистиллят в разы длиннее карточки), поэтому сравниваются
  ЧАНКИ: окна по ``CHUNK_CHARS`` с перекрытием, подписи чанков, порог покрытия.

Из пары дублей остаётся **старейший**. Потоковая обработка идёт в порядке
возрастания времени сессий, поэтому «старейший» — это первый встреченный;
обратный случай (дубль старше уже оставленного) считается отдельно
(``out_of_order_older``) и сигнализирует, что порядок источников нарушен.

Внешних зависимостей нет: ``zlib.crc32`` как дешёвый 64-битный хеш шингла,
MinHash — свой. Память на эпизод — 64 подписи (``array('Q')``), не шинглы.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import re
import zlib
from array import array
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

SHINGLE_SIZE = 5
NUM_PERM = 64
BANDS = 16
ROWS = 4
JACCARD_THRESHOLD = 0.8

#: Потолок числа шинглов на эпизод: точность оценки против времени прогона.
#: При переполнении берётся **bottom-k** — ``max_shingles`` наименьших хешей
#: (консистентная выборка по ЗНАЧЕНИЮ хеша, не по рангу): общий шингл попадает
#: в выборку обеих записей или ни одной.
MAX_SHINGLES = 512

_MASK64 = 0xFFFFFFFFFFFFFFFF
_EMPTY_SIGNATURE = _MASK64
_WHITESPACE = re.compile(r"\s+")

assert BANDS * ROWS == NUM_PERM, "бандинг обязан покрывать все подписи"


def normalize_text(text: str) -> str:
    """Нормализация для сравнения: регистр, пробелы, пустые края."""
    if not text:
        return ""
    return _WHITESPACE.sub(" ", text).strip().casefold()


def episode_text(episode: Any) -> str:
    """Текст эпизода — содержимое ходов подряд, без метаданных."""
    if isinstance(episode, dict):
        turns = episode.get("turns") or []
        parts = [str(turn.get("content", "")) for turn in turns if isinstance(turn, dict)]
    else:
        parts = [getattr(turn, "content", "") for turn in getattr(episode, "turns", [])]
    return "\n".join(part for part in parts if part)


def exact_fingerprint(episode: Any) -> str:
    """sha256 нормализованного текста эпизода (метаданные не участвуют)."""
    return hashlib.sha256(normalize_text(episode_text(episode)).encode("utf-8")).hexdigest()


def _shingle_hash(chunk: str) -> int:
    low = zlib.crc32(chunk.encode("utf-8")) & 0xFFFFFFFF
    high = zlib.crc32(chunk.encode("utf-8"), 0x9E3779B1) & 0xFFFFFFFF
    return (high << 32) | low


def shingle_hashes(text: str, size: int = SHINGLE_SIZE, max_shingles: int = MAX_SHINGLES) -> list[int]:
    """Уникальные хеши 5-грамм текста; при переполнении — консистентная выборка.

    Решение дельты-2 (находка боевого прогона: 32 near-дубля на 6.16M кандидатов
    при истинном Jaccard 0.91): выборка шинглов — **bottom-k по значению хеша**
    (``max_shingles`` наименьших), а не равномерная по рангам отсортированного
    списка. Ранговая выборка сравнивала у двух документов РАЗНЫЕ значения (каждый
    брал свои квантили), поэтому MinHash терял чутьё: замер на синтетике с общим
    содержанием 95% — истинный Jaccard 0.91 против оценки 0.02–0.17 при длине
    документа от 1.4 КБ. Bottom-k консистентна: шингл входит в выборку обеих
    записей или ни одной, оценка следует за истинным Jaccard (0.906 при 2 КБ,
    0.969 при 20 КБ) и остаётся нулевой на несвязанных документах (max 0.016).

    ``MAX_SHINGLES`` при этом НЕ поднимается до p99 длины записи: замер по пробе
    библиотеки (2 810 карточек K, 1 000 дистиллятов D) дал p50/p90/p99 уникальных
    шинглов 1 443/2 056/2 774 у K и 4 241/5 592/6 777 у D, то есть покрытие p99
    стоило бы 5–13× времени подписи на каждой записи боевого прогона (≈100k
    записей) без выигрыша в точности: дисперсия оценки bottom-k при k=512 —
    ≈1.3% при J=0.9, уже ниже разброса самого порога.

    Выборка детерминирована без ГПСЧ (seed не нужен): ключ — значение хеша
    шингла, поэтому подпись записи воспроизводима между прогонами.
    """
    normalized = normalize_text(text)
    if not normalized:
        return []
    if len(normalized) <= size:
        return [_shingle_hash(normalized)]
    unique = {_shingle_hash(normalized[i : i + size]) for i in range(len(normalized) - size + 1)}
    if len(unique) > max_shingles:
        return heapq.nsmallest(max_shingles, unique)
    return sorted(unique)


def _splitmix64(value: int) -> int:
    """Детерминированный 64-битный микшер (Snowflake/splitmix64)."""
    value = (value + 0x9E3779B97F4A7C15) & _MASK64
    mixed = value
    mixed = ((mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    mixed = ((mixed ^ (mixed >> 27)) * 0x94D049BB133111EB) & _MASK64
    return mixed ^ (mixed >> 31)


#: Семейство аффинных хешей h_i(x) = a_i*x + b_i (mod 2^64) — дешёвая замена
#: независимых перестановок в MinHash. Константы детерминированы, поэтому
#: подпись эпизода воспроизводима между прогонами.
_AFFINE: tuple[tuple[int, int], ...] = tuple(
    ((_splitmix64(2 * index + 1) | 1), _splitmix64(2 * index + 2)) for index in range(NUM_PERM)
)


def minhash_signature(
    text: str, num_perm: int = NUM_PERM, max_shingles: int = MAX_SHINGLES
) -> array:
    """MinHash-подпись текста: ``num_perm`` минимальных хешей."""
    hashes = shingle_hashes(text, max_shingles=max_shingles)
    if not hashes:
        return array("Q", [_EMPTY_SIGNATURE] * num_perm)
    signature = [_EMPTY_SIGNATURE] * num_perm
    for shingle in hashes:
        values = [(a * shingle + b) & _MASK64 for a, b in _AFFINE]
        signature = [current if current < value else value for current, value in zip(signature, values)]
    return array("Q", signature)


def jaccard_estimate(signature_a: Sequence[int], signature_b: Sequence[int]) -> float:
    """Оценка Jaccard по доле совпавших подписей."""
    if not signature_a or len(signature_a) != len(signature_b):
        return 0.0
    matches = 0
    for left, right in zip(signature_a, signature_b):
        if left == right:
            matches += 1
    return matches / len(signature_a)


def _band_keys(signature: Sequence[int], bands: int = BANDS, rows: int = ROWS) -> list[int]:
    """Ключи бандов подписи; ключ упакован в одно число (``band << 32 | digest``).

    Упаковка, а не кортеж: индекс контейнмента держит миллионы ключей (160k
    чанков × 16 бандов), кортеж из двух int стоит в памяти вчетверо дороже.
    """
    keys = []
    for band in range(bands):
        start = band * rows
        digest = zlib.crc32(json.dumps(tuple(signature[start : start + rows])).encode("ascii"))
        keys.append((band << 32) | (digest & 0xFFFFFFFF))
    return keys


@dataclass
class DedupStats:
    exact: int = 0
    near: int = 0
    kept: int = 0
    out_of_order_older: int = 0
    candidates: int = 0

    def to_dict(self) -> dict:
        return {
            "exact": self.exact,
            "near": self.near,
            "kept": self.kept,
            "out_of_order_older": self.out_of_order_older,
            "candidates": self.candidates,
        }


class Deduper:
    """Потоковый дедуп: помнит подписи и отпечатки уже оставленных эпизодов."""

    def __init__(self, threshold: float = JACCARD_THRESHOLD) -> None:
        self.threshold = float(threshold)
        self.stats = DedupStats()
        # Хранятся только подписи и отпечатки: сами эпизоды не удерживаются
        # в памяти (иначе полный прогон растёт на сотни МБ).
        self._by_fingerprint: dict[str, str] = {}
        self._signatures: dict[str, array] = {}
        self._started_at: dict[str, str] = {}
        self._bands: dict[tuple[int, int], list[str]] = {}

    # -- API --------------------------------------------------------------- #

    def add(self, episode: Any) -> bool:
        """True — эпизод оставлен, False — снят как дубль."""
        record = episode if isinstance(episode, dict) else episode.to_dict()
        episode_id = str(record.get("id", ""))
        started_at = str(record.get("started_at", ""))
        text = episode_text(record)

        fingerprint = hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()
        kept_id = self._by_fingerprint.get(fingerprint)
        if kept_id is not None:
            self.stats.exact += 1
            self._note_older(started_at, self._started_at.get(kept_id, ""))
            return False

        signature = minhash_signature(text)
        for candidate_id in self._near_candidates(signature):
            other = self._signatures.get(candidate_id)
            if other is None:
                continue
            self.stats.candidates += 1
            if jaccard_estimate(signature, other) >= self.threshold:
                self.stats.near += 1
                self._note_older(started_at, self._started_at.get(candidate_id, ""))
                return False

        self._remember(episode_id, fingerprint, signature, started_at)
        self.stats.kept += 1
        return True

    def add_many(self, episodes: Iterable[Any]) -> Iterable[Any]:
        """Отдать только оставленные эпизоды (порядок сохраняется)."""
        for episode in episodes:
            if self.add(episode):
                yield episode

    @property
    def kept(self) -> int:
        return self.stats.kept

    # -- внутреннее --------------------------------------------------------- #

    def _remember(
        self,
        episode_id: str,
        fingerprint: str,
        signature: array,
        started_at: str,
    ) -> None:
        self._by_fingerprint[fingerprint] = episode_id
        self._signatures[episode_id] = signature
        self._started_at[episode_id] = started_at
        for key in _band_keys(signature):
            self._bands.setdefault(key, []).append(episode_id)

    def _near_candidates(self, signature: array) -> list[str]:
        seen: dict[str, None] = {}
        for key in _band_keys(signature):
            for episode_id in self._bands.get(key, ()):  # noqa: SIM118
                seen.setdefault(episode_id, None)
        return list(seen)

    def _note_older(self, dropped_started_at: str, kept_started_at: str) -> None:
        if dropped_started_at and kept_started_at and dropped_started_at < kept_started_at:
            self.stats.out_of_order_older += 1


# --------------------------------------------------------------------------- #
# Контейнмент: содержание записи внутри другой записи
# --------------------------------------------------------------------------- #

#: Окно ЗАПРОСА (запись-содержимое, K): ширина и перекрытие соседних окон.
CHUNK_CHARS = 400
CHUNK_OVERLAP_CHARS = 200

#: Окно КОНТЕЙНЕРА (запись, внутри которой ищут содержание): уже запроса, шаг —
#: половина ширины. Геометрия выбрана из инварианта, а не подобрана: если окно
#: контейнера шириной w укладывается в окно запроса шириной 1.33·w, то вложенное
#: окно запроса совпадает с лучшим окном контейнера на w символов при ЛЮБОМ сдвиге
#: сеток, то есть схожесть равна ≈0.75 геометрически (300/(400+300−300)), а не
#: «в среднем 0.6 с разбросом». Симметричная нарезка 400/400 давала на худшем
#: сдвиге схожесть 0.6 при пороге 0.55 — на границе разброса оценки (σ≈0.08),
#: то есть треть вложенных окон терялась бы случайно.
CONTAINER_CHUNK_CHARS = 300
CONTAINER_OVERLAP_CHARS = 150

#: Подписи чанков: своё число перестановок и своя сетка бандов. Чанк короче
#: эпизода, поэтому банды мелкие (2 строки): при схожести 0.55 бандинг 16×2 даёт
#: recall ≈1.0 на пару, тогда как 16×4 — лишь 0.78.
CHUNK_NUM_PERM = 32
CHUNK_BANDS = 16
CHUNK_ROWS = 2

#: Порог схожести окна запроса с окном контейнера. 0.55 при геометрическом
#: максимуме вложенного окна 0.75: запас 0.2 ≈ 2.4σ оценки (σ≈0.08 при 32
#: перестановках), то есть вложенное окно находится с вероятностью ≈0.99 на
#: пару «окно запроса ↔ окно контейнера» и ≈1.0 с учётом соседних окон.
CHUNK_JACCARD = 0.55

#: Порог ступени: доля чанков записи, покрытых чанками ОДНОГО контейнера.
CONTAINMENT_THRESHOLD = 0.8

#: Роли ступени по умолчанию: (контейнер, содержимое) — дистиллят D богаче
#: контекстом, чем карточка K, поэтому снимается K.
CONTAINMENT_SIDES: tuple[tuple[str, str], ...] = (("D", "K"),)

assert CHUNK_BANDS * CHUNK_ROWS <= CHUNK_NUM_PERM, "бандинг чанков не покрывает подпись"
#: Уникальных шинглов в окне не больше, чем символов минус размер шингла, —
#: выборка шинглов внутри чанка не включается никогда (подпись чанка точная).
assert max(CHUNK_CHARS, CONTAINER_CHUNK_CHARS) - SHINGLE_SIZE <= MAX_SHINGLES, (
    "окно чанка не должно сэмплироваться"
)
assert CONTAINER_OVERLAP_CHARS <= CONTAINER_CHUNK_CHARS <= CHUNK_CHARS, (
    "окно контейнера обязано укладываться в окно запроса"
)


def chunk_windows(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> list[str]:
    """Окна нормализованного текста шириной ``size`` с перекрытием ``overlap``.

    Нормализация — до нарезки: у обеих сторон сравнения окна считаются в одном
    пространстве (регистр и пробелы не различаются). Последнее окно — хвост
    текста, поэтому короткий хвост не теряется.
    """
    normalized = normalize_text(text)
    if not normalized:
        return []
    if len(normalized) <= size:
        return [normalized]
    step = max(1, size - overlap)
    last_start = len(normalized) - size
    starts = list(range(0, last_start + 1, step))
    if starts[-1] != last_start:
        starts.append(last_start)
    return [normalized[start : start + size] for start in starts]


def required_chunks(total: int, threshold: float = CONTAINMENT_THRESHOLD) -> int:
    """Сколько чанков записи обязаны быть покрыты, чтобы запись считалась вложенной."""
    return max(1, math.ceil(threshold * total))


def _record_fields(record: Any) -> tuple[str, str, str]:
    """``(id, component, text)`` записи — словарём или кортежем.

    Ступень компонентно-агностична: она не знает, что такое K и D, — роли
    (контейнер, содержимое) задаёт вызывающий.
    """
    if isinstance(record, Mapping):
        return (
            str(record.get("id", "")),
            str(record.get("component", "")),
            str(record.get("text", "")),
        )
    record_id, component, text = record
    return str(record_id), str(component), str(text)


def _blocks(text: str, extract_blocks: Callable[[str], Sequence[str]] | None) -> list[str]:
    """Блоки записи, среди которых ищется содержание (по умолчанию — весь текст)."""
    if extract_blocks is None:
        return [text]
    blocks = extract_blocks(text)
    if isinstance(blocks, str):  # защита от «вернул строку вместо последовательности»
        return [blocks]
    return [str(block) for block in blocks]


@dataclass
class ContainmentStats:
    """Числа ступени контейнмента: ни содержимого, ни путей — только счётчики."""

    threshold: float = CONTAINMENT_THRESHOLD
    chunk_jaccard: float = CHUNK_JACCARD
    chunk_chars: int = CHUNK_CHARS
    chunk_overlap_chars: int = CHUNK_OVERLAP_CHARS
    container_chunk_chars: int = CONTAINER_CHUNK_CHARS
    container_overlap_chars: int = CONTAINER_OVERLAP_CHARS
    containers: int = 0
    checked: int = 0
    checked_pairs: int = 0
    dropped: int = 0
    dropped_by_component: Counter = field(default_factory=Counter)
    coverage_histogram: Counter = field(default_factory=Counter)

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "chunk_jaccard": self.chunk_jaccard,
            "chunk_chars": self.chunk_chars,
            "chunk_overlap_chars": self.chunk_overlap_chars,
            "container_chunk_chars": self.container_chunk_chars,
            "container_overlap_chars": self.container_overlap_chars,
            "containers": self.containers,
            "checked": self.checked,
            "checked_pairs": self.checked_pairs,
            "dropped": self.dropped,
            "dropped_by_component": dict(sorted(self.dropped_by_component.items())),
            "coverage_histogram": {
                label: int(self.coverage_histogram[label]) for label in sorted(self.coverage_histogram)
            },
        }


class ContainmentIndex:
    """Индекс чанков записей-контейнеров: кандидаты — по бандам подписей чанков.

    Почему чанки, а не документы: карточка K внутри дистиллята D даёт документный
    Jaccard ≈0.3 (дистиллят в разы длиннее), и документный LSH такую пару почти
    не находит. Совпадающие же чанки (окна запроса и контейнера) похожи почти
    целиком, поэтому находятся бандингом надёжно.

    Почему окна контейнера уже окон запроса (``CONTAINER_CHUNK_CHARS`` против
    ``CHUNK_CHARS``): сетки окон двух записей сдвинуты друг относительно друга,
    и при равной ширине вложенное окно совпадает с лучшим чужим лишь частично.
    Окно контейнера шириной 300 при шаге 150 гарантированно укладывается внутрь
    окна запроса шириной 400 при любом сдвиге, поэтому схожесть вложенного окна
    равна ≈0.75 геометрически — без «хвоста» распределения у самого порога.
    """

    def __init__(
        self,
        container_chars: int = CONTAINER_CHUNK_CHARS,
        container_overlap_chars: int = CONTAINER_OVERLAP_CHARS,
        query_chars: int = CHUNK_CHARS,
        query_overlap_chars: int = CHUNK_OVERLAP_CHARS,
        chunk_jaccard: float = CHUNK_JACCARD,
        num_perm: int = CHUNK_NUM_PERM,
        bands: int = CHUNK_BANDS,
        rows: int = CHUNK_ROWS,
    ) -> None:
        self.container_chars = int(container_chars)
        self.container_overlap_chars = int(container_overlap_chars)
        self.query_chars = int(query_chars)
        self.query_overlap_chars = int(query_overlap_chars)
        self.chunk_jaccard = float(chunk_jaccard)
        self.num_perm = int(num_perm)
        self.bands = int(bands)
        self.rows = int(rows)
        self._signatures: list[array] = []
        self._owners: list[str] = []
        #: Ключ банда → id чанка (int) либо список id (при совпадении ключей).
        self._bands: dict[int, int | list[int]] = {}
        self._owner_ids: dict[str, int] = {}

    # -- наполнение --------------------------------------------------------- #

    def add(self, owner_id: str, text: str, blocks: Sequence[str] | None = None) -> int:
        """Проиндексировать чанки записи-контейнера; вернуть число чанков."""
        self._owner_ids.setdefault(owner_id, len(self._owner_ids))
        chunks = self.chunks(text, blocks)
        for chunk in chunks:
            signature = minhash_signature(chunk, num_perm=self.num_perm)
            chunk_id = len(self._signatures)
            self._signatures.append(signature)
            self._owners.append(owner_id)
            for key in _band_keys(signature, self.bands, self.rows):
                entry = self._bands.get(key)
                if entry is None:
                    self._bands[key] = chunk_id
                elif isinstance(entry, int):
                    self._bands[key] = [entry, chunk_id]
                else:
                    entry.append(chunk_id)
        return len(chunks)

    def chunks(self, text: str, blocks: Sequence[str] | None = None) -> list[str]:
        """Чанки записи-контейнера: окна каждого блока подряд."""
        return self._windows(text, blocks, self.container_chars, self.container_overlap_chars)

    def query_chunks(self, text: str, blocks: Sequence[str] | None = None) -> list[str]:
        """Чанки записи-содержимого (запроса): окна каждого блока подряд."""
        return self._windows(text, blocks, self.query_chars, self.query_overlap_chars)

    @staticmethod
    def _windows(
        text: str, blocks: Sequence[str] | None, size: int, overlap: int
    ) -> list[str]:
        parts = blocks if blocks is not None else [text]
        windows: list[str] = []
        for part in parts:
            windows.extend(chunk_windows(part, size, overlap))
        return windows

    @property
    def owners(self) -> int:
        return len(self._owner_ids)

    @property
    def chunks_total(self) -> int:
        return len(self._signatures)

    # -- запрос ------------------------------------------------------------- #

    def coverage(
        self, text: str, threshold: float = CONTAINMENT_THRESHOLD, blocks: Sequence[str] | None = None
    ) -> tuple[dict[str, int], int]:
        """``({запись-контейнер: покрытых чанков}, всего чанков запроса)``.

        Ранний выход: контейнер, впервые зацепившийся на чанке с номером ``j``,
        может покрыть не более ``n - j`` чанков запроса. Как только все живые
        кандидаты потеряли шанс добрать порог, обход чанков прекращается: ни
        один новый кандидат порога уже не достигнет. Покрытие не снятых пар
        поэтому — нижняя оценка на момент решения (снятых — точное).
        """
        chunks = self.query_chunks(text, blocks)
        total = len(chunks)
        if not total or not self._bands:
            return {}, total
        required = required_chunks(total, threshold)
        allowed_misses = total - required
        hits: dict[str, int] = {}
        alive: set[str] = set()
        for index, chunk in enumerate(chunks):
            # ``index > allowed_misses``: кандидат, впервые зацепившийся на этом
            # чанке, может добрать не более ``total - index`` — порога не хватит.
            if index > allowed_misses and not alive:
                break
            signature = minhash_signature(chunk, num_perm=self.num_perm)
            for owner_id in self._matched_owners(signature):
                hits[owner_id] = hits.get(owner_id, 0) + 1
                alive.add(owner_id)
            remaining = total - index - 1
            alive = {owner for owner in alive if hits[owner] + remaining >= required}
        return hits, total

    def _matched_owners(self, signature: array) -> set[str]:
        """Записи-контейнеры, у которых есть чанк, схожий с подписью запроса."""
        matched: set[str] = set()
        for key in _band_keys(signature, self.bands, self.rows):
            entry = self._bands.get(key)
            if entry is None:
                continue
            candidates = (entry,) if isinstance(entry, int) else entry
            for chunk_id in candidates:
                if jaccard_estimate(signature, self._signatures[chunk_id]) >= self.chunk_jaccard:
                    matched.add(self._owners[chunk_id])
        return matched


def containment_dedup(
    records: Iterable[Any],
    extract_blocks: Callable[[str], Sequence[str]] | None = None,
    threshold: float = CONTAINMENT_THRESHOLD,
    *,
    sides: Sequence[tuple[str, str]] = CONTAINMENT_SIDES,
    container_chars: int = CONTAINER_CHUNK_CHARS,
    container_overlap_chars: int = CONTAINER_OVERLAP_CHARS,
    chunk_chars: int = CHUNK_CHARS,
    overlap_chars: int = CHUNK_OVERLAP_CHARS,
    chunk_jaccard: float = CHUNK_JACCARD,
) -> tuple[list[Any], ContainmentStats]:
    """Снять записи, содержание которых лежит внутри записи-контейнера.

    Контейнмент — не дубль: документный Jaccard пары «карточка K, дистиллят D»
    мал (D в разы длиннее), а содержание карточки совпадает с фрагментом D почти
    целиком. Ступень сравнивает ЧАНКИ: если доля покрытых чанков записи-содержимого
    не меньше ``threshold`` (0.8), запись снимается — контейнер богаче контекстом;
    записи-контейнеры не снимаются никогда.

    ``records`` — записи ``(id, component, text)`` словарём или кортежем; роли
    задаёт ``sides`` (по умолчанию ``(("D", "K"),)``), поэтому ступень не знает
    предметных имён компонентов. ``extract_blocks`` ограничивает текст, среди
    которого ищется содержание (например, тело карточки без синтезированного
    заголовка); по умолчанию берётся весь текст. «Старейшинство» компонентов
    ступень не трогает: снятие всегда на стороне содержимого.

    Порядок записей на решение не влияет: контейнеры индексируются все до того,
    как принимается первое решение. Детерминированности ГПСЧ не нужен — выборка
    шинглов контентная (см. :func:`shingle_hashes`).
    """
    materialized = list(records)
    fields = [_record_fields(record) for record in materialized]
    stats = ContainmentStats(
        threshold=float(threshold),
        chunk_jaccard=float(chunk_jaccard),
        chunk_chars=int(chunk_chars),
        chunk_overlap_chars=int(overlap_chars),
        container_chunk_chars=int(container_chars),
        container_overlap_chars=int(container_overlap_chars),
    )

    indices: dict[str, ContainmentIndex] = {}
    for container, _contained in sides:
        index = ContainmentIndex(
            container_chars,
            container_overlap_chars,
            chunk_chars,
            overlap_chars,
            chunk_jaccard,
        )
        for record_id, component, text in fields:
            if component == container:
                index.add(record_id, text, _blocks(text, extract_blocks))
        stats.containers += index.owners
        indices[container] = index

    contained_roles = {contained for _container, contained in sides}
    kept: list[Any] = []
    for (record_id, component, text), record in zip(fields, materialized):
        if component not in contained_roles:
            kept.append(record)
            continue
        stats.checked += 1
        drop = False
        for container, contained in sides:
            if component != contained:
                continue
            index = indices[container]
            hits, total = index.coverage(text, threshold, _blocks(text, extract_blocks))
            for owner_id, covered in hits.items():
                stats.checked_pairs += 1
                coverage = covered / total if total else 0.0
                # Корзина — нижняя граница («покрытие не ниже 0.8»): иначе 0.86
                # округлилось бы до «0.9» и таблица врала бы у самого порога.
                stats.coverage_histogram[f"{math.floor(coverage * 10) / 10:.1f}"] += 1
                if covered >= required_chunks(total, threshold):
                    drop = True
        if drop:
            stats.dropped += 1
            stats.dropped_by_component[component] += 1
        else:
            kept.append(record)
    return kept, stats
