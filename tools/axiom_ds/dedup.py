"""Ступень 4 пайплайна axiom-domain-ds-v1: точные дубли и near-dup (MinHash).

* **Точный дубль** — совпал sha256 нормализованного текста эпизода
  (регистр и пробелы не различаются, метаданные и id — не участвуют).
* **Near-dup** — 5-граммные шинглы → MinHash на 64 подписи → бандинг 16×4;
  оценка Jaccard по совпавшим подписям ≥ ``JACCARD_THRESHOLD`` (0.8) — дубль.

Из пары дублей остаётся **старейший**. Потоковая обработка идёт в порядке
возрастания времени сессий, поэтому «старейший» — это первый встреченный;
обратный случай (дубль старше уже оставленного) считается отдельно
(``out_of_order_older``) и сигнализирует, что порядок источников нарушен.

Внешних зависимостей нет: ``zlib.crc32`` как дешёвый 64-битный хеш шингла,
MinHash — свой. Память на эпизод — 64 подписи (``array('Q')``), не шинглы.
"""

from __future__ import annotations

import hashlib
import json
import re
import zlib
from array import array
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

SHINGLE_SIZE = 5
NUM_PERM = 64
BANDS = 16
ROWS = 4
JACCARD_THRESHOLD = 0.8

#: Потолок числа шинглов на эпизод: точность оценки против времени прогона.
#: Шинглы берутся равномерной выборкой из отсортированных хешей.
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
    """Уникальные хеши 5-грамм текста (равномерная выборка при переполнении)."""
    normalized = normalize_text(text)
    if not normalized:
        return []
    if len(normalized) <= size:
        return [_shingle_hash(normalized)]
    unique = {_shingle_hash(normalized[i : i + size]) for i in range(len(normalized) - size + 1)}
    if len(unique) > max_shingles:
        ordered = sorted(unique)
        step = len(ordered) / max_shingles
        unique = {ordered[int(index * step)] for index in range(max_shingles)}
    return sorted(unique)


def _signature_values(shingle_hash: int) -> list[int]:
    """64 подписи одного шингла: crc32 с разными зёрнами."""
    values = []
    for seed in range(NUM_PERM):
        mixed = zlib.crc32(shingle_hash.to_bytes(8, "big"), seed * 0x9E3779B1 & 0xFFFFFFFF)
        values.append((mixed & _MASK64))
    return values


def minhash_signature(
    text: str, num_perm: int = NUM_PERM, max_shingles: int = MAX_SHINGLES
) -> array:
    """MinHash-подпись текста: ``num_perm`` минимальных хешей."""
    hashes = shingle_hashes(text, max_shingles=max_shingles)
    if not hashes:
        return array("Q", [_EMPTY_SIGNATURE] * num_perm)
    signature = array("Q", [_EMPTY_SIGNATURE] * num_perm)
    for shingle in hashes:
        values = _signature_values(shingle)
        for index in range(num_perm):
            value = values[index]
            if value < signature[index]:
                signature[index] = value
    return signature


def jaccard_estimate(signature_a: Sequence[int], signature_b: Sequence[int]) -> float:
    """Оценка Jaccard по доле совпавших подписей."""
    if not signature_a or len(signature_a) != len(signature_b):
        return 0.0
    matches = 0
    for left, right in zip(signature_a, signature_b):
        if left == right:
            matches += 1
    return matches / len(signature_a)


def _band_keys(signature: Sequence[int]) -> list[tuple[int, int]]:
    keys = []
    for band in range(BANDS):
        start = band * ROWS
        chunk = tuple(signature[start : start + ROWS])
        digest = zlib.crc32(json.dumps(chunk).encode("ascii")) & 0xFFFFFFFF
        keys.append((band, digest))
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
        self._by_fingerprint: dict[str, dict] = {}
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
        if fingerprint in self._by_fingerprint:
            kept = self._by_fingerprint[fingerprint]
            self.stats.exact += 1
            self._note_older(started_at, self._started_at.get(str(kept.get("id", "")), ""))
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

        self._remember(episode_id, fingerprint, signature, started_at, record)
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
        record: dict,
    ) -> None:
        self._by_fingerprint[fingerprint] = record
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
