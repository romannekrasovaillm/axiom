"""Шард W: веб-корпус FineWeb-Edu (ADR-021, ~17B токенов).

Источник — ``HuggingFaceFW/fineweb-edu``, конфиг ``sample-100BT`` (100B токенов
edu-выборки: цели 17B хватает с запасом, а множество parquet-файлов на порядок
меньше полного ``data/*/*``). Чтение — потоковое (``datasets`` streaming):
датасет не материализуется, в памяти живёт одна запись и множество хешей дедупа.

Фильтры: качество уже отфильтровано edu-классификатором на стороне датасета —
дополнительный порог ``min_int_score`` по умолчанию ВЫКЛЮЧЕН (ручка оставлена
зарезервированной, её значение фиксируется в отчёте). Дедупликация документов —
точная, по sha256 текста, с ограниченным окном (страховка: FineWeb уже
дедуплицирован).

Запись: ``{"text": …, "meta": {"source": …, "id": …, "url": …, "dump": …}}``
в jsonl-шарды ``W-000NN.jsonl.zst`` по ~500 МБ.
"""

from __future__ import annotations

import os
from typing import Any, Iterator

from . import common

SHARD = "W"

#: Канонический репозиторий; HuggingFaceH4/fineweb-edu — тот же корпус-зеркало.
DEFAULT_REPO = "HuggingFaceFW/fineweb-edu"
DEFAULT_CONFIG = "sample-100BT"
DEFAULT_SPLIT = "train"

#: Боевой источник по умолчанию (спецификация JSON-сериализуема и идёт в отчёт).
DEFAULT_SOURCE: dict[str, Any] = {
    "kind": "hf",
    "repo": DEFAULT_REPO,
    "config": DEFAULT_CONFIG,
    "split": DEFAULT_SPLIT,
}

DEFAULT_TARGET_TOKENS = 17_000_000_000


class FinewebDocuments:
    """Нормализация записи FineWeb-Edu → ``(text, meta)``; ``None`` — отброшена.

    Объект копит причины отбраковки в ``stats`` — они попадают в отчёт.
    """

    def __init__(
        self,
        min_int_score: int | None = None,
        max_chars: int | None = None,
        source_name: str = DEFAULT_REPO,
    ) -> None:
        self.min_int_score = min_int_score
        self.max_chars = max_chars
        self.source_name = source_name
        self.stats: dict[str, int] = {
            "dropped_empty_text": 0,
            "dropped_low_int_score": 0,
            "dropped_oversized": 0,
        }

    def __call__(self, record: dict) -> tuple[str, dict] | None:
        text = record.get("text") or ""
        if not isinstance(text, str) or not text.strip():
            self.stats["dropped_empty_text"] += 1
            return None
        if self.min_int_score is not None:
            score = record.get("int_score")
            if score is None or int(score) < self.min_int_score:
                self.stats["dropped_low_int_score"] += 1
                return None
        if self.max_chars is not None and len(text) > self.max_chars:
            self.stats["dropped_oversized"] += 1
            return None
        meta = {
            "source": record.get("source") or self.source_name,
            "id": record.get("id"),
            "dump": record.get("dump"),
            "url": record.get("url"),
            "language": record.get("language"),
            "int_score": record.get("int_score"),
        }
        return text, {key: value for key, value in meta.items() if value is not None}


def iter_documents(source_spec: dict | None = None) -> Iterator[dict]:
    """Поток сырых записей источника W (для проб и отладки)."""
    spec = source_spec or DEFAULT_SOURCE
    func, kwargs = common.source_iterator(spec)
    return common.iter_source(func, **kwargs)


def prepare_w(
    out_dir: str | os.PathLike[str] | None = None,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    source_spec: dict | None = None,
    manifest_path: str | os.PathLike[str] | None = None,
    report_path: str | os.PathLike[str] | None = None,
    min_int_score: int | None = None,
    **kwargs: Any,
) -> dict:
    """Подготовить шард W: поток → фильтр → дедуп → jsonl-шарды в ``out_dir``."""
    root = out_dir or os.path.join(common.DATASET_ROOT, SHARD)
    source = dict(source_spec or DEFAULT_SOURCE)
    source_name = source.get("repo") or source.get("glob") or DEFAULT_REPO
    manifest = manifest_path or os.path.join(root, "manifest-w.json")
    report = report_path or os.path.join(root, "report-w.json")
    return common.run_shard(
        shard=SHARD,
        out_dir=root,
        target_tokens=target_tokens,
        source_spec=source,
        normalize=FinewebDocuments(min_int_score=min_int_score, source_name=str(source_name)),
        manifest_path=manifest,
        report_path=report,
        **kwargs,
    )
