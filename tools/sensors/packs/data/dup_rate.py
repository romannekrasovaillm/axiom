"""Экспортёр DATA-002: доля near-дубликатов (дельта G3, ADR-038).

MinHash переиспользуется из ``tools/axiom_ds/dedup.py`` (``minhash_signature``,
``jaccard_estimate``, ``normalize_text``) — второй реализации нет. Вход — JSONL
с текстовым полем; сравнение детерминированное, без сети.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.sensors.protocol import BaseExporter, Fact, SensorSpec


def _load_texts(path: Path, field: str) -> list[str]:
    texts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if not isinstance(record, dict) or field not in record:
            raise ValueError(f"нет поля {field!r} в записи")
        texts.append(str(record[field]))
    return texts


class DupRateExporter(BaseExporter):
    SPEC = SensorSpec(
        id="DATA-002",
        facts=("dup_rate", "n_records", "n_duplicates"),
        schema={
            "dup_rate": {"unit": "fraction", "quality": "derived", "level": "end_to_end"},
            "n_records": {"unit": "count", "quality": "wrapped", "level": "component"},
            "n_duplicates": {"unit": "count", "quality": "derived", "level": "end_to_end"},
        },
        level="end_to_end",
        raw={"path": "<dataset.jsonl>", "format": "jsonl", "retention_days": None, "in_git": False},
        context=("repo",),
        pack="data",
    )

    def collect(self, subject: dict[str, Any], *, input_path: Any = None, field: str = "text",
                threshold: float = 0.8, **_: Any) -> list[Fact]:
        if not input_path:
            return [self.unavailable(f, subject, "не передан файл набора (input_path)") for f in self.SPEC.facts]
        path = Path(input_path)
        if not path.is_file():
            return [self.unavailable(f, subject, f"набор не найден: {path}") for f in self.SPEC.facts]
        try:
            texts = _load_texts(path, field)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            return [self.unavailable(f, subject, f"набор не читается: {exc}") for f in self.SPEC.facts]
        if not texts:
            return [self.unavailable(f, subject, "набор пуст") for f in self.SPEC.facts]

        from tools.axiom_ds.dedup import jaccard_estimate, minhash_signature, normalize_text

        normalized = [normalize_text(t) for t in texts]
        signatures = [minhash_signature(t) for t in texts]
        keep = [True] * len(texts)
        for i in range(len(texts)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(texts)):
                if not keep[j]:
                    continue
                if normalized[i] == normalized[j] or jaccard_estimate(signatures[i], signatures[j]) >= threshold:
                    keep[j] = False
        duplicates = len(texts) - sum(keep)
        method = f"MinHash (axiom_ds/dedup) по полю {field!r}, threshold={threshold} из {path.name}"
        return [
            self.fact("dup_rate", round(duplicates / len(texts), 6), subject=subject, method=method),
            self.fact("n_records", len(texts), subject=subject, method=method),
            self.fact("n_duplicates", duplicates, subject=subject, method=method),
        ]


EXPORTER = DupRateExporter()
