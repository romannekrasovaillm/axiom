"""Экспортёр DATA-001: отпечаток набора полей и типов (дельта G3, ADR-038).

По JSONL (и Parquet — если доступен pyarrow) строит отпечаток набора
``поле:тип`` и сравнивает с эталонным отпечатком. Только файлы, без сети.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from tools.sensors.protocol import BaseExporter, Fact, SensorSpec


def _normalize_type(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if value is None:
        return "null"
    return type(value).__name__


def _fingerprint(fields: dict[str, str]) -> str:
    canonical = "\n".join(f"{key}:{fields[key]}" for key in sorted(fields))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _from_jsonl(path: Path, sample: int) -> Optional[dict[str, str]]:
    fields: dict[str, str] = {}
    seen = 0
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                return None
            for key, value in record.items():
                tname = _normalize_type(value)
                if key in fields and fields[key] != tname:
                    fields[key] = "mixed"
                else:
                    fields[key] = tname
            seen += 1
            if sample and seen >= sample:
                break
    except (OSError, json.JSONDecodeError):
        return None
    return fields


def _from_parquet(path: Path) -> Optional[dict[str, str]]:
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415 — необязательная зависимость
    except Exception:  # noqa: BLE001
        return None
    try:
        schema = pq.read_schema(path)
    except Exception:  # noqa: BLE001
        return None
    mapping = {
        "int64": "int", "int32": "int", "double": "float", "float": "float",
        "string": "str", "large_string": "str", "bool": "bool",
    }
    fields: dict[str, str] = {}
    for name, dtype in zip(schema.names, schema.types):
        fields[str(name)] = mapping.get(str(dtype), str(dtype))
    return fields


class SchemaFingerprintExporter(BaseExporter):
    SPEC = SensorSpec(
        id="DATA-001",
        facts=("field_fingerprint", "field_count", "drift_vs_reference"),
        schema={
            "field_fingerprint": {"unit": "", "quality": "wrapped", "level": "component"},
            "field_count": {"unit": "count", "quality": "wrapped", "level": "component"},
            "drift_vs_reference": {"unit": "bool", "quality": "wrapped", "level": "end_to_end"},
        },
        level="component",
        raw={"path": "<dataset.jsonl|parquet>", "format": "jsonl", "retention_days": None, "in_git": False},
        context=("repo",),
        pack="data",
    )

    def collect(self, subject: dict[str, Any], *, input_path: Any = None, reference: Optional[str] = None,
                sample: int = 10000, **_: Any) -> list[Fact]:
        if not input_path:
            return [self.unavailable(f, subject, "не передан файл набора (input_path)") for f in self.SPEC.facts]
        path = Path(input_path)
        if not path.is_file():
            return [self.unavailable(f, subject, f"набор не найден: {path}") for f in self.SPEC.facts]
        if path.suffix.lower() == ".parquet":
            fields = _from_parquet(path)
            if fields is None:
                return [self.unavailable(f, subject, "Parquet: нет pyarrow или файл не читается") for f in self.SPEC.facts]
        else:
            fields = _from_jsonl(path, sample)
            if fields is None:
                return [self.unavailable(f, subject, f"JSONL не читается: {path}") for f in self.SPEC.facts]

        fp = _fingerprint(fields)
        method = f"отпечаток {len(fields)} полей из {path.name}"
        facts = [
            self.fact("field_fingerprint", fp, subject=subject, method=method),
            self.fact("field_count", len(fields), subject=subject, method=method),
        ]
        if reference:
            facts.append(self.fact("drift_vs_reference", fp != reference, subject=subject, method=method))
        else:
            facts.append(self.unavailable("drift_vs_reference", subject, "не передан эталонный отпечаток (reference)"))
        return facts


EXPORTER = SchemaFingerprintExporter()
