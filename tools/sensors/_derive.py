"""Помощники производных фактов (дельта C6, ADR-036).

Производный факт (``quality: derived``) обязан ссылаться в ``inputs`` на
исходные записи: файл + sha256 строки. :func:`fact_ref` достаёт последнюю
подходящую запись и возвращает такую ссылку; :func:`missing` собирает список
недоступных входов, чтобы факт не подставлял значение, а честно писал
``unverified``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from . import fact as fact_mod
from ._common import REPO_ROOT


def fact_ref(
    sensor: str,
    fact: str,
    *,
    out_dir: Optional[str | Path] = None,
    subject_filter: Any = None,
    subject: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Ссылка на последнюю запись факта: ``{file, sha256, fact, sensor, value}``."""
    record = fact_mod.read_latest(
        sensor, fact, subject_filter, out_dir=out_dir, subject=subject
    )
    if record is None:
        return None
    raw = record.get("_raw")
    base = Path(out_dir) if out_dir is not None else fact_mod.DEFAULT_FACTS_DIR
    try:
        rel_path = (base / f"{sensor}.jsonl").relative_to(REPO_ROOT).as_posix()
    except ValueError:
        rel_path = f"evidence/facts/{sensor}.jsonl"
    return {
        "file": rel_path,
        "sha256": fact_mod.line_sha256(raw) if raw else None,
        "fact": fact,
        "sensor": sensor,
        "value": record.get("value"),
        "status": record.get("status"),
    }


def missing(*refs: Optional[dict[str, Any]]) -> list[str]:
    """Имена недоступных входов (для ``inputs``/``note``)."""
    names = []
    for ref in refs:
        if ref is None:
            names.append("<нет>")
        elif ref.get("status") == "unverified" or ref.get("value") is None:
            names.append(f"{ref['sensor']}:{ref['fact']}")
    return names
