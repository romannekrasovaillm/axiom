"""Общие помощники обёрток существующих производителей фактов (дельта C3).

Обёртка **не меняет** инструмент — только читает его выход (``quality: wrapped``)
и переводит в запись факта. Отсутствие источника → ``unverified``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from ._common import REPO_ROOT


def run_tool(script_rel: str, args: list[str], *, timeout: int = 120) -> subprocess.CompletedProcess:
    """Запускает существующий инструмент кейса текущим интерпретатором."""
    script = REPO_ROOT / script_rel
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True, text=True, timeout=timeout, cwd=str(REPO_ROOT),
    )


def read_json(path: str | Path) -> Optional[Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return rows
    for line in text.split("\n"):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def median(values: list[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def latest_glob(pattern: str, root: Optional[Path] = None) -> Optional[Path]:
    base = root or REPO_ROOT
    matches = sorted(base.glob(pattern))
    return matches[-1] if matches else None


def sha256_of(path: str | Path) -> Optional[str]:
    from ._common import sha256_file

    return sha256_file(path)
