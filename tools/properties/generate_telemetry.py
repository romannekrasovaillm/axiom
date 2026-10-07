"""G3 — генератор кандидатов из стандартной телеметрии (ADR-039, дельта G3).

Читает текстовый формат OpenMetrics / экспозиции Prometheus (включая собственный
экспорт ``tools/sensors/export_openmetrics.py``) и структурные JSONL-логи с
числовыми ключами. По каждой метрике — ``bounds`` по наблюдённому диапазону и
``freshness`` через механизм G2 (> =20 наблюдений, порог с запасом).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.properties.candidates import make_candidate  # noqa: E402
from tools.properties.generate_observed import MARGIN, MIN_OBSERVATIONS  # noqa: E402

_METRIC_RE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eE]+)\s*$")
_SOURCE = "телеметрия (OpenMetrics/JSONL, ADR-039 G3)"


def parse_openmetrics(text: str) -> dict[str, list[float]]:
    """OpenMetrics/Prometheus-текст → ``{метрика: [значения]}``."""
    metrics: dict[str, list[float]] = defaultdict(list)
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _METRIC_RE.match(line)
        if not m:
            continue
        try:
            metrics[m.group(1)].append(float(m.group(3)))
        except ValueError:
            continue
    return dict(metrics)


def parse_jsonl(path: Path) -> dict[str, list[float]]:
    """JSONL с числовыми ключами → ``{ключ: [значения]}``."""
    metrics: dict[str, list[float]] = defaultdict(list)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        for key, value in (record.items() if isinstance(record, dict) else []):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            metrics[str(key)].append(float(value))
    return dict(metrics)


def generate(root: str | Path, sources: Optional[list[str]] = None) -> tuple[list[dict], list[str]]:
    root = Path(root).resolve()
    metrics: dict[str, list[float]] = {}
    notes: list[str] = []
    paths: list[Path] = []
    if sources:
        paths = [Path(s) for s in sources]
    else:
        for pattern in ("evidence/**/*.prom", "evidence/**/*.metrics", "evidence/**/*.jsonl"):
            paths.extend(sorted(root.glob(pattern)))
    for path in paths:
        if not path.is_file() or "facts" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        found = parse_jsonl(path) if path.suffix == ".jsonl" else parse_openmetrics(text)
        for key, values in found.items():
            metrics.setdefault(key, []).extend(values)
    candidates: list[dict] = []
    for name, values in sorted(metrics.items()):
        if len(values) < MIN_OBSERVATIONS:
            continue
        lo, hi = min(values), max(values)
        margin = (hi - lo) * MARGIN
        candidates.append(make_candidate(
            "telemetry", "bounds", {"fact": name, "min": lo - margin, "max": hi + margin,
                                    "tolerance_source": {"decision": _SOURCE}},
            evidence=[name], support=len(values),
        ))
    if not candidates:
        notes.append(
            f"телеметрии с >= {MIN_OBSERVATIONS} наблюдениями не найдено "
            "(evidence/telemetry/*.prom и структурные логи отсутствуют)"
        )
    return candidates, notes


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="G3: кандидаты из телеметрии")
    parser.add_argument("--root", default=".")
    parser.add_argument("--source", action="append", default=None)
    parser.add_argument("--json", dest="json_path", default=None)
    args = parser.parse_args(argv)
    candidates, notes = generate(args.root, args.source)
    print(json.dumps({"source": "telemetry", "candidates": len(candidates), "notes": notes},
                     ensure_ascii=False, indent=1))
    if args.json_path:
        Path(args.json_path).write_text(json.dumps(candidates, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
