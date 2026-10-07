"""G2 — генератор кандидатов из наблюдений (ADR-039, дельта G2).

Вывод инвариантов в духе Daikon по истории фактов `evidence/facts/`:
константы, диапазоны [min, max] с запасом, равенства/стабильные отношения между
фактами одного предмета, монотонность рядов.

Правила вывода (ADR-039): минимум 20 наблюдений без контрпримеров; выведенный
диапазон всегда помечается «по выборке», порог берётся с запасом, а не по
крайним точкам; факты уровня `diagnostic` дают кандидатов только для `freshness`
и `liveness`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.properties.candidates import make_candidate  # noqa: E402

#: Минимум наблюдений без контрпримеров (ADR-039, дельта G2).
MIN_OBSERVATIONS = 20
#: Запас диапазона «по выборке» (доля размаха).
MARGIN = 0.10
_SOURCE = "по выборке (>=20 наблюдений без контрпримеров, ADR-039 G2)"


def _levels(root: Path) -> dict[tuple[str, str], str]:
    from tools.miniyaml import load_file

    data = load_file(root / "model" / "sensors.yaml")
    out: dict[tuple[str, str], str] = {}
    for s in data if isinstance(data, list) else []:
        if not isinstance(s, dict):
            continue
        base = s.get("level")
        overrides = s.get("fact_levels") or {}
        for fact in s.get("facts") or []:
            out[(str(s.get("id")), str(fact))] = str(overrides.get(fact, base))
    return out


def _series(root: Path) -> dict[tuple[str, str], list[dict]]:
    from tools.sensors.fact import read_records

    by_fact: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for path in sorted((root / "evidence" / "facts").glob("S-*.jsonl")):
        sensor = path.stem
        try:
            for record in read_records(sensor, root / "evidence" / "facts"):
                by_fact[(sensor, str(record.get("fact")))].append(record)
        except Exception:  # noqa: BLE001 — битая история пропускается с отчётом
            continue
    return by_fact


def _numeric(records: list[dict]) -> list[float]:
    out = []
    for r in records:
        if r.get("status") != "ok":
            continue
        v = r.get("value")
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        out.append(float(v))
    return out


def generate(root: str | Path) -> tuple[list[dict], list[str]]:
    root = Path(root).resolve()
    levels = _levels(root)
    series = _series(root)
    candidates: list[dict] = []
    notes: list[str] = []
    for (sensor, fact), records in sorted(series.items()):
        spec = f"{sensor}.{fact}"
        level = levels.get((sensor, fact), "diagnostic")
        values = _numeric(records)
        if len(values) < MIN_OBSERVATIONS:
            continue
        if level == "diagnostic":
            # diagnostic → только freshness/liveness (дельта G2).
            continue
        lo, hi = min(values), max(values)
        if lo == hi:
            candidates.append(make_candidate(
                "observed", "bounds", {"fact": spec, "min": lo, "max": hi,
                                       "tolerance_source": {"decision": _SOURCE}},
                evidence=[f"evidence/facts/{sensor}.jsonl"], support=len(values),
            ))
        else:
            margin = (hi - lo) * MARGIN
            candidates.append(make_candidate(
                "observed", "bounds", {"fact": spec, "min": lo - margin, "max": hi + margin,
                                       "tolerance_source": {"decision": _SOURCE}},
                evidence=[f"evidence/facts/{sensor}.jsonl"], support=len(values),
            ))
    if not candidates:
        notes.append(
            f"наблюдений >= {MIN_OBSERVATIONS} нет ни по одному недекл. факту "
            "(история фактов короткая) — кандидаты из наблюдений не выведены"
        )
    return candidates, notes


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="G2: кандидаты из наблюдений")
    parser.add_argument("--root", default=".")
    parser.add_argument("--json", dest="json_path", default=None)
    args = parser.parse_args(argv)
    candidates, notes = generate(args.root)
    print(json.dumps({"source": "observed", "candidates": len(candidates), "notes": notes},
                     ensure_ascii=False, indent=1))
    if args.json_path:
        Path(args.json_path).write_text(json.dumps(candidates, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
