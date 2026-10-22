"""G1 — генератор кандидатов из деклараций (ADR-039, дельта G1).

Источники: `net/config.json` (declared_equals_actual с S-001), манифесты корпуса
(conservation/identity с S-005), `UPLOAD_MANIFEST` публикаций (identity),
`evidence/budget/*` (bounds с расходом S-022), `evidence/kpi-pins.json` (bounds).
Кандидат создаётся только если датчик нужного факта объявлен в
`model/sensors.yaml`; иначе источник попадает в отчёт «нет датчика».
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.properties.candidates import make_candidate  # noqa: E402


def _sensor_facts(root: Path) -> dict[str, set[str]]:
    from tools.miniyaml import load_file

    data = load_file(root / "model" / "sensors.yaml")
    out: dict[str, set[str]] = {}
    for s in data if isinstance(data, list) else []:
        if isinstance(s, dict) and s.get("id"):
            out[str(s["id"])] = set(s.get("facts") or [])
    return out


def generate(root: str | Path) -> tuple[list[dict], list[str]]:
    root = Path(root).resolve()
    facts = _sensor_facts(root)
    candidates: list[dict] = []
    notes: list[str] = []

    def has(sensor: str, fact: str) -> bool:
        return fact in facts.get(sensor, set())

    # 1. net/config.json → declared_equals_actual против S-001.
    config_path = root / "net" / "config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            config = {}
        for fact_name in sorted(facts.get("S-001", set())):
            if fact_name not in config:
                continue
            candidates.append(make_candidate(
                "declared", "declared_equals_actual",
                {"declared": {"file": "net/config.json", "key": fact_name},
                 "actual": {"fact": f"S-001.{fact_name}"}},
                evidence=["net/config.json"],
            ))
    else:
        notes.append("нет файла net/config.json — декларации конфига не собраны")

    # 2. evidence/kpi-pins.json → bounds (порог ток/с).
    pins_path = root / "evidence" / "kpi-pins.json"
    if pins_path.is_file():
        try:
            pins = json.loads(pins_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pins = {}
        for pin in (pins.get("pins") or []):
            threshold = pin.get("threshold")
            if threshold is None:
                continue
            if has("S-012", "tok_s_median_window"):
                candidates.append(make_candidate(
                    "declared", "bounds",
                    {"fact": "S-012.tok_s_median_window", "min": threshold,
                     "tolerance_source": {"decision": "evidence/kpi-pins.json"}},
                    evidence=["evidence/kpi-pins.json", f"run={pin.get('run')}"],
                ))
            else:
                notes.append("нет датчика S-012.tok_s_median_window для KPI-пина")

    # 3. evidence/budget/* → bounds (расход S-022 против лимита сметы).
    budget_dir = root / "evidence" / "budget"
    if budget_dir.is_dir():
        for path in sorted(budget_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            limit = data.get("limit_usd")
            if limit is None:
                continue
            if has("S-022", "spend_usd"):
                candidates.append(make_candidate(
                    "declared", "bounds",
                    {"fact": "S-022.spend_usd", "max": limit},
                    evidence=[path.relative_to(root).as_posix()],
                ))
            else:
                notes.append("нет датчика S-022.spend_usd для сметы")

    # 4. Манифесты корпуса (S-005) — источники вне git (C-032).
    for sensor, prop in (("S-005", "conservation"), ("S-005", "identity")):
        if facts.get(sensor):
            notes.append(
                f"{prop} с {sensor}: манифест корпуса вне git (C-032) — кандидат не построен "
                "(нет источника в контуре)"
            )

    # 5. UPLOAD_MANIFEST публикаций — файла-манифеста в репозитории нет.
    notes.append("UPLOAD_MANIFEST публикаций: манифест-файл в кейсе отсутствует — identity не построен")

    return candidates, notes


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="G1: кандидаты из деклараций")
    parser.add_argument("--root", default=".")
    parser.add_argument("--json", dest="json_path", default=None)
    args = parser.parse_args(argv)
    candidates, notes = generate(args.root)
    print(json.dumps({"source": "declared", "candidates": len(candidates), "notes": notes},
                     ensure_ascii=False, indent=1))
    if args.json_path:
        Path(args.json_path).write_text(json.dumps(candidates, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
