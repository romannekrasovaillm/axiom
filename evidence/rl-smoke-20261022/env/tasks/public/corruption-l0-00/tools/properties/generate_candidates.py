"""G6 — прогон генераторов и сборка реестра кандидатов (ADR-039, дельта G6).

    python3 -m tools.properties.generate_candidates [--write] [--proposals FILE]

Запускает G1–G4 (и G5 при `--proposals`), сливает кандидатов по отпечатку `fp`
в `model/candidates.yaml` со статусом `proposed`. Исполнитель кандидатов не
принимает: `accepted` ставит только архитектор.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.properties import candidates as store
from tools.properties import (
    generate_declared,
    generate_incidents,
    generate_observed,
    generate_telemetry,
)


def collect(root: Path, proposals: str | None = None) -> tuple[list[dict], dict]:
    all_candidates: list[dict] = []
    report: dict = {}
    for name, fn in (
        ("declared", generate_declared.generate),
        ("observed", generate_observed.generate),
        ("telemetry", generate_telemetry.generate),
        ("incident", generate_incidents.generate),
    ):
        cands, notes = fn(root)
        all_candidates.extend(cands)
        report[name] = {"candidates": len(cands), "notes": notes}
    if proposals:
        from tools.properties.proposer import FileBackend, propose

        cands = propose(FileBackend(proposals))
        all_candidates.extend(cands)
        report["llm"] = {"candidates": len(cands), "notes": []}
    else:
        report["llm"] = {"candidates": 0, "notes": ["предложения не переданы (--proposals)"]}
    return all_candidates, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="G6: кандидаты из деклараций/наблюдений/телеметрии/инцидентов")
    parser.add_argument("--root", default=".")
    parser.add_argument("--proposals", default=None, help="JSON-предложения LLM (G5)")
    parser.add_argument("--write", action="store_true", help="записать model/candidates.yaml")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    incoming, report = collect(root, args.proposals)
    existing = store.load_candidates(root)
    merged, added, skipped = store.merge(existing, incoming)
    if args.write:
        store.save_candidates(root, merged)
    report["merged"] = {"existing": len(existing), "incoming": len(incoming),
                        "added": added, "skipped_duplicates": skipped, "total": len(merged)}
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
