"""G4 — генератор кандидатов из инцидентов (ADR-039, дельта G4).

Читает `evidence/incidents.yaml`, где у каждого инцидента проставлен
``violated_property`` (шаблон каталога), и предлагает экземпляр шаблона как
страж инцидента со ``guard: pending``. Параметры привязки — решение архитектора.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.properties.candidates import make_candidate  # noqa: E402


def _incidents(root: Path) -> list[dict]:
    from tools.miniyaml import load_file

    path = root / "evidence" / "incidents.yaml"
    if not path.is_file():
        return []
    data = load_file(path)
    return [i for i in data if isinstance(i, dict)] if isinstance(data, list) else []


def generate(root: str | Path) -> tuple[list[dict], list[str]]:
    root = Path(root).resolve()
    candidates: list[dict] = []
    notes: list[str] = []
    for inc in _incidents(root):
        prop = inc.get("violated_property")
        if not prop:
            notes.append(f"{inc.get('id')}: нет violated_property — кандидат не предложен")
            continue
        candidates.append(make_candidate(
            "incident", str(prop), {},
            evidence=[f"evidence/incidents.yaml#{inc.get('id')}"],
            reason="guard: pending — шаблон предложен как страж инцидента, привязка параметров ждёт архитектора",
        ))
    return candidates, notes


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="G4: кандидаты из инцидентов")
    parser.add_argument("--root", default=".")
    parser.add_argument("--json", dest="json_path", default=None)
    args = parser.parse_args(argv)
    candidates, notes = generate(args.root)
    print(json.dumps({"source": "incident", "candidates": len(candidates), "notes": notes},
                     ensure_ascii=False, indent=1))
    if args.json_path:
        Path(args.json_path).write_text(json.dumps(candidates, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
