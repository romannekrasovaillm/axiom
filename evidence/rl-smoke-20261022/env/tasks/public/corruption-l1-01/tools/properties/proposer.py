"""G5 — предложения языковой модели: только интерфейс и файловый бэкенд (ADR-039).

Сетевых вызовов и встроенного клиента модели нет: бэкенд читает JSON-предложения,
подготовленные вне репозитория, и превращает их в кандидатов ``source: llm``.
Языковая модель — генератор кандидатов, вне пути вердикта (AD-2); граф импортов
проверяется тестом по образцу C-039.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Protocol

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.properties.candidates import make_candidate  # noqa: E402


class ProposerBackend(Protocol):
    """Источник JSON-предложений модели: список ``{property, params, evidence?}``."""

    def load(self) -> list[dict]:  # pragma: no cover - протокол
        ...


class FileBackend:
    """Читает предложения из JSON-файла, подготовленного вне репозитория."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> list[dict]:
        if not self.path.is_file():
            raise FileNotFoundError(f"нет файла предложений: {self.path}")
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("proposals")
        if not isinstance(data, list):
            raise ValueError(f"{self.path}: ожидался список предложений")
        return [p for p in data if isinstance(p, dict)]


def propose(backend: ProposerBackend) -> list[dict]:
    """Превращает предложения бэкенда в кандидатов ``source: llm`` (без приёмки)."""
    candidates: list[dict] = []
    for proposal in backend.load():
        prop = proposal.get("property")
        if not prop:
            continue
        candidates.append(make_candidate(
            "llm", str(prop), proposal.get("params") or {},
            evidence=list(proposal.get("evidence") or []),
            reason="предложение языковой модели — принимает только архитектор",
        ))
    return candidates


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="G5: предложения LLM (файловый бэкенд)")
    parser.add_argument("proposals", help="JSON-файл предложений")
    parser.add_argument("--json", dest="json_path", default=None)
    args = parser.parse_args(argv)
    candidates = propose(FileBackend(args.proposals))
    payload = {"source": "llm", "candidates": len(candidates),
               "notes": ["сетевых вызовов нет; предложения читаются из файла"]}
    print(json.dumps(payload, ensure_ascii=False, indent=1))
    if args.json_path:
        Path(args.json_path).write_text(json.dumps(candidates, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
