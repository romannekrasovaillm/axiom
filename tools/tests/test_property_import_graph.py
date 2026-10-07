"""G5: ``proposer`` не входит в путь вердикта (по образцу C-039, ADR-039).

Граф импортов строится от корней пути вердикта (``check_claims``, ``preflight``,
шаблоны каталога, ``env/verifier.py``); модуль ``tools.properties.proposer`` в
замыкании быть не должен. Сетевых вызовов в ``proposer`` нет.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]

_VERDICT_ROOTS = (
    "tools/check_claims.py",
    "tools/preflight.py",
    "tools/properties/base.py",
    "tools/properties/bounds.py",
    "tools/properties/declared_equals_actual.py",
    "tools/properties/conservation.py",
    "tools/properties/identity.py",
    "tools/properties/determinism.py",
    "tools/properties/reversibility.py",
    "tools/properties/differential.py",
    "tools/properties/metamorphic.py",
    "tools/properties/monotonic_trend.py",
    "tools/properties/liveness.py",
    "tools/properties/safety.py",
    "tools/properties/freshness.py",
    "env/verifier.py",
)


def _module_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def _resolve(module: str) -> Path | None:
    rel = module.replace(".", "/")
    for cand in (_REPO / rel, _REPO / (rel + ".py")):
        if cand.is_file():
            return cand
        pkg = _REPO / rel / "__init__.py"
        if pkg.is_file():
            return pkg
    return None


def _closure(roots) -> set[str]:
    seen: set[str] = set()
    queue = list(roots)
    while queue:
        rel = queue.pop()
        if rel in seen or not (_REPO / rel).is_file():
            continue
        seen.add(rel)
        for module in _module_names(_REPO / rel):
            target = _resolve(module)
            if target is not None:
                queue.append(target.relative_to(_REPO).as_posix())
    return seen


def test_proposer_not_in_verdict_path():
    closure = _closure(_VERDICT_ROOTS)
    assert "tools/properties/proposer.py" not in closure


def test_proposer_has_no_network():
    source = (_REPO / "tools/properties/proposer.py").read_text(encoding="utf-8")
    for forbidden in ("import requests", "import socket", "import urllib", "from urllib", "http.client"):
        assert forbidden not in source
