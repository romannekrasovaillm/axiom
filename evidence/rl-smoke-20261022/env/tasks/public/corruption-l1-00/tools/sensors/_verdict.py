"""Общий вердикт-помощник датчиков S-007/S-008/S-009 (дельта C2).

Собирает Task Spec restore-gates, гоняет ``env.verifier.verify`` (бинарь
``arch-ml``) и канонизирует вердикт (относительные пути, сортировка) — так
датчик детерминизма сравнивает представление, а не абсолютный tmp-путь.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from env.util import EMPTY_HIDDEN_SHA256, sha256_file
from env.verifier import arch_ml_bin, arch_ml_available, verify

from ._common import REPO_ROOT


def make_spec(clean_root: Path) -> dict[str, Any]:
    return {
        "id": "sensor-probe",
        "source": "corruption",
        "prompt": "Восстанови архитектурные гейты кейса.",
        "objective": {"kind": "restore-gates", "tests_cmd": "true"},
        "verifier": {
            "constraints": "CONSTRAINTS.yaml",
            "spine": True,
            "trace": True,
            "hidden_constraints_sha256": EMPTY_HIDDEN_SHA256,
        },
        "gates_sha256": {
            "constraints": sha256_file(clean_root / "CONSTRAINTS.yaml"),
            "spine": sha256_file(clean_root / "ARCHITECTURE-SPINE.md"),
        },
        "max_tokens": 131072,
    }


def _rel(value: Any, workspace: Path) -> str:
    if not value or not isinstance(value, str):
        return ""
    try:
        p = Path(value)
        if p.is_absolute():
            return p.resolve().relative_to(Path(workspace).resolve()).as_posix() or "."
        return p.as_posix()
    except (ValueError, OSError):
        return Path(value).as_posix()


def canonical_verdict(verdict: Any, workspace: Path) -> str:
    """Канонический sha-payload вердикта: сортировка + относительные пути."""
    violations = []
    for item in getattr(verdict, "violations", []):
        if isinstance(item, (tuple, list)) and len(item) == 2:
            rule, path = str(item[0]), item[1]
        elif isinstance(item, dict):
            rule, path = str(item.get("rule", "")), item.get("file")
        else:
            rule, path = str(item), None
        violations.append((_rel(path, workspace), rule))
    violations.sort()
    payload = {
        "passed": bool(verdict.passed),
        "tests_passed": bool(verdict.tests_passed),
        "gates": {k: bool(v) for k, v in sorted(verdict.gates().items())},
        "violations": violations,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def run_verdict(workspace: Path, clean_root: Path, *, bin: Optional[str] = None):
    """Один вердикт (passed) рабочего каталога. Возвращает (verdict, payload)."""
    spec = make_spec(clean_root)
    verdict = verify(spec, workspace, bin=bin or arch_ml_bin(), base_ws=clean_root)
    return verdict, canonical_verdict(verdict, workspace)


def arch_ml_ready() -> bool:
    return arch_ml_available(arch_ml_bin())


def task_seed(level: str, index: int) -> int:
    import hashlib

    digest = hashlib.sha256(f"feasibility:{level}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


class _NoopVerifier:
    """Заглушка верификатора для ``WorkspaceTools``: починке нужен только edit_file."""

    def run_gates(self, root: Path) -> dict[str, Any]:  # pragma: no cover — не вызывается
        return {}


def prepare_task(clean: Path, ws: Path, level: str, index: int) -> list[Any]:
    """Снапшот + порча + эталонная починка четырьмя инструментами §13.

    Возвращает список применённых повреждений. Механику починки датчики S-007,
    S-008 и S-009 используют общую, чтобы «эталонная починка» была одной и той
    же процедурой, а не тремя разными.
    """
    import sys

    from env import corruption
    from env.sft_block import _insertion_edit, plan_repairable
    from env.util import copy_case_snapshot

    if str(REPO_ROOT / "tools") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "tools"))
    from rollout_harness import WorkspaceTools

    damages = plan_repairable(clean, task_seed(level, index), level)
    copy_case_snapshot(clean, ws)
    for damage in damages:
        corruption.apply_damage(ws, damage)
    tools = WorkspaceTools(ws, _NoopVerifier())
    for damage in damages:
        damaged_text = (ws / damage.file).read_text(encoding="utf-8")
        clean_text = (clean / damage.file).read_text(encoding="utf-8")
        old, new = _insertion_edit(damaged_text, clean_text)
        tools.edit_file(damage.file, old, new)
    return damages


__all__ = [
    "REPO_ROOT",
    "make_spec",
    "canonical_verdict",
    "run_verdict",
    "arch_ml_ready",
    "task_seed",
    "prepare_task",
]
