"""Сквозная оценка прогона: вердикт + награда по финальному состоянию.

Связывает verifier (механические гейты), reward (§5) и пиннинг в один
детерминированный результат прогона. Используется CLI ``verify`` и ``calibrate``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import reward as reward_mod
from .verifier import Verdict, collect_violations, verify


@dataclass
class RunResult:
    verdict: Verdict
    reward: reward_mod.Reward
    base_violations: frozenset


def evaluate_run(
    task_spec: dict,
    base_ws: Path,
    final_ws: Path,
    spent_tokens: int,
    bin: Optional[str] = None,
    hidden_constraints: Optional[Path] = None,
) -> RunResult:
    """Вердикт и награда по финальному состоянию. Детерминированная функция состояния."""
    base_violations = collect_violations(base_ws, task_spec, bin=bin, hidden_constraints=hidden_constraints)
    verdict = verify(task_spec, final_ws, bin=bin, hidden_constraints=hidden_constraints)
    kind = task_spec.get("objective", {}).get("kind", "restore-gates")
    reward = reward_mod.compute(
        objective_kind=kind,
        passed=verdict.passed,
        tests_passed=verdict.tests_passed,
        base_violations=base_violations,
        final_violations=verdict.violations,
        spent_tokens=spent_tokens,
        max_tokens=int(task_spec.get("max_tokens", 0)),
    )
    return RunResult(verdict=verdict, reward=reward, base_violations=base_violations)
