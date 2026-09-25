"""Детерминированная награда (§5). Все компоненты — функции артефактов прогона.

.. code-block:: text

    r = 1.0·[pass]
      + 0.5·soft
      − 0.25·min(new_violations, 4)/4
      − 1.0·[spent_tokens > τ·b₀]

``soft`` (restore)         = 1 − |viol(final)∩viol(base)| / |viol(base)|
``soft`` (keep-implement)  = доля пройденных задачных тестов (v1: 0/1)
``new_violations``         = |viol(final) \\ viol(base)|, сигнатуры (rule, file)
τ = 2, b₀ = max_tokens задачи.
"""

from __future__ import annotations

from dataclasses import dataclass

TAU = 2
SOFT_WEIGHT = 0.5
NEW_VIOLATION_WEIGHT = 0.25
NEW_VIOLATION_CAP = 4


@dataclass(frozen=True)
class Reward:
    passed: bool
    pass_component: float
    soft_fraction: float  # сырая доля soft (0..1)
    soft: float  # взвешенный вклад soft = 0.5·soft_fraction
    new_violations_count: int
    new_violations_penalty: float  # 0.25·min(count,4)/4
    effort_penalty: float  # 1.0·[spent_tokens > τ·b₀]
    total: float

    def to_manifest_dict(self) -> dict:
        return {
            "pass_component": self.pass_component,
            "soft": self.soft,
            "new_violations": self.new_violations_count,
            "effort_penalty": self.effort_penalty,
            "total": self.total,
        }


def soft_fraction_restore(base_violations: frozenset, final_violations: frozenset) -> float:
    """Доля устранённых базовых нарушений (restore)."""
    if not base_violations:
        return 1.0
    remaining = len(final_violations & base_violations)
    return 1.0 - remaining / len(base_violations)


def new_violations_count(base_violations: frozenset, final_violations: frozenset) -> int:
    return len(final_violations - base_violations)


def compute(
    objective_kind: str,
    passed: bool,
    tests_passed: bool,
    base_violations: frozenset,
    final_violations: frozenset,
    spent_tokens: int,
    max_tokens: int,
) -> Reward:
    """Чистая детерминированная функция награды от артефактов прогона."""
    if objective_kind == "keep-gates-implement":
        soft_frac = 1.0 if tests_passed else 0.0
    else:
        soft_frac = soft_fraction_restore(base_violations, final_violations)

    count = new_violations_count(base_violations, final_violations)
    penalty = NEW_VIOLATION_WEIGHT * min(count, NEW_VIOLATION_CAP) / NEW_VIOLATION_CAP
    effort = 1.0 if spent_tokens > TAU * max_tokens else 0.0

    pass_component = 1.0 if passed else 0.0
    soft = SOFT_WEIGHT * soft_frac
    total = pass_component + soft - penalty - effort
    return Reward(
        passed=passed,
        pass_component=pass_component,
        soft_fraction=soft_frac,
        soft=soft,
        new_violations_count=count,
        new_violations_penalty=penalty,
        effort_penalty=effort,
        total=total,
    )
