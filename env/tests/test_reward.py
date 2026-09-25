"""§11(6): повторный вердикт по тому же финальному состоянию даёт тот же reward."""

from __future__ import annotations

from env.reward import compute
from env.run import evaluate_run

SIG = frozenset({("C-001", "docs/adr/x.md")})


def test_compute_is_deterministic():
    a = compute("restore-gates", True, True, SIG, frozenset(), 100, 131072)
    b = compute("restore-gates", True, True, SIG, frozenset(), 100, 131072)
    assert a.total == b.total
    assert a.to_manifest_dict() == b.to_manifest_dict()


def test_full_restore_reward():
    # pass, устранил все базовые нарушения: 1.0 + 0.5*1.0 = 1.5
    r = compute("restore-gates", True, True, SIG, frozenset(), 100, 131072)
    assert r.pass_component == 1.0
    assert r.soft_fraction == 1.0
    assert r.soft == 0.5
    assert r.new_violations_count == 0
    assert r.total == 1.5


def test_no_restore_reward():
    # не восстановил ничего, новых нет: 0 + 0.5*0 = 0
    r = compute("restore-gates", False, True, SIG, SIG, 100, 131072)
    assert r.soft_fraction == 0.0
    assert r.total == 0.0


def test_new_violation_penalty():
    # base пустая, финальная имеет одно новое: new=1, penalty=0.25*1/4
    base = frozenset()
    final = SIG
    r = compute("restore-gates", False, True, base, final, 100, 131072)
    assert r.new_violations_count == 1
    assert abs(r.new_violations_penalty - 0.25 * 1 / 4) < 1e-9
    # restore с пустой базой: soft_frac=1.0 → soft=0.5
    assert r.soft == 0.5
    assert abs(r.total - (0.0 + 0.5 - 0.25 * 1 / 4)) < 1e-9


def test_effort_penalty_triggers_over_budget():
    # spent_tokens > 2 * max_tokens → effort_penalty = 1.0
    under = compute("restore-gates", True, True, SIG, frozenset(), 131072 * 2, 131072)
    over = compute("restore-gates", True, True, SIG, frozenset(), 131072 * 2 + 1, 131072)
    assert under.effort_penalty == 0.0
    assert over.effort_penalty == 1.0
    assert over.total == under.total - 1.0


def test_keep_implement_soft_is_tests():
    # keep-gates-implement: soft = 1.0 при прохождении тестов, 0 иначе
    ok = compute("keep-gates-implement", True, True, frozenset(), frozenset(), 100, 131072)
    fail = compute("keep-gates-implement", False, False, frozenset(), frozenset(), 100, 131072)
    assert ok.soft_fraction == 1.0
    assert fail.soft_fraction == 0.0


def test_repeat_verdict_same_reward(clean_snapshot, arch_ml):
    """Интеграция: тот же финальный стейт → тот же reward (A5-свойство)."""
    spec = {
        "id": "probe",
        "source": "corruption",
        "objective": {"kind": "restore-gates", "tests_cmd": "true"},
        "verifier": {"constraints": "CONSTRAINTS.yaml", "spine": True, "trace": True,
                     "hidden_constraints_sha256": "0" * 64},
        "max_tokens": 131072,
    }
    # base — чистый снапшот (нет базовых нарушений)
    a = evaluate_run(spec, clean_snapshot, clean_snapshot, 100, bin=arch_ml)
    b = evaluate_run(spec, clean_snapshot, clean_snapshot, 100, bin=arch_ml)
    assert a.reward.total == b.reward.total
    assert a.verdict.passed == b.verdict.passed
