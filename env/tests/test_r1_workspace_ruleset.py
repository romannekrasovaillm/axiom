"""R-1 (§7, ADR-016): вердикт среды исполняется на workspace-ruleset снапшота.

Тесты (б)–(г) п.1 handoff E-2.5: чистый кейс → ``verdict.passed``; порченный →
``not passed``; восстановленный по Damage-листу → ``passed``. Требуют бинаря
arch-ml (``arch_ml`` fixture → skip с явной причиной при отсутствии).
"""

from __future__ import annotations

from pathlib import Path

from env import corruption
from env.util import EMPTY_HIDDEN_SHA256
from env.verifier import verify

#: Спека restore-gates: задачных тестов нет (``true``), проверяются только гейты.
SPEC = {
    "id": "probe-r1",
    "source": "corruption",
    "objective": {"kind": "restore-gates", "tests_cmd": "true"},
    "verifier": {
        "constraints": "CONSTRAINTS.yaml",
        "spine": True,
        "trace": True,
        "hidden_constraints_sha256": EMPTY_HIDDEN_SHA256,
    },
    "max_tokens": 131072,
}


def _ruleset_error_rules(v) -> list[str]:
    """Правила error-находок fitness (для проверки отсутствия case-скоупа)."""
    return [it["rule"] for it in v.fitness.errors]


def test_clean_case_verdict_passes(clean_snapshot, arch_ml):
    """(б) чистый кейс на workspace-ruleset → вердикт зелёный.

    Критерий п.1: FitnessReport без error, case-скоуп стражи (C-037/38/40/41)
    в отчёте отсутствуют.
    """
    v = verify(SPEC, clean_snapshot, bin=arch_ml)
    assert _ruleset_error_rules(v) == [], _ruleset_error_rules(v)
    report_text = " ".join(str(it) for it in v.fitness.errors)
    for case_rule in ("C-037", "C-038", "C-040", "C-041"):
        assert case_rule not in report_text, case_rule
    assert v.fitness.passed
    assert v.passed, {
        "fitness": v.fitness.passed,
        "spine": v.spine.passed,
        "trace": v.trace.passed,
        "trace_errors": [it["rule"] for it in v.trace.errors][:5],
    }


def test_corrupted_case_verdict_fails(tmp_path, case_dir, arch_ml):
    """(в) порченный кейс → вердикт красный."""
    ws = tmp_path / "corrupted"
    damages = corruption.corrupt(case_dir, ws, seed=42, level="L1")
    assert damages
    v = verify(SPEC, ws, bin=arch_ml)
    assert not v.passed


def test_restored_case_verdict_passes(tmp_path, case_dir, arch_ml):
    """(г) восстановление по Damage-листу → вердикт зелёный."""
    ws = tmp_path / "restored"
    damages = corruption.corrupt(case_dir, ws, seed=42, level="L1")
    for d in damages:
        corruption.revert_damage(ws, case_dir, d)
    v = verify(SPEC, ws, bin=arch_ml)
    assert v.passed, {
        "fitness": v.fitness.passed,
        "spine": v.spine.passed,
        "trace": v.trace.passed,
        "trace_errors": [it["rule"] for it in v.trace.errors][:5],
    }
