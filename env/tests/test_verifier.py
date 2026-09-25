"""§11(2): верификатор парсит FitnessReport JSON и отличает error от warn."""

from __future__ import annotations

from env.verifier import _run_fitness, _split_issues, verify

HASH = "0" * 64


def spec(kind: str = "restore-gates") -> dict:
    return {
        "id": "probe",
        "source": "corruption",
        "objective": {"kind": kind, "tests_cmd": "true"},
        "verifier": {"constraints": "CONSTRAINTS.yaml", "spine": True, "trace": True,
                     "hidden_constraints_sha256": HASH},
        "max_tokens": 131072,
    }


def test_split_issues_distinguishes_error_warn():
    issues = [
        {"rule": "a", "file": "f1", "severity": "error", "line": 1, "message": "m"},
        {"rule": "b", "file": "f2", "severity": "warn", "line": 2, "message": "m"},
    ]
    errors, warns = _split_issues(issues)
    assert [e["rule"] for e in errors] == ["a"]
    assert [w["rule"] for w in warns] == ["b"]


def test_clean_case_passes(clean_snapshot, arch_ml):
    v = verify(spec(), clean_snapshot, bin=arch_ml)
    assert v.passed
    assert not v.violations
    assert v.fitness.passed and v.spine.passed and v.trace.passed


def test_fitness_report_json_parsed(clean_snapshot, arch_ml):
    gate, report = _run_fitness(clean_snapshot, clean_snapshot / "CONSTRAINTS.yaml", arch_ml)
    assert gate.passed is True
    assert isinstance(report, dict)
    assert "passed" in report and "issues" in report and "summary" in report
    assert report["passed"] is True


def test_todo_is_error_not_warn(clean_snapshot, arch_ml):
    target = sorted((clean_snapshot / "model").glob("AD-*.md"))[0]
    target.write_text(target.read_text(encoding="utf-8") + "\nTODO: probe\n", encoding="utf-8")
    v = verify(spec(), clean_snapshot, bin=arch_ml)
    assert not v.passed
    assert not v.fitness.passed
    # C-007 (заглушки) — error, а не warn.
    assert any(it["severity"] == "error" for it in v.fitness.errors)


def test_warn_does_not_fail_gate(clean_snapshot, arch_ml):
    spine = clean_snapshot / "ARCHITECTURE-SPINE.md"
    spine.write_text(spine.read_text(encoding="utf-8") + "\n- version: latest\n", encoding="utf-8")
    v = verify(spec(), clean_snapshot, bin=arch_ml)
    assert v.spine.warns, "непиннутая версия — warn"
    assert v.passed, "warn не должен ломать гейт"
    assert not v.violations
