"""check_claims — страж реестра утверждений и инцидентов (дельта A/B)."""

from __future__ import annotations

import json
from pathlib import Path

from tools import check_claims as cc

ROOT = Path(__file__).resolve().parents[2]


def test_selftest_green():
    assert cc.run_selftest() == 0


def test_config_bindings_green_on_repo():
    results = cc.config_bindings(ROOT)
    assert results, "нет config_binding-утверждений"
    assert all(r["verdict"] == "pass" for r in results)


def test_evaluate_classes_are_valid():
    results = cc.evaluate(ROOT)
    assert len(results) >= 15
    assert {r["verdict"] for r in results} <= {"pass", "fail", "unverified"}
    # KPI и MFU — не pass (числа не измерены/не сходятся): fail или unverified.
    kpi = next(r for r in results if r["id"] == "CL-001")
    assert kpi["verdict"] in ("fail", "unverified")
    mfu = next(r for r in results if r["id"] == "CL-016")
    assert mfu["verdict"] in ("fail", "unverified")


def test_no_claim_fails_on_repo():
    results = cc.evaluate(ROOT)
    assert [r["id"] for r in results if r["verdict"] == "fail"] == []


def test_incidents_verify_clean():
    sensors = {s["id"]: set(s.get("facts") or []) for s in cc.load_sensors(ROOT)}
    rules = cc.rule_ids_and_kinds(ROOT)
    assert cc._verify_incidents(ROOT, sensors, rules) == []


def test_report_shape():
    rep = cc.report(ROOT)
    assert rep["claims_total"] >= 15
    assert rep["incidents_total"] >= 7
    assert rep["incidents_guarded"] >= 3
    assert set(rep["verdicts"]) == {"pass", "fail", "unverified"}


def test_apply_predicate_ops():
    assert cc.apply_predicate({"op": ">=", "value": 800}, 800) is True
    assert cc.apply_predicate({"op": "between", "value": [0.1, 0.9]}, 0.5) is True
    assert cc.apply_predicate({"op": "approx", "value": 0.35, "tolerance": 0.1}, 0.4) is True
    assert cc.apply_predicate({"op": "==", "value": True}, False) is False
    assert cc.apply_predicate({"op": ">=", "value": 1}, None) is None
    assert cc.apply_predicate({"op": "between", "value": [0.1, 0.9], "path": "aggregate"}, {"aggregate": 0.0}) is False


def test_scan_adr_informational(tmp_path, capsys):
    import sys

    code = cc.main(["--scan-adr", "--root", str(ROOT)])
    assert code == 0
    out = capsys.readouterr().out
    assert "files" in out


def test_verify_reports_missing_rule_refs_in_sandbox_like_fixture(tmp_path):
    # Схема + anchor проверяются на синтетическом корне (без rules-а).
    root = tmp_path
    (root / "docs").mkdir()
    (root / "model").mkdir()
    anchor = "целевой порог приёмки не менее 800 ток/с"
    (root / "docs" / "src.md").write_text(anchor, encoding="utf-8")
    (root / "model" / "sensors.yaml").write_text(
        json.dumps([{"id": "S-001", "facts": ["num_kda_layers"]}], ensure_ascii=False),
        encoding="utf-8",
    )
    (root / "CONSTRAINTS.yaml").write_text("- id: C-049\n  kind: structural\n", encoding="utf-8")
    (root / "model" / "claims.yaml").write_text(
        json.dumps([{
            "id": "CL-1", "source": {"file": "docs/src.md", "anchor": anchor},
            "statement": "s", "kind": "number", "sensor": "S-001", "fact": "num_kda_layers",
            "predicate": {"op": "==", "value": 18}, "subject_match": [],
            "rule": "C-049", "binding": None,
        }], ensure_ascii=False),
        encoding="utf-8",
    )
    assert cc.verify(root, with_incidents=False) == []
