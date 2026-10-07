"""Уровни, прокси, окна, условия, история и флапание (ADR-038, дельты L/M)."""

from __future__ import annotations

import json
from pathlib import Path

from tools import check_claims as cc
from tools import preflight as pf
from tools.sensors.packs.ml import noise_baseline
from tools.sensors.packs.spine import flapping as flapping_mod

ROOT = Path(__file__).resolve().parents[2]
SUBJECT = {"git_sha": "a" * 40, "git_dirty": False, "device_kind": "cpu"}


def test_verify_layer_green_with_LM():
    assert cc.verify_layer(ROOT) == []


def test_L2_mutant_component_fact_in_gate_fails(tmp_path):
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "sensors.yaml").write_text(
        "- id: S-900\n  facts: [v]\n  level: component\n", encoding="utf-8"
    )
    (tmp_path / "model" / "claims.yaml").write_text(
        json.dumps([{"id": "CL-1", "sensor": "S-900", "fact": "v", "kind": "number", "rule": None}], ensure_ascii=False),
        encoding="utf-8",
    )
    (tmp_path / "model" / "opening-gates.yaml").write_text(
        "- gate: g\n  opens: money\n  requires:\n    - {claim: CL-1}\n  invocation: x\n", encoding="utf-8"
    )
    (tmp_path / "CONSTRAINTS.yaml").write_text("- id: C-001\n", encoding="utf-8")
    errors = cc._verify_levels(tmp_path)
    assert any("end_to_end" in e for e in errors)


def test_L3_component_claims_have_e2e_pair():
    claims = {str(c["id"]): c for c in cc.load_claims(ROOT)}
    for cid in ("CL-006", "CL-021"):
        assert claims[cid].get("e2e_pair") == "CL-001"
    assert cc._verify_e2e_pairs(ROOT) == []


def test_L3_proxy_divergence_reported():
    results = cc.evaluate(ROOT)
    lines = cc.proxy_divergences(results, ROOT)
    assert any("прокси-расхождение" in ln for ln in lines)


def test_M1_series_predicates_have_windows():
    claims = {str(c["id"]): c for c in cc.load_claims(ROOT)}
    for cid in ("CL-001", "CL-016", "CL-021"):
        predicate = claims[cid]["predicate"]
        assert predicate.get("window") and predicate.get("tolerance") is not None
        assert predicate.get("tolerance_source")
    assert cc._verify_predicate_windows(ROOT) == []


def test_M1_mutant_series_without_tolerance_fails(tmp_path):
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "sensors.yaml").write_text("- id: S-900\n  facts: [v]\n", encoding="utf-8")
    (tmp_path / "model" / "claims.yaml").write_text(
        json.dumps([{"id": "CL-1", "sensor": "S-900", "fact": "v", "series": True,
                     "predicate": {"op": ">=", "value": 1}}], ensure_ascii=False),
        encoding="utf-8",
    )
    errors = cc._verify_predicate_windows(tmp_path)
    assert any("window" in e for e in errors) and any("tolerance" in e for e in errors)


def test_M3_condition_unmet_gives_unverified():
    claim = next(c for c in cc.load_claims(ROOT) if c["id"] == "CL-001")
    result = cc.evaluate_claim(claim, ROOT, None)
    assert result["verdict"] == "unverified"
    assert "услов" in result["reason"]


def test_M4_history_and_flapping_fact_present():
    from tools.sensors.fact import read_latest

    history = read_latest("S-036", "claim_verdict_history")
    assert history is not None
    assert isinstance(history["value"], dict) and "claim" in history["value"]
    flapping = read_latest("S-037", "flapping")
    assert flapping is not None and flapping["status"] == "ok"


def test_flapping_measure_detects_switching(tmp_path):
    # Синтетическая история: исход скачет → flapping True.
    from tools.sensors.fact import write_fact
    from tools.sensors.subject import build_subject

    subject = build_subject(repo_root=tmp_path, git_sha="a" * 40, dirty=False, device="cpu")
    for i, verdict in enumerate(["pass", "fail", "pass", "fail", "pass", "fail"]):
        write_fact("S-036", "claim_verdict_history", {"claim": "CL-X", "verdict": verdict},
                   unit="", quality="measured", method="t", subject=subject, out_dir=tmp_path)
    data = flapping_mod.measure(tmp_path, window_n=10, threshold=3)
    assert data["flapping"]["CL-X"] is True
    assert data["switches"]["CL-X"] == 5


def test_preflight_flapping_marks_unverified(tmp_path, monkeypatch):
    claims = {"CL-1": {"id": "CL-1", "verdict": "pass", "reason": "ok"}}
    result = pf.evaluate_requirement({"claim": "CL-1"}, tmp_path, claims, flapping={"CL-1": True})
    assert result["verdict"] == "unverified"
    assert "флапа" in result["reason"]


def test_noise_baseline_unverified_without_series(tmp_path):
    data = noise_baseline.measure(tmp_path)
    assert data["cv_tok_s"] is None
    facts = {f.fact: f for f in noise_baseline.EXPORTER.collect(SUBJECT, out_dir=tmp_path)}
    assert facts["cv_tok_s"].status == "unverified"
    assert facts["window_n"].status == "ok"
