"""Единый preflight открывающих гейтов (ADR-038, дельта K)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import check_claims as cc
from tools import preflight as pf
from tools import stage_preflight

ROOT = Path(__file__).resolve().parents[2]


def test_gates_registry_has_expected_gates():
    gates = {g["gate"] for g in pf.load_gates(ROOT)}
    assert {"rental-pretrain-l3", "sft-start", "rl-start", "hf-publish"} <= gates


def test_unknown_gate_raises(tmp_path):
    with pytest.raises(pf.GateError):
        pf.find_gate(ROOT, "нет-такого-гейта")


def test_rental_gate_refused_with_blockers():
    code, report = pf.preflight(ROOT, "rental-pretrain-l3")
    assert code == 1
    assert report["verdict"] == "refused"
    assert report["blockers"], "KPI не подтверждён — ожидался отказ с причинами"


def test_hf_publish_gate_refused_with_reasons():
    code, report = pf.preflight(ROOT, "hf-publish")
    assert code == 1
    assert report["blockers"]
    assert any("карточ" in r["reason"] or "уnverified" in r["reason"] or "факт" in r["reason"] for r in report["blockers"])


def test_preflight_report_written_as_fact():
    from tools.sensors.fact import read_latest

    record = read_latest("S-033", "gate_preflight")
    assert record is not None and record["status"] == "ok"
    assert record["value"]["gate"] in {"rental-pretrain-l3", "hf-publish"}


def test_override_allows_continue_but_records():
    code, report = pf.preflight(ROOT, "sft-start", override="сессия владельца")
    assert code == 0
    assert report["verdict"] == "overridden"
    assert report["override"] == "сессия владельца"


def test_k5_fail_open_guard_requires_verified():
    gates = pf.load_gates(ROOT)
    rental = next(g for g in gates if g["gate"] == "rental-pretrain-l3")
    guard_reqs = [r for r in rental["requires"] if "guard" in r]
    assert guard_reqs
    assert all(r.get("require_verified") for r in guard_reqs)


def test_verify_layer_detects_fail_open_without_verified(tmp_path):
    # Мутант: гейт со стражем fail-open без require_verified → K5 краснеет.
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "opening-gates.yaml").write_text(
        "- gate: demo\n  opens: money\n  requires:\n    - {guard: performance-roofline, run: r}\n"
        "  invocation: x\n",
        encoding="utf-8",
    )
    errors = cc._verify_gates(tmp_path)
    assert any("fail-open" in e and "performance-roofline" in e for e in errors)


def test_stage_preflight_enforce_no_preflight_is_noop():
    assert stage_preflight.enforce("sft-start", None, no_preflight=True) == 0


def test_stage_preflight_enforce_refuses_without_override():
    assert stage_preflight.enforce("sft-start", None) != 0


def test_stage_preflight_enforce_override_continues():
    assert stage_preflight.enforce("sft-start", "аварийный обход") == 0


def test_stage_runners_expose_preflight_flags():
    import sys

    if str(ROOT / "tools") not in sys.path:
        sys.path.insert(0, str(ROOT / "tools"))
    import run_sft_smoke as sft
    import run_rl_smoke as rl

    sft_args = sft.parse_args(["--no-preflight"])
    assert sft_args.preflight_gate == "sft-start"
    rl_args = rl.parse_args(["--no-preflight"])
    assert rl_args.preflight_gate == "rl-start"


def test_preflight_json_output(tmp_path):
    out = tmp_path / "report.json"
    code = pf.main(["rental-pretrain-l3", "--root", str(ROOT), "--json", str(out)])
    assert code == 1
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["gate"] == "rental-pretrain-l3"
