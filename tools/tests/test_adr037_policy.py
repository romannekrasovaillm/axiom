"""ADR-037 дельта E: unverified не открывает расход (roofline + preflight)."""

from __future__ import annotations

import json
from pathlib import Path

from tools import check_budget_gate as budget
from tools import check_claims
from tools import check_performance_roofline as roofline

ROOT = Path(__file__).resolve().parents[2]


def test_require_verified_neutral_exit_3(tmp_path):
    pins = tmp_path / "pins.json"
    pins.write_text(json.dumps([{"run": "r", "kpi_tok_s_baseline": 86, "threshold": 800}]))
    code, report = roofline.run_check("r", tmp_path / "absent.jsonl", pins_path=pins, require_verified=True)
    assert code == roofline.EXIT_UNVERIFIED == 3
    assert report["verdict_class"] == "unverified"


def test_require_verified_no_data_exit_1(tmp_path):
    pins = tmp_path / "pins.json"
    pins.write_text(json.dumps([{"run": "r", "kpi_tok_s_baseline": 86, "threshold": 800}]))
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    code, report = roofline.run_check("r", empty, pins_path=pins, require_verified=True)
    assert code == roofline.EXIT_FAIL
    assert report["verdict_class"] == "no-data"


def test_require_verified_facts_divergence(tmp_path):
    pins = tmp_path / "pins.json"
    pins.write_text(json.dumps([{"run": "r", "kpi_tok_s_baseline": 86, "threshold": 800}]))
    metrics = tmp_path / "m.jsonl"
    metrics.write_text("".join(json.dumps({"step": i, "tok_s": 900.0}) + "\n" for i in range(20)))
    from tools.sensors.fact import write_fact
    from tools.sensors.subject import build_subject

    subject = build_subject(repo_root=tmp_path, git_sha="a" * 40, dirty=False, run_ref="r", device="cpu")
    write_fact("S-012", "tok_s_median_window", 400.0, unit="tok_s", quality="wrapped",
               method="f", subject=subject, out_dir=tmp_path)
    code, report = roofline.run_check("r", metrics, pins_path=pins, facts_dir=tmp_path)
    assert report.get("tok_s_facts") == 400.0
    assert report.get("source_divergence") is True


def test_preflight_refuses_when_unverified(tmp_path):
    # Смета vast.ai без подтверждённых утверждений → отказ.
    code = budget.cmd_preflight(ROOT, "pretreain-l3")
    assert code == 1


def test_preflight_refuses_without_estimate(tmp_path):
    (tmp_path / "evidence" / "budget").mkdir(parents=True)
    assert budget.cmd_preflight(tmp_path, "nope") == 1


def test_vast_estimate_requires_verdicts_field():
    errs = budget.validate_estimate(
        {
            "run_ref": "x", "gpu_type": "g", "gpu_hours_estimate": 1.0,
            "usd_estimate": 1.0, "limit_usd": 2.0,
            "budget_method": "6*N*D recompute", "stop_rule": "s", "created_at": "2026-09-30",
            "provider": "vast.ai",
        },
        "x",
    )
    assert any("requires_verdicts" in e for e in errs)


def test_repo_estimate_structural_verify_passes():
    dates = budget.run_dates(ROOT)
    errs, status = budget.verify_run(ROOT, "pretreain-l3", dates)
    assert errs == [], errs


def test_check_claims_combined_verify_and_bindings_green():
    assert check_claims.main(["--verify", "--mode", "config-bindings", "--root", str(ROOT)]) == 0
    assert check_claims.main(["--evaluate", "--no-history", "--fail-on", "fail", "--root", str(ROOT)]) == 0
