"""§11(5): calibrate.py прогоняется end-to-end на заглушке и пишет отчёт в evidence."""

from __future__ import annotations

from env import calibrate
from env.verifier import collect_violations


def test_calibrate_end_to_end(generated, arch_ml, tmp_path):
    evidence = tmp_path / "evidence"
    report = calibrate.calibrate(
        generated["out"], generated["case"], evidence,
        model_name="stub", model_seed=7, bin=arch_ml,
    )
    assert (evidence / "calibration-stub.json").exists()
    assert len(report["matrix"]["stub"]) >= 20
    pr = report["aggregate"]["pass_rate"]
    assert 0.10 <= pr <= 0.90, f"pass-rate {pr} вне коридора готовности [10%, 90%]"
    assert report["readiness"]["ready"] is True
    # пиннинг обязателен (AD-4)
    assert report["pinning"]["arch_ml_build"]
    assert len(report["pinning"]["constraints_sha256"]) == 64


def test_calibrate_deterministic(generated, arch_ml, tmp_path):
    r1 = calibrate.calibrate(generated["out"], generated["case"], tmp_path / "e1",
                             model_name="stub", model_seed=7, bin=arch_ml)
    r2 = calibrate.calibrate(generated["out"], generated["case"], tmp_path / "e2",
                             model_name="stub", model_seed=7, bin=arch_ml)
    cells1 = r1["matrix"]["stub"]
    cells2 = r2["matrix"]["stub"]
    assert [c["pass"] for c in cells1] == [c["pass"] for c in cells2]
    assert r1["aggregate"]["pass_rate"] == r2["aggregate"]["pass_rate"]


def test_hidden_constraints_pass_on_clean(generated, arch_ml, clean_snapshot):
    hidden = generated["out"] / "holdout" / "hidden_constraints.yaml"
    spec = {
        "id": "probe",
        "source": "corruption",
        "objective": {"kind": "restore-gates", "tests_cmd": "true"},
        "verifier": {"constraints": "CONSTRAINTS.yaml", "spine": True, "trace": True,
                     "hidden_constraints_sha256": "0" * 64},
        "max_tokens": 131072,
    }
    # скрытые правила не дают нарушений на чистом кейсе
    v = collect_violations(clean_snapshot, spec, bin=arch_ml, hidden_constraints=hidden)
    assert v == frozenset()
