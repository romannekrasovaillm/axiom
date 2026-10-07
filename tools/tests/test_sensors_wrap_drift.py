"""S-019 wrap_drift — тесты датчика."""

from tools.sensors import wrap_drift as mod


def test_selftest_green():
    assert mod.run_selftest() == 0


def test_missing_report_unverified(tmp_path):
    written = mod.measure(tmp_path / "none.json", out_dir=tmp_path)
    assert written["bitwise_identical"]["status"] == "unverified"


def test_repo_drift_report(tmp_path):
    written = mod.measure(out_dir=tmp_path)
    assert written["bitwise_identical"]["value"] is True
