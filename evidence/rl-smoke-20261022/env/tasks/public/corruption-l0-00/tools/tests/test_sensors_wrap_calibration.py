"""S-015 wrap_calibration — тесты датчика."""

from tools.sensors import wrap_calibration as mod


def test_selftest_green():
    assert mod.run_selftest() == 0


def test_missing_report_unverified(tmp_path):
    written = mod.measure(tmp_path / "nope.json", out_dir=tmp_path)
    assert written["pass_rate"]["status"] == "unverified"


def test_reads_calibration_on_repo(tmp_path):
    written = mod.measure(out_dir=tmp_path)
    # На репозитории есть evidence/track2-stage-a-*/calibration.json.
    assert written["pass_rate"]["value"] is not None
    assert "aggregate" in written["pass_rate"]["value"]
