"""S-027 derived_mfu — тесты датчика."""

from tools.sensors import derived_mfu as mod
from tools.sensors.fact import write_fact
from tools.sensors.subject import build_subject


def test_selftest_green():
    assert mod.run_selftest() == 0


def test_requires_measured_peak(tmp_path):
    subject = build_subject(repo_root=tmp_path, git_sha="a" * 40, dirty=False, device="cpu")
    write_fact("S-002", "params_active", 500_000_000, unit="count", quality="measured",
               method="f", subject=subject, out_dir=tmp_path)
    write_fact("S-012", "tok_s_median_window", 800.0, unit="tok_s", quality="wrapped",
               method="f", subject=subject, out_dir=tmp_path)
    written = mod.measure(out_dir=tmp_path)
    assert written["mfu_measured"]["status"] == "unverified"
    assert "S-025" in written["mfu_measured"]["note"]
