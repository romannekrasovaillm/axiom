"""S-028 derived_eta — тесты датчика."""

from tools.sensors import derived_eta as mod
from tools.sensors.fact import write_fact
from tools.sensors.subject import build_subject


def test_selftest_green():
    assert mod.run_selftest() == 0


def test_eta_formula(tmp_path):
    subject = build_subject(repo_root=tmp_path, git_sha="a" * 40, dirty=False, device="cpu")
    write_fact("S-012", "tok_s_median_window", 1000.0, unit="tok_s", quality="wrapped",
               method="f", subject=subject, out_dir=tmp_path)
    written = mod.measure(d_tokens=3_600_000, out_dir=tmp_path)
    assert written["eta_hours"]["value"] == 1.0


def test_missing_rate_unverified(tmp_path):
    written = mod.measure(out_dir=tmp_path)
    assert written["eta_hours"]["status"] == "unverified"
