"""S-029 derived_fit — тесты датчика."""

from tools.sensors import derived_fit as mod
from tools.sensors.fact import write_fact
from tools.sensors.subject import build_subject


def test_selftest_green():
    assert mod.run_selftest() == 0


def test_fit_uses_fact_capacity(tmp_path):
    subject = build_subject(repo_root=tmp_path, git_sha="a" * 40, dirty=False, device="cpu")
    write_fact("S-003", "replica_bytes_total", 8_000_000_000, unit="bytes",
               quality="measured", method="f", subject=subject, out_dir=tmp_path)
    written = mod.measure(device_bytes=80 * 1024**3, out_dir=tmp_path)
    assert written["replica_fits_device"]["value"] is True


def test_no_capacity_unverified(tmp_path):
    subject = build_subject(repo_root=tmp_path, git_sha="a" * 40, dirty=False, device="cpu")
    write_fact("S-003", "replica_bytes_total", 8_000_000_000, unit="bytes",
               quality="measured", method="f", subject=subject, out_dir=tmp_path)
    written = mod.measure(out_dir=tmp_path)
    assert written["replica_fits_device"]["status"] == "unverified"
