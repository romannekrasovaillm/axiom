"""S-011 disk_usage — тесты датчика."""

from tools.sensors import disk_usage


def test_selftest_green():
    assert disk_usage.run_selftest() == 0


def test_du_missing_and_present(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    (d / "f").write_bytes(b"x" * 1024)
    size, exists = disk_usage.du_bytes(tmp_path)
    assert exists and size >= 1024
    missing, m_exists = disk_usage.du_bytes(tmp_path / "nope")
    assert missing == 0 and not m_exists


def test_measure_keys(tmp_path):
    written = disk_usage.measure(out_dir=tmp_path)
    assert set(written) == {"pytest_tmp_bytes", "env_workspace_bytes", "staging_bytes"}
    for rec in written.values():
        assert rec["value"] >= 0
