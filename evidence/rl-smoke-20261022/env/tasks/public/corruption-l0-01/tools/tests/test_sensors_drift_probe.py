"""S-026 drift_probe — тесты датчика."""

from tools.sensors import drift_probe as mod


def test_selftest_green():
    assert mod.run_selftest() == 0


def test_compare_hashes():
    assert mod.compare_hashes("ab" * 32, "ab" * 32) is True
    assert mod.compare_hashes("ab" * 32, "cd" * 32) is False
    assert mod.compare_hashes("", "") is False


def test_executor_refuses_device(tmp_path):
    written = mod.measure(allow_device=False, out_dir=tmp_path)
    assert written["short_run_bitwise_identical"]["status"] == "unverified"
