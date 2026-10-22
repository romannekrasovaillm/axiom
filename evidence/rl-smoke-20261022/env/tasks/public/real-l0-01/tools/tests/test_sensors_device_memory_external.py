"""S-021 device_memory_external — тесты датчика."""

from tools.sensors import device_memory_external as mod


def test_selftest_green():
    assert mod.run_selftest() == 0


def test_sum_processes(tmp_path):
    written = mod.measure("1, 512\n2, 256\n", out_dir=tmp_path)
    assert written["device_mem_used_mb"]["value"] == 768.0


def test_unified_memory_unverified(tmp_path):
    written = mod.measure("1, [Not supported]\n", out_dir=tmp_path)
    assert written["device_mem_used_mb"]["status"] == "unverified"
    assert "unified" in written["device_mem_used_mb"]["note"]
