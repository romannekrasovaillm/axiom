"""S-025 peak_tflops_bench — тесты датчика."""

from tools.sensors import peak_tflops_bench as mod


def test_selftest_green():
    assert mod.run_selftest() == 0


def test_formula():
    assert mod.tflops_from_seconds(1024, 0.0) is None
    assert mod.tflops_from_seconds(1024, 1.0) == 2 * 1024**3 / 1e12


def test_executor_refuses_device(tmp_path):
    written = mod.measure(allow_device=False, out_dir=tmp_path)
    rec = written["measured_peak_tflops_bf16"]
    assert rec["value"] is None
    assert rec["status"] == "unverified"
    assert "AD-7" in rec["note"] or "окне" in rec["note"]
