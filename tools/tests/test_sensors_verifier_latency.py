"""S-008 verifier_latency — тесты датчика."""

from tools.sensors import verifier_latency


def test_selftest_green():
    assert verifier_latency.run_selftest() == 0


def test_percentile():
    assert verifier_latency.percentile([1.0, 2.0, 3.0], 0.5) == 2.0
    assert verifier_latency.percentile([], 0.9) is None


def test_zero_samples_unverified(tmp_path):
    written = verifier_latency.measure(n=0, out_dir=tmp_path)
    assert written["verdict_seconds_p50"]["status"] == "unverified"
