"""S-009 verifier_determinism — тесты датчика."""

import os

from tools.sensors import verifier_determinism


def test_selftest_green():
    assert verifier_determinism.run_selftest() == 0


def test_unavailable_binary_unverified(tmp_path, monkeypatch):
    monkeypatch.setenv("ENV_ARCH_ML_BIN", "/nonexistent/arch-ml")
    written = verifier_determinism.measure(out_dir=tmp_path)
    assert written["verdict_identical"]["status"] == "unverified"
