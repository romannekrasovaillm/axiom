"""Selftest шаблона ``liveness`` (ADR-039, дельта P2)."""

from tools.properties.selftest import selftest_one


def test_liveness_selftest():
    checks = selftest_one("liveness")
    assert checks and all(passed for _, passed in checks), checks
