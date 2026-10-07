"""Selftest шаблона ``safety`` (ADR-039, дельта P2)."""

from tools.properties.selftest import selftest_one


def test_safety_selftest():
    checks = selftest_one("safety")
    assert checks and all(passed for _, passed in checks), checks
