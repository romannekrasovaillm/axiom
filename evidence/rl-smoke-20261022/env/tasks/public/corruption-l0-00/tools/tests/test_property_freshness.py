"""Selftest шаблона ``freshness`` (ADR-039, дельта P2)."""

from tools.properties.selftest import selftest_one


def test_freshness_selftest():
    checks = selftest_one("freshness")
    assert checks and all(passed for _, passed in checks), checks
