"""Selftest шаблона ``monotonic_trend`` (ADR-039, дельта P2)."""

from tools.properties.selftest import selftest_one


def test_monotonic_trend_selftest():
    checks = selftest_one("monotonic_trend")
    assert checks and all(passed for _, passed in checks), checks
