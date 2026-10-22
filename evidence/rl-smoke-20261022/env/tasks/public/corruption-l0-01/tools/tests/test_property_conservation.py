"""Selftest шаблона ``conservation`` (ADR-039, дельта P2)."""

from tools.properties.selftest import selftest_one


def test_conservation_selftest():
    checks = selftest_one("conservation")
    assert checks and all(passed for _, passed in checks), checks
