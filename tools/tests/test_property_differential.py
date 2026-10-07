"""Selftest шаблона ``differential`` (ADR-039, дельта P2)."""

from tools.properties.selftest import selftest_one


def test_differential_selftest():
    checks = selftest_one("differential")
    assert checks and all(passed for _, passed in checks), checks
