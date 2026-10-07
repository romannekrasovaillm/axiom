"""Selftest шаблона ``determinism`` (ADR-039, дельта P2)."""

from tools.properties.selftest import selftest_one


def test_determinism_selftest():
    checks = selftest_one("determinism")
    assert checks and all(passed for _, passed in checks), checks
