"""Selftest шаблона ``metamorphic`` (ADR-039, дельта P2)."""

from tools.properties.selftest import selftest_one


def test_metamorphic_selftest():
    checks = selftest_one("metamorphic")
    assert checks and all(passed for _, passed in checks), checks
