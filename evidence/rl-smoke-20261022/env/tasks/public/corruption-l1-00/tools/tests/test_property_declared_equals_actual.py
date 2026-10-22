"""Selftest шаблона ``declared_equals_actual`` (ADR-039, дельта P2)."""

from tools.properties.selftest import selftest_one


def test_declared_equals_actual_selftest():
    checks = selftest_one("declared_equals_actual")
    assert checks and all(passed for _, passed in checks), checks
