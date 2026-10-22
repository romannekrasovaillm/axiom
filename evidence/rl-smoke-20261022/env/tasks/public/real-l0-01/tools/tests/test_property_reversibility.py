"""Selftest шаблона ``reversibility`` (ADR-039, дельта P2)."""

from tools.properties.selftest import selftest_one


def test_reversibility_selftest():
    checks = selftest_one("reversibility")
    assert checks and all(passed for _, passed in checks), checks
