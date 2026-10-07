"""Selftest шаблона ``identity`` (ADR-039, дельта P2)."""

from tools.properties.selftest import selftest_one


def test_identity_selftest():
    checks = selftest_one("identity")
    assert checks and all(passed for _, passed in checks), checks
