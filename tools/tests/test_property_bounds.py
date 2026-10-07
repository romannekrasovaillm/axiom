"""Selftest шаблона ``bounds`` (ADR-039, дельта P2)."""

from tools.properties.selftest import selftest_one


def test_bounds_selftest():
    checks = selftest_one("bounds")
    assert checks and all(passed for _, passed in checks), checks


def test_bounds_boundary_is_not_red():
    # Граница вплотную (значение == min) — не нарушение.
    checks = dict(selftest_one("bounds"))
    assert checks["граница вплотную не краснеет"] is True
