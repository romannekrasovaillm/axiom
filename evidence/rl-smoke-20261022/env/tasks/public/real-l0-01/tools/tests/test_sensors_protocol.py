"""Протокол экспортёров и доменные пакеты (ADR-038, дельта G)."""

from __future__ import annotations

from tools.sensors.packs import discover_exporters, discover_packs
from tools.sensors.protocol import SensorSpec, check_conformance, collect_all
from tools.sensors import export_openmetrics as om

SUBJECT = {"git_sha": "a" * 40, "git_dirty": False, "device_kind": "cpu"}


def test_packs_discovered():
    names = {p.name for p in discover_packs()}
    assert {"service", "iac", "data"} <= names


def test_all_exporters_conform_to_protocol():
    exporters = discover_exporters()
    assert exporters, "не найдено ни одного экспортёра"
    assert all(isinstance(e.describe(), SensorSpec) for e in exporters)
    assert check_conformance(exporters, subject=SUBJECT) == []


def test_collect_on_missing_source_is_unverified_not_exception():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        for exporter in discover_exporters():
            facts = collect_all(exporter, SUBJECT, root=tmp)
            assert facts
            assert all(f.status in ("ok", "unverified") for f in facts)


def test_openmetrics_selftest_green():
    assert om.run_selftest() == 0


def test_openmetrics_render_roundtrip_real_facts():
    records = om.latest_records()
    assert records, "нет числовых фактов в evidence/facts"
    parsed = om.parse(om.render(records))
    assert parsed
    assert all({"sensor", "fact"} <= set(s["labels"]) for s in parsed)
