"""Тест доменного пакета service (ADR-038, дельта G3)."""

from __future__ import annotations

from pathlib import Path

from tools.sensors.packs import discover_exporters, fixture_path
from tools.sensors.protocol import SensorSpec, check_conformance

FIX = Path(__file__).resolve().parents[2] / "tools" / "sensors" / "packs" / "service" / "fixtures"
SUBJECT = {"git_sha": "a" * 40, "git_dirty": False, "device_kind": "cpu"}


def _by_id() -> dict:
    return {e.describe().id: e for e in discover_exporters(only=["service"])}


def test_service_pack_manifest_discovers_two_exporters():
    exporters = discover_exporters(only=["service"])
    assert len(exporters) == 2
    assert all(isinstance(e.describe(), SensorSpec) for e in exporters)


def test_service_exporters_conform():
    assert check_conformance(discover_exporters(only=["service"]), subject=SUBJECT) == []


def test_svc001_openapi_surface_values():
    exp = _by_id()["SVC-001"]
    facts = {f.fact: f for f in exp.collect(SUBJECT, input_path=FIX / "openapi.json")}
    assert facts["endpoint_count"].value == 4
    assert facts["auth_schemes_declared"].value == 1
    assert facts["endpoints_without_auth"].value == 2
    assert all(f.status == "ok" for f in facts.values())


def test_svc001_missing_source_is_unverified():
    exp = _by_id()["SVC-001"]
    facts = exp.collect(SUBJECT, input_path=FIX / "nope.json")
    assert facts and all(f.status == "unverified" for f in facts)


def test_svc002_prom_quantiles_and_error_share():
    exp = _by_id()["SVC-002"]
    facts = {f.fact: f for f in exp.collect(SUBJECT, input_path=FIX / "metrics.txt")}
    assert abs(facts["latency_p50_seconds"].value - 0.1375) < 1e-6
    assert abs(facts["latency_p99_seconds"].value - 1.0) < 1e-6
    assert abs(facts["error_share"].value - 0.05) < 1e-6


def test_svc002_empty_file_is_unverified():
    exp = _by_id()["SVC-002"]
    facts = exp.collect(SUBJECT, input_path=FIX / "empty.txt")
    assert all(f.status == "unverified" for f in facts)


def test_fixture_path_helper():
    assert fixture_path("service", "openapi.json").is_file()
