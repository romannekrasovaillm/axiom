"""Тест доменного пакета data (ADR-038, дельта G3)."""

from __future__ import annotations

from tools.sensors.packs import discover_exporters, fixture_path
from tools.sensors.protocol import check_conformance

SUBJECT = {"git_sha": "a" * 40, "git_dirty": False, "device_kind": "cpu"}


def _by_id() -> dict:
    return {e.describe().id: e for e in discover_exporters(only=["data"])}


def test_data_exporters_conform():
    assert check_conformance(discover_exporters(only=["data"]), subject=SUBJECT) == []


def test_data001_fingerprint_and_count():
    exp = _by_id()["DATA-001"]
    facts = {f.fact: f for f in exp.collect(SUBJECT, input_path=fixture_path("data", "sample.jsonl"))}
    assert facts["field_count"].value == 3
    fp = facts["field_fingerprint"].value
    assert isinstance(fp, str) and len(fp) == 64
    # Тот же отпечаток как эталон → дрейфа нет.
    facts2 = {f.fact: f for f in exp.collect(
        SUBJECT, input_path=fixture_path("data", "sample.jsonl"), reference=fp)}
    assert facts2["drift_vs_reference"].value is False


def test_data001_drift_detected():
    exp = _by_id()["DATA-001"]
    facts = {f.fact: f for f in exp.collect(
        SUBJECT, input_path=fixture_path("data", "sample.jsonl"), reference="0" * 64)}
    assert facts["drift_vs_reference"].value is True


def test_data001_missing_source_unverified():
    exp = _by_id()["DATA-001"]
    facts = exp.collect(SUBJECT, input_path=fixture_path("data", "nope.jsonl"))
    assert all(f.status == "unverified" for f in facts)


def test_data002_dup_rate_uses_minhash():
    exp = _by_id()["DATA-002"]
    facts = {f.fact: f for f in exp.collect(SUBJECT, input_path=fixture_path("data", "dup.jsonl"))}
    assert facts["n_records"].value == 4
    assert facts["n_duplicates"].value == 2
    assert abs(facts["dup_rate"].value - 0.5) < 1e-6


def test_data002_distinct_names_are_not_duplicates():
    exp = _by_id()["DATA-002"]
    facts = {f.fact: f.value for f in exp.collect(
        SUBJECT, input_path=fixture_path("data", "sample.jsonl"), field="b")}
    assert facts["n_records"] == 2
    assert facts["n_duplicates"] == 0
