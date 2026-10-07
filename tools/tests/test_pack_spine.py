"""Доменный пакет spine и две доли как факты (ADR-038, дельта H)."""

from __future__ import annotations

from pathlib import Path

from tools import check_claims as cc
from tools.sensors.packs import discover_exporters
from tools.sensors.packs.spine import coverage, piles_trend
from tools.sensors.protocol import check_conformance

ROOT = Path(__file__).resolve().parents[2]
SUBJECT = {"git_sha": "a" * 40, "git_dirty": False, "device_kind": "cpu"}


def _by_id() -> dict:
    return {e.describe().id: e for e in discover_exporters(only=["spine"])}


def test_spine_exporters_conform():
    assert check_conformance(discover_exporters(only=["spine"]), subject=SUBJECT) == []


def test_s031_matches_check_claims_shares():
    values = coverage.measure(ROOT)
    expected = cc.shares(ROOT)
    assert values["share_measurable_claims_with_sensor"] == expected["share_measurable_claims_with_sensor"]["value"]
    assert values["share_incidents_guarded"] == expected["share_incidents_guarded"]["value"]
    assert values["measurable_claims_numerator"] == expected["share_measurable_claims_with_sensor"]["numerator"]
    assert values["measurable_claims_denominator"] == expected["share_measurable_claims_with_sensor"]["denominator"]


def test_s031_facts_are_derived_with_inputs():
    exp = _by_id()["S-031"]
    facts = exp.collect(SUBJECT, root=ROOT)
    assert {f.fact for f in facts} == set(exp.describe().facts)
    assert all(f.quality == "derived" for f in facts)
    assert all(f.inputs for f in facts), "derived-факт обязан нести inputs"


def test_s031_no_thresholds_only_shares():
    # Доли — наблюдение: значения в [0, 1], порогов в спецификации нет.
    values = coverage.measure(ROOT)
    for key in ("share_measurable_claims_with_sensor", "share_incidents_guarded"):
        assert 0.0 <= values[key] <= 1.0


def test_s032_weekly_series_aligned():
    data = piles_trend.measure(ROOT)
    assert data["n_weeks"] == len(data["weeks"])
    assert len(data["docs"]) == len(data["weeks"])
    assert len(data["behaviour"]) == len(data["weeks"])
    assert len(data["work"]) == len(data["weeks"])


def test_s032_collect_writes_five_facts():
    exp = _by_id()["S-032"]
    facts = {f.fact for f in exp.collect(SUBJECT, root=ROOT)}
    assert facts == {"weeks", "weekly_docs_share", "weekly_behaviour_share", "weekly_work_share", "n_weeks"}


def test_spine_facts_exist_and_pass_contract():
    from tools.sensors.registry import load_sensors, probe_sensor

    sensors = {s["id"]: s for s in load_sensors()}
    for sid in ("S-031", "S-032"):
        assert sid in sensors
        row = probe_sensor(sensors[sid])
        assert row["verdict"] == "pass", row["reason"]
