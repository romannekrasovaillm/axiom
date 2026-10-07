"""Аудит fail-open стражей S-034 (ADR-038, дельта K4)."""

from __future__ import annotations

from pathlib import Path

from tools.sensors.packs.spine import fail_open_audit
from tools.sensors.packs import discover_exporters

ROOT = Path(__file__).resolve().parents[2]
SUBJECT = {"git_sha": "a" * 40, "git_dirty": False, "device_kind": "cpu"}


def test_probe_table_has_one_line_per_guard():
    names = [name for name, *_ in fail_open_audit.PROBES]
    assert len(names) == len(set(names))
    # performance-roofline и gb10-single-load — известные случаи из решения.
    assert "performance-roofline" in names
    assert "gb10-single-load" in names


def test_s034_exporter_registered_and_conforms():
    exporters = {e.describe().id for e in discover_exporters(only=["spine"])}
    assert "S-034" in exporters


def test_s034_fact_records_known_fail_open_and_closed():
    from tools.sensors.fact import read_latest

    record = read_latest("S-034", "fail_open")
    assert record is not None and record["status"] == "ok"
    mapping = record["value"]
    # performance-roofline нейтрален без --require-verified (решение E-3.5).
    assert mapping["performance-roofline"] is True
    # check_gb10_single_load — fail-closed (NOT-VERIFIED, exit 1).
    assert mapping["gb10-single-load"] is False


def test_s034_outcomes_map_to_exit_codes():
    from tools.sensors.fact import read_latest

    outcomes = read_latest("S-034", "outcomes")["value"]
    assert outcomes["performance-roofline"]["exit_code"] == 0
    assert outcomes["gb10-single-load"]["exit_code"] not in (0, None)
    assert all("probe" in v for v in outcomes.values())


def test_collect_writes_four_facts_without_raising():
    # Полный аудит — тяжёлый; проверяем, что collect возвращает 4 факта и не падает
    # на недоступном входе (мутант: несуществующий корень → пробы absent).
    exp = {e.describe().id: e for e in discover_exporters(only=["spine"])}["S-034"]
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        facts = exp.collect(SUBJECT, root=tmp)
    assert {f.fact for f in facts} == set(exp.describe().facts)
    assert all(f.status == "ok" for f in facts)
