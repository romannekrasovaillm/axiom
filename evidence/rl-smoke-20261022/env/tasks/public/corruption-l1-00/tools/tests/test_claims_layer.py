"""Слой ADR-038: наследование, ставки, триаж, очередь (дельты H/I/J)."""

from __future__ import annotations

from pathlib import Path

from tools import check_claims as cc

ROOT = Path(__file__).resolve().parents[2]

DOCUMENTARY_RULES = (
    "C-001", "C-002", "C-003", "C-004", "C-005", "C-006", "C-007", "C-008",
    "C-009", "C-010", "C-011", "C-012", "C-013",
    "C-015", "C-016", "C-017", "C-018", "C-019", "C-020", "C-021", "C-022",
    "C-023", "C-024", "C-025", "C-026", "C-027", "C-028", "C-029", "C-030", "C-031",
)


def test_verify_layer_green():
    assert cc.verify_layer(ROOT) == []


def test_lineage_covers_all_documentary_rules():
    lineage = cc.load_lineage(ROOT)
    assert lineage is not None
    rules = {entry["rule"] for entry in lineage}
    assert rules == set(DOCUMENTARY_RULES)
    kinds = cc.rule_ids_and_kinds(ROOT)
    for entry in lineage:
        if entry.get("successor") is not None:
            assert kinds.get(entry["successor"]) in ("behavioural", "structural")


def test_lineage_has_no_invented_successors():
    # Наследники — только реально существующие не-документарные правила.
    kinds = cc.rule_ids_and_kinds(ROOT)
    lineage = cc.load_lineage(ROOT)
    for entry in lineage:
        if entry.get("successor"):
            assert entry["successor"] in kinds
            assert kinds[entry["successor"]] != "documentary"
        elif entry.get("pending"):
            assert entry["pending"]["owner"] and entry["pending"]["since"] and entry["pending"]["reason"]
        elif entry.get("none"):
            assert entry["none"]["reason"].strip()


def test_every_claim_has_valid_stake():
    for claim in cc.load_claims(ROOT):
        assert claim.get("stake") in cc.STAKES, claim.get("id")


def test_high_stake_claims_have_sensor_or_owner():
    for claim in cc.load_claims(ROOT):
        if claim.get("stake") in ("money", "irreversible", "public"):
            assert claim.get("sensor") is not None or (
                isinstance(claim.get("pending"), dict)
                and claim["pending"].get("owner")
                and claim["pending"].get("since")
            ), claim.get("id")


def test_triage_covers_all_candidates_no_unresolved():
    scan = cc.scan_adr(ROOT)
    assert scan["unresolved_count"] == 0
    assert len(scan["candidates"]) >= 20


def test_triage_dispositions_valid():
    for entry in cc.load_triage(ROOT):
        assert entry["disposition"] in cc.DISPOSITIONS


def test_queue_sorted_by_stake_then_age():
    items = cc.queue(ROOT)
    assert items
    order = [cc._STAKE_ORDER[item["stake"]] for item in items]
    assert order == sorted(order)


def test_shares_have_numerator_and_denominator():
    payload = cc.shares(ROOT)
    for key in ("share_measurable_claims_with_sensor", "share_incidents_guarded"):
        block = payload[key]
        assert block["numerator"] <= block["denominator"] or block["denominator"] == 0
    lines = cc.format_shares(payload)
    assert lines[0].startswith("share_measurable_claims_with_sensor")
    assert "(" in lines[0] and "/" in lines[0]


def test_report_cli_prints_shares_first(capsys):
    code = cc.main(["--report", "--root", str(ROOT)])
    assert code == 0
    first = capsys.readouterr().out.splitlines()[0]
    assert first.startswith("share_measurable_claims_with_sensor")


def test_verify_layer_cli(capsys):
    assert cc.main(["--verify-layer", "--root", str(ROOT)]) == 0
    assert "verify-layer: OK" in capsys.readouterr().out
