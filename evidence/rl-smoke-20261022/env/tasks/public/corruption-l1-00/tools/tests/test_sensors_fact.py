"""Контракт записи факта и пин предмета (ADR-037, дельта C1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.sensors import fact as fact_mod
from tools.sensors.subject import build_subject, device_kind, git_head, sha256_file


def _subject(tmp_path: Path) -> dict:
    return build_subject(repo_root=tmp_path, git_sha="b" * 40, dirty=False, device="cpu")


def test_write_fact_appends_contract_record(tmp_path):
    subject = _subject(tmp_path)
    rec = fact_mod.write_fact(
        "S-900", "demo", 7, unit="count", quality="measured",
        method="fixture", subject=subject, out_dir=tmp_path,
    )
    assert set(fact_mod.FACT_KEYS) <= set(rec)
    assert rec["sensor"] == "S-900"
    assert rec["fact"] == "demo"
    assert rec["value"] == 7
    assert rec["status"] == "ok"
    assert rec["prev_sha256"] is None
    lines = (tmp_path / "S-900.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["fact"] == "demo"


def test_chain_links_previous_line(tmp_path):
    subject = _subject(tmp_path)
    first = fact_mod.write_fact(
        "S-901", "demo", 1, unit="count", quality="measured",
        method="fixture", subject=subject, out_dir=tmp_path,
    )
    second = fact_mod.write_fact(
        "S-901", "demo", 2, unit="count", quality="measured",
        method="fixture", subject=subject, out_dir=tmp_path,
    )
    lines = (tmp_path / "S-901.jsonl").read_text(encoding="utf-8").splitlines(keepends=True)
    assert second["prev_sha256"] == fact_mod.line_sha256(lines[0])
    assert first["prev_sha256"] is None
    ok, reason = fact_mod.verify_chain("S-901", tmp_path)
    assert ok, reason


def test_chain_detects_history_edit(tmp_path):
    subject = _subject(tmp_path)
    for value in (1, 2):
        fact_mod.write_fact(
            "S-902", "demo", value, unit="count", quality="measured",
            method="fixture", subject=subject, out_dir=tmp_path,
        )
    path = tmp_path / "S-902.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    edited = json.loads(lines[-1])
    edited["prev_sha256"] = "0" * 64
    lines[-1] = json.dumps(edited, ensure_ascii=False, sort_keys=True) + "\n"
    path.write_text("".join(lines), encoding="utf-8")
    ok, reason = fact_mod.verify_chain("S-902", tmp_path)
    assert not ok
    assert "цепочка" in reason


def test_unverified_requires_reason(tmp_path):
    subject = _subject(tmp_path)
    with pytest.raises(fact_mod.FactError):
        fact_mod.write_fact(
            "S-903", "demo", None, unit="count", quality="measured",
            method="fixture", subject=subject, out_dir=tmp_path,
            status="unverified",
        )
    rec = fact_mod.write_fact(
        "S-903", "demo", None, unit="count", quality="measured",
        method="fixture", subject=subject, out_dir=tmp_path,
        status="unverified", note="нет GPU",
    )
    assert rec["status"] == "unverified"
    assert rec["value"] is None


def test_ok_requires_value(tmp_path):
    subject = _subject(tmp_path)
    with pytest.raises(fact_mod.FactError):
        fact_mod.write_fact(
            "S-904", "demo", None, unit="count", quality="measured",
            method="fixture", subject=subject, out_dir=tmp_path,
        )


def test_empty_subject_rejected(tmp_path):
    empty = {"git_sha": None, "git_dirty": None, "config_sha256": None}
    with pytest.raises(fact_mod.FactError):
        fact_mod.write_fact(
            "S-905", "demo", 1, unit="count", quality="measured",
            method="fixture", subject=empty, out_dir=tmp_path,
        )


def test_read_latest_filters_by_fact_and_subject(tmp_path):
    subject_a = build_subject(repo_root=tmp_path, git_sha="a" * 40, dirty=False, device="cpu")
    subject_b = build_subject(repo_root=tmp_path, git_sha="c" * 40, dirty=False, device="cpu")
    fact_mod.write_fact("S-906", "alpha", 1, unit="u", quality="measured",
                        method="m", subject=subject_a, out_dir=tmp_path)
    fact_mod.write_fact("S-906", "beta", 2, unit="u", quality="measured",
                        method="m", subject=subject_a, out_dir=tmp_path)
    fact_mod.write_fact("S-906", "alpha", 3, unit="u", quality="measured",
                        method="m", subject=subject_b, out_dir=tmp_path)
    latest_any = fact_mod.read_latest("S-906", "alpha", out_dir=tmp_path)
    assert latest_any["value"] == 3
    latest_a = fact_mod.read_latest(
        "S-906", "alpha", ["git_sha"], out_dir=tmp_path, subject=subject_a
    )
    assert latest_a["value"] == 1
    assert fact_mod.read_latest("S-906", "missing", out_dir=tmp_path) is None


def test_age_hours_parses_offset(tmp_path):
    assert fact_mod.age_hours("2026-10-07T10:00:00+03:00") is not None
    assert fact_mod.age_hours("не дата") is None


def test_sha256_file_and_missing(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"abc")
    assert sha256_file(p) == sha256_file(p)
    assert sha256_file(tmp_path / "nope") is None
    assert sha256_file(None) is None


def test_device_kind_override(tmp_path, monkeypatch):
    assert device_kind("cpu") == "cpu"
    monkeypatch.setenv("AXIOM_DEVICE_KIND", "gpu-fixture")
    assert device_kind() == "gpu-fixture"


def test_git_head_on_non_repo(tmp_path):
    assert git_head(tmp_path) is None
