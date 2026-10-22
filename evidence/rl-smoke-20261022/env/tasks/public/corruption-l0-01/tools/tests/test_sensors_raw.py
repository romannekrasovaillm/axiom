"""Происхождение факта до сырья (ADR-038, дельта F)."""

from __future__ import annotations

import json

from tools.sensors import raw as raw_mod
from tools.sensors.fact import write_fact
from tools.sensors.subject import build_subject


def test_raw_selftest_green():
    assert raw_mod.run_selftest() == 0


def test_selector_parsing():
    assert raw_mod.parse_selector("") == ("file", None)
    assert raw_mod.parse_selector("lines 3-7") == ("lines", (3, 7))
    assert raw_mod.parse_selector("key a.b.c") == ("key", "a.b.c")
    assert raw_mod.parse_selector("нечто иное")[0] == "unknown"


def test_fingerprint_lines_and_key(tmp_path):
    f = tmp_path / "data.jsonl"
    f.write_text("a\nb\nc\nd\ne\n", encoding="utf-8")
    assert raw_mod.fingerprint(f, "lines 2-4") == raw_mod.fingerprint(f, "lines 2-4")
    assert raw_mod.fingerprint(f, "") != raw_mod.fingerprint(f, "lines 2-4")
    j = tmp_path / "c.json"
    j.write_text(json.dumps({"m": {"v": 5}}), encoding="utf-8")
    assert len(raw_mod.fingerprint(j, "key m.v")) == 64
    assert raw_mod.fingerprint(j, "key m.missing") is None


def test_probe_raw_pass_and_history_tamper(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    raw_file = repo / "m.jsonl"
    raw_file.write_text("".join(json.dumps({"i": i, "v": 1 + i}) + "\n" for i in range(20)), encoding="utf-8")
    out = tmp_path / "facts"
    subject = build_subject(repo_root=repo, git_sha="a" * 40, dirty=False, device="cpu")
    sha = raw_mod.fingerprint(raw_file, "lines 1-20")
    write_fact("S-930", "v", 10, unit="count", quality="measured", method="t", subject=subject,
               out_dir=out, raw_ref={"path": "m.jsonl", "sha256": sha, "selector": "lines 1-20"})
    sensor = {"id": "S-930", "raw": {"path": "m.jsonl", "format": "jsonl", "retention_days": 90, "in_git": True}}
    assert raw_mod.probe_raw_sensor(sensor, out_dir=out, repo_root=repo)["verdict"] == "pass"
    raw_file.write_text(raw_file.read_text(encoding="utf-8").replace('"v": 1', '"v": 9'), encoding="utf-8")
    row = raw_mod.probe_raw_sensor(sensor, out_dir=out, repo_root=repo)
    assert row["verdict"] == "fail"
    assert "подмена истории" in row["reason"]


def test_probe_raw_external_unavailable_is_unverified(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    sensor = {"id": "S-931", "raw": {"path": "/нет/такого.jsonl", "format": "jsonl", "in_git": False}}
    row = raw_mod.probe_raw_sensor(sensor, out_dir=tmp_path, repo_root=repo)
    assert row["verdict"] == "unverified"


def test_probe_cli_raw_on_repo_has_no_failures():
    from tools.sensors import probe

    assert probe.main(["--raw"]) == 0


def test_raw_null_sensor_reports_not_applicable(tmp_path):
    row = raw_mod.probe_raw_sensor({"id": "S-003", "raw": {"note": "арифметика"}}, out_dir=tmp_path, repo_root=tmp_path)
    assert row["verdict"] == "unverified"
    assert "не применим" in row["reason"]
