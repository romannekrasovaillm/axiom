"""Реестр датчиков и probe (ADR-036, дельта C1)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tools.sensors import registry as reg
from tools.sensors.fact import write_fact
from tools.sensors.subject import build_subject

FACTS_DIR = Path(__file__).resolve().parents[2] / "evidence" / "facts"


def _sensor(sid: str, **over) -> dict:
    base = {
        "id": sid,
        "name": f"test {sid}",
        "facts": ["v"],
        "context": "cpu",
        "producer": "pytest",
        "output": f"evidence/facts/{sid}.jsonl",
        "quality": "measured",
        "freshness_max_h": None,
        "zone": "line-2",
        "status": "active",
    }
    base.update(over)
    return base


def test_project_registry_is_valid():
    sensors = reg.load_sensors()
    assert sensors, "реестр пуст"
    errors = reg.validate_sensors(sensors)
    assert errors == [], errors
    ids = [s["id"] for s in sensors]
    assert "S-001" in ids and "S-030" in ids


def test_unknown_zone_rejected():
    errors = reg.validate_sensors([_sensor("S-900", zone="line-9")])
    assert any("zone" in e for e in errors)


def test_missing_field_rejected():
    entry = _sensor("S-900")
    del entry["facts"]
    errors = reg.validate_sensors([entry])
    assert any("facts" in e for e in errors)


def test_duplicate_id_rejected():
    errors = reg.validate_sensors([_sensor("S-900"), _sensor("S-900")])
    assert any("дубль" in e for e in errors)


def test_pending_probe_is_unverified(tmp_path):
    row = reg.probe_sensor(_sensor("S-900", status="pending"), out_dir=tmp_path)
    assert row["verdict"] == "unverified"


def test_active_without_file_fails(tmp_path):
    row = reg.probe_sensor(_sensor("S-901"), out_dir=tmp_path)
    assert row["verdict"] == "fail"


def test_active_with_good_fact_passes(tmp_path):
    subject = build_subject(repo_root=tmp_path, git_sha="a" * 40, dirty=False, device="cpu")
    write_fact("S-902", "v", 1, unit="count", quality="measured",
               method="fixture", subject=subject, out_dir=tmp_path)
    row = reg.probe_sensor(_sensor("S-902"), out_dir=tmp_path)
    assert row["verdict"] == "pass", row["reason"]


def test_active_with_unverified_last_is_unverified(tmp_path):
    subject = build_subject(repo_root=tmp_path, git_sha="a" * 40, dirty=False, device="cpu")
    write_fact("S-903", "v", None, unit="count", quality="measured",
               method="fixture", subject=subject, out_dir=tmp_path,
               status="unverified", note="нет источника")
    row = reg.probe_sensor(_sensor("S-903"), out_dir=tmp_path)
    assert row["verdict"] == "unverified"
    assert "нет источника" in row["reason"]


def test_active_stale_fails(tmp_path):
    subject = build_subject(repo_root=tmp_path, git_sha="a" * 40, dirty=False, device="cpu")
    old = datetime(2026, 1, 1, tzinfo=timezone.utc)
    write_fact("S-904", "v", 1, unit="count", quality="measured",
               method="fixture", subject=subject, out_dir=tmp_path,
               ts=old.isoformat(timespec="seconds"))
    now = old + timedelta(hours=48)
    row = reg.probe_sensor(_sensor("S-904", freshness_max_h=1), out_dir=tmp_path, now=now)
    assert row["verdict"] == "fail"


def test_fact_not_declared_in_registry_fails(tmp_path):
    subject = build_subject(repo_root=tmp_path, git_sha="a" * 40, dirty=False, device="cpu")
    write_fact("S-905", "other", 1, unit="count", quality="measured",
               method="fixture", subject=subject, out_dir=tmp_path)
    row = reg.probe_sensor(_sensor("S-905"), out_dir=tmp_path)
    assert row["verdict"] == "fail"


def test_sensor_fact_index_and_find():
    sensors = reg.load_sensors()
    index = reg.sensor_fact_index(sensors)
    assert "S-001" in index["num_kda_layers"]
    assert reg.find_sensor(sensors, "S-003") is not None
    assert reg.find_sensor(sensors, "S-999") is None


def test_probe_cli_selftest_green():
    from tools.sensors import probe

    assert probe.run_selftest() == 0


def test_probe_cli_json_output(tmp_path, capsys):
    from tools.sensors import probe

    code = probe.main(["--registry", str(_write_registry(tmp_path)), "--out-dir", str(tmp_path), "--json", str(tmp_path / "out.json")])
    assert code == 0
    rows = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert any(r["id"] == "S-900" for r in rows)


def _write_registry(tmp_path: Path) -> Path:
    import yaml

    path = tmp_path / "sensors.yaml"
    path.write_text(
        yaml.safe_dump([_sensor("S-900", status="pending")], sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path
