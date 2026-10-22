"""S-013 wrap_rss — тесты датчика."""

import json

from tools.sensors import wrap_rss as mod


def test_selftest_green():
    assert mod.run_selftest() == 0


def test_missing_journal_unverified(tmp_path):
    written = mod.measure(tmp_path / "nope.json", out_dir=tmp_path)
    assert written["host_rss_peak_mb"]["status"] == "unverified"


def test_parses_rss(tmp_path):
    journal = tmp_path / "j.json"
    journal.write_text(json.dumps({"summary": {"peak_rss_mb": 42.0}}), encoding="utf-8")
    written = mod.measure(journal, out_dir=tmp_path)
    assert written["host_rss_peak_mb"]["value"] == 42.0
