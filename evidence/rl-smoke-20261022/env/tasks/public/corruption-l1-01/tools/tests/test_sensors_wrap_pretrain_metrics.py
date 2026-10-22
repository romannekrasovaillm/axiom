"""S-012 wrap_pretrain_metrics — тесты датчика."""

import json

from tools.sensors import wrap_pretrain_metrics as mod


def test_selftest_green():
    assert mod.run_selftest() == 0


def test_missing_source_unverified(tmp_path):
    written = mod.measure(None, None, out_dir=tmp_path)
    assert written["tok_s_median_window"]["status"] == "unverified"


def test_parses_metrics_and_journal(tmp_path):
    metrics = tmp_path / "m.jsonl"
    metrics.write_text(
        "".join(json.dumps({"step": i, "tok_s": 10.0, "loss": 5.0, "step_seconds": 1.0}) + "\n" for i in range(5)),
        encoding="utf-8",
    )
    journal = tmp_path / "j.json"
    journal.write_text(json.dumps({"run_ref": "r", "mfu_median": 0.2}), encoding="utf-8")
    written = mod.measure(metrics, journal, out_dir=tmp_path)
    assert written["tok_s_median_window"]["value"] == 10.0
    assert written["mfu_declared_peak"]["value"] == 0.2
