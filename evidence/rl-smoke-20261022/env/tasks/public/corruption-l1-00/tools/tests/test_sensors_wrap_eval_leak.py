"""S-016 wrap_eval_leak — тесты датчика."""

from tools.sensors import wrap_eval_leak as mod


def test_selftest_green():
    assert mod.run_selftest() == 0


def test_no_holdout_unverified(tmp_path):
    written = mod.measure(tmp_path / "none.jsonl", out_dir=tmp_path)
    assert written["overlap_count"]["status"] == "unverified"
    assert written["holdout_sha256"]["status"] == "unverified"
