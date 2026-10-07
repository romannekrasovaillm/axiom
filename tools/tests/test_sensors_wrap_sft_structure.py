"""S-017 wrap_sft_structure — тесты датчика."""

from tools.sensors import wrap_sft_structure as mod


def test_selftest_green():
    assert mod.run_selftest() == 0


def test_missing_set_unverified(tmp_path):
    written = mod.measure(tmp_path / "none.jsonl", out_dir=tmp_path)
    assert written["sft_defect_shares"]["status"] == "unverified"


def test_canonical_mini_set(tmp_path):
    written = mod.measure(mod.DEFAULT_INPUT, out_dir=tmp_path)
    assert written["dataset_sha256"]["value"] is not None
    assert isinstance(written["sft_defect_shares"]["value"], dict)
