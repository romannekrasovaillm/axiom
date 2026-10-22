"""S-004 tokenizer_artifact — тесты датчика."""

from pathlib import Path

from tools.sensors import tokenizer_artifact


def test_selftest_green():
    assert tokenizer_artifact.run_selftest() == 0


def test_missing_artifact_is_unverified(tmp_path):
    written = tokenizer_artifact.measure(tmp_path / "nope.model", out_dir=tmp_path)
    assert written["tokenizer_sha256"]["status"] == "unverified"
    assert written["tokenizer_sha256"]["value"] is None


def test_reads_vocab_and_hash(tmp_path):
    art = tmp_path / "tokenizer.model"
    art.write_text('{"model": {"vocab": {"a": 0, "b": 1}}}', encoding="utf-8")
    written = tokenizer_artifact.measure(art, out_dir=tmp_path)
    assert written["tokenizer_vocab_size"]["value"] == 2
    assert written["tokenizer_sha256"]["value"] is not None
