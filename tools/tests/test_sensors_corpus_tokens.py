"""S-005 corpus_tokens — тесты датчика."""

import json

from tools.sensors import corpus_tokens
from tools.sensors._common import sha256_file


def test_selftest_green():
    assert corpus_tokens.run_selftest() == 0


def _make_root(tmp_path):
    for stream in ("W", "C", "Q"):
        (tmp_path / stream).mkdir()
    b = tmp_path / "W" / "s.bin"
    b.write_bytes(b"data")
    manifest = {
        "totals": {"stream_tokens": 100},
        "shards": [{"file": "s.bin", "sha256": sha256_file(b)}],
    }
    (tmp_path / "W" / "manifest-W.json").write_text(json.dumps(manifest), encoding="utf-8")
    for stream, n in (("C", 7), ("Q", 3)):
        (tmp_path / stream / f"manifest-{stream}.json").write_text(
            json.dumps({"totals": {"stream_tokens": n}}), encoding="utf-8"
        )
    return tmp_path


def test_sums_manifests(tmp_path):
    root = _make_root(tmp_path)
    written = corpus_tokens.measure(root, out_dir=tmp_path, spotcheck=1)
    assert written["corpus_tokens_total"]["value"] == 110
    assert written["corpus_tokens_w"]["value"] == 100
    assert written["bins_spotcheck_ok"]["value"] is True


def test_unavailable_root_unverified(tmp_path):
    written = corpus_tokens.measure(tmp_path / "absent", out_dir=tmp_path)
    assert written["corpus_tokens_total"]["status"] == "unverified"
