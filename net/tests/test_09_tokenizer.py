"""Acceptance criterion 9 — BPE 160K tokenizer.

Round-trip >= 99.5% (byte-level BPE is lossless => 100%), vocabulary exactly
160K, and the hash is deterministic and pinned in ``config.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

from net.data import synthetic_corpus_texts
from net.tokenizer import BPETokenizer, round_trip_rate

CONFIG = Path(__file__).resolve().parent.parent / "config.json"


def _canonical_tokenizer() -> BPETokenizer:
    texts = synthetic_corpus_texts(seed=0, n_docs=20, words_per_doc=32)
    return BPETokenizer(vocab_size=160_000).train(texts, seed=0)


def test_vocab_size_exactly_160k():
    tok = _canonical_tokenizer()
    assert tok.vocab_size == 160_000
    assert len(tok._id_to_bytes) == 160_000


def test_round_trip_above_threshold():
    tok = _canonical_tokenizer()
    held_out = synthetic_corpus_texts(seed=1, n_docs=50, words_per_doc=20)
    held_out.append("mixed: английский и русский текст — 123 例 😀\n\t\\n")
    rate = round_trip_rate(tok, held_out)
    assert rate >= 0.995, rate


def test_hash_deterministic_and_pinned():
    tok = _canonical_tokenizer()
    tok2 = _canonical_tokenizer()
    assert tok.vocab_hash() == tok2.vocab_hash()
    data = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert tok.vocab_hash() == data["tokenizer_hash"]
