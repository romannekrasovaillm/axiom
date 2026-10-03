"""Acceptance criterion 9 — BPE 160K tokenizer.

Round-trip >= 99.5% (byte-level BPE is lossless => 100%), vocabulary exactly
160K, and the hash is deterministic.

Two hashes live here and must not be conflated (ADR-004):

* the **data** tokenizer's canonical hash is pinned in ``net/config.json``
  (``500f80237b3bd0fb``) — it identifies the BPE trained on the pretrain corpus;
* the tokenizer in ``net/tokenizer.py`` is the skeleton's own **stub**, trained
  in the test context on ``synthetic_corpus_texts``; its hash is a different
  value, pinned here as ``SKELETON_TOKENIZER_HASH``.  The stub is not the data
  tokenizer, so its hash is never compared against the config pin.
"""

from __future__ import annotations

import json
from pathlib import Path

from net.data import synthetic_corpus_texts
from net.tokenizer import BPETokenizer, round_trip_rate

CONFIG = Path(__file__).resolve().parent.parent / "config.json"

#: Canonical hash of the *data* tokenizer (BPE 160K, ADR-004), the value pinned
#: in ``net/config.json``.  It names the corpus BPE, not the skeleton stub.
CANONICAL_DATA_TOKENIZER_HASH = "500f80237b3bd0fb"

#: Deterministic hash of the skeleton stub (``net/tokenizer.py``), pinned in this
#: test context.  Training the stub on ``synthetic_corpus_texts`` is reproducible
#: (same corpus + seed => same hash); the value is deliberately *not* the data
#: tokenizer's canonical hash above — the two vocabularies differ by design.
SKELETON_TOKENIZER_HASH = (
    "9f8d03091c4a44fa8e7c08f57f7d9f4f97b69e8fde622bb9f6ab613da3565857"
)


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
    # Determinism: a repeated generation over the same corpus and seed yields the
    # same vocabulary hash — this is what makes a run reproducible.
    tok = _canonical_tokenizer()
    tok2 = _canonical_tokenizer()
    assert tok.vocab_hash() == tok2.vocab_hash()
    # The skeleton stub's hash is pinned in this test context (not in the config:
    # the stub is not the data tokenizer).
    assert tok.vocab_hash() == SKELETON_TOKENIZER_HASH
    # The canonical *data* tokenizer hash (ADR-004) is the constant pinned in
    # net/config.json.
    data = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert data["tokenizer_hash"] == CANONICAL_DATA_TOKENIZER_HASH
