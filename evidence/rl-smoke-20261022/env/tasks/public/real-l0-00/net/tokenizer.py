"""Byte-level BPE tokenizer, 160K vocab, deterministic (seed-pinned).

MODEL-L3-SKELETON.md section 4: our own BPE trained on a sample of the pretrain
corpus, seed-pinned, with a fixed content hash.  Byte-level BPE is lossless by
construction (every UTF-8 byte is in the base vocabulary), so
detokenise -> tokenise round-trip is 100% for valid UTF-8 (>= 99.5% target).

Vocabulary layout: ``<pad> <bos> <eos> <unk>`` (ids 0..3), 256 byte tokens
(ids 4..259), trained merges from id 260, and reserved tokens to pad the
vocabulary to exactly ``vocab_size`` (standard practice for fixed-size
vocabularies; the reserved slots are unused).
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
N_BYTES = 256


class BPETokenizer:
    def __init__(self, vocab_size: int = 160_000, special_tokens: list[str] | None = None):
        self.vocab_size = vocab_size
        self.specials = special_tokens if special_tokens is not None else SPECIAL_TOKENS
        self.n_special = len(self.specials)
        self.merges: list[tuple[int, int]] = []
        self._id_to_bytes: dict[int, bytes] = {}
        self._merge_id: dict[tuple[int, int], int] = {}
        for i in range(self.n_special):
            self._id_to_bytes[i] = b""

    # -- training -----------------------------------------------------------

    def train(self, texts: list[str], seed: int = 0) -> "BPETokenizer":
        rng = random.Random(seed)
        seqs = [list(t.encode("utf-8")) for t in texts]
        next_id = self.n_special + N_BYTES
        max_merges = self.vocab_size - next_id

        # base byte ids
        for b in range(N_BYTES):
            self._id_to_bytes[self.n_special + b] = bytes([b])

        for _ in range(max_merges):
            counts = self._pair_counts(seqs)
            if not counts:
                break
            best = min(counts.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
            (a, b) = best[0]
            self.merges.append((a, b))
            self._merge_id[(a, b)] = next_id
            self._id_to_bytes[next_id] = self._id_to_bytes[a] + self._id_to_bytes[b]
            seqs = [self._merge_seq(seq, a, b, next_id) for seq in seqs]
            next_id += 1

        # reserved padding to exactly vocab_size
        for rid in range(next_id, self.vocab_size):
            self._id_to_bytes[rid] = b""
        self._freeze_merges = tuple(self.merges)
        return self

    @staticmethod
    def _pair_counts(seqs) -> dict[tuple[int, int], int]:
        counts: dict[tuple[int, int], int] = defaultdict(int)
        for seq in seqs:
            for i in range(len(seq) - 1):
                counts[(seq[i], seq[i + 1])] += 1
        return counts

    @staticmethod
    def _merge_seq(seq, a, b, c):
        out = []
        i = 0
        while i < len(seq):
            if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
                out.append(c)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        return out

    # -- encode / decode ----------------------------------------------------

    def _encode_bytes(self, data: bytes) -> list[int]:
        symbols = [self.n_special + b for b in data]
        for (a, b) in self.merges:
            c = self._merge_id[(a, b)]
            out = []
            i = 0
            while i < len(symbols):
                if i + 1 < len(symbols) and symbols[i] == a and symbols[i + 1] == b:
                    out.append(c)
                    i += 2
                else:
                    out.append(symbols[i])
                    i += 1
            symbols = out
        return symbols

    def encode(self, text: str) -> list[int]:
        return self._encode_bytes(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        chunks = []
        for i in ids:
            b = self._id_to_bytes.get(int(i))
            if b is None:
                b = self._id_to_bytes[self.n_special + 3]  # <unk> placeholder
            chunks.append(b)
        return b"".join(chunks).decode("utf-8", errors="replace")

    # -- persistence / hash -------------------------------------------------

    def vocab_hash(self) -> str:
        payload = json.dumps(
            {"specials": self.specials, "merges": [[a, b] for a, b in self.merges]},
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"specials": self.specials, "merges": [[a, b] for a, b in self.merges]},
                f,
                ensure_ascii=False,
            )

    @classmethod
    def load(cls, path: str | Path, vocab_size: int = 160_000) -> "BPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(vocab_size=vocab_size, special_tokens=data["specials"])
        # replay merges through train would be wasteful; rebuild directly
        next_id = tok.n_special + N_BYTES
        for b in range(N_BYTES):
            tok._id_to_bytes[tok.n_special + b] = bytes([b])
        for (a, b) in data["merges"]:
            tok.merges.append((a, b))
            tok._merge_id[(a, b)] = next_id
            tok._id_to_bytes[next_id] = tok._id_to_bytes[a] + tok._id_to_bytes[b]
            next_id += 1
        for rid in range(next_id, vocab_size):
            tok._id_to_bytes[rid] = b""
        return tok


def round_trip_rate(tok: BPETokenizer, texts: list[str]) -> float:
    """Fraction of texts where ``decode(encode(t)) == t``."""
    if not texts:
        return 1.0
    ok = sum(1 for t in texts if tok.decode(tok.encode(t)) == t)
    return ok / len(texts)
