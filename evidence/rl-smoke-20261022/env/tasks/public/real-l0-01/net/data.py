"""Pretrain data pipeline — 4-domain shards + vision-min, card + hash, packing.

MODEL-L3-SKELETON.md section 4.  The four text domains (Web Text, Code,
Mathematics, Knowledge) follow the K3 taxonomy (:676-677); vision-min covers
captions/OCR (native multimodal).  Data lives canonically on
``~/gb10-shared`` and is mounted by symlink (C-032/C-033) — no copies
are ever made in the working tree.  Each dataset carries a card (source,
license) and a content hash.

For the smoke tests a deterministic synthetic stream is provided so the
pipeline runs without downloading any real corpus.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp

GB10_SHARED = Path(
    os.environ.get("GB10_SHARED") or str(Path.home() / "gb10-shared")
)


class Domain:
    WEB_TEXT = "web_text"
    CODE = "code"
    MATHEMATICS = "mathematics"
    KNOWLEDGE = "knowledge"
    VISION = "vision"


DOMAINS = [Domain.WEB_TEXT, Domain.CODE, Domain.MATHEMATICS, Domain.KNOWLEDGE]


@dataclass(frozen=True)
class DatasetCard:
    name: str
    domain: str
    source: str
    license: str
    hash: str  # SHA-256 of the canonical shard, computed at ingestion


# Public datasets by domain (source URLs / licenses, per ADR-004).  The hashes
# are pinned at ingestion time; here they are empty until a shard is mounted.
PUBLIC_DATASETS: list[DatasetCard] = [
    DatasetCard("c4-subset", Domain.WEB_TEXT, "https://huggingface.co/datasets/allenai/c4", "ODC-BY", ""),
    DatasetCard("the-stack-subset", Domain.CODE, "https://huggingface.co/datasets/bigcode/the-stack", "various OSS", ""),
    DatasetCard("open-web-math", Domain.MATHEMATICS, "https://huggingface.co/datasets/open-web-math/open-web-math", "ODC-BY", ""),
    DatasetCard("wikipedia-2024", Domain.KNOWLEDGE, "https://huggingface.co/datasets/wikimedia/wikipedia", "CC-BY-SA", ""),
    DatasetCard("sbu-captions", Domain.VISION, "https://huggingface.co/datasets/sbu_captions", "custom", ""),
]


def shard_hash(path: Path, chunk: int = 1 << 20) -> str:
    """Content hash of a shard file (streamed, no full read into memory)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def write_manifest(cards: list[DatasetCard], out: Path) -> None:
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"datasets": [asdict(c) for c in cards]}, f, indent=2, ensure_ascii=False)


def symlink_shard(shard_path: Path, name: str, target_dir: Path) -> Path:
    """Create a symlink to a canonical shard; never copy (C-032/C-033)."""
    target_dir.mkdir(parents=True, exist_ok=True)
    link = target_dir / name
    if not link.exists():
        link.symlink_to(shard_path)
    return link


def pack_sequence(token_ids: list[int], context_len: int, pad_id: int = 0, bos_id: int = 1, eos_id: int = 2) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Pack a token sequence to exactly ``context_len`` (labels = input shifted)."""
    seq = [bos_id] + list(token_ids[: context_len - 2]) + [eos_id]
    seq = seq + [pad_id] * (context_len - len(seq))
    seq = seq[:context_len]
    inputs = jnp.asarray(seq[:-1], dtype=jnp.int32)
    labels = jnp.asarray(seq[1:], dtype=jnp.int32)
    return inputs, labels


def synthetic_stream(key, batch_size: int, seq_len: int, vocab_size: int, steps: int):
    """Deterministic synthetic token stream for smoke tests (no corpus)."""
    def gen(step_key):
        k1, k2 = jax.random.split(step_key)
        ids = jax.random.randint(k1, (batch_size, seq_len), 0, vocab_size, dtype=jnp.int32)
        return ids
    for s in range(steps):
        yield gen(jax.random.fold_in(key, s))


def synthetic_corpus_texts(seed: int = 0, n_docs: int = 200, words_per_doc: int = 64) -> list[str]:
    """Deterministic text sample used to train the tokenizer in the smoke tests."""
    import random
    rng = random.Random(seed)
    vocab = ["the", "model", "attention", "delta", "token", "context", "language",
             "код", "математика", "знание", "зрение", "сеть", "обучение", "данные",
             "x", "y", "=", "1", "2", "3", "(", ")", "{", "}", "[", "]", "int", "def"]
    docs = []
    for _ in range(n_docs):
        docs.append(" ".join(rng.choice(vocab) for _ in range(words_per_doc)))
    return docs
