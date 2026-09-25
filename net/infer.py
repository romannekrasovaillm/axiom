"""Deterministic inference loop — NTP head only.

The MTP head is training-only and is dropped at inference (``model.py``
docstring, ``active_params_per_token`` definition); generation reads the
next-token logits of the backbone's tied-embedding head.

``generate(params, cfg, token_ids, max_new_tokens, temperature, seed)``:

* ``temperature <= 0`` — greedy argmax;
* ``temperature > 0`` — sampling from ``jax.random.PRNGKey(seed)`` with an
  optional top-p (nucleus) filter;
* stops at the EOS token or after ``max_new_tokens`` steps.

Determinism (pinned in the run manifest): the same ``(params, token_ids,
temperature, seed, top_p)`` produces an identical token sequence — routing
is seed-pinned by ``cfg.routing_seed`` and sampling by ``seed``.  No KV
cache at skeleton scale: each step re-runs the forward over the grown
sequence.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import ModelConfig
from . import model as model_mod
from .tokenizer import BPETokenizer, SPECIAL_TOKENS

PAD_ID = SPECIAL_TOKENS.index("<pad>")
BOS_ID = SPECIAL_TOKENS.index("<bos>")
EOS_ID = SPECIAL_TOKENS.index("<eos>")


def _top_p_filter(probs: jnp.ndarray, top_p: float) -> jnp.ndarray:
    """Keep the smallest token set whose cumulative probability reaches ``top_p``.

    The top token is always kept; the filtered distribution is renormalised.
    """
    order = jnp.argsort(probs)[::-1]
    sorted_probs = probs[order]
    cum = jnp.cumsum(sorted_probs)
    keep = (cum - sorted_probs) < top_p
    sorted_probs = jnp.where(keep, sorted_probs, 0.0)
    filtered = jnp.zeros_like(probs).at[order].set(sorted_probs)
    return filtered / filtered.sum()


def _next_token(
    logits: jnp.ndarray,
    temperature: float,
    top_p: float,
    key,
) -> jnp.ndarray:
    """One token from the last-position logits (greedy or seeded sampling)."""
    if temperature <= 0.0:
        return jnp.argmax(logits)
    scaled = logits / temperature
    probs = jax.nn.softmax(scaled)
    if top_p < 1.0:
        probs = _top_p_filter(probs, top_p)
    return jax.random.choice(key, logits.shape[0], shape=(), p=probs)


def generate(
    params: model_mod.ModelParams,
    cfg: ModelConfig,
    token_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    seed: int,
    *,
    top_p: float = 1.0,
    eos_token_id: int = EOS_ID,
    chunk_size: int = 64,
) -> list[int]:
    """Autoregressive generation from ``token_ids`` (prompt, without BOS/EOS).

    Returns the generated ids (EOS excluded).  The prompt ids must lie inside
    ``cfg.vocab_size`` — the tokenizer's vocabulary has to match the config's.
    """
    ids = [int(t) for t in token_ids]
    if not ids or max_new_tokens <= 0:
        return []
    if min(ids) < 0 or max(ids) >= cfg.vocab_size:
        raise ValueError(
            f"token id outside cfg.vocab_size={cfg.vocab_size} "
            f"(tokenizer vocabulary must match the config)"
        )
    key = jax.random.PRNGKey(seed)
    out: list[int] = []
    for _ in range(max_new_tokens):
        x = jnp.asarray(ids, dtype=jnp.int32)[None, :]
        logits = model_mod.forward(params, cfg, x, chunk_size=chunk_size)
        key, sub = jax.random.split(key)
        nxt = int(_next_token(logits[0, -1], float(temperature), float(top_p), sub))
        if nxt == eos_token_id:
            break
        out.append(nxt)
        ids.append(nxt)
    return out


def generate_text(
    params: model_mod.ModelParams,
    cfg: ModelConfig,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    seed: int,
    *,
    top_p: float = 1.0,
    chunk_size: int = 64,
) -> tuple[str, list[int]]:
    """Tokenise ``prompt`` → generate → detokenise.  Returns ``(text, ids)``."""
    token_ids = tokenizer.encode(prompt)
    out_ids = generate(
        params, cfg, token_ids, max_new_tokens, temperature, seed,
        top_p=top_p, chunk_size=chunk_size,
    )
    return tokenizer.decode(out_ids), out_ids
