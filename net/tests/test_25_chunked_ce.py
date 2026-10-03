"""Chunked cross-entropy of the NTP/MTP heads (D-8 remainder, ``ce_chunk_tokens``).

``compute_loss`` built the full ``(B, T, vocab)`` logits for both heads; at
V=262144 (the ``l3-full`` power-of-2 padded vocabulary), T=8192 that tensor is
8.6 GiB and the autodiff graph retains ~14 copies of it — a ~69+ GiB constant
floor that keeps ``l3-full@8192`` out of GB10's 76.8 GiB pool even with
per-layer remat (D-8, results note of 03.10).  The fix slices the T axis into
chunks of ``ce_chunk_tokens`` rows, builds one chunk's logits at a time and
reduces them with a *global* sum over a *global* count.

What these tests pin:

* the declared width is read from ``net/config.json`` (``1024`` for the
  ``l3-full`` preset) and validated by ``net.config.validate_config``; the
  schema default stays ``0`` so a config built in code keeps the naive path;
* the mechanism is real: every chunk's logits+loss is a ``remat`` boundary in
  the traced graph (a policy parsed but not consumed would be the AD-9
  failure mode);
* the loss and its gradients are unchanged versus the naive whole-vocabulary
  reduction (``allclose``, not bitwise: the chunked sum reorders the
  floating-point reduction — an XLA property, not a bug);
* the reduction is global sum / global count, *not* a mean of per-chunk means
  (the classic chunking bug), verified with a pad mask that falls across a
  chunk boundary;
* ``0`` reproduces the pre-delta graph bit-for-bit, so the switch is honest in
  both directions.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from conftest import small_config

from net import model
from net.config import load_config, validate_config

#: The case's declarative config (the file C-035-style declarations live in).
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"

#: Sequence length used by the small-config tests (NTP sees T-1 rows, MTP T-2).
SEQ = 40


def _params_ids(cfg, *, seed=0):
    key = jr.PRNGKey(seed)
    params = model.init_params(key, cfg)
    ids = jr.randint(key, (2, SEQ), 1, cfg.vocab_size)  # 1..V-1: keeps targets in range
    return params, ids


def _with_chunk(cfg, n: int):
    return dataclasses.replace(cfg, ce_chunk_tokens=n)


def _tree_allclose(a, b, *, rtol=1e-4, atol=1e-5) -> bool:
    leaves_a = jax.tree_util.tree_leaves(a)
    leaves_b = jax.tree_util.tree_leaves(b)
    assert len(leaves_a) == len(leaves_b)
    return all(
        bool(jnp.allclose(x, y, rtol=rtol, atol=atol, equal_nan=True))
        for x, y in zip(leaves_a, leaves_b)
    )


def _remat_count(cfg, *, policy: str = "none") -> int:
    """Number of ``remat`` boundaries in the traced ``compute_loss`` graph."""
    params, ids = _params_ids(cfg)

    def loss(p):
        return model.compute_loss(p, cfg, ids, chunk_size=16, grad_ckpt_policy=policy)

    closed = jax.make_jaxpr(loss)(params)
    return sum(
        1 for eqn in closed.jaxpr.eqns
        if str(eqn.primitive.name).startswith("remat")
    )


# ---------------------------------------------------------------------------
# declaration: the config file is the switch (spine AD-9 / C-035 form)
# ---------------------------------------------------------------------------


def test_declared_width_is_1024_for_pretrain():
    cfg = load_config(CONFIG_PATH)
    assert cfg.ce_chunk_tokens == 1024
    validate_config(cfg)  # the declared value is one the loss implements


def test_schema_default_is_off():
    """A config built in code (tests/smokes) keeps the naive path."""
    assert small_config().ce_chunk_tokens == 0


def test_negative_width_is_rejected():
    cfg = dataclasses.replace(small_config(), ce_chunk_tokens=-1)
    with pytest.raises(AssertionError):
        validate_config(cfg)


def test_loss_impl_names_the_switch():
    assert model.loss_impl(small_config()) == "naive_ce"
    assert model.loss_impl(_with_chunk(small_config(), 128)) == "chunked_ce"


# ---------------------------------------------------------------------------
# mechanism: per-chunk remat boundaries and the skipped full-vocab projection
# ---------------------------------------------------------------------------


def test_chunked_path_puts_a_remat_per_chunk(cfg):
    """NTP rows = T-1 = 39 -> 5 chunks of 8; MTP rows = T-2 = 38 -> 5 chunks.

    ``grad_ckpt_policy="none"`` leaves the backbone un-rematted, so every
    ``remat`` in the graph is a chunk boundary: 5 + 5 = 10.
    """
    assert _remat_count(_with_chunk(cfg, 8), policy="none") == 10


def test_naive_path_has_no_chunk_remat(cfg):
    assert _remat_count(cfg, policy="none") == 0


def test_emit_logits_false_skips_the_projection(cfg):
    """``forward(..., emit_logits=False)`` returns the hidden state alone."""
    params, ids = _params_ids(cfg)
    full_logits, full_hidden = model.forward(params, cfg, ids, return_hidden=True)
    hidden_only = model.forward(
        params, cfg, ids, return_hidden=True, emit_logits=False
    )
    assert full_logits.shape[-1] == cfg.vocab_size
    assert hidden_only.shape == full_hidden.shape
    assert jnp.array_equal(hidden_only, full_hidden)


# ---------------------------------------------------------------------------
# parity: chunking changes where the logits live, not the numbers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk", [8, 7, 1024])
def test_loss_parity(cfg, chunk):
    """Global sum/count equals the naive mean, whole-chunk or ragged tail."""
    params, ids = _params_ids(cfg)
    naive = model.compute_loss(params, cfg, ids, chunk_size=16)
    chunked = model.compute_loss(params, _with_chunk(cfg, chunk), ids, chunk_size=16)
    assert jnp.allclose(naive, chunked, rtol=1e-4, atol=1e-5)


def test_grad_parity(cfg):
    params, ids = _params_ids(cfg)
    c_chunked = _with_chunk(cfg, 8)
    g_naive = jax.grad(lambda p: model.compute_loss(p, cfg, ids, chunk_size=16))(params)
    g_chunk = jax.grad(lambda p: model.compute_loss(p, c_chunked, ids, chunk_size=16))(params)
    assert _tree_allclose(g_naive, g_chunk, rtol=2e-2, atol=2e-3)


def test_mtp_head_parity(cfg):
    """The MTP head's vocabulary projection is chunked the same way."""
    params, ids = _params_ids(cfg)
    hidden = model.forward(params, cfg, ids, return_hidden=True, emit_logits=False)
    naive = model.mtp_loss(params, cfg, hidden, ids, 16)
    chunked = model.mtp_loss(params, cfg, hidden, ids, 16, ce_chunk_tokens=8)
    assert jnp.allclose(naive, chunked, rtol=1e-4, atol=1e-5)


def test_jit_parity(cfg):
    """Chunked CE survives jit + value_and_grad (the training path)."""
    params, ids = _params_ids(cfg)
    c_chunked = _with_chunk(cfg, 8)

    def run(c):
        return jax.jit(jax.value_and_grad(
            lambda p: model.compute_loss(p, c, ids, chunk_size=16)
        ))(params)

    v_naive, g_naive = run(cfg)
    v_chunk, g_chunk = run(c_chunked)
    assert jnp.allclose(v_naive, v_chunk, rtol=1e-4, atol=1e-5)
    assert _tree_allclose(g_naive, g_chunk, rtol=2e-2, atol=2e-3)


# ---------------------------------------------------------------------------
# global normalization: sum/count, not a mean of per-chunk means
# ---------------------------------------------------------------------------


def test_pad_mask_across_chunk_boundary_is_global(cfg):
    """A pad mask straddling a chunk boundary still divides by the global count.

    Every row's first 11 positions are real and the last 2 are padding; with
    ``ce_chunk_tokens=5`` the third chunk (rows 10..14) straddles the boundary
    and has only one real row.  The chunked helper must equal the whole-tensor
    masked mean, and a deliberately wrong mean-of-means must differ (pins the
    bug the global reduction exists to avoid).
    """
    params, _ = _params_ids(cfg)
    key = jr.PRNGKey(7)
    features = jr.normal(key, (2, 13, cfg.hidden))  # 13 rows, 2 padded
    targets = jr.randint(key, (2, 13), 1, cfg.vocab_size)
    mask = jnp.zeros((2, 13), jnp.float32).at[:, :11].set(1.0)  # 11 real, 2 pad

    emb = params.embedding
    logits = features @ emb.T
    logp = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
    reference = jnp.sum(nll * mask) / jnp.sum(mask)  # global masked mean

    got = model._chunked_cross_entropy(features, targets, emb, 5, loss_mask=mask)
    assert jnp.allclose(got, reference, rtol=1e-4, atol=1e-5)

    # mean-of-means over the chunks [0:5], [5:10], [10:15] (5, 5, 1 real rows)
    # — the classic wrong reduction — weights the 1-row tail as much as a full
    # chunk, so it cannot match the global mean.
    per_chunk = [
        jnp.sum(nll[:, s:s + 5] * mask[:, s:s + 5]) / jnp.sum(mask[:, s:s + 5])
        for s in (0, 5, 10)
    ]
    mean_of_means = jnp.stack(per_chunk).mean()
    assert abs(float(mean_of_means - reference)) > 1e-6
    assert abs(float(mean_of_means - got)) > 1e-6


# ---------------------------------------------------------------------------
# disable flag: ``0`` is the pre-delta graph, bit for bit
# ---------------------------------------------------------------------------


def test_zero_width_is_bitwise_identical_to_default(cfg):
    params, ids = _params_ids(cfg, seed=3)
    default = model.compute_loss(params, cfg, ids, chunk_size=16)
    explicit_off = model.compute_loss(
        params, _with_chunk(cfg, 0), ids, chunk_size=16
    )
    assert jnp.array_equal(default, explicit_off)


def test_chunked_equals_naive_for_short_and_ragged_T(cfg):
    """T below the chunk width and a T that is not a multiple both coincide."""
    params, ids = _params_ids(cfg, seed=5)
    naive = model.compute_loss(params, cfg, ids, chunk_size=16)
    for width in (SEQ + 100, 3, 6):  # single chunk; ragged tails
        got = model.compute_loss(params, _with_chunk(cfg, width), ids, chunk_size=16)
        assert jnp.allclose(naive, got, rtol=1e-4, atol=1e-5)
