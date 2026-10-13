"""Acceptance criterion 15 — FP4 latent KV: toy-recall delta vs FP8 baseline.

ADR-009 criterion 4: with the same weights and the same delta (sparse+SWA) path,
``qat_kv_enabled = True`` (MXFP4 latent) must not lose more than 2 percentage
points of toy needle-recall against the FP8 baseline (``qat_kv_enabled = False``,
threshold from NFR-004).  On failure the branch is rolled back (the flag stays
False); the test reports the measured delta instead of weakening the threshold.

The smoke trains the tiny copy fixture **on the delta path** (dense oracle off),
so FP4 and FP8 are compared under the mechanism that will actually ship — not
under the oracle (ADR-009 D5: the same selection/window is simulated in
training).
"""

from __future__ import annotations

import dataclasses

import pytest

import jax
import jax.numpy as jnp
import jax.random as jr

from conftest import tiny_config

from net import model, optimizer

FILL, KEY_LO, QUERY = 1, 2, 4
KEYS = 8
TRAIN_GAP = 4
EVAL_GAP = 16
STEPS = 300
LR = 1e-2
DELTA_MAX_PP = 0.02  # NFR-004 threshold: 2 percentage points

DELTA_CFG = dict(attn_dense_reference=False, swa_window=8, mla_top_k=512)


def _delta_config(**over):
    return dataclasses.replace(tiny_config(), **{**DELTA_CFG, **over})


@pytest.fixture(scope="module")
def trained_delta():
    cfg = _delta_config()
    key = jr.PRNGKey(0)
    params = model.init_params(key, cfg)

    def loss_fn(p, x):
        return model.compute_loss(p, cfg, x, chunk_size=8)

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    step = optimizer.make_step(cfg)
    state = optimizer.init_state(params)
    for _ in range(STEPS):
        key, k = jr.split(key)
        keys = jr.randint(k, (8,), KEY_LO, KEY_LO + KEYS)
        body = jnp.full((8, TRAIN_GAP), FILL)
        q = jnp.full((8, 1), QUERY)
        x = jnp.concatenate([keys[:, None], body, q, keys[:, None]], axis=1).astype(jnp.int32)
        _, grads = grad_fn(params, x)
        params, state = step(params, grads, state, LR)
    return cfg, params


def _eval_batch(gap, n=64):
    k = jr.PRNGKey(123)
    keys = jr.randint(k, (n,), KEY_LO, KEY_LO + KEYS)
    body = jnp.full((n, gap), FILL)
    q = jnp.full((n, 1), QUERY)
    x = jnp.concatenate([keys[:, None], body, q, jnp.full((n, 1), FILL)], axis=1).astype(jnp.int32)
    _, counts = jnp.unique(keys, return_counts=True)
    degenerate = float(jnp.max(counts)) / n
    return keys, x, degenerate


def _recall(cfg, params, gap, qat_kv, n=64):
    c = dataclasses.replace(cfg, qat_kv_enabled=qat_kv)
    keys, x, degenerate = _eval_batch(gap, n)
    logits = model.forward(params, c, x, chunk_size=8)
    pred = jnp.argmax(logits[:, gap + 1, :], axis=-1)
    recall = float(jnp.mean((pred == keys).astype(jnp.float32)))
    return recall, bool(jnp.all(jnp.isfinite(logits))), degenerate


def test_fp4_latent_recall_delta_within_threshold(trained_delta):
    cfg, params = trained_delta
    r_fp8, finite8, degenerate = _recall(cfg, params, EVAL_GAP, qat_kv=False)
    r_fp4, finite4, _ = _recall(cfg, params, EVAL_GAP, qat_kv=True)
    print(
        f"[criterion 15] gap={EVAL_GAP} recall FP8={r_fp8:.3f} FP4={r_fp4:.3f} "
        f"delta={abs(r_fp8 - r_fp4) * 100:.2f}pp (constant-predictor {degenerate:.3f})"
    )
    assert finite8 and finite4
    assert abs(r_fp8 - r_fp4) <= DELTA_MAX_PP, (
        f"FP4 latent delta {abs(r_fp8 - r_fp4) * 100:.2f}pp > {DELTA_MAX_PP * 100:.0f}pp "
        f"(FP8 {r_fp8:.3f}, FP4 {r_fp4:.3f}); roll back qat_kv_enabled"
    )


def test_fp4_is_a_real_perturbation(trained_delta):
    """The comparison is not vacuous: FP4 changes the logits but not finiteness."""
    cfg, params = trained_delta
    _, x, _ = _eval_batch(EVAL_GAP)
    off = model.forward(params, dataclasses.replace(cfg, qat_kv_enabled=False), x, chunk_size=8)
    on = model.forward(params, dataclasses.replace(cfg, qat_kv_enabled=True), x, chunk_size=8)
    assert bool(jnp.all(jnp.isfinite(on)))
    assert float(jnp.max(jnp.abs(on - off))) > 0.0
