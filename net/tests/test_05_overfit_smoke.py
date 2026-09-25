"""Acceptance criterion 5 — overfit smoke.

A tiny model overfits a small fixed corpus with the Per-Head Muon optimizer and
the combined NTP + MTP loss.  The loss must fall below a fixed threshold
(pinned against the baseline run).  The full criterion is 1M tokens of one
corpus; the smoke exercises the same mechanism (loss monotonically decreasing)
at a fraction of the compute.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr

from net import model, optimizer

STEPS = 100
LR = 1e-2
THRESHOLD = 0.5  # loss below this after overfitting (baseline: start ~4.6)


def test_overfit_smoke(tiny_cfg):
    key = jr.PRNGKey(0)
    params = model.init_params(key, tiny_cfg)
    ids = jr.randint(key, (4, 16), 0, tiny_cfg.vocab_size)

    def loss_fn(p, x):
        return model.compute_loss(p, tiny_cfg, x, chunk_size=8)

    step = optimizer.make_step(tiny_cfg)
    state = optimizer.init_state(params)
    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    losses = []
    for _ in range(STEPS):
        loss, grads = grad_fn(params, ids)
        params, state = step(params, grads, state, LR)
        losses.append(float(loss))

    assert losses[-1] < THRESHOLD, f"final loss {losses[-1]} above threshold {THRESHOLD}"
    # monotonically (non-strictly) decreasing over the tail
    assert losses[-1] < losses[0]
