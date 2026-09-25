"""Acceptance criterion 12 — QB (quantile balancing) load balance.

On a uniform batch, after QB bias tuning (Eq. 14, kda-formulas.md :578-583)
the most loaded routed expert must take at most 2x the mean dispatch share,
and the QB auxiliary loss must decrease over the smoke run.  The router starts
deliberately skewed so the imbalance is real, not accidental.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from net import model, moe

STEPS = 25
BATCH = (4, 128)  # 512 tokens per QB estimate


def _skewed_moe(cfg):
    """A LatentMoE block whose initial dispatch is deliberately concentrated."""
    params = model.init_params(jr.PRNGKey(cfg.routing_seed), cfg)
    n_dense, _ = model.dense_moe_split(cfg)
    blk = params.layers[n_dense].mlp  # first LatentMoE layer
    skew = jnp.linspace(2.0, -2.0, cfg.moe_num_routed)  # expert 0 dominates
    return blk._replace(router_b=blk.router_b + skew)


def _qb_tune(blk, cfg, seed: int = 7):
    """Run the QB bias update on fresh uniform batches (the training mechanic)."""
    key = jr.PRNGKey(seed)
    for _ in range(STEPS):
        key, k = jr.split(key)
        x = jr.normal(k, (*BATCH, cfg.hidden))
        blk = blk._replace(router_b=moe.qb_update_bias(blk, cfg, x))
    return blk


def test_qb_balances_load(cfg):
    blk = _skewed_moe(cfg)
    x = jr.normal(jr.PRNGKey(99), (*BATCH, cfg.hidden))
    mean_share = cfg.moe_top_k / cfg.moe_num_routed
    before = moe.load_fraction(blk, cfg, x)
    assert float(before.max()) > 2.0 * mean_share  # the skew is real
    blk = _qb_tune(blk, cfg)
    after = moe.load_fraction(blk, cfg, x)
    assert float(after.max()) <= 2.0 * mean_share, f"max load {after.max():.3f} vs mean {mean_share:.3f}"


def test_qb_loss_decreases_on_smoke(cfg):
    blk = _skewed_moe(cfg)
    probe = jr.normal(jr.PRNGKey(123), (*BATCH, cfg.hidden))
    losses = [float(moe.qb_aux_loss(blk, cfg, probe))]
    key = jr.PRNGKey(7)
    for _ in range(STEPS):
        key, k = jr.split(key)
        x = jr.normal(k, (*BATCH, cfg.hidden))
        blk = blk._replace(router_b=moe.qb_update_bias(blk, cfg, x))
        losses.append(float(moe.qb_aux_loss(blk, cfg, probe)))
    assert losses[-1] < losses[0], f"QB loss did not decrease: {losses[0]:.3f} -> {losses[-1]:.3f}"
    balanced_floor = float(cfg.moe_top_k)  # n * sum_j (k/n)(1/n) = k at perfect balance
    assert losses[-1] < balanced_floor * 1.5, f"QB loss {losses[-1]:.3f} far from balance floor {balanced_floor}"
