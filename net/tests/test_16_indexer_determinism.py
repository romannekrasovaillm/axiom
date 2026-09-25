"""Acceptance criterion 17 — deterministic indexer selection.

Same input + same ``indexer_seed`` => identical top-k indices (the analogue of
criterion 11 for routing).  The seed is folded into the indexer weights at
initialisation and pinned in ``net/config.json`` / the run manifest (spine AD-4).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr

from conftest import small_config

from net import mla, model
from net.config import ModelConfig

CONFIG = Path(__file__).resolve().parent.parent / "config.json"


def _sparse_cfg(cfg, **over):
    return dataclasses.replace(cfg, attn_dense_reference=False, **over)


def test_repeated_selection_is_identical(cfg):
    params = mla.init_mla(jr.PRNGKey(0), cfg)
    x = jr.normal(jr.PRNGKey(1), (2, 40, cfg.hidden))
    scfg = _sparse_cfg(cfg, swa_window=8, mla_top_k=4)
    a = mla.topk_indices(params, scfg, x)
    b = mla.topk_indices(params, scfg, x)
    assert a.shape == (2, 40, 4)
    assert bool(jnp.array_equal(a, b))


def test_seed_pins_indexer_weights(cfg):
    """The same base key + indexer_seed reproduces the indexer weights."""
    a = mla.init_mla(jr.PRNGKey(0), dataclasses.replace(cfg, indexer_seed=7))
    b = mla.init_mla(jr.PRNGKey(0), dataclasses.replace(cfg, indexer_seed=7))
    assert bool(jnp.array_equal(a.W_idx_q, b.W_idx_q))
    assert bool(jnp.array_equal(a.W_idx_k, b.W_idx_k))
    c = mla.init_mla(jr.PRNGKey(0), dataclasses.replace(cfg, indexer_seed=8))
    assert not bool(jnp.array_equal(a.W_idx_k, c.W_idx_k))


def test_selection_ignores_non_indexer_weights(cfg):
    """Selection depends on the indexer and the latent projection only."""
    params = mla.init_mla(jr.PRNGKey(0), cfg)
    x = jr.normal(jr.PRNGKey(1), (2, 40, cfg.hidden))
    scfg = _sparse_cfg(cfg, swa_window=8, mla_top_k=4)
    base = mla.topk_indices(params, scfg, x)
    # W_o / W_g / SWA projections do not enter the selection.
    other = params._replace(
        W_o=jr.normal(jr.PRNGKey(99), params.W_o.shape),
        W_g=jr.normal(jr.PRNGKey(98), params.W_g.shape),
    )
    assert bool(jnp.array_equal(base, mla.topk_indices(other, scfg, x)))


def test_selection_deterministic_across_network_reinit(cfg):
    """Two model initialisations with the same key + pinned seed agree."""
    key = jr.PRNGKey(cfg.indexer_seed)
    pa = model.init_params(key, cfg)
    pb = model.init_params(jr.PRNGKey(cfg.indexer_seed), cfg)
    mla_layer = cfg.num_layers - 1  # the MLA layer of the [K,K,K,M] block
    x = jr.normal(jr.PRNGKey(2), (1, 33, cfg.hidden))
    scfg = _sparse_cfg(cfg, swa_window=8, mla_top_k=4)
    ia = mla.topk_indices(pa.layers[mla_layer].attn, scfg, x)
    ib = mla.topk_indices(pb.layers[mla_layer].attn, scfg, x)
    assert bool(jnp.array_equal(ia, ib))


def test_indexer_seed_pinned_in_config():
    data = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert data["indexer_seed"] == ModelConfig().indexer_seed
    assert data["mla_top_k"] == ModelConfig().mla_top_k
    assert data["swa_window"] == ModelConfig().swa_window
