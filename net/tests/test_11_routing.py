"""Acceptance criterion 11 — deterministic routing.

Same input + same seed → identical expert assignments.  The routing seed is
pinned in ``net/config.json`` (``routing_seed``); the same file pins the
``active_params_per_token`` counter (checked against the live computation).
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr

from net import model, moe
from net.config import ModelConfig

CONFIG = Path(__file__).resolve().parent.parent / "config.json"


def test_same_input_same_seed_identical_assignments(cfg):
    key = jr.PRNGKey(cfg.routing_seed)
    params_a = model.init_params(key, cfg)
    params_b = model.init_params(jr.PRNGKey(cfg.routing_seed), cfg)
    x = jr.normal(jr.PRNGKey(1), (2, 16, cfg.hidden))
    n_dense, _ = model.dense_moe_split(cfg)
    for layer in range(n_dense, cfg.num_layers):  # every LatentMoE layer
        idx_a = moe.routing_indices(params_a.layers[layer].mlp, cfg, x)
        idx_b = moe.routing_indices(params_b.layers[layer].mlp, cfg, x)
        assert idx_a.shape == (2, 16, cfg.moe_top_k)
        assert bool(jnp.array_equal(idx_a, idx_b)), f"layer {layer} routing differs"


def test_routing_seed_pinned_in_config():
    data = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert data["routing_seed"] == ModelConfig().routing_seed


def test_active_params_counter_in_config():
    data = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert data["active_params_per_token"] == model.active_param_count(ModelConfig())
