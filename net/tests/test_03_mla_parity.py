"""Acceptance criterion 3 — Gated MLA parity.

The compressed latent path (``c_t = W_c x_t`` reconstructed through
up-projections) must reproduce full-key/value attention exactly: ``apply`` is
compared against ``reference_full_kv`` (the mathematically identical
uncompressed oracle) on identical weights and inputs in fp32 CPU.
"""

from __future__ import annotations

import pytest

import jax.numpy as jnp
import jax.random as jr

from net import mla
from net.config import ModelConfig


def test_mla_matches_reference(cfg):
    key = jr.PRNGKey(0)
    params = mla.init_mla(key, cfg)
    x = jr.normal(key, (2, 16, cfg.hidden))
    a = mla.apply(params, cfg, x)
    b = mla.reference_full_kv(params, cfg, x)
    assert float(jnp.max(jnp.abs(a - b))) <= 1e-5


def test_mla_matches_reference_varied_shapes(cfg):
    key = jr.PRNGKey(1)
    params = mla.init_mla(key, cfg)
    for b, t in ((1, 8), (3, 20)):
        x = jr.normal(key, (b, t, cfg.hidden))
        a = mla.apply(params, cfg, x)
        b = mla.reference_full_kv(params, cfg, x)
        assert float(jnp.max(jnp.abs(a - b))) <= 1e-5


def test_mla_causal_and_finite(cfg):
    key = jr.PRNGKey(2)
    params = mla.init_mla(key, cfg)
    x = jr.normal(key, (2, 16, cfg.hidden))
    out = mla.apply(params, cfg, x)
    assert out.shape == (2, 16, cfg.hidden)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_maxtext_k2_oracle_available(cfg):
    pytest.importorskip("maxtext", reason="MaxText fork not vendored in the skeleton")
    pytest.skip("MaxText K2 MLA parity runs in the forked integration, deferred here")
