"""Acceptance criterion 7 — Orbax checkpoint round-trip, bitwise + hash.

Weights survive save -> load bit-identically, and the content hash is written
to the manifest.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr

from net import checkpoint, model


def test_orbax_roundtrip_bitwise(cfg, tmp_path):
    key = jr.PRNGKey(0)
    params = model.init_params(key, cfg)
    directory = tmp_path / "ckpt"
    digest = checkpoint.save_checkpoint(params, directory)

    restored = checkpoint.load_checkpoint(directory, target=params)
    leaves_a = jax.tree_util.tree_leaves(params)
    leaves_b = jax.tree_util.tree_leaves(restored)
    assert jax.tree_util.tree_structure(params) == jax.tree_util.tree_structure(restored)
    for a, b in zip(leaves_a, leaves_b):
        assert bool(jnp.array_equal(a, b)), "round-trip is not bitwise identical"

    manifest = checkpoint.read_manifest(directory / "manifest.json")
    assert manifest["checkpoint_hash"] == digest
    assert manifest["format"] == "orbax"


def test_tree_hash_deterministic(cfg):
    key = jr.PRNGKey(0)
    p1 = model.init_params(key, cfg)
    p2 = model.init_params(key, cfg)
    assert checkpoint.tree_hash(p1) == checkpoint.tree_hash(p2)
