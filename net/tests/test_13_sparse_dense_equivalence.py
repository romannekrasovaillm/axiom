"""Acceptance criterion 13 — sparse path reduces to the dense oracle.

ADR-009 criterion 1: with ``mla_top_k >= T`` and ``swa_window = 0`` the sparse
selection is the full causal prefix and the window is off, so the network output
must coincide with the dense path within 1e-4 (fp32, CPU) — for *any* weights,
because selection only reorders a set, it does not change the attended values.

The dense oracle is ``attn_dense_reference = True`` (the default, ADR-009 D4);
the sparse path is the same config with the flag off.  Both run the real
mechanisms — no re-implementation in the test.

ADR-012 extends the criterion to the hierarchical path: when the candidate pool
covers the causal prefix (``mla_pool_size * mla_pool_block >= T``) the level-2
``reindex`` selection ranges over exactly the same record set as the level-1
full-prefix selection, so the pooled output must be the unpooled one.  A pool
*smaller* than the prefix is an approximation by construction (ADR-012,
Negative) and is not an equivalence case.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import jax.random as jr

from conftest import small_config, tiny_config

from net import mla, model

TOL = 1e-4


def _sparse_cfg(cfg, top_k: int):
    return dataclasses.replace(
        cfg, attn_dense_reference=False, swa_window=0, mla_top_k=top_k
    )


def _pool_cfg(cfg, modes, pool_block: int, pool_size: int, top_k: int):
    """Sparse config with the ADR-012 layout declared (pool + per-layer modes)."""
    return dataclasses.replace(
        cfg,
        attn_dense_reference=False,
        swa_window=0,
        mla_top_k=top_k,
        mla_pool_block=pool_block,
        mla_pool_size=pool_size,
        mla_layer_modes=tuple(modes),
    )


def test_mla_layer_sparse_matches_dense(cfg):
    """Layer-level: selection over the full prefix == dense causal softmax."""
    key = jr.PRNGKey(0)
    params = mla.init_mla(key, cfg)
    x = jr.normal(key, (2, 24, cfg.hidden))
    dense = mla.apply(params, cfg, x)
    sparse = mla.apply(params, _sparse_cfg(cfg, top_k=10**6), x)
    assert float(jnp.max(jnp.abs(dense - sparse))) <= TOL


def test_mla_layer_equivalence_holds_for_arbitrary_weights(cfg):
    """Independent of the trained weights: random params, several seeds/shapes."""
    for seed in (1, 2, 3):
        params = mla.init_mla(jr.PRNGKey(seed), cfg)
        for b, t in ((1, 8), (3, 20), (1, 33)):
            x = jr.normal(jr.PRNGKey(seed * 10 + t), (b, t, cfg.hidden))
            dense = mla.apply(params, cfg, x)
            sparse = mla.apply(params, _sparse_cfg(cfg, top_k=t), x)
            assert float(jnp.max(jnp.abs(dense - sparse))) <= TOL, f"seed={seed} t={t}"


def test_mla_top_k_larger_than_sequence(cfg):
    """top_k > T is clamped to T and stays equivalent."""
    params = mla.init_mla(jr.PRNGKey(7), cfg)
    x = jr.normal(jr.PRNGKey(8), (2, 16, cfg.hidden))
    dense = mla.apply(params, cfg, x)
    sparse = mla.apply(params, _sparse_cfg(cfg, top_k=4096), x)
    assert float(jnp.max(jnp.abs(dense - sparse))) <= TOL


def test_network_output_sparse_matches_dense():
    """Network level: all 24 (tiny) layers, the whole forward, equal to 1e-4."""
    cfg = tiny_config()
    sparse = _sparse_cfg(cfg, top_k=10**6)
    key = jr.PRNGKey(0)
    params = model.init_params(key, cfg)
    ids = jr.randint(key, (1, 20), 0, cfg.vocab_size)
    dense_logits = model.forward(params, cfg, ids, chunk_size=8)
    sparse_logits = model.forward(params, sparse, ids, chunk_size=8)
    assert float(jnp.max(jnp.abs(dense_logits - sparse_logits))) <= TOL


def test_network_equivalence_is_weight_independent():
    cfg = tiny_config()
    sparse = _sparse_cfg(cfg, top_k=10**6)
    for seed in (0, 4):
        params = model.init_params(jr.PRNGKey(seed), cfg)
        ids = jr.randint(jr.PRNGKey(seed + 1), (2, 18), 0, cfg.vocab_size)
        dense_logits = model.forward(params, cfg, ids, chunk_size=8)
        sparse_logits = model.forward(params, sparse, ids, chunk_size=8)
        assert float(jnp.max(jnp.abs(dense_logits - sparse_logits))) <= TOL, seed


def test_sparse_dense_equivalence_longer_than_one_block():
    """The blocked scan must agree across block boundaries (T > block)."""
    cfg = small_config()
    params = mla.init_mla(jr.PRNGKey(3), cfg)
    x = jr.normal(jr.PRNGKey(4), (1, 300, cfg.hidden))
    dense = mla.apply(params, cfg, x)
    sparse = mla.apply(params, _sparse_cfg(cfg, top_k=10**6), x)
    assert float(jnp.max(jnp.abs(dense - sparse))) <= TOL


def test_dense_path_is_the_default():
    """The dense oracle is selected unless the flag is turned off (A4 gate)."""
    import json
    from pathlib import Path

    from net.config import ModelConfig

    data = json.loads((Path(__file__).resolve().parent.parent / "config.json").read_text(encoding="utf-8"))
    assert data["attn_dense_reference"] is True
    assert ModelConfig().attn_dense_reference is True


# --- ADR-012: the hierarchical pool on top of criterion 13 -------------------


def test_reindex_over_pool_covering_prefix_is_exact(cfg):
    """A pool that covers the prefix makes level 2 exact, not approximate.

    ``mla_pool_size * mla_pool_block >= T`` means the level-2 layer scores every
    causal record — through the pool instead of the raw prefix — so its top_k
    selection is the same *set*, and the attended values are identical (the
    selection only reorders the union).
    """
    pooled = _pool_cfg(cfg, ("full", "reindex"), pool_block=64, pool_size=2, top_k=8)
    plain = dataclasses.replace(pooled, mla_pool_size=0, mla_layer_modes=())
    params = mla.init_mla(jr.PRNGKey(0), cfg)
    x = jr.normal(jr.PRNGKey(1), (2, 64, cfg.hidden))

    full_prefix = mla.apply(params, plain, x)
    _, pool = mla.apply_with_pool(params, pooled, x, mode="full")
    assert pool is not None and pool.blocks.shape[:2] == (2, 64)
    pooled_out, pool_out = mla.apply_with_pool(params, pooled, x, pool=pool, mode="reindex")

    assert pool_out is pool, "a consumer layer must hand the pool on unchanged"
    assert float(jnp.max(jnp.abs(full_prefix - pooled_out))) <= TOL


def test_pool_builder_output_is_unchanged_by_the_pool(cfg):
    """Mode ``full`` is still the full-prefix selection: building the pool is free.

    The builder publishes the pool *in addition to* its own exact selection, so
    its output must be bit-identical to the pool-disabled path.
    """
    pooled = _pool_cfg(cfg, ("full", "reindex"), pool_block=8, pool_size=2, top_k=4)
    plain = dataclasses.replace(pooled, mla_pool_size=0, mla_layer_modes=())
    params = mla.init_mla(jr.PRNGKey(2), cfg)
    x = jr.normal(jr.PRNGKey(3), (2, 40, cfg.hidden))

    a = mla.apply(params, plain, x)
    b, pool = mla.apply_with_pool(params, pooled, x, mode="full")
    assert pool is not None
    assert float(jnp.max(jnp.abs(a - b))) == 0.0


def test_reuse_reproduces_the_builders_selection(cfg):
    """Mode ``reuse`` is exactly the builder's top_k — no scoring of its own.

    Driven with the *same* layer parameters, reuse over the published pool must
    reproduce the mode-``full`` output bit-for-bit; a different selection would
    show up immediately, since the attended record set would differ.
    """
    reuse = _pool_cfg(cfg, ("full", "reuse"), pool_block=8, pool_size=2, top_k=4)
    params = mla.init_mla(jr.PRNGKey(4), cfg)
    x = jr.normal(jr.PRNGKey(5), (1, 40, cfg.hidden))

    full_out, pool = mla.apply_with_pool(params, reuse, x, mode="full")
    assert pool is not None and pool.selection is not None
    assert pool.selection.shape[-1] == 4
    reuse_out, _ = mla.apply_with_pool(params, reuse, x, pool=pool, mode="reuse")
    assert float(jnp.max(jnp.abs(full_out - reuse_out))) == 0.0


def test_network_with_pool_layout_still_matches_dense():
    """Criterion 13 at the network level with the pooling layout declared.

    ``mla_top_k >= T`` must keep the network equal to the dense oracle whichever
    pooling layout is configured — the dense path is the reference and the pool
    never changes the premise of the criterion.
    """
    base = small_config(num_layers=8, num_kda_layers=6, num_mla_layers=2)
    pooled = _pool_cfg(base, ("full", "reindex"), pool_block=8, pool_size=3, top_k=10**6)
    params = model.init_params(jr.PRNGKey(0), base)
    ids = jr.randint(jr.PRNGKey(1), (2, 20), 0, base.vocab_size)

    dense_logits = model.forward(params, base, ids, chunk_size=8)
    pooled_logits = model.forward(params, pooled, ids, chunk_size=8)
    assert float(jnp.max(jnp.abs(dense_logits - pooled_logits))) <= TOL


def test_network_pool_covering_prefix_matches_unpooled():
    """The whole forward with a prefix-covering pool equals the unpooled forward.

    Two MLA layers (``full`` then ``reindex``) share a pool of one block that
    covers T = 64, so the level-2 layer sees the whole prefix and the network
    output must be bit-identical to the pool-disabled configuration.
    """
    base = small_config(num_layers=8, num_kda_layers=6, num_mla_layers=2)
    pooled = _pool_cfg(base, ("full", "reindex"), pool_block=64, pool_size=1, top_k=8)
    plain = dataclasses.replace(pooled, mla_pool_size=0, mla_layer_modes=())
    params = model.init_params(jr.PRNGKey(0), base)
    ids = jr.randint(jr.PRNGKey(1), (1, 64), 0, base.vocab_size)

    a = model.forward(params, plain, ids, chunk_size=8)
    b = model.forward(params, pooled, ids, chunk_size=8)
    assert float(jnp.max(jnp.abs(a - b))) <= TOL


# --- criterion 16 delta: the fused union assembly is the same arithmetic ------


def test_window_duplicate_mask_matches_the_pairwise_compare():
    """The scatter-based duplicate mask is the pairwise ``(Q, W, k_sel)`` compare.

    The fused assembly marks a window slot by scattering the selected records
    into their slot (``O(Q * k_sel)``) instead of comparing every window slot
    against every selected index (``O(Q * W * k_sel)``).  This drives the
    scatter over records that land inside the window, before it and past its end,
    so the equivalence is asserted on all three cases, not only on the easy one.
    """
    from net.attn_sparse import _window_duplicates

    B, Q, W, k_sel, T = 2, 6, 4, 3, 10
    pos_c = jnp.asarray([0, 3, 5, 9, 5, 7])
    # Hand-picked selections covering the three cases the mask must get right:
    # records inside the window (a real duplicate), records older than the window
    # start (negative slot offset) and records past its end (offset >= W) — the
    # last two must come out *not* duplicates.
    top_idx = jnp.asarray(
        [
            [[0, 0, 2], [1, 2, 3], [4, 4, 5], [6, 8, 9], [5, 5, 2], [7, 4, 6]],
            [[0, 3, 3], [3, 0, 2], [5, 5, 5], [9, 9, 0], [4, 1, 5], [7, 6, 3]],
        ],
        dtype=jnp.int32,
    )
    assert top_idx.shape == (B, Q, k_sel)

    widx_raw = pos_c[:, None] + (jnp.arange(W)[None, :] - (W - 1))  # (Q, W)
    pairwise = jnp.any(widx_raw[None, :, :, None] == top_idx[:, :, None, :], axis=-1)
    scatter = _window_duplicates(top_idx, pos_c - (W - 1), W)
    assert scatter.shape == (B, Q, W)
    assert bool(jnp.all(scatter == pairwise)), f"scatter={scatter} pairwise={pairwise}"
    # The case is not vacuous: at least one slot really is a duplicate.
    assert bool(jnp.any(scatter))


def test_fused_assembly_matches_the_unfused_one(cfg):
    """The fused union assembly differs from the verbatim one by rounding only.

    Both legs run the *same* selection, the same masked scores and the same
    softmax over the concatenated scores; the fused leg applies the softmax
    weights to each branch's gathered values in place instead of to a
    concatenated value tensor.  That is the same sum of the same terms in a
    different association, so the outputs may differ by rounding — asserted
    against the criterion-13 tolerance, on shapes that cross the query-block
    boundary (``_QUERY_BLOCK = 128``) and with a window wide enough that the
    duplicate mask is genuinely in play (``swa_window >= T`` duplicates the
    whole selection).
    """
    params = mla.init_mla(jr.PRNGKey(0), cfg)
    for window in (8, 40):  # window < T, then window >= T (whole prefix)
        fused = dataclasses.replace(
            cfg, attn_dense_reference=False, swa_window=window, mla_top_k=4,
            attn_fused_assembly=True,
        )
        unfused = dataclasses.replace(fused, attn_fused_assembly=False)
        for b, t in ((1, 8), (3, 40), (1, 300)):
            x = jr.normal(jr.PRNGKey(t + window), (b, t, cfg.hidden))
            a = mla.apply(params, fused, x)
            u = mla.apply(params, unfused, x)
            assert float(jnp.max(jnp.abs(a - u))) <= TOL, f"window={window} b={b} t={t}"


def test_fused_assembly_holds_under_the_pool_layout(cfg):
    """The fused leg is equivalent under ADR-012's builder/consumer split too.

    The fuse is an assembly property, not a pool property, so it must not change
    the pooled output either: ``full`` builds the pool, ``reindex`` selects
    inside it, and both legs consume the same published pool.
    """
    modes = ("full", "reindex")
    pooled = _pool_cfg(cfg, modes, pool_block=8, pool_size=3, top_k=4)
    fused = dataclasses.replace(pooled, swa_window=8, attn_fused_assembly=True)
    unfused = dataclasses.replace(fused, attn_fused_assembly=False)
    params = mla.init_mla(jr.PRNGKey(9), cfg)
    x = jr.normal(jr.PRNGKey(10), (2, 40, cfg.hidden))

    def run(c):
        out, pool = mla.apply_with_pool(params, c, x, mode="full")
        out, pool = mla.apply_with_pool(params, c, x, pool=pool, mode="reindex")
        assert pool is not None
        return out

    assert float(jnp.max(jnp.abs(run(fused) - run(unfused)))) <= TOL


def test_fused_assembly_is_the_declared_default():
    """The fused assembly is pinned; the unfused leg exists for the A/B only."""
    import json
    from pathlib import Path

    from net.config import ModelConfig

    data = json.loads((Path(__file__).resolve().parent.parent / "config.json").read_text(encoding="utf-8"))
    assert data["attn_fused_assembly"] is True
    assert ModelConfig().attn_fused_assembly is True
