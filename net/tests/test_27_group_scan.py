"""Group-scan of the repeated ``[K,K,K,M]`` layer stack — host-compile RAM /6.

The 24-layer backbone is the periodic pattern ``[KDA, KDA, KDA, Gated-MLA] x 6``
(``net/config.py``'s ``num_layers % 4 == 0`` invariant).  Unrolled, XLA traces a
distinct region per layer and the host working set of the full ``l3-full`` graph
at T=8192 blows past the box's RAM (>250 GB spike).  The group-scan runs the
repeated tail groups through **one** ``jax.lax.scan`` body, so XLA reuses one
unit's host buffers across the scanned units (``tools/bench_block_merge.py``
measures the per-record cost flat in the layer count inside a scan).

Design points pinned by this module:

* the switch is **declarative** — ``scan_layers`` in ``net/config.json`` (spine
  AD-9 / guard C-035 form, the same mechanism as ``grad_ckpt_policy`` and
  ``kda_chunked_backward``); ``false`` is the unrolled parity reference;
* the leading dense-MLP unit stays in python (group 0 is heterogeneous: layer 0
  is a dense SiTU-GLU MLP, the scanned groups are LatentMoE) — ``first_unit =
  ceil(moe_dense_layers / 4)``;
* the plan is admitted only when the scanned units are uniform: one MLA
  ``mode`` and a candidate pool whose pytree structure does not change across
  iterations (under the pinned ``attn_dense_reference`` the pool is never
  built).  A rejected plan is a journal warning, not an error — the unrolled
  path is the same arithmetic;
* **transient stacking**: the external parameter tree stays flat (24 separate
  ``BlockParams``); the group axis exists only inside the scan's ``xs``.  So the
  optimizer sees exactly the pre-delta tree and the per-head Muon branch
  (``_PER_HEAD_LEAVES``) still fires for every Q/K/V projection — the design's
  stacked-tree Muon adapter is unnecessary by construction, and this module
  pins both the flatness and the per-head geometry;
* AttnRes crosses the scan as a fixed-shape ``(num_layers+1, B, T, hidden)``
  buffer read through :func:`net.attnres.apply_layer_masked` — bit-for-bit the
  variable-length :func:`net.attnres.apply_layer` over the live prefix.

All parity oracles run on the small CPU config; the scan needs at least two
whole units, so ``small_config`` (one unit) is extended to 12 layers / 3 units.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from conftest import small_config

from net import checkpoint, model, optimizer
from net.config import load_config, validate_config

#: The case's declarative config (where ``scan_layers`` lives).
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"

#: Sequence length for the small-config oracles (kept short — CPU).
SEQ = 16

#: A multi-unit layout: 3 whole ``[K,K,K,M]`` units (12 layers), so groups 1..2
#: are actually scanned (a one-unit config never scans — see the single-unit
#: guard test).  Keeps the small model's 3:1 KDA:MLA ratio.
_SCAN_KW = dict(num_layers=12, num_kda_layers=9, num_mla_layers=3)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _scan_cfg(**overrides):
    return small_config(**{**_SCAN_KW, **overrides})


def _with_flag(cfg, on: bool):
    return dataclasses.replace(cfg, scan_layers=on)


def _ids(cfg, *, seed=0, T=SEQ):
    return jr.randint(jr.PRNGKey(seed), (2, T), 0, cfg.vocab_size)


def _tree_allclose(a, b, *, rtol=1e-5, atol=1e-6) -> bool:
    la = jax.tree_util.tree_leaves(a)
    lb = jax.tree_util.tree_leaves(b)
    assert len(la) == len(lb)
    return all(
        bool(jnp.allclose(x, y, rtol=rtol, atol=atol, equal_nan=True))
        for x, y in zip(la, lb)
    )


def _unrolled_reference(params, cfg, ids, *, chunk_size=16, use_attnres=True):
    """The plain layer loop, written out here so a bitwise pin does not depend
    on :func:`net.model.forward` calling itself.  Pool is ``None`` throughout
    because the small config is the dense oracle (``attn_dense_reference``)."""
    emb = params.embedding[ids]
    h = emb
    deltas: list[jnp.ndarray] = []
    for i, block in enumerate(params.layers):
        is_kda = model._layer_is_kda(i)
        mode = "full" if is_kda else model.mla_mod.layer_mode(
            cfg, model._mla_ordinal(i)
        )
        aw = params.attnres.w[i] if use_attnres else None
        h, delta, _qb, _pool = model._layer_transition(
            block, aw, h, None, tuple(deltas), emb,
            cfg=cfg, chunk_size=chunk_size, collect_qb=False,
            use_attnres=use_attnres, index=i, is_kda=is_kda, mode=mode,
        )
        deltas.append(delta)
    h = model.rms_norm(h, params.norm_final)
    return h @ params.embedding.T


# ---------------------------------------------------------------------------
# declaration: net/config.json is the switch (spine AD-9 / C-035 form)
# ---------------------------------------------------------------------------


def test_declared_flag_is_on_for_pretrain():
    cfg = load_config(CONFIG_PATH)
    assert cfg.scan_layers is True
    validate_config(cfg)  # the declared value is one forward implements


def test_pinned_config_is_scannable():
    """``l3-full`` admits the scan: units 1..5, one MLA mode (``reindex``)."""
    cfg = load_config(CONFIG_PATH)
    assert model._group_scan_units(cfg) == (1, "reindex")


def test_schema_default_is_on():
    assert small_config().scan_layers is True


def test_non_bool_is_rejected():
    """A truthy non-bool would flip the graph by accident, so it is refused."""
    cfg = dataclasses.replace(small_config(), scan_layers=1)
    with pytest.raises(AssertionError):
        validate_config(cfg)


# ---------------------------------------------------------------------------
# plan guards: a rejected layout loses the saving, not correctness
# ---------------------------------------------------------------------------


def test_single_unit_never_scans():
    """``small_config`` is one unit: nothing repeats, so it stays unrolled."""
    assert model._group_scan_units(small_config()) is None


def test_flag_off_yields_no_plan():
    assert model._group_scan_units(_with_flag(_scan_cfg(), False)) is None


def test_uniform_units_are_scannable():
    first_unit, mode = model._group_scan_units(_scan_cfg())
    assert first_unit == 1  # layer 0 is the dense-MLP prefix
    assert mode == "full"  # no mla_layer_modes declared -> full for all


def test_num_layers_not_multiple_of_four_falls_back():
    cfg = dataclasses.replace(
        _scan_cfg(), num_layers=6, num_kda_layers=4, num_mla_layers=2
    )
    assert model._group_scan_units(cfg) is None


def test_heterogeneous_mla_modes_fall_back():
    """A shared body cannot specialise a per-iteration MLA mode."""
    cfg = _scan_cfg(mla_layer_modes=("full", "reindex", "full"))
    assert model._group_scan_units(cfg) is None


def test_live_pool_falls_back():
    """The pool's pytree structure flips under the sparse path -> not stable."""
    cfg = dataclasses.replace(
        _scan_cfg(), attn_dense_reference=False, mla_pool_size=16
    )
    assert model._group_scan_units(cfg) is None


def test_dense_oracle_keeps_pool_none_and_scans():
    """Under the dense oracle the pool is never built, so a positive
    ``mla_pool_size`` does not block the scan (the pinned layout)."""
    cfg = dataclasses.replace(
        _scan_cfg(), attn_dense_reference=True, mla_pool_size=16
    )
    assert model._group_scan_units(cfg) is not None


# ---------------------------------------------------------------------------
# parity: recompute changes where the graph lives, not the numbers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_attnres", [False, True])
def test_forward_parity(use_attnres):
    cfg = _scan_cfg()
    params = model.init_params(jr.PRNGKey(0), cfg)
    ids = _ids(cfg)
    on = model.forward(
        params, _with_flag(cfg, True), ids, chunk_size=16,
        use_attnres=use_attnres,
    )
    off = model.forward(
        params, _with_flag(cfg, False), ids, chunk_size=16,
        use_attnres=use_attnres,
    )
    assert _tree_allclose(on, off, rtol=1e-5, atol=1e-6)


def test_forward_parity_output_modes():
    """The scanned hidden state and QB term match the unrolled ones too."""
    cfg = _scan_cfg()
    params = model.init_params(jr.PRNGKey(0), cfg)
    ids = _ids(cfg)
    kw = dict(chunk_size=16, emit_logits=False, return_hidden=True, collect_qb=True)
    on = model.forward(params, _with_flag(cfg, True), ids, **kw)
    off = model.forward(params, _with_flag(cfg, False), ids, **kw)
    assert len(on) == 3 and len(off) == 3
    assert _tree_allclose(on, off, rtol=1e-5, atol=1e-6)


def test_model_loss_parity():
    cfg = _scan_cfg()
    params = model.init_params(jr.PRNGKey(0), cfg)
    ids = _ids(cfg)
    on = model.compute_loss(params, _with_flag(cfg, True), ids, chunk_size=16)
    off = model.compute_loss(params, _with_flag(cfg, False), ids, chunk_size=16)
    assert bool(jnp.allclose(on, off, rtol=1e-5, atol=1e-6))


def test_grad_parity():
    cfg = _scan_cfg()
    params = model.init_params(jr.PRNGKey(0), cfg)
    ids = _ids(cfg)
    g_on = jax.grad(
        lambda p: model.compute_loss(p, _with_flag(cfg, True), ids, chunk_size=16)
    )(params)
    g_off = jax.grad(
        lambda p: model.compute_loss(p, _with_flag(cfg, False), ids, chunk_size=16)
    )(params)
    assert _tree_allclose(g_on, g_off, rtol=2e-2, atol=2e-3)


def test_jit_parity():
    """The mechanism survives jit + value_and_grad (the training path)."""
    cfg = _scan_cfg()
    params = model.init_params(jr.PRNGKey(0), cfg)
    ids = _ids(cfg)

    def run(c):
        return jax.jit(jax.value_and_grad(
            lambda p: model.compute_loss(p, c, ids, chunk_size=16)
        ))(params)

    v_on, g_on = run(_with_flag(cfg, True))
    v_off, g_off = run(_with_flag(cfg, False))
    assert bool(jnp.allclose(v_on, v_off, rtol=1e-5, atol=1e-6))
    assert _tree_allclose(g_on, g_off, rtol=2e-2, atol=2e-3)


def test_mtp_head_parity():
    cfg = _scan_cfg()
    params = model.init_params(jr.PRNGKey(0), cfg)
    ids = _ids(cfg, T=SEQ + 8)
    hidden = model.forward(
        params, cfg, ids, chunk_size=16, return_hidden=True, emit_logits=False
    )
    on = model.mtp_loss(params, _with_flag(cfg, True), hidden, ids, 16)
    off = model.mtp_loss(params, _with_flag(cfg, False), hidden, ids, 16)
    assert bool(jnp.allclose(on, off, rtol=1e-5, atol=1e-6))


def test_flag_off_is_bitwise_unrolled():
    """Off reproduces the plain layer loop bit-for-bit (the gated graph)."""
    cfg = _scan_cfg()
    params = model.init_params(jr.PRNGKey(0), cfg)
    ids = _ids(cfg, seed=2)
    off = model.forward(params, _with_flag(cfg, False), ids, chunk_size=16)
    reference = _unrolled_reference(params, _with_flag(cfg, False), ids)
    assert jnp.array_equal(off, reference)


def test_scan_changes_the_graph_not_the_param_tree():
    """Transient stacking: the external tree is identical either way.

    This is why the Muon adapter the design anticipated for a stacked external
    tree is unnecessary — the optimizer never sees a group axis.
    """
    cfg = _scan_cfg()
    p_on = model.init_params(jr.PRNGKey(0), _with_flag(cfg, True))
    p_off = model.init_params(jr.PRNGKey(0), _with_flag(cfg, False))
    assert jax.tree_util.tree_structure(p_on) == jax.tree_util.tree_structure(p_off)
    assert all(
        a.shape == b.shape
        for a, b in zip(
            jax.tree_util.tree_leaves(p_on), jax.tree_util.tree_leaves(p_off)
        )
    )


# ---------------------------------------------------------------------------
# Muon semantics: per-head orthogonalization preserved per layer
# ---------------------------------------------------------------------------


def test_per_head_leaves_stay_two_dimensional():
    """Every Q/K/V projection is ndim 2, so the optimizer takes the per-head
    branch (``_PER_HEAD_LEAVES``), not the batched one."""
    cfg = _scan_cfg()
    params = model.init_params(jr.PRNGKey(0), cfg)

    seen = {"n": 0, "bad": 0}

    def visit(path, leaf):
        name = getattr(path[-1], "name", None) if path else None
        if name in optimizer._PER_HEAD_LEAVES:
            seen["n"] += 1
            if leaf.ndim != 2:
                seen["bad"] += 1

    jax.tree_util.tree_map_with_path(visit, params)
    assert seen["n"] > 0, "no per-head leaf found — the walker is wrong"
    assert seen["bad"] == 0, "a per-head leaf was stacked; Muon branch would change"


def test_per_head_orthogonalises_each_head_independently():
    """``per_head_newtonschulz5`` is exactly per-head NS concatenated — the
    property the scan must not perturb."""
    H, dim_in, dh = 3, 8, 4
    m = jr.normal(jr.PRNGKey(0), (dim_in, H * dh))
    out = optimizer.per_head_newtonschulz5(m, H)
    assert out.shape == m.shape
    for h in range(H):
        cols = slice(h * dh, (h + 1) * dh)
        manual = optimizer.newtonschulz5(m[:, cols])
        assert bool(jnp.allclose(out[:, cols], manual, rtol=1e-5, atol=1e-6))


def test_group_axis_flatten_preserves_per_head():
    """The design's Muon adapter lemma: folding the group axis into the head
    axis before the batched Newton-Schulz keeps each layer's per-head
    orthogonalization.  (The implementation avoids the adapter entirely by
    transient stacking, but the geometry it relies on is pinned here.)"""
    G, H, dim_in, dh = 6, 3, 8, 4
    g = jr.normal(jr.PRNGKey(0), (G, dim_in, H * dh))

    separate = jnp.stack(
        [optimizer.per_head_newtonschulz5(g[u], H) for u in range(G)]
    )
    per_head_view = (
        g.reshape(G, dim_in, H, dh).transpose(0, 2, 1, 3).reshape(G * H, dim_in, dh)
    )
    batched = (
        jax.vmap(optimizer.newtonschulz5)(per_head_view)
        .reshape(G, H, dim_in, dh).transpose(0, 2, 1, 3)
        .reshape(G, dim_in, H * dh)
    )
    assert _tree_allclose(separate, batched, rtol=1e-4, atol=1e-5)


def test_optimizer_step_parity_scan_vs_unrolled():
    """One Muon step on the scan gradients equals one on the unrolled
    gradients — the optimizer sees the same tree and the same leaf names."""
    cfg = _scan_cfg()
    params = model.init_params(jr.PRNGKey(0), cfg)
    ids = _ids(cfg)
    g_on = jax.grad(
        lambda p: model.compute_loss(p, _with_flag(cfg, True), ids, chunk_size=16)
    )(params)
    g_off = jax.grad(
        lambda p: model.compute_loss(p, _with_flag(cfg, False), ids, chunk_size=16)
    )(params)
    step = optimizer.make_step(cfg)
    state = optimizer.init_state(params)
    p_on, s_on = step(params, g_on, state, 0.01)
    p_off, s_off = step(params, g_off, state, 0.01)
    assert _tree_allclose(p_on, p_off, rtol=2e-2, atol=2e-3)
    assert _tree_allclose(s_on, s_off, rtol=2e-2, atol=2e-3)


# ---------------------------------------------------------------------------
# checkpoint: the flat param tree round-trips on the scan-enabled model
# ---------------------------------------------------------------------------


def test_checkpoint_roundtrip_on_scan_model(tmp_path):
    cfg = _scan_cfg()
    params = model.init_params(jr.PRNGKey(0), cfg)
    directory = tmp_path / "ckpt"
    digest = checkpoint.save_checkpoint(params, directory)
    restored = checkpoint.load_checkpoint(directory, target=params)
    assert jax.tree_util.tree_structure(params) == jax.tree_util.tree_structure(
        restored
    )
    for a, b in zip(
        jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(restored)
    ):
        assert bool(jnp.array_equal(a, b)), "round-trip is not bitwise identical"
    manifest = checkpoint.read_manifest(directory / "manifest.json")
    assert manifest["checkpoint_hash"] == digest
    assert checkpoint.tree_hash(params) == checkpoint.tree_hash(restored)


# ---------------------------------------------------------------------------
# full-geometry manifest (opt-in, for the GB10 host-compile confirmation run)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("AXIOM_SCAN_HOST_MEMORY") != "1",
    reason="full-geometry host-compile manifest is for the GB10 run "
    "(set AXIOM_SCAN_HOST_MEMORY=1)",
)
def test_pinned_config_plan_manifest():
    """The pinned ``l3-full`` layout admits a plan of 5 scanned units with one
    body — the quantity the host-compile RAM measurement is taken on."""
    cfg = load_config(CONFIG_PATH)
    plan = model._group_scan_units(cfg)
    n_units = cfg.num_layers // 4
    print(
        f"[group-scan] plan={plan} units={n_units} "
        f"scanned={n_units - plan[0] if plan else 0}"
    )
    assert plan is not None
    assert plan[0] == 1
    assert n_units - plan[0] == 5
