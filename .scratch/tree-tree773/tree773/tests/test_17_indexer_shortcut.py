"""Regression for the sparse-path cost bottleneck (criterion 16 follow-up).

The 8K cost is dominated by the indexer's ``top_k`` over the whole sequence —
an ``O(T^2 log T)`` selection per forward (see
``docs/research/attn-cost-and-matmul-precision-2026-09-12.md`` section 3 and the
A4 delta report).  Two invariants pin the implementation so that bottleneck does
not silently regress:

1. The indexer runs **exactly once** per sparse forward when ``mla_top_k < T``
   (``top_k`` appears once in the jaxpr — no recompute per query block).
2. When ``mla_top_k >= T`` the selection is the whole causal prefix and the
   sparse path short-circuits to dense causal attention: ``top_k`` must **not**
   appear at all (criterion 13, and it would otherwise be pure ``O(T^2 log T)``
   overhead).

ADR-012 moves the bottleneck from "every layer scans the prefix" to "one layer
scans, the rest score the pool", which is a property of the layer *stack*, so it
needs its own pins:

3. The pool is built **exactly once per forward** — the level-1 selection costs
   one ``top_k`` over blocks, in the single builder layer, however many MLA
   layers consume it (its ``k`` is ``mla_pool_size``, distinct from the record
   selection width of every layer).
4. A ``reuse`` layer renders **no scores at all**: only the builder's ``top_k``
   appears in the forward, because the selection is reused verbatim.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import jax.random as jr

from conftest import small_config

from net import mla, model


def _count_primitive(jaxpr, name: str, **want) -> int:
    """Count primitive ``name`` across a (possibly nested) jaxpr.

    ``want`` filters on equality of the primitive's params, so e.g. ``k=3``
    counts only the ``top_k`` calls of width 3.
    """
    n = 0
    for eq in jaxpr.eqns:
        if eq.primitive.name == name and all(eq.params.get(key) == val for key, val in want.items()):
            n += 1
        for param in eq.params.values():
            if hasattr(param, "jaxpr"):
                n += _count_primitive(param.jaxpr, name, **want)
            elif isinstance(param, (tuple, list)):
                for q in param:
                    if hasattr(q, "jaxpr"):
                        n += _count_primitive(q.jaxpr, name, **want)
            elif isinstance(param, dict):
                for q in param.values():
                    if hasattr(q, "jaxpr"):
                        n += _count_primitive(q.jaxpr, name, **want)
    return n


def _topk_count(cfg, T: int) -> int:
    params = mla.init_mla(jr.PRNGKey(0), cfg)
    x = jr.normal(jr.PRNGKey(1), (1, T, cfg.hidden))
    closed = jax.make_jaxpr(mla.apply, static_argnums=(1,))(params, cfg, x)
    return _count_primitive(closed.jaxpr, "top_k")


def test_indexer_runs_once_for_partial_selection(cfg):
    """top_k < T: the indexer selection runs exactly once per forward."""
    cfg = dataclasses.replace(cfg, attn_dense_reference=False, swa_window=8, mla_top_k=4)
    assert _topk_count(cfg, T=40) == 1


def test_indexer_short_circuits_for_full_selection(cfg):
    """top_k >= T: the O(T^2 log T) top_k is skipped (criterion 13 fast path)."""
    cfg = dataclasses.replace(cfg, attn_dense_reference=False, swa_window=0, mla_top_k=10**6)
    assert _topk_count(cfg, T=24) == 0


def test_short_circuit_still_matches_dense(cfg):
    """The fast path is exact, not a different attention (criterion 13)."""
    sparse = dataclasses.replace(cfg, attn_dense_reference=False, swa_window=0, mla_top_k=10**6)
    params = mla.init_mla(jr.PRNGKey(3), cfg)
    x = jr.normal(jr.PRNGKey(4), (2, 33, cfg.hidden))
    dense = mla.apply(params, cfg, x)
    out = mla.apply(params, sparse, x)
    assert float(jnp.max(jnp.abs(dense - out))) <= 1e-4


# --- ADR-012: the shared pool is built once, not once per layer --------------


def _topk_widths(closed) -> list[int]:
    """Every ``top_k`` width in a closed jaxpr, in trace order (nested included)."""
    widths: list[int] = []

    def walk(jaxpr):
        for eq in jaxpr.eqns:
            if eq.primitive.name == "top_k":
                widths.append(int(eq.params["k"]))
            for param in eq.params.values():
                if hasattr(param, "jaxpr"):
                    walk(param.jaxpr)
                elif isinstance(param, (tuple, list)):
                    for q in param:
                        if hasattr(q, "jaxpr"):
                            walk(q.jaxpr)
                elif isinstance(param, dict):
                    for q in param.values():
                        if hasattr(q, "jaxpr"):
                            walk(q.jaxpr)

    walk(closed.jaxpr)
    return widths


def _model_topk_counts(cfg, T: int):
    """``Counter`` of every ``top_k`` width in one whole-model forward."""
    import collections

    params = model.init_params(jr.PRNGKey(0), cfg)
    ids = jr.randint(jr.PRNGKey(1), (1, T), 0, cfg.vocab_size)
    closed = jax.make_jaxpr(lambda p, i: model.forward(p, cfg, i, chunk_size=8))(params, ids)
    return collections.Counter(_topk_widths(closed))


def _abstract_topk_counts(cfg, T: int):
    """``Counter`` of ``top_k`` widths for the *pinned* config, shapes only.

    ``eval_shape`` builds the parameter tree without allocating it, so the full
    1B skeleton can be traced in seconds instead of being materialised.
    """
    import collections

    shapes = jax.eval_shape(lambda k: model.init_params(k, cfg), jr.PRNGKey(0))
    abstract = jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), shapes
    )
    closed = jax.make_jaxpr(lambda p, i: model.forward(p, cfg, i, chunk_size=8))(
        abstract, jax.ShapeDtypeStruct((1, T), jnp.int32)
    )
    return collections.Counter(_topk_widths(closed))


def _two_mla_cfg(modes, **over):
    """Two MLA layers (indices 3 and 7) so the pool can be shared between them.

    The widths are chosen apart from ``moe_top_k`` (the LatentMoE router also
    calls ``top_k``, at width ``moe_top_k = 2``, once per MoE layer): the pool
    build is 5, every record selection is 4.
    """
    base = small_config(
        num_layers=8, num_kda_layers=6, num_mla_layers=2,
        attn_dense_reference=False, swa_window=8, mla_top_k=4,
        mla_pool_block=8, mla_pool_size=5, mla_layer_modes=tuple(modes),
    )
    return dataclasses.replace(base, **over)


def test_pool_is_built_exactly_once_per_forward():
    """The level-1 pool selection appears once, however many layers consume it.

    With T = 40, ``mla_pool_block = 8`` and ``mla_pool_size = 5`` the pool
    ``top_k`` has width 5 while every record selection has width 4 = mla_top_k,
    so the two are counted apart: one pool build, one full-prefix selection
    (builder) and one in-pool selection (the reindex layer).
    """
    cfg = _two_mla_cfg(("full", "reindex"))
    counts = _model_topk_counts(cfg, T=40)
    assert counts[5] == 1, f"pool must be built exactly once, got {dict(counts)}"
    assert counts[4] == 2, f"each MLA layer selects its top_k once, got {dict(counts)}"


def test_reuse_layers_do_not_reselect():
    """A ``reuse`` layer scores nothing: only the builder's top_k is left."""
    cfg = _two_mla_cfg(("full", "reuse"))
    counts = _model_topk_counts(cfg, T=40)
    assert counts[5] == 1, f"pool must be built exactly once, got {dict(counts)}"
    assert counts[4] == 1, f"reuse must not select again, got {dict(counts)}"


def test_pool_off_leaves_the_adr009_path_untouched(cfg):
    """``mla_pool_size = 0`` keeps the ADR-009 D2 selection exactly as it was."""
    off = dataclasses.replace(cfg, attn_dense_reference=False, swa_window=8, mla_top_k=4)
    counts = _model_topk_counts(off, T=40)
    # small_config has a single MLA layer: one selection, and no pool build.
    assert counts[4] == 1, dict(counts)
    assert 5 not in counts, dict(counts)


def test_pinned_config_pool_layout_at_skeleton_scale():
    """The pinned 1B config runs one pool build and one selection per MLA layer.

    Traced abstractly (``eval_shape``, no allocation) with the dense oracle
    switched off — that is the code the flag-off run executes.  The pool width
    (16) and the selection width (512) are distinct from each other and from the
    LatentMoE router's ``moe_top_k`` (2), so the counts are unambiguous: one
    level-1 pool ``top_k`` for the whole forward, six level-2 selections.
    """
    import json
    from pathlib import Path

    from net.config import load_config, validate_config

    data = json.loads((Path(__file__).resolve().parent.parent / "config.json").read_text(encoding="utf-8"))
    cfg = dataclasses.replace(load_config(Path(__file__).resolve().parent.parent / "config.json"),
                              attn_dense_reference=False)
    validate_config(cfg)

    assert len(data["mla_layer_modes"]) == data["num_mla_layers"]
    assert data["mla_layer_modes"][0] == "full"
    assert [mla.layer_mode(cfg, i) for i in range(cfg.num_mla_layers)] == list(data["mla_layer_modes"])

    counts = _abstract_topk_counts(cfg, T=1024)
    assert counts[data["mla_pool_size"]] == 1, dict(counts)
    assert counts[data["mla_top_k"]] == data["num_mla_layers"], dict(counts)


def test_reindex_without_a_pool_falls_back_to_full(cfg):
    """A consumer with no published pool selects the full prefix, not garbage."""
    modes = dataclasses.replace(
        cfg, attn_dense_reference=False, swa_window=0, mla_top_k=4,
        mla_pool_block=8, mla_pool_size=5, mla_layer_modes=("reindex",),
    )
    params = mla.init_mla(jr.PRNGKey(5), cfg)
    x = jr.normal(jr.PRNGKey(6), (1, 40, cfg.hidden))
    out, pool = mla.apply_with_pool(params, modes, x, mode="reindex")
    plain = mla.apply(params, dataclasses.replace(modes, mla_pool_size=0, mla_layer_modes=()), x)
    assert float(jnp.max(jnp.abs(out - plain))) <= 1e-4
    # Having fallen back to ``full`` it also becomes a builder: the pool it
    # publishes is what the layers after it consume.
    assert pool is not None and pool.blocks.shape[-1] == 5
