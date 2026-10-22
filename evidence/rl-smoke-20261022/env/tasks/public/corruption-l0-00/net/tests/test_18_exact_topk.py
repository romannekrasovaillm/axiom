"""Criterion 16 delta — the exact tiled top-k selection (``net/topk_exact.py``).

The selection is an *implementation* of "take the top ``mla_top_k`` records":
the acceptance requirement for the delta is that it selects the same records as
``jax.lax.top_k``, including where the tiling does not divide the row and where
the scores have duplicates or edge values.  These tests pin exactly that, and
the two invariants the delta must not break:

* exactly one ``top_k`` primitive of width ``k`` per selection — the count
  pinned by ``test_17_indexer_shortcut.py`` (the final merge level is a
  ``jax.lax.top_k``, everything else is ``sort``);

The kernel is *exact but not faster* in the MLA stack, so the pinned leg stays
``jax.lax.top_k`` (``attn_topk_exact = false``, see
``docs/research/topk-exact-delta-2026-09-13.md``); ``attn_topk_exact = true`` is
the A/B leg these tests exercise.

Ties are broken arbitrarily by both implementations (sort networks do not
define an order among equals), so the equality asserted is on the *selected
values* — the multiset is exact in every case — and on the *index set* wherever
the ``k``-th and ``(k+1)``-th largest values differ, i.e. wherever the top-k
set is unambiguous.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr

from conftest import small_config

from net import mla, topk_exact

CONFIG = Path(__file__).resolve().parent.parent / "config.json"


# --- helpers -----------------------------------------------------------------


def _descending(sc: jnp.ndarray) -> jnp.ndarray:
    """The row sorted descending (the ground truth the selection must match)."""
    return -jnp.sort(-sc, axis=-1)


def _selected(sc: jnp.ndarray, idx: jnp.ndarray) -> jnp.ndarray:
    return jnp.take_along_axis(sc, idx, axis=-1)


def _assert_same_records(sc: jnp.ndarray, k: int, idx: jnp.ndarray, label: str) -> None:
    """``idx`` selects exactly the ``k`` largest values of ``sc``.

    Checks (1) the selected values equal the ``k`` largest, in order, and
    (2) the index set equals ``jax.lax.top_k``'s whenever no tie straddles the
    boundary (so the set is unambiguous).
    """
    assert idx.shape[-1] == min(k, sc.shape[-1]), label
    assert idx.dtype == jnp.int32, label
    assert int(idx.min()) >= 0 and int(idx.max()) < sc.shape[-1], label
    got = _selected(sc, idx)
    want = _descending(sc)[..., : idx.shape[-1]]
    assert jnp.array_equal(got, want), f"{label}: selected values differ"

    kk = idx.shape[-1]
    T = sc.shape[-1]
    if kk < T:
        srt = _descending(sc)
        kth, next_ = srt[..., kk - 1], srt[..., kk]
        if bool(jnp.all(kth != next_)):  # no boundary tie: the set is unique
            ref = jax.lax.top_k(sc, kk)[1]
            assert jnp.array_equal(
                jnp.sort(idx, axis=-1), jnp.sort(ref, axis=-1)
            ), f"{label}: index set differs from jax.lax.top_k"


def _walk_topk(jaxpr) -> list[int]:
    """Every ``top_k`` width in a (nested) jaxpr, in trace order."""
    widths: list[int] = []

    def walk(j):
        for eq in j.eqns:
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

    walk(jaxpr)
    return widths


# --- equivalence against jax.lax.top_k ---------------------------------------


def test_matches_lax_top_k_on_random_rows():
    """The tiled selection picks the same records on ordinary random rows."""
    sc = jr.normal(jr.PRNGKey(0), (4, 1000))
    for k in (1, 7, 100, 512, 999):
        idx = topk_exact.topk_indices(sc, k, chunk=256)
        _assert_same_records(sc, k, idx, f"random T=1000 k={k}")


def test_matches_lax_top_k_across_lengths_and_chunks():
    """Lengths on both sides of the tile bounds, and tile sizes that divide or not."""
    key = jr.PRNGKey(1)
    for T in (1, 2, 7, 255, 256, 257, 512, 1000, 1025, 4096):
        sc = jr.normal(jr.split(key, 2)[0], (2, T))
        key = jr.split(key, 2)[1]
        for k in {1, 3, max(1, T // 3), max(1, T - 1)}:
            for chunk in (None, 64, 256, 1024):
                idx = topk_exact.topk_indices(sc, k, chunk=chunk)
                _assert_same_records(sc, k, idx, f"T={T} k={k} chunk={chunk}")


def test_row_shorter_than_the_tile_uses_the_single_tile_path():
    """``T <= chunk`` is the declared single-tile branch, and still exact."""
    sc = jr.normal(jr.PRNGKey(2), (3, 200))
    idx = topk_exact.topk_indices(sc, 50, chunk=256)
    _assert_same_records(sc, 50, idx, "T=200 chunk=256")
    assert int(topk_exact.chunk_for(200)) == 256  # the default takes that branch


def test_length_not_divisible_by_the_tile():
    """The padded tail must not leak into the result (T % chunk != 0)."""
    T, chunk = 1000, 256  # 4 tiles: 3 x 256 + 232 real, 24 padded with -inf
    sc = jr.normal(jr.PRNGKey(3), (3, T))
    idx = topk_exact.topk_indices(sc, 300, chunk=chunk)
    _assert_same_records(sc, 300, idx, f"T={T} chunk={chunk}")
    assert int(idx.max()) < T  # no padded position (>= T) survives

    # with the largest values sitting exactly in the last, partial tile
    sc2 = jnp.zeros((1, T)).at[..., 900:].set(jnp.arange(100.0) + 10.0)
    idx2 = topk_exact.topk_indices(sc2, 50, chunk=chunk)
    _assert_same_records(sc2, 50, idx2, "tail-heavy")
    assert bool(jnp.all(idx2 >= T - 100))


def test_duplicates_and_ties_are_selected_exactly():
    """Repeated values: the value multiset is exact, the indices are valid."""
    # all-equal row: any k records are a valid answer
    flat = jnp.ones((2, 600))
    idx = topk_exact.topk_indices(flat, 100, chunk=256)
    _assert_same_records(flat, 100, idx, "all-equal")

    # a tie straddling the k-th boundary, on both sides of a tile
    sc = jnp.zeros((1, 600))
    sc = sc.at[..., :520].set(1.0)  # 520 ones, 80 zeros
    idx = topk_exact.topk_indices(sc, 500, chunk=256)
    got = _selected(sc, idx)
    assert bool(jnp.all(got == 1.0))
    assert len(set(idx[0].tolist())) == 500  # 500 distinct ones, none repeated

    # duplicates spread across every tile
    sc = jnp.tile(jnp.array([3.0, 1.0, 2.0, 2.0, 1.0, 3.0]), (1, 100))
    idx = topk_exact.topk_indices(sc, 120, chunk=64)
    _assert_same_records(sc, 120, idx, "tiled duplicates")


def test_extreme_values():
    """``+/-inf`` and very large/small finite values must order exactly."""
    sc = jnp.array([[-jnp.inf, 0.0, jnp.inf, 1e30, -1e30, 3.0, jnp.inf, -jnp.inf]])
    for k in (1, 2, 3, 5, 8):
        idx = topk_exact.topk_indices(sc, k, chunk=4)
        _assert_same_records(sc, k, idx, f"inf k={k}")
    # the two +inf records must be in the top-2 (ties, so only the values)
    idx2 = topk_exact.topk_indices(sc, 2, chunk=2)
    assert bool(jnp.all(jnp.isinf(_selected(sc, idx2))))


def test_causal_masked_rows_match_the_legacy_selection():
    """The full-prefix shape: ``-inf`` outside the causal prefix (the real use)."""
    T = 1200
    sc = jr.normal(jr.PRNGKey(4), (1, T))
    causal = jnp.arange(T)[None, :] <= jnp.arange(T)[:, None] * 3  # a mask that moves
    sc = jnp.where(causal[None, :, :], sc, jnp.finfo(jnp.float32).min)
    idx = topk_exact.topk_indices(sc, 200, chunk=256)
    _assert_same_records(sc, 200, idx, "causal")
    # Where the prefix holds at least k records, every selected one is inside it.
    # (Shorter prefixes are padded by -inf records in both implementations; the
    # caller's ``sparse_valid`` mask drops those, exactly as for ``jax.lax.top_k``.)
    pos = jnp.arange(T) * 3  # last causal position of each query
    deep = pos >= 200
    assert bool(jnp.all(idx[0, deep] <= pos[deep, None]))


def test_selection_is_deterministic():
    """Same input, same selection — the analogue of criterion 17 for the kernel."""
    sc = jr.normal(jr.PRNGKey(5), (2, 900))
    a = topk_exact.topk_indices(sc, 128, chunk=256)
    b = topk_exact.topk_indices(sc, 128, chunk=256)
    assert bool(jnp.array_equal(a, b))


# --- pinned invariants -------------------------------------------------------


def test_kernel_keeps_exactly_one_top_k_of_width_k():
    """The tiled selection contains one ``top_k`` (width k); the rest is ``sort``."""
    sc = jax.ShapeDtypeStruct((2, 4096), jnp.float32)
    for k in (16, 512, 4095):
        widths = _walk_topk(jax.make_jaxpr(lambda s: topk_exact.topk_indices(s, k))(sc).jaxpr)
        assert widths == [k], f"k={k}: expected exactly one top_k of width k, got {widths}"


def _count_primitive(jaxpr, name: str, **want) -> int:
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


def test_layer_selection_still_has_one_top_k_per_layer(cfg):
    """At a row length above the tile bound the layer keeps the pinned count."""
    sparse = dataclasses.replace(
        cfg, attn_dense_reference=False, swa_window=8, mla_top_k=64, attn_topk_exact=True
    )
    params = mla.init_mla(jr.PRNGKey(6), sparse)
    x = jr.normal(jr.PRNGKey(7), (1, 1024, sparse.hidden))
    closed = jax.make_jaxpr(mla.apply, static_argnums=(1,))(params, sparse, x)
    assert _count_primitive(closed.jaxpr, "top_k", k=64) == 1


def test_the_flag_is_pinned_and_the_legs_agree(cfg):
    """``attn_topk_exact`` is declared in the config; both legs select the same.

    The pinned value is ``False``: the tiled kernel is exact but slower inside
    the MLA stack (see the module docstring), so the legacy ``jax.lax.top_k``
    leg stays the one the pinned config runs.
    """
    data = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert data["attn_topk_exact"] is False
    assert "attn_topk_exact" in cfg.as_dict()

    sc = jr.normal(jr.PRNGKey(8), (2, 900))
    exact = topk_exact.topk_indices(sc, 128)
    legacy = jax.lax.top_k(sc, 128)[1]
    assert jnp.array_equal(_selected(sc, exact), _selected(sc, legacy))


def test_model_path_agrees_with_the_legacy_leg(cfg):
    """End to end through the sparse layer: both legs give the same output."""
    base = dataclasses.replace(
        cfg, attn_dense_reference=False, swa_window=8, mla_top_k=32,
        mla_pool_block=8, mla_pool_size=4,
        mla_layer_modes=("full", "reindex"),
    )
    exact = dataclasses.replace(base, attn_topk_exact=True)
    legacy = dataclasses.replace(base, attn_topk_exact=False)
    params = mla.init_mla(jr.PRNGKey(9), base)
    x = jr.normal(jr.PRNGKey(10), (1, 600, base.hidden))
    a, pool_a = mla.apply_with_pool(params, exact, x, mode="full")
    b, _ = mla.apply_with_pool(params, legacy, x, mode="full")
    assert float(jnp.max(jnp.abs(a - b))) <= 1e-6
    c, _ = mla.apply_with_pool(params, exact, x, pool=pool_a, mode="reindex")
    d, _ = mla.apply_with_pool(params, legacy, x, pool=pool_a, mode="reindex")
    assert float(jnp.max(jnp.abs(c - d))) <= 1e-6
