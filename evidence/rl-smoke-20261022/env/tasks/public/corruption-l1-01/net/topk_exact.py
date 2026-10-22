"""Exact top-k selection by tiling + exact merge (criterion 16 delta).

``jax.lax.top_k`` on the GPU backend does not have a selection kernel: XLA lowers
it to a **full bitonic sort** of the whole row with an iota payload, then slices
the first ``k``.  For the builder's selection (``T = 8192``, ``k = 512``) that is
a 91-substage network on the full row plus the marshalling of the
``(value, index)`` operand pair (the ``input_transpose_fusion`` /
``input_concatenate_fusion`` kernels in the profile), to produce 6% of the row.

This module computes the *same* selection with a tiling + merge scheme, without
changing what is selected:

* the row is cut into ``ceil(T / chunk)`` tiles, each tile is sorted descending
  carrying its positions, and truncated to its top ``min(k, chunk)`` records;
* tiles are merged pairwise, every merge keeping only its top ``k`` — exact,
  because a record outside a part's top ``k`` can never be in the union's top
  ``k``;
* the last merge is ``jax.lax.top_k`` over the two surviving lists, so the
  selection still contains exactly one ``top_k`` primitive of width ``k`` — the
  invariant pinned by ``net/tests/test_17_indexer_shortcut.py``.

Exactness is by construction: there is no approximation, no sampling and no
threshold.  The only lossy step is the truncation to ``k`` at each merge, and
that truncation is lossless for the top-``k`` of the union (a merge of two lists
whose top-``k`` are kept computes the top-``k`` of the union exactly).

Ties: both this kernel and ``jax.lax.top_k`` return *some* ``k`` records among
equals — the value multiset is identical and exact, the index set is identical
whenever the ``k``-th and ``(k+1)``-th largest values differ (see
``net/tests/test_18_exact_topk.py``).  ``jax.lax.sort`` is deterministic for a
fixed executable, so the selection is reproducible (criterion 17).

The tiling constants below are *implementation* tiling, not mechanism
parameters: no selection width, window, pool layout or threshold depends on
them.  ``_TILES`` targets a balanced merge tree of depth ~3, ``_MAX_CHUNK``
keeps a tile cache-resident at long context and ``_MIN_CHUNK`` keeps the tree
from degenerating into many tiny merge levels on short rows.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# Tiles per row (before clamping), and the tile-size bounds.
_TILES = 8
_MIN_CHUNK = 256
_MAX_CHUNK = 1024


def chunk_for(T: int) -> int:
    """Tile size for a row of length ``T`` (implementation tiling, see module doc)."""
    return max(_MIN_CHUNK, min(_MAX_CHUNK, int(T) // _TILES))


def _sort_desc(vals: jnp.ndarray, idx: jnp.ndarray):
    """Sort along the last axis by ``vals`` descending, carrying ``idx``.

    Descending is ascending on the negated keys: negation is exact on floats and
    reverses the order, so ties stay ties (``-inf`` maps to ``+inf`` and ranks
    last, which is what the causal mask relies on).  ``jax.lax.sort`` on a pair
    of operands with ``num_keys=1`` compares only the first and permutes both —
    a single ``sort`` primitive, no ``top_k``, so the selection keeps its pinned
    ``top_k`` count.
    """
    v, i = jax.lax.sort((-vals, idx), dimension=-1, is_stable=False, num_keys=1)
    return -v, i


def topk_indices(sc: jnp.ndarray, k: int, chunk: int | None = None) -> jnp.ndarray:
    """Exact top-``k`` **indices** along the last axis of ``sc``.

    The result is the same set of records as ``jax.lax.top_k(sc, k)[1]`` up to
    the tie-break among equal scores (documented in the module docstring):
    ``(..., min(k, T))`` int32 indices into the last axis, ordered by descending
    score.  Rows shorter than ``chunk`` fall back to ``jax.lax.top_k`` — the
    tiling would be a single tile anyway.

    Args:
        sc: ``(..., T)`` score array; positions outside the causal prefix are
            expected to be ``-inf`` (they rank last and are dropped first).
        k: selection width; clipped to ``T``.
        chunk: tile size; ``None`` uses :func:`chunk_for`.
    """
    *lead, T = sc.shape
    T = int(T)
    k = min(int(k), T)
    if chunk is None:
        chunk = chunk_for(T)
    chunk = min(int(chunk), T)
    if T <= chunk:
        # Single tile: the row is already the merge result.
        return jax.lax.top_k(sc, k)[1].astype(jnp.int32)

    n_tiles = -(-T // chunk)
    pad = n_tiles * chunk - T
    if pad:
        sc = jnp.concatenate(
            [sc, jnp.full((*lead, pad), -jnp.inf, sc.dtype)], axis=-1
        )
    pos = jnp.broadcast_to(jnp.arange(n_tiles * chunk, dtype=jnp.int32), sc.shape)
    v, i = _sort_desc(sc.reshape(*lead, n_tiles, chunk), pos.reshape(*lead, n_tiles, chunk))
    w = min(k, chunk)
    v, i = v[..., :w], i[..., :w]

    # Pairwise merge, each level keeping the top k of the pair.  An odd tile
    # count is padded with -inf, which never enters a top-k while real records
    # remain (and the caller's causal mask drops them otherwise).
    while v.shape[-2] > 2:
        *L, g, kk = v.shape
        if g % 2:
            v = jnp.concatenate([v, jnp.full((*L, 1, kk), -jnp.inf, v.dtype)], axis=-2)
            i = jnp.concatenate([i, jnp.full((*L, 1, kk), -1, i.dtype)], axis=-2)
            g += 1
        v = v.reshape(*L, g // 2, 2 * kk)
        i = i.reshape(*L, g // 2, 2 * kk)
        v, i = _sort_desc(v, i)
        w = min(k, 2 * kk)
        v, i = v[..., :w], i[..., :w]

    # Final level: at most two sorted lists.  This is the pinned ``top_k`` of
    # width k (test_17 counts it); it is over at most 2 * min(k, ...) records.
    *L, g, kk = v.shape
    vv = v.reshape(*L, g * kk)
    ii = i.reshape(*L, g * kk)
    if vv.shape[-1] < k:  # defensive: cannot happen while k <= T
        extra = k - vv.shape[-1]
        vv = jnp.concatenate([vv, jnp.full((*L, extra), -jnp.inf, vv.dtype)], axis=-1)
        ii = jnp.concatenate([ii, jnp.full((*L, extra), -1, ii.dtype)], axis=-1)
    _, order = jax.lax.top_k(vv, k)
    idx = jnp.take_along_axis(ii, order, axis=-1)
    # Padded tiles carry positions >= T only when the row has fewer than k real
    # records; clip them into range (the caller masks non-causal positions).
    return jnp.clip(idx, 0, T - 1).astype(jnp.int32)
