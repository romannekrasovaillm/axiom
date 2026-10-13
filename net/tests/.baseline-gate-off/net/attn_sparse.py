"""Sparse-selection and sliding-window attention primitives (ADR-009, D1/D2).

Both mechanisms process queries in blocks so that the activations never
materialise a ``(T, T)`` score matrix: the whole point of the delta is that the
attention cost grows as ``O(T * (top_k + n_win))`` rather than ``O(T^2)``
(V4.1-Flash section 2.3, ADR-009 criteria 5/6).  Blocking over queries keeps the
peak activation at ``O(top_k + n_win)`` per query block and makes the 64K
measurement (criterion 16) allocatable in fp32 on CPU.

``window_attention`` is the shared local branch (D1): exact causal softmax over
the last ``window`` positions, gathered per query.  ``sparse_union_attention``
is the MLA branch (D2): a light indexer scores every causal record, ``top_k``
records are selected, and attention runs over the union of the selected records
and the local window — duplicates between the two are dropped by mask, so local
accuracy does not depend on the selection.

ADR-012 adds a *hierarchical* selection on top of D2: the first MLA layer
(mode ``full``) also scores the causal prefix **by pool block** and publishes a
shared candidate pool — the top ``m`` blocks by blockwise max (the source's
"общий пул кандидатов", V4.1-Flash section 2.3.2).  The remaining MLA layers
select their own ``top_k`` *inside that pool* instead of over the whole prefix
(``reindex``: own re-scoring of the pool records; ``reuse``: the pool builder's
selection, no scoring), which is what makes the indexer's cost stop growing
with the context length in every layer but the first.  The pool is a config
parameter (``mla_pool_block`` / ``mla_pool_size`` / ``mla_layer_modes``); with
``mla_pool_size = 0`` the path is exactly ADR-009 D2.

ADR-018 groups the history into blocks *before* the selection (Step-5-Preview
model card, see ``mla_block_merge`` in ``net/config.json``).  Aggregation is
mean pooling — the minimal honest choice: it is parameter-free, deterministic
and permutation-invariant, so the delta adds no trainable machinery of its own
and cannot learn its way around the gate.  One merged record per complete block
carries both the main KV and the indexer keys, the block's score is the score of
that merged record (the selection ranks exactly what attention will read), and
attention reads merged records instead of raw ones — which is where the
per-record indexer and top-k cost is given up.  A block is eligible only when
all of its records precede the query (a partially causal block would pool the
future), so the query's own partially filled block stays the window branch's
job and the mechanism requires ``swa_window >= block`` (checked in
``net.config.validate_config``).  The window copy of a position inside a
selected *eligible* block is masked out — the D2 duplicate rule lifted to
blocks; an ineligible block the ranking padded itself with marks nothing, or the
very tail the window is there to cover would be lost twice over.  The
mechanism is *declarative*: :func:`net.config.declared_block_merge` reads
``mla_block_merge`` from ``net/config.json`` (spine AD-9, guard C-035) and
:func:`sparse_union_attention` consumes it as the default of its ``block_merge``
parameter — the flag is flipped in the config, never by editing this path; an
explicit ``block_merge`` argument overrides the declaration for tests and A/B.
The reader sits in the schema module (which imports no numerics) so the guard
can compare declaration and code on a plain ``python3`` without jax.  With the
flag off the unmerged path is bit-for-bit what it was.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import compute_dtype, topk_exact
from .config import declared_block_merge

# Query-block size for the blocked scan.  Small enough that the largest
# gathered activation (B, block, top_k + window, H, dq) stays well under a few
# hundred MB at the skeleton scale, large enough to keep the scan short.
_QUERY_BLOCK = 128


def _n_blocks(T: int, block: int) -> int:
    return max(1, -(-T // block))


def mean_blocks(x: jnp.ndarray, block: int, n_blocks: int) -> jnp.ndarray:
    """Mean of each complete block of ``block`` records along the time axis.

    ``x`` is ``(B, T, ...)`` and the result is ``(B, n_blocks, ...)`` with
    ``n_blocks = T // block``: the trailing partial block is dropped, because
    its remaining records are the query's own partially filled block — causal
    eligibility excludes it and the window branch covers it (ADR-018).  Mean
    pooling is the aggregation the mechanism declares: no parameters, no learned
    gate, exactly reproducible, and the score of a merged record is the mean of
    the scores of the records it merges (so the selection ranks the same
    representation attention reads).
    """
    if n_blocks < 1:
        return x[:, :0]
    cut = x[:, : n_blocks * block]
    shape = (cut.shape[0], n_blocks, block) + cut.shape[2:]
    return cut.reshape(shape).mean(axis=2)


def merged_block_selection(
    idx_q: jnp.ndarray,
    idx_k_merged: jnp.ndarray,
    pos: jnp.ndarray,
    block: int,
    top_k: int,
    exact_topk: bool = True,
) -> jnp.ndarray:
    """Selected merged-block ids per query (ADR-018).

    ``idx_q`` is ``(B, Q, Hi, Di)``, ``idx_k_merged`` the pooled indexer keys
    ``(B, n_blocks, Hi, Di)`` and ``pos`` the ``(Q,)`` query positions.  Returns
    ``(B, Q, min(top_k, n_blocks))`` block ids, ranked by the merged records'
    own indexer affinity.

    Eligibility is causal at block granularity: a block is a candidate for a
    query at ``p`` only when every one of its records precedes it
    (``(j + 1) * block <= p + 1``), since the merged representation is the mean
    over the block and a partially causal block would mix future records into
    it.  Ineligible blocks keep ``-inf`` and are ranked last; the caller masks
    whatever the padding selects (``sparse_valid`` below).
    """
    n_blk = idx_k_merged.shape[1]
    width = min(int(top_k), int(n_blk))
    scores = compute_dtype.gemm_einsum(
        "bqhi,bjhi->bqj", idx_q.astype(jnp.float32), idx_k_merged.astype(jnp.float32)
    ) / idx_q.shape[2]
    ids = jnp.arange(n_blk)[None, :]
    eligible = (ids + 1) * int(block) <= (pos + 1)[:, None]  # (Q, n_blocks)
    scores = jnp.where(eligible[None, :, :], scores, jnp.finfo(jnp.float32).min)
    if exact_topk:
        return topk_exact.topk_indices(scores, width)
    return jax.lax.top_k(scores, width)[1]


def _block_duplicates(
    block_idx: jnp.ndarray,
    start: jnp.ndarray,
    block: int,
    window: int,
    valid: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """``(B, Q, W)`` mask of window slots covered by a selected merged block.

    Each merged block occupies ``block`` consecutive records, so its window
    slots are ``block_idx * block - start + arange(block)``; a slot is marked
    when that offset lands inside ``[0, W)`` — the merged record covers it, so
    the window's raw copy is the duplicate.  ``start`` is the first window
    position of each query, as in :func:`_window_duplicates`.

    This is the D2 duplicate rule (ADR-009) lifted from records to blocks: the
    surviving copy is the *main* one, only here the main copy is the merged
    representation.  The scatter is ``O(Q * k_blocks * block)`` — with a block
    of 16 and ``top_k`` blocks that stays far below the gathered KV tensors the
    same step materialises.

    ``valid`` — the ``(B, Q, k_blocks)`` eligibility of each selected block (the
    caller's ``sparse_valid``) — matters because a *selection* of ``top_k``
    blocks is not always a selection of ``top_k`` *eligible* ones: when fewer
    blocks are eligible than ``top_k`` (an early query), the ranking pads
    itself with ineligible blocks, which the sparse branch then drops.  Such a
    block must not take the window down with it: the padding block that sits
    over the window is exactly the query's own partially filled block, and its
    causal prefix is the window branch's job (ADR-018).  An ineligible block
    therefore marks nothing; the default ``None`` treats every id as valid,
    which is the pure geometric reading of the rule (the unit test's case).
    """
    off = (
        block_idx[..., None] * int(block)
        - start[None, :, None, None]
        + jnp.arange(int(block))
    )  # (B, Q, k_blocks, block) — the trailing axis is explicit, block = 1 included
    inside = (off >= 0) & (off < int(window))
    if valid is not None:
        inside = inside & valid[..., None]
    slot = jnp.clip(off, 0, int(window) - 1)
    B, Q, _ = block_idx.shape
    dup = jnp.zeros((B, Q, int(window)), dtype=jnp.bool_)
    return dup.at[
        jnp.arange(B)[:, None, None, None], jnp.arange(Q)[None, :, None, None], slot
    ].max(inside)


def _pos_grid(T: int, block: int, n_blocks: int):
    """Static (block,)-shaped positions for block ``i`` (clipped to ``T-1``)."""
    return jnp.arange(block)


def _gather_time(x: jnp.ndarray, idx: jnp.ndarray) -> jnp.ndarray:
    """Gather along the time axis per batch element.

    ``x`` is (B, T, H, D) and ``idx`` is (B, Q, K); returns (B, Q, K, H, D).
    ``x[:, idx]`` would tile the batch axis, so the batch offset is folded into
    a flat index instead.
    """
    B, T, H, D = x.shape
    Q, K = idx.shape[1], idx.shape[2]
    off = (jnp.arange(B) * T)[:, None, None]
    flat = (idx + off).reshape(B, Q * K)
    return jnp.take(x.reshape(B * T, H, D), flat.reshape(-1), axis=0).reshape(B, Q, K, H, D)


def _window_duplicates(top_idx: jnp.ndarray, start: jnp.ndarray, W: int) -> jnp.ndarray:
    """``(B, Q, W)`` mask of window slots already taken by the indexer.

    A window slot is a duplicate when the record it points at is also in that
    query's own ``top_k``: the window's main-KV copy is the one that survives, so
    the slot must be masked out of the softmax.  ``start`` is the first window
    position of each query (``pos_c - (W - 1)``), so a selected record at
    position ``p`` lands in slot ``p - start`` whenever that is inside the window.

    Scattering the ``k_eff`` selected records into their window slots is
    ``O(Q * k_eff)``; the pairwise ``(Q, W, k_eff)`` comparison it replaces is
    ``W`` times larger and materialises a ``(B, Q, W, k_eff)`` predicate per
    query block.  Duplicates land in the same slot, so the scatter is a ``max``
    (any selected record in the slot marks it) rather than a ``set``.
    """
    B, Q, k_sel = top_idx.shape
    off = top_idx - start[None, :, None]  # (B, Q, k_sel): slot of each selected record
    inside = (off >= 0) & (off < W)
    slot = jnp.clip(off, 0, W - 1)
    dup = jnp.zeros((B, Q, W), dtype=jnp.bool_)
    return dup.at[jnp.arange(B)[:, None, None], jnp.arange(Q)[None, :, None], slot].max(inside)


class CandidatePool(NamedTuple):
    """Shared candidate pool published by an MLA layer in mode ``full`` (ADR-012).

    ``blocks`` is the level-1 selection: ``(B, T, m)`` block ids (into the
    global grid of ``mla_pool_block`` records) per query, chosen by blockwise
    max of the builder's indexer scores.  ``selection`` is the builder's own
    ``top_k`` record indices ``(B, T, k_eff)`` — only materialised when a later
    layer is in mode ``reuse`` (it is what ``reuse`` consumes instead of
    scoring); it is ``None`` otherwise, so the pool costs nothing extra.
    """

    blocks: jnp.ndarray
    selection: jnp.ndarray | None


def _blockwise_max(sc: jnp.ndarray, pool_block: int) -> jnp.ndarray:
    """Max indexer score per pool block — the level-1 block score (ADR-012).

    ``sc`` is ``(B, Q, T)`` with ``-inf`` outside the causal prefix.  Returns
    ``(B, Q, n_blocks)`` with ``n_blocks = ceil(T / pool_block)``; a block whose
    records are all masked keeps ``-inf`` and is ranked last by the level-1
    ``top_m``, so the causal prefix is respected without a second mask.
    """
    B, Q, T = sc.shape
    nb = _n_blocks(T, pool_block)
    pad = nb * pool_block - T
    if pad:
        sc = jnp.pad(sc, ((0, 0), (0, 0), (0, pad)), constant_values=jnp.finfo(sc.dtype).min)
    return sc.reshape(B, Q, nb, pool_block).max(axis=-1)


def _expand_pool(blocks: jnp.ndarray, pool_block: int, T: int):
    """Pool block ids ``(B, Q, m)`` -> ``(records, raw)`` of width ``m*pool_block``.

    ``raw`` keeps the unclipped positions so validity can be tested against
    ``T`` (the last block may be padded); ``records`` is clipped to ``[0, T-1]``
    for the gather.
    """
    raw = blocks[..., None] * pool_block + jnp.arange(pool_block)  # (B, Q, m, pb)
    B, Q, m, pb = raw.shape
    return jnp.clip(raw, 0, T - 1).reshape(B, Q, m * pb), raw.reshape(B, Q, m * pb)


def _reindex_from_pool(idx_q, idx_k, records, raw, pos_c, k_eff, T, neg, exact_topk=True):
    """Reindex: score only the pool records, select ``top_k`` among them.

    Own indexer queries against the pool's indexer keys, so the scored width is
    ``|pool|`` instead of ``T`` (ADR-012 level 2: "своя переоценка скоров
    внутри пула").  Returns the selected record positions
    ``(B, Q, min(k_eff, |pool|))``.  ``exact_topk`` selects between the tiled
    exact selection of :mod:`net.topk_exact` and ``jax.lax.top_k`` (same set of
    records; only the selection algorithm differs).
    """
    Hi = idx_q.shape[2]
    kg = _gather_time(idx_k, records)  # (B, Q, P, Hi, Di)
    scp = compute_dtype.gemm_einsum("bqhi,bqphi->bqp", idx_q, kg) / Hi  # (B, Q, P)
    valid = (raw < T) & (raw <= pos_c[:, None])  # causal + real records
    scp = jnp.where(valid, scp, neg)
    kk = min(int(k_eff), scp.shape[-1])
    order = topk_exact.topk_indices(scp, kk) if exact_topk else jax.lax.top_k(scp, kk)[1]
    return jnp.take_along_axis(records, order, axis=2)


def _dense_causal_blocked(
    q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, block: int
) -> jnp.ndarray:
    """Memory-bounded exact causal softmax over the full prefix.

    Used as the ``top_k >= T`` fast path of :func:`sparse_union_attention`
    (criterion 13): when the selection width covers the whole sequence, the
    indexer's ``top_k`` over ``T`` records degenerates to "select everything",
    the window is a duplicate subset that the union mask drops entirely, and the
    result is exactly dense causal attention.  Running the indexer and a full
    ``O(T^2 log T)`` ``top_k`` here would be pure overhead — and would silently
    break criterion 13 if the selection ever moved to an approximate top-k.
    """
    B, T, H, D = q.shape
    scale = 1.0 / jnp.sqrt(jnp.asarray(D, q.dtype))
    neg = jnp.finfo(jnp.float32).min
    blk = min(int(block), T)
    n = _n_blocks(T, blk)
    all_pos = jnp.arange(T)

    def step(_carry, i):
        s = i * blk
        pos = s + jnp.arange(blk)
        pos_c = jnp.minimum(pos, T - 1)
        alive = pos < T
        qb = jnp.take(q, pos_c, axis=1)
        causal = all_pos[None, :] <= pos_c[:, None]
        sc = compute_dtype.gemm_einsum("bqhd,bshd->bqhs", qb, k) * scale
        sc = jnp.where(causal[None, :, None, :], sc, neg)
        attn = jax.nn.softmax(sc, axis=-1)
        o = compute_dtype.gemm_einsum("bqhs,bshd->bqhd", attn, v)
        return None, jnp.where(alive[None, :, None, None], o, 0.0)

    _, ys = jax.lax.scan(step, None, jnp.arange(n))
    ys = jnp.transpose(ys, (1, 0, 2, 3, 4)).reshape(B, n * blk, H, D)
    return ys[:, :T]


def window_attention(q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, window: int) -> jnp.ndarray:
    """Causal sliding-window attention over ``q, k, v`` of shape (B, T, H, D).

    For query ``i`` the attended positions are ``[i - window + 1, i]`` clipped to
    ``[0, T - 1]``; the softmax is exact over that set (D1).  ``window = 0``
    yields zeros (the branch is off).
    """
    B, T, H, D = q.shape
    if window <= 0:
        return jnp.zeros_like(q)
    W = min(int(window), T)
    scale = 1.0 / jnp.sqrt(jnp.asarray(D, q.dtype))
    block = min(int(_QUERY_BLOCK), T)
    n = _n_blocks(T, block)
    q = q.astype(jnp.float32)
    k = k.astype(jnp.float32)
    v = v.astype(jnp.float32)

    def step(_carry, i):
        s = i * block
        pos = s + _pos_grid(T, block, n)  # (Q,)
        pos_c = jnp.minimum(pos, T - 1)
        rel = jnp.arange(W)[None, :] - (W - 1)  # (1, W), <= 0
        widx = pos_c[:, None] + rel  # (Q, W)
        valid = (pos[:, None] < T) & (widx >= 0) & (widx <= pos_c[:, None])
        widx = jnp.clip(widx, 0, T - 1)
        wb = jnp.broadcast_to(widx[None], (B,) + widx.shape)  # (B, Q, W)
        qb = jnp.take(q, pos_c, axis=1)  # (B, Q, H, D)
        kg = _gather_time(k, wb)  # (B, Q, W, H, D)
        vg = _gather_time(v, wb)
        scores = compute_dtype.gemm_einsum("bqhd,bqwhd->bqhw", qb, kg) * scale
        neg = jnp.finfo(jnp.float32).min
        scores = jnp.where(valid[None, :, None, :], scores, neg)
        attn = jax.nn.softmax(scores, axis=-1)
        return None, compute_dtype.gemm_einsum("bqhw,bqwhd->bqhd", attn, vg)

    _, ys = jax.lax.scan(step, None, jnp.arange(n))
    ys = jnp.transpose(ys, (1, 0, 2, 3, 4)).reshape(B, n * block, H, D)
    return ys[:, :T]


def sparse_union_attention(
    q: jnp.ndarray,
    k_main: jnp.ndarray,
    v_main: jnp.ndarray,
    k_swa: jnp.ndarray | None,
    v_swa: jnp.ndarray | None,
    idx_q: jnp.ndarray | None,
    idx_k: jnp.ndarray | None,
    top_k: int,
    window: int,
    window_bias: jnp.ndarray,
    *,
    mode: str = "full",
    pool: CandidatePool | None = None,
    pool_block: int = 0,
    pool_size: int = 0,
    want_selection: bool = False,
    fused: bool = True,
    exact_topk: bool = True,
    block_merge: int | None = None,
) -> tuple[jnp.ndarray, CandidatePool | None]:
    """Sparse main-KV selection unioned with the SWA window (ADR-009 D2 + ADR-012).

    Args:
        q, k_main, v_main: (B, T, H, D) query and main KV from the latent.
        k_swa, v_swa: (B, T, H, D) local-window KV, or None when ``window == 0``.
        idx_q: (B, T, Hi, Di) indexer queries (from the hidden state); None in
            mode ``reuse`` (that mode does not score at all).
        idx_k: (B, T, Hi, Di) indexer keys (projected from the main KV latent);
            None in mode ``reuse``.
        top_k: selection width; the effective width is ``min(top_k, T)``.
        window: local window size.
        window_bias: learnable per-layer scalar added to the window logits (the
            learnable window share of D1).
        mode: selection mode (ADR-012) — ``full`` scores the whole causal
            prefix and (when ``pool_size > 0``) publishes the candidate pool;
            ``reindex`` scores only the pool records; ``reuse`` takes the pool
            builder's selection.  A ``reindex``/``reuse`` layer without a pool
            falls back to ``full`` (exact — the pool can never make the result
            *unlike* the full prefix selection when it is absent).
        pool: the pool published by the earlier ``full`` layer, or None.
        pool_block: record block size of the pool (``mla_pool_block``).
        pool_size: ``m``, number of blocks in the pool (0 disables the pool).
        want_selection: also publish the builder's own ``top_k`` indices — set
            only when a later layer is in mode ``reuse``, since the array is
            ``O(T * top_k)``.
        fused: assemble the union attention without materialising the gathered
            ``(B, Q, k_eff, H, D)`` keys/values (see :func:`_mark_positions` and
            the per-branch value reduction below).  ``False`` runs the verbatim
            unfused assembly — same selection, same arithmetic — for A/B timing
            and cross-checks.
        exact_topk: select the records with :func:`net.topk_exact.topk_indices`
            (tiled exact merge) instead of ``jax.lax.top_k``.  The two select the
            same records (ties broken arbitrarily in both); the flag exists to
            A/B the selection *algorithm*, not the selection.
        block_merge: records per merged block (ADR-018), or ``None`` for "read
            the declared value from ``net/config.json``" (see
            :func:`net.config.declared_block_merge`).  ``<= 0``, a sequence shorter than one
            block, an absent window branch or an indexer-less layer (mode
            ``reuse``) all keep the unmerged path: the merged path needs a block
            to merge, the window to cover the query's own partially filled block
            and indexer scores to rank blocks with.  With ``block_merge = 1`` the
            mechanism degenerates to the unmerged path exactly (T-m1).

    Returns:
        ``(out, pool_out)``: (B, T, H, D) attention output over the *union* of
        the selected records and the window (duplicate positions keep the
        main-KV copy, the window copy is masked out), and the pool to hand to
        the next MLA layer (None unless this layer is the pool builder).

    Block-wise merging (``block_merge > 0``) replaces the per-record selection
    with a per-block one: the indexer scores one merged record per complete block
    and attention reads the merged records, so the selected width is ``top_k``
    *blocks* rather than ``top_k`` records.  The merged path is its own
    full-prefix selection: it neither publishes nor consumes the ADR-012 pool
    (that pool is a record-level mechanism — combining the two is a separate
    decision, not this delta), so it returns ``pool_out = None``.
    """
    B, T, H, D = q.shape
    k_eff = min(int(top_k), T)
    scale = 1.0 / jnp.sqrt(jnp.asarray(D, q.dtype))
    block = min(int(_QUERY_BLOCK), T)
    n = _n_blocks(T, block)
    q = q.astype(jnp.float32)
    k_main = k_main.astype(jnp.float32)
    v_main = v_main.astype(jnp.float32)
    if idx_q is not None:
        idx_q = idx_q.astype(jnp.float32)
        idx_k = idx_k.astype(jnp.float32)
        Hi = idx_q.shape[2]
    dtype = q.dtype
    neg = jnp.finfo(jnp.float32).min
    all_pos = jnp.arange(T)

    # Criterion 13 fast path: ``top_k >= T`` selects the whole causal prefix, so
    # the sparse union reduces to dense causal attention (the window is fully
    # masked as a duplicate subset).  Exact, and avoids an ``O(T^2 log T)``
    # ``top_k`` over ``T`` records that would be pure overhead.  It applies
    # regardless of the pooling layout: the premise of criterion 13 is that the
    # selection covers the prefix, and no pool is needed to reproduce it.
    if k_eff >= T:
        return _dense_causal_blocked(q, k_main, v_main, block).astype(dtype), None

    # A consumer without the pool it needs (no builder before it, or a builder
    # that did not publish its selection) falls back to ``full``: an exact
    # selection is always available, so the fallback can never be *less* faithful
    # than the full prefix — only more expensive.
    if mode == "reindex" and pool is None:
        mode = "full"
    if mode == "reuse" and (pool is None or pool.selection is None):
        mode = "full"
    pooling = int(pool_size) > 0 and mode == "full"
    m_eff = min(int(pool_size), _n_blocks(T, pool_block)) if pooling else 0
    sel_width = k_eff  # width of the published builder selection (mode ``full``)

    if window > 0:
        W = min(int(window), T)
        k_swa = k_swa.astype(jnp.float32)
        v_swa = v_swa.astype(jnp.float32)
    else:
        W = 0
        k_swa = v_swa = None

    # --- block-wise token merging (ADR-018) --------------------------------
    # ``None`` means "whatever the declaration says": the case config is the
    # switch (spine AD-9, guard C-035), an explicit width is the test/A-B hook.
    if block_merge is None:
        block_merge = declared_block_merge()
    merge_block = int(block_merge)
    n_blk = T // merge_block if merge_block > 0 else 0
    # The merged path needs something to merge (a complete block), the window to
    # cover the query's own partially filled block (otherwise up to
    # ``block - 1`` records before the window would go unattended) and indexer
    # scores to rank blocks with — a ``reuse`` layer renders none, so it keeps
    # the unmerged path.  ``k_eff >= T`` never reaches this point (the criterion
    # 13 fast path above returns first): a selection covering the prefix is the
    # dense oracle, block merging or not.
    merge = (
        merge_block > 0 and n_blk > 0 and W >= merge_block and idx_q is not None
    )
    if merge:
        # The merged candidates are the complete blocks of the prefix; the
        # selection scans them (not the records), which is the saving the
        # mechanism buys, and it is its own full-prefix selection, so the
        # ADR-012 pool is neither published nor consumed here.
        k_merged = mean_blocks(k_main, merge_block, n_blk)
        v_merged = mean_blocks(v_main, merge_block, n_blk)
        idx_k_merged = mean_blocks(idx_k, merge_block, n_blk)
        mode = "full"
        pooling = False

    def step(_carry, i):
        s = i * block
        pos = s + _pos_grid(T, block, n)  # (Q,)
        pos_c = jnp.minimum(pos, T - 1)
        alive = pos < T  # padded tail queries

        # --- selection: merged blocks / full prefix / in-pool ---------------
        blocks_b = jnp.zeros((B, pos_c.shape[0], max(m_eff, 1)), jnp.int32)
        sel_b = jnp.zeros((B, pos_c.shape[0], sel_width), jnp.int32)
        if merge:
            qi = jnp.take(idx_q, pos_c, axis=1)  # (B, Q, Hi, Di)
            top_idx = merged_block_selection(
                qi, idx_k_merged, pos_c, merge_block, k_eff, exact_topk=exact_topk
            )
            # Causality at block granularity: a block whose records are all
            # before the query.  The padding the top-k fills with is masked out
            # of the softmax here (the same role ``sparse_valid`` plays below).
            sparse_valid = ((top_idx + 1) * merge_block) <= (pos_c + 1)[None, :, None]
        elif mode == "reuse":
            # No scoring at all: the pool builder's selection is reused verbatim.
            top_idx = jnp.take(pool.selection, pos_c, axis=1)  # (B, Q, k_eff)
        else:
            qi = jnp.take(idx_q, pos_c, axis=1)  # (B, Q, Hi, Di)
            if mode == "reindex":
                blocks = jnp.take(pool.blocks, pos_c, axis=1)  # (B, Q, m)
                records, raw = _expand_pool(blocks, pool_block, T)
                top_idx = _reindex_from_pool(
                    qi, idx_k, records, raw, pos_c, k_eff, T, neg, exact_topk=exact_topk
                )
            else:  # full: score the whole causal prefix
                sc = compute_dtype.gemm_einsum("bqhi,bshi->bqs", qi, idx_k) / Hi  # (B, Q, T)
                causal = all_pos[None, :] <= pos_c[:, None]  # (Q, T)
                sc = jnp.where(causal[None, :, :], sc, neg)
                if exact_topk:
                    top_idx = topk_exact.topk_indices(sc, k_eff)  # (B, Q, k_eff)
                else:
                    top_idx = jax.lax.top_k(sc, k_eff)[1]  # (B, Q, k_eff)
                if pooling:
                    # ADR-012 level 1: block score = max score in the block,
                    # pool = its top-m blocks ("общий пул кандидатов").
                    bs = _blockwise_max(sc, pool_block)  # (B, Q, nb)
                    blocks_b = jax.lax.top_k(bs, m_eff)[1]
                    if want_selection:
                        sel_b = top_idx
        if not merge:
            sparse_valid = top_idx <= pos_c[:, None]  # (B, Q, k_sel)

        # --- window records, duplicates removed by mask --------------------
        if W > 0:
            rel = jnp.arange(W)[None, :] - (W - 1)
            widx_raw = pos_c[:, None] + rel  # (Q, W)
            wvalid = (widx_raw >= 0) & (widx_raw <= pos_c[:, None]) & alive[:, None]
            widx = jnp.clip(widx_raw, 0, T - 1)
            # a position already selected by the indexer keeps its main-KV copy
            if merge:
                # Block granularity: every window slot inside a selected merged
                # block is that block's duplicate (the merged record survives).
                # Only the *eligible* selected blocks count: a padding block the
                # sparse branch drops must not mask its raw window copy either,
                # or the query's own partially filled block would go unattended
                # (early queries have fewer eligible blocks than ``top_k``).
                dup = _block_duplicates(
                    top_idx, pos_c - (W - 1), merge_block, W, valid=sparse_valid
                )
            elif fused:
                # Scatter the selected records into their window slots once,
                # instead of comparing every window slot with every selected
                # index: O(Q * k_eff) against O(Q * W * k_eff).
                dup = _window_duplicates(top_idx, pos_c - (W - 1), W)  # (B,Q,W)
            else:
                dup = jnp.any(widx_raw[None, :, :, None] == top_idx[:, :, None, :], axis=-1)
            wvalid = wvalid[None, :, :] & ~dup

        k_sel = top_idx.shape[-1]
        mkg = _gather_time(k_merged if merge else k_main, top_idx)  # (B, Q, k_sel, H, D)
        mvg = _gather_time(v_merged if merge else v_main, top_idx)
        qb = jnp.take(q, pos_c, axis=1)  # (B, Q, H, D)
        o_s = compute_dtype.gemm_einsum("bqhd,bqkhd->bqhk", qb, mkg) * scale
        o_s = jnp.where(sparse_valid[..., None, :], o_s, neg)
        if W > 0:
            wb = jnp.broadcast_to(widx[None], (B,) + widx.shape)
            skg = _gather_time(k_swa, wb)  # (B, Q, W, H, D)
            svg = _gather_time(v_swa, wb)
            o_w = compute_dtype.gemm_einsum("bqhd,bqwhd->bqhw", qb, skg) * scale + window_bias
            o_w = jnp.where(wvalid[..., None, :], o_w, neg)

        if fused:
            # The union softmax is over the concatenated *scores* (small:
            # ``(B, Q, H, k_sel+W)``); each branch's weights are then applied to
            # its own gathered values.  That is the same sum of the same terms as
            # ``attn @ concat(v_main, v_swa)`` — only the association of the value
            # reduction changes (~1e-7 relative) — but the gathered keys/values
            # keep a single consumer, so the ``(B, Q, k_sel, H, D)`` tensors are
            # never materialised.  See ``test_fused_assembly_*``.
            score = (
                jnp.concatenate([o_s, o_w], axis=-1) if W > 0 else o_s
            )  # (B, Q, H, k_sel+W)
            attn = jax.nn.softmax(score, axis=-1)
            o = compute_dtype.gemm_einsum("bqhk,bqkhd->bqhd", attn[..., :k_sel], mvg)
            if W > 0:
                o = o + compute_dtype.gemm_einsum("bqhw,bqwhd->bqhd", attn[..., k_sel:], svg)
        else:
            vals = [mvg]
            scores = [o_s]
            if W > 0:
                vals.append(svg)
                scores.append(o_w)
            score = jnp.concatenate(scores, axis=-1)  # (B, Q, H, k_sel+W)
            val = jnp.concatenate(vals, axis=2)  # (B, Q, k_sel+W, H, D)
            attn = jax.nn.softmax(score, axis=-1)
            o = compute_dtype.gemm_einsum("bqhs,bqshd->bqhd", attn, val)
        o = jnp.where(alive[None, :, None, None], o, 0.0)
        zero = jnp.zeros((B, pos_c.shape[0], 0), jnp.int32)
        return None, (
            o,
            blocks_b[:, :, :m_eff] if pooling else zero,
            sel_b[:, :, :sel_width] if (pooling and want_selection) else zero,
        )

    _, (ys, pblocks, psel) = jax.lax.scan(step, None, jnp.arange(n))
    ys = jnp.transpose(ys, (1, 0, 2, 3, 4)).reshape(B, n * block, H, D)
    out = ys[:, :T].astype(dtype)
    if merge:
        # A merged layer publishes no pool: its selection is block-level, and a
        # record-level consumer without a pool falls back to its own exact
        # full-prefix selection (above), never to a stale one.
        return out, None
    if not pooling:
        # A consumer layer hands the *same* pool on to the next MLA layer.
        return out, pool
    blocks_out = jnp.transpose(pblocks, (1, 0, 2, 3)).reshape(B, n * block, m_eff)[:, :T]
    sel_out = None
    if want_selection:
        sel_out = jnp.transpose(psel, (1, 0, 2, 3)).reshape(B, n * block, sel_width)[:, :T]
    return out, CandidatePool(blocks=blocks_out, selection=sel_out)
