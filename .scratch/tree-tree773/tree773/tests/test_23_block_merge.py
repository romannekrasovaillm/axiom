"""ADR-018 — block-wise token merging in the MLA layers, behind a config flag.

The mechanism (Step-5-Preview model card, 20.09.2026; ADR-018) aggregates the
history into blocks of ``B`` records *before* the sparse selection: the indexer
scores one merged record per block and attention reads the merged records, so
the indexer cost and the selection granularity stop growing with the record
count.  The delta is **declarative** — the switch is ``mla_block_merge`` in
``net/config.json`` (spine AD-9, guard C-035), not a code edit — and reversible:
with the flag off the unmerged path must be bit-for-bit what it was.

* **T-m1** — flag off ⇒ bit-identical output; the merged machinery with a
  degenerate block of one record reproduces the unmerged path exactly; and the
  criterion-13 fast path (``top_k >= T`` ⇒ dense oracle) is unaffected by the
  flag.
* **T-m2** — flag on, ``B = 16``: finite output, stable shape, at most ``top_k``
  selected blocks, selection = top-``top_k`` of the *eligible* blocks (brute
  force), the window copy of a position inside a selected block is masked (the
  merged record survives), merged records are what attention reads, and the
  declared flag — not a code edit — flips the live layer.
* **T-m3** — a config without the field keeps the unmerged path (backward
  compatible).
* **T-m4** — the declaration exists, is pinned off, and its ``deviations`` line
  names the decision and its source.

T-m5 (the C-035 guard: config ↔ code consistency) lives in
``tools/tests/test_check_declarative_context.py``.

The tests are CPU-only and use hand-sized tensors, so they stay fast; the
merged path is driven through the primitive (``net/attn_sparse.py``) with an
explicit ``block_merge``, and once through ``net/mla.py`` via the *declared*
config file — which is exactly how the mechanism reaches the live stack.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from conftest import small_config

from net import attn_sparse, mla
from net import config as net_config
from net.config import BlockMergeConfig, ModelConfig, load_config, validate_config

CONFIG = Path(__file__).resolve().parent.parent / "config.json"
TOL = 1e-6


# --- helpers -----------------------------------------------------------------


def _primitive_inputs(
    cfg, T: int, B: int = 2, seed: int = 0, window: int = 64, top_k: int = 8,
):
    """Random q/k/v/swa/indexer tensors for one ``sparse_union_attention`` call.

    Built directly (not through ``mla._sparse_apply``) so the merged and
    unmerged legs are handed *identical* inputs — the equivalence claims of
    T-m1 are about the mechanism, not about the projections feeding it.
    """
    keys = jr.split(jr.PRNGKey(seed), 7)
    H, dq = cfg.num_heads, cfg.mla_head_dim
    Hi, Di = cfg.mla_index_heads, cfg.mla_index_dim
    shape = (B, T, H, dq)
    return dict(
        q=jr.normal(keys[0], shape),
        k_main=jr.normal(keys[1], shape),
        v_main=jr.normal(keys[2], shape),
        k_swa=jr.normal(keys[3], shape),
        v_swa=jr.normal(keys[4], shape),
        idx_q=jr.normal(keys[5], (B, T, Hi, Di)),
        idx_k=jr.normal(keys[6], (B, T, Hi, Di)),
        top_k=top_k,
        window=window,
    )


def _run(inputs, *, block_merge, exact_topk: bool = True):
    """One primitive call with a fixed assembly (fused, no ADR-012 pool)."""
    return attn_sparse.sparse_union_attention(
        inputs["q"],
        inputs["k_main"],
        inputs["v_main"],
        inputs["k_swa"],
        inputs["v_swa"],
        inputs["idx_q"],
        inputs["idx_k"],
        top_k=inputs["top_k"],
        window=inputs["window"],
        window_bias=jnp.zeros(()),
        block_merge=block_merge,
        exact_topk=exact_topk,
    )


def _declared_swap(tmp_path: Path, monkeypatch, payload: dict) -> Path:
    """Point the declarative reader at a fixture config (the switch, not code)."""
    path = tmp_path / "declared.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(net_config, "CONFIG_PATH", path)
    return path


# ---------------------------------------------------------------------------
# T-m1 — the flag off is the pre-delta path, bit for bit
# ---------------------------------------------------------------------------


def test_t_m1_flag_off_is_bit_identical_to_the_unmerged_path(cfg):
    """``block_merge = 0`` and the declared default (off) agree exactly.

    The declared case config does not enable the mechanism, so the primitive's
    default (``None`` ⇒ read the declaration) and an explicit ``0`` must both be
    the unmerged path — the rollback gate of the delta.
    """
    assert net_config.declared_block_merge() == 0
    inputs = _primitive_inputs(cfg, T=300)
    declared, _ = _run(inputs, block_merge=None)
    explicit, _ = _run(inputs, block_merge=0)
    assert bool(jnp.array_equal(declared, explicit))
    assert bool(jnp.all(jnp.isfinite(explicit)))


def test_t_m1_block_of_one_reproduces_the_unmerged_path(cfg):
    """A merged block of one record is the record: the new path is its degenerate case.

    ``B = 1`` pools each record with itself, selects the same records (the
    eligibility rule then reads ``record <= query``, the causal rule of D2) and
    masks the same window slots — so the output must equal the unmerged one
    exactly, not approximately.  This is the strongest available statement that
    the delta did not perturb the arithmetic of the existing path.
    """
    inputs = _primitive_inputs(cfg, T=300, top_k=8, window=64)
    unmerged, _ = _run(inputs, block_merge=0)
    degenerate, _ = _run(inputs, block_merge=1)
    assert bool(jnp.array_equal(unmerged, degenerate))


def test_t_m1_top_k_over_the_prefix_still_matches_the_dense_oracle(cfg):
    """Criterion 13 is untouched by the flag: ``top_k >= T`` ⇒ dense attention.

    The merged path is a *selection* mechanism; when the selection covers the
    whole prefix the fast path (dense causal softmax) already applies and the
    merge is bypassed, so the oracle equivalence holds with the flag on too.
    """
    inputs = _primitive_inputs(cfg, T=64, top_k=10**6, window=0)
    inputs["k_swa"] = inputs["v_swa"] = None
    merged, _ = _run(inputs, block_merge=16)

    # Independent reference: the plain (T, T) causal softmax, written out here.
    q = inputs["q"].astype(jnp.float32)
    k = inputs["k_main"].astype(jnp.float32)
    v = inputs["v_main"].astype(jnp.float32)
    scale = 1.0 / jnp.sqrt(jnp.asarray(q.shape[-1], jnp.float32))
    scores = jnp.einsum("bthd,bshd->bhts", q, k) * scale
    causal = jnp.arange(64)[None, :] <= jnp.arange(64)[:, None]
    attn = jax.nn.softmax(jnp.where(causal, scores, jnp.finfo(jnp.float32).min), axis=-1)
    reference = jnp.einsum("bhts,bshd->bthd", attn, v)
    assert float(jnp.max(jnp.abs(merged - reference))) <= TOL


# ---------------------------------------------------------------------------
# T-m2 — the flag on: finite, shaped, ≤ top_k blocks, exact semantics
# ---------------------------------------------------------------------------


def test_t_m2_merged_output_is_finite_and_keeps_the_shape(cfg):
    """``B = 16``: finite output, stable shape/dtype, no pool published."""
    inputs = _primitive_inputs(cfg, T=512, B=2, top_k=8, window=64)
    out, pool = _run(inputs, block_merge=16)
    assert out.shape == inputs["q"].shape
    assert out.dtype == inputs["q"].dtype
    assert bool(jnp.all(jnp.isfinite(out)))
    assert pool is None, "the merged path selects blocks, it publishes no ADR-012 pool"


def test_t_m2_selection_returns_at_most_top_k_blocks(cfg):
    """The selection width is ``min(top_k, T // B)`` and every id is a real block."""
    T, B_, Hi, Di = 512, 2, cfg.mla_index_heads, cfg.mla_index_dim
    block, top_k, nq = 16, 8, 128
    n_blk = T // block
    inputs = _primitive_inputs(cfg, T=T, B=B_, top_k=top_k, window=64)
    qi = inputs["idx_q"][:, :nq].astype(jnp.float32)
    # Merged indexer keys: mean over each complete block (pooled here, not by
    # the implementation, so the width claim is checked against a literal).
    ki = inputs["idx_k"].astype(jnp.float32)[:, : n_blk * block]
    ki = ki.reshape(B_, n_blk, block, Hi, Di).mean(axis=2)
    sel = attn_sparse.merged_block_selection(qi, ki, jnp.arange(nq), block, top_k)
    assert sel.shape == (B_, nq, 8)  # min(top_k, n_blocks) = min(8, 32) = 8
    assert sel.shape[-1] <= inputs["top_k"]
    assert int(sel.min()) >= 0 and int(sel.max()) < 512 // 16


def test_t_m2_selection_is_the_top_k_of_the_eligible_blocks(cfg):
    """Brute force: the selected blocks are the highest-scoring *eligible* ones.

    Eligibility is causal at block granularity — a block whose records are all
    before the query (``(j + 1) * B <= p + 1``); a partially causal block would
    pool future records into the merged representation.  The reference is
    computed independently of the implementation (own scoring, own masking).
    """
    T, B_, block, top_k, nq = 512, 2, 16, 8, 64
    n_blk = T // block
    keys = jr.split(jr.PRNGKey(3), 2)
    qi = jr.normal(keys[0], (B_, nq, 1, 8))
    kim = jr.normal(keys[1], (B_, n_blk, 1, 8))
    pos = jnp.asarray([(i * 7) % T for i in range(nq)], dtype=jnp.int32)

    sel = attn_sparse.merged_block_selection(qi, kim, pos, block, top_k)
    assert sel.shape == (B_, nq, min(top_k, n_blk))
    assert sel.dtype == jnp.int32

    scb = jnp.einsum("bqhi,bjhi->bqj", qi, kim) / 1.0
    j = jnp.arange(n_blk)[None, :]
    elig = (j + 1) * block <= (pos + 1)[:, None]
    masked = jnp.where(elig[None, :, :], scb, jnp.finfo(jnp.float32).min)
    got = jnp.take_along_axis(masked, sel, axis=-1)
    want = -jnp.sort(-masked, axis=-1)[..., : sel.shape[-1]]
    assert bool(jnp.array_equal(got, want)), "selected blocks are not the top-k"
    # Where the eligible prefix is wider than top_k, no ineligible block is taken.
    wide = (pos + 1) >= top_k * block
    assert bool(jnp.all(((sel + 1) * block <= (pos + 1)[None, :, None]) | ~wide[None, :, None]))


def _hand_inputs(T: int = 8, block: int = 4, window: int = 4, top_k: int = 1):
    """Hand-built tensors where block 0 is the high-affinity one (exact reference).

    Queries are ``e_0``; block 0's records carry key ``[1, 0]`` and value
    ``[3, 0]`` (mean pooling is exact there), block 1's carry key ``[0, 1]`` and
    value ``[0, 5]`` — the same values in the window branch, so the reference
    can be written out by hand.
    """
    q = jnp.zeros((1, T, 1, 2)).at[..., 0].set(1.0)
    k = jnp.zeros((1, T, 1, 2))
    v = jnp.zeros((1, T, 1, 2))
    k = k.at[:, :block, :, 0].set(1.0).at[:, block:, :, 1].set(1.0)
    v = v.at[:, :block, :, 0].set(3.0).at[:, block:, :, 1].set(5.0)
    iq = jnp.zeros((1, T, 1, 2)).at[..., 0].set(1.0)  # only the query at p = T-1 matters
    ik = jnp.zeros((1, T, 1, 2)).at[:, :block, :, 0].set(1.0).at[:, block:, :, 1].set(1.0)
    return dict(
        q=q, k_main=k, v_main=v, k_swa=k, v_swa=v, idx_q=iq, idx_k=ik,
        top_k=top_k, window=window,
    )


def _hand_reference(values, logits):
    """Softmax over ``logits`` applied to ``values`` (fp32, as the primitive)."""
    w = jax.nn.softmax(jnp.asarray(logits, jnp.float32), axis=-1)
    return (w[:, None] * jnp.asarray(values, jnp.float32)).sum(axis=0)


def test_t_m2_attention_reads_the_merged_block_record():
    """A selected block enters attention as its merged (mean) representation.

    ``top_k = 1`` selects block 0 only; the window (positions 4..7) does not
    overlap it, so the output at ``p = 7`` is the softmax over the merged record
    of block 0 and the four window records — written out by hand here.
    """
    out, _ = _run(_hand_inputs(top_k=1), block_merge=4)
    scale = 1.0 / jnp.sqrt(jnp.asarray(2, jnp.float32))
    logits = [scale * 1.0] + [scale * 0.0] * 4  # merged block 0 + 4 window slots
    values = [[3.0, 0.0]] + [[0.0, 5.0]] * 4
    assert float(jnp.max(jnp.abs(out[0, 7, 0] - _hand_reference(values, logits)))) <= TOL


def test_t_m2_window_copy_of_a_selected_block_is_masked():
    """Duplicate rule at block granularity: the window copy of a merged block dies.

    ``top_k = 2`` also selects block 1, whose span is exactly the window
    (4..7): those raw window records are the duplicate copy, so they are masked
    out of the softmax and the block's *merged* record is what survives — the
    same rule D2 applies to individual records (ADR-009), lifted to blocks.  A
    double count would be visible immediately in the reference below.
    """
    out, _ = _run(_hand_inputs(top_k=2), block_merge=4)
    scale = 1.0 / jnp.sqrt(jnp.asarray(2, jnp.float32))
    logits = [scale * 1.0, scale * 0.0]  # only the two merged blocks survive
    values = [[3.0, 0.0], [0.0, 5.0]]
    assert float(jnp.max(jnp.abs(out[0, 7, 0] - _hand_reference(values, logits)))) <= TOL


def test_t_m2_padding_block_does_not_mask_the_window_copy():
    """An *ineligible* selected block must not take the window down with it.

    ``top_k`` ranks blocks, and an early query has fewer eligible blocks than
    ``top_k``: blocks 0 and 1 are eligible at ``p = 9`` (``(j + 1) * 4 <= 10``),
    so the third pick pads itself with block 2 — the query's *own*, only
    partially causal block ([8, 12) at ``p = 9``).  The sparse branch drops that
    block (its mean would contain future records), so its window slots 8 and 9
    must stay alive: they are the causal prefix of the query's own block, and
    covering it is the window's job (ADR-018) — the whole reason
    ``validate_config`` demands ``swa_window >= block``.  The expected value is
    written out by hand: the merged records of blocks 0 and 1 (logits 1 and 0)
    plus the two raw window records 8, 9 (logit 0) — block 1's span (slots 6, 7)
    being masked, since that block *is* eligible and its merged record is the
    surviving copy.
    """
    inputs = _hand_inputs(T=12, block=4, window=4, top_k=3)
    out, _ = _run(inputs, block_merge=4)
    scale = 1.0 / jnp.sqrt(jnp.asarray(2, jnp.float32))
    logits = [scale * 1.0] + [scale * 0.0] * 3  # merged blocks 0, 1 + raw 8, 9
    values = [[3.0, 0.0], [0.0, 5.0]] + [[0.0, 5.0]] * 2
    reference = _hand_reference(values, logits)
    assert float(jnp.max(jnp.abs(out[0, 9, 0] - reference))) <= TOL, (
        "the padded block masked the window copy of the query's own causal prefix"
    )
    # Independent control: the padding block contributes nothing, so dropping it
    # from the ranking (top_k = 2 — exactly the two eligible blocks) cannot
    # change the output at p = 9.
    narrower, _ = _run(_hand_inputs(T=12, block=4, window=4, top_k=2), block_merge=4)
    assert float(jnp.max(jnp.abs(narrower[0, 9, 0] - out[0, 9, 0]))) <= TOL


def test_t_m2_block_duplicate_mask_matches_the_pairwise_compare():
    """The scatter-based block mask is the pairwise slot ⊂ block test.

    ``_block_duplicates`` marks a window slot by scattering each selected
    block's ``B`` offsets into the window (``O(Q * k * B)``); the three cases
    that must come out right are a block inside the window, one entirely before
    it and one past its end.
    """
    block, W, Q = 4, 6, 5
    start = jnp.asarray([0, 3, 10, 20, 7])  # first window position per query
    # (B=1, Q, k=2) block ids; per query the two picks cover a block inside the
    # window, one before it (negative offset) and one starting past its end.
    idx = jnp.asarray([[[0, 6], [1, 2], [1, 5], [3, 4], [2, 7]]], dtype=jnp.int32)

    # Pairwise reference: is window position r inside some selected block's span?
    win = start[:, None] + jnp.arange(W)[None, :]                     # (Q, W)
    lo = idx[..., None] * block                                       # (B, Q, k, 1)
    hi = (idx[..., None] + 1) * block
    covered = (win[None, :, None, :] >= lo) & (win[None, :, None, :] < hi)
    pairwise = covered.any(-2)                                        # (B, Q, W)

    scatter = attn_sparse._block_duplicates(idx, start, block, W)
    assert scatter.shape == pairwise.shape == (1, Q, W)
    assert bool(jnp.all(scatter == pairwise))
    assert bool(jnp.any(scatter))  # the case is not vacuous

    # The eligibility gate (``valid``): only the second pick counts here, so
    # exactly the slots it covers may be marked — the picks that cover the
    # window through *ineligible* blocks (queries 0 and 4) go unmarked.
    second_only = jnp.zeros_like(idx, dtype=jnp.bool_).at[..., 1].set(True)
    gated = attn_sparse._block_duplicates(idx, start, block, W, valid=second_only)
    assert bool(jnp.all(gated == covered[..., 1, :]))
    assert bool(jnp.any(pairwise & ~gated)), "the gate case is not vacuous"


def test_t_m2_declared_flag_switches_the_live_layer(cfg, tmp_path, monkeypatch):
    """The declaration — not a code edit — turns the mechanism on in ``mla.apply``.

    ``net/mla.py`` is untouched by the delta, so the only way the mechanism can
    reach the live layer is the declared config; pointing the reader at a
    fixture that enables it must change the layer output, while the case's own
    declaration (off) leaves it alone.
    """
    sparse_cfg = dataclasses.replace(
        cfg, attn_dense_reference=False, swa_window=64, mla_top_k=8
    )
    params = mla.init_mla(jr.PRNGKey(0), sparse_cfg)
    x = jr.normal(jr.PRNGKey(1), (1, 256, cfg.hidden))

    off = mla.apply(params, sparse_cfg, x)
    _declared_swap(tmp_path, monkeypatch, {"mla_block_merge": {"enabled": True, "block": 16}})
    on = mla.apply(params, sparse_cfg, x)

    assert bool(jnp.all(jnp.isfinite(on)))
    assert on.shape == off.shape
    assert not bool(jnp.allclose(off, on)), "the declared flag did not reach the layer"


def test_t_m2_merged_stack_publishes_no_pool_and_consumers_fall_back(
    cfg, tmp_path, monkeypatch
):
    """A merged layer and the ADR-012 pool are not combined in this delta.

    The pool is a *record*-level mechanism, so a merged layer neither publishes
    one nor consumes a stale one: a following ``reindex`` layer, given no pool,
    falls back to its own exact full-prefix selection (the existing rule for a
    consumer without a builder).  Asserted on a two-layer stack driven through
    ``net/mla.py``, with the declaration switched on.
    """
    stack_cfg = small_config(num_layers=8, num_kda_layers=6, num_mla_layers=2)
    pooled = dataclasses.replace(
        stack_cfg, attn_dense_reference=False, swa_window=64, mla_top_k=8,
        mla_pool_block=8, mla_pool_size=2, mla_layer_modes=("full", "reindex"),
    )
    _declared_swap(tmp_path, monkeypatch, {"mla_block_merge": {"enabled": True, "block": 16}})
    params = [mla.init_mla(key, pooled) for key in jr.split(jr.PRNGKey(5), 2)]
    x = jr.normal(jr.PRNGKey(6), (1, 128, pooled.hidden))

    out, pool = mla.apply_with_pool(params[0], pooled, x, mode="full")
    assert pool is None, "a merged layer must not publish a record-level pool"
    assert bool(jnp.all(jnp.isfinite(out)))

    out2, pool2 = mla.apply_with_pool(params[1], pooled, x, pool=pool, mode="reindex")
    assert pool2 is None
    assert bool(jnp.all(jnp.isfinite(out2)))


def test_t_m2_config_requires_the_window_to_cover_the_block():
    """Declared invariant: the window must cover the intra-block tail.

    Eligible blocks are the whole ones before the query, so the query's own
    partially filled block (up to ``B - 1`` records) is the window branch's job;
    a window narrower than the block would leave that tail unattended.
    """
    ok = small_config(swa_window=16, mla_block_merge=BlockMergeConfig(enabled=True, block=16))
    validate_config(ok)  # no window obligation while the mechanism is off
    validate_config(small_config(mla_block_merge=BlockMergeConfig(enabled=False, block=64)))

    too_small = small_config(swa_window=8, mla_block_merge=BlockMergeConfig(enabled=True, block=16))
    with pytest.raises(AssertionError):
        validate_config(too_small)
    zero = small_config(mla_block_merge=BlockMergeConfig(enabled=True, block=0))
    with pytest.raises(AssertionError):
        validate_config(zero)


# ---------------------------------------------------------------------------
# T-m3 — a config that never declared the mechanism keeps the unmerged path
# ---------------------------------------------------------------------------


def test_t_m3_config_without_the_field_keeps_the_unmerged_path(tmp_path, cfg):
    """Backward compatibility: no field ⇒ off, in the schema and in the reader."""
    declared = json.loads(CONFIG.read_text(encoding="utf-8"))
    declared.pop("mla_block_merge")
    path = tmp_path / "config-without-the-field.json"
    path.write_text(json.dumps(declared, ensure_ascii=False), encoding="utf-8")

    assert net_config.declared_block_merge(path) == 0
    legacy = load_config(path)
    assert legacy.mla_block_merge == BlockMergeConfig(enabled=False, block=16)
    validate_config(legacy)

    inputs = _primitive_inputs(cfg, T=192)
    out, _ = _run(inputs, block_merge=None)
    unmerged, _ = _run(inputs, block_merge=0)
    assert bool(jnp.array_equal(out, unmerged))


def test_t_m3_reader_defaults_to_off_and_never_guesses_a_width(tmp_path):
    """Absent field, disabled flag, non-positive block, missing file ⇒ ``0`` (off)."""
    def declared(payload: dict) -> Path:
        path = tmp_path / f"declared-{len(payload)}-{abs(hash(json.dumps(payload, sort_keys=True)))}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    assert net_config.declared_block_merge(declared({})) == 0
    assert net_config.declared_block_merge(
        declared({"mla_block_merge": {"enabled": False, "block": 16}})
    ) == 0
    assert net_config.declared_block_merge(
        declared({"mla_block_merge": {"enabled": True, "block": 16}})
    ) == 16
    assert net_config.declared_block_merge(
        declared({"mla_block_merge": {"enabled": True, "block": 0}})
    ) == 0
    assert net_config.declared_block_merge(
        declared({"mla_block_merge": {"enabled": True}})
    ) == 0
    assert net_config.declared_block_merge(tmp_path / "missing.json") == 0


# ---------------------------------------------------------------------------
# T-m4 — the declaration itself: pinned off, and sourced in ``deviations``
# ---------------------------------------------------------------------------


def test_t_m4_flag_is_pinned_off_in_the_declared_config(cfg):
    """ADR-018 p. 3: off until the 64K gap measurement is green (criteria 1-7)."""
    declared = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert declared["mla_block_merge"] == {"enabled": False, "block": 16}
    assert ModelConfig().mla_block_merge == BlockMergeConfig(enabled=False, block=16)
    assert net_config.declared_block_merge() == 0
    assert load_config(CONFIG).mla_block_merge.enabled is False


def test_t_m4_deviation_line_names_adr_018_and_the_source():
    """A borrowed mechanism without a sourced ``deviations`` line is a silent import."""
    declared = json.loads(CONFIG.read_text(encoding="utf-8"))
    lines = [line for line in declared["deviations"] if "ADR-018" in line]
    assert len(lines) == 1, "the block-merge deviation must be declared exactly once"
    line = lines[0]
    assert "block-wise token merging" in line.lower()
    assert "mla_block_merge" in line and "block = 16" in line
    assert "Step-5" in line and "20.09.2026" in line
    assert "ADR-009" in line  # the gate the mechanism is accepted through
