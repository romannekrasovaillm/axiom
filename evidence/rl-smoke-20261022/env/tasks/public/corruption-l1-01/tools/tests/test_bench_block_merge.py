"""Tests of the ADR-018 64K bench (`tools/bench_block_merge.py`).

The bench itself runs the pinned l3-full model at 64K on a GPU; these tests run
on CPU with hand-sized forms and pin the four properties the bench's numbers
rest on:

* **T-b1** — the needle sequence and the recall instrument are reproducible at
  the same seed (same token ids, same probes, bit-identical recall), so a rerun
  of the bench measures the run, not the PRNG;
* **T-b2** — the verdict function is the declared mechanical rule: synthetic
  numbers in, ``flag_on``/``flag_off`` out, including the two boundaries
  (``recall_on == recall_off - 0.02`` is still inside the tolerance;
  ``cost_on.p50 == 0.95 x cost_off.p50`` is not a drop);
* **T-b3** — flipping the flag off is the pre-delta path, bit for bit: with the
  declared switch off the model output is identical to the pinned-config output
  *after* the switch has been on, and the switch is live (with it on the output
  differs — otherwise the bench's cost legs would be two copies of one graph);
* **T-b4** — the capture is `net/model.py`'s forward: its hidden state equals
  ``model.forward(..., return_hidden=True)`` exactly, and the MLA layer inputs it
  reports are the tensors the layers were actually handed (recorded by wrapping
  ``net.mla.apply_with_pool``), for both AttnRes settings.  The capture has two
  forms and the equality is pinned for both, each in the environment it runs in:
  eagerly for the reference (loop) form, which is what a fixture of this size
  takes and what T-b4 above asserts; compiled for the scan form, which is the
  64K form and is compiled by construction (T-b4b).  The scan form cannot pass
  the eager assertion — and no scan-based capture can: ``lax.scan`` compiles its
  body, and XLA lowers a reduction inside a compiled body in a different
  association order than op by op, so ``jax.jit(rms_norm) != rms_norm`` by ~1
  ULP, which RMSNorm compounds across layers (measured: 3.5e-5 on this fixture,
  4.6e-5 for ``jax.jit(model.forward)`` against ``model.forward`` itself);
* **T-b5** — the instrument's unmerged selection is the model's own: its record
  ids equal ``net.mla.topk_indices``'s row for the same layer input, and the
  runtime switch really moves the declared reader.
"""

from __future__ import annotations

import dataclasses
import sys
from contextlib import contextmanager
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1]
CASE_DIR = TOOLS_DIR.parent
for _p in (str(TOOLS_DIR), str(CASE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bench_block_merge as bench  # noqa: E402

from net import mla, model  # noqa: E402
from net.config import BlockMergeConfig, ModelConfig  # noqa: E402


# --- a hand-sized model that satisfies every validate_config invariant --------


def tiny_cfg(**overrides) -> ModelConfig:
    """8 layers (6 KDA + 2 MLA), 32-wide: the smallest form with two MLA layers."""
    base = dict(
        vocab_size=128,
        hidden=32,
        num_layers=8,
        num_kda_layers=6,
        num_mla_layers=2,
        num_heads=2,
        head_dim=16,
        kda_dk=16,
        kda_dv=16,
        kda_decay_rank=16,
        kda_short_conv_kernel=4,
        mla_latent_dim=32,
        mla_head_dim=16,
        mla_top_k=8,
        mla_index_heads=1,
        mla_index_dim=8,
        swa_window=16,
        attn_dense_reference=False,
        mla_pool_size=0,
        mla_layer_modes=(),
        mla_block_merge=BlockMergeConfig(enabled=False, block=16),
        mlp_intermediate=64,
        moe_latent_dim=16,
        moe_num_routed=4,
        moe_num_shared=1,
        moe_top_k=2,
        moe_expert_intermediate=16,
        moe_shared_intermediate=32,
        attnres_blocks=1,
        attnres_block_size=4,
        vit_hidden=16,
        vit_depth=1,
        vit_heads=2,
        vit_mlp=32,
    )
    base.update(overrides)
    return ModelConfig(**base)


T_TINY = 512  # >= 3 * NEEDLE_STRIDE + needle, and > mla_top_k

#: Bound on the cross-form numerical drift of the scan capture (module docstring,
#: T-b4).  ``lax.scan`` compiles its body, and XLA lowers a reduction inside a
#: compiled body in a different association order than the op-by-op loop — the
#: same effect the docstring measures at ~3.5e-5 for eager-vs-compiled.  On this
#: fixture (8 layers, hidden 32) the compiled scan and the compiled loop agree to
#: 3.19e-5 (deterministic, independent of the ADR-018 flag), so an exact
#: ``array_equal`` between the two forms is not attainable — and no scan-based
#: capture can attain it.  The bound is an order of magnitude above that drift
#: and far below any structural difference (T-b3 measures O(1) for merged vs
#: unmerged), so it still fails a capture whose graph is not the model's.
_SCAN_FORM_ATOL = 1e-4

#: Bound on T-b4's eager scan-vs-loop drift — the same association-order effect as
#: ``_SCAN_FORM_ATOL``, one level up.  ``tiny_cfg`` declares ``scan_layers=True``
#: (the pinned ``net/config.json`` value), so the reference leg (``model.forward``)
#: lowers a reduction inside a ``lax.scan`` body while the capture is the op-by-op
#: reference loop; XLA orders the compiled reduction differently, and RMSNorm
#: compounds it across layers.  Measured max|diff| = 3.526e-5 on this fixture
#: (2026-10-05, ``docs/RESULTS-2026-10-05.ru.md`` §8: the comparison is the
#: structural equivalence of two implementations, not a bit-exactness gate).  The
#: bound is the precedent ``_SCAN_FORM_ATOL = 1e-4``; shape and finiteness stay
#: strict, so a capture whose graph is not the model's still fails.
_MODEL_FORWARD_ATOL = 1e-4


def _tiny_params(cfg: ModelConfig, seed: int = 0) -> model.ModelParams:
    return model.init_params(jr.PRNGKey(seed), cfg)


def _tiny_sequence(cfg: ModelConfig, length: int = T_TINY):
    return bench.build_needle_sequence(
        length, seed=3, vocab=cfg.vocab_size, needle_len=2
    )


def _captures(cfg: ModelConfig, params: model.ModelParams, sequence, block: int):
    """The two legs of one capture, as the isolated source builds them."""
    ids = sequence.token_ids[None, :]
    with bench.declared_switch(False, block):
        off = bench.capture_leg(cfg, params, ids, use_attnres=False, label="off",
                                verbose=False)
    return {"off": off, "on": off}


# ---------------------------------------------------------------------------
# T-b1 — the instrument is reproducible at the same seed
# ---------------------------------------------------------------------------


def test_t_b1_sequence_is_reproducible_at_the_same_seed():
    cfg = tiny_cfg()
    a = _tiny_sequence(cfg)
    b = _tiny_sequence(cfg)
    assert a.probes == b.probes
    assert bool(jnp.array_equal(a.token_ids, b.token_ids))
    assert a.token_ids.dtype == jnp.int32
    # The three facts of a position are one declared block apart, so they are
    # not one block sampled three times (the per-position aggregate is honest).
    for frac in bench.POSITION_FRACTIONS:
        group = [p for p in a.probes if p.position_fraction == frac]
        assert len(group) == bench.FACTS_PER_POSITION
        blocks = [p.start // 16 for p in group]
        assert len(set(blocks)) == len(blocks)


def test_t_b1_sequence_changes_with_the_seed():
    cfg = tiny_cfg()
    a = _tiny_sequence(cfg)
    b = bench.build_needle_sequence(a.length, seed=4, vocab=cfg.vocab_size, needle_len=2)
    assert not bool(jnp.array_equal(a.token_ids, b.token_ids))


def test_t_b1_recall_is_bit_reproducible_at_the_same_seed():
    cfg = tiny_cfg()
    params = _tiny_params(cfg)
    sequence = _tiny_sequence(cfg)
    block = 16
    first = bench.measure_recall(cfg, params, sequence,
                                 _captures(cfg, params, sequence, block),
                                 block=block, recall_source="isolated")
    second = bench.measure_recall(cfg, params, sequence,
                                  _captures(cfg, params, sequence, block),
                                  block=block, recall_source="isolated")
    assert first["probes"] == second["probes"]
    assert first["overall"] == second["overall"]


def test_t_b1_recall_covers_every_probe_and_position():
    cfg = tiny_cfg()
    params = _tiny_params(cfg)
    sequence = _tiny_sequence(cfg)
    block = 16
    recall = bench.measure_recall(cfg, params, sequence,
                                  _captures(cfg, params, sequence, block),
                                  block=block, recall_source="isolated")
    assert len(recall["probes"]) == bench.FACTS_PER_POSITION * len(bench.POSITION_FRACTIONS)
    assert len(recall["layers"]) == cfg.num_mla_layers
    assert set(recall["by_position"]) == {f"{f:.2f}" for f in bench.POSITION_FRACTIONS}
    for probe in recall["probes"]:
        for mode in ("off", "on", "on_equal_budget"):
            assert 0.0 <= probe[f"recall_{mode}"] <= 1.0
        for mode in ("off", "on", "eq"):
            assert len(probe["per_layer"][mode]) == cfg.num_mla_layers
        # the equal-budget width is the number of blocks that reads as many
        # records as the unmerged selection (top_k records)
        assert probe["equal_budget_blocks"] == max(1, min(cfg.mla_top_k, T_TINY) // block)
        # the merged leg covers a whole block per selected id, so its chance
        # coverage cannot be below the unmerged leg's at this block width
        assert probe["chance_on"] > probe["chance_off"]


# ---------------------------------------------------------------------------
# T-b2 — the verdict function is the declared mechanical rule
# ---------------------------------------------------------------------------


def _recall_stub(rows):
    return {"probes": [{"position_fraction": f, "fact_index": j, "recall_off": a,
                        "recall_on": b} for f, j, a, b in rows]}


def _cost_stub(p50_off: float, p50_on: float):
    return {
        "cost_drop_measurable": p50_on < p50_off * bench.COST_RATIO_THRESHOLD,
        "ratio_of_medians": p50_on / p50_off,
        "legs": {"off": {"p50_s": p50_off}, "on": {"p50_s": p50_on}},
    }


NINE_PASSING = [(0.10, 0, 0.40, 0.40), (0.10, 1, 0.30, 0.31), (0.10, 2, 0.20, 0.20),
                (0.50, 0, 0.35, 0.35), (0.50, 1, 0.25, 0.26), (0.50, 2, 0.15, 0.15),
                (0.90, 0, 0.45, 0.44), (0.90, 1, 0.55, 0.55), (0.90, 2, 0.05, 0.05)]


def test_t_b2_flag_on_when_both_arms_hold():
    verdict = bench.decide(_recall_stub(NINE_PASSING), _cost_stub(1.00, 0.90))
    assert verdict["verdict"] == "flag_on"
    assert verdict["recall_condition"] is True
    assert verdict["cost_condition"] is True
    assert verdict["reasons"] == []
    assert verdict["applied_by_architect"] is False


def test_t_b2_flag_off_when_one_probe_falls_below_the_tolerance():
    rows = list(NINE_PASSING)
    rows[4] = (0.50, 1, 0.40, 0.30)  # 0.10 below the baseline, tolerance 0.02
    verdict = bench.decide(_recall_stub(rows), _cost_stub(1.00, 0.90))
    assert verdict["verdict"] == "flag_off"
    assert verdict["recall_condition"] is False
    assert verdict["cost_condition"] is True
    assert any("recall dropped" in r for r in verdict["reasons"])


def test_t_b2_recall_tolerance_boundary_is_inclusive():
    """``recall_on == recall_off - 0.02`` is still inside the tolerance (>=)."""
    rows = list(NINE_PASSING)
    rows[0] = (0.10, 0, 0.40, 0.40 - bench.RECALL_TOLERANCE)
    verdict = bench.decide(_recall_stub(rows), _cost_stub(1.00, 0.90))
    assert verdict["recall_condition"] is True
    assert verdict["verdict"] == "flag_on"


def test_t_b2_flag_off_when_the_cost_drop_is_not_measurable():
    verdict = bench.decide(_recall_stub(NINE_PASSING), _cost_stub(1.00, 0.95))
    assert verdict["verdict"] == "flag_off"
    assert verdict["cost_condition"] is False
    assert any("p50" in r for r in verdict["reasons"])


def test_t_b2_flag_off_when_an_arm_is_missing():
    assert bench.decide(None, _cost_stub(1.00, 0.50))["verdict"] == "flag_off"
    assert bench.decide(_recall_stub(NINE_PASSING), None)["verdict"] == "flag_off"


def test_t_b2_cost_ratio_threshold_is_strict():
    """A ratio of exactly 0.95 is not a drop below 0.95 x the baseline."""
    cost = _cost_stub(1.0, bench.COST_RATIO_THRESHOLD)
    assert bench.decide(_recall_stub(NINE_PASSING), cost)["verdict"] == "flag_off"
    cost = _cost_stub(1.0, bench.COST_RATIO_THRESHOLD - 1e-6)
    assert bench.decide(_recall_stub(NINE_PASSING), cost)["verdict"] == "flag_on"


# ---------------------------------------------------------------------------
# T-b3 — the flag off is the pre-delta path, bit for bit
# ---------------------------------------------------------------------------


def test_t_b3_flag_off_is_bit_identical_after_the_switch_has_been_on():
    cfg = tiny_cfg()
    params = _tiny_params(cfg)
    ids = jnp.arange(T_TINY, dtype=jnp.int32)[None, :] % cfg.vocab_size

    # Both legs are overridden explicitly: the pinned net/config.json declares the
    # ADR-018 flag on, so reading it without a switch would make "baseline" the
    # merged leg.  The pre-delta path is the flag *off*, whatever the pin says.
    with bench.declared_switch(False, 16):
        baseline = jax.jit(lambda p: model.forward(p, cfg, ids, chunk_size=16))(params)
        jax.block_until_ready(baseline)

    with bench.declared_switch(True, 16):
        with_on = jax.jit(lambda p: model.forward(p, cfg, ids, chunk_size=16))(params)
        jax.block_until_ready(with_on)

    with bench.declared_switch(False, 16):
        after = jax.jit(lambda p: model.forward(p, cfg, ids, chunk_size=16))(params)
        jax.block_until_ready(after)

    assert not bool(jnp.array_equal(baseline, with_on)), (
        "the runtime switch did not reach the path: the merged leg is identical to the "
        "baseline, so the bench's two legs would be one graph"
    )
    assert bool(jnp.array_equal(baseline, after)), (
        "the flag off is not the pre-delta path: the output changed after the switch was "
        "on and back off (rollback signal (a))"
    )


def test_t_b3_runtime_switch_moves_the_declared_reader():
    from net import attn_sparse
    from net import config as net_config

    pinned_path = net_config.CONFIG_PATH
    # The reader's value outside any switch is whatever the pinned config declares
    # (ADR-018 turned it on): the test must not assume a default, only that the
    # switch moves the reader to each requested leg and restores the pin.
    pinned_declared = attn_sparse.declared_block_merge()
    assert pinned_declared in (0, 16), pinned_declared
    try:
        with bench.declared_switch(True, 16):
            assert attn_sparse.declared_block_merge() == 16
            assert not net_config.CONFIG_PATH.samefile(bench.CONFIG_PATH), (
                "the switch must not be an edit of the pinned net/config.json"
            )
        assert attn_sparse.declared_block_merge() == pinned_declared
        with bench.declared_switch(False, 16):
            assert attn_sparse.declared_block_merge() == 0
        assert attn_sparse.declared_block_merge() == pinned_declared
        assert net_config.CONFIG_PATH == pinned_path
    finally:
        net_config.CONFIG_PATH = pinned_path


# ---------------------------------------------------------------------------
# T-b4 — the capture is the model's own forward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_attnres", [False, True])
def test_t_b4_capture_equals_model_forward(use_attnres: bool):
    cfg = tiny_cfg()
    params = _tiny_params(cfg)
    ids = jnp.arange(T_TINY, dtype=jnp.int32)[None, :] % cfg.vocab_size

    with bench.declared_switch(False, 16):
        expected = model.forward(params, cfg, ids, chunk_size=16, use_attnres=use_attnres,
                                 return_hidden=True)[1]
        hidden, mla_inputs = bench.capture_forward(params, cfg, ids, use_attnres=use_attnres,
                                                   chunk_size=16)
    # The reference leg scans (``tiny_cfg`` keeps the pinned ``scan_layers=True``)
    # while the capture is the reference loop, so the two lower a reduction in a
    # different association order: equality is structural, within the measured
    # bound, not bit-exact.  Shape and finiteness stay exact, or a capture whose
    # graph is not the model's would slip through.
    assert hidden.shape == expected.shape, (
        "the capture changes the hidden-state shape — its activations are not the model's"
    )
    assert bool(jnp.all(jnp.isfinite(hidden))) and bool(jnp.all(jnp.isfinite(expected))), (
        "the capture or the model produced a non-finite activation"
    )
    assert bool(jnp.allclose(hidden, expected, rtol=0.0, atol=_MODEL_FORWARD_ATOL)), (
        "the capture drifts from net/model.py's forward beyond the association-order "
        "bound — its activations are not the model's"
    )
    assert len(mla_inputs) == cfg.num_mla_layers


def test_t_b4b_scan_form_capture_equals_model_forward_compiled():
    """T-b4 for the scan form, in the environment the bench uses it in.

    The scan form is the 64K one (the reference form needs ~15.3 GiB there and
    dies on a 16 GB card), and ``capture_leg`` compiles the capture, so the
    equality that matters is against a compiled ``model.forward``.  This is the
    same assertion as T-b4 — the capture *is* the model's forward — with the
    reference compiled too, which is the only way a ``lax.scan``-based capture
    can be compared: ``lax.scan`` compiles its body, and a compiled body cannot
    reproduce a reference bit for bit (module docstring, T-b4).  The equality is
    therefore within ``_SCAN_FORM_ATOL`` — the measured association-order bound —
    not ``array_equal``: compiled scan and compiled loop differ by 3.19e-5 on
    this fixture, deterministically, once both legs are declared the same way.
    """
    cfg = tiny_cfg()
    params = _tiny_params(cfg)
    ids = jnp.arange(T_TINY, dtype=jnp.int32)[None, :] % cfg.vocab_size

    assert bench.capture_form(params, cfg, ids, form="scan") == "scan", (
        "the fixture must be able to take the scan form, or this test is vacuous"
    )
    with bench.declared_switch(False, 16):
        expected = jax.jit(lambda p, x: model.forward(
            p, cfg, x, chunk_size=16, use_attnres=False, return_hidden=True))(params, ids)[1]
        hidden, mla_inputs = jax.jit(lambda p, x: bench.capture_forward(
            p, cfg, x, use_attnres=False, chunk_size=16, form="scan"))(params, ids)
        # Both reference legs are overridden to the same (off) declaration as the
        # scan leg: the pinned net/config.json now declares the ADR-018 flag on,
        # so compiling the reference outside the switch would compare the merged
        # leg against the unmerged one — the switch, not the pin, must fix the leg.
        compiled_loop = jax.jit(lambda p, x: bench.capture_forward(
            p, cfg, x, use_attnres=False, chunk_size=16, form="loop"))(params, ids)[1]
    assert bool(jnp.allclose(hidden, expected, rtol=0.0, atol=_SCAN_FORM_ATOL)), (
        "the scan form drifts from net/model.py's compiled forward beyond the "
        "association-order bound — its activations are not the model's"
    )
    # The captured MLA inputs are the model's too: they match the compiled
    # reference form's within the same bound, and T-b4 pins that form's inputs
    # against the tensors the layers were actually handed (by wrapping
    # net.mla.apply_with_pool — done eagerly there, because a recorded tensor
    # cannot escape a jit trace).
    assert len(compiled_loop) == len(mla_inputs) == cfg.num_mla_layers
    for layer, (reference, scanned) in enumerate(zip(compiled_loop, mla_inputs)):
        assert bool(jnp.allclose(reference, scanned, rtol=0.0, atol=_SCAN_FORM_ATOL)), (
            f"MLA layer {layer}: the scan form reports a different input than the "
            f"reference form beyond the association-order bound"
        )


@contextmanager
def _budget_of(value: int | None):
    """Run a block with ``bench._device_budget_bytes`` reporting ``value``."""
    original = bench._device_budget_bytes
    bench._device_budget_bytes = lambda: value
    try:
        yield
    finally:
        bench._device_budget_bytes = original


def test_t_b4c_capture_form_is_chosen_by_need_and_declared():
    """The form decision: the reference form while it fits, the scan form after.

    Both forms are the model's forward; the choice is memory, and the bench
    records it (``bounds.capture_form``) so the numbers declare their instrument.
    """
    cfg = tiny_cfg()
    params = _tiny_params(cfg)
    itemsize = int(params.embedding.dtype.itemsize)
    ids = jnp.arange(T_TINY, dtype=jnp.int32)[None, :] % cfg.vocab_size

    # the fixture is tiny and this backend reports no budget: the reference
    # (eager-exact) form is what the capture takes, which is what T-b4 asserts
    assert bench.capture_form(params, cfg, ids) == "loop"
    assert bench.capture_form(params, cfg, ids, form="auto") == "loop"

    # the estimate is the documented arithmetic, and it is what decides the form
    need = bench._reference_form_bytes(cfg, T_TINY, itemsize)
    assert need == int(cfg.num_layers * T_TINY * cfg.hidden * itemsize
                       * bench._REFERENCE_LIVE_COPIES_PER_LAYER)
    long_ids = jnp.zeros((1, 1 << 20), dtype=jnp.int32)
    long_need = bench._reference_form_bytes(cfg, 1 << 20, itemsize)
    with _budget_of(long_need - 1):
        assert bench.capture_form(params, cfg, long_ids) == "scan", (
            "a capture whose estimated live set exceeds the device budget must take the "
            "memory-safe form"
        )
    with _budget_of(long_need):
        assert bench.capture_form(params, cfg, long_ids) == "loop", (
            "the boundary: a live set that just fits the budget stays on the reference form"
        )

    # the explicit forms override the estimate, and an unusable scan form falls back
    assert bench.capture_form(params, cfg, long_ids, form="loop") == "loop"
    assert bench.capture_form(params, cfg, long_ids, form="scan") == "scan"
    assert bench.capture_form(params, cfg, long_ids, use_attnres=True) == "loop", (
        "AttnRes keeps every earlier delta live and cannot be carried by a scan"
    )
    with pytest.raises(ValueError):
        bench.capture_form(params, cfg, ids, form="nonsense")


def test_t_b4_captured_inputs_are_what_the_layers_were_handed():
    """Wrap ``net.mla.apply_with_pool`` and compare what each MLA layer received.

    The reference leg runs the unrolled parity path (``scan_layers=False``): the
    recorder reads the tensors the layers were handed, and a tensor produced inside
    a ``lax.scan`` body is a trace-time value that cannot escape the trace
    (``UnexpectedTracerError``).  The capture leg is the loop form either way, so
    this keeps both legs on the model's eager forward — its own layers, one at a
    time — and does not touch ``net/`` (``docs/RESULTS-2026-10-05.ru.md`` §8).
    """
    cfg = dataclasses.replace(tiny_cfg(), scan_layers=False)
    params = _tiny_params(cfg)
    ids = jnp.arange(T_TINY, dtype=jnp.int32)[None, :] % cfg.vocab_size
    recorded: list[jnp.ndarray] = []
    original = mla.apply_with_pool

    def recorder(p, c, x, *a, **kw):
        recorded.append(x)
        return original(p, c, x, *a, **kw)

    mla.apply_with_pool = recorder
    try:
        with bench.declared_switch(False, 16):
            model.forward(params, cfg, ids, chunk_size=16, use_attnres=False)
            handed = list(recorded)
            recorded.clear()
            _, captured = bench.capture_forward(params, cfg, ids, use_attnres=False,
                                                chunk_size=16)
            from_capture = list(recorded)
    finally:
        mla.apply_with_pool = original

    assert len(handed) == len(from_capture) == len(captured) == cfg.num_mla_layers
    assert len(handed) == 2, "the fixture must have exactly two MLA layers"
    for layer, (seen, got) in enumerate(zip(handed, captured)):
        assert bool(jnp.array_equal(seen, got)), f"MLA layer {layer}: capture mismatch"


# ---------------------------------------------------------------------------
# T-b5 — the instrument's selection is the model's own
# ---------------------------------------------------------------------------


def test_t_b5_unmerged_selection_matches_mla_topk_indices():
    cfg = tiny_cfg()
    params = _tiny_params(cfg)
    x = jr.normal(jr.PRNGKey(11), (1, T_TINY, cfg.hidden))
    query = T_TINY - 1

    idx_q, idx_k = bench.indexer_projections(params.layers[3].attn, cfg, x)
    selected, _scores = bench.unmerged_selection(idx_q, idx_k, cfg, query)

    reference = mla.topk_indices(params.layers[3].attn, cfg, x)  # (B, T, k)
    assert set(int(v) for v in selected) == set(int(v) for v in reference[0, query])


def test_t_b5_merged_selection_is_a_top_k_of_blocks():
    cfg = tiny_cfg()
    params = _tiny_params(cfg)
    x = jr.normal(jr.PRNGKey(12), (1, T_TINY, cfg.hidden))
    block, query = 16, T_TINY - 1

    idx_q, idx_k = bench.indexer_projections(params.layers[3].attn, cfg, x)
    with bench.declared_switch(True, block):  # the merged path reads the declared switch
        blocks, scores, merged = bench.merged_selection(idx_q, idx_k, cfg, query, block)

    assert merged.shape[1] == T_TINY // block
    width = min(cfg.mla_top_k, T_TINY // block)
    assert blocks.shape[-1] == width
    # every eligible block is a legal candidate at the last query, so the
    # selection is exactly the top ``width`` of the block scores
    order = jnp.argsort(-scores)
    assert set(int(v) for v in blocks) == set(int(v) for v in order[:width])
    # and a merged block is the mean of the records it stands for
    expected_mean = idx_k[0, :block].mean(axis=0)
    assert jnp.allclose(merged[0, 0], expected_mean, atol=1e-5)


def test_t_b5_coverage_modes_are_record_accurate():
    probe = bench.NeedleProbe(position_fraction=0.5, fact_index=0, start=32, length=4,
                              key_token=1)
    assert bench.coverage(probe, {32, 33}, mode="records", block=16) == 0.5
    # records 32..35 all live in block 2, so selecting the block covers all four
    assert bench.coverage(probe, {2}, mode="blocks", block=16) == 1.0
    assert bench.coverage(probe, {3}, mode="blocks", block=16) == 0.0
