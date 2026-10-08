"""Tests for ``tools/mfu_ladder.py`` — the MFU ceiling ladder (L0..L4).

Coverage contract of the ladder (TASK.md):
  * analytic FLOPs models, cross-checked against the *parameter trees* of the
    real ``net/`` blocks (an independent MAC count, not a restatement of the
    formula under test);
  * the MFU denominator is read from ``evidence/kpi-pins.json`` and the tool
    fails closed when it is absent — the declared peak must never be
    substituted (project invariant, tools/sensors/derived_mfu.py);
  * no CUDA device → every level is EMPTY-PENDING with a reason, the tool never
    silently measures on the CPU and reports it as a GB10 number;
  * ``XLA_PYTHON_CLIENT_MEM_FRACTION`` is set *before* ``import jax``
    (ADR-041, OOM incident of 08.10), respects an existing override, and the
    CPU smoke path still produces a non-zero TFLOPS;
  * the L3-MLA rung is gate-aware: it runs under the ``AXIOM_COMPUTE_DTYPE``
    mode its ``--dtype`` selects and takes that mode's denominator, instead of
    pinning fp32 (and the fp32 pin) whatever the flag said.

Run from the repository root::

    python -m pytest tools/tests/test_mfu_ladder.py -q
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# ``mfu_ladder`` imports jax at module scope (the ladder measures on a device);
# where jax is absent the module skips with a reason rather than erroring at
# collection — the same guard the sibling net/tools compute-dtype tests use.
pytest.importorskip(
    "jax",
    reason="mfu_ladder импортирует jax на уровне модуля; на этой машине jax нет",
)

REPO = Path(__file__).resolve().parents[2]

from tools import mfu_ladder as mfu  # noqa: E402  (path set by pytest rootdir=REPO)

PIN_FILE = REPO / "evidence" / "kpi-pins.json"


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def pins() -> dict:
    return mfu.load_denominator(PIN_FILE)


@pytest.fixture(scope="module")
def l3_cfg():
    import net.config as config

    return config.load_config(str(REPO / "net" / "config.json"))


def _macs_from_tree(tree, *, hidden_axis: int = -1) -> int:
    """MACs per token from a module's own parameter tree.

    Counts one multiply-accumulate per weight element of every matrix that the
    token vector touches: for a ``(in, out)`` weight that is ``in * out`` MACs
    per token.  This is deliberately a *different* derivation from the analytic
    formula in ``mfu_ladder`` — it reads the real shapes the ``net/`` blocks
    allocate, so the two agreeing is evidence and not a tautology.
    """
    import jax

    leaves = jax.tree_util.tree_leaves(tree)

    def _shape(x):
        try:
            return tuple(x.shape) if hasattr(x, "shape") else ()
        except Exception:  # pragma: no cover - defensive
            return ()

    total = 0
    for leaf in leaves:
        shape = _shape(leaf)
        if len(shape) == 2:
            total += shape[0] * shape[1]
        elif len(shape) == 3:
            # expert stack (n_experts, in, out) — every expert is materialised
            total += shape[0] * shape[1] * shape[2]
    return total


# --------------------------------------------------------------------------- #
# L0 — GEMM FLOPs and the measurement harness
# --------------------------------------------------------------------------- #
def test_flops_matmul_reference_shapes():
    # 2 * M * K * N, the textbook GEMM count (MAC x 2).
    assert mfu.flops_matmul(4096, 4096, 4096) == 2 * 4096**3
    assert mfu.flops_matmul(8192, 1536, 1536) == 2 * 8192 * 1536 * 1536
    assert mfu.flops_matmul(8192, 1536, 6144) == 2 * 8192 * 1536 * 6144
    # batched: the batch axis multiplies the count
    assert mfu.flops_matmul(1024, 1536, 1536, batch=8) == 8 * 2 * 1024 * 1536 * 1536


def test_bench_reports_median_and_phase_timings():
    import jax.numpy as jnp

    res = mfu.bench_matmul(jnp.float32, 128, 128, 128, iters=4, warmup=2)
    assert set(res) >= {"seconds", "tflops", "warmup_s", "compile_s", "iters"}
    assert res["iters"] == 4
    assert res["seconds"] > 0.0
    assert res["tflops"] > 0.0
    assert res["compile_s"] >= 0.0
    assert res["warmup_s"] >= 0.0
    # median of a positive-latency kernel must equal the measured FLOPs / time
    expected = mfu.flops_matmul(128, 128, 128) / res["seconds"] / 1e12
    assert res["tflops"] == pytest.approx(expected, rel=1e-9)


def test_cpu_smoke_l0_small_shape_nonzero_tflops():
    """Requirement 8: a CPU run of a *small* L0 form yields non-zero TFLOPS."""
    import jax.numpy as jnp

    res = mfu.bench_matmul(jnp.float32, 256, 256, 256, iters=3, warmup=1)
    assert res["tflops"] > 0.0


def test_mfu_pct_uses_the_measured_denominator():
    assert mfu.mfu_pct(49.1, 98.2) == pytest.approx(50.0, rel=1e-6)
    assert mfu.mfu_pct(0.0, 98.2) == 0.0


# --------------------------------------------------------------------------- #
# L1/L2 — analytic FLOPs vs the real parameter trees
# --------------------------------------------------------------------------- #
def test_dense_ffn_flops_match_param_tree(l3_cfg):
    import jax
    import net.mlp as mlp

    params = mlp.init_mlp(jax.random.PRNGKey(0), l3_cfg)
    macs_per_token = _macs_from_tree(params)
    # three matmuls: hidden->inter (g), hidden->inter (u), inter->hidden (down)
    assert macs_per_token == 3 * l3_cfg.hidden * l3_cfg.mlp_intermediate

    tokens = 8192
    inter = l3_cfg.mlp_intermediate  # score the layer as net/ actually builds it
    assert mfu.flops_dense_ffn(tokens, l3_cfg, inter=inter) == 2 * macs_per_token * tokens
    # fwd+bwd = 3x fwd (2 forward + 1 backward) — documented convention
    assert mfu.flops_dense_ffn(tokens, l3_cfg, inter=inter, fwd_bwd=True) == 3 * mfu.flops_dense_ffn(
        tokens, l3_cfg, inter=inter
    )
    # scaling in tokens is linear
    assert mfu.flops_dense_ffn(32768, l3_cfg, inter=inter) == 4 * mfu.flops_dense_ffn(
        8192, l3_cfg, inter=inter
    )


def test_moe_flops_split_active_vs_executed(l3_cfg):
    """LatentMoE: the *executed* routed FLOPs are all experts, not top-k.

    ``net/moe.py:apply`` builds ``(N, n_routed, expert_inter)`` with an einsum
    over every expert and only then gathers top-k (``take_along_axis``).  The
    hardware therefore pays for all ``n_routed`` experts while the algorithm
    needs ``top_k``; the ladder must report both, otherwise L2's ceiling is
    attributed to the wrong cause.
    """
    import jax
    import net.moe as moe

    cfg = l3_cfg
    params = moe.init_moe(jax.random.PRNGKey(0), cfg)

    # --- independent MAC count of the dense (executed) form -----------------
    nr, ns = cfg.moe_num_routed, cfg.moe_num_shared
    lat, ei = cfg.moe_latent_dim, cfg.moe_expert_intermediate
    si, hid = cfg.moe_shared_intermediate, cfg.hidden
    macs_exec_per_token = (
        2 * hid * lat  # W_down, W_up (latent projections)
        + nr * 3 * lat * ei  # every routed expert is materialised
        + ns * 3 * hid * si  # shared experts, full width
        + hid * nr  # router
    )
    # The tree-based count is the independent oracle: it must reproduce the
    # analytic executed MACs exactly, router included (router_w is a real
    # (hidden, n_routed) matrix; only the 1-D norm vector drops out).
    assert _macs_from_tree(params) == (
        2 * hid * lat + nr * 3 * lat * ei + ns * 3 * hid * si + hid * nr
    )

    tokens = 8192
    assert mfu.flops_moe_ffn(tokens, cfg, active=False) == 2 * macs_exec_per_token * tokens
    # active: only top_k routed experts are algorithmically required
    macs_active_per_token = (
        2 * hid * lat + cfg.moe_top_k * 3 * lat * ei + ns * 3 * hid * si + hid * nr
    )
    assert mfu.flops_moe_ffn(tokens, cfg, active=True) == 2 * macs_active_per_token * tokens
    # the dispatch gap is exactly n_routed / top_k on the routed term
    exec_flops = mfu.flops_moe_ffn(tokens, cfg, active=False)
    active_flops = mfu.flops_moe_ffn(tokens, cfg, active=True)
    assert exec_flops > active_flops
    assert mfu.flops_moe_ffn(tokens, cfg, active=False, fwd_bwd=True) == 3 * exec_flops


def test_dense_model_flops_is_six_nd(l3_cfg):
    """L4 reuses the project's own 6ND convention (sensors/derived_mfu.py)."""
    import net.model as model

    active = model.active_param_count(l3_cfg)
    assert active > 0
    assert mfu.flops_dense_model(l3_cfg, 8192, fwd_bwd=True) == 6 * active * 8192


# --------------------------------------------------------------------------- #
# The rollback criterion, mechanised: the analytic model must agree with an
# independent count of the COMPILED program within 10%.  A divergence is a bug
# in the ladder (the instrument), not a property of the hardware.
# --------------------------------------------------------------------------- #
TOLERANCE = 0.10
PROBE_T = 256


def _ratio(analytic: int, fn, *args) -> float:
    compiled = mfu.flops_from_jaxpr(fn, *args)
    assert compiled > 0, "the compiled program has no dot-products to count"
    return compiled / analytic


def _f32_to_bf16(tree):
    import jax
    import jax.numpy as jnp

    return jax.tree_util.tree_map(
        lambda a: a.astype(jnp.bfloat16) if getattr(a, "dtype", None) == jnp.float32 else a,
        tree,
    )


def test_flops_from_jaxpr_is_exact_on_a_matmul():
    import jax.numpy as jnp

    a, b = jnp.ones((4, 8)), jnp.ones((8, 16))
    assert mfu.flops_from_jaxpr(lambda x, y: x @ y, a, b) == mfu.flops_matmul(4, 8, 16)


def test_scan_combines_per_chunk_model():
    # a two-pass tree, not C-1 combines; used by the KDA scan term
    assert mfu.scan_combines_per_chunk(2) == 1
    assert mfu.scan_combines_per_chunk(64) == 120
    assert mfu.scan_combines_per_chunk(64) < 2 * 64 + 1
    assert mfu.scan_combines_per_chunk(256) > mfu.scan_combines_per_chunk(64)


def test_dense_ffn_defaults_to_the_task_expansion(l3_cfg):
    # the task pins L1 at hidden 1536 -> inter 6144 (4x), which is NOT the
    # model's own dense layer-0 width (cfg.mlp_intermediate = 4096)
    assert mfu.dense_ffn_macs_per_token(l3_cfg) == 3 * l3_cfg.hidden * 4 * l3_cfg.hidden
    assert mfu.dense_ffn_macs_per_token(l3_cfg, l3_cfg.mlp_intermediate) == (
        3 * l3_cfg.hidden * l3_cfg.mlp_intermediate
    )


def test_analytic_matches_compiled_program_dense_ffn(l3_cfg):
    import dataclasses

    import jax
    import jax.numpy as jnp
    import net.mlp as mlp

    cfg = dataclasses.replace(l3_cfg, mlp_intermediate=4 * l3_cfg.hidden)
    p = _f32_to_bf16(mlp.init_mlp(jax.random.PRNGKey(0), cfg))
    x = jax.random.normal(jax.random.PRNGKey(1), (PROBE_T, cfg.hidden)).astype(jnp.bfloat16)

    def loss(pp, xx):
        return jnp.sum(mlp.apply(pp, cfg, xx).astype(jnp.float32) ** 2)

    fwd = lambda pp, xx: mlp.apply(pp, cfg, xx)  # noqa: E731
    ratio_fwd = _ratio(mfu.flops_dense_ffn(PROBE_T, cfg, fwd_bwd=False), fwd, p, x)
    assert abs(ratio_fwd - 1.0) <= TOLERANCE, f"fwd divergence {ratio_fwd:.3f}"

    ratio = _ratio(
        mfu.flops_dense_ffn(PROBE_T, cfg, fwd_bwd=True),
        jax.value_and_grad(loss, argnums=(0, 1)),
        p,
        x,
    )
    assert abs(ratio - 1.0) <= TOLERANCE, f"fwd+bwd divergence {ratio:.3f}"
    # and the 3x training convention itself is confirmed by the program
    compiled_fwd = mfu.flops_from_jaxpr(fwd, p, x)
    compiled_bwd = mfu.flops_from_jaxpr(jax.value_and_grad(loss, argnums=(0, 1)), p, x)
    assert compiled_bwd / compiled_fwd == pytest.approx(3.0, rel=1e-6)


def test_analytic_matches_compiled_program_moe(l3_cfg):
    import jax
    import jax.numpy as jnp
    import net.moe as moe

    cfg = l3_cfg
    p = _f32_to_bf16(moe.init_moe(jax.random.PRNGKey(0), cfg))
    x = jax.random.normal(jax.random.PRNGKey(1), (PROBE_T, cfg.hidden)).astype(jnp.bfloat16)

    ratio = _ratio(
        mfu.flops_moe_ffn(PROBE_T, cfg, fwd_bwd=False),
        lambda pp, xx: moe.apply(pp, cfg, xx),
        p,
        x,
    )
    assert abs(ratio - 1.0) <= TOLERANCE, f"MoE exec divergence {ratio:.3f}"


def test_moe_dispatch_gap_is_the_routed_expert_ratio(l3_cfg):
    """Executed / active FLOPs == n_routed / top_k on the routed term."""
    cfg = l3_cfg
    exec_flops = mfu.flops_moe_ffn(8192, cfg, active=False)
    active_flops = mfu.flops_moe_ffn(8192, cfg, active=True)
    routed_active = cfg.moe_top_k * 3 * cfg.moe_latent_dim * cfg.moe_expert_intermediate
    routed_exec = cfg.moe_num_routed * routed_active // cfg.moe_top_k
    assert exec_flops - active_flops == 2 * 8192 * (routed_exec - routed_active)


def test_analytic_matches_compiled_program_kda(l3_cfg):
    import jax
    import jax.numpy as jnp
    import net.kda as kda

    cfg = l3_cfg
    p = _f32_to_bf16(kda.init_kda(jax.random.PRNGKey(0), cfg))
    x = jax.random.normal(jax.random.PRNGKey(1), (PROBE_T, cfg.hidden)).astype(jnp.bfloat16)

    ratio = _ratio(
        mfu.flops_kda(PROBE_T, cfg, fwd_bwd=False),
        lambda pp, xx: kda.apply_kda(pp, cfg, xx),
        p,
        x,
    )
    assert abs(ratio - 1.0) <= TOLERANCE, f"KDA divergence {ratio:.3f}"


def test_kda_scan_term_dominates_the_layer(l3_cfg):
    """Documents the finding: the chunked delta-rule is scan-bound, not proj-bound.

    The affine-transition monoid over ``(dk,dk)``/``(dk,dv)`` matrices is the
    bulk of the layer's FLOPs; this is what the WY/UT form (ADR-031) removes.
    """
    terms = mfu.kda_macs_per_token(l3_cfg, 8192)
    total = sum(terms.values())
    assert terms["scan"] / total > 0.80
    assert terms["scan"] > 5 * terms["proj"]


def test_analytic_matches_compiled_program_mla(l3_cfg):
    import jax
    import jax.numpy as jnp
    import net.mla as mla

    cfg = l3_cfg
    p = mla.init_mla(jax.random.PRNGKey(0), cfg)  # fp32 by construction
    x = jax.random.normal(jax.random.PRNGKey(1), (1, PROBE_T, cfg.hidden))

    ratio = _ratio(
        mfu.flops_mla(PROBE_T, cfg, fwd_bwd=False),
        lambda pp, xx: mla.apply(pp, cfg, xx),
        p,
        x,
    )
    assert abs(ratio - 1.0) <= TOLERANCE, f"MLA divergence {ratio:.3f}"


def test_gate_mode_maps_the_ladder_dtype_or_fails_closed():
    """``--dtype`` -> the ``AXIOM_COMPUTE_DTYPE`` mode, or an explicit refusal."""
    assert mfu._gate_mode("bf16") == "bf16"
    assert mfu._gate_mode("bfloat16") == "bf16"
    assert mfu._gate_mode("fp32") == "fp32"
    assert mfu._gate_mode("float32") == "fp32"
    # a dtype the gate cannot express must not be silently measured as fp32
    with pytest.raises(ValueError):
        mfu._gate_mode("fp16")


def test_l3_mla_cell_is_gate_aware(l3_cfg):
    """The MLA rung declares — and runs under — the mode ``--dtype`` picked.

    Regression guard for the MFU-fix: the cell used to pin
    ``compute_dtype="fp32"`` and never touch ``AXIOM_COMPUTE_DTYPE``, so it
    produced an fp32 point of reference whatever ``--dtype`` said (and whatever
    state ``net/mla.py`` was in).
    """
    import net.compute_dtype as compute_dtype

    for requested, mode in (("bf16", "bf16"), ("fp32", "fp32")):
        cases = mfu.build_level3(requested, scale=8, cfg=l3_cfg)
        mla = [c for c in cases if c.shape.startswith("MLA ")]
        assert len(mla) == 1, "the L3 rung carries exactly one MLA cell"
        cell = mla[0]
        assert cell.compute_dtype == mode
        assert cell.dtype == mode
        # the variable name comes from the gate module's own constant, not a copy
        assert cell.env == {compute_dtype.MODE_ENV: mode}
        assert mode in cell.notes


def test_run_levels_holds_the_mla_gate_env_and_takes_that_mode_s_pin(monkeypatch, tmp_path):
    """The MLA cell benches under its gate; the pin follows the mode it measured.

    ``bench`` is stubbed out: this pins the *wiring* (env around the cell's own
    window, restored afterwards; denominator chosen from ``case.compute_dtype``),
    not the kernel — the GB10 run is the architect's.
    """
    import net.compute_dtype as compute_dtype

    monkeypatch.delenv(compute_dtype.MODE_ENV, raising=False)
    seen: list[str | None] = []

    def spy_bench(fn, **kwargs):
        seen.append(os.environ.get(compute_dtype.MODE_ENV))
        return {
            "seconds": 1.0,
            "samples": [1.0],
            "warmup_s": 0.0,
            "compile_s": 0.0,
            "iters": kwargs.get("iters", 1),
        }

    monkeypatch.setattr(mfu, "has_cuda", lambda: True)
    monkeypatch.setattr(mfu, "bench", spy_bench)
    monkeypatch.setattr(mfu, "T_SEQ", 64)  # keep the cell build cheap

    def mla_row(requested: str, out: Path) -> dict:
        report = mfu.run_levels(
            levels=["L3"],
            dtype=requested,
            pins_path=PIN_FILE,
            out=out,
            iters=1,
            warmup=1,
        )
        row = next(r for r in report["results"] if r["shape"].startswith("MLA "))
        assert row["status"] == "OK"
        return row

    row = mla_row("bf16", tmp_path / "bf16.json")
    assert row["compute_dtype"] == "bf16"
    assert row["denominator_name"] == "bf16"
    assert row["denominator_tflops"] == pytest.approx(98.2)  # bf16 pin, by mode
    # KDA declares no gate; MLA runs with it on; nothing leaks after the rung
    assert seen == [None, "bf16"]
    assert os.environ.get(compute_dtype.MODE_ENV) is None

    seen.clear()
    row = mla_row("fp32", tmp_path / "fp32.json")
    assert row["compute_dtype"] == "fp32"
    assert row["denominator_tflops"] == pytest.approx(45.2)  # fp32 pin, by mode
    assert seen == [None, "fp32"]


def test_env_override_restores_a_preexisting_value(monkeypatch):
    monkeypatch.setenv("AXIOM_TEST_MODE", "fp32")
    with mfu._env_override({"AXIOM_TEST_MODE": "bf16"}):
        assert os.environ["AXIOM_TEST_MODE"] == "bf16"
    assert os.environ["AXIOM_TEST_MODE"] == "fp32"


def test_kda_reference_shape_terms():
    """Hand-computed reference terms for the pinned l3-full shapes."""
    import dataclasses

    import net.config as config

    cfg = config.load_config(str(REPO / "net" / "config.json"))
    hid, H, dk, dv, r, K, C = (
        cfg.hidden,
        cfg.num_heads,
        cfg.kda_dk,
        cfg.kda_dv,
        cfg.kda_decay_rank,
        cfg.kda_short_conv_kernel,
        cfg.kda_wyut_chunk,
    )
    terms = mfu.kda_macs_per_token(cfg, 8192)
    assert terms["proj"] == (
        hid * H * dk * 3 + hid * hid + hid * H + hid * r + r * H * dk + H * dv * hid
    )
    assert terms["conv"] == 3 * K * H * dk
    assert terms["scan"] == H * (dk**3 + dk * dk * dv) * mfu.scan_combines_per_chunk(C) // C
    assert terms["intra"] == H * dk * dv


# --------------------------------------------------------------------------- #
# Denominator: fail-closed pinning
# --------------------------------------------------------------------------- #
def test_load_denominator_from_pins(pins):
    assert pins["bf16"] == pytest.approx(98.2)
    assert pins["fp32"] == pytest.approx(45.2)
    assert pins["source"]


def test_load_denominator_fail_closed_when_missing(tmp_path):
    with pytest.raises(mfu.DenominatorUnavailable):
        mfu.load_denominator(tmp_path / "nope.json")


def test_load_denominator_rejects_declared_peak(tmp_path):
    """No measured peak in the file → fail closed, never fall back to the theory."""
    declared = {
        "mfu_reference": {
            "peak_sources": ["datasheet"],
            "peak_used": {"bf16_dense_tflops_theoretical": 125},
        }
    }
    path = tmp_path / "pins.json"
    path.write_text(json.dumps(declared))
    with pytest.raises(mfu.DenominatorUnavailable):
        mfu.load_denominator(path)


def test_load_denominator_rejects_nonpositive(tmp_path):
    bad = {"mfu_reference": {"peak_used": {"bf16_dense_tflops_measured": 0.0}}}
    path = tmp_path / "pins.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(mfu.DenominatorUnavailable):
        mfu.load_denominator(path)


# --------------------------------------------------------------------------- #
# No-CUDA behaviour: EMPTY-PENDING, never a CPU number dressed as a GB10 one
# --------------------------------------------------------------------------- #
def test_no_cuda_marks_every_level_empty_pending(monkeypatch, tmp_path):
    monkeypatch.setattr(mfu, "has_cuda", lambda: False)
    report = mfu.run_levels(
        levels=["L0", "L1", "L4"],
        dtype="bf16",
        pins_path=PIN_FILE,
        out=tmp_path / "r.json",
        iters=2,
        warmup=1,
        allow_cpu=False,
    )
    assert report["status"] == "empty-pending"
    assert report["results"], "levels must still be listed, with a reason"
    for row in report["results"]:
        assert row["status"] == "EMPTY-PENDING"
        assert "CUDA" in row["notes"]
        assert row["seconds"] is None
        assert row["tflops"] is None
        assert row["mfu_pct"] is None
        # FLOPs and shape are analytic, so they survive the absence of a device
        assert row["flops"] > 0
        assert row["shape"]
    # L4 says why it cannot be faked on the CPU
    l4 = [r for r in report["results"] if r["level"] == "L4"]
    assert l4 and all("GB10" in r["notes"] for r in l4)


def test_l4_is_not_run_on_cpu_even_with_allow_cpu(monkeypatch, tmp_path):
    """L4 needs the GB10; --allow-cpu must not silently produce a CPU step."""
    monkeypatch.setattr(mfu, "has_cuda", lambda: False)
    report = mfu.run_levels(
        levels=["L4"],
        dtype="bf16",
        pins_path=PIN_FILE,
        out=tmp_path / "r.json",
        iters=1,
        warmup=1,
        allow_cpu=True,
        cpu_scale=8,
    )
    assert report["status"] == "empty-pending"
    assert all(r["status"] == "EMPTY-PENDING" for r in report["results"])
    # six-ND convention, computed from the config rather than from a run
    assert report["results"][0]["flops"] > 0


def test_allow_cpu_runs_scaled_and_says_so(monkeypatch, tmp_path):
    """--allow-cpu is an explicit opt-in; the row must record the scaling.

    And the rung must *measure the shape it declares*: a rung that reports the
    full-size shape while running a scaled one inflates its own MFU by the
    scaling ratio.
    """
    monkeypatch.setattr(mfu, "has_cuda", lambda: False)
    report = mfu.run_levels(
        levels=["L0"],
        dtype="float32",
        pins_path=PIN_FILE,
        out=tmp_path / "r.json",
        iters=2,
        warmup=1,
        allow_cpu=True,
        cpu_scale=16,
    )
    assert report["status"] == "ok"
    assert report["cpu_scale"] == 16
    rows = [r for r in report["results"] if r["status"] == "OK"]
    assert rows
    for row in rows:
        assert row["seconds"] > 0.0
        assert row["tflops"] > 0.0
        assert "CPU" in row["notes"]
        # declared shape == measured shape
        parts = row["shape"].split("x")
        dims = [int(p) for p in parts]
        if len(dims) == 4:
            assert row["flops"] == mfu.flops_matmul(dims[0], dims[1], dims[2], batch=dims[3])
        else:
            assert row["flops"] == mfu.flops_matmul(*dims)
    # the unbatched pinned L0 shapes are 4096^3 / 8192x1536x1536 / 8192x1536x6144
    assert "256x256x256" in {r["shape"] for r in rows}


# --------------------------------------------------------------------------- #
# Report shape and artefact
# --------------------------------------------------------------------------- #
def test_report_row_fields(monkeypatch, tmp_path):
    monkeypatch.setattr(mfu, "has_cuda", lambda: False)
    report = mfu.run_levels(
        levels=["L0"],
        dtype="bf16",
        pins_path=PIN_FILE,
        out=tmp_path / "r.json",
        iters=1,
        warmup=1,
    )
    required = {"level", "dtype", "shape", "flops", "seconds", "tflops", "mfu_pct", "notes"}
    for row in report["results"]:
        assert required <= set(row)
    # the denominator is carried with the report so the numbers are auditable
    assert report["denominator_bf16"] == pytest.approx(98.2)
    assert report["denominator_source"]


def test_main_writes_json_report(tmp_path, monkeypatch):
    monkeypatch.setattr(mfu, "has_cuda", lambda: False)
    out = tmp_path / "ladder-report.json"
    rc = mfu.main(["--levels", "L0", "--out", str(out), "--iters", "1", "--warmup", "1"])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["schema"].startswith("mfu-ladder/")
    assert payload["results"]


def test_report_table_renders_every_row(monkeypatch, tmp_path):
    monkeypatch.setattr(mfu, "has_cuda", lambda: False)
    report = mfu.run_levels(
        levels=["L0", "L1"],
        dtype="bf16",
        pins_path=PIN_FILE,
        out=tmp_path / "r.json",
        iters=1,
        warmup=1,
    )
    table = mfu.format_table(report)
    assert "EMPTY-PENDING" in table
    for row in report["results"]:
        assert row["level"] in table


def test_unknown_level_rejected(tmp_path):
    with pytest.raises(ValueError):
        mfu.run_levels(
            levels=["L9"],
            dtype="bf16",
            pins_path=PIN_FILE,
            out=tmp_path / "r.json",
        )


# --------------------------------------------------------------------------- #
# ADR-041: the memory fraction is pinned before jax is imported
# --------------------------------------------------------------------------- #
def test_mem_fraction_is_set_before_jax_import():
    """Structural check: the env write precedes ``import jax`` in the source."""
    src = Path(mfu.__file__).read_text()
    tree = ast.parse(src)
    env_line = None
    jax_line = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = getattr(fn, "attr", getattr(fn, "id", ""))
            if name in {"setdefault", "environ"} or "MEM_FRACTION" in ast.dump(node):
                if node.lineno and (env_line is None or node.lineno < env_line):
                    env_line = node.lineno
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = getattr(node, "module", "") or ""
            names = [a.name for a in node.names]
            if mod.split(".")[0] == "jax" or any(n.split(".")[0] == "jax" for n in names):
                if jax_line is None or node.lineno < jax_line:
                    jax_line = node.lineno
    assert env_line is not None, "module must set the memory fraction itself"
    assert jax_line is not None, "module must import jax"
    assert env_line < jax_line, (
        "XLA_PYTHON_CLIENT_MEM_FRACTION must be written before `import jax` "
        f"(env at line {env_line}, jax at line {jax_line}) — ADR-041"
    )


def test_mem_fraction_default_applied_on_fresh_import(tmp_path):
    """A fresh interpreter without the variable gets the 0.5 default."""
    code = (
        "import sys; sys.path.insert(0, %r);\n"
        "from tools import mfu_ladder as m;\n"
        "import os; print(os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'])\n" % str(REPO)
    )
    env = {k: v for k, v in os.environ.items() if k != mfu.MEM_FRACTION_ENV}
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=str(REPO)
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == mfu.DEFAULT_MEM_FRACTION


def test_mem_fraction_override_is_preserved(tmp_path):
    """An operator-set fraction wins: setdefault, never overwrite."""
    code = (
        "import sys; sys.path.insert(0, %r);\n"
        "from tools import mfu_ladder as m;\n"
        "import os; print(os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'])\n" % str(REPO)
    )
    env = dict(os.environ)
    env[mfu.MEM_FRACTION_ENV] = "0.15"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=str(REPO)
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "0.15"


# --------------------------------------------------------------------------- #
# Boundaries (requirement 7): no network, writes stay inside the worktree
# --------------------------------------------------------------------------- #
def test_ladder_makes_no_network_calls():
    src = Path(mfu.__file__).read_text()
    for forbidden in (
        "import requests",
        "import socket",
        "import urllib",
        "from urllib",
        "http.client",
        "import httpx",
    ):
        assert forbidden not in src


def test_default_outputs_live_inside_the_repo():
    for path in (mfu.DEFAULT_OUT, mfu.DEFAULT_PINS, mfu.DEFAULT_L4_CONFIG):
        assert REPO in Path(path).resolve().parents
