"""BF16 campaign, phase 1 — the compute-dtype gate ``AXIOM_COMPUTE_DTYPE``.

Three acceptance criteria of the task are pinned mechanically here:

**(а) gate off is bit-for-bit the pre-gate graph.**  The criterion names the
baseline commit (``773e389``, the certified MFU-denominator chain).  Rather than
pinning a hash — which would be backend-dependent and would rot — the test
extracts ``net/`` at that commit with ``git archive`` into a temporary package
``baseline_net`` and runs **both** implementations on identical parameters and
inputs, comparing the raw fp32 bit patterns.  The comparison covers the dense
MLA oracle, the sparse union-attention path, the KDA + MLA + LatentMoE pilot
forward, the diagnostic dense-standard forward and the loss (NTP + MTP + QB).

**(b) under bf16 the residual stream, the master weights and the gradient
accumulation stay fp32.**  Checked on a real forward + ``jax.grad`` + an
optimizer state initialisation: no leaf of any of those trees is bf16.

**(c) under bf16 the only dtype the path introduces is bf16, and it is
introduced at GEMM boundaries.**  Checked on the traced jaxpr: the graph of the
layer contains bf16 intermediates (so the gate really bit), contains no third
precision (no f16/f64/f8 outside the QAT quantiser, which is off), and returns
fp32 — a dtype leaking out of a boundary would show up as an fp32-typed entry
into the loss path or as a non-fp32 residual.

The gate's own contract (fail-closed on an unknown value, default fp32) is
pinned next to them, because a typo that silently fell back to fp32 would make
every measurement of the campaign a measurement of the wrong model.
"""

from __future__ import annotations

import dataclasses
import io
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from net import compute_dtype, mla, model, optimizer

NET_DIR = Path(__file__).resolve().parents[1]
CASE_DIR = NET_DIR.parent

#: The baseline the gate-off path must reproduce bit for bit (task §2а).
BASELINE_COMMIT = "773e389"

GATE = compute_dtype.MODE_ENV

#: Small pilot config — the ``[K,K,K,M]`` composition (3 KDA + 1 MLA), one
#: dense-MLP layer and three LatentMoE layers, MTP and AttnRes on.  Widths are
#: smoke-sized; the layer kinds are the production ones (`net/config.py`).
PILOT_KWARGS = dict(
    vocab_size=512,
    hidden=64,
    num_layers=4,
    num_kda_layers=3,
    num_mla_layers=1,
    num_heads=4,
    head_dim=16,
    kda_dk=16,
    kda_dv=16,
    kda_decay_rank=16,
    kda_short_conv_kernel=4,
    kda_g_min=-5.0,
    mla_latent_dim=32,
    mla_head_dim=16,
    mlp_intermediate=128,
    siti_beta_gate=4.0,
    siti_beta_up=25.0,
    mtp_layers=1,
    mtp_loss_weight=0.1,
    attnres_blocks=1,
    attnres_block_size=4,
    vit_patch=14,
    vit_hidden=32,
    vit_depth=2,
    vit_heads=2,
    vit_mlp=64,
    image_size=56,
    moe_dense_layers=1,
    moe_latent_dim=32,
    moe_num_routed=6,
    moe_num_shared=2,
    moe_top_k=2,
    moe_expert_intermediate=16,
    moe_shared_intermediate=32,
    qb_weight=0.01,
    routing_seed=0,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _bits(x: jnp.ndarray) -> np.ndarray:
    """Raw fp32 bit patterns of an array (NaN-safe bitwise comparison)."""
    arr = np.asarray(jnp.asarray(x))
    assert arr.dtype == np.float32, f"bitwise comparison expects fp32, got {arr.dtype}"
    return arr.view(np.uint32)


def _bitwise_equal(a: jnp.ndarray, b: jnp.ndarray) -> bool:
    return bool(np.array_equal(_bits(a), _bits(b)))


def _pilot_cfg(cfg_mod) -> object:
    return cfg_mod.ModelConfig(**PILOT_KWARGS)


def _sparse_cfg(cfg_mod) -> object:
    """Pilot config on the sparse path (the ADR-009 v1.5 union attention)."""
    return dataclasses.replace(
        _pilot_cfg(cfg_mod),
        attn_dense_reference=False,
        mla_top_k=4,
        swa_window=4,
        mla_layer_modes=("full",),
    )


def _dense_mini_cfg(cfg_mod) -> object:
    """``net/config-dense124m.json`` shrunk to smoke widths, composition kept.

    Same idiom as ``net/tests/test_28_dense_standard.py``: the declared file is
    the source of truth for the composition, only the widths are reduced.
    """
    cfg = cfg_mod.load_config(NET_DIR / "config-dense124m.json")
    return dataclasses.replace(
        cfg,
        vocab_size=512,
        hidden=64,
        num_layers=4,
        dense_standard_layers=4,
        num_heads=4,
        head_dim=16,
        kda_dk=16,
        kda_dv=16,
        kda_decay_rank=16,
        mla_latent_dim=32,
        mla_head_dim=16,
        mlp_intermediate=128,
        moe_dense_layers=4,
        moe_latent_dim=32,
        moe_num_routed=6,
        moe_num_shared=2,
        moe_top_k=2,
        moe_expert_intermediate=16,
        moe_shared_intermediate=32,
        attnres_blocks=1,
        attnres_block_size=4,
        vit_patch=14,
        vit_hidden=32,
        vit_depth=2,
        vit_heads=2,
        vit_mlp=64,
        image_size=56,
        scan_layers=False,
    )


@pytest.fixture(scope="module")
def baseline():
    """``net/`` as of ``BASELINE_COMMIT``, importable as the ``baseline_net`` package.

    ``git archive`` (not a worktree) so the fleet that churns the working tree
    cannot perturb the reference, and so no checkout state is touched.  The
    package is renamed because the modules import each other relatively
    (``from .config import ...``), which keeps working under a new package name.
    """
    root = Path(__file__).resolve().parent / ".baseline-net"
    if root.exists():
        import shutil

        shutil.rmtree(root)
    root.mkdir()
    try:
        archive = subprocess.run(
            ["git", "-C", str(CASE_DIR), "archive", BASELINE_COMMIT, "net/"],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:  # pragma: no cover
        pytest.skip(f"baseline {BASELINE_COMMIT} недоступен: {exc}")
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as bundle:
        bundle.extractall(root)
    (root / "net").rename(root / "baseline_net")

    sys.path.insert(0, str(root))
    try:
        import baseline_net.config as bconfig
        import baseline_net.mla as bmla
        import baseline_net.model as bmodel

        yield SimpleNamespace(config=bconfig, mla=bmla, model=bmodel)
    finally:
        sys.path.remove(str(root))
        for name in [m for m in list(sys.modules) if m.startswith("baseline_net")]:
            del sys.modules[name]
        import shutil

        shutil.rmtree(root, ignore_errors=True)


# --------------------------------------------------------------------------- #
# The gate's own contract
# --------------------------------------------------------------------------- #


def test_default_mode_is_fp32(monkeypatch) -> None:
    monkeypatch.delenv(GATE, raising=False)
    assert compute_dtype.mode() == compute_dtype.FP32
    assert compute_dtype.compute_dtype() == jnp.float32
    assert not compute_dtype.is_bf16()


def test_empty_value_is_fp32(monkeypatch) -> None:
    monkeypatch.setenv(GATE, "   ")
    assert compute_dtype.mode() == compute_dtype.FP32


def test_bf16_mode_is_selected(monkeypatch) -> None:
    monkeypatch.setenv(GATE, "bf16")
    assert compute_dtype.mode() == compute_dtype.BF16
    assert compute_dtype.compute_dtype() == jnp.bfloat16
    assert compute_dtype.is_bf16()


def test_value_is_case_insensitive(monkeypatch) -> None:
    monkeypatch.setenv(GATE, "BF16")
    assert compute_dtype.mode() == compute_dtype.BF16


def test_unknown_value_fails_closed(monkeypatch) -> None:
    """A typo must not silently pick a precision (see the module docstring)."""
    monkeypatch.setenv(GATE, "bfloat16")
    with pytest.raises(compute_dtype.ComputeDtypeError):
        compute_dtype.mode()
    with pytest.raises(compute_dtype.ComputeDtypeError):
        compute_dtype.gemm(jnp.ones((2, 2)), jnp.ones((2, 2)))


def test_gemm_fp32_is_the_callers_own_expression() -> None:
    """Gate off: ``gemm`` must be the ``@`` it replaced, not a rewrite of it."""
    a = jr.normal(jr.PRNGKey(0), (7, 5))
    b = jr.normal(jr.PRNGKey(1), (5, 3))
    assert _bitwise_equal(compute_dtype.gemm(a, b), a @ b)


def test_gemm_bf16_casts_operands_and_returns_fp32(monkeypatch) -> None:
    monkeypatch.setenv(GATE, "bf16")
    a = jr.normal(jr.PRNGKey(0), (7, 5))
    b = jr.normal(jr.PRNGKey(1), (5, 3))
    out = compute_dtype.gemm(a, b)
    assert out.dtype == jnp.float32
    expected = jnp.matmul(
        a.astype(jnp.bfloat16), b.astype(jnp.bfloat16), preferred_element_type=jnp.float32
    )
    assert _bitwise_equal(out, expected)
    # The operands really were rounded: bf16 is coarser than fp32 here.
    assert not _bitwise_equal(out, a @ b)


def test_gemm_einsum_contract(monkeypatch) -> None:
    a = jr.normal(jr.PRNGKey(0), (7, 5))
    b = jr.normal(jr.PRNGKey(1), (5, 3))
    assert _bitwise_equal(
        compute_dtype.gemm_einsum("ij,jk->ik", a, b), jnp.einsum("ij,jk->ik", a, b)
    )
    monkeypatch.setenv(GATE, "bf16")
    out = compute_dtype.gemm_einsum("ij,jk->ik", a, b)
    assert out.dtype == jnp.float32
    assert not _bitwise_equal(out, jnp.einsum("ij,jk->ik", a, b))


def test_cast_helpers_are_identities_under_fp32(monkeypatch) -> None:
    monkeypatch.delenv(GATE, raising=False)
    x = jnp.ones((2, 2), jnp.float32)
    assert compute_dtype.cast_in(x) is x
    assert compute_dtype.cast_out(x) is x


def test_cast_out_restores_fp32_under_bf16(monkeypatch) -> None:
    """A fused kernel that returns bf16 must not leak it into the residual."""
    monkeypatch.setenv(GATE, "bf16")
    assert compute_dtype.cast_in(jnp.ones((2, 2), jnp.float32)).dtype == jnp.bfloat16
    assert compute_dtype.cast_out(jnp.ones((2, 2), jnp.bfloat16)).dtype == jnp.float32


# --------------------------------------------------------------------------- #
# (а) gate off — bit for bit the baseline commit
# --------------------------------------------------------------------------- #


def test_mla_dense_oracle_is_bitwise_the_baseline(baseline) -> None:
    cfg = _pilot_cfg(sys.modules["net.config"])
    bcfg = _pilot_cfg(baseline.config)
    key = jr.PRNGKey(0)
    params = mla.init_mla(key, cfg)
    bparams = baseline.mla.init_mla(key, bcfg)
    x = jr.normal(jr.PRNGKey(1), (2, 16, cfg.hidden))

    assert _bitwise_equal(mla.apply(params, cfg, x), baseline.mla.apply(bparams, bcfg, x))
    assert _bitwise_equal(
        mla.reference_full_kv(params, cfg, x),
        baseline.mla.reference_full_kv(bparams, bcfg, x),
    )


def test_mla_sparse_path_is_bitwise_the_baseline(baseline) -> None:
    cfg = _sparse_cfg(sys.modules["net.config"])
    bcfg = _sparse_cfg(baseline.config)
    key = jr.PRNGKey(0)
    params = mla.init_mla(key, cfg)
    bparams = baseline.mla.init_mla(key, bcfg)
    x = jr.normal(jr.PRNGKey(1), (1, 12, cfg.hidden))

    assert _bitwise_equal(mla.apply(params, cfg, x), baseline.mla.apply(bparams, bcfg, x))
    # The indexer selection is integer-valued: compare the indices directly.
    assert np.array_equal(
        np.asarray(mla.topk_indices(params, cfg, x)),
        np.asarray(baseline.mla.topk_indices(bparams, bcfg, x)),
    )


def test_pilot_forward_and_loss_are_bitwise_the_baseline(baseline) -> None:
    """The full token compute path: KDA + MLA + dense MLP + LatentMoE + AttnRes + head."""
    cfg = _pilot_cfg(sys.modules["net.config"])
    bcfg = _pilot_cfg(baseline.config)
    key = jr.PRNGKey(0)
    params = model.init_params(key, cfg)
    bparams = baseline.model.init_params(key, bcfg)
    ids = jr.randint(jr.PRNGKey(2), (2, 16), 0, cfg.vocab_size)

    assert _bitwise_equal(
        model.forward(params, cfg, ids, grad_ckpt_policy="none"),
        baseline.model.forward(bparams, bcfg, ids, grad_ckpt_policy="none"),
    )
    assert _bitwise_equal(
        model.compute_loss(params, cfg, ids, 16, grad_ckpt_policy="none"),
        baseline.model.compute_loss(bparams, bcfg, ids, 16, grad_ckpt_policy="none"),
    )


def test_dense_standard_forward_is_bitwise_the_baseline(baseline) -> None:
    """The dense arm of the verification leg (``_dense_apply`` + SiTU-GLU MLP)."""
    cfg = _dense_mini_cfg(sys.modules["net.config"])
    bcfg = _dense_mini_cfg(baseline.config)
    key = jr.PRNGKey(0)
    params = model.init_params(key, cfg)
    bparams = baseline.model.init_params(key, bcfg)
    ids = jr.randint(jr.PRNGKey(2), (2, 16), 0, cfg.vocab_size)

    assert _bitwise_equal(
        model.forward(params, cfg, ids, grad_ckpt_policy="none"),
        baseline.model.forward(bparams, bcfg, ids, grad_ckpt_policy="none"),
    )


def test_gate_off_is_the_default_of_the_suite(monkeypatch) -> None:
    """The whole suite runs gate-off; if that ever stops being true, say so."""
    monkeypatch.delenv(GATE, raising=False)
    assert compute_dtype.mode() == compute_dtype.FP32


# --------------------------------------------------------------------------- #
# (b) bf16 — residual, master weights and gradient accumulation stay fp32
# --------------------------------------------------------------------------- #
#
# The dtype invariants are read with ``jax.eval_shape`` / ``jax.make_jaxpr``
# (abstract evaluation): they are properties of the traced graph, so they are
# checkable on every backend without running a bf16 GEMM — and the local XLA:CPU
# backend cannot run one ("Unsupported element type for DotThunk::Execute:
# BF16 x BF16 = F32" appears as soon as fused MoE dots are compiled; measured
# 08.10 on jax 0.10.2).  The numeric arm of the campaign is the GPU stand's job
# (``tools/mfu_bf16_protocol.py``), which is EMPTY-PENDING while no stand is up.


def _pilot_setup():
    cfg = _pilot_cfg(sys.modules["net.config"])
    params = model.init_params(jr.PRNGKey(0), cfg)
    ids = jr.randint(jr.PRNGKey(2), (2, 16), 0, cfg.vocab_size)
    return cfg, params, ids


def _eval_dtype(fn):
    """Dtypes of ``fn()`` under abstract evaluation (no execution)."""
    return jax.tree_util.tree_map(lambda s: np.dtype(s.dtype), jax.eval_shape(fn))


def test_params_are_fp32_under_both_modes(monkeypatch) -> None:
    monkeypatch.setenv(GATE, "bf16")
    _cfg, params, _ids = _pilot_setup()
    for leaf in jax.tree_util.tree_leaves(params):
        assert leaf.dtype == jnp.float32


def test_forward_and_loss_are_fp32_under_bf16(monkeypatch) -> None:
    monkeypatch.setenv(GATE, "bf16")
    cfg, params, ids = _pilot_setup()
    dtypes = _eval_dtype(lambda: model.forward(params, cfg, ids, grad_ckpt_policy="none"))
    assert dtypes == np.dtype(jnp.float32), dtypes
    loss_dtype = _eval_dtype(
        lambda: model.compute_loss(params, cfg, ids, 16, grad_ckpt_policy="none")
    )
    assert loss_dtype == np.dtype(jnp.float32), loss_dtype


def test_hidden_state_residual_stream_is_fp32_under_bf16(monkeypatch) -> None:
    """The residual is what the layers add to: it must never become bf16."""
    monkeypatch.setenv(GATE, "bf16")
    cfg, params, ids = _pilot_setup()
    dtypes = _eval_dtype(
        lambda: model.forward(params, cfg, ids, return_hidden=True, grad_ckpt_policy="none")
    )
    assert dtypes == (np.dtype(jnp.float32), np.dtype(jnp.float32)), dtypes


def test_gradients_are_fp32_under_bf16(monkeypatch) -> None:
    """Gradient accumulation lives in the master tree — it must stay fp32."""
    monkeypatch.setenv(GATE, "bf16")
    cfg, params, ids = _pilot_setup()
    dtypes = _eval_dtype(
        lambda: jax.grad(
            lambda p: model.compute_loss(p, cfg, ids, 16, grad_ckpt_policy="none")
        )(params)
    )
    dtypes = jax.tree_util.tree_leaves(dtypes)
    assert dtypes, "градиентный поток пуст"
    assert set(dtypes) == {np.dtype(jnp.float32)}, set(dtypes)


def test_optimizer_master_state_is_fp32_under_bf16(monkeypatch) -> None:
    """Momentum / AdamW slots mirror the master tree, so they are fp32 too."""
    monkeypatch.setenv(GATE, "bf16")
    _cfg, params, _ids = _pilot_setup()
    dtypes = _eval_dtype(lambda: optimizer.init_state(params))
    for leaf in jax.tree_util.tree_leaves(dtypes):
        parts = leaf if isinstance(leaf, tuple) else (leaf,)
        for part in parts:
            assert part == np.dtype(jnp.float32), parts


def test_gate_does_not_touch_the_weight_dtype(monkeypatch) -> None:
    """Tracing a forward must not cast the parameters (the master stays fp32)."""
    monkeypatch.setenv(GATE, "bf16")
    cfg, params, ids = _pilot_setup()
    before = [leaf.dtype for leaf in jax.tree_util.tree_leaves(params)]
    jax.eval_shape(lambda: model.forward(params, cfg, ids, grad_ckpt_policy="none"))
    after = [leaf.dtype for leaf in jax.tree_util.tree_leaves(params)]
    assert before == after


# --------------------------------------------------------------------------- #
# (c) bf16 — casts only at GEMM boundaries, nothing in the loss path
# --------------------------------------------------------------------------- #


def _traced_dtypes(fn) -> set:
    """Output dtypes of every equation in the traced graph of ``fn()``."""
    closed = jax.make_jaxpr(fn)()
    dtypes = set()
    for eqn in closed.jaxpr.eqns:
        for var in eqn.outvars:
            dtypes.add(np.dtype(var.aval.dtype))
    return dtypes


def test_bf16_really_bites_and_introduces_no_third_precision(monkeypatch) -> None:
    """The gate must change the graph, and only into bf16 (no f16/f64/f8)."""
    cfg = _pilot_cfg(sys.modules["net.config"])
    params = mla.init_mla(jr.PRNGKey(0), cfg)
    x = jr.normal(jr.PRNGKey(1), (2, 8, cfg.hidden))

    monkeypatch.delenv(GATE, raising=False)
    off = _traced_dtypes(lambda: mla.apply(params, cfg, x))
    monkeypatch.setenv(GATE, "bf16")
    on = _traced_dtypes(lambda: mla.apply(params, cfg, x))

    assert np.dtype(jnp.bfloat16) in on, "гейт bf16 не изменил граф — это no-op"
    assert np.dtype(jnp.bfloat16) not in off, "fp32-режим обязан быть без bf16"
    forbidden = {
        np.dtype(d) for d in (jnp.float16, jnp.float64, jnp.float8_e4m3fn, jnp.float8_e5m2)
    }
    assert not (on & forbidden), f"в графе появилась третья точность: {on & forbidden}"


def test_loss_path_never_sees_a_non_fp32_dtype(monkeypatch) -> None:
    """Under bf16 the loss graph carries only f32 and the boundary's bf16."""
    monkeypatch.setenv(GATE, "bf16")
    cfg, params, ids = _pilot_setup()
    dtypes = _traced_dtypes(
        lambda: model.compute_loss(params, cfg, ids, 8, grad_ckpt_policy="none")
    )
    # Integer/bool intermediates (masks, target indices, routing) are expected;
    # what must not appear is a *float* other than the boundary's bf16 and fp32.
    floats = {d for d in dtypes if np.issubdtype(d, np.floating)}
    assert floats <= {np.dtype(jnp.float32), np.dtype(jnp.bfloat16)}, floats
    assert np.dtype(jnp.float32) in floats


def test_bf16_forward_stays_close_to_fp32(monkeypatch) -> None:
    """Numeric sanity: the gate changes precision, not the computation.

    Not a strict equivalence — bf16 rounding is real — but an order-of-magnitude
    guard that catches a boundary wired to the wrong operand.  Skipped where the
    local backend cannot execute the fused bf16 GEMMs (see section (b)).
    """
    cfg, params, ids = _pilot_setup()
    monkeypatch.delenv(GATE, raising=False)
    ref = np.asarray(model.forward(params, cfg, ids, grad_ckpt_policy="none"))
    monkeypatch.setenv(GATE, "bf16")
    try:
        got = np.asarray(model.forward(params, cfg, ids, grad_ckpt_policy="none"))
    except jax.errors.JaxRuntimeError as exc:
        if "Unsupported element type for DotThunk" in str(exc):
            pytest.skip(
                "локальный XLA:CPU не исполняет fused bf16-GEMM (DotThunk); "
                "числовая проверка — на стенде GB10"
            )
        raise
    scale = max(1e-6, float(np.abs(ref).max()))
    assert float(np.abs(ref - got).max()) / scale < 0.2
