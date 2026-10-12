"""Fused ``chunked_cc`` body under ``jax.grad`` (ADR-054 stage-2b, milestone 2): CPU verification.

The backward half of the fused body is the analytic adjoint
(:func:`net.kernels.kda_cc_fused_bwd.reference_bwd`) wired into a
``jax.custom_vjp`` (:data:`net.kernels.kda_cc_fused.cc_step`), so the
``AXIOM_KDA_CC_IMPL=mosaic`` body is trainable: the forward rule runs the
Mosaic kernel, the backward rule runs the jnp adjoint.  Variant B of the
delta: fwd and remat-fwd collapse into one launch per chunk, the backward
pass runs as fused XLA/jnp — the fused backward *kernel* is a scoped follow-up
(its blocker and the derived design are recorded in the backward module's
docstring).

What is pinned on this CPU node (and what is not):

* ``reference_bwd`` against ``jax.grad`` of ``kda_cc_fused.reference`` — the
  fp32 replica the milestone-1 tests pin against the committed
  ``cc_chunk_step`` at 2e-5.  fp32 inputs: worst |diff| ~1e-6 (the adjoint is
  the same algebra jax differentiates, up to summation order); bf16 inputs:
  ``jax`` computes the backward *in bf16*, this adjoint in fp32 and rounds the
  cotangent once — agreement is at the bf16 ulp class (declared 2e-2 budget);
* end-to-end gradient parity of the **real mux**: ``AXIOM_KDA_CC_IMPL=mosaic``
  with ``fused.device_available`` forced true and ``fused.kernel`` substituted
  by the fp32 replica (the only substitution — everything else is the shipped
  ``_cc_chunk_step_mosaic``: casts, beta32 plumbing, output gate) vs
  ``AXIOM_KDA_CC_IMPL=jax``, gradients w.r.t. the layer input at the campaign
  geometry.  Measured deviation is ~1e-3 (three to thirty times under the
  declared 2e-2 operand budget; the gap is the bf16 operand path, which exists
  in the mosaic mode by design);
* jvp/finite-difference cross-check of the adjoint (numerical vjp reference);
* the custom_vjp routing (forward = kernel seam, backward = the sibling
  adjoint), forward-value preservation, and stability over repeated calls;
* **not verifiable here**: numerical parity of the *executed* kernel (no GPU;
  jax 0.11.2 interpret mode has no rule for ``plgpu.mma``) and the launch
  counts / timing of the training step — the architect's GB10 A/B (the
  ``open_questions`` of this delta: tokens/s of a full ``AXIOM_KDA_CC_IMPL=mosaic``
  step, nsys kernels/step against the jax arm, 4-step loss parity).
"""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from net import kda
from net.config import ModelConfig
from net.kernels import kda_cc_fused as fused
from net.kernels import kda_cc_fused_bwd as bwd

C = 32
DK = 128
DV = 128

#: Declared operand budget of the bf16 path (the milestone-1 tolerance class).
BF16_ATOL = 2e-2
BF16_RTOL = 2e-2

#: fp32 gradcheck tolerance (the convention of the sibling gradcheck in
#: ``test_kernel_kda_ut_solve.py``: 1e-4 / 1e-3 over a measured ~1e-6).
F32_ATOL = 1e-4
F32_RTOL = 1e-3


def _cfg(H=2, *, chunk=32, g_min=-5.0):
    """A KDA-only config at the kernel's geometry (hidden = H*dv, structural)."""
    return ModelConfig(
        hidden=H * DV, num_heads=H, head_dim=DK,
        kda_dk=DK, kda_dv=DV, kda_decay_rank=128, kda_short_conv_kernel=4,
        kda_g_min=g_min, kda_wyut_chunk=chunk,
    )


def _inputs(Cc, Dk, Dv, H, seed, *, dtype, decay=2.5, damp=0.5):
    keys = jr.split(jr.PRNGKey(seed), 6)
    q = (jr.normal(keys[0], (Cc, H * Dk)) * damp).astype(dtype)
    k = (jr.normal(keys[1], (Cc, H * Dk)) * damp).astype(dtype)
    v = (jr.normal(keys[2], (Cc, H * Dv)) * damp).astype(dtype)
    alpha = jnp.exp(-decay * jr.uniform(keys[3], (Cc, H * Dk)))
    beta = jr.uniform(keys[4], (Cc, H))
    beta32 = jnp.broadcast_to(beta[:, :, None], (Cc, H, 32)).reshape(Cc, H * 32)
    S = jr.normal(keys[5], (H, Dk, Dv)) * 0.1
    return q, k, v, alpha, beta32, S


def _assert_grad_close(got, ref, name, *, shape, seed, tol):
    """Scale-relative grad comparison.

    The adjoint's fp32 noise is *absolute* in the summation order of the same
    algebra, i.e. it is naturally measured against the largest entry of the
    gradient — an elementwise ``atol`` would be dominated by the many small
    entries of a large-gradient sample.  Observed at the campaign geometry:
    ``max|d| / max|ref|`` ~2e-5 at the campaign decay; ``tol=1e-3`` is a
    declared bound with >1 order of headroom, not a fitted one.
    """
    scale = max(1.0, float(jnp.max(jnp.abs(ref.astype(jnp.float32)))))
    err = float(jnp.max(jnp.abs(got.astype(jnp.float32) - ref.astype(jnp.float32))))
    assert err <= tol * scale, (
        f"d{name} shape={shape} seed={seed}: max|d|={err:.3e} scale={scale:.3e} "
        f"(bound {tol * scale:.3e})")


# ---------------------------------------------------------------------------
# the analytic adjoint: gradcheck against jax.grad of the fp32 replica
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype,tol", [
    (jnp.float32, "tight"),
    (jnp.bfloat16, "bf16"),
])
@pytest.mark.parametrize("shape,seed", [
    ((8, 4, 2, 1), 0), ((16, 8, 4, 2), 1), ((32, 16, 8, 2), 2),
    ((32, 128, 128, 2), 3),
])
def test_reference_bwd_matches_grad_of_reference(dtype, tol, shape, seed):
    """``reference_bwd`` == ``jax.grad(reference)`` (the adjoint is exact).

    fp32: both sides are the same algebra — the residual is summation order,
    bounded scale-relative (see :func:`_assert_grad_close`).  bf16: jax
    computes the whole backward in bf16 while this adjoint accumulates fp32
    and rounds the cotangent once, so the honest assertion is the declared
    operand budget (2e-2).
    """
    Cc, Dk, Dv, H = shape
    q, k, v, alpha, beta32, S = _inputs(Cc, Dk, Dv, H, seed, dtype=dtype)
    go = jr.normal(jr.PRNGKey(seed + 50), (Cc, H * Dv))
    gS = jr.normal(jr.PRNGKey(seed + 51), (H, Dk, Dv))

    def loss(*a):
        o, s_new = fused.reference(*a)
        return jnp.sum(o * go) + jnp.sum(s_new * gS)

    grads = jax.grad(loss, argnums=(0, 1, 2, 3, 4, 5))(q, k, v, alpha, beta32, S)
    mine = bwd.reference_bwd(q, k, v, alpha, beta32, S, go, gS, tile=16)

    for name, ref, got in zip("qkvabS", grads, mine):
        assert got.dtype == ref.dtype
        if tol == "tight":
            _assert_grad_close(got, ref, name, shape=shape, seed=seed, tol=1e-3)
        else:
            assert jnp.allclose(got.astype(jnp.float32), ref.astype(jnp.float32),
                                atol=BF16_ATOL, rtol=BF16_RTOL), (
                f"d{name} shape={shape} seed={seed}: "
                f"max|d|={float(jnp.max(jnp.abs(got.astype(jnp.float32) - ref.astype(jnp.float32)))):.3e}")


def test_reference_bwd_gamma_c_axis_is_observable_with_weak_decay():
    """The gamma^C row-broadcast is exercised at a geometry where it is not ~0.

    At the campaign decay (``g_min=-5``, C=32) ``gamma^C`` underflows to ~1e-35
    and a row/column axis mix-up in that term is numerically invisible — it
    was in fact present in the milestone-1 emulation.  With a weak decay the
    term is observable; the adjoint and the replica must still agree tightly
    (fp32) on this geometry, which pins the axis in both.
    """
    key = jr.PRNGKey(7)
    Cc, Dk, Dv, H = 8, 4, 2, 1
    q, k, v, alpha, beta32, S = _inputs(Cc, Dk, Dv, H, 7, dtype=jnp.float32,
                                        decay=0.05)
    # weak decay -> gamma^C = exp(sum la) is O(1): the gc term carries mass
    la = jnp.log(alpha)
    assert float(jnp.exp(jnp.sum(la, axis=0)).min()) > 0.1
    go = jr.normal(jr.PRNGKey(77), (Cc, H * Dv))
    gS = jr.normal(jr.PRNGKey(78), (H, Dk, Dv))

    def loss(*a):
        o, s_new = fused.reference(*a)
        return jnp.sum(o * go) + jnp.sum(s_new * gS)

    grads = jax.grad(loss, argnums=(0, 1, 2, 3, 4, 5))(q, k, v, alpha, beta32, S)
    mine = bwd.reference_bwd(q, k, v, alpha, beta32, S, go, gS, tile=16)
    for name, ref, got in zip("qkvabS", grads, mine):
        assert jnp.allclose(got, ref, atol=F32_ATOL, rtol=F32_RTOL), (
            f"d{name}: max|d|={float(jnp.max(jnp.abs(got - ref))):.3e}")


# ---------------------------------------------------------------------------
# jvp / numerical vjp cross-check
# ---------------------------------------------------------------------------


def test_adjoint_matches_jvp_and_finite_differences():
    """Directional derivative: ``<adjoint cotangents, tangents>`` == jvp == FD.

    ``jax.jvp`` gives the exact directional derivative of the scalar loss; the
    inner product with the adjoint's cotangents is the vjp identity; central
    finite differences are the *numerical* reference that assumes no algebra.
    """
    Cc, Dk, Dv, H = 8, 4, 2, 1
    q, k, v, alpha, beta32, S = _inputs(Cc, Dk, Dv, H, 3, dtype=jnp.float32)
    go = jr.normal(jr.PRNGKey(30), (Cc, H * Dv))
    gS = jr.normal(jr.PRNGKey(31), (H, Dk, Dv))

    def loss(q, k, v, alpha, beta32, S):
        o, s_new = fused.reference(q, k, v, alpha, beta32, S)
        return jnp.sum(o * go) + jnp.sum(s_new * gS)

    tangents = _inputs(Cc, Dk, Dv, H, 4, dtype=jnp.float32)
    cot = (go, gS)
    dq, dk, dv, da, db, ds = bwd.reference_bwd(
        q, k, v, alpha, beta32, S, *cot, tile=16)

    jvp_exact = jax.jvp(loss, (q, k, v, alpha, beta32, S), tangents)[1]
    from_adjoint = (jnp.sum(dq * tangents[0]) + jnp.sum(dk * tangents[1])
                    + jnp.sum(dv * tangents[2]) + jnp.sum(da * tangents[3])
                    + jnp.sum(db * tangents[4]) + jnp.sum(ds * tangents[5]))
    assert jnp.allclose(from_adjoint, jvp_exact, atol=1e-4, rtol=1e-5), (
        f"vjp identity: adjoint={float(from_adjoint):.6f} jvp={float(jvp_exact):.6f}")

    eps = 1e-3
    hi = loss(*[x + eps * t for x, t in zip((q, k, v, alpha, beta32, S), tangents)])
    lo = loss(*[x - eps * t for x, t in zip((q, k, v, alpha, beta32, S), tangents)])
    fd = (hi - lo) / (2 * eps)
    assert jnp.allclose(fd, jvp_exact, atol=2e-4, rtol=2e-4), (
        f"finite differences: fd={float(fd):.6f} jvp={float(jvp_exact):.6f}")


# ---------------------------------------------------------------------------
# end-to-end: the real mux under jax.grad, via the CPU kernel substitution
# ---------------------------------------------------------------------------


def _cpu_kernel_stub(q, k, v, alpha, beta32, S, tile=16):
    """The kernel seam: the fp32 replica (the only substitution in these tests)."""
    return fused.reference(q, k, v, alpha, beta32, S, tile=tile)


def _enable_mosaic_on_cpu(monkeypatch):
    """Force the mosaic mux on this node with the kernel stubbed to the replica."""
    monkeypatch.setenv("AXIOM_KDA_CC_IMPL", "mosaic")
    monkeypatch.setattr(fused, "device_available", lambda: True)
    monkeypatch.setattr(fused, "kernel", _cpu_kernel_stub)


@pytest.mark.parametrize("H,T", [(2, 128), (2, 256)])
@pytest.mark.parametrize("seed", [0, 1])
def test_grad_of_the_mux_matches_the_jax_body(monkeypatch, H, T, seed):
    """``jax.grad`` through the mosaic body == through the jax body (2e-2 class).

    Both arms are the *shipped* code path: the flag mux, the projections, the
    short-conv, the post-processing and (mosaic arm) the bf16 operand cast are
    the real ones; the kernel execution is substituted by its fp32 replica,
    which the milestone-1 tests pin to ``cc_chunk_step`` at 2e-5.  The
    remaining deviation is therefore the bf16 operand path — the declared
    budget of the mode.
    """
    cfg = _cfg(H)
    params = kda.init_kda(jr.PRNGKey(seed), cfg)
    x = jr.normal(jr.PRNGKey(seed + 5), (T, cfg.hidden))
    w = jr.normal(jr.PRNGKey(seed + 6), (T, cfg.hidden))

    monkeypatch.setenv("AXIOM_KDA_CC_IMPL", "jax")
    loss_jax = lambda xx: jnp.sum(kda.apply_chunked_cc(params, cfg, xx, chunk_size=C) * w)
    g_jax = jax.grad(loss_jax)(x)

    _enable_mosaic_on_cpu(monkeypatch)
    loss_mosaic = lambda xx: jnp.sum(kda.apply_chunked_cc(params, cfg, xx, chunk_size=C) * w)
    g_mosaic = jax.grad(loss_mosaic)(x)

    _assert_grad_close(g_mosaic, g_jax, "x", shape=(T, cfg.hidden), seed=seed,
                       tol=BF16_RTOL)


def test_grad_of_the_mux_matches_the_jax_body_wrt_params(monkeypatch):
    """The same parity for parameter gradients (the training-relevant direction)."""
    H, T, seed = 2, 128, 2
    cfg = _cfg(H)
    params = kda.init_kda(jr.PRNGKey(seed), cfg)
    x = jr.normal(jr.PRNGKey(seed + 5), (T, cfg.hidden))
    w = jr.normal(jr.PRNGKey(seed + 6), (T, cfg.hidden))

    monkeypatch.setenv("AXIOM_KDA_CC_IMPL", "jax")
    f = lambda p: jnp.sum(kda.apply_chunked_cc(p, cfg, x, chunk_size=C) * w)
    g_jax = jax.grad(f)(params)

    _enable_mosaic_on_cpu(monkeypatch)
    g_mosaic = jax.grad(f)(params)

    for name in ("W_q", "W_k", "W_v", "W_o", "W_g", "W_beta", "W_a_down",
                 "W_a_up", "conv_q", "conv_k", "conv_v"):
        _assert_grad_close(getattr(g_mosaic, name), getattr(g_jax, name), name,
                           shape=(T, cfg.hidden), seed=seed, tol=BF16_RTOL)


# ---------------------------------------------------------------------------
# the custom_vjp wiring: routing, forward preservation, repeatability
# ---------------------------------------------------------------------------


def test_custom_vjp_routes_forward_and_backward(monkeypatch):
    """The shipped ``cc_step`` uses the kernel seam forward and the adjoint back.

    A bare ``pallas_call`` cannot be differentiated, so observing the adjoint
    on the backward of ``fused.cc_step`` proves the ``custom_vjp`` path is in
    force (the sibling-kernel-routing test of ``kda_ut_solve``, transposed to
    this body).  The forward value must be exactly the seam's value.
    """
    calls = {"fwd": 0, "bwd": 0}
    real_bwd = bwd.reference_bwd

    def stub(q, k, v, alpha, beta32, S, tile=16):
        calls["fwd"] += 1
        return fused.reference(q, k, v, alpha, beta32, S, tile=tile)

    def spy_bwd(q, k, v, alpha, beta32, S, do, dS, tile=16):
        calls["bwd"] += 1
        return real_bwd(q, k, v, alpha, beta32, S, do, dS, tile=tile)

    monkeypatch.setattr(fused, "kernel", stub)
    monkeypatch.setattr(bwd, "reference_bwd", spy_bwd)

    Cc, Dk, Dv, H = 16, 8, 4, 1
    args = _inputs(Cc, Dk, Dv, H, 11, dtype=jnp.bfloat16)
    o_step, s_step = fused.cc_step(*args, 16)
    o_ref, s_ref = stub(*args, 16)
    # the fwd rule runs once for the jaxpr of the primal and once for the fwd
    # rule itself (custom_vjp_call_jaxpr): at least one call, no backward yet
    assert calls["fwd"] >= 1 and calls["bwd"] == 0, calls
    assert jnp.array_equal(o_step, o_ref) and jnp.array_equal(s_step, s_ref), (
        "custom_vjp must not perturb the forward value")

    calls.update(fwd=0, bwd=0)
    go = jr.normal(jr.PRNGKey(40), o_ref.shape)
    gS = jr.normal(jr.PRNGKey(41), s_ref.shape)
    jax.grad(lambda *a: jnp.sum(fused.cc_step(*a, 16)[0] * go)
             + jnp.sum(fused.cc_step(*a, 16)[1] * gS), argnums=(0, 1, 2, 3, 4, 5))(*args)
    assert calls["fwd"] >= 1 and calls["bwd"] >= 1, calls


def test_grad_calls_are_repeatable_and_buffer_safe(monkeypatch):
    """Repeated grad calls across retraced shapes are stable and raise nothing.

    The mosaic body's contract on the device is buffer reuse across chunk
    launches; a host-side regression in the wiring would surface as a changed
    value or an exception on a repeated call.  Five calls, two shapes, jitted:
    each repeat must be bit-identical to the first.
    """
    _enable_mosaic_on_cpu(monkeypatch)
    cfg = _cfg(2)
    params = kda.init_kda(jr.PRNGKey(0), cfg)
    results = []
    for rep in range(5):
        T = 128 if rep % 2 == 0 else 64
        x = jr.normal(jr.PRNGKey(rep), (T, cfg.hidden))
        w = jr.normal(jr.PRNGKey(rep + 20), (T, cfg.hidden))
        g = jax.jit(jax.grad(
            lambda xx: jnp.sum(kda.apply_chunked_cc(params, cfg, xx, chunk_size=C) * w)))(x)
        g2 = jax.jit(jax.grad(
            lambda xx: jnp.sum(kda.apply_chunked_cc(params, cfg, xx, chunk_size=C) * w)))(x)
        assert jnp.array_equal(g, g2), f"repeat {rep} changed the gradient"
        results.append(g)
    assert all(jnp.all(jnp.isfinite(r)) for r in results)


def test_cc_step_and_kernel_contract():
    """``cc_step`` dispatches to cached custom_vjp wrappers; the kernel stays forward-only.

    ``cc_step`` itself is a plain host dispatcher on the static ``tile`` (the
    ``custom_vjp`` object lives one level down — an int argument through
    ``custom_vjp`` would arrive inside the rules as a weak int32 *tracer* and
    the kernel's ``int(tile)`` would fail with ``TracerIntegerConversionError``);
    the wrappers are cached per tile width.
    """
    assert fused.cc_step is not fused.kernel
    assert not hasattr(fused.kernel, "defvjp"), "the bare kernel is forward-only"
    wrapper = fused.make_cc_step()
    assert hasattr(wrapper, "defvjp")
    assert fused.make_cc_step() is not wrapper  # factory returns fresh wrappers
    stub = fused.make_cc_step(forward=_cpu_kernel_stub, backward=bwd.reference_bwd)
    assert hasattr(stub, "defvjp")
    assert fused.cc_step.__doc__ and "custom_vjp" in fused.cc_step.__doc__


# ---------------------------------------------------------------------------
# regression: the declared-algebra fixes of this delta (forward kernel)
# ---------------------------------------------------------------------------


def test_forward_phase4_lam_uses_the_full_row_sum():
    """The phase-4 ``lam`` anchor is the full row sum, not the cumsum.

    The milestone-1 kernel built ``glast`` with the ``tril`` (cumsum) pattern
    and then subtracted the cumsum ``G`` — identically zero, i.e. ``lam = 1``.
    Measured on the declared algebra, that mis-states the state transfer by up
    to ~5e-1 at the campaign geometry; the emulation (and the reference) had
    the correct form, so no CPU test could see the kernel tex.  This pins the
    kernel source (the device check remains the architect's).
    """
    src = inspect.getsource(fused._build_kernel)
    assert "(cidx2 <= C - 1).astype(dt)" in src, "phase-4 full-sum pattern missing"
    # the cumsum pattern must not be used as the lam anchor inside _phase4
    phase4 = src.split("def _phase4")[1]
    assert "(ridx2 >= cidx2).astype(dt)" not in phase4, (
        "phase-4 lam anchor is the cumsum again (lam would be identically 1)")


def test_forward_phase4_gamma_c_is_the_row_broadcast():
    """``gamma^C`` decays the key channels — rows of the ``(dk, dv)`` state.

    The milestone-1 kernel loaded the vector with the column-vector reduce and
    broadcast it along axis 1 (a column scaling).  At the campaign decay the
    term underflows to ~1e-35 and the axis mix-up was numerically invisible
    (both here and on the launch-count probe of milestone 1); the reference
    (pinned to ``cc_chunk_step``) scales rows.  Pinned on the kernel source.
    """
    src = inspect.getsource(fused._build_kernel)
    assert "broadcast_in_dim(gcv, (dk, dv), (0,))" in src
    assert "broadcast_in_dim(gcv, (dk, dv), (1,))" not in src
    assert "ACC_DK.reduce(1)" in src
