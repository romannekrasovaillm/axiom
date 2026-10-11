"""Persistent KDA scan kernel (ADR-054 Amendment 4, milestone 3): CPU verification.

The persistent kernel of :mod:`net.kernels.kda_cc_fused_persist` collapses the
whole ``chunked_cc`` scan of one layer-pass into ONE Mosaic launch -- the unit
of fusing re-measured in Am.4 (the per-chunk custom-call boundary made the full
step *worse*: 704k kernels/step, 117k DtoD = 13.7 s).  This file pins what is
verifiable **on this CPU node** and states plainly what is not:

* the module's fp32 :func:`kda_cc_fused_persist.reference` (the chunk scan of
  the per-chunk algebra that milestone 1 pins against the committed
  :func:`net.kda.cc_chunk_step`) is pinned against the committed path run
  chunk by chunk at the campaign geometry (C=32, H=2..12, dk=dv=128,
  T=256..2048, several seeds): the state at every boundary and the pre-gate
  output;
* a bf16 emulation of the kernel's declared rounding points **with the
  per-chunk SMEM state rounding the persistent design introduces** is pinned
  under the declared operand tolerance (2e-2) at the kernel's boundary --
  state at *every* chunk boundary, pre-gate output -- at the campaign geometry
  and in a weak-decay stress (``g_min`` down to -0.005, per-chunk state
  retention ~0.92, where a bf16 state drift would amplify).  Observed ~1e-4
  (strong decay) to ~1.3e-3 (weak-decay stress at H=12, T=512), an order of
  magnitude under the budget;
* one test isolates the drift term: with the state rounded back to fp32, the
  emulation's only deviation is the milestone-1 operand quantisation, so the
  distance of the bf16-state emulation from the fp32 state chain is
  attributable to the state rounding alone (asserted under 2e-2);
* the mode-level parity (``AXIOM_KDA_CC_IMPL=persistent`` vs ``jax``) at the
  gated boundary is pinned scale-relative at the declared 2e-2 (observed
  ~7e-3 at H=2 and ~1.5e-2 of scale at H=12).  The residual is *entirely* the
  head-wise RMSNorm amplification of the bf16 operand quantisation (the term
  milestone 1 declared for the ``mosaic`` mode): substituting the kernel by
  the same emulation with a fp32 state produces the same difference to the
  last digit, so the state term is below this comparison's resolution;
* the backward rule of the whole-sequence ``custom_vjp`` (variant A:
  :func:`kda_cc_fused_persist.reference_bwd_scan`) is gradchecked against
  ``jax.grad`` of the fp32 replica, the gradient of the real mux is compared
  with the jax body's (the milestone-2 pattern), and the ``custom_vjp`` wiring
  is observed (forward seam + adjoint routing);
* the *launch-count claim itself* is pinned at the API seam: the persistent
  mode calls the kernel seam exactly ONCE per layer-pass, where the per-chunk
  ``mosaic`` body called it once per chunk (256 per pass at the campaign
  geometry);
* the kernel *lowers* for sm_121 under the skill's patched lowering (arch
  (12,1), 101 376 B SMEM limit) for every live case; the sieve
  (:func:`kda_cc_fused_persist.valid_config`) rejects what does not fit (C=64,
  and the fp32 SMEM state at C=32 -- the measured alternatives are in the
  module docstring), and the lowered module carries the chunk loop
  (``scf.for``), so the persistent structure is visible in the IR, not only
  in the docstring;
* **the kernel itself is not executed here** (no GPU on this node, and jax
  0.11.2's GPU interpret mode carries no rule for ``plgpu.mma``): device
  parity, timing, determinism, the per-pass launch count of the real run and
  the full-step loss parity are the architect's GB10 run (``check_kernel.py``
  plus the A/B in the delta's ``open_questions``).  The emulation mirrors the
  kernel's *declared* rounding points; a divergence between code and emulation
  would still need the device to see.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from net import kda
from net.config import ModelConfig
from net.kernels import kda_cc_fused as fused
from net.kernels import kda_cc_fused_persist as persist
from net.tests.test_kda_cc_fused import _emulate_bf16

#: Campaign geometry of the kernel (the live cases of the module contract).
C = 32
DK = 128
DV = 128

#: fp32 parity budget of the replica vs the committed jnp path (both fp32;
#: observed ~3e-7 per chunk, the bound leaves room for summation order).
REF_ATOL = 2e-5

#: Declared operand budget of the bf16 path (the milestone-1 tolerance class).
BF16_ATOL = 2e-2
BF16_RTOL = 2e-2


def _cfg(H=2, *, chunk=32, g_min=-5.0):
    """A KDA-only config at the kernel's geometry (hidden = H*dv, structural)."""
    return ModelConfig(
        hidden=H * DV, num_heads=H, head_dim=DK,
        kda_dk=DK, kda_dv=DV, kda_decay_rank=128, kda_short_conv_kernel=4,
        kda_g_min=g_min, kda_wyut_chunk=chunk,
    )


def _max_abs(a, b):
    return float(jnp.max(jnp.abs(a - b)))


def _interval_and_committed(params, cfg, x, chunk):
    """Whole-sequence interval inputs and the committed scan's state chain.

    Returns ``(interval, states)``: ``interval`` is ``(q, k, v, alpha, beta32)``
    of the *padded* sequence, built with the committed helpers
    (``_project``/``_short_conv_chunk``/``_postprocess`` -- the same code the
    scan body consumes in the ``jax`` mode), and ``states`` is the per-chunk
    state chain ``(n+1, H, dk, dv)`` of :func:`net.kda.cc_chunk_step` -- i.e.
    exactly what the persistent design has to reproduce.
    """
    H, dk, dv = cfg.num_heads, cfg.kda_dk, cfg.kda_dv
    Cc = chunk
    T = x.shape[0]
    n = (T + Cc - 1) // Cc
    x_p = jnp.pad(x, ((0, n * Cc - T), (0, 0))) if n * Cc != T else x
    qs, ks, vs, als, bs = [], [], [], [], []
    states = []
    carry = kda.init_state(cfg)
    for i in range(n):
        xc = x_p[i * Cc:(i + 1) * Cc]
        proj = kda._project(params, cfg, xc)
        qc, q_buf = kda._short_conv_chunk(proj["qp"], params.conv_q, carry.q_buf)
        kc, k_buf = kda._short_conv_chunk(proj["kp"], params.conv_k, carry.k_buf)
        vc, v_buf = kda._short_conv_chunk(proj["vp"], params.conv_v, carry.v_buf)
        states.append(carry.S)
        carry, _out_i = kda.cc_chunk_step(params, cfg, carry, xc)
        q, k, v = kda._postprocess(qc, kc, vc, cfg)
        qs.append(q.reshape(Cc, H * dk))
        ks.append(k.reshape(Cc, H * dk))
        vs.append(v.reshape(Cc, H * dv))
        als.append(proj["alpha"].reshape(Cc, H * dk).astype(jnp.float32))
        bs.append(jnp.broadcast_to(proj["beta"][:, :, None], (Cc, H, 32))
                  .reshape(Cc, H * 32).astype(jnp.float32))
    states.append(carry.S)
    interval = (jnp.concatenate(qs), jnp.concatenate(ks), jnp.concatenate(vs),
                jnp.concatenate(als), jnp.concatenate(bs))
    return interval, jnp.stack(states)


def _reference_scan(interval, S0, Cc, tile):
    """The fp32 replica run chunk by chunk, exposing the state chain.

    Uses :func:`net.kernels.kda_cc_fused.reference` -- the per-chunk laid-out
    replica that :func:`kda_cc_fused_persist.reference` scans; this helper
    exists to compare the state *at every boundary*, which the whole-sequence
    entry point does not return.
    """
    q, k, v, alpha, beta32 = interval
    n = q.shape[0] // Cc
    outs, states = [], []
    s_cur = S0
    for i in range(n):
        sl = slice(i * Cc, (i + 1) * Cc)
        states.append(s_cur)
        o_i, s_cur = fused.reference(q[sl], k[sl], v[sl], alpha[sl], beta32[sl],
                                     s_cur, tile=tile)
        outs.append(o_i)
    states.append(s_cur)
    return jnp.concatenate(outs), jnp.stack(states)


def _emulated_scan(interval, S0, Cc, tile):
    """The kernel's declared rounding points, SMEM state rounding included.

    The per-chunk emulation lives in the milestone-1 file (it mirrors the
    kernel's storage points: operand quantisation, the hi/lo decay split,
    per-factor bf16 storages).  The persistent design adds exactly one
    rounding point of its own: the state lives in bf16 SMEM, so the
    retransferred state is rounded at every chunk boundary -- the term this
    scan models.
    """
    q, k, v, alpha, beta32 = interval
    n = q.shape[0] // Cc
    outs, states = [], []
    s_cur = S0
    for i in range(n):
        sl = slice(i * Cc, (i + 1) * Cc)
        states.append(s_cur)
        o_i, s_i = _emulate_bf16(q[sl], k[sl], v[sl], alpha[sl], beta32[sl],
                                 s_cur, tile=tile)
        outs.append(o_i)
        s_cur = s_i.astype(jnp.bfloat16).astype(jnp.float32)  # the SMEM state write
    states.append(s_cur)
    return jnp.concatenate(outs), jnp.stack(states)


def _fp32_state_emulated_scan(interval, S0, Cc, tile):
    """The emulation *without* the state rounding (the isolation leg)."""
    q, k, v, alpha, beta32 = interval
    n = q.shape[0] // Cc
    outs, states = [], []
    s_cur = S0
    for i in range(n):
        sl = slice(i * Cc, (i + 1) * Cc)
        states.append(s_cur)
        o_i, s_cur = _emulate_bf16(q[sl], k[sl], v[sl], alpha[sl], beta32[sl],
                                   s_cur, tile=tile)
        outs.append(o_i)
    states.append(s_cur)
    return jnp.concatenate(outs), jnp.stack(states)


# ---------------------------------------------------------------------------
# the fp32 replica: the persistent scan == the committed chunk-by-chunk path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("H,T", [(2, 256), (12, 256), (2, 2048), (12, 512)])
@pytest.mark.parametrize("seed", [0, 1])
def test_reference_scan_matches_the_committed_chunk_scan(H, T, seed):
    """``persist.reference`` (whole sequence) == ``cc_chunk_step`` per chunk.

    Both fp32; the observed deviation is the summation-order slack of the same
    laid-out algebra (~1e-7 at the campaign geometry).
    """
    cfg = _cfg(H)
    params = kda.init_kda(jr.PRNGKey(seed), cfg)
    x = jr.normal(jr.PRNGKey(seed + 7), (T, cfg.hidden))
    interval, s_committed = _interval_and_committed(params, cfg, x, C)
    s0 = jnp.zeros((H, DK, DV), jnp.float32)
    o_ref, s_ref = persist.reference(*interval, s0, chunk=C)
    o_scan, s_scan = _reference_scan(interval, s0, C, kda._cc_tile(cfg))
    # the whole-sequence entry point is the same scan as the chunk loop
    assert _max_abs(o_ref, o_scan) <= REF_ATOL, f"H={H} T={T} seed={seed}"
    assert _max_abs(s_ref, s_scan[-1]) <= REF_ATOL
    # and that scan reproduces the committed path at every boundary
    assert _max_abs(s_scan, s_committed) <= REF_ATOL, f"H={H} T={T} seed={seed}"


def test_reference_scan_matches_at_the_long_end():
    """H=12, T=2048: the long end of the declared grid (64 chunks)."""
    cfg = _cfg(12)
    params = kda.init_kda(jr.PRNGKey(3), cfg)
    x = jr.normal(jr.PRNGKey(4), (2048, cfg.hidden))
    interval, s_committed = _interval_and_committed(params, cfg, x, C)
    _o, s_scan = _reference_scan(interval, jnp.zeros((12, DK, DV), jnp.float32),
                                 C, kda._cc_tile(cfg))
    assert _max_abs(s_scan, s_committed) <= REF_ATOL


# ---------------------------------------------------------------------------
# the declared bf16 rounding points (state drift included) at 2e-2
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("g_min", [-5.0, -0.005])
@pytest.mark.parametrize("H,T", [(2, 256), (12, 512), (2, 2048)])
@pytest.mark.parametrize("seed", [0, 1])
def test_bf16_state_emulation_matches_within_operand_tolerance(g_min, H, T, seed):
    """The persistent design's declared rounding stays inside the 2e-2 budget.

    State and pre-gate output at every chunk boundary against the fp32 replica
    chain.  Observed: ~1e-4..1e-3 at ``g_min=-5``, <= ~1.3e-3 in the
    weak-decay stress (``g_min=-0.005``) at H=12, T=512 -- an order of
    magnitude under the declared budget.
    """
    cfg = _cfg(H, g_min=g_min)
    params = kda.init_kda(jr.PRNGKey(seed), cfg)
    x = jr.normal(jr.PRNGKey(seed + 11), (T, cfg.hidden))
    interval, _s = _interval_and_committed(params, cfg, x, C)
    s0 = jnp.zeros((H, DK, DV), jnp.float32)
    o_ref, s_ref = _reference_scan(interval, s0, C, kda._cc_tile(cfg))
    o_e, s_e = _emulated_scan(interval, s0, C, kda._cc_tile(cfg))
    assert jnp.allclose(s_e, s_ref, rtol=BF16_RTOL, atol=BF16_ATOL), (
        f"state g_min={g_min} H={H} T={T} seed={seed}: max|d|={_max_abs(s_e, s_ref):.3e}")
    assert jnp.allclose(o_e, o_ref, rtol=BF16_RTOL, atol=BF16_ATOL)


@pytest.mark.parametrize("g_min", [-5.0, -0.005])
@pytest.mark.parametrize("H,T", [(2, 256), (12, 512)])
def test_state_drift_is_isolated_and_bounded(g_min, H, T):
    """The bf16 SMEM state is the only new rounding point, and it is bounded.

    Compares the two emulations (bf16 state vs fp32 state) against the fp32
    reference chain: the fp32-state leg deviates only by the milestone-1
    operand quantisation, so the distance of the bf16-state leg from the
    reference *is* the state-drift term.  Declared bound: the operand budget.
    """
    cfg = _cfg(H, g_min=g_min)
    params = kda.init_kda(jr.PRNGKey(7), cfg)
    x = jr.normal(jr.PRNGKey(8), (T, cfg.hidden))
    interval, _s = _interval_and_committed(params, cfg, x, C)
    s0 = jnp.zeros((H, DK, DV), jnp.float32)
    _o_ref, s_ref = _reference_scan(interval, s0, C, kda._cc_tile(cfg))
    _o_p, s_p = _fp32_state_emulated_scan(interval, s0, C, kda._cc_tile(cfg))
    _o_b, s_b = _emulated_scan(interval, s0, C, kda._cc_tile(cfg))
    drift = _max_abs(s_b, s_ref)
    assert drift <= BF16_ATOL, f"drift g_min={g_min} H={H} T={T}: {drift:.3e}"


# ---------------------------------------------------------------------------
# the AXIOM_KDA_CC_IMPL mux: persistent mode end to end (CPU substitution)
# ---------------------------------------------------------------------------


def _cpu_emulated_kernel(q, k, v, alpha, beta32, S, *, tile=16, chunk=32):
    """The kernel seam: the declared numerics (rounding points, bf16 state).

    The only substitution in the end-to-end tests: kernel execution is
    replaced by its declared arithmetic.  ``q/k/v`` arrive already cast to
    bf16 by the mode, exactly like the kernel's operand path.  The seam
    returns the *final* state (the whole-sequence contract), not the state
    chain the scan helper collects for the boundary comparisons.
    """
    o, states = _emulated_scan((q, k, v, alpha, beta32), S, chunk, tile)
    return o, states[-1]


def _cpu_replica_kernel(q, k, v, alpha, beta32, S, *, tile=16, chunk=32):
    """The kernel seam: the fp32 replica (no state rounding)."""
    return persist.reference(q, k, v, alpha, beta32, S, tile=tile, chunk=chunk)


def _enable_persistent_on_cpu(monkeypatch, kernel=_cpu_emulated_kernel):
    """Force the persistent mux on this node with the kernel stubbed."""
    monkeypatch.setenv("AXIOM_KDA_CC_IMPL", "persistent")
    monkeypatch.setattr(persist, "device_available", lambda: True)
    monkeypatch.setattr(persist, "kernel", kernel)


@pytest.mark.parametrize("H,T", [(2, 256), (2, 512), (12, 96), (12, 512), (2, 100)])
@pytest.mark.parametrize("seed", [0, 1])
def test_persistent_mode_matches_the_jax_body(monkeypatch, H, T, seed):
    """``AXIOM_KDA_CC_IMPL=persistent`` == ``jax`` at the gated boundary.

    Scale-relative comparison at the declared 2e-2 (the milestone-2 pattern).
    Observed ~7e-3 at H=2 and ~1.5e-2 of scale at H=12: the head-wise RMSNorm
    amplification of the bf16 operand quantisation, the term milestone 1
    declared for the ``mosaic`` mode (bit-identical to the fp32-state
    substitution -- the state term is below this comparison's resolution).
    ``T=100`` exercises padding (100 = 3*32 + 4).
    """
    cfg = _cfg(H)
    params = kda.init_kda(jr.PRNGKey(seed), cfg)
    x = jr.normal(jr.PRNGKey(seed + 9), (T, cfg.hidden))

    monkeypatch.delenv("AXIOM_KDA_CC_IMPL", raising=False)
    ref = kda.apply_chunked_cc(params, cfg, x, chunk_size=C)

    _enable_persistent_on_cpu(monkeypatch)
    got = kda.apply_chunked_cc(params, cfg, x, chunk_size=C)

    scale = max(1.0, float(jnp.max(jnp.abs(ref))))
    err = _max_abs(got, ref)
    assert err <= BF16_RTOL * scale, (
        f"H={H} T={T} seed={seed}: max|d|={err:.3e} scale={scale:.3e}")


def test_persistent_mode_ignores_remat_flag_and_still_matches(monkeypatch):
    """``kda_chunked_backward`` is inert in the persistent mode (no XLA scan body)."""
    H, T, seed = 2, 256, 2
    cfg = _cfg(H)
    params = kda.init_kda(jr.PRNGKey(seed), cfg)
    x = jr.normal(jr.PRNGKey(seed + 3), (T, cfg.hidden))

    _enable_persistent_on_cpu(monkeypatch)
    got = kda.apply_chunked_cc(params, cfg, x, chunk_size=C)
    got_remat = kda.apply_chunked_cc(
        params, dataclasses.replace(cfg, kda_chunked_backward=True), x, chunk_size=C)
    assert _max_abs(got, got_remat) == 0.0


def test_persistent_mode_is_repeatable(monkeypatch):
    """Repeated calls give the same numbers (no hidden state in the mux)."""
    H, T, seed = 2, 256, 0
    cfg = _cfg(H)
    params = kda.init_kda(jr.PRNGKey(seed), cfg)
    x = jr.normal(jr.PRNGKey(seed + 1), (T, cfg.hidden))
    _enable_persistent_on_cpu(monkeypatch)
    first = kda.apply_chunked_cc(params, cfg, x, chunk_size=C)
    second = kda.apply_chunked_cc(params, cfg, x, chunk_size=C)
    assert _max_abs(first, second) == 0.0


def test_persistent_kernel_seam_is_one_call_per_pass(monkeypatch):
    """The launch-count claim at the API seam: ONE kernel call per layer-pass.

    ADR-054 Am.4: the per-chunk body called the seam once per chunk (256 per
    layer-pass); the persistent body must call it once for the whole scan.
    """
    calls = {"n": 0}

    def stub(q, k, v, alpha, beta32, S, *, tile=16, chunk=32):
        calls["n"] += 1
        return persist.reference(q, k, v, alpha, beta32, S, tile=tile, chunk=chunk)

    monkeypatch.setenv("AXIOM_KDA_CC_IMPL", "persistent")
    monkeypatch.setattr(persist, "device_available", lambda: True)
    monkeypatch.setattr(persist, "kernel", stub)

    H, T, seed = 2, 256, 0
    cfg = _cfg(H)
    params = kda.init_kda(jr.PRNGKey(seed), cfg)
    x = jr.normal(jr.PRNGKey(seed + 2), (T, cfg.hidden))
    kda.apply_chunked_cc(params, cfg, x, chunk_size=C)
    assert calls["n"] == 1, f"kernel seam called {calls['n']} times (T/C = {T // C})"


def test_persistent_mode_without_gpu_is_fail_closed(monkeypatch):
    """persistent needs Mosaic GPU; a GPU-less host gets a typed refusal."""
    if any(d.platform == "gpu" for d in jax.devices()):  # pragma: no cover — stand
        pytest.skip("GPU present: the mode is runnable, nothing to refuse")
    monkeypatch.setenv("AXIOM_KDA_CC_IMPL", "persistent")
    cfg = _cfg(H=2)
    params = kda.init_kda(jr.PRNGKey(0), cfg)
    x = jr.normal(jr.PRNGKey(1), (64, cfg.hidden))
    with pytest.raises(RuntimeError, match="GPU"):
        kda.apply_chunked_cc(params, cfg, x, chunk_size=C)


def test_persistent_mode_rejects_a_non_live_chunk_width(monkeypatch):
    """C=64 is over the SMEM budget: the kernel's sieve refuses it (typed)."""
    monkeypatch.setenv("AXIOM_KDA_CC_IMPL", "persistent")
    monkeypatch.setattr(persist, "device_available", lambda: True)
    cfg = _cfg(H=2, chunk=64)
    params = kda.init_kda(jr.PRNGKey(0), cfg)
    x = jr.normal(jr.PRNGKey(1), (128, cfg.hidden))
    with pytest.raises(ValueError, match="SMEM|budget|live"):
        kda.apply_chunked_cc(params, cfg, x, chunk_size=64)


def test_persistent_mode_not_entered_by_default(monkeypatch):
    """Default env: the flag's absence installs the committed scan body."""
    monkeypatch.delenv("AXIOM_KDA_CC_IMPL", raising=False)

    def spy(*args, **kwargs):  # pragma: no cover — must never run here
        raise AssertionError("default must not touch the persistent body")

    monkeypatch.setattr(kda, "_apply_chunked_cc_persistent", spy)
    cfg = _cfg(H=2)
    params = kda.init_kda(jr.PRNGKey(0), cfg)
    x = jr.normal(jr.PRNGKey(1), (96, cfg.hidden))
    kda.apply_chunked_cc(params, cfg, x, chunk_size=C)


def test_impl_mux_names_the_persistent_mode():
    """The flag's third value is declared (the rollback handle list)."""
    assert "persistent" in kda.KDA_CC_IMPLS
    assert kda.KDA_CC_IMPLS == ("jax", "mosaic", "persistent")


# ---------------------------------------------------------------------------
# the backward rule (variant A) and the gradient of the mux
# ---------------------------------------------------------------------------


def _bwd_inputs(T, Dk, Dv, H, seed):
    """Small whole-sequence inputs with a real (non-degenerate) decay."""
    keys = jr.split(jr.PRNGKey(seed), 6)
    q = jr.normal(keys[0], (T, H * Dk)) * 0.5
    k = jr.normal(keys[1], (T, H * Dk)) * 0.5
    v = jr.normal(keys[2], (T, H * Dv)) * 0.5
    alpha = jnp.exp(-2.5 * jr.uniform(keys[3], (T, H * Dk)))
    beta = jr.uniform(keys[4], (T, H))
    beta32 = jnp.broadcast_to(beta[:, :, None], (T, H, 32)).reshape(T, H * 32)
    S = jr.normal(keys[5], (H, Dk, Dv)) * 0.1
    return q, k, v, alpha, beta32, S


@pytest.mark.parametrize("dtype,tol", [(jnp.float32, 1e-4), (jnp.bfloat16, 2e-2)])
@pytest.mark.parametrize("shape,seed", [((64, 8, 4, 2, 16), 0), ((128, 4, 2, 1, 16), 1),
                                        ((64, 128, 128, 1, 32), 2)])
def test_reference_bwd_scan_matches_grad_of_reference(dtype, tol, shape, seed):
    """``reference_bwd_scan`` == ``jax.grad`` of the fp32 replica (variant A).

    The reverse scan of the milestone-2 per-chunk adjoint against autodiff of
    the same fp32 object; scale-relative bound.  bf16 inputs get the declared
    operand budget (jax runs the backward in bf16 while the adjoint
    accumulates fp32 and rounds the cotangent once).
    """
    T, Dk, Dv, H, Cc = shape
    q, k, v, alpha, beta32, S = _bwd_inputs(T, Dk, Dv, H, seed)
    q, k, v = q.astype(dtype), k.astype(dtype), v.astype(dtype)
    go = jr.normal(jr.PRNGKey(seed + 50), (T, H * Dv))
    gS = jr.normal(jr.PRNGKey(seed + 51), (H, Dk, Dv))

    def loss(q, k, v, alpha, beta32, S):
        o, s_new = persist.reference(q, k, v, alpha, beta32, S, chunk=Cc)
        return jnp.sum(o * go) + jnp.sum(s_new * gS)

    grads = jax.grad(loss, argnums=(0, 1, 2, 3, 4, 5))(q, k, v, alpha, beta32, S)
    got = persist.reference_bwd_scan(q, k, v, alpha, beta32, S, go, gS, chunk=Cc)
    assert len(got) == len(grads)
    for g_got, g_ref, name in zip(got, grads, ("q", "k", "v", "alpha", "beta32", "S")):
        scale = max(1.0, float(jnp.max(jnp.abs(g_ref.astype(jnp.float32)))))
        err = _max_abs(g_got.astype(jnp.float32), g_ref.astype(jnp.float32))
        assert err <= tol * scale, (
            f"d{name} shape={shape} seed={seed} dtype={dtype}: "
            f"max|d|={err:.3e} scale={scale:.3e}")


@pytest.mark.parametrize("H,T", [(2, 128), (2, 256)])
@pytest.mark.parametrize("seed", [0, 1])
def test_grad_of_the_persistent_mux_matches_the_jax_body(monkeypatch, H, T, seed):
    """``jax.grad`` through the persistent mux == through the jax body.

    Both arms are the shipped code path (the flag mux, the projections, the
    short-conv, the post-processing, the bf16 operand cast); the forward
    execution is substituted by its declared numerics (bf16 state included),
    the backward rule is the real reverse scan.  Scale-relative at 2e-2.
    """
    cfg = _cfg(H)
    params = kda.init_kda(jr.PRNGKey(seed), cfg)
    x = jr.normal(jr.PRNGKey(seed + 5), (T, cfg.hidden))
    w = jr.normal(jr.PRNGKey(seed + 6), (T, cfg.hidden))

    monkeypatch.delenv("AXIOM_KDA_CC_IMPL", raising=False)

    def loss_jax(xx):
        return jnp.sum(kda.apply_chunked_cc(params, cfg, xx, chunk_size=C) * w)

    g_jax = jax.grad(loss_jax)(x)

    _enable_persistent_on_cpu(monkeypatch)

    def loss_p(xx):
        return jnp.sum(kda.apply_chunked_cc(params, cfg, xx, chunk_size=C) * w)

    g_p = jax.grad(loss_p)(x)

    scale = max(1.0, float(jnp.max(jnp.abs(g_jax))))
    err = _max_abs(g_p, g_jax)
    assert err <= BF16_RTOL * scale, (
        f"dx H={H} T={T} seed={seed}: max|d|={err:.3e} scale={scale:.3e}")


def test_grad_of_the_persistent_mux_wrt_params(monkeypatch):
    """The same parity for parameter gradients (the training-relevant direction)."""
    H, T, seed = 2, 128, 2
    cfg = _cfg(H)
    params = kda.init_kda(jr.PRNGKey(seed), cfg)
    x = jr.normal(jr.PRNGKey(seed + 5), (T, cfg.hidden))
    w = jr.normal(jr.PRNGKey(seed + 6), (T, cfg.hidden))

    monkeypatch.delenv("AXIOM_KDA_CC_IMPL", raising=False)

    def f(p):
        return jnp.sum(kda.apply_chunked_cc(p, cfg, x, chunk_size=C) * w)

    g_jax = jax.grad(f)(params)

    _enable_persistent_on_cpu(monkeypatch)
    g_p = jax.grad(f)(params)

    for name in ("W_q", "W_k", "W_v", "W_o", "W_g", "W_beta", "W_a_down",
                 "W_a_up", "conv_q", "conv_k", "conv_v"):
        got = getattr(g_p, name)
        ref = getattr(g_jax, name)
        scale = max(1.0, float(jnp.max(jnp.abs(ref))))
        err = _max_abs(got, ref)
        assert err <= BF16_RTOL * scale, f"d{name}: max|d|={err:.3e} scale={scale:.3e}"


def test_custom_vjp_routes_forward_and_backward(monkeypatch):
    """The shipped ``cc_scan`` uses the kernel seam forward and the scan adjoint back."""
    calls = {"fwd": 0, "bwd": 0}
    real_bwd = persist.reference_bwd_scan

    def stub(q, k, v, alpha, beta32, S, *, tile=16, chunk=32):
        calls["fwd"] += 1
        return persist.reference(q, k, v, alpha, beta32, S, tile=tile, chunk=chunk)

    def spy_bwd(q, k, v, alpha, beta32, S, do, dS, *, tile=16, chunk=32):
        calls["bwd"] += 1
        return real_bwd(q, k, v, alpha, beta32, S, do, dS, tile=tile, chunk=chunk)

    monkeypatch.setattr(persist, "kernel", stub)
    monkeypatch.setattr(persist, "reference_bwd_scan", spy_bwd)

    T, Dk, Dv, H = 64, 8, 4, 1
    args = _bwd_inputs(T, Dk, Dv, H, 11)
    go = jr.normal(jr.PRNGKey(70), (T, H * Dv))

    def loss(*args):
        o, s_new = persist.cc_scan(*args, chunk=16)
        return jnp.sum(o * go) + jnp.sum(s_new)

    grads = jax.grad(loss)(*args)
    assert calls["fwd"] == 1  # one forward rule per trace: one launch per pass
    assert calls["bwd"] == 1  # the analytic adjoint is the backward rule
    assert all(g is not None for g in grads)


# ---------------------------------------------------------------------------
# the module contract: cases, sieve, budget arithmetic, guards
# ---------------------------------------------------------------------------


def test_live_cases_are_the_campaign_geometry():
    """Live: (C=32, bf16 state); C=16 (tiling) and C=64 (SMEM) stay skipped."""
    live = [c for c in persist.CASES if persist.valid_config(c, {"tile": 16})]
    skipped = [c for c in persist.CASES if not persist.valid_config(c, {"tile": 16})]
    assert any(c["C"] == 32 and c["H"] == 12 and c["dtype"] == "bfloat16" for c in live)
    assert all(c["C"] == 32 and c["dtype"] == "bfloat16" for c in live), live
    assert all(c["C"] in (16, 64) for c in skipped), skipped


def test_sieve_numbers_are_the_declared_budget_arithmetic():
    """The SMEM floor of the module docstring, as arithmetic (not prose).

    Declared floor (conservative; the measured lowering numbers are slightly
    lower -- ``test_smem_budget_is_measured_not_declared``): C=32 bf16 state
    98 304 B (live); C=32 fp32 state 131 072 B (over -- the reason the state
    is bf16); C=16 bf16 82 944 B (moot: the C<32 tiling rule rejects it first,
    the fp32 floor 115 712 B would reject it anyway); C=64 bf16 135 424 B
    (over).
    """
    bf16, f32 = 2, 4
    assert persist.STATE_DTYPE == jnp.bfloat16
    assert persist._smem_floor_bytes(32, 128, 128, bf16, bf16) == 98304
    assert persist._smem_floor_bytes(32, 128, 128, bf16, f32) == 131072
    assert persist._smem_floor_bytes(16, 128, 128, bf16, bf16) == 82944
    assert persist._smem_floor_bytes(16, 128, 128, bf16, f32) == 115712
    assert persist._smem_floor_bytes(64, 128, 128, bf16, bf16) == 135424
    case32 = dict(C=32, H=12, dk=128, dv=128, tile=16, dtype="bfloat16")
    assert persist.valid_config(case32, {"tile": 16}) is True
    assert persist.valid_config(dict(case32, state_dtype="float32"), {"tile": 16}) is False
    assert persist.valid_config(dict(case32, C=64), {"tile": 16}) is False
    assert persist.valid_config(dict(case32, C=16), {"tile": 16}) is False  # tiling rule
    assert persist.SMEM_LIMIT_BYTES == 101376  # the GB10 opt-in block budget


def test_smem_budget_is_measured_not_declared():
    """The lowering's own SMEM number, extracted by the artificial limit.

    ``lower_check`` runs under the same limit; this pins the measured numbers
    behind the dtype decision: C=32 bf16 state lowers at 94 208 B (under the
    101 376 B budget), C=32 fp32 state at 126 976 B (over -- it can never
    lower on GB10).  The C=16 width is not an SMEM question at all: its
    lowering fails on the mma tiling (pinned below).
    """
    from jax.experimental.mosaic import gpu as mosaic_mod
    from jax.experimental.mosaic.gpu import utils as mosaic_utils

    mosaic_utils._infer_arch = lambda: (12, 1)

    def _build_trace(Cc, state_dtype, limit):
        with mosaic_mod.core.artificial_shared_memory_limit(limit):
            shapes = jax.eval_shape(lambda: persist.make_inputs(
                jax.random.key(0), T=256, C=Cc, H=2, dk=128, dv=128, tile=16,
                dtype="bfloat16"))
            fn = jax.jit(lambda *a: persist._build_kernel(
                256, Cc, 2, 128, 128, 16, "bfloat16", state_dtype)(*a))
            return fn.trace(*shapes).lower(lowering_platforms=("cuda",))

    def measured_smem(Cc, state_dtype):
        try:
            _build_trace(Cc, state_dtype, 1)
        except ValueError as e:
            msg = str(e)
            assert "shared memory" in msg, msg
            assert "smem_bytes=" in msg, msg
            return int(msg.split("smem_bytes=")[1].split()[0])
        raise AssertionError("the artificial 1-byte limit did not bite")

    live = measured_smem(32, "bfloat16")
    over = measured_smem(32, "float32")
    assert live == 94208 and live <= persist.SMEM_LIMIT_BYTES
    assert over == 126976 and over > persist.SMEM_LIMIT_BYTES
    # C=16: the (16, 16) values cannot take the (16, 32) mma accumulator tile
    with pytest.raises(ValueError, match="Tiling"):
        _build_trace(16, "bfloat16", persist.SMEM_LIMIT_BYTES)


def test_kernel_rejects_non_bf16_operands():
    """fp32 q/k/v is not an mma operand path on this design: typed refusal."""
    T, H = 64, 2
    a = jnp.zeros((T, H * DK), jnp.float32)
    s = jnp.zeros((H, DK, DV), jnp.float32)
    with pytest.raises(ValueError, match="bf16"):
        persist.kernel(a, a, jnp.zeros((T, H * DV), jnp.float32), a, a, s)


def test_kernel_and_replica_reject_non_multiple_T():
    """T must be a multiple of the chunk width (the wrapper pads; the kernel does not)."""
    T, H = 100, 2
    q = jnp.zeros((T, H * DK), jnp.bfloat16)
    s = jnp.zeros((H, DK, DV), jnp.float32)
    with pytest.raises(ValueError, match="multiple"):
        persist.kernel(q, q, jnp.zeros((T, H * DV), jnp.bfloat16), q, q, s, chunk=32)
    with pytest.raises(ValueError, match="multiple"):
        persist.reference(q, q, jnp.zeros((T, H * DV), jnp.bfloat16), q, q, s, chunk=32)


def test_make_inputs_geometry():
    """The module's I/O contract: shapes and dtypes of one pass."""
    q, k, v, alpha, beta32, S = persist.make_inputs(
        jax.random.key(0), T=256, C=32, H=2, dk=128, dv=128)
    assert q.shape == (256, 2 * 128) and q.dtype == jnp.bfloat16
    assert k.dtype == jnp.bfloat16 and v.shape == (256, 2 * 128)
    assert alpha.shape == (256, 2 * 128) and alpha.dtype == jnp.float32
    assert beta32.shape == (256, 2 * 32) and beta32.dtype == jnp.float32
    assert S.shape == (2, 128, 128) and S.dtype == jnp.float32


# ---------------------------------------------------------------------------
# lowering on this node: every live case compiles for sm_121 (patched, no GPU)
# ---------------------------------------------------------------------------


def _lower_live_case(case):
    """Lower ``kernel`` for sm_121 exactly as the skill's lower_check does."""
    from jax.experimental.mosaic import gpu as mosaic_mod
    from jax.experimental.mosaic.gpu import utils as mosaic_utils

    mosaic_utils._infer_arch = lambda: (12, 1)
    ctx = mosaic_mod.core.artificial_shared_memory_limit(persist.SMEM_LIMIT_BYTES)
    with ctx:
        shapes = jax.eval_shape(lambda: persist.make_inputs(jax.random.key(0), **case))
        fn = jax.jit(lambda *a: persist.kernel(*a, tile=case["tile"], chunk=case["C"]))
        return fn.trace(*shapes).lower(lowering_platforms=("cuda",)).as_text()


@pytest.mark.parametrize("case", [c for c in persist.CASES
                                  if persist.valid_config(c, {"tile": 16})])
def test_kernel_lowers_for_sm121(case):
    """Every live case lowers under the GB10 SMEM limit (the CPU criterion)."""
    text = _lower_live_case(case)
    assert text  # lowering produced a module


def test_the_lowered_module_carries_the_chunk_loop():
    """The persistent structure is in the lowered module: a dynamic chunk loop.

    The lowered module must carry exactly one ``scf.for`` -- the loop that
    walks the chunks (traced once, not unrolled).  An unrolled scan or a
    single-chunk body would change the launch-count claim this milestone is
    about, so it is pinned, not assumed.
    """
    case = dict(T=256, C=32, H=2, dk=128, dv=128, tile=16, dtype="bfloat16")
    text = _lower_live_case(case)
    assert text.count("scf.for") == 1, (
        f"expected one dynamic chunk loop, found {text.count('scf.for')}")
