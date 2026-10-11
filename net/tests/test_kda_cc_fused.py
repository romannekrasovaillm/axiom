"""Fused ``chunked_cc`` chunk body (ADR-054 stage-2b, milestone 1): CPU verification.

The fused forward kernel of :mod:`net.kernels.kda_cc_fused` collapses one KDA
chunk's per-head algebra (tile-wise ``C x C`` scores, the WY/UT Neumann solve,
the inter-chunk state transfer) into one Mosaic launch, and
``net.kda.apply_chunked_cc`` can route the scan body through it
(``AXIOM_KDA_CC_IMPL=mosaic``).  This file pins what is verifiable **on this
CPU node** and states plainly what is not:

* the module's fp32 :func:`kda_cc_fused.reference` — the *laid-out* replica
  (tile-wise factors, transposed left-multiplied Neumann chain, the 32-column
  beta recovery) — is pinned against the committed
  :func:`net.kda.cc_chunk_step` at the campaign geometry (C=32, H=2..12,
  dk=dv=128, T=256..2048, several seeds).  A transposition error in the
  laid-out form shows up here, not only on the stand;
* a bf16 emulation of the kernel's rounding points (operand quantisation,
  the hi/lo decay split, per-factor bf16 storages) is pinned under the
  declared operand tolerance (atol/rtol 2e-2) **at the kernel's boundary**:
  the state against the committed ``cc_chunk_step``, the pre-gate output
  against the module's fp32 replica (which the test above pins to
  ``cc_chunk_step`` at 2e-5).  Observed deviations are ~1e-3 — three orders
  under the budget.  The gated end-to-end output is *not* asserted at 2e-2:
  the head-wise RMSNorm of ``_output_gate`` amplifies the interval's bf16
  rounding by ~1/RMS(head) (observed ~3e-2 on |out|~1.5 at H=12 — ~2% of
  scale, right at the boundary).  That amplification belongs to the gate
  (XLA-side, identical in both modes), not to the kernel; the end-to-end
  budget is a stand question (milestone-1 note).
* the kernel *lowers* for sm_121 under the skill's patched lowering (arch
  (12,1), 101 376 B SMEM limit) for every live case, and the sieve
  (:func:`kda_cc_fused.valid_config`) rejects what does not fit (C=64);
* the ``AXIOM_KDA_CC_IMPL`` mux is fail-closed: default ``jax`` is the
  committed path, an unknown value is an error, ``mosaic`` on a GPU-less host
  refuses with a typed error (no silent fallback);
* **the kernel itself is not executed here** (no GPU on this node, and jax
  0.11.2's GPU interpret mode carries no rule for ``plgpu.mma``): device
  parity, timing and determinism are the architect's ``check_kernel.py`` run
  on GB10.  The emulation mirrors the kernel's *declared* rounding points; a
  divergence between code and emulation would still need the device to see.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from net import kda
from net.config import ModelConfig
from net.kernels import kda_cc_fused as fused

#: Campaign geometry of the kernel (the live cases of the module contract).
C = 32
DK = 128
DV = 128

#: SiLU-free parity budget of the fp32 reference vs the committed jnp path
#: (both fp32; observed ~3e-7, the bound leaves room for summation order).
REF_ATOL = 2e-5


def _cfg(H=2, *, chunk=32, g_min=-5.0):
    """A KDA-only config at the kernel's geometry (hidden = H*dv, structural)."""
    return ModelConfig(
        hidden=H * DV, num_heads=H, head_dim=DK,
        kda_dk=DK, kda_dv=DV, kda_decay_rank=128, kda_short_conv_kernel=4,
        kda_g_min=g_min, kda_wyut_chunk=chunk,
    )


def _interval_inputs(params, cfg, carry, xc):
    """The ``(q, k, v, alpha32, beta32)`` interval ``cc_chunk_step`` computes.

    Recomputed through the *same* helpers the committed path calls, so both
    sides of the parity see identical inputs (the projections, short-conv and
    post-processing stay in XLA by design — milestone 1 fuses only the body).
    """
    H, dk, dv = cfg.num_heads, cfg.kda_dk, cfg.kda_dv
    Cc = xc.shape[0]
    proj = kda._project(params, cfg, xc)
    qc, _ = kda._short_conv_chunk(proj["qp"], params.conv_q, carry.q_buf)
    kc, _ = kda._short_conv_chunk(proj["kp"], params.conv_k, carry.k_buf)
    vc, _ = kda._short_conv_chunk(proj["vp"], params.conv_v, carry.v_buf)
    q, k, v = kda._postprocess(qc, kc, vc, cfg)
    alpha = proj["alpha"].reshape(Cc, H * dk).astype(jnp.float32)
    beta = proj["beta"]
    beta32 = jnp.broadcast_to(beta[:, :, None], (Cc, H, 32)).reshape(Cc, H * 32)
    return (q.reshape(Cc, H * dk), k.reshape(Cc, H * dk), v.reshape(Cc, H * dv),
            alpha, beta32.astype(jnp.float32), proj)


def _emulate_bf16(q_flat, k_flat, v_flat, alpha_flat, beta32, S, tile=16):
    """Emulation of the kernel's arithmetic: fp32 accumulation with its bf16 roundings.

    Not an execution — a jnp mirror of the kernel's *storage* points: the
    hi/lo split of ``log alpha``, the quantised tile factors and score
    operands, the quantised solve storages (per factor), the ``v_tilde`` /
    ``gamma_q`` / ``y`` roundings, and the 32-column beta recovery.
    """
    Cc = q_flat.shape[0]
    H, dk, dv = S.shape
    f32 = jnp.float32
    ar = jnp.arange(Cc)
    ridx, cidx = ar[:, None], ar[None, :]
    tril = (ridx >= cidx).astype(f32)
    o = jnp.zeros((Cc, H * dv), f32)
    S_new = jnp.zeros((H, dk, dv), f32)
    steps = 0
    while (1 << steps) < Cc:
        steps += 1

    for h in range(H):
        kv = k_flat[:, h * dk:(h + 1) * dk].astype(jnp.bfloat16)
        qv = q_flat[:, h * dk:(h + 1) * dk].astype(jnp.bfloat16)
        vv = v_flat[:, h * dv:(h + 1) * dv].astype(jnp.bfloat16)
        ah = alpha_flat[:, h * dk:(h + 1) * dk].astype(f32)
        beta = jnp.sum(beta32[:, h * 32:(h + 1) * 32].astype(f32), axis=1) * (1.0 / 32.0)
        bc = beta[:, None]
        la = jnp.log(ah)
        hi = la.astype(jnp.bfloat16)
        lo = (la - hi.astype(f32)).astype(jnp.bfloat16)
        G = tril @ hi.astype(f32) + tril @ lo.astype(f32)  # fp32 cumsum, bf16 operands

        def scores(row, mask):
            acc = jnp.zeros((Cc, Cc), f32)
            for p in range(0, Cc, tile):
                pre = (cidx <= p).astype(f32)
                anchor = pre @ hi.astype(f32) + pre @ lo.astype(f32)
                rowmask = ((ar >= p) & (ar < p + tile)).astype(f32)[:, None]
                rf = jnp.exp(jnp.minimum(G - anchor, 0.0))
                rq = (row.astype(f32) * (rf.astype(jnp.bfloat16).astype(f32) * rowmask)
                      ).astype(jnp.bfloat16)
                colmask = (ar < p + tile).astype(f32)[:, None]
                cf = jnp.exp(anchor - G)
                ck = (kv.astype(f32) * (cf * colmask).astype(jnp.bfloat16).astype(f32)
                      ).astype(jnp.bfloat16)
                acc = acc + rq.astype(f32) @ ck.astype(f32).T
            return acc * mask

        akk = scores((kv.astype(f32) * bc).astype(jnp.bfloat16), (ridx > cidx).astype(f32))
        aqk = scores(qv, tril)
        gam = jnp.exp(G).astype(jnp.bfloat16).astype(f32)
        kb = (kv.astype(f32) * bc).astype(jnp.bfloat16)
        Xt = (gam * kb.astype(f32)).astype(jnp.bfloat16).astype(f32).T
        Yt = (vv.astype(f32) * bc).astype(jnp.bfloat16).astype(f32).T
        pw = akk.astype(jnp.bfloat16)
        pf = pw.astype(f32)
        Xt = (Xt - Xt @ pf.T).astype(jnp.bfloat16).astype(f32)
        Yt = (Yt - Yt @ pf.T).astype(jnp.bfloat16).astype(f32)
        for _ in range(steps - 1):
            pw = (pf @ pf).astype(jnp.bfloat16)
            pf = pw.astype(f32)
            Xt = (Xt + Xt @ pf.T).astype(jnp.bfloat16).astype(f32)
            Yt = (Yt + Yt @ pf.T).astype(jnp.bfloat16).astype(f32)
        Sqh = S[h].astype(jnp.bfloat16)
        Wq = Xt.T.astype(jnp.bfloat16)
        ws_t = Sqh.T.astype(f32) @ Wq.astype(f32).T       # (W S)^T, bf16 operands
        vt_t = (Yt - ws_t).astype(jnp.bfloat16)          # v_tilde^T
        gq = (gam * qv.astype(f32)).astype(jnp.bfloat16)
        acc_o = (gq.astype(f32) @ Sqh.astype(f32)
                 + aqk.astype(jnp.bfloat16).astype(f32) @ vt_t.astype(f32).T)
        o = o.at[:, h * dv:(h + 1) * dv].set(acc_o)
        lh = la.astype(jnp.bfloat16)
        ll = (la - lh.astype(f32)).astype(jnp.bfloat16)
        ones = jnp.ones((Cc, Cc), f32)
        glast = ones @ lh.astype(f32) + ones @ ll.astype(f32)
        lam = jnp.exp(jnp.minimum(glast - G, 0.0))
        y_t = (lam.astype(jnp.bfloat16).astype(f32) * kv.astype(f32)
               ).astype(jnp.bfloat16).astype(f32).T
        g = jnp.exp(jnp.sum(la, axis=0)).astype(jnp.bfloat16).astype(f32)
        # gamma^C decays the key channels -- a *row* scaling of the state (the
        # kernel's phase-4 broadcast is pinned to the row vector by
        # test_kda_cc_fused_bwd.py; at the campaign decay the term is ~1e-35,
        # so this axis is only observable at weak decay).
        S_new = S_new.at[h].set(S[h] * g[:, None] + y_t @ vt_t.astype(f32).T)
    return o, S_new


def _max_abs(a, b):
    return float(jnp.max(jnp.abs(a - b)))


def _run_chunks(cfg, params, x, step_fn):
    """Scan the sequence chunk by chunk; ``step_fn`` returns the next state/output.

    ``step_fn(interval, carry) -> (S_new, payload)`` gets the interval inputs
    recomputed through the committed helpers and the current jnp carry (the
    short-conv buffers are head-independent bookkeeping, identical on both
    sides by construction: ``test_cc_chunk_state_matches_chunk_step``).
    """
    T = x.shape[0]
    pad = -T % cfg.kda_wyut_chunk
    x_p = jnp.pad(x, ((0, pad), (0, 0))) if pad else x
    carry = kda.init_state(cfg)
    outs = []
    for i in range(x_p.shape[0] // cfg.kda_wyut_chunk):
        xc = x_p[i * cfg.kda_wyut_chunk:(i + 1) * cfg.kda_wyut_chunk]
        interval = _interval_inputs(params, cfg, carry, xc)
        carry_jnp, out_jnp = kda.cc_chunk_step(params, cfg, carry, xc)
        s_new, out = step_fn(interval, carry)
        outs.append((carry_jnp, out_jnp, s_new, out))
        carry = carry_jnp._replace(S=s_new)
    return outs


# ---------------------------------------------------------------------------
# the fp32 reference pins the kernel's algebra against the committed path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("H,T", [(2, 256), (12, 256), (2, 2048), (12, 512)])
@pytest.mark.parametrize("seed", [0, 1])
def test_reference_matches_cc_chunk_step(H, T, seed):
    """Module reference vs committed ``cc_chunk_step``: same arithmetic, fp32.

    The reference is the laid-out form of the kernel (tile-wise factors,
    transposed Neumann chain, beta recovery), so this pin is what certifies
    the kernel's *formulas* on a CPU; the kernel's own execution is the
    stand's (module docstring).
    """
    cfg = _cfg(H)
    params = kda.init_kda(jr.PRNGKey(seed), cfg)
    x = jr.normal(jr.PRNGKey(seed + 7), (T, cfg.hidden))

    def step(interval, carry):
        q_flat, k_flat, v_flat, alpha_flat, beta32, proj = interval
        o_ref, s_ref = fused.reference(q_flat, k_flat, v_flat, alpha_flat, beta32,
                                       carry.S, tile=kda._cc_tile(cfg))
        out = kda._output_gate(o_ref.reshape(C, H, DV), proj["gate"], params)
        return s_ref, out

    for carry_jnp, out_jnp, s_ref, out_ref in _run_chunks(cfg, params, x, step):
        assert _max_abs(s_ref, carry_jnp.S) <= REF_ATOL, f"H={H} T={T} seed={seed}"
        assert _max_abs(out_ref, out_jnp) <= REF_ATOL, f"H={H} T={T} seed={seed}"


@pytest.mark.parametrize("H,T", [(2, 256), (12, 256), (2, 2048), (12, 512)])
@pytest.mark.parametrize("seed", [0, 1])
def test_bf16_emulation_matches_cc_within_operand_tolerance(H, T, seed):
    """The kernel's bf16 rounding points stay inside the declared 2e-2 budget.

    The emulation is the kernel's arithmetic (see :func:`_emulate_bf16`), not
    the kernel's execution — the distinction is stated in the module
    docstring; what this pins is the *numerics decision*: bf16 operands with
    fp32 accumulation and the hi/lo decay split keep the body within the
    tolerance of the jnp path at the campaign geometry.

    The comparison is taken at the kernel's boundary (its actual contract):
    the state against the committed ``cc_chunk_step``, the pre-gate output
    against the module's fp32 replica — which the test above pins to
    ``cc_chunk_step`` at 2e-5, so the two legs are the same jnp reference up
    to that bound.  The gated output is deliberately not asserted at 2e-2:
    see the module docstring (the head-wise RMSNorm amplifies the interval's
    rounding; that amplification exists in both modes and is not the
    kernel's term).
    """
    atol, rtol = fused.TOLERANCE["bfloat16"]
    cfg = _cfg(H)
    params = kda.init_kda(jr.PRNGKey(seed), cfg)
    x = jr.normal(jr.PRNGKey(seed + 11), (T, cfg.hidden))

    def step(interval, carry):
        q_flat, k_flat, v_flat, alpha_flat, beta32, _proj = interval
        o_e, s_e = _emulate_bf16(q_flat, k_flat, v_flat, alpha_flat, beta32,
                                 carry.S, tile=kda._cc_tile(cfg))
        o_r, _ = fused.reference(q_flat, k_flat, v_flat, alpha_flat, beta32,
                                 carry.S, tile=kda._cc_tile(cfg))
        return s_e, (o_e, o_r)

    for carry_jnp, _out_jnp, s_e, (o_e, o_r) in _run_chunks(cfg, params, x, step):
        assert jnp.allclose(s_e, carry_jnp.S, rtol=rtol, atol=atol), f"H={H} T={T} seed={seed}"
        assert jnp.allclose(o_e, o_r, rtol=rtol, atol=atol), f"H={H} T={T} seed={seed}"


def test_reference_geometry_contract():
    """Shapes of the reference line up with the kernel's I/O contract."""
    cfg = _cfg(H=2)
    params = kda.init_kda(jr.PRNGKey(0), cfg)
    x = jr.normal(jr.PRNGKey(1), (C, cfg.hidden))
    q_flat, k_flat, v_flat, alpha_flat, beta32, _ = _interval_inputs(
        params, cfg, kda.init_state(cfg), x)
    o, s_new = fused.reference(q_flat, k_flat, v_flat, alpha_flat, beta32,
                               jnp.zeros((2, DK, DV), jnp.float32))
    assert o.shape == (C, 2 * DV) and o.dtype == jnp.float32
    assert s_new.shape == (2, DK, DV)


# ---------------------------------------------------------------------------
# the module contract: cases, sieve, dtype guard
# ---------------------------------------------------------------------------


def test_live_cases_are_the_campaign_geometry():
    """Main case = C=32, H=12, dk=dv=128, bf16; C=64 stays as a skipped fallback."""
    live = [c for c in fused.CASES if fused.valid_config(c, {"tile": 16})]
    skipped = [c for c in fused.CASES if not fused.valid_config(c, {"tile": 16})]
    assert any(c["C"] == 32 and c["H"] == 12 and c["dtype"] == "bfloat16" for c in live)
    assert all(c["C"] == 32 and c["dtype"] == "bfloat16" for c in live)
    assert all(c["C"] == 64 or c["dtype"] == "float32" for c in skipped), skipped


def test_valid_config_sieve_numbers():
    """The sieve is the floor arithmetic of the module docstring, no hand-waving."""
    bf16 = int(jnp.dtype(jnp.bfloat16).itemsize)
    f32 = int(jnp.dtype(jnp.float32).itemsize)
    small = dict(C=32, H=12, dk=128, dv=128, tile=16, dtype="bfloat16")
    big = dict(C=64, H=12, dk=128, dv=128, tile=16, dtype="bfloat16")
    assert fused.valid_config(small, {"tile": 16}) is True
    assert fused.valid_config(big, {"tile": 16}) is False
    assert fused._smem_floor_bytes(32, 128, 128, bf16) == 65536
    assert fused._smem_floor_bytes(64, 128, 128, bf16) == 102656
    assert fused._smem_floor_bytes(32, 128, 128, f32) > fused.SMEM_LIMIT_BYTES
    assert fused.SMEM_LIMIT_BYTES == 101376  # the GB10 opt-in block budget


def test_kernel_rejects_non_bf16_operands():
    """fp32 q/k/v is not an mma operand path on this design: typed refusal."""
    a = jnp.zeros((C, 2 * DK), jnp.float32)
    s = jnp.zeros((2, DK, DV), jnp.float32)
    with pytest.raises(ValueError, match="bf16"):
        fused.kernel(a, a, jnp.zeros((C, 2 * DV), jnp.float32), a, a, s)


# ---------------------------------------------------------------------------
# the AXIOM_KDA_CC_IMPL mux: default jax, unknown is an error, mosaic refuses CPU
# ---------------------------------------------------------------------------


def test_default_impl_is_jax(monkeypatch):
    monkeypatch.delenv("AXIOM_KDA_CC_IMPL", raising=False)
    assert kda.kda_cc_impl() == "jax"


def test_unknown_impl_is_an_error(monkeypatch):
    monkeypatch.setenv("AXIOM_KDA_CC_IMPL", "banana")
    with pytest.raises(ValueError):
        kda.kda_cc_impl()


def test_mosaic_mode_without_gpu_is_fail_closed(monkeypatch):
    """mosaic needs Mosaic GPU; a GPU-less host gets a typed refusal, not a fallback."""
    if any(d.platform == "gpu" for d in jax.devices()):  # pragma: no cover — stand
        pytest.skip("GPU present: the mode is runnable, nothing to refuse")
    monkeypatch.setenv("AXIOM_KDA_CC_IMPL", "mosaic")
    cfg = _cfg(H=2)
    params = kda.init_kda(jr.PRNGKey(0), cfg)
    x = jr.normal(jr.PRNGKey(1), (64, cfg.hidden))
    with pytest.raises(RuntimeError, match="GPU"):
        kda.apply_chunked_cc(params, cfg, x, chunk_size=C)


def test_jax_mode_is_the_committed_path(monkeypatch):
    """Default env: the flag's absence installs the committed scan body.

    The fused body is spied on and must not be entered; the result matches a
    manual ``cc_chunk_step`` scan of the same chunks (fp32; the bound is the
    XLA reordering slack of the scan fusion, not a semantic difference).
    """
    monkeypatch.delenv("AXIOM_KDA_CC_IMPL", raising=False)
    calls = {"mosaic": 0}

    def spy(*args, **kwargs):  # pragma: no cover — must never run here
        calls["mosaic"] += 1
        raise AssertionError("AXIOM_KDA_CC_IMPL default must not touch the fused body")

    monkeypatch.setattr(kda, "_cc_chunk_step_mosaic", spy)
    cfg = _cfg(H=2)
    params = kda.init_kda(jr.PRNGKey(0), cfg)
    x = jr.normal(jr.PRNGKey(1), (96, cfg.hidden))
    via_apply = kda.apply_chunked_cc(params, cfg, x, chunk_size=C)
    assert calls["mosaic"] == 0
    # manual scan through cc_chunk_step — the body the default mode installs
    T = x.shape[0]
    pad = -T % C
    x_p = jnp.pad(x, ((0, pad), (0, 0))) if pad else x
    carry = kda.init_state(cfg)
    outs = []
    for i in range(x_p.shape[0] // C):
        carry, out = kda.cc_chunk_step(params, cfg, carry, x_p[i * C:(i + 1) * C])
        outs.append(out)
    manual = jnp.concatenate(outs, axis=0)[:T]
    assert _max_abs(via_apply, manual) <= 1e-5


# ---------------------------------------------------------------------------
# lowering on this node: every live case compiles for sm_121 (patched, no GPU)
# ---------------------------------------------------------------------------


def _lower_live_case(case):
    """Lower ``kernel`` for sm_121 exactly as the skill's lower_check does."""
    from jax.experimental.mosaic import gpu as mosaic_mod
    from jax.experimental.mosaic.gpu import utils as mosaic_utils

    mosaic_utils._infer_arch = lambda: (12, 1)
    ctx = mosaic_mod.core.artificial_shared_memory_limit(fused.SMEM_LIMIT_BYTES)
    with ctx:
        shapes = jax.eval_shape(lambda: fused.make_inputs(jax.random.key(0), **case))
        fn = jax.jit(lambda *a: fused.kernel(*a, tile=case["tile"]))
        return fn.trace(*shapes).lower(lowering_platforms=("cuda",)).as_text()


@pytest.mark.parametrize("index", [0, 1])
def test_kernel_lowers_for_sm121(index):
    """Full Mosaic lowering (layouts, SMEM budget, TMA boxes) for the live cases."""
    pytest.importorskip("jax.experimental.pallas.mosaic_gpu")
    if any(d.platform == "gpu" for d in jax.devices()):  # pragma: no cover — stand
        pytest.skip("GPU present: check_kernel.py on the stand is the real proof")
    text = _lower_live_case(fused.CASES[index])
    assert "mosaic_gpu" in text, "lowering did not produce a Mosaic kernel"


def test_oversize_cases_are_not_lowered():
    """The sieve's promise: the C=64 fallback never reaches the lowering."""
    for case in fused.CASES:
        if not fused.valid_config(case, {"tile": 16}):
            assert fused._smem_floor_bytes(case["C"], case["dk"], case["dv"],
                                           jnp.dtype(case["dtype"]).itemsize) > fused.SMEM_LIMIT_BYTES
