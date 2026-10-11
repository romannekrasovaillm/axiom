"""Fused forward-body kernel for the ``chunked_cc`` KDA chunk step (ADR-054, stage-2b, milestone 1).

Why one launch (the factorisation, ADR-054 Am.3)
------------------------------------------------
The step profile attributes ~29.5 s of the 33.5 s step to the submission cost
of ~100k kernel launches, and ~100k of the 102k launches belong to the KDA
scan family: every one of the 6912 chunk iterations (128 chunks x 18 layers x
3 passes -- ``evidence/mfu-55/donate-ab/KERNEL-FACTORIZATION.md``) issues
8-14 small kernels: elementwise micro-kernels (5.5 per iteration), per-head
kernels (12 = H), einsums, and the batched triangle solve that is the most
expensive KDA class on the stand (``grid=(12,1,1)``, 89 registers,
``evidence/mfu-55/env0112/REPORT-kernels-0112.md``).  None of them is
expensive on the GPU (5 % occupancy); each costs a driver submission.  The
single structural lever left by the measurement is therefore to collapse one
chunk's body into ONE launch -- this module.

Scope of milestone 1 (forward only; boundary documented honestly)
-----------------------------------------------------------------
The kernel covers the per-head algebra of ``net.kda.cc_chunk_step`` *inside*
the chunk:

* the tile-wise decay factors of :func:`net.kda._cc_scores` (tile <= 16, the
  ``_cc_tile`` budget rule), the two ``C x C`` score matrices,
* the WY/UT solve ``(I + L)^{-1} [xw | vw]`` in Neumann-product form,
* the inter-chunk state transfer and the chunk output ``o``.

``S`` is a GMEM input *and* output: successive chunk launches are tied by the
XLA scan/while loop, which is the declared milestone-1 structure (a persistent
kernel over all chunks is a later milestone).

The dense parameter projections (``_project``), the short-conv and swish/L2
post-processing, and the output gate (``_output_gate``: head-wise RMSNorm +
``W_o`` gemm) stay in XLA for this milestone.  Rationale: the output gate
sums over heads (a per-head program would need a cross-program reduction),
the projections and the gate are plain large GEMMs XLA already runs as a few
big kernels, and the short-conv is a positional shift that a one-program-per-
head kernel cannot express without a staging buffer that does not fit the
GB10 SMEM budget (99 KiB) next to the chunk algebra.  The kernel therefore
consumes the *post-processed* ``q``, ``k``, ``v`` of the existing XLA path and
returns ``o`` before the gate -- exactly the quantities ``cc_chunk_step``
computes between ``_postprocess`` and ``_output_gate``.  The reference below
reads them with the same formulas.

Milestone 2 (separate delta, ADR-054 stage-2b) is the backward pass:
``custom_vjp`` (analytic adjoint through a transposed kernel, as
``kda_ut_solve_mosaic`` does, or a jnp backward).  The forward kernel here
carries no autodiff by construction and must not be placed under
``jax.grad`` before that delta lands.

Verification status (honest)
----------------------------
* ``lower_check.py`` (skill ``pallas-gb10-kernel``) lowers every ``CASES``
  entry for sm_121 on this CPU node -- the declared CPU-side criterion;
* ``reference`` below is the fp32 jnp replica of the *same* algorithm
  (tile-wise scores, Neumann-product solve), pinned on CPU against
  ``net.kda.cc_chunk_step`` by ``net/tests/test_kda_cc_fused.py``;
* **the kernel itself has not been executed on a device** (node is CPU-only):
  numerical parity on GB10, timing and determinism are the architect's to
  measure (``check_kernel.py``), as the delta says.

Layout recipes used (each validated by lowering on this node; the mma/SMEM
core patterns come from the GB10-verified ``net/kernels/kda_ut_solve_mosaic``):

* ``plgpu.mma(acc, a, b)`` = ``acc + a @ b``: the LHS is loaded straight from
  a row-major SMEM buffer, the RHS from ``buf.T`` (the value then is the
  transposed view, ``(k, n)`` with ``k``-contiguous memory).
* SMEM is the only way to change a register layout: computed values are
  stored and re-loaded with ``Layout.MMA_LHS/RHS``, ``Layout.WGMMA`` or
  ``Layout.MMA_ACC``.
* row scalings of a ``(C, d)`` value are free only when the vector's layout
  is the value tiling reduced on that axis: ``reduce(1)`` -> explicit
  ``jax.lax.broadcast_in_dim(v, (C, d), (0,))``; column scalings use
  ``reduce(0)`` and ``(1,)``.  Implicit ``v[:, None]`` broadcasting is not
  inferred and is not used.
* iota masks are built in registers (WG-strided layout) and rounded through
  SMEM before meeting a tiled value of a different layout.
* transposes are materialised by storing a value into a transposed SMEM view
  (``buf.transpose((1, 0))[...] = value``); operands that need a transposed
  memory layout are produced this way.
* ``plgpu.store`` does not exist in 0.11.2: GMEM writes are ``ref[...] = v``
  over ``ref.at[pl.ds(...), pl.ds(...)]`` column slices; a GMEM region moved
  in one statement must hold at least 128 elements (transfer granularity).
* decay cumsums are accumulated in fp32 by a hi/lo bf16 split of ``log
  alpha`` (two mma's): bf16 rounding of ``log alpha`` alone would bias the
  cumsum by ~1e-2 per position, which the 2e-2 tolerance does not absorb.

Grid: one program per head (``grid=(H,)``, ``grid_names=("head",)``) -- the
structure the measured kernel classes point at.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Module contract (pallas-gb10-kernel skill)
# ---------------------------------------------------------------------------

#: Real campaign shapes: C=64, H=12, dk=dv=128, tile=16 (|g_min|=5 budget),
#: bf16 activations with fp32 accumulation, plus a short-width lowering case.
CASES = [
    dict(C=64, H=12, dk=128, dv=128, tile=16, dtype="bfloat16"),
    dict(C=64, H=2, dk=128, dv=128, tile=16, dtype="bfloat16"),
    dict(C=64, H=12, dk=128, dv=128, tile=16, dtype="float32"),
]

#: bf16-operand / fp32-accumulator path; the fp32 path is the exact reference.
TOLERANCE = {"bfloat16": (2e-2, 2e-2), "float32": (1e-5, 1e-5)}

#: Static knobs of :func:`kernel`.
TUNE_SPACE = dict(num_threads=[1, 2])


def valid_config(case, params):
    """Static sieve: MMA block rules and the SMEM floor of the persistent set."""
    C, H, dk, dv = (int(case[k]) for k in ("C", "H", "dk", "dv"))
    tile = int(case.get("tile", 16))
    if C % 64 or dk % 64 or dv % 16 or tile > 16 or tile < 1:
        return False
    if (C * dk) % 128 or (C * dv) % 128:  # transfer granularity of SMEM rows
        return False
    dt = jnp.dtype(case.get("dtype", "bfloat16"))
    if dt not in (jnp.bfloat16, jnp.float32):
        return False
    itemsize = dt.itemsize
    # q/k/hi/lo/ltr patterns + score matrices + the solve work buffers; the
    # phase-local scratch is scoped with pl.run_scoped and does not add up.
    persistent = itemsize * (
        3 * C * dk              # q_s, k_s, la_hi_t
        + C * dk                # la_lo_t
        + 2 * C * dk            # vw_t, op scratch
        + 2 * C * C             # aqk, L
        + C * C                 # power patterns
    )
    return 4 * 1024 + persistent <= 101376


def cost(C, H, dk, dv, tile, dtype="bfloat16", **kw):
    """Minimal FLOPs/bytes of one fused chunk body (roofline bookkeeping)."""
    ntile = -(-C // int(tile))
    flops = (
        2 * 2 * C * C * dk                       # aqk + akk (tile-rebuilt)
        + 2 * 2 * ntile * C * C * dk             # tile-wise rebuild (4 passes)
        + 2 * 2 * C * C * (dk + dv)              # solve RHS + power chain
        + 2 * C * (dk + dv) * dk                 # wS, inter
        + 2 * C * dv * C                         # intra
        + 2 * dk * dv * C                        # state update
    )
    b = jnp.dtype(dtype).itemsize
    bytes_ = (
        C * (2 * dk + dv) * b                    # q, k, v
        + C * dk * 4 * 2                         # alpha + hi/lo side buffer
        + C * 32 * 4                             # replicated beta
        + 2 * dk * dv * 4                        # S in + out
    )
    return dict(flops=flops, bytes=bytes_)


def make_inputs(key, C, H, dk, dv, tile=16, dtype="bfloat16", **kw):
    """Inputs of one chunk body: ``(q, k, v, alpha, beta32, S)``.

    * ``q``, ``k``: ``(C, H*dk)`` post-shortconv+swish+L2Norm projections,
    * ``v``: ``(C, H*dv)`` post-shortconv+swish projection,
    * ``alpha``: ``(C, H*dk)`` fp32 channel decay ``exp(g)`` in ``(0, 1]``,
    * ``beta32``: ``(C, H*32)`` fp32 write gate replicated over 32 columns
      (a replicated 32-wide block is the loadable shape of a per-row vector),
    * ``S``: ``(H, dk, dv)`` fp32 incoming state.
    """
    dt = jnp.dtype(dtype)
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    scale = 1.0 / (dk ** 0.25)
    q = (jax.random.normal(k1, (C, H * dk), jnp.float32) * scale).astype(dt)
    k = (jax.random.normal(k2, (C, H * dk), jnp.float32) * scale).astype(dt)
    v = (jax.random.normal(k3, (C, H * dv), jnp.float32) * scale).astype(dt)
    g = -0.5 * jax.random.uniform(k4, (C, H * dk), jnp.float32)  # log alpha
    alpha = jnp.exp(g)
    beta = jax.random.uniform(k5, (C, H), jnp.float32)
    beta32 = jnp.broadcast_to(beta[:, :, None], (C, H, 32)).reshape(C, H * 32)
    S = jax.random.normal(k6, (H, dk, dv), jnp.float32) * 0.1
    return q, k, v, alpha, beta32, S


# ---------------------------------------------------------------------------
# The kernel (one program per head)
# ---------------------------------------------------------------------------


def _build_kernel(C, H, dk, dv, tile, dtype, num_threads=1):
    """Build the Pallas Mosaic GPU kernel: one program per head."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import mosaic_gpu as plgpu

    dt = jnp.dtype(dtype)
    f32 = jnp.float32
    L_ACC = plgpu.Layout.MMA_ACC(dt)
    L_LHS = plgpu.Layout.MMA_LHS(dt)
    L_RHS = plgpu.Layout.MMA_RHS(dt)
    WG = plgpu.Layout.WGMMA

    def _l(ref, layout):
        return plgpu.load(ref, layout=layout, optimized=False)

    def _zero(shape):
        return plgpu.layout_cast(jnp.zeros(shape, f32), L_ACC)

    def _iota(shape, axis):
        return plgpu.layout_cast(
            plgpu.broadcasted_iota(jnp.int32, shape, axis),
            plgpu.Layout.WG_STRIDED(shape, 2))

    ridx = _iota((C, dk), 0)
    cidx = _iota((C, dk), 1)
    ridx2 = _iota((C, C), 0)
    cidx2 = _iota((C, C), 1)

    def _decay(la_hi_t, la_lo_t, pat_s):
        """log(alpha) -> hi/lo bf16 halves -> fp32 cumsum via the tril mma."""
        av = _l(ag.at[pl.ds(0, C), pl.ds(qo, dk)], WG).astype(f32)
        la = jnp.log(av)                                  # (C, dk) fp32
        la_hi = la.astype(dt)
        la_lo = (la - la_hi.astype(f32)).astype(dt)
        la_hi_t.transpose((1, 0))[...] = la_hi
        la_lo_t.transpose((1, 0))[...] = la_lo
        pat_s[...] = (ridx2 >= cidx2).astype(dt)          # tril ones
        acc = _zero((C, dk))
        acc = plgpu.mma(acc, _l(pat_s, L_LHS), _l(la_hi_t.T, L_RHS))
        acc = plgpu.mma(acc, _l(pat_s, L_LHS), _l(la_lo_t.T, L_RHS))
        return la, acc

    def _body(qg, kg, vg, ag, bg, sg, og, sog, k_s, aqk_s, lkk_s, lkk_t_s):
        h = jax.lax.axis_index("head")
        qo, vo, bo = h * dk, h * dv, h * 32
        qv = _l(qg.at[pl.ds(0, C), pl.ds(qo, dk)], WG).astype(f32)
        kv = _l(kg.at[pl.ds(0, C), pl.ds(qo, dk)], WG)
        k_s[...] = kv                                    # persistent k operand
        cv = _l(bg.at[pl.ds(0, C), pl.ds(bo, 32)], WG).astype(f32)
        betav = jnp.sum(cv, axis=1)                      # (C,) row vector
        beta_c = jax.lax.broadcast_in_dim(betav, (C, dk), (0,))
        beta_v = jax.lax.broadcast_in_dim(betav, (C, dv), (0,))
        return qv, kv, beta_c, beta_v

    def _anchor(p, pat_s, la_hi_t, la_lo_t):
        """All rows := reference row p of G, via the prefix pattern x la."""
        pat_s[...] = (cidx2 <= p).astype(dt)
        out = _zero((C, dk))
        out = plgpu.mma(out, _l(pat_s, L_LHS), _l(la_hi_t.T, L_RHS))
        return plgpu.mma(out, _l(pat_s, L_LHS), _l(la_lo_t.T, L_RHS))

    def _score_matrix(row_v, strict, G, kv, op_s, pat_s, la_hi_t, la_lo_t):
        """Tile-wise decay-weighted C x C scores (net.kda._cc_scores math)."""
        acc = _zero((C, C))
        for p in range(0, C, tile):
            anchor = _anchor(p, pat_s, la_hi_t, la_lo_t)
            op_s[...] = ((ridx >= p) & (ridx < p + tile)).astype(dt)
            rmk = _l(op_s, WG).astype(f32)
            rf = jnp.exp(jnp.minimum(G - anchor, 0.0))   # rows c in the tile
            op_s[...] = rf.astype(dt)
            rf_wg = _l(op_s, WG).astype(f32) * rmk
            op_s[...] = (row_v.astype(f32) * rf_wg).astype(dt)
            rq = _l(op_s, L_LHS)
            op_s[...] = (ridx < p + tile).astype(dt)
            cmk = _l(op_s, WG).astype(f32)
            cf = jnp.exp(anchor - G)                     # cols i < p + tile
            op_s[...] = (cf * cmk).astype(dt)
            cf_wg = _l(op_s, WG).astype(f32)
            op_s[...] = (kv.astype(f32) * cf_wg).astype(dt)
            ck = _l(op_s.T, L_RHS)
            acc = plgpu.mma(acc, rq, ck)
        pat_s[...] = ((ridx2 > cidx2) if strict else (ridx2 >= cidx2)).astype(dt)
        mk = _l(pat_s, L_ACC).astype(f32)                # structural mask
        return acc * mk

    def _solve(xw_t, vw_t, pw_s, pw_t, op2_s, kb, beta_v, G, vv):
        """(I + L)^{-1} [xw | vw] on the transposed right-hand sides."""
        pw_s[...] = lkk_s[...]                            # P := L (natural)
        pw_t[...] = lkk_t_s[...]                          # P^T (for the mma RHS)
        # xw = gamma * kb, vw = beta * v
        vw_t.transpose((1, 0))[...] = (vv.astype(f32) * beta_v).astype(dt)
        op2_s[...] = jnp.exp(G).astype(dt)                # gamma (<=1)
        gam_wg = _l(op2_s, WG).astype(f32)
        xw_t.transpose((1, 0))[...] = (gam_wg * kb.astype(f32)).astype(dt)
        # factors: (I - L), then (I + L^2), (I + L^4), ... (L is nilpotent)
        acc = _l(xw_t, L_ACC).astype(f32)
        acc = plgpu.mma(acc, _l(xw_t, L_LHS), -_l(pw_s.T, L_RHS))
        xw_t[...] = acc.astype(dt)
        acc = _l(vw_t, L_ACC).astype(f32)
        acc = plgpu.mma(acc, _l(vw_t, L_LHS), -_l(pw_s.T, L_RHS))
        vw_t[...] = acc.astype(dt)
        steps = 0
        while (1 << steps) < C:
            steps += 1
        for _ in range(steps - 1):
            p2 = plgpu.mma(_zero((C, C)), _l(pw_s, L_LHS), _l(pw_t, L_RHS))
            pw_s[...] = p2.astype(dt)
            pw_t.transpose((1, 0))[...] = p2.astype(dt)
            acc = _l(xw_t, L_ACC).astype(f32)
            acc = plgpu.mma(acc, _l(xw_t, L_LHS), _l(pw_s.T, L_RHS))
            xw_t[...] = acc.astype(dt)
            acc = _l(vw_t, L_ACC).astype(f32)
            acc = plgpu.mma(acc, _l(vw_t, L_LHS), _l(pw_s.T, L_RHS))
            vw_t[...] = acc.astype(dt)
        return _l(xw_t, L_ACC).astype(f32), _l(vw_t, L_ACC).astype(f32)

    def _body(qg, kg, vg, ag, bg, sg, og, sog, k_s, aqk_s, lkk_s, lkk_t_s):
        h = jax.lax.axis_index("head")
        qo, vo, bo = h * dk, h * dv, h * 32
        qv = _l(qg.at[pl.ds(0, C), pl.ds(qo, dk)], WG).astype(f32)
        kv = _l(kg.at[pl.ds(0, C), pl.ds(qo, dk)], WG)
        k_s[...] = kv
        cv = _l(bg.at[pl.ds(0, C), pl.ds(bo, 32)], WG).astype(f32)
        betav = jnp.sum(cv, axis=1)
        beta_c = jax.lax.broadcast_in_dim(betav, (C, dk), (0,))
        beta_v = jax.lax.broadcast_in_dim(betav, (C, dv), (0,))

        def _phase1(la_hi_t, la_lo_t, op_s, pat_s):
            la, G = _decay(la_hi_t, la_lo_t, pat_s)
            row_k = (kv.astype(f32) * beta_c).astype(dt)
            acc_kk = _score_matrix(row_k, True, G, kv, op_s, pat_s, la_hi_t, la_lo_t)
            lkk_s[...] = acc_kk.astype(dt)
            lkk_t_s.transpose((1, 0))[...] = acc_kk.astype(dt)
            acc_qk = _score_matrix(qv.astype(dt), False, G, kv, op_s, pat_s,
                                   la_hi_t, la_lo_t)
            aqk_s[...] = acc_qk.astype(dt)
            return G

        G = pl.run_scoped(
            _phase1,
            plgpu.SMEM((dk, C), dt), plgpu.SMEM((dk, C), dt),
            plgpu.SMEM((C, dk), dt), plgpu.SMEM((C, C), dt),
        )

        vv = _l(vg.at[pl.ds(0, C), pl.ds(vo, dv)], WG)
        kb = (kv.astype(f32) * beta_c).astype(dt)

        def _phase2(xw_t, vw_t, pw_s, pw_t, op2_s):
            return _solve(xw_t, vw_t, pw_s, pw_t, op2_s, kb, beta_v, G, vv)

        w_acc, u_acc = pl.run_scoped(
            _phase2,
            plgpu.SMEM((dk, C), dt), plgpu.SMEM((dv, C), dt),
            plgpu.SMEM((C, C), dt), plgpu.SMEM((C, C), dt),
            plgpu.SMEM((C, dk), dt),
        )

        def _phase3(w_s, u_s, s_t, vt_t, opb_s):
            # S^T in SMEM: the RHS of w S / gamma_q S needs the k-contiguous
            # transposed view.
            sv = _l(sg.at[h], WG)
            s_t.transpose((1, 0))[...] = sv.astype(dt)
            w_s[...] = w_acc.astype(dt)
            u_s[...] = u_acc.astype(dt)
            ws = plgpu.mma(_zero((C, dv)), _l(w_s, L_LHS), _l(s_t.T, L_RHS))
            vt = u_acc - ws
            vt_t.transpose((1, 0))[...] = vt.astype(dt)
            # o = gamma_q S + aqk vt   (gamma_q reuses the freed w_s buffer)
            opb_s[...] = jnp.exp(G).astype(dt)
            gq = (_l(opb_s, WG).astype(f32) * qv).astype(dt)
            w_s[...] = gq
            acc_o = plgpu.mma(_zero((C, dv)), _l(w_s, L_LHS), _l(s_t.T, L_RHS))
            acc_o = plgpu.mma(acc_o, _l(aqk_s, L_LHS), _l(vt_t.T, L_RHS))
            og.at[pl.ds(0, C), pl.ds(vo, dv)][...] = acc_o

        pl.run_scoped(
            _phase3,
            plgpu.SMEM((C, dk), dt), plgpu.SMEM((C, dv), dt),
            plgpu.SMEM((dv, dk), dt), plgpu.SMEM((dv, C), dt),
            plgpu.SMEM((C, dk), dt),
        )

        def _phase4(y_t, opb_s, gl_hi_t, gl_lo_t):
            la2 = jnp.log(_l(ag.at[pl.ds(0, C), pl.ds(qo, dk)], WG).astype(f32))
            lh = la2.astype(dt)
            ll = (la2 - lh.astype(f32)).astype(dt)
            gl_hi_t.transpose((1, 0))[...] = lh
            gl_lo_t.transpose((1, 0))[...] = ll
            opb_s.at[pl.ds(0, C), pl.ds(0, C)][...] = (ridx >= 0).astype(dt)
            glast = plgpu.mma(_zero((C, dk)), _l(opb_s.at[pl.ds(0, C), pl.ds(0, C)], L_LHS),
                              _l(gl_hi_t.T, L_RHS))
            glast = plgpu.mma(glast, _l(opb_s.at[pl.ds(0, C), pl.ds(0, C)], L_LHS),
                              _l(gl_lo_t.T, L_RHS))
            lam = jnp.exp(jnp.minimum(glast - G, 0.0))
            opb_s[...] = lam.astype(dt)
            yv = (_l(opb_s, WG).astype(f32) * kv.astype(f32)).astype(dt)
            y_t.transpose((1, 0))[...] = yv
            gc = jnp.exp(jnp.sum(la2, axis=0))                 # (dk,) column vector
            gc_b = jax.lax.broadcast_in_dim(gc, (dk, dv), (1,))
            sv = _l(sg.at[h], WG)
            acc_s = plgpu.layout_cast(sv * gc_b, L_ACC)
            s_out = plgpu.mma(acc_s, _l(y_t, L_LHS), _l(_lt_s_guard(), L_RHS))
            sog.at[h][...] = s_out

        return None
