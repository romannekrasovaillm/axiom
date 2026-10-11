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

Milestone 2 (this round): the backward pass.  ``make_cc_step`` /
:func:`cc_step` wrap the body in ``jax.custom_vjp``: the forward rule runs the
kernel, the backward rule runs the analytic jnp adjoint of
:mod:`net.kernels.kda_cc_fused_bwd` (variant B of the milestone-2 delta —
fwd and remat-fwd collapse into one launch per chunk, the backward pass runs as
fused XLA/jnp; the fused backward *kernel* is a scoped follow-up, its blocker
recorded in that module's docstring).  The bare :func:`kernel` remains
forward-only by construction (a ``pallas_call`` carries no JVP) and must not be
placed under ``jax.grad`` directly — the wrapper is the differentiable entry
point.

Campaign case and the SMEM fact that fixes it (round 2)
-------------------------------------------------------
The round-1 design was declared at the campaign width C=64 -- and does not
fit.  The persistent set of one program (q/k patterns, score matrices, the
solve work buffers, the cross-phase scratch) floors at 122 880 B (bf16) for
C=64/dk=dv=128, against the 101 376 B (99 KiB) SMEM budget of GB10: all CASES
were honestly rejected by :func:`valid_config` and nothing lowered.  The
design is arithmetically chunk-size agnostic (the jnp path ``chunked_cc`` is
pinned for every chunk width), so **the campaign case is C=32**:

====  ========  ============================  =====================
C     dtype     SMEM floor (persistent + max phase + 4 KiB)  vs 101 376 B
====  ========  ============================  =====================
64    bf16      102 656 B                   over by the declared epilogue margin
32    bf16      65 536 B                    fits, ~35 KiB margin
32    fp32      126 976 B                   over (and fp32 has no mma path)
64    fp32      201 216 B                   over (and fp32 has no mma path)
====  ========  ============================  =====================

``valid_config`` applies the floor so the oversize configurations stay in
``CASES`` as declared, *skipped* fallbacks rather than pretend-runnable ones.
The cost of C=32 is 256 chunk iterations per pass instead of 128 -- accepted:
the milestone's currency is *launches per iteration* (~4.6k launches/step
against ~90k), not chunks per pass.  The C=64 two-kernel split (intra:
scores+solve; inter: state+output) remains the documented alternative if the
stand ever needs C=64 (see ``docs`` note in the delta), not implemented here.

m_warps: M=C=32 MMA tiles lower **only** with ``Layout.MMA_*(dt, m_warps=1)``
(verified by lowering on the CPU node: the default m_warps=4 tiling
``((64, 16), ...)`` does not apply to a 32-row shape); the M=dk=128
accumulators keep ``m_warps=4`` (128x128 register tile = 128 f32/thread).
Both choices are applied per-MMA-site through :func:`_mma_m_warps`.

Verification status (honest)
----------------------------
* ``lower_check.py`` (skill ``pallas-gb10-kernel``) lowers every **live**
  ``CASES`` entry (``valid_config`` true) for sm_121 on this CPU node -- the
  declared CPU-side criterion; the oversize entries are reported as skipped;
* ``reference`` below is the fp32 jnp replica of the *same* algebra, written
  in the same transposed/left-multiplied dataflow as the kernel body (so a
  transposition error in the laid-out form shows up as a parity failure, not
  only on the stand), pinned on CPU against ``net.kda.cc_chunk_step`` by
  ``net/tests/test_kda_cc_fused.py`` (observed ~3e-7 at the campaign
  geometry, H=2..12, T=256..2048);
* the tests additionally pin a bf16 *emulation* of the kernel's rounding
  points (operand quantisation, hi/lo decay split, per-factor storages)
  against ``cc_chunk_step``/the replica at the kernel's boundary under the
  declared 2e-2 operand tolerance (observed ~1e-3).  The gated end-to-end
  output is not asserted at 2e-2: the head-wise RMSNorm of ``_output_gate``
  amplifies the interval's bf16 rounding by ~1/RMS(head) (~3e-2 on |out|~1.5
  at H=12, ~2% of scale) — a property of the gate, identical in both modes;
* **the kernel itself has not been executed on a device** (node is CPU-only,
  and jax 0.11.2's GPU interpret mode has no rule for ``plgpu.mma``):
  numerical parity of the *executed* kernel, timing and determinism are the
  architect's to measure on GB10 (``check_kernel.py``), as the delta says.
  The emulation mirrors the kernel's *declared* rounding points; only the
  stand can prove the code and the emulation agree.

Layout recipes used (each validated by lowering on this node; the mma/SMEM
core patterns come from the GB10-verified ``net/kernels/kda_ut_solve_mosaic``):

* ``plgpu.mma(acc, a, b)`` = ``acc + a @ b``: the LHS is loaded straight from
  a row-major SMEM buffer (the memory is ``k``-contiguous), the RHS from
  ``buf.T`` (``buf`` is ``(n, k)`` row-major, so the transposed view is
  ``(k, n)`` with ``k``-contiguous memory).
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
  over ``ref.at[pl.ds(...), pl.ds(...)]`` column slices.
* decay cumsums are accumulated in fp32 by a hi/lo bf16 split of ``log
  alpha`` (two mma's): bf16 rounding of ``log alpha`` alone would bias the
  cumsum by ~1e-2 per position, which the 2e-2 tolerance does not absorb.

Grid: one program per head (``grid=(H,)``, ``grid_names=("head",)``) -- the
structure the measured kernel classes point at.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Module contract (pallas-gb10-kernel skill)
# ---------------------------------------------------------------------------

#: Case shapes.  The two C=32 entries are the live campaign cases (H=12 is the
#: model, H=2 keeps the lowering/verification runs cheap on the CPU node); the
#: C=64 entries are the declared fallback, honestly *skipped* by
#: :func:`valid_config` (their SMEM floor is over the GB10 budget -- see the
#: module docstring).  ``valid_config`` is the arbiter, not this list.
CASES = [
    dict(C=32, H=12, dk=128, dv=128, tile=16, dtype="bfloat16"),
    dict(C=32, H=2, dk=128, dv=128, tile=16, dtype="bfloat16"),
    dict(C=64, H=12, dk=128, dv=128, tile=16, dtype="bfloat16"),  # oversize: skipped
    dict(C=64, H=12, dk=128, dv=128, tile=16, dtype="float32"),   # oversize: skipped
]

#: bf16-operand / fp32-accumulator path; the fp32 path is the exact reference.
TOLERANCE = {"bfloat16": (2e-2, 2e-2), "float32": (1e-5, 1e-5)}

#: Static knobs of :func:`kernel`.  ``tile`` is the live knob (the tile-wide
#: decay-factor budget of ``net.kda._cc_tile``); a second warpgoup
#: (``num_threads``) needs a ``thread_name`` axis and stays a milestone-2 knob.
TUNE_SPACE = dict(tile=[16])

#: GB10 SMEM budget of one block (99 KiB, the value the skill's lower_check
#: patches into the Mosaic lowering on a GPU-less node) minus the epilogue
#: staging margin (4 KiB, one ``(C, dk)`` f32 store at C=32).
SMEM_LIMIT_BYTES = 101376
SMEM_EPILOGUE_BYTES = 4 * 1024

#: The tile budget rule of ``net.kda._cc_tile``, mirrored as a static bound:
#: the column factor inside a tile is ``exp(|g_min| * (tile - 1))`` (<= e^80).
TILE_MAX = 16


def _smem_floor_bytes(C, dk, dv, itemsize):
    """Declared SMEM floor of one program (persistent + largest phase arena).

    The phases are ``pl.run_scoped`` arenas with disjoint lifetimes, so the
    floor is the persistent set plus the *largest* phase, not their sum
    (the liveness-pooling contract of ``pl.run_scoped``; checked by
    ``lower_check`` on every live case, not assumed silently).

    * persistent: ``aqk`` and ``L`` ``(C, C)`` cross-phase scratch plus the
      ``vt`` ``(dv, C)`` buffer that phase 3 writes and phase 4 reads;
    * phase 1: ``la_hi_t``/``la_lo_t`` ``(dk, C)`` + ``op`` ``(C, dk)`` +
      ``pat`` ``(C, C)``;
    * phase 2: ``xw_t`` ``(dk, C)`` + ``vw_t`` ``(dv, C)`` + ``P``/``P^T``
      ``(C, C)`` + ``op2`` ``(C, dk)``;
    * phase 3: ``W``/``gamma_q`` ``(C, dk)`` + ``S^T`` ``(dv, dk)``  --
      the binding phase, its ``dv*dk`` buffer dominates;
    * phase 4: ``y``/``gl_hi``/``gl_lo`` ``(dk, C)`` + ``op`` ``(C, dk)``.
    """
    el = (
        2 * C * C + dv * C                                                # persistent
    )
    phases = (
        3 * C * dk + C * C,                                               # phase 1
        2 * C * dk + dv * C + 2 * C * C,                                  # phase 2
        2 * C * dk + dk * dv,                                             # phase 3
        4 * C * dk + dk,                                                  # phase 4
    )
    return itemsize * (el + max(phases)) + SMEM_EPILOGUE_BYTES


def valid_config(case, params):
    """Static sieve: MMA tiling rules and the SMEM floor of the design.

    Returns False for anything the GB10 cannot run -- in particular the C=64
    fallback entries of :data:`CASES` (SMEM floor over :data:`SMEM_LIMIT_BYTES`;
    see the module docstring) -- so ``lower_check`` skips them instead of
    failing on them, which is the honest form of "does not fit".
    """
    C, H, dk, dv = (int(case[k]) for k in ("C", "H", "dk", "dv"))
    tile = int(params.get("tile", case.get("tile", TILE_MAX)))
    if C % 16 or C < 16 or dk % 64 or dv % 8 or dv > 128 or dk > 128 or tile > TILE_MAX or tile < 1:
        return False
    if (C * dk) % 128 or (C * dv) % 128:  # transfer granularity of the loads
        return False
    dt = jnp.dtype(case.get("dtype", "bfloat16"))
    if dt not in (jnp.bfloat16, jnp.float32):
        return False
    return _smem_floor_bytes(C, dk, dv, dt.itemsize) <= SMEM_LIMIT_BYTES


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


def device_available() -> bool:
    """True when a GPU platform device is visible to JAX.

    The kernel's *execution* needs Mosaic GPU; without a device JAX raises at
    compile time, so the callers that pick the impl (``net.kda``) refuse early
    with a typed error instead of letting the XLA/pallas error escape.  The
    *lowering* does not need a device (``lower_check`` runs exactly like that).
    """
    try:
        return any(d.platform == "gpu" for d in jax.devices())
    except Exception:  # noqa: BLE001 — a broken plugin must not read as "GPU present"
        return False


# ---------------------------------------------------------------------------
# The kernel (one program per head)
# ---------------------------------------------------------------------------


def _mma_m_warps(m: int) -> int:
    """Warp-group width of the MMA layouts for an accumulator with ``m`` rows.

    Lowering fact (this node, jax 0.11.2, sm_121): the default ``m_warps=4``
    tiling is a 64-row tile and does not apply to 32-row shapes
    (``Tiling ((64, 16), ...) does not apply to shape (32, ...)``); with
    ``m_warps=1`` the same shapes lower.  The 128-row accumulators of the
    state algebra keep the wide ``m_warps=4`` register tile.
    """
    return 1 if m < 64 else 4


def _build_kernel(C, H, dk, dv, tile, dtype):
    """Build the Pallas Mosaic GPU kernel: one program per head."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import mosaic_gpu as plgpu

    dt = jnp.dtype(dtype)
    f32 = jnp.float32
    def _l(ref, layout):
        return plgpu.load(ref, layout=layout, optimized=False)

    def _mma_layouts(m):
        mw = _mma_m_warps(m)
        return (
            plgpu.Layout.MMA_ACC(dt, m_warps=mw),
            plgpu.Layout.MMA_LHS(dt, m_warps=mw),
            plgpu.Layout.MMA_RHS(dt, m_warps=mw),
        )

    # Layout triples per accumulator row-count used in this body.  Non-MMA
    # values ride these ACC layouts too (WGMMA needs a 64-row tile, which
    # C=32 does not have), and a TiledLayout is what supports the single-axis
    # reductions of the row/column vectors (the rmsnorm recipe).
    ACC_C, L_C, R_C = _mma_layouts(C)          # (C, .) accumulators
    ACC_DK, L_DK, R_DK = _mma_layouts(dk)      # (dk, .) accumulators

    def _zero(shape, acc_layout):
        return plgpu.layout_cast(jnp.zeros(shape, f32), acc_layout)

    def _body(qg, kg, vg, ag, bg, sg, og, sog, aqk_s, lkk_s, vt_s):
        h = jax.lax.axis_index("head")
        qo, vo, bo = h * dk, h * dv, h * 32

        def _iota(shape, axis):
            return plgpu.layout_cast(
                plgpu.broadcasted_iota(jnp.int32, shape, axis),
                plgpu.Layout.WG_STRIDED(shape, 2))

        ridx = _iota((C, dk), 0)
        cidx = _iota((C, dk), 1)
        ridx2 = _iota((C, C), 0)
        cidx2 = _iota((C, C), 1)

        qv = _l(qg.at[pl.ds(0, C), pl.ds(qo, dk)], ACC_C).astype(f32)
        kv = _l(kg.at[pl.ds(0, C), pl.ds(qo, dk)], ACC_C)
        cv = _l(bg.at[pl.ds(0, C), pl.ds(bo, 32)], ACC_C).astype(f32)
        # beta32 replicates beta over the 32 columns (the loadable shape of
        # a per-row vector): the sum recovers 32 * beta, exactly (2**5).
        betav = jnp.sum(cv, axis=1) * (1.0 / 32.0)       # (C,) row vector
        # Broadcasts need an explicit target layout on this lowering (rmsnorm
        # recipe): the inferred candidates do not cover the C=32 tilings.
        beta_c = plgpu.layout_cast(jax.lax.broadcast_in_dim(betav, (C, dk), (0,)), ACC_C)
        beta_v = plgpu.layout_cast(jax.lax.broadcast_in_dim(betav, (C, dv), (0,)), ACC_C)

        def _decay(la_hi_t, la_lo_t, pat_s):
            """log(alpha) -> hi/lo bf16 halves -> fp32 cumsum via the tril mma."""
            av = _l(ag.at[pl.ds(0, C), pl.ds(qo, dk)], ACC_C).astype(f32)
            la = jnp.log(av)                                  # (C, dk) fp32
            la_hi = la.astype(dt)
            la_lo = (la - la_hi.astype(f32)).astype(dt)
            la_hi_t.transpose((1, 0))[...] = la_hi
            la_lo_t.transpose((1, 0))[...] = la_lo
            pat_s[...] = (ridx2 >= cidx2).astype(dt)          # tril ones
            acc = _zero((C, dk), ACC_C)
            acc = plgpu.mma(acc, _l(pat_s, L_C), _l(la_hi_t.T, R_C))
            acc = plgpu.mma(acc, _l(pat_s, L_C), _l(la_lo_t.T, R_C))
            return la, acc

        def _anchor(p, pat_s, la_hi_t, la_lo_t):
            """All rows := reference row p of G, via the prefix pattern x la."""
            pat_s[...] = (cidx2 <= p).astype(dt)
            out = _zero((C, dk), ACC_C)
            out = plgpu.mma(out, _l(pat_s, L_C), _l(la_hi_t.T, R_C))
            return plgpu.mma(out, _l(pat_s, L_C), _l(la_lo_t.T, R_C))

        def _score_matrix(row_v, strict, G, op_s, pat_s, la_hi_t, la_lo_t):
            """Tile-wise decay-weighted C x C scores (net.kda._cc_scores math)."""
            acc = _zero((C, C), ACC_C)
            for p in range(0, C, tile):
                anchor = _anchor(p, pat_s, la_hi_t, la_lo_t)
                op_s[...] = ((ridx >= p) & (ridx < p + tile)).astype(dt)
                rmk = _l(op_s, ACC_C).astype(f32)
                rf = jnp.exp(jnp.minimum(G - anchor, 0.0))   # rows c in the tile
                op_s[...] = rf.astype(dt)
                rf_wg = _l(op_s, ACC_C).astype(f32) * rmk
                op_s[...] = (row_v.astype(f32) * rf_wg).astype(dt)
                rq = _l(op_s, L_C)
                op_s[...] = (ridx < p + tile).astype(dt)
                cmk = _l(op_s, ACC_C).astype(f32)
                cf = jnp.exp(anchor - G)                     # cols i < p + tile
                op_s[...] = (cf * cmk).astype(dt)
                cf_wg = _l(op_s, ACC_C).astype(f32)
                op_s[...] = (kv.astype(f32) * cf_wg).astype(dt)
                ck = _l(op_s.T, R_C)
                acc = plgpu.mma(acc, rq, ck)
            pat_s[...] = ((ridx2 > cidx2) if strict else (ridx2 >= cidx2)).astype(dt)
            mk = _l(pat_s, ACC_C).astype(f32)                # structural mask
            return acc * mk

        def _phase1(la_hi_t, la_lo_t, op_s, pat_s):
            la, G = _decay(la_hi_t, la_lo_t, pat_s)
            row_k = (kv.astype(f32) * beta_c).astype(dt)
            acc_kk = _score_matrix(row_k, True, G, op_s, pat_s, la_hi_t, la_lo_t)
            lkk_s[...] = acc_kk.astype(dt)
            acc_qk = _score_matrix(qv.astype(dt), False, G, op_s, pat_s,
                                   la_hi_t, la_lo_t)
            aqk_s[...] = acc_qk.astype(dt)
            return G

        G = pl.run_scoped(
            _phase1,
            plgpu.SMEM((dk, C), dt), plgpu.SMEM((dk, C), dt),
            plgpu.SMEM((C, dk), dt), plgpu.SMEM((C, C), dt),
        )

        vv = _l(vg.at[pl.ds(0, C), pl.ds(vo, dv)], ACC_C)
        kb = (kv.astype(f32) * beta_c).astype(dt)

        def _solve(xw_t, vw_t, pw_s, pw_t, op2_s):
            """(I + L)^{-1} [xw | vw], left-multiplied on the transposed RHS.

            The accumulated value is ``X^T`` (memory ``(d, C)``): each factor
            is an MMA ``X^T + X^T @ (P^{2^k})^T`` = ``[X (I + ...)]^T`` with
            the alternating series ``(I - L)(I + L^2)(I + L^4)... = (I+L)^{-1}``
            (L nilpotent: the chain is cut at ``ceil(log2(C))`` steps).
            """
            # P := L and P^T, both materialised from one register read of the
            # phase-1 scratch (a direct SMEM->SMEM transposed copy does not
            # vectorise: "'vector.load' op most minor memref dim must have unit
            # stride").
            lv = _l(lkk_s, ACC_C)
            pw_s[...] = lv.astype(dt)
            pw_t.transpose((1, 0))[...] = lv.astype(dt)
            # xw = gamma * beta * k, vw = beta * v
            vw_t.transpose((1, 0))[...] = (vv.astype(f32) * beta_v).astype(dt)
            op2_s[...] = jnp.exp(G).astype(dt)              # gamma (<= 1)
            gam_wg = _l(op2_s, ACC_C).astype(f32)
            xw_t.transpose((1, 0))[...] = (gam_wg * kb.astype(f32)).astype(dt)
            acc = _l(xw_t, ACC_DK).astype(f32)
            acc = plgpu.mma(acc, _l(xw_t, L_DK), -_l(pw_s.T, R_DK))
            xw_t[...] = acc.astype(dt)
            acc = _l(vw_t, ACC_DK).astype(f32)
            acc = plgpu.mma(acc, _l(vw_t, L_DK), -_l(pw_s.T, R_DK))
            vw_t[...] = acc.astype(dt)
            steps = 0
            while (1 << steps) < C:
                steps += 1
            for _ in range(steps - 1):
                p2 = plgpu.mma(_zero((C, C), ACC_C), _l(pw_s, L_C), _l(pw_t.T, R_C))
                pw_s[...] = p2.astype(dt)
                # The squaring above reads *both* buffers (as A and as B), so
                # the transposed copy has to follow the natural one: without
                # this write the second iteration would compute P^2 @ P = P^3.
                pw_t.transpose((1, 0))[...] = p2.astype(dt)
                acc = _l(xw_t, ACC_DK).astype(f32)
                acc = plgpu.mma(acc, _l(xw_t, L_DK), _l(pw_s.T, R_DK))
                xw_t[...] = acc.astype(dt)
                acc = _l(vw_t, ACC_DK).astype(f32)
                acc = plgpu.mma(acc, _l(vw_t, L_DK), _l(pw_s.T, R_DK))
                vw_t[...] = acc.astype(dt)
            return _l(xw_t, ACC_DK).astype(f32), _l(vw_t, ACC_DK).astype(f32)

        def _phase2(xw_t, vw_t, pw_s, pw_t, op2_s):
            return _solve(xw_t, vw_t, pw_s, pw_t, op2_s)

        w_acc, u_acc = pl.run_scoped(
            _phase2,
            plgpu.SMEM((dk, C), dt), plgpu.SMEM((dv, C), dt),
            plgpu.SMEM((C, C), dt), plgpu.SMEM((C, C), dt),
            plgpu.SMEM((C, dk), dt),
        )

        def _phase3(wt_s, s_t, opb_s):
            """Output and the transposed pseudo-value.

            ``w_acc``/``u_acc`` come back from the solve as the *transposed*
            right-hand sides (``(dk, C)`` / ``(dv, C)``); the output algebra
            needs the natural orientation, so the transposes are materialised
            through SMEM.  The pseudo-value is then built in the transposed
            space directly: ``vt^T = U^T - S^T W^T`` -- one MMA with ``S^T``
            as the left operand and ``W`` (via its transposed view) as the
            right one.
            """
            sv = _l(sg.at[h], ACC_DK)                 # (dk, dv) = S
            s_t.transpose((1, 0))[...] = sv.astype(dt)     # s_t = S^T (dv, dk)
            wt_s.transpose((1, 0))[...] = w_acc.astype(dt)  # wt_s = W (C, dk)
            ws_t = plgpu.mma(_zero((dv, C), ACC_DK), _l(s_t, L_DK),
                             _l(wt_s.T, R_DK))             # (W S)^T (dv, C)
            vt_s[...] = (u_acc - ws_t).astype(dt)          # vt^T = U^T - (W S)^T
            # o = gamma_q S + aqk vt   (gamma_q reuses the wt_s buffer)
            opb_s[...] = jnp.exp(G).astype(dt)             # gamma (C, dk)
            gq = (_l(opb_s, ACC_C).astype(f32) * qv).astype(dt)
            wt_s[...] = gq
            acc_o = plgpu.mma(_zero((C, dv), ACC_C), _l(wt_s, L_C), _l(s_t.T, R_C))
            acc_o = plgpu.mma(acc_o, _l(aqk_s, L_C), _l(vt_s.T, R_C))
            og.at[pl.ds(0, C), pl.ds(vo, dv)][...] = acc_o

        pl.run_scoped(
            _phase3,
            plgpu.SMEM((C, dk), dt), plgpu.SMEM((dv, dk), dt),
            plgpu.SMEM((C, dk), dt),
        )
        def _phase4(y_t, opb_s, gl_hi_t, gl_lo_t, g_s):
            la2 = jnp.log(_l(ag.at[pl.ds(0, C), pl.ds(qo, dk)], ACC_C).astype(f32))
            lh = la2.astype(dt)
            ll = (la2 - lh.astype(f32)).astype(dt)
            gl_hi_t.transpose((1, 0))[...] = lh
            gl_lo_t.transpose((1, 0))[...] = ll
            # All-rows := the *full* row sum of ``log alpha`` (pattern ``cidx2 <= C-1``
            # is the anchor idiom with ``p = C-1``), NOT the cumsum: ``lam`` needs
            # ``sum_j la_j - G_c`` = ``gamma^{c+1->C}``.  With the cumsum the
            # difference would be identically zero and ``lam`` would come out as
            # ones -- a state-transfer error (measured up to ~5e-1 on the chunk
            # state at the campaign geometry; caught by the declared-algebra audit
            # ``test_kernel_declared_phase4_patterns`` in milestone 2, never on a
            # device: milestone-1's GB10 probe measured launches only, not values).
            opb_s.at[pl.ds(0, C), pl.ds(0, C)][...] = (cidx2 <= C - 1).astype(dt)
            glast = plgpu.mma(_zero((C, dk), ACC_C),
                              _l(opb_s.at[pl.ds(0, C), pl.ds(0, C)], L_C),
                              _l(gl_hi_t.T, R_C))
            glast = plgpu.mma(glast,
                              _l(opb_s.at[pl.ds(0, C), pl.ds(0, C)], L_C),
                              _l(gl_lo_t.T, R_C))
            lam = jnp.exp(jnp.minimum(glast - G, 0.0))     # gamma^{c+1->C}
            opb_s[...] = lam.astype(dt)
            yv = (_l(opb_s, ACC_C).astype(f32) * kv.astype(f32)).astype(dt)
            y_t.transpose((1, 0))[...] = yv
            # gamma^C as a *row* vector of the (dk, dv)-shaped state (it decays the
            # key channels -- rows of S).  The vector is a side of a different
            # layout family ((C, dk) source vs the (dk, dv)-shaped destination
            # value), so it goes through SMEM and is re-loaded with the
            # destination layout's own *row-vector* reduce (the rmsnorm recipe:
            # `load(vec, layout.reduce(1))` for a value tiling reduced on the
            # column axis), with an explicit target layout on the broadcast --
            # implicit broadcasting is not inferred.
            g_s[...] = jnp.exp(jnp.sum(la2, axis=0)).astype(dt)
            gcv = _l(g_s, ACC_DK.reduce(1)).astype(f32)
            gc_b = plgpu.layout_cast(
                jax.lax.broadcast_in_dim(gcv, (dk, dv), (0,)), ACC_DK)
            sv = _l(sg.at[h], ACC_DK)
            acc_s = plgpu.layout_cast(sv * gc_b, ACC_DK)
            s_out = plgpu.mma(acc_s, _l(y_t, L_DK), _l(vt_s.T, R_DK))
            sog.at[h][...] = s_out

        pl.run_scoped(
            _phase4,
            plgpu.SMEM((dk, C), dt), plgpu.SMEM((C, dk), dt),
            plgpu.SMEM((dk, C), dt), plgpu.SMEM((dk, C), dt),
            plgpu.SMEM((dk,), dt),
        )

    return plgpu.kernel(
        _body,
        out_type=(
            jax.ShapeDtypeStruct((C, H * dv), jnp.float32),
            jax.ShapeDtypeStruct((H, dk, dv), jnp.float32),
        ),
        scratch_types=[
            plgpu.SMEM((C, C), dt),      # aqk (phase 1 -> 3)
            plgpu.SMEM((C, C), dt),      # L (phase 1 -> 2)
            plgpu.SMEM((dv, C), dt),     # v_tilde^T (phase 3 -> 4)
        ],
        compiler_params=dataclasses.replace(
            plgpu.CompilerParams(), lowering_semantics=plgpu.LoweringSemantics.Lane
        ),
        grid=(H,),
        grid_names=("head",),
    )


@dataclasses.dataclass(frozen=True)
class _KernelSpec:
    """Static shapes the kernel was built for (jit cache key material)."""

    C: int
    H: int
    dk: int
    dv: int
    tile: int
    dtype: str


def kernel(q, k, v, alpha, beta32, S, *, tile=16):
    """Fused forward chunk body: ``(q, k, v, alpha, beta32, S) -> (o, S_new)``.

    ``q``/``k``: ``(C, H*dk)`` (bf16); ``v``: ``(C, H*dv)``; ``alpha``:
    ``(C, H*dk)`` fp32 decay; ``beta32``: ``(C, H*32)`` fp32 write gate
    replicated over 32 columns; ``S``: ``(H, dk, dv)`` fp32 incoming state.
    Returns ``o`` ``(C, H*dv)`` fp32 (before the output gate) and the new
    state ``(H, dk, dv)`` fp32 -- exactly the interval ``cc_chunk_step``
    covers between ``_postprocess`` and ``_output_gate``.

    Execution needs Mosaic GPU (GB10); the *lowering* runs anywhere
    (``lower_check.py`` on this CPU node).  ``tile`` is the static tile width
    of the decay factors and must match ``net.kda._cc_tile`` (<= 16).
    """
    C = q.shape[0]
    H, dk, dv = S.shape
    dt = q.dtype
    if v.dtype != dt or dt not in (jnp.bfloat16,):
        raise ValueError(
            f"fused kernel consumes bf16 q/k/v (the stand-validated operand path "
            f"of the sm_121 mma); got q.dtype={dt}, v.dtype={v.dtype}")
    spec = _KernelSpec(C, H, dk, dv, int(tile), jnp.dtype(dt).name)
    run = _build_kernel(spec.C, spec.H, spec.dk, spec.dv, spec.tile, spec.dtype)
    return jax.jit(run)(q, k, v, alpha, beta32, S)


def reference(q, k, v, alpha, beta32, S, *, tile=16):
    """fp32 jnp replica of the kernel's algebra (the CPU-side numerical pin).

    Written in the same *laid-out* form as the kernel body: the two score
    matrices are built tile-wise against each tile's first position, the WY/UT
    solve is the alternating Neumann chain left-multiplying the transposed
    right-hand sides, and the decay is accumulated as a fp32 sum over bf16
    hi/lo halves of ``log alpha``.  Pinning this against
    ``net.kda.cc_chunk_step`` (``net/tests/test_kda_cc_fused.py``) therefore
    checks the transposed dataflow, not just the final formula.
    """
    C = q.shape[0]
    H, dk, dv = S.shape
    f32 = jnp.float32
    ar = jnp.arange(C)
    ridx = ar[:, None]
    cidx = ar[None, :]
    tril = (ridx >= cidx).astype(f32)
    o = jnp.zeros((C, H * dv), f32)
    S_new = jnp.zeros((H, dk, dv), f32)
    steps = 0
    while (1 << steps) < C:
        steps += 1

    def solve(X, P):
        """(I + L)^{-1} X via the kernel's chain: X <- (I-L)(I+L^2)(I+L^4)... X."""
        Xt = X.T
        pw = P
        Xt = Xt - Xt @ pw.T
        for _ in range(steps - 1):
            pw = pw @ pw
            Xt = Xt + Xt @ pw.T
        return Xt.T

    for h in range(H):
        qh = q[:, h * dk:(h + 1) * dk].astype(f32)
        kh = k[:, h * dk:(h + 1) * dk].astype(f32)
        vh = v[:, h * dv:(h + 1) * dv].astype(f32)
        ah = alpha[:, h * dk:(h + 1) * dk].astype(f32)
        bh = beta32[:, h * 32:(h + 1) * 32].astype(f32)
        beta = jnp.sum(bh, axis=1) * (1.0 / 32.0)      # (C,), replicated input
        la = jnp.log(ah)
        G = tril @ la                                  # (C, dk) cumsum

        def scores(row_v, mask):
            acc = jnp.zeros((C, C), f32)
            for p in range(0, C, tile):
                anchor = (cidx <= p).astype(f32) @ la          # (C, dk) = G_p
                rmk = ((ar >= p) & (ar < p + tile)).astype(f32)[:, None]
                rf = jnp.exp(jnp.minimum(G - anchor, 0.0)) * rmk
                cmk = (ar < p + tile).astype(f32)[:, None]
                cf = jnp.exp(anchor - G) * cmk
                acc = acc + (row_v * rf) @ (kh * cf).T
            return acc * mask

        # L = strict_tril(diag(beta) Akk): the row operand carries beta.
        akk = scores(kh * beta[:, None], (ridx > cidx).astype(f32))
        aqk = scores(qh, tril)
        xw = (jnp.exp(G) * kh) * beta[:, None]         # (C, dk)
        vw = vh * beta[:, None]                        # (C, dv)
        w = solve(xw, akk)                             # rows already beta-scaled
        u = solve(vw, akk)
        v_tilde = u - w @ S[h]
        gamma_q = jnp.exp(G) * qh
        o = o.at[:, h * dv:(h + 1) * dv].set(gamma_q @ S[h] + aqk @ v_tilde)
        gc = jnp.exp(jnp.sum(la, axis=0))              # (dk,) gamma^C
        lam = jnp.exp(jnp.minimum(jnp.sum(la, axis=0)[None, :] - G, 0.0))
        y = lam * kh
        S_new = S_new.at[h].set(gc[:, None] * S[h] + y.T @ v_tilde)
    return o, S_new


# ---------------------------------------------------------------------------
# Differentiable wrapper (ADR-054 stage-2b, milestone 2)
#
# The bare ``kernel`` is a Pallas ``pallas_call``: it has no JVP, so a training
# graph that inlines it fails ("Linearization failed to produce known values
# for all output primals").  ``cc_step`` wraps it in ``jax.custom_vjp``: the
# forward rule runs the kernel, the backward rule runs the analytic adjoint of
# :mod:`net.kernels.kda_cc_fused_bwd` (derived, CPU-gradchecked and pinned
# there).  Both rules resolve their implementation on the *module* at call
# time, so tests can observe or substitute which forward/backward runs; the
# factory takes explicit implementations for the GPU-less-node tests.
# ---------------------------------------------------------------------------


def _cc_forward(q, k, v, alpha, beta32, S, tile):
    """Forward rule body — the fused kernel, resolved per call (module global)."""
    return kernel(q, k, v, alpha, beta32, S, tile=tile)


def _cc_backward(q, k, v, alpha, beta32, S, do, dS, tile):
    """Backward rule body — the analytic adjoint, resolved per call.

    The adjoint lives in the sibling module (lazy import: the forward module
    stays importable without it; the attribute is looked up per call so tests
    can spy on the routing exactly like ``kda_ut_solve``'s kernel spies).
    """
    from . import kda_cc_fused_bwd

    return kda_cc_fused_bwd.reference_bwd(q, k, v, alpha, beta32, S, do, dS,
                                          tile=tile)


def make_cc_step(tile=16, forward=None, backward=None):
    """Build the differentiable chunk body ``(q, k, v, alpha, beta32, S)``.

    Returns a ``jax.custom_vjp`` function: forward = :func:`kernel` (or the
    ``forward`` substitution), backward = :func:`kda_cc_fused_bwd.reference_bwd`
    (or the ``backward`` substitution).  ``tile`` is **captured in the closure**
    and not an argument: ``custom_vjp`` is jit-like, so a Python int passed as
    an argument arrives inside the fwd rule as a weak int32 *tracer*
    (``TracerIntegerConversionError`` on the kernel's ``int(tile)``); capturing
    it keeps it a host constant.  The substitutions exist for the CPU test
    suite (the kernel itself needs Mosaic GPU; the analytic adjoint does not).
    """

    def _fwd(q, k, v, alpha, beta32, S):
        fn = forward if forward is not None else _cc_forward
        return fn(q, k, v, alpha, beta32, S, tile)

    def _bwd(q, k, v, alpha, beta32, S, do, dS):
        fn = backward if backward is not None else _cc_backward
        return fn(q, k, v, alpha, beta32, S, do, dS, tile)

    @jax.custom_vjp
    def step(q, k, v, alpha, beta32, S):
        return _fwd(q, k, v, alpha, beta32, S)

    def step_fwd(q, k, v, alpha, beta32, S):
        o, s_new = _fwd(q, k, v, alpha, beta32, S)
        return (o, s_new), (q, k, v, alpha, beta32, S)

    def step_bwd(res, g):
        q, k, v, alpha, beta32, S = res
        do, dS = g
        dq, dk, dv, dalpha, dbeta32, dS_in = _bwd(
            q, k, v, alpha, beta32, S, do, dS)
        return dq, dk, dv, dalpha, dbeta32, dS_in

    step.defvjp(step_fwd, step_bwd)
    return step


#: Cache of the wrappers by static tile width (``net.kda._cc_tile`` is the
#: declared source of the value; one wrapper per width).  The wrappers resolve
#: ``kernel``/``reference_bwd`` on their modules at call time, so observing or
#: substituting them keeps working through this cache.
_CC_STEPS: dict[int, object] = {}


def cc_step(q, k, v, alpha, beta32, S, tile=16):
    """Differentiable body the ``AXIOM_KDA_CC_IMPL=mosaic`` mode installs.

    ``jax.custom_vjp``: forward rule = the fused kernel, backward rule = the
    analytic adjoint of :mod:`net.kernels.kda_cc_fused_bwd`.  ``tile`` is a
    *static* host argument (the decay-factor tile width the kernel was built
    with); the wrapper for each width is cached — the host dispatch happens at
    trace time, before ``custom_vjp`` turns values into tracers.
    """
    step = _CC_STEPS.get(tile)
    if step is None:
        step = make_cc_step(tile=tile)
        _CC_STEPS[tile] = step
    return step(q, k, v, alpha, beta32, S)
