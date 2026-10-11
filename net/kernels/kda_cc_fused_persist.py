"""Persistent KDA ``chunked_cc`` scan kernel (ADR-054 Amendment 4, milestone 3).

Why this module exists (the measured unit of fusing)
----------------------------------------------------
Milestone 1 collapsed one chunk's KDA algebra into one Mosaic launch and the
isolated layer probe measured a real win (6203 -> 896 launches per layer, GPU
time -47 %), but the *full training step* with the per-chunk ``custom_vjp``
came out worse than the plain jax path: 704k kernels/step against ~100k,
117k ``cuMemcpyDtoDAsync`` = 13.7 s/step (ADR-054 Amendment 4,
``evidence/mfu-55/kda-fused/full-step/``).  The diagnosis: the unit of fusing
was too small -- the XLA custom-call *boundary* (one per chunk, 256 per
layer-pass) generates ~50 small operations and layout copies per iteration,
and the boundary costs more than the launches it saves.

Milestone 3 (this module) therefore changes the unit: **one kernel launch per
layer-pass** (``grid=(H,)``), the program walking the whole scan of the
sequence *inside itself*: the state ``S`` lives in SMEM between chunk
iterations, the per-chunk inputs stream from GMEM with dynamic offsets, the
output is written chunk by chunk, and exactly one custom-call boundary per
layer-pass survives (54 per step for 18 layers x 3 passes, against 704k
kernels).

Scope and structure
-------------------
* :func:`kernel` -- the persistent Mosaic kernel over the whole sequence:
  ``(q, k, v, alpha, beta32, S0) -> (o, S_new)``, all chunks of one head in
  one program.  Per chunk the body is exactly the milestone-1 algebra
  (:mod:`net.kernels.kda_cc_fused`, pinned to ``net.kda.cc_chunk_step``):
  tile-wise decay factors, the two ``C x C`` score matrices, the WY/UT
  Neumann solve, the state transfer and the output -- the only structural
  change is that ``S`` is read from and written to an SMEM buffer instead of
  GMEM, and the chunk offsets are dynamic loop indices.
* :func:`reference` -- the fp32 jnp replica of the *same* design, chunk by
  chunk (the CPU-side numerical pin, exactly the milestone-1 contract).
* the *declared bf16 rounding points* of the persistent design -- including
  the per-chunk SMEM state rounding -- are emulated by
  ``net/tests/test_kda_cc_fused_persist.py`` (the per-chunk emulation lives in
  the milestone-1 test file; this file's tests wrap it with the state
  rounding), so the CPU suite measures the drift term the persistent design
  introduces instead of assuming it away;
* :func:`make_cc_scan` / :func:`cc_scan` -- the differentiable whole-sequence
  entry point (``jax.custom_vjp``): forward rule = the persistent kernel,
  backward rule = a reverse scan over chunks of the analytic adjoint
  (:mod:`net.kernels.kda_cc_fused_bwd`, derived and CPU-gradchecked in
  milestone 2).  This is **variant A** of the milestone-3 delta: the forward
  pass (and, where rematerialisation is used, its recomputation) collapse
  into one launch per layer-pass; the backward pass stays a multi-operation
  jnp scan (variant B -- a persistent transposed backward kernel -- is the
  documented follow-up, see ``open_questions``).

The state in SMEM: dtype decision (measured, not assumed)
---------------------------------------------------------
``dk x dv = 128 x 128`` is 64 KiB fp32 / 32 KiB bf16, the GB10 SMEM budget is
101 376 B, and the phase arenas of the chunk body are pooled by
``pl.run_scoped`` liveness (floor = ``S + persistent scratch + max(phase)``).
The decision is taken by the *lowering*, not by prose: the numbers below are
the module's own ``smem_bytes``, read out of the Mosaic lowering with the
skill's artificial SMEM limit (the same machinery ``lower_check`` uses), in
``net/tests/test_kda_cc_fused_persist.py``:

================  ================  ==========  ==========================
configuration     SMEM (measured)   vs budget   verdict
================  ================  ==========  ==========================
C=32, S bf16      94 208 B          +7 168 B    **shipped** (live)
C=32, S fp32      126 976 B         -25 600 B   rejected (sieve + lowering)
C=16, S bf16      79 872 B          +21 504 B   tiling: C<32 rejected
C=16, S fp32      112 640 B         -11 264 B   rejected (tiling and budget)
C=64, S bf16      134 144 B         -32 768 B   rejected (milestone-1 fallback)
================  ================  ==========  ==========================

The C=16 rows are worth stating plainly.  The delta sketched C=16/fp32 as a
fitting compromise ("~27 KiB chunk set + 64 KiB state"), but two measured
facts close that door: the ``(C, C)`` values of the chunk body ride the
``MMA_ACC(dt, m_warps=1)`` tile ``(16, 32)``, which does not apply to a
``(16, 16)`` shape (the effort of a second layout family for an unused width
was not spent), and even with a working tiling the fp32 state would not fit
next to the milestone-1 phase set.  C=16 therefore sits in :data:`CASES` as an
honestly *skipped* entry, like C=64.  The bf16 state at C=32 is the shipped
design.

The drift question is answered by measurement, not by hope: the declared bf16
state adds a per-chunk rounding of the retransferred state, and the CPU
emulation (``net/tests/test_kda_cc_fused_persist.py``) reproduces it exactly.  At the campaign
geometry the state magnitude is small and the decay strong, so the term is
~1e-4; even in a weak-decay stress (``g_min`` down to -0.005, i.e. a
per-token decay of ~0.9975 and a per-chunk state retention of ~0.92) the
pre-gate output deviation against the fp32 replica stays <= ~1.3e-3 and the
state drift <= ~1e-3 -- an order of magnitude under the declared 2e-2 operand
budget (numbers pinned in ``net/tests/test_kda_cc_fused_persist.py``, where a
dedicated test isolates the drift term from the milestone-1 operand
quantisation).  The state drift is *not* zero and is declared here: at a
trained model's much longer effective memory the term could grow -- the GB10
loss-parity check is listed in ``open_questions``.  The documented follow-up
that would remove it entirely (if ever needed) is to keep ``S^T`` in the SMEM
buffer phase 3 already uses as its staging copy and drop that copy: the same
94 208 B budget then carries an fp32 state (measured: fp32 + no staging
buffer would need the phase-3 ``(dv, dk)`` arenas gone; it is a phase-3/4
restructuring with the state written in the transposed orientation).

Backward variant (A vs B) -- the choice, stated
-----------------------------------------------
Variant A (shipped): the forward pass *and* its rematerialisation collapse
into one kernel per layer-pass; the backward pass is a jnp reverse scan of the
milestone-2 analytic per-chunk adjoint (a multi-operation backward, the honest
boundary).  Variant B (a persistent *transposed* backward kernel, so all three
passes collapse) needs the same ~20-mma hand-written backward discussed in
:mod:`net.kernels.kda_cc_fused_bwd`'s "Variant A" section -- and this node
cannot verify it beyond lowering, which the milestone-2 declared-algebra audit
showed is not enough for a hand-written Mosaic body.  Variant A is therefore
the shipped unit; B stays a scoped follow-up in ``open_questions``.

Verification status (honest, same standard as milestones 1-2)
-------------------------------------------------------------
* ``reference`` is pinned on CPU against ``net.kda.cc_chunk_step`` (the
  committed scan body) chunk by chunk at the campaign geometry;
* the bf16 emulation (the kernel's declared rounding points, bf16 SMEM state
  included) is pinned against the fp32 replica under the declared 2e-2 operand
  tolerance at the kernel's boundary, at every chunk boundary, at the campaign
  geometry and in the weak-decay stress; the drift term is isolated in a
  dedicated test;
* the mode mux (``AXIOM_KDA_CC_IMPL=persistent``) is pinned end to end on CPU
  with the kernel seam substituted by its declared numerics: forward parity,
  gradient parity (inputs and parameters), repeatability, the fail-closed
  GPU-less refusal, the padding path and the inert remat flag;
* the *launch-count claim at the API seam* is pinned: exactly one kernel call
  per layer-pass (the per-chunk ``mosaic`` body called it once per chunk);
* the kernel *lowers* for sm_121 under the skill's patched lowering (the
  ``lower_check`` contract: every live case OK, oversize cases honestly
  skipped by :func:`valid_config`), and the lowered module carries exactly one
  ``scf.for`` -- the dynamic chunk loop, not an unrolled scan;
* **the kernel is not executed on this node** (CPU-only; jax 0.11.2's GPU
  interpret mode has no rule for ``plgpu.mma``).  Device parity, timing,
  determinism and the per-pass launch count of the real run are the
  architect's GB10 measurement (``check_kernel.py``); ``open_questions`` lists
  exactly what to measure.

Layout recipes (all validated by lowering on this node; the mma/SMEM core
patterns are the GB10-verified ones of ``net/kernels/kda_ut_solve_mosaic`` and
milestone 1):

* ``plgpu.mma(acc, a, b)`` = ``acc + a @ b``; the LHS is loaded from a
  row-major SMEM buffer (``k``-contiguous memory), the RHS from ``buf.T``.
* SMEM is the only way to change a register layout; transposes are
  materialised by storing into ``buf.transpose((1, 0))`` views.
* row scalings use ``reduce(1)``, column scalings ``reduce(0)``, with an
  explicit ``jax.lax.broadcast_in_dim`` on the broadcast (implicit
  broadcasting is not inferred).
* ``plgpu.store`` does not exist in 0.11.2: GMEM writes are ``ref[...] = v``
  over ``ref.at[pl.ds(...), pl.ds(...)]`` slices.
* decay cumsums run through the tril mma on hi/lo bf16 halves of ``log
  alpha`` (a plain bf16 rounding biases the cumsum past the 2e-2 budget).
* the chunk loop is ``pl.loop`` (dynamic ``fori``): the per-chunk loads use
  dynamic ``pl.ds(i * C, C)`` offsets; loop-invariant scratch (``aqk``,
  ``L``, ``v_tilde^T``) and the SMEM state are closed over.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Module contract (pallas-gb10-kernel skill)
# ---------------------------------------------------------------------------

#: Case shapes.  ``T`` is the whole-sequence length (a multiple of ``C``; the
#: wrapper pads).  The first two entries are the live campaign cases (H=12 is
#: the model, H=2 keeps the CPU-side verification cheap); the C=64 entries are
#: the declared fallback, honestly *skipped* by :func:`valid_config` (their
#: SMEM floor is over the GB10 budget -- the milestone-1 arithmetic, now with
#: the SMEM state added).
CASES = [
    dict(T=2048, C=32, H=12, dk=128, dv=128, tile=16, dtype="bfloat16"),
    dict(T=256, C=32, H=2, dk=128, dv=128, tile=16, dtype="bfloat16"),
    dict(T=256, C=16, H=2, dk=128, dv=128, tile=16, dtype="bfloat16"),   # C=16: mma tiling, skipped
    dict(T=512, C=64, H=12, dk=128, dv=128, tile=16, dtype="bfloat16"),  # oversize: skipped
]

#: bf16-operand / fp32-accumulator path; the fp32 path is the exact reference.
TOLERANCE = {"bfloat16": (2e-2, 2e-2), "float32": (1e-5, 1e-5)}

#: Static knobs of :func:`kernel`.  ``tile`` is the live knob (the tile-wide
#: decay-factor budget of ``net.kda._cc_tile``).
TUNE_SPACE = dict(tile=[16])

#: GB10 SMEM budget of one block (99 KiB, the value the skill's lower_check
#: patches into the Mosaic lowering on a GPU-less node) minus the epilogue
#: staging margin (4 KiB, one ``(C, dk)`` f32 store at C=32).
SMEM_LIMIT_BYTES = 101376
SMEM_EPILOGUE_BYTES = 4 * 1024

#: The tile budget rule of ``net.kda._cc_tile``, mirrored as a static bound.
TILE_MAX = 16

#: The declared chunk width of the persistent scan (the campaign value of
#: ``cfg.kda_wyut_chunk``; the kernel takes it as a static parameter).
CHUNK = 32

#: The persistent state lives in SMEM; its dtype is the milestone-3 decision.
#: bf16 fits the GB10 budget next to the milestone-1 phase arenas (fp32 does
#: not -- see the module docstring), and the measured drift at the declared
#: tolerance is in the tests.  fp32 stays supported by the sieve arithmetic
#: (it lowers on no live case at C=32; the honest skip, not a silent one).
STATE_DTYPE = jnp.bfloat16


def _smem_floor_bytes(C, dk, dv, itemsize=2, state_itemsize=2):
    """Declared SMEM floor of one *chunk iteration* plus the persistent set.

    The phases are ``pl.run_scoped`` arenas with disjoint lifetimes (the
    liveness-pooling contract of ``pl.run_scoped``; checked by the lowering on
    every live case, not assumed silently).  Persistent: the SMEM state
    ``dk*dv``, the cross-phase scratch ``aqk``/``L`` ``(C, C)``, the ``vt``
    ``(dv, C)`` buffer, plus the ``S^T`` staging ``(dv, dk)`` that phase 3
    keeps live across the state read (it is written from the state read and
    read by two mma sites of the same phase -- a phase-3 arena entry, not a
    persistent one).

    * phase 1: ``la_hi_t``/``la_lo_t`` ``(dk, C)`` + ``op`` ``(C, dk)`` +
      ``pat`` ``(C, C)``;
    * phase 2: ``xw_t`` ``(dk, C)`` + ``vw_t`` ``(dv, C)`` + ``P``/``P^T``
      ``(C, C)`` + ``op2`` ``(C, dk)``;
    * phase 3: ``W``/``gamma_q`` ``(C, dk)`` + ``S^T`` ``(dv, dk)`` -- the
      binding phase, its ``dv*dk`` buffer dominates;
    * phase 4: ``y``/``gl_hi``/``gl_lo`` ``(dk, C)`` + ``op`` ``(C, dk)``.
    """
    el = dv * C + 2 * C * C                                        # persistent scratch
    phases = (
        3 * C * dk + C * C,                                        # phase 1
        2 * C * dk + dv * C + 2 * C * C,                           # phase 2
        2 * C * dk + dk * dv,                                      # phase 3
        4 * C * dk + dk,                                           # phase 4
    )
    return (itemsize * (el + max(phases)) + state_itemsize * dk * dv
            + SMEM_EPILOGUE_BYTES)


def valid_config(case, params):
    """Static sieve: MMA tiling rules and the SMEM floor of the design.

    Returns False for anything the GB10 cannot run -- in particular the C=16
    and C=64 fallback entries of :data:`CASES`, so ``lower_check`` skips them
    instead of failing on them, which is the honest form of "does not fit":

    * ``C < 32``: the ``(C, C)`` values of the chunk body ride the
      ``Layout.MMA_ACC(dt, m_warps=1)`` accumulator layout, whose tile is
      ``(16, 32)``; a ``C < 32`` shape cannot take that tiling (measured on
      this node: ``ValueError: Tiling ((16, 32), ...) does not apply to shape
      (16, 16)`` for C=16) -- another reason the C=16/fp32 compromise of the
      delta is not available without a second layout family;
    * the SMEM floor over :data:`SMEM_LIMIT_BYTES` (C=64, and the fp32 SMEM
      state at C=32).
    """
    C, H, dk, dv = (int(case[k]) for k in ("C", "H", "dk", "dv"))
    tile = int(params.get("tile", case.get("tile", TILE_MAX)))
    if C % 16 or C < 32 or dk % 64 or dv % 8 or dv > 128 or dk > 128 or tile > TILE_MAX or tile < 1:
        return False
    if (C * dk) % 128 or (C * dv) % 128:  # transfer granularity of the loads
        return False
    dt = jnp.dtype(case.get("dtype", "bfloat16"))
    if dt not in (jnp.bfloat16, jnp.float32):
        return False
    st = jnp.dtype(case.get("state_dtype", STATE_DTYPE))
    return _smem_floor_bytes(C, dk, dv, dt.itemsize, st.itemsize) <= SMEM_LIMIT_BYTES


def cost(T, C, H, dk, dv, tile, dtype="bfloat16", **kw):
    """Minimal FLOPs/bytes of one persistent pass (roofline bookkeeping)."""
    ntile = -(-C // int(tile))
    per_chunk = (
        2 * 2 * C * C * dk                       # aqk + akk (tile-rebuilt)
        + 2 * 2 * ntile * C * C * dk             # tile-wise rebuild (4 passes)
        + 2 * 2 * C * C * (dk + dv)              # solve RHS + power chain
        + 2 * C * (dk + dv) * dk                 # wS, inter
        + 2 * C * dv * C                         # intra
        + 2 * dk * dv * C                        # state update
    )
    n_chunks = T // C
    b = jnp.dtype(dtype).itemsize
    per_chunk_bytes = (
        C * (2 * dk + dv) * b                    # q, k, v
        + C * dk * 4 * 2                         # alpha + hi/lo side buffer
        + C * 32 * 4                             # replicated beta
        + C * dv * 4                             # output o (pre-gate)
    )
    return dict(flops=n_chunks * per_chunk,
                bytes=n_chunks * per_chunk_bytes + 2 * dk * dv * 4)


def make_inputs(key, T, C, H, dk, dv, tile=16, dtype="bfloat16", **kw):
    """Inputs of one persistent pass: ``(q, k, v, alpha, beta32, S)``.

    * ``q``, ``k``: ``(T, H*dk)`` post-shortconv+swish+L2Norm projections,
    * ``v``: ``(T, H*dv)`` post-shortconv+swish projection,
    * ``alpha``: ``(T, H*dk)`` fp32 channel decay ``exp(g)`` in ``(0, 1]``,
    * ``beta32``: ``(T, H*32)`` fp32 write gate replicated over 32 columns,
    * ``S``: ``(H, dk, dv)`` fp32 incoming state.
    """
    if T % C:
        raise ValueError(f"T={T} must be a multiple of C={C} (the wrapper pads)")
    dt = jnp.dtype(dtype)
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    scale = 1.0 / (dk ** 0.25)
    q = (jax.random.normal(k1, (T, H * dk), jnp.float32) * scale).astype(dt)
    k = (jax.random.normal(k2, (T, H * dk), jnp.float32) * scale).astype(dt)
    v = (jax.random.normal(k3, (T, H * dv), jnp.float32) * scale).astype(dt)
    g = -0.5 * jax.random.uniform(k4, (T, H * dk), jnp.float32)  # log alpha
    alpha = jnp.exp(g)
    beta = jax.random.uniform(k5, (T, H), jnp.float32)
    beta32 = jnp.broadcast_to(beta[:, :, None], (T, H, 32)).reshape(T, H * 32)
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
# The kernel (one program per head, the whole scan inside)
# ---------------------------------------------------------------------------


def _mma_m_warps(m: int) -> int:
    """Warp-group width of the MMA layouts for an accumulator with ``m`` rows.

    Lowering fact (milestone 1, this node, jax 0.11.2, sm_121): the default
    ``m_warps=4`` tiling is a 64-row tile and does not apply to 32-row shapes;
    with ``m_warps=1`` the same shapes lower.  The 128-row accumulators of the
    state algebra keep the wide ``m_warps=4`` register tile.
    """
    return 1 if m < 64 else 4


def _build_kernel(T, C, H, dk, dv, tile, dtype, state_dtype):
    """Build the persistent Mosaic GPU kernel: one program per head, all chunks."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import mosaic_gpu as plgpu

    dt = jnp.dtype(dtype)
    stdt = jnp.dtype(state_dtype)
    f32 = jnp.float32
    n_chunks = T // C

    def _l(ref, layout):
        return plgpu.load(ref, layout=layout, optimized=False)

    def _mma_layouts(m):
        mw = _mma_m_warps(m)
        return (
            plgpu.Layout.MMA_ACC(dt, m_warps=mw),
            plgpu.Layout.MMA_LHS(dt, m_warps=mw),
            plgpu.Layout.MMA_RHS(dt, m_warps=mw),
        )

    ACC_C, L_C, R_C = _mma_layouts(C)          # (C, .) accumulators
    ACC_DK, L_DK, R_DK = _mma_layouts(dk)      # (dk, .) accumulators

    def _zero(shape, acc_layout):
        return plgpu.layout_cast(jnp.zeros(shape, f32), acc_layout)

    def _body(qg, kg, vg, ag, bg, sg, og, sog, s_state, aqk_s, lkk_s, vt_s):
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

        # --- the loop-invariant state: S in SMEM between chunk iterations ----
        s_state[...] = _l(sg.at[h], ACC_DK).astype(stdt)

        def _chunk(i):
            qv = _l(qg.at[pl.ds(i * C, C), pl.ds(qo, dk)], ACC_C).astype(f32)
            kv = _l(kg.at[pl.ds(i * C, C), pl.ds(qo, dk)], ACC_C)
            cv = _l(bg.at[pl.ds(i * C, C), pl.ds(bo, 32)], ACC_C).astype(f32)
            # beta32 replicates beta over the 32 columns (the loadable shape of
            # a per-row vector): the sum recovers 32 * beta, exactly (2**5).
            betav = jnp.sum(cv, axis=1) * (1.0 / 32.0)       # (C,) row vector
            # Broadcasts need an explicit target layout on this lowering (rmsnorm
            # recipe): the inferred candidates do not cover the C=32 tilings.
            beta_c = plgpu.layout_cast(jax.lax.broadcast_in_dim(betav, (C, dk), (0,)), ACC_C)
            beta_v = plgpu.layout_cast(jax.lax.broadcast_in_dim(betav, (C, dv), (0,)), ACC_C)

            def _decay(la_hi_t, la_lo_t, pat_s):
                """log(alpha) -> hi/lo bf16 halves -> fp32 cumsum via the tril mma."""
                av = _l(ag.at[pl.ds(i * C, C), pl.ds(qo, dk)], ACC_C).astype(f32)
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

            vv = _l(vg.at[pl.ds(i * C, C), pl.ds(vo, dv)], ACC_C)
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

                The state read is the *persistent* SMEM state (milestone 3):
                ``s_t`` is the bf16 staging copy the two mma sites consume --
                the same rounding the milestone-1 kernel applies to the GMEM
                state, but now re-read from SMEM every chunk.
                """
                sv = _l(s_state, ACC_DK)
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
                og.at[pl.ds(i * C, C), pl.ds(vo, dv)][...] = acc_o

            pl.run_scoped(
                _phase3,
                plgpu.SMEM((C, dk), dt), plgpu.SMEM((dv, dk), dt),
                plgpu.SMEM((C, dk), dt),
            )

            def _phase4(y_t, opb_s, gl_hi_t, gl_lo_t, g_s):
                la2 = jnp.log(_l(ag.at[pl.ds(i * C, C), pl.ds(qo, dk)], ACC_C).astype(f32))
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
                sv = _l(s_state, ACC_DK)
                acc_s = plgpu.layout_cast(sv.astype(f32) * gc_b, ACC_DK)
                s_out = plgpu.mma(acc_s, _l(y_t, L_DK), _l(vt_s.T, R_DK))
                # The persistent state write: the retransferred state is stored
                # into SMEM and is the operand of the next chunk's state read.
                s_state[...] = s_out.astype(stdt)

            pl.run_scoped(
                _phase4,
                plgpu.SMEM((dk, C), dt), plgpu.SMEM((C, dk), dt),
                plgpu.SMEM((dk, C), dt), plgpu.SMEM((dk, C), dt),
                plgpu.SMEM((dk,), dt),
            )

        pl.loop(0, n_chunks)(_chunk)

        # The final state, written once after the scan (GMEM out).
        sog.at[h][...] = _l(s_state, ACC_DK).astype(f32)

    return plgpu.kernel(
        _body,
        out_type=(
            jax.ShapeDtypeStruct((T, H * dv), jnp.float32),
            jax.ShapeDtypeStruct((H, dk, dv), jnp.float32),
        ),
        scratch_types=[
            plgpu.SMEM((dk, dv), stdt),  # persistent state S (milestone 3)
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

    T: int
    C: int
    H: int
    dk: int
    dv: int
    tile: int
    dtype: str
    state_dtype: str


def kernel(q, k, v, alpha, beta32, S, *, tile=16, chunk=CHUNK):
    """Persistent scan: ``(q, k, v, alpha, beta32, S) -> (o, S_new)``, one launch.

    ``q``/``k``: ``(T, H*dk)`` bf16; ``v``: ``(T, H*dv)``; ``alpha``:
    ``(T, H*dk)`` fp32 decay; ``beta32``: ``(T, H*32)`` fp32 write gate
    replicated over 32 columns; ``S``: ``(H, dk, dv)`` fp32 incoming state.
    ``chunk`` is the *static* chunk width of the scan (``C``); ``T`` must be a
    multiple of it (the caller pads; the mode wrapper in ``net.kda`` does).
    Returns ``o`` ``(T, H*dv)`` fp32 (before the output gate) and the final
    state ``(H, dk, dv)`` fp32 -- exactly the interval
    :func:`net.kda.cc_chunk_step` covers, scanned.

    Execution needs Mosaic GPU (GB10); the *lowering* runs anywhere
    (``lower_check.py`` on this CPU node).
    """
    T, _ = q.shape
    H, dk, dv = S.shape
    C = int(chunk)
    dt = q.dtype
    if v.dtype != dt or dt not in (jnp.bfloat16,):
        raise ValueError(
            f"persistent kernel consumes bf16 q/k/v (the stand-validated operand "
            f"path of the sm_121 mma); got q.dtype={dt}, v.dtype={v.dtype}")
    if T % C:
        raise ValueError(f"T={T} must be a multiple of the chunk width C={C}")
    if not valid_config(dict(T=T, C=C, H=H, dk=dk, dv=dv, tile=int(tile),
                            dtype=jnp.dtype(dt).name), {"tile": int(tile)}):
        raise ValueError(
            f"persistent kernel: configuration (C={C}, H={H}, dk={dk}, dv={dv}, "
            f"dtype={jnp.dtype(dt).name}) is not among the live cases of the sieve "
            f"(mma tiling rules and the GB10 SMEM floor -- see valid_config); "
            f"no silent fallback")
    spec = _KernelSpec(T, C, H, dk, dv, int(tile), jnp.dtype(dt).name,
                       jnp.dtype(STATE_DTYPE).name)
    run = _build_kernel(spec.T, spec.C, spec.H, spec.dk, spec.dv, spec.tile,
                        spec.dtype, spec.state_dtype)
    return jax.jit(run)(q, k, v, alpha, beta32, S)


# ---------------------------------------------------------------------------
# The fp32 replica (CPU-side numerical pin) and the scan-of-chunks adjoint
# ---------------------------------------------------------------------------


def _chunks(x, C):
    """Split the leading axis into chunks: ``(n, C, ...)``."""
    T = x.shape[0]
    return x.reshape(T // C, C, *x.shape[1:])


def reference(q, k, v, alpha, beta32, S, *, tile=16, chunk=CHUNK):
    """fp32 jnp replica of the persistent design (chunk scan, fp32 state).

    Same algebra as :func:`net.kernels.kda_cc_fused.reference` per chunk (the
    laid-out form of the kernel body, pinned to ``net.kda.cc_chunk_step`` by
    the milestone-1 tests), applied chunk by chunk with the state carried in
    fp32 -- i.e. the *exact* semantics the persistent kernel approximates.
    The kernel's declared rounding points (including the per-chunk bf16 SMEM
    state) are emulated by the tests, not here.
    """
    from . import kda_cc_fused

    T = q.shape[0]
    C = int(chunk)
    if T % C:
        raise ValueError(f"T={T} must be a multiple of the chunk width C={C}")
    o = jnp.zeros((T, v.shape[1]), jnp.float32)
    s_cur = S
    for i in range(T // C):
        sl = slice(i * C, (i + 1) * C)
        o_i, s_cur = kda_cc_fused.reference(
            q[sl], k[sl], v[sl], alpha[sl], beta32[sl], s_cur, tile=tile)
        o = o.at[sl].set(o_i)
    return o, s_cur


def reference_bwd_scan(q, k, v, alpha, beta32, S, do, dS, *, tile=16, chunk=CHUNK):
    """Adjoint of :func:`reference` as a reverse scan over chunks (variant A).

    Two passes, both pure jnp:

    1. a *forward* scan recomputes the per-chunk incoming states ``S_i``
       (``kda_cc_fused.reference`` per chunk, the same function the tests pin
       to ``cc_chunk_step``) -- the whole-sequence ``custom_vjp`` saves only
       the inputs, so the states are rematerialised here;
    2. a *reverse* scan runs the milestone-2 analytic adjoint
       (:func:`net.kernels.kda_cc_fused_bwd.reference_bwd`) per chunk, passing
       the state cotangent ``dS`` from each chunk to its predecessor.

    Returns ``(dq, dk, dv, dalpha, dbeta32, dS_in)`` with primal dtypes --
    exactly the cotangent structure the per-chunk ``custom_vjp`` of milestone
    2 produced, now aligned with one forward launch per layer-pass and a
    multi-operation jnp backward (the honest boundary of variant A).
    """
    from . import kda_cc_fused, kda_cc_fused_bwd

    T = q.shape[0]
    C = int(chunk)
    if T % C:
        raise ValueError(f"T={T} must be a multiple of the chunk width C={C}")
    qc, kc, vc = _chunks(q, C), _chunks(k, C), _chunks(v, C)
    ac, bc = _chunks(alpha, C), _chunks(beta32, C)
    doc = _chunks(do, C)

    def _fwd(carry, xs):
        qf, kf, vf, af, bf = xs
        _o, s_new = kda_cc_fused.reference(qf, kf, vf, af, bf, carry, tile=tile)
        return s_new, carry

    _, s_ins = jax.lax.scan(_fwd, S, (qc, kc, vc, ac, bc))   # (n, H, dk, dv)

    def _bwd(ds_next, xs):
        qf, kf, vf, af, bf, dof, s_in = xs
        dq_f, dk_f, dv_f, da_f, db_f, ds_in = kda_cc_fused_bwd.reference_bwd(
            qf, kf, vf, af, bf, s_in, dof, ds_next, tile=tile)
        return ds_in, (dq_f, dk_f, dv_f, da_f, db_f)

    dS_in, (dq, dk_out, dv_out, dalpha, dbeta32) = jax.lax.scan(
        _bwd, dS, (qc, kc, vc, ac, bc, doc, s_ins), reverse=True)
    return (dq.reshape(q.shape), dk_out.reshape(k.shape), dv_out.reshape(v.shape),
            dalpha.reshape(alpha.shape), dbeta32.reshape(beta32.shape), dS_in)


# ---------------------------------------------------------------------------
# Differentiable wrapper (the whole-sequence ``custom_vjp``)
#
# Same wiring as milestone 2's per-chunk wrapper, moved to the whole scan: the
# forward rule runs the persistent kernel (one launch per layer-pass), the
# backward rule runs the reverse scan of the analytic adjoint.  Both rules
# resolve their implementations on the *module* at call time, so tests can
# observe or substitute which forward/backward runs.
# ---------------------------------------------------------------------------


def _scan_forward(q, k, v, alpha, beta32, S, tile, chunk):
    """Forward rule body -- the persistent kernel, resolved per call."""
    return kernel(q, k, v, alpha, beta32, S, tile=tile, chunk=chunk)


def _scan_backward(q, k, v, alpha, beta32, S, do, dS, tile, chunk):
    """Backward rule body -- the scan-of-chunks adjoint, resolved per call.

    The adjoint lives in the sibling modules (lazy import: this module stays
    importable without them; the attributes are looked up per call so tests
    can spy on the routing exactly like the milestone-2 wrapper's).
    """
    return reference_bwd_scan(q, k, v, alpha, beta32, S, do, dS,
                              tile=tile, chunk=chunk)


def make_cc_scan(tile=16, chunk=CHUNK, forward=None, backward=None):
    """Build the differentiable whole-sequence body ``(q, k, v, alpha, beta32, S)``.

    Returns a ``jax.custom_vjp`` function: forward = :func:`kernel` (or the
    ``forward`` substitution), backward = :func:`reference_bwd_scan` (or the
    ``backward`` substitution).  ``tile`` and ``chunk`` are **captured in the
    closure** and not arguments: ``custom_vjp`` is jit-like, so Python ints
    passed as arguments arrive inside the rules as weak-int tracers (the same
    ``TracerIntegerConversionError`` milestone 1 documented); capturing them
    keeps them host constants.  The substitutions exist for the CPU test suite
    (the kernel itself needs Mosaic GPU; the adjoint does not).
    """

    def _fwd(q, k, v, alpha, beta32, S):
        fn = forward if forward is not None else _scan_forward
        return fn(q, k, v, alpha, beta32, S, tile, chunk)

    def _bwd(q, k, v, alpha, beta32, S, do, dS):
        fn = backward if backward is not None else _scan_backward
        return fn(q, k, v, alpha, beta32, S, do, dS, tile, chunk)

    @jax.custom_vjp
    def scan(q, k, v, alpha, beta32, S):
        return _fwd(q, k, v, alpha, beta32, S)

    def scan_fwd(q, k, v, alpha, beta32, S):
        o, s_new = _fwd(q, k, v, alpha, beta32, S)
        return (o, s_new), (q, k, v, alpha, beta32, S)

    def scan_bwd(res, g):
        q, k, v, alpha, beta32, S = res
        do, dS = g
        dq, dk, dv, dalpha, dbeta32, dS_in = _bwd(
            q, k, v, alpha, beta32, S, do, dS)
        return dq, dk, dv, dalpha, dbeta32, dS_in

    scan.defvjp(scan_fwd, scan_bwd)
    return scan


#: Cache of the wrappers by static ``(tile, chunk)`` pair.  The wrappers
#: resolve ``kernel``/``reference_bwd_scan`` on their modules at call time, so
#: observing or substituting them keeps working through this cache.
_CC_SCANS: dict[tuple[int, int], object] = {}


def cc_scan(q, k, v, alpha, beta32, S, tile=16, chunk=CHUNK):
    """Differentiable entry point the ``AXIOM_KDA_CC_IMPL=persistent`` mode installs.

    ``jax.custom_vjp`` over the whole sequence: forward rule = the persistent
    kernel (one launch per layer-pass), backward rule = the reverse scan of
    the analytic per-chunk adjoint.  ``tile``/``chunk`` are *static* host
    arguments (the decay-factor tile width and the scan's chunk width); the
    wrapper for each pair is cached -- the host dispatch happens at trace
    time, before ``custom_vjp`` turns values into tracers.
    """
    key = (int(tile), int(chunk))
    scan = _CC_SCANS.get(key)
    if scan is None:
        scan = make_cc_scan(tile=key[0], chunk=key[1])
        _CC_SCANS[key] = scan
    return scan(q, k, v, alpha, beta32, S)
