"""Persistent MLA attention kernels (fwd + bwd) — Mosaic GPU, sm_121 (milestone 13).

Why this module exists (the measured disease of cycle-16)
---------------------------------------------------------
The NVTX attribution of ADR-054 Am.27 (stationary step): the blocking episodes
of the step sit in the *backward structure* — ``while.236`` (the VJP loop with
its nested remat recompute) is 4.46 s in ONE span, five checkpoint/remat spans
of 150-300 ms follow, and the jax-level flash-backward of milestone 12 pays a
host tile loop of ~12 s/step on top.  The KDA milestones 3-4 showed the cure
class: raise the unit of fusing to *one kernel launch per layer-pass*, the
state in SMEM, the whole sequence walked inside the program, instead of
per-chunk/per-tile boundaries.  This module is the MLA analogue: **one
persistent kernel for the forward of the attention interval and one for its
backward**, so neither the host tile loop nor an XLA while/remat structure
over the attention survives.

Scope: the attention interval ``(q, k, v) -> o`` of the dense MLA path
(``net/mla.py``), exactly the interval the milestone-12 flash-backward gated
(``AXIOM_MLA_FLASH_BWD``) — the dense-bwd semantics.  The layer's latent cache
``c = W_c x`` and the full-rank output gate ``y = W_o [sigmoid(W_g x) ⊙ õ]``
stay in jnp, per kda-formulas.md section 2; the kernels carry the T^2 coupling,
which is where the host/remat price lives.  The custom_vjp boundary is
``(q, k, v) -> o`` (no ``mask``: the persistent kernel implements the causal
case; an explicit mask is refused, not silently dropped).

Structure (per the KDA milestones 3-4)
--------------------------------------
* :func:`kernel` and :func:`kernel_bwd` — persistent Mosaic GPU kernels,
  ``grid=(B*H,)``: one program per (batch, head) walks the whole sequence
  inside itself.  Forward: query tiles outer, KV tiles inner, the online
  softmax state ``(O, m, l)`` carried in ACC registers between iterations,
  per-row statistics written once.  Backward: KV tiles outer, query tiles
  inner; ``dk``/``dv`` accumulate in ACC registers and are written once per KV
  tile (complete in one visit); ``dq`` contributions are read-modify-written
  into the GMEM output (the one quantity the KV-outer schedule cannot hold;
  each (batch, head) owns disjoint slices, so the RMW is program-private and
  ordered by the language semantics — no atomics, no cross-program races).
* :func:`reference_attention` — the fp32 jnp replica of the same interval (the
  exact semantics: full-row softmax, fp32 throughout).
* :func:`emulate_attention` / :func:`emulate_attention_bwd` — the *declared
  numerics* of the kernel design: bf16 operand casts at exactly the boundary
  points the kernels cast (q/k/v loads, the ``p`` staging, the ``ds`` staging,
  the ``do`` cast), fp32 accumulation everywhere else.  The CPU tests pin the
  drift term of these roundings against :func:`reference_attention`.
* :func:`mla_attention` — the differentiable entry point the
  ``AXIOM_MLA_PERSIST=1`` mode installs in ``net/mla.py`` (a
  ``jax.custom_vjp``: forward rule = the persistent kernel, backward rule =
  the persistent backward kernel, one launch per direction per layer-pass).

Declared numerics (the operand budget, stated before the stand)
---------------------------------------------------------------
The GB10 ``plgpu.mma`` path is bf16-operand / fp32-accumulate (the operand path
validated on the stand by the KDA milestone probes); the persistent kernels
cannot be more precise than their operands.  The declared rounding points of
this design, in the order of the data flow:

1. the ``q``/``k``/``v`` loads are cast bf16 at the kernel boundary (the same
   operand quantization the ``AXIOM_COMPUTE_DTYPE=bf16`` mode of
   ``net/compute_dtype.py`` applies to the dense GEMM operands; the cast lives
   *inside* the ``custom_vjp`` rule, so the op boundary itself stays fp32 —
   the ``gemm_einsum`` shape);
2. the softmax numerator ``p`` (forward: ``exp(s-m)``; backward:
   ``exp(s-m)/l``) is staged as a **bf16 hi/lo pair** (``p_hi = bf16(p)``,
   ``p_lo = bf16(p - p_hi)``, two accumulating mmas — the KDA milestone-4
   hi/lo idiom): a plain single-bf16 staging would leave the kernel's own
   staging term at the 1e-3 boundary, the pair resolves it to ~2^-17;
3. the backward score cotangent ``ds`` is staged the same way (two mmas per
   cotangent site);
4. the ``do`` cotangent of ``o`` is cast bf16 at the kernel boundary (declared
   operand);
5. the backward cotangents ``dq``/``dk``/``dv`` are fp32 accumulations and
   leave the ``custom_vjp`` rule in fp32 (no boundary re-rounding).

Everything else — score accumulators, the online-softmax algebra, the
statistics, the ``D_i = rowsum(do ⊙ o)`` reduction, the cotangent
accumulations — is fp32.

Measured consequences (pinned in ``net/tests/test_44_mla_persist.py``):

* **the design's algebra is fp32-exact**: the tiled forward against the dense
  fp32 attention is ~7e-8 relative (bit-level fp32 reordering only), and the
  declared-numerics adjoint against ``jax.grad`` of the same forward is
  ~1e-7 -- both far inside the milestone's ``<=1e-3`` budget.  With the hi/lo
  staging and bf16-representable operands the whole interval sits at the fp32
  rounding level;
* **the operand-path term** (the remaining >1e-3 number) is the bf16
  *quantization of the inputs* (``q``/``k``/``v`` as they come out of the fp32
  projections, and ``do``): at the layer level (``net.mla`` dense reference,
  mini configuration) the forward measures ~1.0e-3..2.0e-3 and the gradients
  ~1.5e-3..6.5e-3 relative depending on the compute-dtype mode -- the size of
  a bf16 rounding of the operands (~2^-9), i.e. the same class the campaign's
  own bf16 mode carries (milestone 12 documented ~3e-3 for the bf16-mode
  gradients of the flash-backward).  The number is *documented, not hidden*:
  the tests print the per-mode table.  The levers that would remove it are
  declared follow-ups (hi/lo for the input operands -- 2x streaming and mma
  sites; or an f16 operand path -- never probed on the GB10 stand).

SMEM budget (the declared floor; the measured numbers are pinned by tests)
--------------------------------------------------------------------------
One block's persistent SMEM (everything the loop body reuses; bf16 unless
noted):

======================  ===========================  =====================
buffer                  forward                      backward
======================  ===========================  =====================
K tile                  ``bk x dq``                  ``bk x dq``
V tile                  ``bk x dq``                  ``bk x dq``
Q tile                  (loaded GMEM->LHS directly)   ``bq x dq``
do tile                 —                            ``bq x dq``
``p`` hi/lo pair        ``bq x bk`` x2               ``bq x bk`` x2
``ds`` hi/lo pair       —                            ``bq x bk`` x2
======================  ===========================  =====================

The causal pattern is built in registers (a direct ``layout_cast`` of the iota
product into the accumulator layout; the ``WG_STRIDED`` intermediate does not
satisfy the constraint system -- measured) — no pattern buffer.  At the
campaign tiling ``bq = bk = 64``, ``dq = 128``: forward 50 KiB, backward
96 KiB — both under the 99 KiB GB10 block budget.  Wider tiles are sieved by
arithmetic (:func:`valid_config`, :func:`valid_config_bwd`), not by hope; the
surviving/skipped cases are pinned in the tests.

Verification status (honest)
----------------------------
* the kernels *lower* for sm_121 under the skill's patched lowering
  (``lower_check.py``, no GPU needed) in the module's live cases; the backward
  kernel's companion contract module is ``net/tests/mla_persist_bwd_lower.py``;
* the declared numerics (:func:`emulate_attention`,
  :func:`emulate_attention_bwd`) are pinned on CPU against the fp32
  :func:`reference_attention` at the campaign geometry and against the real
  dense interval of ``net/mla.py`` (parity, measured numbers in the tests);
* **the kernels are not executed on this node** (CPU-only; jax 0.11.2's GPU
  interpret mode has no rule for ``plgpu.mma``).  Device parity, timing,
  determinism and the per-pass launch count of a real GB10 run are the stand
  leg of the milestone (``check_kernel.py``); ``open_questions`` lists them.
"""

from __future__ import annotations

import functools

import numpy as np

import jax
import jax.numpy as jnp

from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

# ---------------------------------------------------------------------------
# Module contract (pallas-gb10-kernel skill)
# ---------------------------------------------------------------------------

#: Case shapes.  ``B``/``T``/``H``/``dq`` are the data geometry; ``bq``/``bk``
#: are the static tiles (they equal the kernel defaults, so the default-mode
#: lowering runs exactly what the case declares).  Entries 0-1 are the live
#: campaign forms (l3-full: H=12, dq=128, T=8192, B=2 and B=4); entry 2 is the
#: CPU-cheap live case of the test suite.  The oversize geometry (``bk=128``:
#: the backward SMEM floor is over the GB10 budget) is *skipped* by the sieve
#: in the ``--configs all`` sweep — see :data:`TUNE_SPACE` and
#: :func:`valid_config_bwd`; it is pinned as a skip in the tests.
CASES = [
    dict(B=2, T=8192, H=12, dq=128, bq=64, bk=64, dtype="bfloat16"),
    dict(B=4, T=8192, H=12, dq=128, bq=64, bk=64, dtype="bfloat16"),
    dict(B=1, T=256, H=2, dq=128, bq=64, bk=64, dtype="bfloat16"),
]

#: bf16-operand / fp32-accumulator path; the fp32 path is the exact reference.
TOLERANCE = {"bfloat16": (1e-2, 1e-2), "float32": (1e-5, 1e-5)}

#: Static knobs of the kernels: the tile widths.  ``lower_check --configs all``
#: sweeps them through the sieve; only the (64, 64) geometry survives for the
#: *pair* of kernels (see the floors) — the sweep therefore doubles as a pin
#: of the honest skips.
TUNE_SPACE = dict(bk=[64, 128])

#: GB10 SMEM budget of one block (99 KiB — the value the skill's lower_check
#: patches into the Mosaic lowering on a GPU-less node).
SMEM_LIMIT_BYTES = 101376

#: The lowering's own overhead beyond the arithmetic sum of the declared
#: scratch buffers (measured on this node with the artificial limit: +2048 B
#: on the forward kernel, +0 on the backward one; the floor keeps the larger
#: of the two for both kernels, so the sieve stays conservative but never
#: false-rejects a geometry the lowering accepts — the milestone-13 lesson of
#: the first floor draft, which carried a 4 KiB epilogue margin and rejected
#: the live backward tiling that in fact lowers at 98 304 B).
SMEM_OVERHEAD_BYTES = 2 * 1024

#: Default tile widths (campaign value; the sieve admits other multiples).
BQ = 64
BK = 64


def _attn_scale(dq: int) -> np.float32:
    """``1/sqrt(dq)`` as the f32 constant the dense path multiplies scores by.

    Computed with f32 IEEE ops (the same correctly-rounded values
    ``1.0 / jnp.sqrt(dq)`` produces in the dense path) but as a *host* value,
    so the kernel builders and the emulation stay traceable: a jnp scalar
    computed inside a trace is a tracer and cannot cross the Python-level
    branches (the ConcretizationTypeError this helper exists to avoid).
    """
    return np.float32(1.0) / np.sqrt(np.float32(dq))


def _smem_floor_bytes_fwd(bq, bk, dq, itemsize=2):
    """Declared SMEM floor of the forward kernel (see the module docstring).

    Persistent buffers: ``k_s``/``v_s`` ``bk x dq`` and the ``p`` hi/lo pair
    ``bq x bk`` x2.  The Q tile is loaded GMEM->LHS directly (the admitted
    GMEM load form), the causal pattern is built in registers, the
    online-softmax state lives in ACC registers.
    """
    return (2 * bk * dq + 2 * bq * bk) * itemsize + SMEM_OVERHEAD_BYTES


def _smem_floor_bytes_bwd(bq, bk, dq, itemsize=2):
    """Declared SMEM floor of the backward kernel (see the module docstring).

    Persistent buffers: ``k_s``/``v_s`` ``bk x dq``, ``q_s``/``do_s``
    ``bq x dq``, the ``p`` and ``ds`` hi/lo pairs ``bq x bk`` x4.  The
    ``dk``/``dv`` accumulators are ACC registers (fp32); ``dq`` is a GMEM RMW.
    """
    return (2 * bk * dq + 2 * bq * dq + 4 * bq * bk) * itemsize + SMEM_OVERHEAD_BYTES


def _tile_rules_ok(bq, bk):
    """The sm_121 mma tiling rules of the skill, as a predicate.

    ``bm % 64 == 0`` for the warpgroup MMA tile (both widths are an ``M`` side
    somewhere: ``bq`` for the score rows, ``bk`` for the ``dk``/``dv``
    accumulators), ``bn <= 128`` for the width that lands on an ``N`` side,
    ``bk`` a multiple of 16 (the bf16 mma K step).
    """
    return (bq >= 64 and bq % 64 == 0 and 16 <= bk <= 128 and bk % 16 == 0)


def _tiles_from(case, params):
    params = params or {}
    bq = int(params.get("bq", case.get("bq", BQ)))
    bk = int(params.get("bk", case.get("bk", BK)))
    return bq, bk


def valid_config(case, params=None):
    """Static sieve: tile rules + the SMEM floors of *both* kernels.

    The live set of the module is the *pair's* set: :func:`kernel` and
    :func:`kernel_bwd` each refuse a geometry where either floor exceeds the
    GB10 budget (no silent fallback), so the sweep of ``lower_check.py``
    honest-skips exactly those cases (the KDA-persistence contract).  Tile
    widths in ``params`` override the case fields (the sweep form).
    """
    bq, bk = _tiles_from(case, params)
    if not _tile_rules_ok(bq, bk):
        return False
    dq = int(case["dq"])
    return (_smem_floor_bytes_fwd(bq, bk, dq) <= SMEM_LIMIT_BYTES
            and _smem_floor_bytes_bwd(bq, bk, dq) <= SMEM_LIMIT_BYTES)


def valid_config_bwd(case, params=None):
    """Static sieve of the backward kernel (the same pair rule as above)."""
    bq, bk = _tiles_from(case, params)
    if not _tile_rules_ok(bq, bk):
        return False
    dq = int(case["dq"])
    return (_smem_floor_bytes_fwd(bq, bk, dq) <= SMEM_LIMIT_BYTES
            and _smem_floor_bytes_bwd(bq, bk, dq) <= SMEM_LIMIT_BYTES)


def cost(B, T, H, dq, bq=64, bk=64, dtype="bfloat16", **kw):
    """Minimal bytes moved by the forward interval (roofline bookkeeping)."""
    b = jnp.dtype(dtype).itemsize
    f = jnp.dtype(jnp.float32).itemsize
    reads = 3 * B * T * H * dq * b                     # q, k, v
    writes = B * T * H * dq * f + 2 * B * H * T * f    # o, rowmax, rowsum
    return dict(flops=2 * B * H * (T * T // 2) * dq, bytes=reads + writes)


def make_inputs(key, B, T, H, dq, bq=64, bk=64, dtype="bfloat16", **kw):
    """Forward operands: ``(q, k, v)`` bf16 ``(B, T, H, dq)``.

    Magnitudes follow the projection boundaries of the model (``0.02``-scale
    weights over ``hidden = 1536``: k/v ~ 0.3-0.8); the scale is part of the
    fixture so the parity tests measure the drift term at the geometry it will
    see, not at unit variance.
    """
    dt = jnp.dtype(dtype)
    k1, k2, k3 = jax.random.split(key, 3)
    s = 0.6
    q = (jax.random.normal(k1, (B, T, H, dq), jnp.float32) * s).astype(dt)
    k = (jax.random.normal(k2, (B, T, H, dq), jnp.float32) * s).astype(dt)
    v = (jax.random.normal(k3, (B, T, H, dq), jnp.float32) * s).astype(dt)
    return q, k, v


def make_inputs_bwd(key, B, T, H, dq, bq=64, bk=64, dtype="bfloat16", **kw):
    """Backward operands: ``(q, k, v, do, D, m, l)``.

    ``D``/``m``/``l`` carry the forward's statistics at the same geometry
    (``D`` = the per-row ``rowsum(do ⊙ o)``; here a plausible fixture — the
    parity tests recompute the real ones).
    """
    q, k, v = make_inputs(key, B, T, H, dq, bq=bq, bk=bk, dtype=dtype)
    k1 = jax.random.fold_in(key, 1)
    do = (jax.random.normal(k1, (B, T, H, dq), jnp.float32) * 0.3).astype(dtype)
    B_, H_, T_ = B, H, T
    return (q, k, v, do, jnp.zeros((B_, H_, T_), jnp.float32),
            jnp.zeros((B_, H_, T_), jnp.float32), jnp.ones((B_, H_, T_), jnp.float32))


def device_available() -> bool:
    """True when a GPU platform device is visible to JAX.

    The kernel's *execution* needs Mosaic GPU; without a device JAX raises at
    compile time, so the mux in ``net.mla`` refuses early with a typed error
    instead of letting the XLA/pallas error escape.  The *lowering* does not
    need a device (``lower_check`` runs exactly like that).
    """
    try:
        return any(d.platform == "gpu" for d in jax.devices())
    except Exception:  # noqa: BLE001 — a broken plugin must not read as "GPU present"
        return False


# ---------------------------------------------------------------------------
# The kernels (one program per (batch, head), the whole sequence inside)
# ---------------------------------------------------------------------------


def _mma_m_warps(m: int) -> int:
    """Warp-group width of the MMA layouts for an accumulator with ``m`` rows.

    Lowering fact (the KDA milestones, this node, jax 0.11.2, sm_121): the
    default ``m_warps=4`` tiling is a 64-row tile; shapes with ``m < 64`` need
    ``m_warps=1``.  The live tiling is 64/64, so this only matters for the
    sieve's smaller shapes.
    """
    return 1 if m < 64 else 4


def _iota(shape, axis):
    """An int32 index grid in the layout the SMEM pattern stores use (KDA idiom)."""
    return plgpu.layout_cast(
        plgpu.broadcasted_iota(jnp.int32, shape, axis),
        plgpu.Layout.WG_STRIDED(shape, 2))


def _load(ref, layout):
    return plgpu.load(ref, layout=layout, optimized=False)


def _zero(shape, acc_layout):
    return plgpu.layout_cast(jnp.zeros(shape, jnp.float32), acc_layout)


@functools.lru_cache(maxsize=8)
def _build_kernel(B, T, H, dq, bq, bk, dtype):
    """Build the persistent forward kernel: one program per (batch, head).

    Per program the body walks the query tiles (outer dynamic ``pl.loop``) and
    per query tile the KV tiles ``j <= i`` (inner dynamic ``pl.loop`` with the
    online-softmax state ``(O, m, l)`` as the loop carry).  Full tiles
    (``j < i``) need no mask by tile arithmetic (the minimum row of frame
    ``i`` is >= the maximum column of every frame ``j < i``); the diagonal
    tile carries the static causal pattern.  ``o`` (fp32) and the per-row
    statistics ``rowmax``/``rowsum`` (fp32 ``(B, H, T)``) are written once per
    row.

    Cached by the static spec (primitives only -- the tuple *is* the key), the
    same invariant the KDA persist module states: one spec -> one built body
    and one jitted run per geometry instead of a rebuild per call.
    """
    dt = jnp.dtype(dtype)
    f32 = jnp.float32
    nq = T // bq
    scale = float(_attn_scale(dq))
    mw_q, mw_k = _mma_m_warps(bq), _mma_m_warps(bk)
    ACC_Q = plgpu.Layout.MMA_ACC(dt, m_warps=mw_q)
    ACC_K = plgpu.Layout.MMA_ACC(dt, m_warps=mw_k)
    L_Q = plgpu.Layout.MMA_LHS(dt, m_warps=mw_q)
    R_Q = plgpu.Layout.MMA_RHS(dt, m_warps=mw_q)

    def _body(qg, kg, vg, og, mg, lg, k_s, v_s, p_s, pl_s):
        prog = jax.lax.axis_index("program")
        b, h = prog // H, prog % H
        # The static causal pattern is built in registers (direct layout_cast
        # of the iota product into the accumulator layout -- the WG_STRIDED
        # intermediate does not satisfy the constraint system, measured).
        ridx = plgpu.broadcasted_iota(jnp.int32, (bq, bk), 0)
        cidx = plgpu.broadcasted_iota(jnp.int32, (bq, bk), 1)
        mk = plgpu.layout_cast((ridx >= cidx).astype(f32), ACC_Q)

        def _step(qv, j, carry, diag):
            """One (query tile, KV tile) update of the online softmax.

            ``s = q k^T * scale`` is the mma of the bf16 operands into the fp32
            accumulator; masked entries carry ``finfo.min`` exactly like the
            dense path's ``where``, so ``exp(s - m_new)`` is 0 there and the
            running statistics converge to the dense full-row statistics.
            ``O`` is rescaled by ``exp(m_old - m_new)`` before the ``p @ v``
            accumulation.  ``p`` is staged as a bf16 *hi/lo pair* (the KDA
            milestone-4 idiom): ``p_hi = bf16(p)``, ``p_lo = bf16(p - p_hi)``,
            and the two mmas accumulate both parts -- a plain bf16 rounding of
            ``p`` alone would put the operand drift at the 1e-3 boundary (the
            measured numbers are in the tests).
            """
            acc, m, l = carry
            k_s[...] = _load(kg.at[b, pl.ds(j * bk, bk), h, pl.ds(0, dq)], ACC_K).astype(dt)
            v_s[...] = _load(vg.at[b, pl.ds(j * bk, bk), h, pl.ds(0, dq)], ACC_K).astype(dt)
            s = plgpu.mma(_zero((bq, bk), ACC_Q), qv, _load(k_s.T, R_Q))
            s = plgpu.layout_cast(s * scale, ACC_Q)
            if diag:
                sm = jnp.where(mk > 0, s, jnp.finfo(f32).min)
            else:
                sm = s
            rm = jnp.max(sm, axis=1)
            mnew = jnp.maximum(m, rm)
            alpha = jnp.exp(m - mnew)
            mb = plgpu.layout_cast(
                jax.lax.broadcast_in_dim(mnew, (bq, bk), (0,)), ACC_Q)
            p = jnp.exp(sm - mb)
            l = l * alpha + jnp.sum(p, axis=1)
            p_hi = p.astype(dt)
            p_lo = (p - p_hi.astype(f32)).astype(dt)
            p_s[...] = p_hi
            pl_s[...] = p_lo
            ab = plgpu.layout_cast(
                jax.lax.broadcast_in_dim(alpha, (bq, dq), (0,)), ACC_Q)
            acc = plgpu.mma(plgpu.layout_cast(acc * ab, ACC_Q),
                            _load(p_s, L_Q), _load(v_s, R_Q))
            acc = plgpu.mma(acc, _load(pl_s, L_Q), _load(v_s, R_Q))
            return acc, mnew, l

        def _tile(i):
            qv = _load(qg.at[b, pl.ds(i * bq, bq), h, pl.ds(0, dq)], L_Q)
            acc0 = _zero((bq, dq), ACC_Q)
            m0 = plgpu.layout_cast(
                jnp.full((bq,), jnp.finfo(f32).min, f32), ACC_Q.reduce(1))
            l0 = plgpu.layout_cast(jnp.zeros((bq,), f32), ACC_Q.reduce(1))

            def _inner(j, carry):
                return _step(qv, j, carry, False)

            acc, m, l = pl.loop(0, i, init_carry=(acc0, m0, l0))(_inner)
            acc, m, l = _step(qv, i, (acc, m, l), True)
            lb = plgpu.layout_cast(jax.lax.broadcast_in_dim(l, (bq, dq), (0,)), ACC_Q)
            og.at[b, pl.ds(i * bq, bq), h, pl.ds(0, dq)][...] = acc / lb
            mg.at[b, h, pl.ds(i * bq, bq)][...] = m
            lg.at[b, h, pl.ds(i * bq, bq)][...] = l

        pl.loop(0, nq)(_tile)

    return plgpu.kernel(
        _body,
        out_type=(
            jax.ShapeDtypeStruct((B, T, H, dq), f32),   # o
            jax.ShapeDtypeStruct((B, H, T), f32),       # rowmax
            jax.ShapeDtypeStruct((B, H, T), f32),       # rowsum
        ),
        scratch_types=[
            plgpu.SMEM((bk, dq), dt),   # k_s
            plgpu.SMEM((bk, dq), dt),   # v_s
            plgpu.SMEM((bq, bk), dt),   # p_s (hi)
            plgpu.SMEM((bq, bk), dt),   # p_lo
        ],
        grid=(B * H,),
        grid_names=("program",),
    )


@functools.lru_cache(maxsize=8)
def _forward_run(B, T, H, dq, bq, bk, dtype):
    """The jitted persistent forward run of one static spec.

    One spec -> one built body AND one ``jax.jit`` wrapper (the KDA cache
    contract: the milestone-3 profile measured the re-build per call as the
    barrier; a cached call is a lookup).
    """
    return jax.jit(_build_kernel(B, T, H, dq, bq, bk, dtype))


# ===PART_BWD===
def kernel(q, k, v, *, bq=BQ, bk=BK):
    """Persistent forward: ``(q, k, v) -> (o, rowmax, rowsum)``, one launch.

    ``q``/``k``/``v``: ``(B, T, H, dq)`` bf16 — the declared mma operand path
    (the caller casts; see :func:`mla_attention`).  ``T`` must be a multiple of
    ``bq`` and ``bk`` (the caller pads; the :func:`mla_attention` wrapper
    does).  Returns ``o`` ``(B, T, H, dq)`` fp32 and the per-row softmax
    statistics ``rowmax``/``rowsum`` ``(B, H, T)`` fp32 — exactly the interval
    the dense causal attention covers.

    Execution needs Mosaic GPU (GB10); the *lowering* runs anywhere
    (``lower_check.py`` on this CPU node).
    """
    B, T, H, dq = q.shape
    if k.shape != q.shape or v.shape != q.shape:
        raise ValueError(
            f"persistent MLA kernel: q/k/v shapes must match; got "
            f"{q.shape}, {k.shape}, {v.shape}")
    dt = jnp.dtype(q.dtype)
    if dt != jnp.bfloat16 or k.dtype != dt or v.dtype != dt:
        raise ValueError(
            "persistent MLA kernel consumes bf16 q/k/v (the declared mma "
            "operand path; the caller casts — see mla_attention); got "
            f"q={dt}, k={k.dtype}, v={v.dtype}")
    if T % int(bq) or T % int(bk):
        raise ValueError(
            f"T={T} must be a multiple of the tile widths bq={bq}, bk={bk} "
            "(the wrapper pads, see mla_attention)")
    case = dict(B=B, T=T, H=H, dq=dq, bq=int(bq), bk=int(bk), dtype=dt.name)
    if not (valid_config(case) and valid_config_bwd(case)):
        raise ValueError(
            f"persistent MLA kernel: configuration {case} is not among the "
            "live cases of the sieve (mma tiling rules and the GB10 SMEM "
            "floor — see valid_config/valid_config_bwd); no silent fallback")
    run = _forward_run(B, T, H, dq, int(bq), int(bk), dt.name)
    return run(q, k, v)


# ---------------------------------------------------------------------------
# The backward kernel (one program per (batch, head), the reverse walk inside)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=8)
def _build_bwd_kernel(B, T, H, dq, bq, bk, dtype):
    """Build the persistent backward kernel: one program per (batch, head).

    KV tiles outer, query tiles ``i >= j`` inner.  Per pair the score is
    *recomputed* (``s = q k^T * scale``, the bf16 operand path, exactly the
    forward's tile algebra), ``p`` is rebuilt from the forward's statistics
    (``p = exp(s - m_i) / l_i``, masked entries 0 by ``finfo.min``), and the
    three cotangent contributions are formed:

    * ``dk_j += ds^T q_i`` and ``dv_j += p^T do_i`` accumulate in ACC
      registers and are written once per KV tile (complete in one visit of
      the outer loop);
    * ``dq_i += ds k_j`` is read-modify-written into the GMEM output —
      program-private (each (batch, head) owns its slices), ordered by the
      language semantics.  The ``dq`` slices of the program are zeroed by the
      program itself before the walk (``out_type`` buffers start
      uninitialized; the zeroing pass is the declared cost of the RMW design —
      it removes the need for atomics or a second launch).

    ``D`` (the per-row ``rowsum(do ⊙ o)``), ``m``/``l`` (the forward's
    statistics) are inputs computed outside (the milestone-12 flash-backward
    reads the same quantities).  ``do`` is cast bf16 at the kernel boundary
    (declared); the cotangents are fp32 accumulators.

    Cached by the static spec, like :func:`_build_kernel`.
    """
    dt = jnp.dtype(dtype)
    f32 = jnp.float32
    nq, nk = T // bq, T // bk
    scale = float(_attn_scale(dq))
    mw_q, mw_k = _mma_m_warps(bq), _mma_m_warps(bk)
    ACC_Q = plgpu.Layout.MMA_ACC(dt, m_warps=mw_q)
    ACC_K = plgpu.Layout.MMA_ACC(dt, m_warps=mw_k)
    L_Q = plgpu.Layout.MMA_LHS(dt, m_warps=mw_q)
    L_K = plgpu.Layout.MMA_LHS(dt, m_warps=mw_k)
    R_Q = plgpu.Layout.MMA_RHS(dt, m_warps=mw_q)
    R_K = plgpu.Layout.MMA_RHS(dt, m_warps=mw_k)

    def _body(qg, kg, vg, dog, Dg, mg, lg,
              dqg, dkg, dvg,
              k_s, v_s, q_s, do_s, p_s, pl_s, ds_s, dsl_s):
        prog = jax.lax.axis_index("program")
        b, h = prog // H, prog % H
        # The static causal pattern is built in registers (direct layout_cast
        # of the iota product into the accumulator layout, the forward
        # kernel's idiom).
        ridx = plgpu.broadcasted_iota(jnp.int32, (bq, bk), 0)
        cidx = plgpu.broadcasted_iota(jnp.int32, (bq, bk), 1)
        mk = plgpu.layout_cast((ridx >= cidx).astype(f32), ACC_Q)

        # Zero the program's dq slices (the RMW accumulator; out_type memory
        # starts uninitialized).  Every (batch, head) owns one column block,
        # so the zeroing is program-private like the accumulation.
        def _zero_dq(i):
            dqg.at[b, pl.ds(i * bq, bq), h, pl.ds(0, dq)][...] = jnp.zeros(
                (bq, dq), f32)

        pl.loop(0, nq)(_zero_dq)

        def _step(i, j, carry, diag):
            """One (query tile i, KV tile j) pair of the adjoint."""
            dk_acc, dv_acc = carry
            q_s[...] = _load(qg.at[b, pl.ds(i * bq, bq), h, pl.ds(0, dq)], ACC_Q).astype(dt)
            do_s[...] = _load(dog.at[b, pl.ds(i * bq, bq), h, pl.ds(0, dq)], ACC_Q).astype(dt)
            s = plgpu.mma(_zero((bq, bk), ACC_Q), _load(q_s, L_Q), _load(k_s.T, R_Q))
            s = plgpu.layout_cast(s * scale, ACC_Q)
            if diag:
                sm = jnp.where(mk > 0, s, jnp.finfo(f32).min)
            else:
                sm = s
            mi = _load(mg.at[b, h, pl.ds(i * bq, bq)], ACC_Q.reduce(1))
            li = _load(lg.at[b, h, pl.ds(i * bq, bq)], ACC_Q.reduce(1))
            Di = _load(Dg.at[b, h, pl.ds(i * bq, bq)], ACC_Q.reduce(1))
            mb = plgpu.layout_cast(jax.lax.broadcast_in_dim(mi, (bq, bk), (0,)), ACC_Q)
            lb = plgpu.layout_cast(jax.lax.broadcast_in_dim(li, (bq, bk), (0,)), ACC_Q)
            db = plgpu.layout_cast(jax.lax.broadcast_in_dim(Di, (bq, bk), (0,)), ACC_Q)
            p = jnp.exp(sm - mb) / lb          # masked entries: exp(min - m) = 0
            dp = plgpu.mma(_zero((bq, bk), ACC_Q), _load(do_s, L_Q), _load(v_s.T, R_Q))
            ds = plgpu.layout_cast((p * (dp - db)) * scale, ACC_Q)
            # hi/lo bf16 staging of both mma operands (the forward kernel's
            # idiom): the plain single-bf16 rounding would put the gradient
            # drift at the 2-4e-3 level (measured; the tests pin the numbers).
            p_hi = p.astype(dt)
            p_lo = (p - p_hi.astype(f32)).astype(dt)
            ds_hi = ds.astype(dt)
            ds_lo = (ds - ds_hi.astype(f32)).astype(dt)
            p_s[...] = p_hi
            pl_s[...] = p_lo
            ds_s[...] = ds_hi
            dsl_s[...] = ds_lo
            dk_acc = plgpu.mma(dk_acc, _load(ds_s.T, L_K), _load(q_s, R_K))
            dk_acc = plgpu.mma(dk_acc, _load(dsl_s.T, L_K), _load(q_s, R_K))
            dv_acc = plgpu.mma(dv_acc, _load(p_s.T, L_K), _load(do_s, R_K))
            dv_acc = plgpu.mma(dv_acc, _load(pl_s.T, L_K), _load(do_s, R_K))
            accq = _load(dqg.at[b, pl.ds(i * bq, bq), h, pl.ds(0, dq)], ACC_Q)
            accq = plgpu.mma(accq, _load(ds_s, L_Q), _load(k_s, R_Q))
            accq = plgpu.mma(accq, _load(dsl_s, L_Q), _load(k_s, R_Q))
            dqg.at[b, pl.ds(i * bq, bq), h, pl.ds(0, dq)][...] = accq
            return dk_acc, dv_acc

        def _outer(j):
            k_s[...] = _load(kg.at[b, pl.ds(j * bk, bk), h, pl.ds(0, dq)], ACC_K).astype(dt)
            v_s[...] = _load(vg.at[b, pl.ds(j * bk, bk), h, pl.ds(0, dq)], ACC_K).astype(dt)
            dk0 = _zero((bk, dq), ACC_K)
            dv0 = _zero((bk, dq), ACC_K)
            dk_acc, dv_acc = _step(j, j, (dk0, dv0), True)

            def _inner(i, carry):
                return _step(i, j, carry, False)

            dk_acc, dv_acc = pl.loop(j + 1, nq, init_carry=(dk_acc, dv_acc))(_inner)
            dkg.at[b, pl.ds(j * bk, bk), h, pl.ds(0, dq)][...] = dk_acc
            dvg.at[b, pl.ds(j * bk, bk), h, pl.ds(0, dq)][...] = dv_acc

        pl.loop(0, nk)(_outer)

    return plgpu.kernel(
        _body,
        out_type=(
            jax.ShapeDtypeStruct((B, T, H, dq), f32),   # dq (RMW-accumulated)
            jax.ShapeDtypeStruct((B, T, H, dq), f32),   # dk (written per KV tile)
            jax.ShapeDtypeStruct((B, T, H, dq), f32),   # dv
        ),
        scratch_types=[
            plgpu.SMEM((bk, dq), dt),   # k_s
            plgpu.SMEM((bk, dq), dt),   # v_s
            plgpu.SMEM((bq, dq), dt),   # q_s
            plgpu.SMEM((bq, dq), dt),   # do_s
            plgpu.SMEM((bq, bk), dt),   # p_s (hi)
            plgpu.SMEM((bq, bk), dt),   # p_lo
            plgpu.SMEM((bq, bk), dt),   # ds_s (hi)
            plgpu.SMEM((bq, bk), dt),   # ds_lo
        ],
        grid=(B * H,),
        grid_names=("program",),
    )


@functools.lru_cache(maxsize=8)
def _backward_run(B, T, H, dq, bq, bk, dtype):
    """The jitted persistent backward run of one static spec (see the cache note)."""
    return jax.jit(_build_bwd_kernel(B, T, H, dq, bq, bk, dtype))


def kernel_bwd(q, k, v, do, D, m, l, *, bq=BQ, bk=BK):
    """Persistent backward: ``(q, k, v, do, D, m, l) -> (dq, dk, dv)``, one launch.

    ``q``/``k``/``v``/``do``: ``(B, T, H, dq)`` bf16 (the declared operand
    path).  ``D``: ``rowsum(do ⊙ o)`` per row, ``m``/``l``: the forward's
    ``rowmax``/``rowsum`` — all fp32 ``(B, H, T)``, computed outside exactly
    like the milestone-12 flash-backward consumes them.  Returns fp32
    cotangents ``(B, T, H, dq)`` (the custom_vjp boundary casts them to the
    bf16 primal dtype — see :func:`mla_attention`).

    Execution needs Mosaic GPU (GB10); the *lowering* runs anywhere
    (``net/tests/mla_persist_bwd_lower.py`` + ``lower_check.py``).
    """
    B, T, H, dq = q.shape
    if k.shape != q.shape or v.shape != q.shape or do.shape != q.shape:
        raise ValueError(
            f"persistent MLA backward kernel: q/k/v/do shapes must match; got "
            f"{q.shape}, {k.shape}, {v.shape}, {do.shape}")
    if (D.shape != (B, H, T) or m.shape != (B, H, T) or l.shape != (B, H, T)):
        raise ValueError(
            f"D/m/l must be fp32 (B,H,T)={(B, H, T)}; got {D.shape}, {m.shape}, "
            f"{l.shape}")
    dt = jnp.dtype(q.dtype)
    if dt != jnp.bfloat16 or k.dtype != dt or v.dtype != dt or do.dtype != dt:
        raise ValueError(
            "persistent MLA backward kernel consumes bf16 q/k/v/do (the "
            f"declared mma operand path); got q={dt}, k={k.dtype}, v={v.dtype}, "
            f"do={do.dtype}")
    if T % int(bq) or T % int(bk):
        raise ValueError(
            f"T={T} must be a multiple of the tile widths bq={bq}, bk={bk} "
            "(the wrapper pads, see mla_attention)")
    case = dict(B=B, T=T, H=H, dq=dq, bq=int(bq), bk=int(bk), dtype=dt.name)
    if not (valid_config(case) and valid_config_bwd(case)):
        raise ValueError(
            f"persistent MLA backward kernel: configuration {case} is not among "
            "the live cases of the sieve (mma tiling rules and the GB10 SMEM "
            "floor — see valid_config/valid_config_bwd); no silent fallback")
    run = _backward_run(B, T, H, dq, int(bq), int(bk), dt.name)
    return run(q, k, v, do, D, m, l)


# ---------------------------------------------------------------------------
# The fp32 reference and the declared-numerics emulation (CPU-side pins)
# ---------------------------------------------------------------------------


def reference_attention(q, k, v):
    """The exact semantics of the interval: full-row causal softmax in fp32.

    This is the fp32 replica of ``(q, k, v) -> (o, rowmax, rowsum)`` — what the
    kernel's tiled online-softmax algebra approximates.  ``q``/``k``/``v`` are
    consumed as given (callers pass bf16 — that is the declared operand
    rounding of the *design*; the emulation models the tile algebra on top of
    it).  Returns ``o`` ``(B,T,H,dq)`` fp32 and the statistics ``(B,H,T)`` fp32
    in the dense convention (``m`` = max of the masked row, ``l`` = row sum of
    ``exp(masked - m)``).
    """
    f32 = jnp.float32
    qf, kf, vf = q.astype(f32), k.astype(f32), v.astype(f32)
    B, T, H, dq = qf.shape
    scale = _attn_scale(dq)
    scores = jnp.einsum("bthd,bshd->bhts", qf, kf) * scale
    causal = jnp.arange(T)[None, :] <= jnp.arange(T)[:, None]
    sm = jnp.where(causal[None, None, :, :], scores, jnp.finfo(f32).min)
    m = jnp.max(sm, axis=-1)                       # (B, H, T)
    l = jnp.sum(jnp.exp(sm - m[..., None]), axis=-1)
    attn = jnp.exp(sm - m[..., None]) / l[..., None]
    o = jnp.einsum("bhts,bshd->bthd", attn, vf)
    return o, m, l


def _tiles(x, width):
    """Split the T axis into tiles: ``(B, n, width, H, dq)`` from ``(B, T, H, dq)``."""
    B, T, H, dq = x.shape
    return x.reshape(B, T // width, width, H, dq)


def emulate_attention(q, k, v, *, bq=BQ, bk=BK, casts=True):
    """The kernel's declared numerics, tile by tile, in jnp (bf16 casts included).

    Mirrors :func:`_build_kernel`'s loop order exactly: per (batch, head)
    query tiles outer, KV tiles ``j < i`` full (no mask), the diagonal tile
    last with the static causal pattern; ``p`` is staged as the bf16 hi/lo
    pair before its two ``@ v`` mmas, the online-softmax rescale and
    statistics stay fp32.  The mma itself is modelled as an exact fp32 matmul
    of the bf16 operands (the design's declared accumulator; the instruction's
    internal accumulation order is not modelled — it is inside the tolerance
    budget).

    ``casts=False`` turns the declared bf16 staging off (the operands enter at
    their storage precision): that is the *declared-algebra audit* form — the
    tiled fp32 algebra itself, which the tests pin against the fp32 reference
    (the rounding term is pinned separately, with ``casts=True``).
    """
    f32, dt = jnp.float32, jnp.dtype(q.dtype)
    B, T, H, dq = q.shape
    scale = float(_attn_scale(dq))
    nq = T // bq
    o_out = jnp.zeros((B, T, H, dq), f32)
    m_out = jnp.full((B, H, T), jnp.finfo(f32).min, f32)
    l_out = jnp.zeros((B, H, T), f32)
    ridx = jnp.arange(bq)[:, None]
    cidx = jnp.arange(bk)[None, :]
    diag_pat = (ridx >= cidx)
    for b in range(B):
        for h in range(H):
            for i in range(nq):
                qi = q[b, i * bq:(i + 1) * bq, h, :].astype(f32)
                acc = jnp.zeros((bq, dq), f32)
                m = jnp.full((bq,), jnp.finfo(f32).min, f32)
                l = jnp.zeros((bq,), f32)

                def _pair(j, acc, m, l, mask):
                    kj = k[b, j * bk:(j + 1) * bk, h, :].astype(f32)
                    vj = v[b, j * bk:(j + 1) * bk, h, :].astype(f32)
                    s = (qi @ kj.T) * scale
                    sm = jnp.where(mask, s, jnp.finfo(f32).min) if mask is not None else s
                    mnew = jnp.maximum(m, jnp.max(sm, axis=1))
                    alpha = jnp.exp(m - mnew)
                    p = jnp.exp(sm - mnew[:, None])
                    l = l * alpha + jnp.sum(p, axis=1)
                    acc = acc * alpha[:, None]
                    if casts:
                        p_hi = p.astype(dt).astype(f32)
                        p_lo = (p - p_hi).astype(dt).astype(f32)
                        acc = acc + p_hi @ vj + p_lo @ vj
                    else:
                        acc = acc + p @ vj
                    return acc, mnew, l

                for j in range(i):
                    acc, m, l = _pair(j, acc, m, l, None)
                acc, m, l = _pair(i, acc, m, l, diag_pat)
                o_out = o_out.at[b, i * bq:(i + 1) * bq, h, :].set(acc / l[:, None])
                m_out = m_out.at[b, h, i * bq:(i + 1) * bq].set(m)
                l_out = l_out.at[b, h, i * bq:(i + 1) * bq].set(l)
    return o_out, m_out, l_out


def emulate_attention_bwd(q, k, v, do, D, m, l, *, bq=BQ, bk=BK, casts=True):
    """The backward kernel's declared numerics, pair by pair, in jnp.

    Mirrors :func:`_build_bwd_kernel`'s algebra and cast points: ``p`` and
    ``ds`` are staged as bf16 *hi/lo pairs* before their mmas (the KDA
    milestone-4 idiom), ``do`` enters as bf16 (the kernel's operand), the
    accumulations are fp32; the KV-outer/query-inner order matches the kernel
    (irrelevant in exact arithmetic, kept for a faithful pin).  Returns
    ``(dq, dk, dv)`` fp32 (the kernels' accumulators).

    ``casts=False`` is the *declared-algebra audit* form: the same adjoint
    with the bf16 stagings off, which the tests pin against ``jax.grad`` of
    :func:`emulate_attention` (the algebra — transposes, signs, the scale —
    must be the exact adjoint; the cast term is pinned separately).
    """
    f32, dt = jnp.float32, jnp.dtype(q.dtype)
    B, T, H, dq = q.shape
    scale = float(_attn_scale(dq))
    nq, nk = T // bq, T // bk
    dq_out = jnp.zeros((B, T, H, dq), f32)
    dk_out = jnp.zeros((B, T, H, dq), f32)
    dv_out = jnp.zeros((B, T, H, dq), f32)
    ridx = jnp.arange(bq)[:, None]
    cidx = jnp.arange(bk)[None, :]
    diag_pat = (ridx >= cidx)
    do32 = (do.astype(dt).astype(f32) if casts else do.astype(f32))
    for b in range(B):
        for h in range(H):
            for j in range(nk):
                kj = k[b, j * bk:(j + 1) * bk, h, :].astype(f32)
                vj = v[b, j * bk:(j + 1) * bk, h, :].astype(f32)
                dk_acc = jnp.zeros((bk, dq), f32)
                dv_acc = jnp.zeros((bk, dq), f32)

                def _pair(i, dk_acc, dv_acc, mask):
                    qi = q[b, i * bq:(i + 1) * bq, h, :].astype(f32)
                    doi = do32[b, i * bq:(i + 1) * bq, h, :]
                    s = (qi @ kj.T) * scale
                    sm = jnp.where(mask, s, jnp.finfo(f32).min) if mask is not None else s
                    mi = m[b, h, i * bq:(i + 1) * bq]
                    li = l[b, h, i * bq:(i + 1) * bq]
                    Di = D[b, h, i * bq:(i + 1) * bq]
                    p = jnp.exp(sm - mi[:, None]) / li[:, None]
                    dp = doi @ vj.T
                    ds = (p * (dp - Di[:, None])) * scale
                    if casts:
                        p_hi = p.astype(dt).astype(f32)
                        p_lo = (p - p_hi).astype(dt).astype(f32)
                        ds_hi = ds.astype(dt).astype(f32)
                        ds_lo = (ds - ds_hi).astype(dt).astype(f32)
                        dk_acc = dk_acc + ds_hi.T @ qi + ds_lo.T @ qi
                        dv_acc = dv_acc + p_hi.T @ doi + p_lo.T @ doi
                        dq_delta = ds_hi @ kj + ds_lo @ kj
                    else:
                        dk_acc = dk_acc + ds.T @ qi
                        dv_acc = dv_acc + p.T @ doi
                        dq_delta = ds @ kj
                    return dk_acc, dv_acc, dq_delta

                dk_acc, dv_acc, dq_delta = _pair(j, dk_acc, dv_acc, diag_pat)
                dq_out = dq_out.at[b, j * bq:(j + 1) * bq, h, :].add(dq_delta)
                for i in range(j + 1, nq):
                    dk_acc, dv_acc, dq_delta = _pair(i, dk_acc, dv_acc, None)
                    dq_out = dq_out.at[b, i * bq:(i + 1) * bq, h, :].add(dq_delta)
                dk_out = dk_out.at[b, j * bk:(j + 1) * bk, h, :].set(dk_acc)
                dv_out = dv_out.at[b, j * bk:(j + 1) * bk, h, :].set(dv_acc)
    return dq_out, dk_out, dv_out


# ---------------------------------------------------------------------------
# The differentiable entry point (the whole attention interval, one launch
# per direction per layer-pass)
#
# The boundary is ``(q, k, v) -> o`` with bf16 primals: the caller casts the
# fp32 projections to the kernel's declared operand path (the cast is a graph
# op, so its own vjp carries the cotangent back into the fp32 graph — the
# dtype contract of the milestone-9/12 lineage).  The forward rule runs the
# persistent kernel and keeps ``(q, k, v, o, m, l)``; the backward rule
# computes ``D = rowsum(do ⊙ o)`` (fp32, exactly the milestone-12 boundary),
# runs the persistent backward kernel and returns bf16 cotangents.
#
# The rules resolve ``kernel``/``kernel_bwd`` on the *module* at call time, so
# the CPU test suite can substitute the declared numerics
# (:func:`emulate_attention` / :func:`emulate_attention_bwd`) at the seam.
# ---------------------------------------------------------------------------


def _attention_fwd_rule(q, k, v, bq, bk):
    # The operand cast lives *inside* the rule — the fp32 op boundary with the
    # bf16 operand path behind it is exactly the shape of
    # ``compute_dtype.gemm_einsum``'s bf16 mode (a fp32-looking op that casts
    # its operands); the cotangents can therefore stay fp32 accumulations
    # (no extra boundary rounding).
    dt = jnp.bfloat16
    o, m, l = kernel(q.astype(dt), k.astype(dt), v.astype(dt), bq=bq, bk=bk)
    return o, (q, k, v, o, m, l)


def _attention_bwd_rule(res, g, bq, bk):
    q, k, v, o, m, l = res
    f32, dt = jnp.float32, jnp.bfloat16
    # D_i = rowsum(do ⊙ o) — the milestone-12 flash-backward boundary (fp32
    # elementwise reduction; the kernel consumes it as an input).
    D = jnp.einsum("bthd,bthd->bht", g, o)
    dq, dk, dv = kernel_bwd(q.astype(dt), k.astype(dt), v.astype(dt),
                            g.astype(dt), D, m, l, bq=bq, bk=bk)
    return dq.astype(f32), dk.astype(f32), dv.astype(f32)


def _make_attention_op(bq, bk):
    """One ``custom_vjp`` attention op per tile pair (the jit-cache stability form)."""

    @jax.custom_vjp
    def op(q, k, v):
        return _attention_fwd_rule(q, k, v, bq, bk)[0]

    def op_fwd(q, k, v):
        return _attention_fwd_rule(q, k, v, bq, bk)

    def op_bwd(res, g):
        return _attention_bwd_rule(res, g, bq, bk)

    op.defvjp(op_fwd, op_bwd)
    return op


#: Cache of the ops by static ``(bq, bk)``.  The ops resolve
#: ``kernel``/``kernel_bwd`` on this module at call time, so observing or
#: substituting them keeps working through this cache (the KDA `_CC_SCANS`
#: contract).
_ATTN_OPS: dict[tuple[int, int], object] = {}


def _attention_op(bq, bk):
    key = (int(bq), int(bk))
    op = _ATTN_OPS.get(key)
    if op is None:
        op = _make_attention_op(key[0], key[1])
        _ATTN_OPS[key] = op
    return op


def mla_attention(q, k, v, bq=BQ, bk=BK):
    """The ``AXIOM_MLA_PERSIST=1`` entry: ``(q, k, v) -> o``, one launch each way.

    ``q``/``k``/``v``: ``(B, T, H, dq)`` float tensors (the dense path's fp32
    projections).  ``T`` is padded to a multiple of the tile widths (the
    padded rows never enter a real row's causal window; the cotangents of the
    padded rows are sliced away by the pad's own vjp).  The operands are cast
    bf16 (the declared operand path) and the interval runs through the
    persistent kernels; the output is fp32.

    Requires a GPU: on a GPU-less host the call refuses with a typed error
    instead of falling back silently (the KDA-persistence contract — the CPU
    test suite substitutes the kernel seams, see the module docstring).
    """
    if not device_available():
        raise RuntimeError(
            "AXIOM_MLA_PERSIST=1: персистентный MLA-кернел исполняется только на "
            "GPU (Mosaic sm_121: один запуск на слой-проход fwd и bwd); на CPU-узле "
            "выберите AXIOM_MLA_PERSIST=0. Тихого отката нет — путь выбирается явно.")
    B, T, H, dq = q.shape
    if k.shape != q.shape or v.shape != q.shape:
        raise ValueError(
            f"q/k/v shapes must match: {q.shape}, {k.shape}, {v.shape}")
    width = max(int(bq), int(bk))
    Tp = (T + width - 1) // width * width
    pad = Tp - T
    if pad:
        q, k, v = (jnp.pad(x, ((0, 0), (0, pad), (0, 0), (0, 0))) for x in (q, k, v))
    op = _attention_op(bq, bk)
    o = op(q, k, v)
    return o[:, :T]

