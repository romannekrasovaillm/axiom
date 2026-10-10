"""Batched unit-triangular solve ``(I + L) X = B`` for the KDA chunk (Pallas).

Why a kernel at all
-------------------
The KDA chunk needs the intra-chunk correction ``X = (I + L)^{-1} B`` where
``I + L`` is a unit lower-triangular ``(H, C, C)`` and ``B`` is
``(H, C, dk + dv)`` (``net/kda.py``: ``Y = T Diag(beta) (Gamma . K)`` and
``U = T Diag(beta) V`` with ``T = (I + L)^{-1}``).  XLA lowers the batched
``triangular_solve``/``inv`` to one 64x4-tile kernel *per (batch, RHS tile)*
plus its pointer-building companion: the nsys profile of the stationary phase
counts **115 200** ``batch_trsm_left_kernel<...,64,4,...>`` launches and
**230 400** ``MakeBatchPointers`` over 120 s at 3.5 % GPU occupancy — the
textbook "XLA does not build one big kernel out of this" case of the
``pallas-gb10-kernel`` skill (non-standard op, tiny per-launch work, pure
launch/pointer overhead).  Geometry: ``H=12, C=64, dk=dv=128, hidden=1536``
(``net/config.json``).

The kernel
----------
One Pallas program per ``(head, RHS column tile)``: it loads the head's
``(C, C)`` factor and a ``(C, bn)`` slice of ``B``, forms the inverse of the
unit lower-triangular factor by the **Neumann product form** and multiplies.
Because ``N = -L`` is strictly lower triangular it is nilpotent (``N**C = 0``),
so ``(I + L)^{-1} = (I - N)^{-1} = (I + N)(I + N^2)(I + N^4) ... (I + N**C/2)``
— ``log2(C)`` small matmuls, no data-dependent indexing, no sequential
dependency chain.  The whole kernel body is matmul + elementwise, which is
what the Triton backend of Pallas on JAX 0.10.2 lowers.

Block layout
------------
Grid ``(H, N / bn)``: program ``(h, j)`` owns head ``h`` over RHS columns
``[j * bn, (j + 1) * bn)``.  The head axis of every array is a *squeezed* block
dimension (``None``), not a tiled one: grid program ``h`` takes head ``h``
whole, so the refs the body receives keep the ``(C, C)`` / ``(C, bn)`` shapes
the solve is written for, and no block is ever split across programs.  A block
shape must have one entry per array dimension — ``(C, C)`` for an ``(H, C, C)``
array is a rank mismatch and is rejected, see :func:`_block_specs`.  An index
map returns a *block* index, not an element offset (``start = block * index``),
except on a squeezed dim, where it is the element index itself.

Launch count, statically
------------------------
Grid ``(H, N / bn)``: **one kernel launch** per solve, 12 x 4 = 48 programs at
the default ``bn=64`` — the GB10 has 48 SMs, which is why ``bn`` is the tuning
knob.  The XLA form under the profile's ``batch_trsm_left_kernel<...,64,4,...>``
emits at least ``H * (N / 4) = 768`` trsm launches plus two
``MakeBatchPointers`` per launch (``H * N / 4 * 2 = 1536``) per invocation —
2304 launches.  The two profiled totals are consistent with exactly that
reading: ``115200 / 768 = 150`` and ``230400 / 1536 = 150`` invocations over
the 120 s of the stationary phase.  Assumption stated, not hidden: that the
template's ``(64, 4)`` are the row/RHS-column tile and that the pointer kernels
are per-operand.

Backend
-------
Mosaic GPU MMA is unavailable on this stand (``mgpu_mma_bf16`` FAILs on
jax 0.10.2: no ``Layout.MMA_ACC``), while ``triton_dot``/``triton_elementwise``
PASS in ``gb10_caps_full.json`` — so this is the legacy Pallas-Triton path
(``pl.pallas_call`` + ``pltriton``), pinned to JAX 0.10.2.

Host dispatch (documented, not a silent fallback)
-------------------------------------------------
Pallas can only *execute* on a GPU, so :func:`kernel` runs the same jnp
program (:func:`solve_jax`) on a non-GPU host.  That is what makes the CPU
parity test possible.  The two paths share :func:`_solve_head`, so the CPU
verdict is a verdict on the algorithm the kernel body runs — **not** on the
kernel.

That dispatch is also why a host-side static lowering of :func:`kernel` is
vacuous: with no GPU, ``lower_check.py`` lowers :func:`solve_jax`, and the
``pallas_call`` is never traced, so a bad ``block_shape``/``index_map``/``grid``
cannot surface there.  The tests therefore build and trace the call directly:
``test_pallas_block_specs_tile_every_array_dimension`` pins the layout and
``test_pallas_call_traces_on_the_host`` runs the same ``BlockSpec.to_block_mapping``
rank check that fails at compile time on the GB10, while
``test_pallas_layout_is_numerically_right_on_the_host`` *executes* the layout
through Pallas' interpreter, on the host.  What remains GPU-only is the
compiled Triton kernel itself — execution and numerics of the real build:
``check_kernel.py`` on the GB10.

Accuracy note
-------------
The product form is *exactly* the inverse in exact arithmetic, and the
rounding of ``log2(C) = 6`` f32 matmuls does not compound: on the case
geometry the measured ``max|X_kernel - X_solve|`` is ``<= 5e-6`` for f32
(see ``tests/test_kernel_kda_ut_solve.py``, where the tolerance is derived).
A *stable* alternative (blocked forward substitution) would add sequential
depth and data-dependent indexing for no accuracy that is needed here.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

# ---------------------------------------------------------------------------
# Module contract (pallas-gb10-kernel): real task geometry only, never a
# stand-in square problem.
# ---------------------------------------------------------------------------

#: ``H`` heads, ``C`` chunk, ``dk``/``dv`` head dims — ``net/config.json``.
#: Both dtypes of the case: f32 (the arithmetic the KDA layer actually runs)
#: and bf16 (the low-precision input the kernel must stay inside its stated
#: tolerance on).
CASES = [
    dict(H=12, C=64, dk=128, dv=128, dtype="float32"),
    dict(H=12, C=64, dk=128, dv=128, dtype="bfloat16"),
]

#: Tolerances for ``check_kernel.py`` on the GB10.  f32: the f32 product form
#: is measured at <= 5e-6 max-abs, so 1e-4 keeps a >20x margin.  bf16: the
#: verdict is dominated by rounding the bf16 *inputs* and the bf16 *output*
#: (2^-9 = 2e-3 relative; 2e-2 is the skill's default bf16 pair).
TOLERANCE = {"float32": (1e-4, 1e-4), "bfloat16": (2e-2, 2e-2)}

#: Uniform (-0.25, 0.25) strictly-lower entries: |L_ij| is the magnitude of the
#: KDA score ``beta_r * Akk_{r,i}`` (|Akk| <= 1 for l2-normalised k, beta <= 1),
#: sampled away from the degenerate all-zero case without making ||L|| large
#: enough for the product form to lose digits.
_L_SCALE = 0.25

#: Register-footprint proxy per thread (f32 values), the *pre-compile* sieve of
#: ``valid_config``.  GB10 has 64K registers of 32 bit per SM and <= 255 per
#: thread; 128 leaves headroom for the Triton-generated temporaries, and the
#: product-form body holds ``A``, ``B``, ``N``, ``T``, ``P`` live at once.
_REGS_PER_THREAD = 128

#: Tuning knobs.  ``bn`` tiles the RHS columns: the grid becomes
#: ``(H, N / bn)``, i.e. 12 programs at bn=256 against 48 SMs, and
#: 12 * 4 = 48 programs at bn=64 — exactly the SM count of the GB10, which is
#: the reason the tile is a real knob and not decoration.  4 * 3 * 2 = 24
#: configurations, sieved by :func:`valid_config` before any compilation.
TUNE_SPACE = dict(
    bn=[32, 64, 128, 256],
    num_warps=[4, 8, 16],
    num_stages=[1, 2],
)

#: Defaults used by ``bench.py``/``check_kernel.py`` when no params are passed.
DEFAULT_PARAMS = dict(bn=64, num_warps=8, num_stages=1)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def make_inputs(key, H, C, dk, dv, dtype):
    """``(a, b)`` with ``a`` unit lower triangular ``(H, C, C)``, ``b`` ``(H, C, dk+dv)``.

    ``a = I + tril(U(-_L_SCALE, _L_SCALE))`` — the shape the kernel contract
    names ("единичная нижняя треугольная"); ``b`` is standard normal.  Both are
    cast to the case dtype, so the f32 case carries no quantisation and the
    bf16 case carries only the input rounding.
    """
    dt = jnp.dtype(dtype)
    k_l, k_b = jax.random.split(key)
    lower = jax.random.uniform(
        k_l, (H, C, C), dtype=jnp.float32, minval=-_L_SCALE, maxval=_L_SCALE
    )
    a = jnp.eye(C, dtype=jnp.float32) + jnp.tril(lower, -1)
    b = jax.random.normal(k_b, (H, C, dk + dv), dtype=jnp.float32)
    return a.astype(dt), b.astype(dt)


# ---------------------------------------------------------------------------
# The algorithm: one head, shared by the Pallas body and the host path.
# ---------------------------------------------------------------------------


def _inverse_steps(c: int) -> int:
    """Number of ``(I + N^(2**k))`` factors after the first — ``log2(c) - 1``."""
    return max(1, (c - 1).bit_length() - 1)


def _solve_head(av, bv):
    """Solve ``av @ x = bv`` for a unit lower-triangular ``av`` (one head).

    ``av`` is ``(C, C)`` and ``bv`` is ``(C, n)``; the caller guarantees the
    unit diagonal.  Pure matmul + elementwise: the only ops the Triton backend
    of Pallas is asked to lower.
    """
    c = av.shape[0]
    eye = jnp.eye(c, dtype=jnp.float32)
    n_low = -jnp.tril(av, -1)  # N = -L; strictly lower, so N**c == 0
    t_mat = eye + n_low  # (I + N)
    power = n_low @ n_low  # N**2
    for _ in range(_inverse_steps(c)):
        t_mat = t_mat @ (power + eye)  # (I+N)(I+N^2)(I+N^4)...
        power = power @ power  # N**4, N**8, ...
    return t_mat @ bv


def solve_jax(a, b):
    """Host implementation of the kernel's arithmetic (batched over ``H``)."""
    c = a.shape[1]
    x = jax.vmap(lambda av, bv: _solve_head(av, bv))(a.astype(jnp.float32), b.astype(jnp.float32))
    return x.astype(a.dtype)


# ---------------------------------------------------------------------------
# Oracles
# ---------------------------------------------------------------------------


def reference(a, b):
    """Exact oracle: ``jnp.linalg.solve`` in f32 (LU with pivoting), cast back."""
    x = jnp.linalg.solve(a.astype(jnp.float32), b.astype(jnp.float32))
    return x.astype(a.dtype)


def baseline(a, b):
    """The XLA path this kernel replaces — ``lax.linalg.triangular_solve``."""
    x = jax.lax.linalg.triangular_solve(
        a.astype(jnp.float32),
        b.astype(jnp.float32),
        left_side=True,
        lower=True,
        unit_diagonal=True,
    )
    return x.astype(a.dtype)


# ---------------------------------------------------------------------------
# Pallas kernel
# ---------------------------------------------------------------------------

_KERNEL_CACHE: dict = {}


def _block_specs(H, C, N, bn):
    """``(grid, in_specs, out_specs)`` — the block layout, pure and inspectable.

    One program per ``(head, RHS column tile)``.  The head axis is **squeezed**
    (``None``): a head is picked by the grid, not tiled inside the program.
    Squeezed dims are dropped from the ref shape (``_get_ref_block_shape``), so
    the body sees the ``(C, C)`` factor and the ``(C, bn)`` RHS slice the solve
    is written for, and the index map's ``h`` is that head's *element* index
    (a squeezed/``Element`` dim takes the loop index as its start index).

    Coverage.  ``grid = (H, N // bn)``; the head dim takes start ``h`` with
    block size 1, so ``h in range(H)`` covers every head exactly once; the RHS
    dim takes start ``bn * j`` with block size ``bn``, and
    :func:`valid_config` guarantees ``bn | N``, so ``j in range(N // bn)``
    tiles the RHS exactly once.  The ``C`` dims are taken whole (block == dim,
    index 0).

    Index convention: an index map returns a **block index**, not an element
    offset — the framework multiplies it by the block size (``start =
    block_size * index``), except for squeezed/``Element`` dims, where it *is*
    the element index.  So the RHS dim is ``j`` and not ``j * bn``; the latter
    would start the tiles at ``bn * j * bn`` and read past the array.

    Every entry of a block shape must correspond to one array dimension:
    ``BlockSpec.to_block_mapping`` rejects a rank mismatch at trace time, e.g.
    ``(C, C)`` for an ``(H, C, C)`` array.  That check is *not* a lowering-time
    (GB10-only) check — see ``tests/test_kernel_kda_ut_solve.py``.
    """
    rhs_tile = lambda h, j: (h, 0, j)  # noqa: E731 — index map, not a def
    return (
        (H, N // bn),
        [
            pl.BlockSpec((None, C, C), lambda h, j: (h, 0, 0)),
            pl.BlockSpec((None, C, bn), rhs_tile),
        ],
        pl.BlockSpec((None, C, bn), rhs_tile),
    )


def _build_kernel(H, C, N, dtype, bn, num_warps, num_stages, interpret=False):
    """Build (and jit) the ``pallas_call`` for one static configuration.

    ``interpret=True`` runs the same body over the same block layout through
    Pallas' interpreter instead of the Triton compiler.  That is what lets a
    host *without* a GPU execute the layout and compare it numerically (see
    ``tests/test_kernel_kda_ut_solve.py``); :func:`kernel` never uses it.
    """
    from jax.experimental.pallas import triton as pltriton

    def body(a_ref, b_ref, o_ref):
        av = pltriton.load(a_ref).astype(jnp.float32)
        bv = pltriton.load(b_ref).astype(jnp.float32)
        pltriton.store(o_ref, _solve_head(av, bv).astype(o_ref.dtype))

    grid, in_specs, out_specs = _block_specs(H, C, N, bn)
    call = pl.pallas_call(
        body,
        out_shape=jax.ShapeDtypeStruct((H, C, N), dtype),
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        interpret=interpret,
        compiler_params=pltriton.CompilerParams(num_warps=num_warps, num_stages=num_stages),
    )
    return jax.jit(call)


def _on_gpu() -> bool:
    try:
        return any(d.platform == "gpu" for d in jax.devices())
    except Exception:  # noqa: BLE001 — a broken plugin must not read as "GPU"
        return False


def kernel(a, b, **params):
    """``(I + L) X = B``, batched over ``H``; the module's kernel entry point.

    ``params`` are the :data:`TUNE_SPACE` keys (``bn``, ``num_warps``,
    ``num_stages``); on a non-GPU host they are inert because the host path
    (:func:`solve_jax`) has no blocks, and the Pallas build/execution is
    skipped — see the module docstring on host dispatch.
    """
    cfg = {**DEFAULT_PARAMS, **params}
    if not _on_gpu():
        return solve_jax(a, b)

    H, C, _ = a.shape
    N = b.shape[-1]
    bn = int(cfg["bn"])
    # The sieve reads only ``C`` and the RHS width, which the arrays carry; the
    # case dict is reconstructed from the shapes rather than re-derived.
    case = dict(H=H, C=C, dk=N // 2, dv=N // 2, dtype=str(a.dtype))
    if not valid_config(case, cfg):
        raise ValueError(
            f"invalid kernel configuration {cfg} for (H={H}, C={C}, N={N}); "
            "see valid_config()"
        )
    key = (H, C, N, str(a.dtype), bn, int(cfg["num_warps"]), int(cfg["num_stages"]))
    fn = _KERNEL_CACHE.get(key)
    if fn is None:
        fn = _build_kernel(H, C, N, a.dtype, bn, int(cfg["num_warps"]), int(cfg["num_stages"]))
        _KERNEL_CACHE[key] = fn
    return fn(a, b)


# ---------------------------------------------------------------------------
# Cost / tuning contract
# ---------------------------------------------------------------------------


def cost(**case):
    """Necessary work of the solve, for the roofline.

    ``flops`` is the *necessary* arithmetic — the substitution touches
    ``C(C-1)/2`` coefficients per RHS column, i.e. ``2 * C(C-1)/2 * N`` per
    head; ``bytes`` is the necessary traffic (read ``A``, read ``B``, write
    ``X``).  ``kernel_flops`` carries what this kernel actually spends, so an
    MFU report cannot be flattered by the minimal figure: the product form
    pays ``_inverse_steps(C) + 1`` extra ``C x C x C`` matmuls per head.
    """
    H, C, dk, dv = case["H"], case["C"], case["dk"], case["dv"]
    n_rhs = dk + dv
    itemsize = jnp.dtype(case.get("dtype", "float32")).itemsize
    n_cubes = _inverse_steps(C) + 1
    return dict(
        flops=H * (2 * (C * (C - 1) // 2) * n_rhs),
        bytes=H * (C * C + 2 * C * n_rhs) * itemsize,
        kernel_flops=H * (2 * n_cubes * C**3 + 2 * C * C * n_rhs),
    )


def valid_config(case, params):
    """Pre-compile sieve: divisibility, power-of-two blocks, register budget.

    Rejects what the Triton backend cannot represent (non power-of-two block,
    ``N % bn != 0``) and what would spill registers, before spending a
    compilation on it.  The budget is :data:`_REGS_PER_THREAD` f32 values per
    thread over the live set ``A, B, N, T, P`` — a documented proxy, since the
    Pallas-Triton path keeps these in registers, not SMEM.
    """
    c = int(case["C"])
    n_rhs = int(case["dk"]) + int(case["dv"])
    bn = int(params["bn"])
    num_warps = int(params["num_warps"])
    if bn <= 0 or bn > n_rhs or (bn & (bn - 1)) != 0:
        return False
    if n_rhs % bn:
        return False
    if c <= 0 or num_warps <= 0:
        return False
    live_elems = c * c  # A
    live_elems += c * bn  # B slice
    live_elems += c * bn  # output block
    live_elems += 2 * c * c  # N/... T and P hold (C, C) each
    return live_elems <= _REGS_PER_THREAD * num_warps * 32
