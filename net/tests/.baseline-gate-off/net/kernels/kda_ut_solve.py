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

Precision, pinned in the body (not inherited)
---------------------------------------------
``check_kernel.py`` runs on the GB10 *without* the suite's
``net/tests/conftest.py``, so a bare ``a @ b`` in the body traced
``dot_general`` with ``precision=None``.  The Pallas-Triton lowerer
(``jax/_src/pallas/triton/lowering.py::_dot_general_lowering``) maps ``None``
to ``(Precision.DEFAULT, Precision.DEFAULT)`` and that pair to Triton's
**tf32** for f32 operands, so the GPU verdict came back at ``max_abs ~ 3.6e-3``
with 61 % of entries outside ``1e-4`` — against the ``<=5e-6`` the CPU path
measures.  That is tf32's 10-bit mantissa, not the algorithm.

The body therefore pins **every** matmul to
``lax.DotAlgorithmPreset.F32_F32_F32`` for the f32 case (see
:func:`_matmul_precision`); the same lowerer turns that preset into
``tt.dot(input_precision=IEEE)``.  ``F32_F32_F32`` is the explicit
"f32 x f32 -> f32, full precision" preset — the ``tl.dot(input_precision="ieee")``
of the Triton API — so the f32 verdict no longer depends on the ambient
``jax_default_matmul_precision``.  (ADR-010 pins that policy for the *suite*;
what this module guarantees is that the kernel holds on its own under
``check_kernel.py`` on the GB10.)

The four matmul sites inside :func:`_solve_head` are the only ``dot`` in the
body (``eye``/``tril``/the adds are elementwise or select); each one carries the
pinned precision, and the last ``power @ power`` of the final loop step is dead
and dropped by DCE.

The bf16 case keeps the backend default (tf32 on tensor cores) on purpose: it
is already PASS (0 % of entries outside the ``(2e-2, 2e-2)`` pair), its verdict
is dominated by the bf16 input/output rounding, and the fix is scoped to the
f32 arithmetic.  ``None`` there is not a silent tf32: it is the pre-existing
behaviour, left untouched.

Backward: a transposed solve wrapped in ``custom_vjp``
------------------------------------------------------
A bare ``pallas_call`` has no JVP, so integrating the forward-only kernel into a
training graph failed with *"Linearization failed to produce known values for all
output primals ... an operation with no defined JVP"*.  The fix is an explicit
adjoint, and the adjoint of a triangular solve is again a triangular solve —
this time **transposed**.

Forward is ``X = M^{-1} B`` with ``M = I + L``.  Differentiating ``M X = B``
gives ``dM X + M dX = dB``; the standard adjoint of a linear solve is

    dB = M^{-T} dX = T^T dX ,      dM = -M^{-T} dX X^T = -T^T dX X^T ,

with ``T = (I + L)^{-1}`` the very factor the kernel builds.  ``dM = dL``
(``M = I + L``), and the module's only trainable input is ``L = tril(A, -1)``, so

    dB = kernel_t(A, dX) ,         dA = strict_lower(-(kernel_t(A, dX)) X^T) .

``kernel_t`` solves ``M^T Y = dX``, i.e. it *is* the ``T^T`` multiplier; no
``jnp.linalg.inv`` is used anywhere in this module.

For the **transposed** forward ``M^T X = B`` the same rule applies with
``M -> M^T``: ``dB = M^{-1} dX = kernel(A, dX)`` and ``dM^T = -M^{-1} dX X^T``;
transposing back, ``dA = strict_lower(-X (kernel(A, dX))^T)``.  So each
orientation's backward reads its multiplier from the *other* kernel.

Why ``dA`` is masked.  ``A`` enters only through ``tril(A, -1)``: the unit
diagonal and the zero upper triangle are not parameters.  The mask is therefore
the shape of the map, not an approximation.  An *unconstrained* oracle
``jnp.linalg.solve(A, B)`` places non-zero mass on the diagonal/upper triangle and
disagrees with a correct constrained gradient on exactly those entries — the
gradcheck compares the strictly-lower block against the raw ``jnp.linalg.solve``
gradient and asserts the rest is exactly zero.

Transposed kernel.  :func:`kernel_t` / :func:`solve_jax_t` / :func:`_solve_head_t`
solve ``(I + L)^T Y = B`` by the *same* Neumann product form on the strictly
upper nilpotent ``N = -L^T``:  ``(I + L^T)^{-1} = (I - L^T)(I + L^{2T})(I + L^{4T})...``
(same ``log2(C)`` matmuls, same precision pin as :func:`_solve_head`).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
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

#: Matmul precision the **f32** case pins inside the kernel body.
#:
#: ``check_kernel.py`` runs on the GB10 without ``net/tests/conftest.py``, so
#: the ambient ``jax_default_matmul_precision`` is unset there.  A bare
#: ``a @ b`` then traces ``dot_general`` with ``precision=None``, which the
#: Pallas-Triton lowerer maps to ``(Precision.DEFAULT, Precision.DEFAULT)`` —
#: Triton's **tf32** for f32 operands (``_dot_general_lowering``).  This preset
#: is that lowerer's explicit IEEE branch (``tt.dot(input_precision=IEEE)``):
#: the ``input_precision="ieee"`` of the task, spelled through the jax API the
#: body actually uses (``lax.dot``).  Chosen over ``Precision.HIGHEST`` because
#: it states the whole algorithm — f32 in, f32 accumulator, f32 out — and does
#: not depend on the lowerer's ``_TF32_PRECISIONS`` table.  See the module
#: docstring ("Precision, pinned in the body").
_F32_MATMUL_PRECISION = lax.DotAlgorithmPreset.F32_F32_F32

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


def _matmul_precision(dtype):
    """Precision pinned for every matmul of :func:`_solve_head` in a case.

    f32 -> :data:`_F32_MATMUL_PRECISION` (full IEEE): the f32 tolerance is a
    pure fp32 one and a ``precision=None`` dot is tf32 on the Triton backend —
    see the module docstring.  bf16 -> ``None``, i.e. the pre-existing behaviour
    (backend default), left untouched because that case already passes.

    Used by both the Pallas body (``_build_kernel``) and :func:`solve_jax`, so
    the two paths run the *same* arithmetic and the CPU parity verdict stays a
    verdict on the kernel's program.
    """
    return _F32_MATMUL_PRECISION if jnp.dtype(dtype) == jnp.float32 else None


def _solve_head(av, bv, precision):
    """Solve ``av @ x = bv`` for a unit lower-triangular ``av`` (one head).

    ``av`` is ``(C, C)`` and ``bv`` is ``(C, n)``; the caller guarantees the
    unit diagonal.  Pure matmul + elementwise: the only ops the Triton backend
    of Pallas is asked to lower.

    ``precision`` is forwarded to **all four** matmul sites below — the ``N**2``
    seed, the ``(I + N^(2**k))`` accumulate and the ``N -> N**2 -> N**4 ...``
    squaring inside the loop, and the final ``T @ bv`` — so no ``dot`` in the
    body can fall back to the ambient default.  Pass
    :func:`_matmul_precision` of the case dtype.
    """
    c = av.shape[0]
    eye = jnp.eye(c, dtype=jnp.float32)
    n_low = -jnp.tril(av, -1)  # N = -L; strictly lower, so N**c == 0
    t_mat = eye + n_low  # (I + N)
    power = lax.dot(n_low, n_low, precision=precision)  # N**2
    for _ in range(_inverse_steps(c)):
        t_mat = lax.dot(t_mat, power + eye, precision=precision)  # (I+N)(I+N^2)(I+N^4)...
        power = lax.dot(power, power, precision=precision)  # N**4, N**8, ...
    return lax.dot(t_mat, bv, precision=precision)


def _solve_head_t(av, bv, precision):
    """Solve ``av.T @ x = bv`` — the transpose of :func:`_solve_head`.

    ``av`` is unit lower triangular ``(C, C)``, so ``av.T = I + L^T`` is unit
    *upper* triangular.  The adjoint ``T^T`` of the forward solve is exactly the
    inverse of this matrix, so the backward pass needs this routine rather than
    ``jnp.linalg.inv``.

    Same Neumann product form as :func:`_solve_head`, on the strictly **upper**
    nilpotent factor ``N = -L^T`` (``L = tril(av, -1)``): ``N`` is strictly upper,
    so ``N**C == 0`` and ``(I + L^T)^{-1} = (I - N)^{-1} = (I + N)(I + N^2)...``.
    Note this is *not* ``_solve_head(av.T, ...)``: ``tril(av.T, -1)`` is zero for
    a unit-upper matrix, so the factor must be transposed *explicitly*.

    Only ``tril(av, -1).T`` is transposed (a ``C x C`` cheap op); the loop is the
    identical ``log2(C)``-matmul body, so the two solves cost the same.
    """
    c = av.shape[0]
    eye = jnp.eye(c, dtype=jnp.float32)
    n_up = -jnp.tril(av, -1).T  # N = -L^T; strictly upper, so N**c == 0
    t_mat = eye + n_up  # (I + N)
    power = lax.dot(n_up, n_up, precision=precision)  # N**2
    for _ in range(_inverse_steps(c)):
        t_mat = lax.dot(t_mat, power + eye, precision=precision)  # (I+N)(I+N^2)(I+N^4)...
        power = lax.dot(power, power, precision=precision)  # N**4, N**8, ...
    return lax.dot(t_mat, bv, precision=precision)


def solve_jax(a, b):
    """Host implementation of the kernel's arithmetic (batched over ``H``).

    Same pinned precision as the kernel body (:func:`_matmul_precision`), so the
    CPU parity verdict is a verdict on the program the kernel actually runs — not
    on a separately-configured host path.
    """
    precision = _matmul_precision(a.dtype)
    x = jax.vmap(lambda av, bv: _solve_head(av, bv, precision))(
        a.astype(jnp.float32), b.astype(jnp.float32)
    )
    return x.astype(a.dtype)


def solve_jax_t(a, b):
    """Host transpose of :func:`solve_jax`: solves ``(I + L)^T X = B``.

    ``a`` is unit lower triangular ``(H, C, C)``.  Same pinned precision as the
    kernel body, so the CPU verdict on the transposed path is a verdict on the
    program :func:`kernel_t` runs.
    """
    precision = _matmul_precision(a.dtype)
    x = jax.vmap(lambda av, bv: _solve_head_t(av, bv, precision))(
        a.astype(jnp.float32), b.astype(jnp.float32)
    )
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


def reference_t(a, b):
    """Exact oracle for the transposed solve: ``jnp.linalg.solve(a.T, b)`` in f32."""
    x = jnp.linalg.solve(
        jnp.swapaxes(a.astype(jnp.float32), -1, -2), b.astype(jnp.float32)
    )
    return x.astype(a.dtype)


def baseline_t(a, b):
    """The XLA path for the transposed solve — ``triangular_solve`` on ``a.T``.

    ``(I + L)^T`` is unit *upper* triangular, hence ``lower=False``.
    """
    x = jax.lax.linalg.triangular_solve(
        jnp.swapaxes(a.astype(jnp.float32), -1, -2),
        b.astype(jnp.float32),
        left_side=True,
        lower=False,
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


def _build_kernel(H, C, N, dtype, bn, num_warps, num_stages, interpret=False, transpose=False):
    """Build (and jit) the ``pallas_call`` for one static configuration.

    ``interpret=True`` runs the same body over the same block layout through
    Pallas' interpreter instead of the Triton compiler.  That is what lets a
    host *without* a GPU execute the layout and compare it numerically (see
    ``tests/test_kernel_kda_ut_solve.py``); :func:`kernel` never uses it.

    ``transpose=True`` builds the **transposed** solve ``(I + L)^T X = B``: the
    body is byte-for-byte the forward body with :func:`_solve_head` replaced by
    :func:`_solve_head_t`.  Same grid, same block specs, same precision pin.
    """
    from jax.experimental.pallas import triton as pltriton

    solve = _solve_head_t if transpose else _solve_head

    def body(a_ref, b_ref, o_ref):
        av = pltriton.load(a_ref).astype(jnp.float32)
        bv = pltriton.load(b_ref).astype(jnp.float32)
        # Precision is picked from the *declared* case dtype, not from the
        # ambient policy: this is what makes the body self-contained under
        # check_kernel.py on the GB10 (no conftest there).  See _matmul_precision.
        precision = _matmul_precision(dtype)
        pltriton.store(o_ref, solve(av, bv, precision).astype(o_ref.dtype))

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


def _dispatch_kernel(a, b, params, transpose):
    """Shared body of :func:`kernel`/:func:`kernel_t` (host vs Pallas dispatch)."""
    cfg = {**DEFAULT_PARAMS, **params}
    if not _on_gpu():
        return solve_jax_t(a, b) if transpose else solve_jax(a, b)

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
    key = (H, C, N, str(a.dtype), bn, int(cfg["num_warps"]), int(cfg["num_stages"]),
           bool(transpose))
    fn = _KERNEL_CACHE.get(key)
    if fn is None:
        fn = _build_kernel(
            H, C, N, a.dtype, bn, int(cfg["num_warps"]), int(cfg["num_stages"]),
            transpose=transpose,
        )
        _KERNEL_CACHE[key] = fn
    return fn(a, b)


def kernel(a, b, **params):
    """``(I + L) X = B``, batched over ``H``; the module's kernel entry point.

    ``params`` are the :data:`TUNE_SPACE` keys (``bn``, ``num_warps``,
    ``num_stages``); on a non-GPU host they are inert because the host path
    (:func:`solve_jax`) has no blocks, and the Pallas build/execution is
    skipped — see the module docstring on host dispatch.
    """
    return _dispatch_kernel(a, b, params, transpose=False)


def kernel_t(a, b, **params):
    """``(I + L)^T X = B``, batched over ``H``; the transposed kernel entry point.

    Solves the transpose of :func:`kernel`'s system; this is the ``T^T``
    multiplier the backward rule of :data:`solve` needs.  Same dispatch
    contract as :func:`kernel` (host path :func:`solve_jax_t` without a GPU).
    """
    return _dispatch_kernel(a, b, params, transpose=True)


# ---------------------------------------------------------------------------
# Differentiable wrapper: transposed-solve backward under ``jax.custom_vjp``
#
# The bare ``kernel``/``kernel_t`` are forward-only: a Pallas ``pallas_call`` has
# no JVP, so putting one in a training graph fails with "Linearization failed to
# produce known values for all output primals".  ``solve``/``solve_t`` wrap the
# kernel with the analytic adjoint (module docstring, "Backward"); the adjoint is
# itself a triangular solve, taken **transposed** from the sibling kernel — never
# from ``jnp.linalg.inv``.
# ---------------------------------------------------------------------------


def _strict_lower_mask(c, dtype=jnp.bool_):
    """Strictly lower-triangular mask ``(C, C)`` — the trainable part of ``A``."""
    return jnp.tril(jnp.ones((c, c), dtype), -1)


def _backward(a, x, dx, transpose, params):
    """Analytic adjoint ``(dA, dB)`` of the (transposed) solve — see the docstring.

    ``a`` is ``(H, C, C)`` unit lower triangular, ``x`` the forward solution
    ``(H, C, N)``, ``dx`` its cotangent.  Both branches get ``T^T`` from a
    **kernel**, never from ``linalg.inv``:

    * forward ``M X = B``:   ``T^T dX = kernel_t(a, dX)`` and ``dM = -Y X^T``;
    * forward ``M^T X = B``: ``T dX = kernel(a, dX)`` and ``dM^T = -Z X^T``.

    ``dA`` is masked strictly lower: ``A`` enters the map only through
    ``tril(A, -1)`` (unit diagonal, zero upper triangle are fixed), so the
    diagonal/upper entries of the true gradient are exactly zero.
    """
    precision = _matmul_precision(a.dtype)
    mask = _strict_lower_mask(a.shape[-1])
    if not transpose:
        y = kernel_t(a, dx, **params)  #  Y = M^{-T} dX  (transposed kernel!)
        db = y
        d_full = jax.vmap(lambda yh, xh: lax.dot(yh, xh.T, precision=precision))(y, x)
    else:
        z = kernel(a, dx, **params)  #  Z = M^{-1} dX   (non-transposed kernel)
        db = z
        d_full = jax.vmap(lambda xh, zh: lax.dot(xh, zh.T, precision=precision))(x, z)
    da = jnp.where(mask, -d_full, jnp.zeros_like(d_full))
    return da, db


def _solve_forward(a, b, transpose, params):
    """Forward dispatch shared by the custom_vjp primal and its fwd rule.

    The kernel is looked up on the *module* (not captured), so a test can observe
    which kernel the forward and the backward actually run.
    """
    kernel_fn = kernel_t if transpose else kernel
    return kernel_fn(a, b, **params)


def make_solve(transpose=False, **params):
    """Wrap the (transposed) solve in ``jax.custom_vjp`` with the analytic adjoint.

    Forward is :func:`kernel`/:func:`kernel_t` with ``params``; the fwd rule
    stashes ``(a, x)`` and the bwd rule is :func:`_backward`.  This is the entry
    point a training graph must use — it carries the JVP/transpose the bare
    ``pallas_call`` lacks, which is what the integrated forward-only kernel was
    missing.
    """

    @jax.custom_vjp
    def solve(a, b):
        return _solve_forward(a, b, transpose, params)

    def solve_fwd(a, b):
        x = _solve_forward(a, b, transpose, params)
        return x, (a, x)

    def solve_bwd(res, dx):
        a, x = res
        return _backward(a, x, dx, transpose=transpose, params=params)

    solve.defvjp(solve_fwd, solve_bwd)
    return solve


#: Differentiable forward solve ``(I + L) X = B``; adjoint via :func:`kernel_t`.
solve = make_solve()

#: Differentiable transposed solve ``(I + L)^T X = B``; adjoint via :func:`kernel`.
solve_t = make_solve(transpose=True)


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
