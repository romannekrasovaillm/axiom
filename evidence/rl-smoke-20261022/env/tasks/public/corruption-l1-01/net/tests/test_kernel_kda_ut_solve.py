"""Acceptance test for the first Pallas kernel: ``net/kernels/kda_ut_solve.py``.

What can be verified on a machine without a GPU (this suite):

* the **algorithm** the kernel body runs — ``kernel`` dispatches to the same
  jnp program on a non-GPU host, so ``kernel`` vs ``reference`` here is a
  numerics verdict on :func:`net.kernels.kda_ut_solve._solve_head`;
* the **contract** of the module (shapes/dtypes, ``valid_config``,
  ``TUNE_SPACE``/``cost`` well-formedness);
* the **envelope** of the Neumann product form (it needs ``||L|| < 1``; the
  KDA factor is ``I + beta * tril(Akk)`` with ``beta <= 1``, ``|Akk| <= 1``);
* the **precision** the body pins — f32 matmuls carry the explicit IEEE preset
  rather than the ambient policy, read back from the traced ``pallas_call``
  (section 4).  That is the regression behind the tf32 GPU verdict.

What it cannot: execution and numerics of the compiled Triton kernel.  That is
``check_kernel.py``/``bench.py`` on the GB10 — see the module docstring and
``.arch-handoff/result.json``.

One GPU-only failure *is* reproducible here.  ``lower_check.py`` lowers
``kernel``, and on a host without a GPU ``kernel`` dispatches to ``solve_jax``,
so the ``pallas_call`` is never traced and a wrong ``block_shape`` rank cannot
surface there.  Section 3 builds and traces the call itself, which runs jax's
``BlockSpec.to_block_mapping`` rank check — exactly the check that fired on the
GB10 as "Block shape for args[0] ... must have the same number of dimensions as
the array shape" — pins the block tiling, and executes the layout in Pallas'
interpreter, where a wrong block *offset* shows up numerically.
"""

from __future__ import annotations

import ast
import itertools

import jax
import jax.numpy as jnp
import pytest

from net.kernels import kda_ut_solve as K

#: Seeds the parity runs average over (the error is seed-dependent through the
#: conditioning of ``I + L``, so a single seed would undersample it).
SEEDS = (0, 1, 2, 3)


def _max_abs(x, y) -> float:
    return float(jnp.abs(x.astype(jnp.float32) - y.astype(jnp.float32)).max())


def _frac_bad(x, y, atol, rtol) -> float:
    d = jnp.abs(x.astype(jnp.float32) - y.astype(jnp.float32))
    ref = y.astype(jnp.float32)
    return float((d > atol + rtol * jnp.abs(ref)).mean())


def _case(dtype):
    return next(c for c in K.CASES if c["dtype"] == dtype)


# ---------------------------------------------------------------------------
# 1. Numerics: kernel == reference on the real geometry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", K.CASES, ids=lambda c: c["dtype"])
def test_kernel_matches_reference(case):
    """``kernel`` reproduces the f32 LU oracle inside the declared tolerance."""
    atol, rtol = K.TOLERANCE[case["dtype"]]
    worst = 0.0
    ref_scale = 0.0
    for seed in SEEDS:
        a, b = K.make_inputs(jax.random.key(seed), **case)
        x = K.kernel(a, b)
        ref = K.reference(a, b)
        assert x.shape == ref.shape
        bad = _frac_bad(x, ref, atol, rtol)
        assert bad == 0.0, (
            f"{case['dtype']} seed {seed}: {bad:.2%} of entries outside "
            f"atol={atol:g} rtol={rtol:g} (max_abs={_max_abs(x, ref):.3g})"
        )
        worst = max(worst, _max_abs(x, ref))
        ref_scale = max(ref_scale, float(jnp.abs(ref).max()))

    # The tolerance is a *declared* number, not a fitted one: keep the margin
    # visible so a regression that eats it fails here rather than at the edge.
    if case["dtype"] == "float32":
        # Measured <= 4e-6 on this geometry; 1e-4 is a >20x margin.
        assert worst < 1e-5, f"f32 max_abs {worst:.3g} lost the margin to {atol:g}"
    else:
        # bf16: the f32 internal rounding (~4e-6) is far below the bf16 output
        # quantum, so the residual is a cast verdict — a few bf16 ulps of the
        # solution's magnitude at most (ulp(v) <= 2**-8 * v).
        assert worst <= 4 * (2.0**-8) * ref_scale, (
            f"bf16 max_abs {worst:.3g} exceeds a few ulps of |X|max={ref_scale:.3g}"
        )


def test_baseline_is_the_right_target():
    """Sanity of the replacement target: XLA's triangular_solve agrees with LU."""
    case = _case("float32")
    a, b = K.make_inputs(jax.random.key(0), **case)
    assert _max_abs(K.baseline(a, b), K.reference(a, b)) == 0.0


def test_product_form_envelope():
    """The product form needs ||L|| < 1; pin the edge of the KDA envelope.

    ``L = beta * tril(Akk)`` with ``beta <= 1`` and ``|Akk| <= 1``, so the KDA
    factor lives at ``|L_ij| <= 1``; measured stable (rel. err ~ 5e-7) up to
    uniform |L| = 1 and broken at 1.5 — the documented validity range.
    """
    H, C, N = 12, 64, 256

    def rel_err(scale):
        k1, k2 = jax.random.split(jax.random.key(7))
        low = jax.random.uniform(k1, (H, C, C), jnp.float32, -scale, scale)
        a = jnp.eye(C, dtype=jnp.float32) + jnp.tril(low, -1)
        b = jax.random.normal(k2, (H, C, N), dtype=jnp.float32)
        ref = jnp.linalg.solve(a, b)
        return _max_abs(K.solve_jax(a, b), ref) / float(jnp.abs(ref).max())

    assert rel_err(1.0) < 1e-5, "product form left the KDA envelope ||L|| <= 1"


# ---------------------------------------------------------------------------
# 2. Module contract: shapes, configuration sieve, cost/tune declarations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", K.CASES, ids=lambda c: c["dtype"])
def test_output_shape_and_dtype(case):
    a, b = K.make_inputs(jax.random.key(0), **case)
    assert a.shape == (case["H"], case["C"], case["C"])
    assert b.shape == (case["H"], case["C"], case["dk"] + case["dv"])
    x = K.kernel(a, b)
    assert x.shape == (case["H"], case["C"], case["dk"] + case["dv"])
    assert x.dtype == jnp.dtype(case["dtype"])


def test_make_inputs_is_unit_lower_triangular():
    a, _ = K.make_inputs(jax.random.key(0), **_case("float32"))
    eye = jnp.eye(a.shape[-1], dtype=jnp.float32)
    assert float(jnp.abs(jnp.diagonal(a, axis1=-2, axis2=-1) - 1.0).max()) == 0.0
    assert float(jnp.abs(jnp.triu(a, 1)).max()) == 0.0


def test_valid_config_sieves_before_compilation():
    """``valid_config`` must reject what the backend cannot take, and only that."""
    case = _case("float32")
    n_rhs = case["dk"] + case["dv"]

    assert K.valid_config(case, K.DEFAULT_PARAMS)
    # bn must divide the RHS width ...
    assert not K.valid_config(case, dict(K.DEFAULT_PARAMS, bn=96))
    assert not K.valid_config(case, dict(K.DEFAULT_PARAMS, bn=n_rhs * 2))
    # ... be a power of two (Triton blocks) and positive ...
    assert not K.valid_config(case, dict(K.DEFAULT_PARAMS, bn=24))
    assert not K.valid_config(case, dict(K.DEFAULT_PARAMS, bn=0))
    # ... and the live set of the body must fit the register budget.
    assert not K.valid_config(case, dict(K.DEFAULT_PARAMS, bn=256, num_warps=4))
    assert K.valid_config(case, dict(K.DEFAULT_PARAMS, bn=256, num_warps=16))


def test_valid_config_partitions_the_tune_space():
    """Every TUNE_SPACE point is decidable before any compilation is attempted."""
    from itertools import product

    keys = list(K.TUNE_SPACE)
    points = [dict(zip(keys, vals)) for vals in product(*(K.TUNE_SPACE[k] for k in keys))]
    assert len(points) >= 10, "TUNE_SPACE must offer tens of configurations"

    for case in K.CASES:
        kept = [p for p in points if K.valid_config(case, p)]
        assert kept, f"no config survives for {case['dtype']}"
        assert K.valid_config(case, K.DEFAULT_PARAMS), "the default must survive the sieve"
        # bn tiles the grid (H, n_rhs / bn) -> the number of programs is a knob.
        assert len({p["bn"] for p in kept}) > 1


def test_cost_is_declared_and_consistent():
    case = _case("float32")
    c = K.cost(**case)
    h, ch, dk, dv = case["H"], case["C"], case["dk"], case["dv"]
    n_rhs = dk + dv
    assert c["flops"] == h * (ch * (ch - 1)) * n_rhs  # necessary substitution flops
    assert c["bytes"] == h * (ch * ch + 2 * ch * n_rhs) * 4
    # The algorithm spends more than the necessary flops; the honest figure is
    # reported separately so an MFU report cannot pick the flattering one.
    assert c["kernel_flops"] >= c["flops"]
    assert c["bytes"] > 0


def test_kernel_dispatches_to_the_host_program_without_gpu():
    """Documented host dispatch: no GPU here, so the jnp program is the verdict."""
    if any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("GPU present: the Pallas path is exercised instead")
    case = _case("float32")
    a, b = K.make_inputs(jax.random.key(0), **case)
    assert bool(jnp.array_equal(K.kernel(a, b), K.solve_jax(a, b)))


# ---------------------------------------------------------------------------
# 3. Block layout: rank, coverage and offsets of block_shape/index_map/grid
#
# ``lower_check.py`` lowers ``kernel``, which dispatches to the host program
# when no GPU is present, so on this machine it never sees a block spec.  These
# tests build and trace the ``pallas_call`` directly instead — the same specs
# the GB10 compiles — and one of them executes the layout in the interpreter.
# ---------------------------------------------------------------------------


def _tile_defect(block_shape, index_map, grid, array_shape, one_writer=False):
    """Why the block grid does not tile ``array_shape`` exactly, or ``""``.

    ``None`` is a squeezed dim: block size 1, and the index map's value is the
    element's start index (not a block index).  Every other dim is ``Blocked``
    with start ``block * index``.  The grid tiles the array iff every block is
    tile-aligned (its start is a multiple of its size, per dim) and the distinct
    blocks are all ``prod(dim // size)`` of them — i.e. exactly the set of
    tile-aligned blocks, which partitions the array.

    ``one_writer`` additionally forbids two programs sharing a block: that is
    required of an output spec (double writes), not of an input spec, where
    several programs legitimately re-read one block (every RHS tile needs the
    whole head factor).
    """
    if len(block_shape) != len(array_shape):
        return (f"block rank {len(block_shape)} != array rank {len(array_shape)}"
                f" (array {array_shape})")
    sizes = [1 if b is None else int(b) for b in block_shape]
    starts, boxes, n_grid, n_tiles = [[] for _ in array_shape], set(), 1, 1
    for dim, size in zip(array_shape, sizes):
        if dim % size:
            return f"block {size} does not divide array dim {dim}"
        n_tiles *= dim // size
    for g in grid:
        n_grid *= g
    for idx in itertools.product(*(range(g) for g in grid)):
        box = tuple(s if b is None else s * size
                    for b, s, size in zip(block_shape, index_map(*idx), sizes))
        boxes.add(box)
        for d, s in enumerate(box):
            starts[d].append(s)
    for d, (size, dim) in enumerate(zip(sizes, array_shape)):
        if sorted(set(starts[d])) != list(range(0, dim, size)):
            return (f"dim {d}: block starts {sorted(set(starts[d]))} are not"
                    f" multiples of {size} covering array dim {dim}")
    if len(boxes) != n_tiles:
        return (f"{len(boxes)} distinct blocks cover {n_tiles} tiles of"
                f" {array_shape} — some region is left uncovered")
    if one_writer and len(boxes) != n_grid:
        return (f"{n_grid} programs write {len(boxes)} distinct blocks —"
                " two programs write the same region")
    return ""


@pytest.mark.parametrize("case", K.CASES, ids=lambda c: c["dtype"])
def test_pallas_block_specs_tile_every_array_dimension(case):
    """Block specs have one dim per array dim and tile it exactly once.

    Covers the head axis explicitly: ``H`` is the grid's first dimension and a
    squeezed block dim, so a program with grid index ``h`` reads head ``h`` and
    no grid index reaches outside ``range(H)``.  Checked for every ``bn`` the
    sieve admits, since ``bn`` moves the grid's second dimension.
    """
    H, C, N = case["H"], case["C"], case["dk"] + case["dv"]
    array_shapes = [(H, C, C), (H, C, N), (H, C, N)]  # args[0], args[1], out

    for bn in K.TUNE_SPACE["bn"]:
        if not K.valid_config(case, dict(K.DEFAULT_PARAMS, bn=bn)):
            continue
        grid, in_specs, out_specs = K._block_specs(H, C, N, bn)
        for spec, shape, one_writer in zip(
            [*in_specs, out_specs], array_shapes, [False, False, True]
        ):
            defect = _tile_defect(
                spec.block_shape, spec.index_map, grid, shape, one_writer=one_writer
            )
            assert not defect, (
                f"{case['dtype']} bn={bn}: block_shape={spec.block_shape} on"
                f" {shape}: {defect}"
            )


@pytest.mark.parametrize("case", K.CASES, ids=lambda c: c["dtype"])
def test_pallas_call_traces_on_the_host(case):
    """The real ``pallas_call`` traces here — the GB10 rank check runs on CPU.

    Tracing calls ``BlockSpec.to_block_mapping`` for every spec, which is where
    jax rejects a block shape of the wrong rank.  A GPU-less host cannot execute
    the kernel, but it can build it, and a spec that does not build cannot run.
    """
    shapes = jax.eval_shape(lambda: K.make_inputs(jax.random.key(0), **case))
    fn = K._build_kernel(
        case["H"], case["C"], case["dk"] + case["dv"],
        jnp.dtype(case["dtype"]), **K.DEFAULT_PARAMS,
    )
    fn.trace(*shapes)  # ValueError: "Block shape for args[0] ..." if ranks differ


@pytest.mark.parametrize("case", K.CASES, ids=lambda c: c["dtype"])
def test_pallas_layout_is_numerically_right_on_the_host(case):
    """Execute the block/index/grid layout on the host, in Pallas' interpreter.

    The parity test above dispatches to :func:`solve_jax`, so it pins the
    *arithmetic* and nothing about the layout.  Here the ``pallas_call`` itself
    runs — same body, same index maps, same grid — with the interpreter rather
    than the Triton compiler.  An index map that starts a block at the wrong
    offset, or a grid that leaves a region of the array unowned, then shows up
    as a mismatch or a NaN here instead of only on the GB10.

    This is stronger than ``lower_check.py`` on a host without a GPU: that
    lowers ``kernel``, which dispatches to the host program, so the block specs
    never reach the lowerer at all.
    """
    atol, rtol = K.TOLERANCE[case["dtype"]]
    fn = K._build_kernel(
        case["H"], case["C"], case["dk"] + case["dv"],
        jnp.dtype(case["dtype"]), interpret=True, **K.DEFAULT_PARAMS,
    )
    for seed in SEEDS[:2]:  # the layout does not depend on the seed
        a, b = K.make_inputs(jax.random.key(seed), **case)
        x = fn(a, b)
        ref = K.reference(a, b)
        assert x.shape == ref.shape
        assert bool(jnp.isfinite(x.astype(jnp.float32)).all()), (
            f"{case['dtype']} seed {seed}: unowned/unwritten region in the output"
        )
        bad = _frac_bad(x, ref, atol, rtol)
        assert bad == 0.0, (
            f"{case['dtype']} seed {seed}: {bad:.2%} of entries outside "
            f"atol={atol:g} rtol={rtol:g} (max_abs={_max_abs(x, ref):.3g})"
        )


# ---------------------------------------------------------------------------
# 4. Precision: the f32 body must not fall back to tf32 (ADR-010, C-042)
#
# The GPU verdict of case0 (f32) came back at max_abs ~ 3.6e-3 with 61 % of
# entries outside 1e-4 — tf32, not the algorithm.  check_kernel.py runs on the
# GB10 *without* this suite's conftest, so the ambient
# jax_default_matmul_precision is unset there; a bare ``a @ b`` in the body then
# traced dot_general with precision=None, which the Pallas-Triton lowerer maps
# to (DEFAULT, DEFAULT) and Triton resolves to tf32 for f32 operands.  These
# tests read the traced body and assert the explicit pin, so the regression
# cannot come back silently — the GB10 is the only other place it shows up.
# ---------------------------------------------------------------------------

#: The preset the f32 case must carry: the lowerer's explicit IEEE branch
#: (``tt.dot(input_precision=IEEE)``), i.e. the ``input_precision="ieee"`` of
#: the task.  Written out literally so a change of the module constant cannot
#: silently relax this test.
_IEEE = jax.lax.DotAlgorithmPreset.F32_F32_F32


def _body_dot_precisions(case):
    """``precision`` of every ``dot_general`` in the traced ``pallas_call`` body.

    The body jaxpr sits under ``jit -> pallas_call -> jaxpr``.  ``jnp.matmul``
    and ``lax.dot`` both lower to ``dot_general``, and its ``precision`` is what
    the Pallas-Triton lowerer turns into Triton's ``input_precision`` — so for
    f32, ``None``/``(DEFAULT, DEFAULT)``/``(HIGH, HIGH)`` mean tf32 and
    ``F32_F32_F32`` means IEEE.  Reading it back is reading the kernel's verdict.
    """
    shapes = jax.eval_shape(lambda: K.make_inputs(jax.random.key(0), **case))
    fn = K._build_kernel(
        case["H"], case["C"], case["dk"] + case["dv"],
        jnp.dtype(case["dtype"]), **K.DEFAULT_PARAMS,
    )
    found = []

    def walk(jaxpr):
        for eqn in jaxpr.eqns:
            if eqn.primitive.name == "dot_general":
                found.append(eqn.params["precision"])
            for key in ("jaxpr", "compute_jaxpr", "call_jaxpr"):
                value = eqn.params.get(key)
                if value is None:
                    continue
                # both spellings occur: the outer ``jit`` carries a ClosedJaxpr,
                # the ``pallas_call`` carries the body as a bare Jaxpr.
                sub = getattr(value, "jaxpr", value)
                if hasattr(sub, "eqns"):
                    walk(sub)
                elif isinstance(sub, (list, tuple)):
                    for item in sub:
                        item = getattr(item, "jaxpr", item)
                        if hasattr(item, "eqns"):
                            walk(item)

    walk(jax.make_jaxpr(fn)(*shapes).jaxpr)
    return found


def test_f32_body_pins_every_dot_to_ieee():
    """No matmul in the f32 body may carry a tf32-class precision.

    The mask is total over all four sites of ``_solve_head`` at once, so a new
    un-pinned matmul fails *here* instead of only on the GB10.
    """
    assert K._F32_MATMUL_PRECISION == _IEEE, (
        "the module must declare the IEEE preset as its f32 precision"
    )
    precisions = _body_dot_precisions(_case("float32"))
    assert precisions, "the body must contain the product-form matmuls"
    off = [p for p in precisions if p != _IEEE]
    assert not off, f"{len(off)}/{len(precisions)} f32 dots are not IEEE: {off}"


def test_f32_body_dot_sites_are_enumerated():
    """The unrolled body carries exactly the product form's dots, no more.

    ``N**2`` (1), the ``(I + N^(2**k))`` accumulate and the squaring per step
    (``_inverse_steps(C)`` each), and the final ``T @ bv`` (1).  The last
    ``power @ power`` is dead after the loop and dropped by DCE, so accept
    either count — this fails if a matmul is added or one stops being traced.
    """
    steps = K._inverse_steps(_case("float32")["C"])
    n = len(_body_dot_precisions(_case("float32")))
    assert n in (2 * steps + 1, 2 * steps + 2), (
        f"{n} dots in the f32 body; the product form unrolls to {2 * steps + 1} "
        f"(last square DCE'd) or {2 * steps + 2}"
    )


def test_f32_pin_is_independent_of_the_matmul_policy():
    """The f32 pin must survive a hostile ambient policy (ADR-010).

    conftest pins the policy to ``highest`` for the suite; ``check_kernel.py``
    on the GB10 runs without it (default -> tf32).  Under a tf32-class policy
    *and* under ``highest`` the f32 body must trace the same explicit preset,
    while bf16 keeps tracking the policy — the case this fix deliberately
    leaves alone.
    """
    for policy in ("default", "high"):
        with jax.default_matmul_precision(policy):
            f32 = _body_dot_precisions(_case("float32"))
            assert f32 and all(p == _IEEE for p in f32), (
                f"f32 dots changed with the ambient policy {policy!r}: {set(map(str, f32))}"
            )
            bf16 = _body_dot_precisions(_case("bfloat16"))
            assert bf16 and all(p != _IEEE for p in bf16), (
                "bf16 must keep the backend default, not the f32 IEEE pin"
            )


# ---------------------------------------------------------------------------
# 5. Backward: the transposed-solve adjoint under ``custom_vjp``
#
# The bare ``pallas_call`` has no JVP — that is exactly the "Linearization failed
# to produce known values for all output primals" error the integrated
# forward-only kernel hit on the GB10.  ``K.solve``/``K.solve_t`` add the analytic
# adjoint (module docstring, "Backward"): for ``X = M^{-1} B`` (``M = I + L``) the
# cotangents are ``dB = T^T dX`` and ``dL = strict_lower(-T^T dX X^T)``, with
# ``T^T`` taken from the *sibling* kernel (``kernel_t``), not from
# ``jnp.linalg.inv``.  These tests pin that numerics, the mask, and the routing.
# ---------------------------------------------------------------------------

#: CPU gradcheck tolerances.  Both sides are f32 but they are *different*
#: algorithms — the forward here is the Neumann product form (max_abs ~ 2e-7 vs
#: LU on the task geometry) and the oracle is ``jnp.linalg.solve`` autodiff.  On
#: the task shape (H=2, C=8, d=4) over seeds 0..5, both orientations, the measured
#: worst |grad diff| is 9.5e-7 (relative 1.6e-5 over entries with |ref| >= 1e-3);
#: a (H, C, N) sweep {(2,8,4),(2,16,8),(4,8,4),(2,32,16)} x 6 seeds measures worst
#: |grad diff| = 4.3e-6.  ``atol=1e-4`` keeps >= 20x over the whole sweep's worst,
#: ``rtol=1e-3`` is > 60x over the measured relative error — a declared tolerance,
#: not a fitted one.
GRADCHECK_ATOL = 1e-4
GRADCHECK_RTOL = 1e-3

#: The task's gradcheck geometry: ``H`` heads, ``C`` chunk, ``d = dk + dv`` RHS
#: width.  Small enough to run on CPU and to keep the LU oracle and the product
#: form within a few f32 ulps.
GRADCHECK_SHAPE = dict(H=2, C=8, dk=2, dv=2)


def _jaxpr_primitives(fn, *args):
    """Set of primitive names in the (nested) jaxpr of ``fn(*args)``."""
    found = set()

    def walk(jaxpr):
        for eqn in jaxpr.eqns:
            found.add(eqn.primitive.name)
            for key in ("jaxpr", "compute_jaxpr", "call_jaxpr"):
                value = eqn.params.get(key)
                if value is None:
                    continue
                sub = getattr(value, "jaxpr", value)
                if hasattr(sub, "eqns"):
                    walk(sub)
                elif isinstance(sub, (list, tuple)):
                    for item in sub:
                        item = getattr(item, "jaxpr", item)
                        if hasattr(item, "eqns"):
                            walk(item)

    walk(jax.make_jaxpr(fn)(*args).jaxpr)
    return found


def _constrained_oracle(a, b, transpose=False):
    """``jnp.linalg.solve`` of the *unit-lower* map ``A -> I + tril(A, -1)``.

    Differentiating this, rather than the raw ``jnp.linalg.solve(a, b)``, gives
    the oracle the same domain as the kernel: ``A`` enters only through
    ``tril(A, -1)``, so the oracle's gradient is exactly zero on the diagonal and
    the upper triangle, exactly like a correct constrained ``dA``.  This is the
    comparison that pins a *correct* gradient; the mask's shape is checked
    separately against the unconstrained oracle in
    :func:`test_backward_dA_is_masked_to_strict_lower`.
    """
    m = jnp.eye(a.shape[-1], dtype=jnp.float32) + jnp.tril(a.astype(jnp.float32), -1)
    if transpose:
        m = jnp.swapaxes(m, -1, -2)
    return jnp.linalg.solve(m, b.astype(jnp.float32))


def _raw_oracle(a, b, transpose=False):
    """Unconstrained ``jnp.linalg.solve`` — mass on the whole triangle."""
    m = a.astype(jnp.float32)
    if transpose:
        m = jnp.swapaxes(m, -1, -2)
    return jnp.linalg.solve(m, b.astype(jnp.float32))


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_transposed_solve_matches_reference(dtype):
    """Forward parity is not broken by the new path: ``kernel_t`` == ``solve(a.T)``."""
    atol, rtol = K.TOLERANCE[dtype]
    for seed in SEEDS:
        a, b = K.make_inputs(jax.random.key(seed), **_case(dtype))
        x = K.kernel_t(a, b)
        ref = K.reference_t(a, b)
        assert x.shape == ref.shape and x.dtype == ref.dtype
        assert _frac_bad(x, ref, atol, rtol) == 0.0, (
            f"kernel_t {dtype} seed {seed}: max_abs={_max_abs(x, ref):.3g}"
        )


def test_transposed_baseline_is_the_right_target():
    """The replacement target for the transposed solve agrees with LU."""
    case = _case("float32")
    a, b = K.make_inputs(jax.random.key(0), **case)
    assert _max_abs(K.baseline_t(a, b), K.reference_t(a, b)) == 0.0


def test_solve_wrapper_forward_equals_the_bare_kernel():
    """``custom_vjp`` wrapping must not perturb the forward value."""
    a, b = K.make_inputs(jax.random.key(1), **GRADCHECK_SHAPE, dtype="float32")
    assert _max_abs(K.solve(a, b), K.kernel(a, b)) == 0.0
    assert _max_abs(K.solve_t(a, b), K.kernel_t(a, b)) == 0.0


@pytest.mark.parametrize("transpose", [False, True], ids=["forward", "transpose"])
@pytest.mark.parametrize("seed", SEEDS)
def test_gradcheck_custom_vjp_matches_linalg_solve(transpose, seed):
    """``jax.grad`` of the custom_vjp path == ``jnp.linalg.solve`` autodiff (both args)."""
    a, b = K.make_inputs(jax.random.key(seed), **GRADCHECK_SHAPE, dtype="float32")
    f = K.solve_t if transpose else K.solve
    g = jax.random.normal(jax.random.key(1000 + seed), b.shape, jnp.float32)
    da_c, db_c = jax.grad(lambda a, b: jnp.sum(f(a, b) * g), argnums=(0, 1))(a, b)
    da_r, db_r = jax.grad(
        lambda a, b: jnp.sum(_constrained_oracle(a, b, transpose) * g), argnums=(0, 1)
    )(a, b)
    assert jnp.allclose(db_c, db_r, atol=GRADCHECK_ATOL, rtol=GRADCHECK_RTOL), (
        f"dB (transpose={transpose}, seed={seed}) max|d|={float(jnp.abs(db_c - db_r).max()):.3e}"
    )
    assert jnp.allclose(da_c, da_r, atol=GRADCHECK_ATOL, rtol=GRADCHECK_RTOL), (
        f"dA (transpose={transpose}, seed={seed}) max|d|={float(jnp.abs(da_c - da_r).max()):.3e}"
    )


@pytest.mark.parametrize("transpose", [False, True], ids=["forward", "transpose"])
def test_backward_dA_is_masked_to_strict_lower(transpose):
    """``dA`` is exactly 0 outside the strictly-lower block; inside it is the true grad.

    ``A`` enters the map only through ``tril(A, -1)``, so the unit diagonal and
    the zero upper triangle are not parameters.  The strictly-lower block is
    compared against the *unconstrained* ``jnp.linalg.solve`` gradient, which puts
    non-zero mass there — agreement on that block plus exact zeros elsewhere is
    the whole point of the mask.
    """
    C = GRADCHECK_SHAPE["C"]
    a, b = K.make_inputs(jax.random.key(0), **GRADCHECK_SHAPE, dtype="float32")
    f = K.solve_t if transpose else K.solve
    g = jax.random.normal(jax.random.key(7), b.shape, jnp.float32)
    da_c, _ = jax.grad(lambda a, b: jnp.sum(f(a, b) * g), argnums=(0, 1))(a, b)
    da_u, _ = jax.grad(
        lambda a, b: jnp.sum(_raw_oracle(a, b, transpose) * g), argnums=(0, 1)
    )(a, b)
    mask = jnp.tril(jnp.ones((C, C), jnp.bool_), -1)
    off = float(jnp.where(~mask, jnp.abs(da_c), 0.0).max())
    assert off == 0.0, f"dA leaks {off:.3e} outside the strictly-lower block"
    same = jnp.allclose(da_c[:, mask], da_u[:, mask], atol=GRADCHECK_ATOL, rtol=GRADCHECK_RTOL)
    assert same, (
        f"strictly-lower dA (transpose={transpose}) disagrees with raw linalg.solve: "
        f"max|d|={float(jnp.abs(da_c[:, mask] - da_u[:, mask]).max()):.3e}"
    )


def test_custom_vjp_backward_routes_through_the_sibling_kernel(monkeypatch):
    """The backward runs the *other* kernel — proof the ``custom_vjp`` path is taken.

    A bare ``pallas_call`` cannot be differentiated, so if the ``custom_vjp`` were
    bypassed jax would differentiate the host solve directly and ``kernel_t`` would
    never be called.  Observing ``kernel_t`` on the ``K.solve`` backward (and
    ``kernel`` on the ``K.solve_t`` backward) therefore proves ``custom_vjp`` is in
    force *and* that the ``T^T`` multiplier comes from the transposed kernel.
    """
    calls = {"n": 0, "t": 0}
    real_n, real_t = K.kernel, K.kernel_t

    def spy_n(a, b, **params):
        calls["n"] += 1
        return real_n(a, b, **params)

    def spy_t(a, b, **params):
        calls["t"] += 1
        return real_t(a, b, **params)

    monkeypatch.setattr(K, "kernel", spy_n)
    monkeypatch.setattr(K, "kernel_t", spy_t)
    a, b = K.make_inputs(jax.random.key(0), **GRADCHECK_SHAPE, dtype="float32")
    g = jax.random.normal(jax.random.key(3), b.shape, jnp.float32)

    # forward solve: primal -> kernel; backward -> the transposed kernel
    calls.update(n=0, t=0)
    K.solve(a, b)
    assert calls == {"n": 1, "t": 0}, calls
    calls.update(n=0, t=0)
    jax.grad(lambda a, b: jnp.sum(K.solve(a, b) * g), argnums=(0, 1))(a, b)
    assert calls["n"] >= 1 and calls["t"] >= 1, calls

    # transposed solve: primal -> kernel_t; backward -> the forward kernel
    calls.update(n=0, t=0)
    K.solve_t(a, b)
    assert calls == {"n": 0, "t": 1}, calls
    calls.update(n=0, t=0)
    jax.grad(lambda a, b: jnp.sum(K.solve_t(a, b) * g), argnums=(0, 1))(a, b)
    assert calls["n"] >= 1 and calls["t"] >= 1, calls


def test_backward_is_matmul_only_no_dense_solve_primitive():
    """The backward jaxpr is the product-form adjoint, not a dense ``lu``/``inv``.

    ``jnp.linalg.inv``/``solve`` lower to ``lu`` + ``custom_linear_solve``; a
    backward that used either would show those primitives *and* lack the
    transposed product-form matmuls.  Requiring ``dot_general`` present and none
    of the dense-solve primitives present pins the mechanism, not just the numbers.
    """
    a, b = K.make_inputs(jax.random.key(0), **GRADCHECK_SHAPE, dtype="float32")
    x = K.solve(a, b)
    banned = {"lu", "custom_linear_solve", "triangular_solve", "solve"}
    for transpose in (False, True):
        names = _jaxpr_primitives(
            lambda a, x, g, t=transpose: K._backward(a, x, g, t, {}), a, x, b
        )
        assert "dot_general" in names, f"no matmul in the backward (transpose={transpose})"
        assert not (names & banned), (
            f"backward (transpose={transpose}) uses a dense solve: {names & banned}"
        )


def test_module_has_no_matrix_inverse():
    """The module calls no ``.inv`` — the multiplier is the kernel, by construction."""
    inv = [
        node.attr
        for node in ast.walk(ast.parse(open(K.__file__).read()))
        if isinstance(node, ast.Attribute) and node.attr == "inv"
    ]
    assert not inv, f"module calls .inv: {inv}"


def test_module_exposes_differentiable_entry_points():
    """``solve``/``solve_t`` are custom_vjp wrappers, distinct from the bare kernels."""
    assert K.solve is not K.kernel and K.solve_t is not K.kernel_t
    assert hasattr(K.solve, "defvjp") and hasattr(K.solve_t, "defvjp")
