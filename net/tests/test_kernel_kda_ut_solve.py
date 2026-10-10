"""Acceptance test for the first Pallas kernel: ``net/kernels/kda_ut_solve.py``.

What can be verified on a machine without a GPU (this suite):

* the **algorithm** the kernel body runs — ``kernel`` dispatches to the same
  jnp program on a non-GPU host, so ``kernel`` vs ``reference`` here is a
  numerics verdict on :func:`net.kernels.kda_ut_solve._solve_head`;
* the **contract** of the module (shapes/dtypes, ``valid_config``,
  ``TUNE_SPACE``/``cost`` well-formedness);
* the **envelope** of the Neumann product form (it needs ``||L|| < 1``; the
  KDA factor is ``I + beta * tril(Akk)`` with ``beta <= 1``, ``|Akk| <= 1``).

What it cannot: the compiled Triton kernel itself.  That is
``lower_check.py`` (static lowering, run from the stand) plus
``check_kernel.py``/``bench.py`` on the GB10 — see the module docstring and
``.arch-handoff/result.json``.
"""

from __future__ import annotations

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
