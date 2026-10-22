"""MFU-фикс №1 — the *operands* of ``_dense_apply``'s matmuls follow the gate.

The MFU ladder localises the L3-MLA cell at 20.7% MFU
(``evidence/mfu-ladder/ladder-report-bf16.json``) and its ``notes`` attribute
that to the layer running in fp32.  ``net/compute_dtype.py`` already owns that
axis (``AXIOM_COMPUTE_DTYPE``), and ``net/mla.py`` routes every matmul of the
dense oracle through it — but nothing pinned *where* the cast happens, so the
pre-gate spelling (``x32 @ params.W.astype(jnp.float32)``, an fp32 matmul with
no gate in it) is indistinguishable from the gated one at the level of the
layer *output*: both return fp32.

This module pins the invariant one level lower — on the operands of the
``dot_general`` ops that JAX actually builds:

* under ``AXIOM_COMPUTE_DTYPE=bf16`` **every** ``dot_general`` operand of
  :func:`net.mla._dense_apply` is bf16, and the layer still returns fp32 (the
  Beam recipe: bf16 operands, fp32 accumulation — ``preferred_element_type``);
* under the default (``AXIOM_COMPUTE_DTYPE`` unset → ``fp32``) no operand is
  bf16 at all, so the ADR-009 oracle keeps its pinned graph;
* ``net/tests/test_29_compute_dtype.py`` remains the owner of the byte-exact
  gate-off comparison against the baseline commit — this module deliberately
  does not repeat it.

The traced arguments are passed explicitly (``make_jaxpr(f)(params, x)``) rather
than closed over: an op whose operands are *both* constants is folded during
tracing, so a closure-based check could quietly observe an empty graph instead
of the matmuls.  Passing them as arguments makes them tracers and the
``dot_general`` ops necessarily appear — the emptiness assert below is the
guard that the check still has something to look at.

``test_the_pre_fix_fp32_spelling_is_what_this_check_rejects`` keeps the teeth:
it reproduces the pre-gate spelling inline and asserts the same predicate
*rejects* it, so a future regression of exactly the kind the task describes
cannot pass unnoticed (the idiom is borrowed from
``net/tests/test_30_mla_flash_parity.py``, which pins its own fix the same way).

jax is required to trace a graph; where it is absent the module skips with a
reason rather than erroring.  Note that on such a machine the whole
``net/tests`` package fails earlier than this guard: ``net/tests/conftest.py``
imports jax unconditionally (ADR-010 pinning), so collection stops at the
conftest.  That is an environment fact, not a result of this test.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip(
    "jax",
    reason="MFU-фикс №1 проверяется на jaxpr: нужен jax, чтобы построить граф "
    "(_dense_apply) и прочитать dtype операндов dot_general",
)

import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402

from net import compute_dtype, mla  # noqa: E402

#: The gate's environment variable (a typo here would silently test the default).
GATE = compute_dtype.MODE_ENV

#: Boundary dtype under the bf16 gate (what every matmul operand must be).
BF16 = np.dtype(jnp.bfloat16)

#: Boundary result / accumulation dtype (what the layer must return).
FP32 = np.dtype(jnp.float32)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _dot_general_operands(fn, *args) -> list[tuple[np.dtype, ...]]:
    """Operand dtypes of every ``dot_general`` in the traced graph of ``fn(*args)``.

    Traced, not executed: the question is what the *graph* feeds the matmul, and
    an eager run would only show the result — which is fp32 under both modes and
    therefore cannot tell the two apart.
    """
    closed = jax.make_jaxpr(fn)(*args)
    operands: list[tuple[np.dtype, ...]] = []
    for eqn in closed.jaxpr.eqns:
        if eqn.primitive.name != "dot_general":
            continue
        operands.append(tuple(np.dtype(var.aval.dtype) for var in eqn.invars))
    return operands


def _non_bf16(operands: list[tuple[np.dtype, ...]]) -> list[np.dtype]:
    """Operands that are not bf16 — precisely the fp32-forced ones the fix removes."""
    return [dtype for dtypes in operands for dtype in dtypes if dtype != BF16]


def _dense_setup(cfg):
    """Oracle parameters and a ``(B, T, hidden)`` input for ``_dense_apply``."""
    params = mla.init_mla(jr.PRNGKey(0), cfg)
    x = jr.normal(jr.PRNGKey(1), (2, 8, cfg.hidden))
    return params, x


# --------------------------------------------------------------------------- #
# The invariant: operands follow the gate
# --------------------------------------------------------------------------- #


def test_bf16_gate_sends_every_dense_apply_matmul_operand_as_bf16(cfg, monkeypatch) -> None:
    """Under the bf16 gate no matmul operand of ``_dense_apply`` stays fp32."""
    params, x = _dense_setup(cfg)
    monkeypatch.setenv(GATE, compute_dtype.BF16)

    operands = _dot_general_operands(
        lambda p, xx: mla._dense_apply(p, cfg, xx), params, x
    )

    assert operands, (
        "в графе _dense_apply не осталось ни одного dot_general — проверка "
        "потеряла зубы (матмулы переписаны мимо JAX или сложились в константу)"
    )
    assert _non_bf16(operands) == [], (
        "под AXIOM_COMPUTE_DTYPE=bf16 операнд матмула остался fp32: "
        f"dot_general операнды = {operands}"
    )


def test_default_gate_keeps_every_dense_apply_matmul_operand_fp32(cfg, monkeypatch) -> None:
    """Gate off (the documented default): the graph carries no bf16 operand."""
    params, x = _dense_setup(cfg)
    monkeypatch.delenv(GATE, raising=False)

    operands = _dot_general_operands(
        lambda p, xx: mla._dense_apply(p, cfg, xx), params, x
    )

    assert operands, "в графе _dense_apply не осталось ни одного dot_general"
    offenders = [dtype for dtypes in operands for dtype in dtypes if dtype != FP32]
    assert offenders == [], (
        f"fp32-режим (гейт выключен) обязан оставаться fp32, а в графе: {operands}"
    )


def test_dense_apply_returns_fp32_under_both_gates(cfg, monkeypatch) -> None:
    """bf16 operands must accumulate *and return* in fp32 (the residual contract)."""
    params, x = _dense_setup(cfg)
    for mode in (compute_dtype.FP32, compute_dtype.BF16):
        monkeypatch.setenv(GATE, mode)
        out = jax.eval_shape(lambda p, xx: mla._dense_apply(p, cfg, xx), params, x)
        assert out.dtype == jnp.float32, (mode, out.dtype)


def test_the_public_apply_used_by_the_mfu_ladder_is_gated_too(cfg, monkeypatch) -> None:
    """The ladder calls ``mla.apply`` (dense oracle by default) — same invariant."""
    params, x = _dense_setup(cfg)
    monkeypatch.setenv(GATE, compute_dtype.BF16)

    operands = _dot_general_operands(lambda p, xx: mla.apply(p, cfg, xx), params, x)

    assert operands, "mla.apply не построил ни одного dot_general"
    assert _non_bf16(operands) == [], (
        f"mla.apply под bf16 оставил операнд матмула в fp32: {operands}"
    )


# --------------------------------------------------------------------------- #
# Teeth: the pre-fix spelling must be rejected by the very same predicate
# --------------------------------------------------------------------------- #


def test_the_pre_fix_fp32_spelling_is_what_this_check_rejects(cfg, monkeypatch) -> None:
    """Reproduce the fp32-forced matmul inline; the invariant must reject it.

    This is the regression the task is about: ``x32 @ W.astype(jnp.float32)``
    bypasses the gate, so its operands are fp32 even with the bf16 gate on —
    while its *output* is fp32 too, which is why an output-level check would
    have called it correct.
    """
    params, x = _dense_setup(cfg)
    w = params.W_q
    monkeypatch.setenv(GATE, compute_dtype.BF16)

    pre_fix = _dot_general_operands(
        lambda a, b: a.astype(jnp.float32) @ b.astype(jnp.float32), x, w
    )
    assert pre_fix == [(FP32, FP32)], pre_fix
    assert _non_bf16(pre_fix) != [], "проверка не поймала форс fp32 — зубы потеряны"

    fixed = _dot_general_operands(
        lambda a, b: compute_dtype.gemm(a.astype(jnp.float32), b.astype(jnp.float32)),
        x,
        w,
    )
    assert fixed == [(BF16, BF16)], fixed
    assert _non_bf16(fixed) == [], fixed
