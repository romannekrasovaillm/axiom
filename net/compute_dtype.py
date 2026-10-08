"""Compute-dtype gate — ``AXIOM_COMPUTE_DTYPE`` (BF16 campaign, phase 1).

The gate selects the dtype of a **GEMM boundary** in the model package.  It is
deliberately *not* a parameter dtype (that axis already exists in the training
loop: ``net/train_loop.py``'s ``param_dtype`` casts a *working copy* of the
master tree) and it is not the checkpoint format or the metrics-journal schema,
neither of which this module can reach.

Modes
-----

``fp32`` (default)
    The pinned compute path.  :func:`gemm` / :func:`gemm_einsum` return the
    expression the caller had before the gate existed (``a @ b`` /
    ``jnp.einsum(eq, *ops)``), so the graph is bit-for-bit the pre-gate graph
    (baseline ``773e389``).  The oracle path of ADR-009 is byte-exact under the
    gate off by construction, not by tolerance.

``bf16``
    The Beam recipe: the **operands** of a dense matmul or an attention
    contraction are cast to bf16 at the boundary, the product **accumulates and
    returns in fp32** (``preferred_element_type=jnp.float32``).  Nothing else
    changes dtype — the residual stream, the normalisation, the activations,
    the loss path and the optimizer's master tree stay fp32.  Concretely:

    * *master weights stay fp32* — the gate never touches a parameter leaf; it
      casts the operand of a GEMM, and the weight a caller passes in is cast
      inside the boundary only.  ``optimizer.init_state`` (momentum/AdamW) and
      the gradient accumulation in ``net/train_loop.py`` therefore keep their
      fp32 dtype;
    * *residual connections stay fp32* — every block returns an fp32 tensor
      (the GEMM output dtype is forced to fp32), so ``h + delta + corr`` adds
      fp32 to fp32;
    * *casts happen only at the GEMM input boundary* — no other site in the
      model package introduces a dtype conversion, so the loss path cannot pick
      up a silent promotion or demotion.

Fail-closed
-----------

An unrecognised ``AXIOM_COMPUTE_DTYPE`` value raises :class:`ComputeDtypeError`
instead of silently running fp32.  A typo that quietly fell back to fp32 would
invalidate a bf16 campaign measurement; a typo that quietly ran bf16 would
perturb the pinned oracle.  Neither is acceptable, so the gate refuses to guess
(the opposite of ``net/tests/conftest.py``'s backend selector, whose fallback
direction is safe because both branches are exercised by the gate profile).

Not covered (declared, so the boundary is reviewable): ``net/vit.py`` (the
vision tower runs on image tokens, not on the pretrain compute axis).  The
MXFP4 QAT quantiser (``net/quant.py``), the checkpoint format and the journal
schema are out of scope by the task and are not referenced here.
"""

from __future__ import annotations

import os

import jax.numpy as jnp

#: The gate's environment variable.
MODE_ENV = "AXIOM_COMPUTE_DTYPE"

#: ``fp32`` — the pinned compute path (default; the pre-gate graph verbatim).
FP32 = "fp32"

#: ``bf16`` — bf16 GEMM operands with an fp32 accumulator (the Beam recipe).
BF16 = "bf16"

#: Declared modes, in the order the report prints them.
MODES = (FP32, BF16)

#: Mode when the variable is unset (or empty).
DEFAULT_MODE = FP32


class ComputeDtypeError(ValueError):
    """Unrecognised ``AXIOM_COMPUTE_DTYPE`` value (fail-closed, no guessing)."""


def mode() -> str:
    """The gate's mode: ``fp32`` | ``bf16`` (read per call, ADR-4 determinism).

    Read on every call rather than pinned into an import-time constant so a
    test, a bench script or a sweep can select the mode in-process without
    reloading modules.  The branch is resolved when a JAX graph is **traced**
    (the same idiom as ``net/mla.py``'s flash switch), so a compiled graph
    carries the mode it was traced with.

    The caveat that follows is worth stating: ``jax.jit``'s cache is keyed on the
    function and the argument signatures, *not* on this variable, so flipping the
    gate under a live jitted function would reuse the graph traced under the old
    mode.  Entry points must therefore set the variable **before** the first
    trace — which is what they all do: the acceptance tests call the eager
    functions, ``net/train_loop.train`` builds a fresh ``jit`` per call, and the
    BF16 instruments (``tools/mfu_bf16_protocol.py``,
    ``tools/loss_parity_bf16.py``) run every cell in a process of its own.
    """
    raw = os.environ.get(MODE_ENV)
    if raw is None:
        return DEFAULT_MODE
    value = raw.strip().lower()
    if not value:
        return DEFAULT_MODE
    if value not in MODES:
        raise ComputeDtypeError(
            f"{MODE_ENV}={raw!r}: неизвестный режим вычислительного dtype; "
            f"ожидается один из {list(MODES)}"
        )
    return value


def is_bf16() -> bool:
    """True when the gate selects the bf16 GEMM boundary."""
    return mode() == BF16


def compute_dtype() -> jnp.dtype:
    """The dtype of a GEMM *operand* under the current gate (the boundary type)."""
    return jnp.bfloat16 if is_bf16() else jnp.float32


def cast_in(x: jnp.ndarray) -> jnp.ndarray:
    """Cast a GEMM *operand* to the boundary dtype (identity under ``fp32``).

    For boundaries that are not expressible as :func:`gemm` / :func:`gemm_einsum`
    — e.g. a fused attention kernel that takes (and returns) its operands
    directly.  The ``fp32`` branch returns the array untouched, so the gate-off
    graph keeps the caller's own expression.
    """
    return x.astype(jnp.bfloat16) if is_bf16() else x


def cast_out(x: jnp.ndarray) -> jnp.ndarray:
    """Cast a boundary *result* back to fp32 (identity under ``fp32``).

    A fused kernel run in bf16 returns bf16; leaving that in the residual stream
    would let the next ``fp32 * bf16`` promotion happen outside a boundary.  The
    result of every GEMM boundary is fp32 by contract, so this restores it
    explicitly rather than relying on promotion.
    """
    return x.astype(jnp.float32) if is_bf16() else x


def gemm(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Dense matmul with the gate's boundary cast.

    ``fp32`` returns ``a @ b`` — the caller's own expression, character for
    character, so the gate-off graph is identical to the pre-gate one.  ``bf16``
    casts both operands at the boundary and accumulates in fp32.
    """
    if mode() != BF16:
        return a @ b
    return jnp.matmul(
        a.astype(jnp.bfloat16),
        b.astype(jnp.bfloat16),
        preferred_element_type=jnp.float32,
    )


def gemm_einsum(equation: str, *operands: jnp.ndarray) -> jnp.ndarray:
    """``jnp.einsum`` with the gate's boundary cast (attention/scores contractions).

    Used where the contraction is expressed as an einsum rather than ``@`` — the
    attention score and value matmuls of the dense oracle and of the sparse
    union attention.  The fp32 mode returns the caller's own ``jnp.einsum``
    call back, so the gate-off numerics and reduction order are unchanged.
    """
    if mode() != BF16:
        return jnp.einsum(equation, *operands)
    cast = tuple(op.astype(jnp.bfloat16) for op in operands)
    return jnp.einsum(equation, *cast, preferred_element_type=jnp.float32)


def gemm_batched(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Batched matmul ``(..., M, K) @ (..., K, N)`` with the boundary cast.

    Exists because *some* einsum patterns that mix a batch axis into one operand
    only — the MoE expert application ``"nej,ejl->nel"`` is the case in this
    repo — lower to a ``dot_general`` the **CPU** backend refuses in
    bf16-with-fp32-accumulate ("Unsupported element type for DotThunk::Execute:
    BF16 x BF16 = F32"), while the equivalent ``jnp.matmul`` runs.  The math is
    the caller's; only the spelling differs, so the fp32 branch is the plain
    ``jnp.matmul`` and the gate-off graph stays the pre-gate one.
    """
    if mode() != BF16:
        return jnp.matmul(a, b)
    return jnp.matmul(
        a.astype(jnp.bfloat16),
        b.astype(jnp.bfloat16),
        preferred_element_type=jnp.float32,
    )
