"""Normalization and activation primitives.

RMSNorm (head-wise / weight-wise), L2Norm and Swish are used verbatim by the
KDA and Gated-MLA layers (kda-formulas.md section 1-2).
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp


def swish(x: jnp.ndarray) -> jnp.ndarray:
    """Swish(x) = x * sigmoid(x)."""
    return x * jax.nn.sigmoid(x)


@partial(jax.custom_jvp, nondiff_argnums=(1, 2))
def _l2_norm_guarded(x: jnp.ndarray, eps: float, axis: int) -> jnp.ndarray:
    """Primal of :func:`l2_norm`: ``x / (‖x‖ + eps)``, verbatim (ADR-010)."""
    return x / (jnp.linalg.norm(x, axis=axis, keepdims=True) + eps)


@_l2_norm_guarded.defjvp
def _l2_norm_guarded_jvp(eps, axis, primals, tangents):
    """Analytic JVP of ``x / (‖x‖ + eps)``, regularised at the origin.

    ``d‖x‖ = xᵀẋ / ‖x‖`` is ``0/0`` when ``x = 0``; multiplying it by a zero
    cotangent still yields NaN in reverse mode (``0 · NaN = NaN``), which is
    exactly how padded KDA rows poison the gradient.  The norm has no
    derivative at the origin, so the tangent there is *defined* as zero (the
    standard subgradient convention): every ``x ≠ 0`` keeps the exact analytic
    derivative, so forward *and* backward are unchanged off the singular set.
    """
    (x,), (x_dot,) = primals, tangents
    norm = jnp.linalg.norm(x, axis=axis, keepdims=True)
    nonsingular = norm > 0.0
    # ``where`` keeps the guarded quotient off the singular point: at x = 0 the
    # unit vector is unmasked to 0 instead of 0/0.
    unit = x * jnp.reciprocal(jnp.where(nonsingular, norm, 1.0))
    denom = norm + eps  # > 0 thanks to eps
    proj = jnp.sum(unit * x_dot, axis=axis, keepdims=True)
    # d[x/(‖x‖+eps)] = ẋ/(‖x‖+eps) − x (xᵀẋ)/(‖x‖(‖x‖+eps)²)
    tangent = (x_dot - x * proj / denom) / denom
    tangent = jnp.where(nonsingular, tangent, jnp.zeros_like(tangent))
    return _l2_norm_guarded(x, eps, axis), tangent


def l2_norm(x: jnp.ndarray, eps: float = 1e-12, axis: int = -1) -> jnp.ndarray:
    """L2-normalise along ``axis`` (used for KDA q/k per head).

    Forward is bit-identical to ``x / (‖x‖ + eps)``; the only difference from a
    plain expression is the tangent at a strictly zero vector, which is zero
    instead of NaN (see :func:`_l2_norm_guarded_jvp`).  Padded rows of
    ``kda.apply_chunked`` hit that point, and a NaN there reaches the KDA
    parameters (``W_q/W_k/conv_q/conv_k``) even though the rows carry no
    upstream gradient.
    """
    return _l2_norm_guarded(x, eps, axis)


def rms_norm(x: jnp.ndarray, weight: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """Weighted RMSNorm: ``x / sqrt(mean(x^2) + eps) * weight``."""
    ms = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    return x * jnp.reciprocal(jnp.sqrt(ms + eps)) * weight


def headwise_rms_norm(x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """Unweighted RMSNorm per head (KDA output, Eq. 6)."""
    ms = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    return x * jnp.reciprocal(jnp.sqrt(ms + eps))
