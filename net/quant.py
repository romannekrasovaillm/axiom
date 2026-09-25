"""QAT — fake-quant MXFP4 weights (straight-through estimator, pure JAX).

Per MODEL-L3-SKELETON.md section 3: QAT begins at the SFT stage (not
pretrain), weights MXFP4, activations MXFP8.  The skeleton implements only the
weight branch (MXFP4); MXFP8 activations are the full-scale follow-up.  This is
our own JAX implementation (FIDELITY section 3) — flagged via
``config.qat_enabled`` and off for pretrain.

MXFP4 (OCP microscaling): a shared power-of-two scale per block (32 elements)
with a 4-bit (sign + 3-bit mantissa) value per element.  The straight-through
estimator passes the gradient through the quantised value unchanged.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

MXFP4_BLOCK = 32
MXFP4_MANTISSA_LEVELS = 8  # 2^(3 mantissa bits)

# --- ADR-009 D3: FP4 main KV (OCP MXFP4, E2M1 element + one E4M3 scale/16 ch) --
# E2M1 representable magnitudes (sign + 2-bit exponent + 1-bit mantissa).
MXFP4_LATENT_BLOCK = 16
E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E4M3_MAX = 448.0


def fake_quant_mxfp4(w: jnp.ndarray, block_size: int = MXFP4_BLOCK) -> jnp.ndarray:
    """Fake-quantise a weight tensor to MXFP4 with a straight-through gradient.

    ``w`` may have any leading shape; quantisation is applied over the last
    axis in blocks of ``block_size``.
    """
    *lead, n = w.shape
    w2 = w.reshape(-1, block_size)
    max_abs = jnp.max(jnp.abs(w2), axis=-1, keepdims=True)
    scale = jnp.power(2.0, jnp.floor(jnp.log2(max_abs + 1e-12)))
    norm = w2 / scale
    q = jnp.clip(norm, -1.0, 1.0)
    q = jnp.round(q * (MXFP4_MANTISSA_LEVELS - 1)) / (MXFP4_MANTISSA_LEVELS - 1)
    deq = (q * scale).reshape(*lead, n)
    return w + jax.lax.stop_gradient(deq - w)


def quantize_dequantize(w: jnp.ndarray, block_size: int = MXFP4_BLOCK) -> jnp.ndarray:
    """Deterministic quantise -> dequantise round-trip (no straight-through)."""
    return jax.lax.stop_gradient(fake_quant_mxfp4(w, block_size))


def apply_fake_quant_tree(params, block_size: int = MXFP4_BLOCK):
    """Fake-quantise all weight matrices in a parameter tree.

    Covers 2-D matrices and the stacked LatentMoE expert weights (ndim >= 2);
    1-D norms and biases pass through unchanged.  Quantisation blocks run over
    the flattened tensor, so any trailing dimension is admissible.
    """
    return jax.tree_util.tree_map(
        lambda leaf: fake_quant_mxfp4(leaf, block_size) if leaf.ndim >= 2 else leaf, params
    )


def quant_error(w: jnp.ndarray, block_size: int = MXFP4_BLOCK) -> jnp.ndarray:
    """Relative L2 error of the MXFP4 round-trip: ``||q(w)-w|| / ||w||``."""
    dq = quantize_dequantize(w, block_size)
    return jnp.linalg.norm(dq - w) / (jnp.linalg.norm(w) + 1e-8)


# ---------------------------------------------------------------------------
# ADR-009 D3 — MXFP4 on the MLA latent (E2M1 + one E4M3 scale per 16 channels)
# ---------------------------------------------------------------------------


def _round_to_e2m1(v: jnp.ndarray) -> jnp.ndarray:
    """Round ``v`` (expected in [-6, 6]) to the nearest E2M1 magnitude, sign kept."""
    a = jnp.abs(v)
    levels = jnp.asarray(E2M1_LEVELS, dtype=v.dtype)
    idx = jnp.argmin(jnp.abs(a[..., None] - levels), axis=-1)
    return jnp.sign(v) * levels[idx]


def _to_e4m3(v: jnp.ndarray) -> jnp.ndarray:
    """Round-trip through E4M3 (the shared scale format), saturating at 448."""
    clipped = jnp.clip(v, -E4M3_MAX, E4M3_MAX)
    return clipped.astype(jnp.float8_e4m3fn).astype(v.dtype)


def mxfp4_latent_roundtrip(c: jnp.ndarray, block: int = MXFP4_LATENT_BLOCK) -> jnp.ndarray:
    """Quantise/dequantise the latent over blocks of ``block`` channels.

    One E4M3 scale per ``block`` channels, **no second-level global scale**
    (ADR-009 D3, V4.1-Flash section 2.4.4).  The scale maps the block maximum
    onto the E2M1 top level ``6.0``; it is itself rounded through E4M3 so the
    dequantised values stay in the declared format.
    """
    *lead, n = c.shape
    assert n % block == 0, f"latent dim {n} not divisible by MXFP4 latent block {block}"
    cb = c.reshape(*lead, n // block, block)
    max_abs = jnp.max(jnp.abs(cb), axis=-1, keepdims=True)
    scale = _to_e4m3(max_abs / 6.0)                     # zero stays zero
    safe = jnp.where(scale > 0, scale, jnp.ones_like(scale))
    q = _round_to_e2m1(cb / safe)
    deq = (jnp.where(scale > 0, q, jnp.zeros_like(q)) * safe).reshape(*lead, n)
    return deq


def fake_quant_mxfp4_latent(c: jnp.ndarray, block: int = MXFP4_LATENT_BLOCK) -> jnp.ndarray:
    """Straight-through fake-quant of the latent ``c_t`` in MXFP4 (ADR-009 D3).

    Dequantised values are used in the forward pass; the gradient passes
    through the quantiser unchanged (QAT).
    """
    deq = mxfp4_latent_roundtrip(c, block)
    return c + jax.lax.stop_gradient(deq - c)


def latent_quant_error(c: jnp.ndarray, block: int = MXFP4_LATENT_BLOCK) -> jnp.ndarray:
    """Relative L2 error of the MXFP4 latent round-trip."""
    deq = mxfp4_latent_roundtrip(c, block)
    return jnp.linalg.norm(deq - c) / (jnp.linalg.norm(c) + 1e-8)


# ---------------------------------------------------------------------------
# SWA KV — stays FP8 (E4M3), per ADR-009 D3 / V4.1-Flash section 2.4.4
# ---------------------------------------------------------------------------


def fake_quant_fp8_e4m3(x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """Straight-through fake-quant of the SWA KV in FP8 (E4M3).

    A power-of-two per-vector scale keeps the block maximum inside the E4M3
    range; the format itself is the declared 8-bit float (no second-level
    scale, matching the SWA-KV convention of the source).
    """
    max_abs = jnp.max(jnp.abs(x), axis=axis, keepdims=True)
    exp = jnp.ceil(jnp.log2(jnp.maximum(max_abs, 1e-30) / E4M3_MAX))
    scale = jnp.power(2.0, exp)
    deq = _to_e4m3(x / scale) * scale
    return x + jax.lax.stop_gradient(deq - x)
