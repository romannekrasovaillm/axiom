"""ShortConv — causal depthwise 1D convolution (KDA Eq. 2).

``short_conv`` applies a causal depthwise convolution over the sequence of
per-head projections.  ``short_conv_step`` is the streaming (recurrent) form
that carries the last ``K-1`` inputs; the two are numerically identical and
this is exercised by the KDA parity tests.
"""

from __future__ import annotations

import jax.numpy as jnp


def short_conv(x: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    """Causal depthwise conv over a sequence.

    Args:
        x: (T, D) input sequence.
        w: (K, D) depthwise kernel, ``w[i]`` weights the input ``i`` steps back.

    Returns:
        (T, D) where ``out[t] = sum_i w[i] * x[t-i]`` (``x[t<0] = 0``).
    """
    k = w.shape[0]
    t = x.shape[0]
    xp = jnp.pad(x, ((k - 1, 0), (0, 0)))  # left causal padding
    out = jnp.zeros_like(x)
    for i in range(k):
        out = out + w[i][None, :] * xp[k - 1 - i : t + k - 1 - i]
    return out


def short_conv_step(x_t: jnp.ndarray, w: jnp.ndarray, buf: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Streaming ShortConv step.

    Args:
        x_t: (D,) current input.
        w: (K, D) depthwise kernel.
        buf: (K-1, D) the last ``K-1`` inputs, oldest first.

    Returns:
        (out_t, new_buf) with ``out_t`` (D,) and ``new_buf`` (K-1, D).
    """
    k = w.shape[0]
    if k == 1:
        return w[0] * x_t, buf
    # buf: (K-1, D), oldest-first: buf[j] = x_{t-K+1+j}.
    # out_t = w[0]*x_t + sum_{j=0}^{K-2} w[K-1-j] * buf[j]
    hist = jnp.einsum("kd,kd->d", w[:0:-1], buf)
    out_t = w[0] * x_t + hist
    new_buf = jnp.concatenate([buf[1:], x_t[None, :]], axis=0)
    return out_t, new_buf


def init_short_conv_buf(kernel: int, dim: int) -> jnp.ndarray:
    """Zero-initialised ShortConv buffer of shape (K-1, D)."""
    return jnp.zeros((kernel - 1, dim))
