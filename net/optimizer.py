"""Optimizer — Per-Head Muon + weight clipping, cosine LR + 1% warmup, wd 0.1.

Follows MODEL-L3-SKELETON.md section 3 and kda-formulas.md section 5:

* **Muon** orthogonalises the momentum of matrix parameters via Newton-Schulz
  (:661-671).  Attention projections (Q/K/V) use the **per-head** variant: the
  momentum is split along the head axis and orthogonalised per head.
* 1-D parameters (norms, biases) use AdamW.
* Stacked expert weights (n, d_in, d_out) are orthogonalised per matrix
  (batched Newton-Schulz over the leading axis).
* **Weight clipping** (MuonClip, our value since the exact clip is not
  disclosed) clamps matrix magnitudes after each step.
* Cosine schedule with a 1% linear warmup and weight decay 0.1 (:772-773).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# Newton-Schulz order-5 coefficients (Muon).
NS_COEFFS = (3.4445, -4.7750, 2.0315)

# Attention-projection leaf names that get per-head orthogonalisation.
_PER_HEAD_LEAVES = {"W_q", "W_k", "W_v", "W_k_up", "W_v_up"}


def newtonschulz5(g: jnp.ndarray, steps: int = 5, eps: float = 1e-7) -> jnp.ndarray:
    """Newton-Schulz orthogonalisation of a 2-D matrix."""
    a, b, c = NS_COEFFS
    x = g / (jnp.linalg.norm(g) + eps)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        A = x @ x.T
        B = b * A + c * (A @ A)
        x = a * x + B @ x
    if transposed:
        x = x.T
    return x


def per_head_newtonschulz5(g: jnp.ndarray, heads: int) -> jnp.ndarray:
    """Orthogonalise a Q/K/V momentum matrix ``(dim_in, heads*dim_head)`` per head."""
    dim_in = g.shape[0]
    dim_head = g.shape[-1] // heads
    g = g.reshape(dim_in, heads, dim_head).transpose(1, 0, 2)  # (H, dim_in, dim_head)
    out = jax.vmap(newtonschulz5)(g)  # (H, dim_in, dim_head)
    return out.transpose(1, 0, 2).reshape(dim_in, heads * dim_head)


def init_state(params):
    """Momentum (matrix) / AdamW m,v (vector) state trees mirroring ``params``."""
    def _init(leaf):
        if leaf.ndim >= 2:
            return jnp.zeros_like(leaf)
        return (jnp.zeros_like(leaf), jnp.zeros_like(leaf))
    return jax.tree_util.tree_map(_init, params)


def make_step(cfg, muon_momentum: float = 0.95, adam_b1: float = 0.9, adam_b2: float = 0.95, weight_clip: float | None = 1.0):
    """Build a jittable optimizer step closing over the hyperparameters."""
    heads = cfg.num_heads
    wd = cfg.weight_decay

    def _matrix_step(p, g, m, lr, per_head):
        m = muon_momentum * m + (1.0 - muon_momentum) * g
        upd = per_head_newtonschulz5(m, heads) if per_head else newtonschulz5(m)
        p = p * (1.0 - lr * wd) - lr * upd
        if weight_clip is not None:
            p = jnp.clip(p, -weight_clip, weight_clip)
        return p, m

    def _batched_matrix_step(p, g, m, lr):
        """Muon for stacked matrices (n, d_in, d_out) — LatentMoE expert weights."""
        m = muon_momentum * m + (1.0 - muon_momentum) * g
        flat = m.reshape(-1, *m.shape[-2:])
        upd = jax.vmap(newtonschulz5)(flat).reshape(m.shape)
        p = p * (1.0 - lr * wd) - lr * upd
        if weight_clip is not None:
            p = jnp.clip(p, -weight_clip, weight_clip)
        return p, m

    def _vector_step(p, g, st, lr):
        m, v = st
        m = adam_b1 * m + (1.0 - adam_b1) * g
        v = adam_b2 * v + (1.0 - adam_b2) * (g * g)
        mh = m / (1.0 - adam_b1)
        vh = v / (1.0 - adam_b2)
        p = p * (1.0 - lr * wd) - lr * mh / (jnp.sqrt(vh) + 1e-8)
        return p, (m, v)

    def step(params, grads, state, lr):
        def _leaf_name(entry):
            return getattr(entry, "name", None) or getattr(entry, "key", None) or getattr(entry, "idx", None) or ""

        def _one(path, p, g, s):
            name = _leaf_name(path[-1]) if path else ""
            if p.ndim == 2:
                per_head = name in _PER_HEAD_LEAVES
                return _matrix_step(p, g, s, lr, per_head)  # (new_p, momentum)
            if p.ndim >= 3:
                return _batched_matrix_step(p, g, s, lr)  # (new_p, momentum)
            return _vector_step(p, g, s, lr)  # (new_p, (m, v))

        # Two passes so params and state come back as separate trees (the tuple
        # return of ``_one`` is otherwise treated as nested pytree structure).
        new_params = jax.tree_util.tree_map_with_path(
            lambda path, p, g, s: _one(path, p, g, s)[0], params, grads, state
        )
        new_state = jax.tree_util.tree_map_with_path(
            lambda path, p, g, s: _one(path, p, g, s)[1], params, grads, state
        )
        return new_params, new_state

    return step


def cosine_schedule(peak_lr: float, total_steps: int, warmup_ratio: float = 0.01, min_ratio: float = 0.0):
    """Cosine decay with a linear warmup; returns ``lr(step)`` as a function."""
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_at(step: int) -> jnp.ndarray:
        step = jnp.asarray(step, dtype=jnp.float32)
        warm = peak_lr * (step / warmup_steps)
        t = (step - warmup_steps) / jnp.maximum(total_steps - warmup_steps, 1.0)
        t = jnp.clip(t, 0.0, 1.0)
        cos = peak_lr * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + jnp.cos(jnp.pi * t)))
        return jnp.where(step < warmup_steps, warm, cos)

    return lr_at
