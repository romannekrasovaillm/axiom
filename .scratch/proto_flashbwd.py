"""Прототип jax-уровня flash-backward MLA (веха-12): custom_vjp на (q,k,v,mask).

Цель прототипа — проверить три вещи до правок net/:
1) forward кастомного vjp побитово равен dense-выражению;
2) backward (тайловый, внешний цикл по KV-тайлам, внутренний по Q) даёт те же
   градиенты, что autodiff dense-пути (допуск);
3) в jaxpr backward нет полного (B,H,T,T), только тайлы.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

TILE_TAG = "TILE"


def _gemm_einsum(eq, *ops):
    # прототип: без гейта compute-dtype (fp32)
    return jnp.einsum(eq, *ops)


def _attn_out_flashbwd_fwd(q, k, v, mask, tile):
    dq = q.shape[-1]
    scale = 1.0 / jnp.sqrt(dq)
    scores = _gemm_einsum("bthd,bshd->bhts", q, k) * scale
    masked = jnp.where(mask, scores, jnp.finfo(jnp.float32).min)
    attn = jax.nn.softmax(masked, axis=-1)
    o = _gemm_einsum("bhts,bshd->bthd", attn, v)
    rowmax = jnp.max(masked, axis=-1, keepdims=True)   # (B,H,T,1)
    rowsum = jnp.sum(jnp.exp(masked - rowmax), axis=-1, keepdims=True)
    return o, (q, k, v, mask, o, rowmax, rowsum)


def _attn_out_flashbwd_bwd(res, g, tile):
    q, k, v, mask, o, rowmax, rowsum = res
    B, T, H, dq = q.shape
    scale = 1.0 / jnp.sqrt(dq)
    # D_i = rowsum_s(do_i ⊙ o_i) = rowsum(dp ⊙ p) — нужен до циклов
    d = jnp.sum(g * o, axis=-1)                     # (B,T,H)
    dq_acc = jnp.zeros_like(q)                      # (B,T,H,dq)
    dk_acc = jnp.zeros_like(k)
    dv_acc = jnp.zeros_like(v)
    j_starts = range(0, T, tile)
    i_starts = range(0, T, tile)
    for j0 in j_starts:                             # внешний цикл: KV-тайлы
        j1 = min(j0 + tile, T)
        kj, vj = k[:, j0:j1], v[:, j0:j1]
        dk_j = jnp.zeros((B, j1 - j0, H, dq), jnp.float32)
        dv_j = jnp.zeros((B, j1 - j0, H, dq), jnp.float32)
        for i0 in i_starts:                         # внутренний: Q-тайлы
            i1 = min(i0 + tile, T)
            qi, oi, do_i = q[:, i0:i1], o[:, i0:i1], g[:, i0:i1]
            s = _gemm_einsum("bthd,bshd->bhts", qi, kj) * scale  # (B,H,ti,tj)
            m_ij = mask[..., i0:i1, j0:j1]
            rm_i = rowmax[:, :, i0:i1, :]           # (B,H,ti,1)
            rs_i = rowsum[:, :, i0:i1, :]
            p = jnp.where(m_ij, jnp.exp(s - rm_i), 0.0) / rs_i
            dp = _gemm_einsum("bthd,bshd->bhts", do_i, vj)
            d_i = jnp.transpose(d[:, i0:i1, :], (0, 2, 1))[:, :, :, None]  # (B,H,ti,1)
            ds = p * (dp - d_i)
            grad_s = ds * scale
            dq_acc = dq_acc.at[:, i0:i1, :, :].add(
                _gemm_einsum("bhts,bshd->bthd", grad_s, kj)
            )
            dk_j = dk_j + _gemm_einsum("bhts,bthd->bshd", grad_s, qi)
            dv_j = dv_j + _gemm_einsum("bhts,bthd->bshd", p, do_i)
        dk_acc = dk_acc.at[:, j0:j1, :, :].add(dk_j)
        dv_acc = dv_acc.at[:, j0:j1, :, :].add(dv_j)
    return dq_acc, dk_acc, dv_acc, None


@functools.lru_cache(maxsize=None)
def _flashbwd_op(tile: int):
    @jax.custom_vjp
    def op(q, k, v, mask):
        return _attn_out_flashbwd_fwd(q, k, v, mask, tile)[0]

    op.defvjp(
        lambda q, k, v, mask: _attn_out_flashbwd_fwd(q, k, v, mask, tile),
        lambda res, g: _attn_out_flashbwd_bwd(res, g, tile),
    )
    return op


def dense_ref(q, k, v, mask):
    dq = q.shape[-1]
    scale = 1.0 / jnp.sqrt(dq)
    scores = jnp.einsum("bthd,bshd->bhts", q, k) * scale
    masked = jnp.where(mask, scores, jnp.finfo(jnp.float32).min)
    attn = jax.nn.softmax(masked, axis=-1)
    return jnp.einsum("bhts,bshd->bthd", attn, v)


def main():
    B, T, H, dq = 2, 8, 3, 4
    tile = 2
    key = jr.PRNGKey(0)
    q = jr.normal(key, (B, T, H, dq))
    k = jr.normal(jr.PRNGKey(1), (B, T, H, dq))
    v = jr.normal(jr.PRNGKey(2), (B, T, H, dq))
    mask = jnp.tril(jnp.ones((T, T), bool))

    f_op = _flashbwd_op(tile)
    o_op = f_op(q, k, v, mask)
    o_ref = dense_ref(q, k, v, mask)
    print("fwd bitwise:", bool(jnp.array_equal(o_op, o_ref)))

    def loss_op(q, k, v):
        return jnp.sum(f_op(q, k, v, mask) ** 2)

    def loss_ref(q, k, v):
        return jnp.sum(dense_ref(q, k, v, mask) ** 2)

    g_op = jax.grad(loss_op, argnums=(0, 1, 2))(q, k, v)
    g_ref = jax.grad(loss_ref, argnums=(0, 1, 2))(q, k, v)
    for name, a, b in zip("qkv", g_op, g_ref):
        rel = float(np.max(np.abs(np.asarray(a) - np.asarray(b))) /
                    max(1e-30, float(np.max(np.abs(np.asarray(b))))))
        print(f"grad {name}: max_rel={rel:.3e}")

    # jaxpr pullback: сканируем формы
    def scalar_op(q, k, v):
        return jnp.sum(f_op(q, k, v, mask))

    _, pb = jax.vjp(scalar_op, q, k, v)
    closed = jax.make_jaxpr(pb)(jnp.ones(()))
    shapes = []
    for eqn in closed.jaxpr.eqns:
        for var in eqn.outvars:
            av = getattr(var, "aval", None)
            if av is not None:
                shapes.append((str(eqn.primitive.name), tuple(av.shape), str(av.dtype)))
    full = [s for s in shapes if len(s[1]) == 4 and s[1][2] == T and s[1][3] == T
            and "float" in s[2]]
    tileish = [s for s in shapes if len(s[1]) == 4 and s[1][2] <= tile and s[1][3] <= tile
               and "float" in s[2] and s[1][2] == tile]
    print("bwd: full-score float vars:", full)
    print("bwd: tile-shaped float vars count:", len(tileish), tileish[:2])

    # jit + lower
    jf = jax.jit(loss_op)
    lo = jf.lower(q, k, v)
    hlo = lo.as_text() if hasattr(lo, "as_text") else lo.as_hlo_text()
    for line in hlo.splitlines():
        if f"{T},{T}" in line or "stablehlo" in line[:20]:
            pass
    print("hlo has tile check:", [l.strip()[:110] for l in hlo.splitlines()
                                  if f"[{B},{H},{tile},{tile}]" in l or f"[{B},{T},{H},{dq}]" in l][:3])


if __name__ == "__main__":
    main()
