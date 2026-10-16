import sys, os
sys.path.insert(0, "net/tests"); sys.path.insert(0, ".")
import conftest  # noqa
import functools
import jax, jax.numpy as jnp, jax.random as jr
import numpy as np
from net import compute_dtype

os.environ["AXIOM_COMPUTE_DTYPE"] = "bf16"

B, T, H, dq, tile = 2, 16, 4, 16, 4
q, k, v = (jr.normal(jr.PRNGKey(i), (B, T, H, dq)) for i in range(3))
mask = jnp.tril(jnp.ones((T, T), bool))


def dense_ref(q, k, v):
    scale = 1.0 / jnp.sqrt(dq)
    s = compute_dtype.gemm_einsum("bthd,bshd->bhts", q, k) * scale
    a = jax.nn.softmax(jnp.where(mask, s, jnp.finfo(jnp.float32).min), axis=-1)
    return compute_dtype.gemm_einsum("bhts,bshd->bthd", a, v)


def make_bwd(kind):
    def bwd(res, g, tile):
        q, k, v, mask_, o, rmax_, rsum_ = res
        scale = 1.0 / jnp.sqrt(dq)
        d = jnp.transpose(jnp.sum(g * o, axis=-1), (0, 2, 1))
        dq_acc = jnp.zeros_like(q); dk_acc = jnp.zeros_like(k); dv_acc = jnp.zeros_like(v)
        for j0 in range(0, T, tile):
            j1 = min(j0 + tile, T); kj, vj = k[:, j0:j1], v[:, j0:j1]
            dk_j = jnp.zeros((B, j1 - j0, H, dq)); dv_j = jnp.zeros((B, j1 - j0, H, dq))
            for i0 in range(0, T, tile):
                i1 = min(i0 + tile, T)
                qi, do_i = q[:, i0:i1], g[:, i0:i1]
                if kind == "cast":
                    s = compute_dtype.gemm_einsum("bthd,bshd->bhts", qi, kj) * scale
                    dp = compute_dtype.gemm_einsum("bthd,bshd->bhts", do_i, vj)
                else:
                    s = jnp.einsum("bthd,bshd->bhts", qi, kj) * scale
                    dp = jnp.einsum("bthd,bshd->bhts", do_i, vj)
                p = jnp.where(mask_[..., i0:i1, j0:j1],
                              jnp.exp(s - rmax_[:, :, i0:i1, :]), 0.0) / rsum_[:, :, i0:i1, :]
                grad_s = p * (dp - d[:, :, i0:i1, None]) * scale
                if kind == "cast":
                    dq_acc = dq_acc.at[:, i0:i1].add(
                        compute_dtype.gemm_einsum("bhts,bshd->bthd", grad_s, kj))
                    dk_j = dk_j + compute_dtype.gemm_einsum("bhts,bthd->bshd", grad_s, qi)
                    dv_j = dv_j + compute_dtype.gemm_einsum("bhts,bthd->bshd", p, do_i)
                else:
                    dq_acc = dq_acc.at[:, i0:i1].add(jnp.einsum("bhts,bshd->bthd", grad_s, kj))
                    dk_j = dk_j + jnp.einsum("bhts,bthd->bshd", grad_s, qi)
                    dv_j = dv_j + jnp.einsum("bhts,bthd->bshd", p, do_i)
            dk_acc = dk_acc.at[:, j0:j1].add(dk_j); dv_acc = dv_acc.at[:, j0:j1].add(dv_j)
        return dq_acc, dk_acc, dv_acc
    return bwd


def fwd(q, k, v, mask_, tile):
    scale = 1.0 / jnp.sqrt(dq)
    s = compute_dtype.gemm_einsum("bthd,bshd->bhts", q, k) * scale
    masked = jnp.where(mask_, s, jnp.finfo(jnp.float32).min)
    a = jax.nn.softmax(masked, axis=-1)
    o = compute_dtype.gemm_einsum("bhts,bshd->bthd", a, v)
    rmax = jnp.max(masked, axis=-1, keepdims=True)
    rsum = jnp.sum(jnp.exp(masked - rmax), axis=-1, keepdims=True)
    return o, (q, k, v, mask_, o, rmax, rsum)


def op(kind):
    @jax.custom_vjp
    def f(q, k, v):
        return fwd(q, k, v, mask, tile)[0]
    f.defvjp(lambda q, k, v: fwd(q, k, v, mask, tile),
             lambda res, g: make_bwd(kind)(res, g, tile))
    return f


ref_g = jax.grad(lambda q, k, v: jnp.sum(dense_ref(q, k, v) ** 2), argnums=(0, 1, 2))(q, k, v)
for kind in ("cast", "fp32bwd"):
    g = jax.grad(lambda q_, k_, v_: jnp.sum(op(kind)(q_, k_, v_) ** 2), argnums=(0, 1, 2))(q, k, v)
    rel = max(float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) for a, b in zip(g, ref_g)) / max(
        1e-30, max(float(np.max(np.abs(np.asarray(b)))) for b in ref_g))
    print(f"kind={kind}: grad max_rel vs bf16-autodiff ref = {rel:.3e}")
os.environ.pop("AXIOM_COMPUTE_DTYPE", None)
