import sys, os
sys.path.insert(0, os.path.abspath("."))
import jax, jax.numpy as jnp
from net.kernels import mla_persist as P

dt = jnp.bfloat16
def dense(q32, k32, v32, cast):
    """dense attention with opt. internal bf16 operand casts (bf16 compute mode)."""
    B, T, H, dq = q32.shape
    scale = P._attn_scale(dq)
    if cast:
        f = lambda x: x.astype(dt).astype(jnp.float32)
    else:
        f = lambda x: x
    s = jnp.einsum("bthd,bshd->bhts", f(q32), f(k32)) * scale
    causal = jnp.arange(T)[None, :] <= jnp.arange(T)[:, None]
    sm = jnp.where(causal[None, None], s, jnp.finfo(jnp.float32).min)
    m = jnp.max(sm, -1); attn = jnp.exp(sm - m[..., None]); l = jnp.sum(attn, -1); attn = attn / l[..., None]
    o = jnp.einsum("bhts,bshd->bthd", f(attn), f(v32))
    return o, m, l

for seed, (B, T, H, dq) in [(0, (1, 64, 1, 16)), (1, (1, 128, 2, 64)), (2, (1, 256, 2, 128)), (3, (2, 256, 2, 128))]:
    qb, kb, vb = P.make_inputs(jax.random.key(seed), B=B, T=T, H=H, dq=dq)
    q32, k32, v32 = (x.astype(jnp.float32) for x in (qb, kb, vb))   # bf16-valued fp32 (the projection outputs)
    go = jax.random.normal(jax.random.fold_in(jax.random.key(seed), 3), (B, T, H, dq))

    # (1) ALGEBRA AUDIT: fp32 tiled vs fp32 autodiff (bf16 values in, no casts)
    def loss_alg(q, k, v):
        o, _, _ = P.emulate_attention(q, k, v, casts=False)
        return jnp.sum(o * go)
    dq_r, dk_r, dv_r = jax.grad(loss_alg, argnums=(0, 1, 2))(q32, k32, v32)
    o_e, m_e, l_e = P.emulate_attention(q32, k32, v32, casts=False)
    D = jnp.einsum("bthd,bthd->bht", go, o_e)
    dq_e, dk_e, dv_e = P.emulate_attention_bwd(q32, k32, v32, go, D, m_e, l_e, casts=False)
    rels_alg = [float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(b))) for a, b in [(dq_e, dq_r), (dk_e, dk_r), (dv_e, dv_r)]]
    # fp32 tiled vs fp32 dense fwd
    o_d, m_d, l_d = dense(q32, k32, v32, cast=False)
    fwd_alg = float(jnp.max(jnp.abs(o_e - o_d)) / jnp.max(jnp.abs(o_d)))

    # (2) OPERAND DRIFT: declared numerics (casts on) vs fp32 autodiff of the dense
    def loss_bf16(q, k, v):
        o, _, _ = dense(q, k, v, cast=True)
        return jnp.sum(o * go)
    dq_b, dk_b, dv_b = jax.grad(loss_bf16, argnums=(0, 1, 2))(q32, k32, v32)
    o_e2, m_e2, l_e2 = P.emulate_attention(q32, k32, v32, casts=True)
    D2 = jnp.einsum("bthd,bthd->bht", go, o_e2)
    dq_e2, dk_e2, dv_e2 = P.emulate_attention_bwd(q32, k32, v32, go, D2, m_e2, l_e2, casts=True)
    rels_bf = [float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(b))) for a, b in [(dq_e2, dq_b), (dk_e2, dk_b), (dv_e2, dv_b)]]
    fwd_bf = float(jnp.max(jnp.abs(o_e2 - dense(q32, k32, v32, cast=True)[0])) / jnp.max(jnp.abs(o_d)))
    print(f"seed{seed} T{T} H{H} dq{dq}:")
    print(f"  algebra: fwd rel={fwd_alg:.2e}  adjoint rel=" + ", ".join(f"{r:.2e}" for r in rels_alg))
    print(f"  operand drift vs dense-bf16: fwd rel={fwd_bf:.2e}  grads rel=" + ", ".join(f"{r:.2e}" for r in rels_bf))
