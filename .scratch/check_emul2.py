import sys, os
sys.path.insert(0, os.path.abspath("."))
import jax, jax.numpy as jnp
from net.kernels import mla_persist as P

def dense_bf16_attention(q, k, v):
    """The dense path in AXIOM_COMPUTE_DTYPE=bf16 mode: operands cast bf16,
    fp32 accumulations (cast_in/cast_out of net/compute_dtype.py)."""
    B, T, H, dq = q.shape
    dt = jnp.bfloat16
    scale = P._attn_scale(dq)
    scores = jnp.einsum("bthd,bshd->bhts", q.astype(dt).astype(jnp.float32),
                        k.astype(dt).astype(jnp.float32)) * scale
    causal = jnp.arange(T)[None, :] <= jnp.arange(T)[:, None]
    sm = jnp.where(causal[None, None, :, :], scores, jnp.finfo(jnp.float32).min)
    m = jnp.max(sm, axis=-1)
    attn = jnp.exp(sm - m[..., None])
    l = jnp.sum(attn, axis=-1)
    attn = attn / l[..., None]
    o = jnp.einsum("bhts,bshd->bthd", attn.astype(dt).astype(jnp.float32),
                   v.astype(dt).astype(jnp.float32))
    return o, m, l

for seed, (B, T, H, dq) in [(0, (1, 256, 2, 128)), (1, (2, 256, 2, 128)), (2, (1, 256, 2, 128))]:
    q, k, v = P.make_inputs(jax.random.key(seed), B=B, T=T, H=H, dq=dq)
    o_b, m_b, l_b = dense_bf16_attention(q, k, v)
    o_e, m_e, l_e = P.emulate_attention(q, k, v)
    rel = float(jnp.max(jnp.abs(o_e - o_b)) / jnp.max(jnp.abs(o_b)))
    print(f"seed{seed} fwd emu vs dense-bf16: max_abs={float(jnp.max(jnp.abs(o_e-o_b))):.3e} rel={rel:.3e}")
    # bwd in the matched mode: autodiff of the dense-bf16 attention
    go = jax.random.normal(jax.random.fold_in(jax.random.key(seed), 9), (B, T, H, dq))
    def loss(q, k, v):
        o, _, _ = dense_bf16_attention(q, k, v)
        return jnp.sum(o * go)
    dq_r, dk_r, dv_r = jax.grad(loss, argnums=(0, 1, 2))(q, k, v)
    D = jnp.einsum("bthd,bthd->bht", go, o_e)
    dq_e, dk_e, dv_e = P.emulate_attention_bwd(q, k, v, go.astype(q.dtype), D, m_e, l_e)
    for nm, a, b in [("dq", dq_e, dq_r), ("dk", dk_e, dk_r), ("dv", dv_e, dv_r)]:
        rel = float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(b)))
        print(f"seed{seed} bwd emu vs grad(dense-bf16) {nm}: rel={rel:.3e}")
