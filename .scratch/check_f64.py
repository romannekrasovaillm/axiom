import sys, os
sys.path.insert(0, os.path.abspath("."))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from net.kernels import mla_persist as P

def fwd_f64(q, k, v):
    B, T, H, dq = q.shape
    scale = float(P._attn_scale(dq))
    o = jnp.zeros((B, T, H, dq), jnp.float64)
    m_o = jnp.zeros((B, H, T), jnp.float64); l_o = jnp.zeros((B, H, T), jnp.float64)
    causal = jnp.arange(T)[None, :] <= jnp.arange(T)[:, None]
    for b in range(B):
        for h in range(H):
            qi = q[b, :, h, :].astype(jnp.float64); kj = k[b, :, h, :].astype(jnp.float64); vj = v[b, :, h, :].astype(jnp.float64)
            s = (qi @ kj.T) * scale
            sm = jnp.where(causal, s, jnp.finfo(jnp.float64).min)
            m = jnp.max(sm, -1); p = jnp.exp(sm - m[:, None]); l = jnp.sum(p, -1)
            o = o.at[b, :, h, :].set((p / l[:, None]) @ vj)
            m_o = m_o.at[b, h].set(m); l_o = l_o.at[b, h].set(l)
    return o, m_o, l_o

def manual_adjoint_f64(q, k, v, go):
    B, T, H, dq = q.shape
    scale = float(P._attn_scale(dq))
    causal = jnp.arange(T)[None, :] <= jnp.arange(T)[:, None]
    dq_o = jnp.zeros((B, T, H, dq), jnp.float64); dk_o = jnp.zeros_like(dq_o); dv_o = jnp.zeros_like(dq_o)
    for b in range(B):
        for h in range(H):
            qi = q[b, :, h, :].astype(jnp.float64); kj = k[b, :, h, :].astype(jnp.float64); vj = v[b, :, h, :].astype(jnp.float64)
            doi = go[b, :, h, :].astype(jnp.float64)
            s = (qi @ kj.T) * scale
            sm = jnp.where(causal, s, jnp.finfo(jnp.float64).min)
            m = jnp.max(sm, -1); p = jnp.exp(sm - m[:, None]); l = jnp.sum(p, -1); attn = p / l[:, None]
            o = attn @ vj
            d = jnp.sum(doi * o, -1)
            dp = doi @ vj.T
            ds = attn * (dp - d[:, None]) * scale
            dq_o = dq_o.at[b, :, h, :].set(ds @ kj)
            dk_o = dk_o.at[b, :, h, :].set(ds.T @ qi)
            dv_o = dv_o.at[b, :, h, :].set(attn.T @ doi)
    return dq_o, dk_o, dv_o

B, T, H, dq = 1, 64, 1, 16
q, k, v = P.make_inputs(jax.random.key(0), B=B, T=T, H=H, dq=dq)
go = jax.random.normal(jax.random.key(9), (B, T, H, dq))
def loss(q, k, v):
    o, _, _ = fwd_f64(q, k, v)
    return jnp.sum(o * go.astype(jnp.float64))
dq_r, dk_r, dv_r = jax.grad(loss, argnums=(0, 1, 2))(q, k, v)
dq_m, dk_m, dv_m = manual_adjoint_f64(q, k, v, go)
print("f64 manual vs autodiff:", [f"{float(jnp.max(jnp.abs(a-b))/jnp.max(jnp.abs(b))):.2e}" for a, b in [(dq_m, dq_r), (dk_m, dk_r), (dv_m, dv_r)]])

# and the fp32 emulation vs the f64 adjoint (the true conditioning-limited error)
o_e, m_e, l_e = P.emulate_attention(q, k, v, casts=False)
D = jnp.einsum("bthd,bthd->bht", go, o_e)
dq_e, dk_e, dv_e = P.emulate_attention_bwd(q, k, v, go.astype(q.dtype), D, m_e, l_e, casts=False)
print("fp32 emul-bwd vs f64 adjoint:", [f"{float(jnp.max(jnp.abs(a.astype(jnp.float64)-b))/jnp.max(jnp.abs(b))):.2e}" for a, b in [(dq_e, dq_r), (dk_e, dk_r), (dv_e, dv_r)]])
