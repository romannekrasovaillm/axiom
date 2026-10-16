import sys, os
sys.path.insert(0, os.path.abspath("."))
import jax, jax.numpy as jnp
from net.kernels import mla_persist as P

def dense_bf16_attn(q, k, v):
    dt = jnp.bfloat16
    B, T, H, dq = q.shape
    scale = P._attn_scale(dq)
    scores = jnp.einsum("bthd,bshd->bhts", q.astype(dt).astype(jnp.float32),
                        k.astype(dt).astype(jnp.float32)) * scale
    causal = jnp.arange(T)[None, :] <= jnp.arange(T)[:, None]
    sm = jnp.where(causal[None, None], scores, jnp.finfo(jnp.float32).min)
    m = jnp.max(sm, axis=-1)
    attn = jnp.exp(sm - m[..., None]); l = jnp.sum(attn, axis=-1); attn = attn / l[..., None]
    o = jnp.einsum("bhts,bshd->bthd", attn.astype(dt).astype(jnp.float32),
                   v.astype(dt).astype(jnp.float32))
    return o, m, l

B, T, H, dq = 1, 256, 2, 128
q, k, v = P.make_inputs(jax.random.key(0), B=B, T=T, H=H, dq=dq)
go = jax.random.normal(jax.random.key(7), (B, T, H, dq))

def loss(q, k, v):
    o, _, _ = dense_bf16_attn(q, k, v)
    return jnp.sum(o * go)
dq_r, dk_r, dv_r = jax.grad(loss, argnums=(0, 1, 2))(q, k, v)

o_e, m_e, l_e = P.emulate_attention(q, k, v)
D = jnp.einsum("bthd,bthd->bht", go, o_e)

# emulation with selective casts disabled (diagnostic only)
def emulate_bwd(cast_p=True, cast_ds=True, cast_do=True):
    dt = jnp.bfloat16; f32 = jnp.float32
    scale = float(P._attn_scale(dq)); bq = bk = 64
    nq, nk = T // bq, T // bk
    dq_o = jnp.zeros_like(o_e); dk_o = jnp.zeros_like(o_e); dv_o = jnp.zeros_like(o_e)
    ridx = jnp.arange(bq)[:, None]; cidx = jnp.arange(bk)[None, :]
    diag = ridx >= cidx
    do32 = go.astype(f32) if not cast_do else go.astype(dt).astype(f32)
    for b in range(B):
        for h in range(H):
            for j in range(nk):
                kj = k[b, j*bk:(j+1)*bk, h].astype(f32); vj = v[b, j*bk:(j+1)*bk, h].astype(f32)
                dk_a = jnp.zeros((bk, dq), f32); dv_a = jnp.zeros((bk, dq), f32)
                for i in range(j, nq):
                    qi = q[b, i*bq:(i+1)*bq, h].astype(f32)
                    doi = do32[b, i*bq:(i+1)*bq, h]
                    s = (qi @ kj.T) * scale
                    mask = diag if i == j else None
                    sm = jnp.where(mask, s, jnp.finfo(f32).min) if mask is not None else s
                    mi = m_e[b, h, i*bq:(i+1)*bq]; li = l_e[b, h, i*bq:(i+1)*bq]; Di = D[b, h, i*bq:(i+1)*bq]
                    p = jnp.exp(sm - mi[:, None]) / li[:, None]
                    dp = doi @ vj.T
                    ds = (p * (dp - Di[:, None])) * scale
                    p16 = p.astype(dt).astype(f32) if cast_p else p
                    ds16 = ds.astype(dt).astype(f32) if cast_ds else ds
                    dk_a = dk_a + ds16.T @ qi
                    dv_a = dv_a + p16.T @ doi
                    dq_o = dq_o.at[b, i*bq:(i+1)*bq, h].add(ds16 @ kj)
                dk_o = dk_o.at[b, j*bk:(j+1)*bk, h].set(dk_a)
                dv_o = dv_o.at[b, j*bk:(j+1)*bk, h].set(dv_a)
    return dq_o, dk_o, dv_o

for name, kw in [("full", dict()), ("no_p_cast", dict(cast_p=False)), ("no_ds_cast", dict(cast_ds=False)),
                 ("no_do_cast", dict(cast_do=False)), ("none", dict(cast_p=False, cast_ds=False, cast_do=False))]:
    dq_e, dk_e, dv_e = emulate_bwd(**kw)
    rels = []
    for a, b in [(dq_e, dq_r), (dk_e, dk_r), (dv_e, dv_r)]:
        rels.append(float(jnp.max(jnp.abs(a-b))/jnp.max(jnp.abs(b))))
    print(f"{name:12s} " + " ".join(f"{r:.2e}" for r in rels))
print("cast-relative scale check: p rel err, ds rel err ~ 2^-9 =", 2**-9)
