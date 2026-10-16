import sys, os
sys.path.insert(0, os.path.abspath("."))
import jax, jax.numpy as jnp
from net.kernels import mla_persist as P

# single tile: T == bq == bk = 64, H=1, dq=16; only the diagonal pair exists
B, T, H, dq = 1, 64, 1, 16
q, k, v = P.make_inputs(jax.random.key(0), B=B, T=T, H=H, dq=dq)
go = jax.random.normal(jax.random.key(9), (B, T, H, dq))

def loss(q, k, v):
    o, m, l = P.emulate_attention(q, k, v, casts=False)
    return jnp.sum(o * go)
dq_r, dk_r, dv_r = jax.grad(loss, argnums=(0, 1, 2))(q, k, v)
o_e, m_e, l_e = P.emulate_attention(q, k, v, casts=False)
print("shapes", o_e.shape)
D = jnp.einsum("bthd,bthd->bht", go, o_e)
dq_e, dk_e, dv_e = P.emulate_attention_bwd(q, k, v, go.astype(q.dtype), D, m_e, l_e, casts=False)
for nm, a, b in [("dq", dq_e, dq_r), ("dk", dk_e, dk_r), ("dv", dv_e, dv_r)]:
    print(f"{nm}: rel={float(jnp.max(jnp.abs(a-b))/jnp.max(jnp.abs(b))):.3e}")

# manual single-tile reference
f32 = jnp.float32
scale = float(P._attn_scale(dq))
qk = q[0, :, 0, :].astype(f32); kk = k[0, :, 0, :].astype(f32); vv = v[0, :, 0, :].astype(f32)
s = (qk @ kk.T) * scale
causal = jnp.arange(T)[:, None] >= jnp.arange(T)[None, :]
s = jnp.where(causal, s, jnp.finfo(f32).min)
m = jnp.max(s, -1); e = jnp.exp(s - m[:, None]); l = jnp.sum(e, -1); attn = e / l[:, None]
o_man = attn @ vv
print("emul vs manual o:", float(jnp.max(jnp.abs(o_e[0,:,0,:] - o_man))), "maxabs o", float(jnp.max(jnp.abs(o_man))))
print("m parity:", float(jnp.max(jnp.abs(m_e[0,0] - m))), " l parity:", float(jnp.max(jnp.abs(l_e[0,0] - l))/jnp.max(l)))
# manual adjoint
do = go[0, :, 0, :]
d = jnp.sum(do * o_man, -1)
dp = do @ vv.T
ds = attn * (dp - d[:, None]) * scale
dq_m = ds @ kk
dk_m = ds.T @ qk
dv_m = attn.T @ do
print("manual adjoint vs emul-bwd:", [float(jnp.max(jnp.abs(a[0,:,0,:]-b))/jnp.max(jnp.abs(b))) for a, b in [(dq_e, dq_m), (dk_e, dk_m), (dv_e, dv_m)]])
print("manual adjoint vs autodiff:", [float(jnp.max(jnp.abs(a-b))/jnp.max(jnp.abs(b))) for a, b in [(dq_m, dq_r[0,:,0,:]), (dk_m, dk_r[0,:,0,:]), (dv_m, dv_r[0,:,0,:])]])
print("emul-bwd vs autodiff:", [float(jnp.max(jnp.abs(a[0,:,0,:]-b))/jnp.max(jnp.abs(b))) for a, b in [(dq_e, dq_r), (dk_e, dk_r), (dv_e, dv_r)]])
