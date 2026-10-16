import sys, os
sys.path.insert(0, os.path.abspath("."))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
f64 = jnp.float64
T, D = 4, 3
key = jax.random.key(0)
k1, k2, k3, k4 = jax.random.split(key, 4)
q = jax.random.normal(k1, (T, D), f64)
k = jax.random.normal(k2, (T, D), f64)
v = jax.random.normal(k3, (T, D), f64)
go = jax.random.normal(k4, (T, D), f64)
scale = 1.0 / jnp.sqrt(D)
causal = jnp.arange(T)[None, :] <= jnp.arange(T)[:, None]

def fwd(q, k, v):
    s = (q @ k.T) * scale
    sm = jnp.where(causal, s, jnp.finfo(f64).min)
    m = jnp.max(sm, -1); p = jnp.exp(sm - m[:, None]); l = jnp.sum(p, -1)
    attn = p / l[:, None]
    return attn @ v
o = fwd(q, k, v)
dq_r, dk_r, dv_r = jax.grad(lambda q, k, v: jnp.sum(fwd(q, k, v) * go), argnums=(0, 1, 2))(q, k, v)

# manual
s = (q @ k.T) * scale
sm = jnp.where(causal, s, jnp.finfo(f64).min)
m = jnp.max(sm, -1); p = jnp.exp(sm - m[:, None]); l = jnp.sum(p, -1); attn = p / l[:, None]
d = jnp.sum(go * o, -1)
dp = go @ v.T
ds = attn * (dp - d[:, None]) * scale
dq_m = ds @ k; dk_m = ds.T @ q; dv_m = attn.T @ go
print("d    :", d)
print("dp   :\n", dp)
print("attn :\n", attn)
print("ds_manual   :\n", ds)
# autodiff of the scores: get ds via grad of sum(attn * dp)... direct check
def attn_only(sm_):
    mm = jnp.max(sm_, -1); pp = jnp.exp(sm_ - mm[:, None]); return pp / jnp.sum(pp, -1)[:, None]
ds_ad = jax.grad(lambda sm_: jnp.sum(attn_only(sm_) * dp))(sm)
print("ds_autodiff :\n", ds_ad)
print("ds diff     :\n", ds - ds_ad)
print("dq_m vs dq_r:", float(jnp.max(jnp.abs(dq_m - dq_r))), "dk:", float(jnp.max(jnp.abs(dk_m - dk_r))), "dv:", float(jnp.max(jnp.abs(dv_m - dv_r))))
print("rowsum(dp*attn):", jnp.sum(dp * attn, -1), " d:", d)
