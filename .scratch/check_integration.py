import sys, os
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("net/tests"))
import jax, jax.numpy as jnp
from conftest import small_config
from net.kernels import mla_persist as P
import net.mla as mla_mod

cfg = small_config()
B, T = 1, 256
key = jax.random.key(3)
k1, k2 = jax.random.split(key)
params = mla_mod.init_mla(k1, cfg)
x = jax.random.normal(k2, (B, T, cfg.hidden)) * 0.3

# subst seams: kernel -> declared numerics
def emu_kernel(q, k, v, *, bq=P.BQ, bk=P.BK):
    return P.emulate_attention(q, k, v, bq=int(bq), bk=int(bk))
def emu_kernel_bwd(q, k, v, do, D, m, l, *, bq=P.BQ, bk=P.BK):
    return P.emulate_attention_bwd(q, k, v, do, D, m, l, bq=int(bq), bk=int(bk))
P.kernel = emu_kernel
P.kernel_bwd = emu_kernel_bwd
P.device_available = lambda: True

os.environ["AXIOM_MLA_PERSIST"] = "0"
o_dense = mla_mod.apply(params, cfg, x)
os.environ["AXIOM_MLA_PERSIST"] = "1"
o_persist = mla_mod.apply(params, cfg, x)
rel = float(jnp.max(jnp.abs(o_persist - o_dense)) / jnp.max(jnp.abs(o_dense)))
print(f"fwd layer parity persist-vs-dense (fp32 mode): max_abs={float(jnp.max(jnp.abs(o_persist-o_dense))):.3e} rel={rel:.3e}")

go = jax.random.normal(jax.random.key(5), o_dense.shape)
def loss(cfg_x, flag):
    os.environ["AXIOM_MLA_PERSIST"] = flag
    return jnp.sum(mla_mod.apply(params, cfg, cfg_x) * go)
g_dense = jax.grad(lambda x_: loss(x_, "0"))(x)
g_persist = jax.grad(lambda x_: loss(x_, "1"))(x)
relg = float(jnp.max(jnp.abs(g_persist - g_dense)) / jnp.max(jnp.abs(g_dense)))
print(f"grad parity (dx) persist-vs-dense: max_abs={float(jnp.max(jnp.abs(g_persist-g_dense))):.3e} rel={relg:.3e}")

# params grads
def ploss(p, flag):
    os.environ["AXIOM_MLA_PERSIST"] = flag
    return jnp.sum(mla_mod.apply(p, cfg, x) * go)
gp_d = jax.grad(lambda p_: ploss(p_, "0"))(params)
gp_p = jax.grad(lambda p_: ploss(p_, "1"))(params)
for nm, a, b in zip(mla_mod.MLAParams._fields, gp_d, gp_p):
    denom = jnp.max(jnp.abs(a))
    print(f"  d{nm}: max_abs={float(jnp.max(jnp.abs(a-b))):.3e} rel={float(jnp.max(jnp.abs(a-b))/denom):.3e}")

# semantic parity: fp32 tiled reference vs fp32 dense attention interval
x32 = x.astype(jnp.float32)
c = x32 @ params.W_c.astype(jnp.float32)
H, dq = cfg.num_heads, cfg.mla_head_dim
scale = 1.0 / jnp.sqrt(dq)
scores = jnp.einsum("bthd,bshd->bhts", q32 := (x32 @ params.W_q.astype(jnp.float32)).reshape(B, T, H, dq),
                    (c @ params.W_k_up.astype(jnp.float32)).reshape(B, T, H, dq)) * scale
mask = jnp.tril(jnp.ones((T, T), jnp.bool_))
sm = jnp.where(mask[None, None], scores, jnp.finfo(jnp.float32).min)
mm = jnp.max(sm, axis=-1); ll = jnp.sum(jnp.exp(sm - mm[..., None]), axis=-1)
attn = jnp.exp(sm - mm[..., None]) / ll[..., None]
v32 = (c @ params.W_v_up.astype(jnp.float32)).reshape(B, T, H, dq)
o_dense_attn = jnp.einsum("bhts,bshd->bthd", attn, v32)
qb, kb, vb = q32.astype(jnp.bfloat16), (c @ params.W_k_up.astype(jnp.float32)).reshape(B, T, H, dq).astype(jnp.bfloat16), v32.astype(jnp.bfloat16)
o_tiled, m_t, l_t = P.emulate_attention(qb, kb, vb)
rel_sem = float(jnp.max(jnp.abs(o_tiled - o_dense_attn)) / jnp.max(jnp.abs(o_dense_attn)))
print(f"operand-drift (bf16 tiled vs fp32 dense): rel={rel_sem:.3e}")
o_ref32, _, _ = P.reference_attention(q32, (c @ params.W_k_up.astype(jnp.float32)).reshape(B, T, H, dq), v32)
rel_sem2 = float(jnp.max(jnp.abs(o_ref32 - o_dense_attn)) / jnp.max(jnp.abs(o_dense_attn)))
print(f"semantic parity (fp32 tiled reference vs fp32 dense attention): rel={rel_sem2:.3e}")
