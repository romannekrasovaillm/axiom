import sys, os
sys.path.insert(0, os.path.abspath("."))
import jax, jax.numpy as jnp
from net.kernels import mla_persist as P

B, T, H, dq = 1, 256, 2, 128
q, k, v = P.make_inputs(jax.random.key(0), B=B, T=T, H=H, dq=dq)

# 1) forward: emulation vs fp32 reference
o_ref, m_ref, l_ref = P.reference_attention(q, k, v)
o_emu, m_emu, l_emu = P.emulate_attention(q, k, v)
rel_o = float(jnp.max(jnp.abs(o_emu - o_ref)) / jnp.max(jnp.abs(o_ref)))
print(f"fwd emu vs ref: max_abs={float(jnp.max(jnp.abs(o_emu-o_ref))):.3e} rel={rel_o:.3e}")
print(f"stats: m {float(jnp.max(jnp.abs(m_emu-m_ref))):.3e} l {float(jnp.max(jnp.abs(l_emu-l_ref))/jnp.max(l_ref)):.3e}")

# 2) backward: emulation vs autodiff of the *fp32 reference* (algebra check)
def loss_ref(q, k, v, go):
    o, _, _ = P.reference_attention(q, k, v)
    return jnp.sum(o * go)
go = jax.random.normal(jax.random.key(7), (B, T, H, dq))
dq_r, dk_r, dv_r = jax.grad(loss_ref, argnums=(0, 1, 2))(q, k, v, go)

o_emu, m_emu, l_emu = P.emulate_attention(q, k, v)
D = jnp.einsum("bthd,bthd->bht", go, o_emu)
dq_e, dk_e, dv_e = P.emulate_attention_bwd(q, k, v, go.astype(q.dtype), D, m_emu, l_emu)
for nm, a, b in [("dq", dq_e, dq_r), ("dk", dk_e, dk_r), ("dv", dv_e, dv_r)]:
    denom = jnp.max(jnp.abs(b))
    print(f"bwd emu vs autodiff {nm}: max_abs={float(jnp.max(jnp.abs(a-b))):.3e} rel={float(jnp.max(jnp.abs(a-b))/denom):.3e}  (|{nm}|max={float(denom):.3e})")
