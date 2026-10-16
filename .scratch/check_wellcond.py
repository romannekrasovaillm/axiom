import sys, os
sys.path.insert(0, os.path.abspath("."))
import jax, jax.numpy as jnp
from net.kernels import mla_persist as P

SHARP = 8.0  # scale q,k -> peaked softmax -> well-conditioned adjoint
for seed, (B, T, H, dq) in [(0, (1, 64, 1, 16)), (1, (1, 128, 2, 64)), (2, (1, 256, 2, 128)), (3, (2, 128, 2, 64))]:
    q, k, v = P.make_inputs(jax.random.key(seed), B=B, T=T, H=H, dq=dq)
    q = (q.astype(jnp.float32) * SHARP).astype(q.dtype)
    go = jax.random.normal(jax.random.fold_in(jax.random.key(seed), 3), (B, T, H, dq))
    for casts in (False, True):
        def loss(q, k, v, c=casts):
            o, _, _ = P.emulate_attention(q, k, v, casts=c)
            return jnp.sum(o * go)
        dq_r, dk_r, dv_r = jax.grad(loss, argnums=(0, 1, 2))(q, k, v)
        o_e, m_e, l_e = P.emulate_attention(q, k, v, casts=casts)
        D = jnp.einsum("bthd,bthd->bht", go, o_e)
        dq_e, dk_e, dv_e = P.emulate_attention_bwd(q, k, v, go.astype(q.dtype), D, m_e, l_e, casts=casts)
        rels = [float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(b))) for a, b in [(dq_e, dq_r), (dk_e, dk_r), (dv_e, dv_r)]]
        print(f"seed{seed} T{T} dq{dq} casts={casts}: adjoint rel = " + ", ".join(f"{r:.2e}" for r in rels))
