import sys, os
sys.path.insert(0, os.path.abspath("."))
import jax, jax.numpy as jnp
from net.kernels import mla_persist as P

# gradcheck: emulate_attention_bwd must be the exact adjoint of emulate_attention
for seed, (B, T, H, dq) in [(0, (1, 128, 2, 128)), (1, (1, 256, 2, 128)), (2, (2, 128, 2, 64))]:
    q, k, v = P.make_inputs(jax.random.key(seed), B=B, T=T, H=H, dq=dq)
    go = jax.random.normal(jax.random.fold_in(jax.random.key(seed), 3), (B, T, H, dq))
    for casts in (False, True):
        def loss_ref(q, k, v, c=casts):
            o, _, _ = P.emulate_attention(q, k, v, casts=c)
            return jnp.sum(o * go)
        dq_r, dk_r, dv_r = jax.grad(loss_ref, argnums=(0, 1, 2))(q, k, v)
        o_e, m_e, l_e = P.emulate_attention(q, k, v, casts=casts)
        D = jnp.einsum("bthd,bthd->bht", go, o_e)
        dq_e, dk_e, dv_e = P.emulate_attention_bwd(q, k, v, go.astype(q.dtype), D, m_e, l_e, casts=casts)
        rels = []
        for a, b in [(dq_e, dq_r), (dk_e, dk_r), (dv_e, dv_r)]:
            den = float(jnp.max(jnp.abs(b)))
            rels.append(float(jnp.max(jnp.abs(a - b))) / den)
        print(f"seed{seed} casts={casts}: adjoint rel dq/dk/dv = " + ", ".join(f"{r:.3e}" for r in rels))
