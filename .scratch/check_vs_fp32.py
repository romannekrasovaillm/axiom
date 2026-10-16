import sys, os
sys.path.insert(0, os.path.abspath("."))
import jax, jax.numpy as jnp
from net.kernels import mla_persist as P

B, T, H, dq = 1, 256, 2, 128
for seed in (0, 1, 2):
    qb, kb, vb = P.make_inputs(jax.random.key(seed), B=B, T=T, H=H, dq=dq)
    q32, k32, v32 = (x.astype(jnp.float32) for x in (qb, kb, vb))
    go = jax.random.normal(jax.random.fold_in(jax.random.key(seed), 3), (B, T, H, dq))

    # the fp32 oracle (net/mla.py's dense attention semantics)
    def loss_fp32(q, k, v):
        o, _, _ = P.reference_attention(q, k, v)
        return jnp.sum(o * go)
    dq_r, dk_r, dv_r = jax.grad(loss_fp32, argnums=(0, 1, 2))(q32, k32, v32)
    o_r, m_r, l_r = P.reference_attention(q32, k32, v32)

    # the kernel's declared numerics (hi/lo staging)
    o_e, m_e, l_e = P.emulate_attention(q32, k32, v32, casts=True)
    D = jnp.einsum("bthd,bthd->bht", go, o_e)
    dq_e, dk_e, dv_e = P.emulate_attention_bwd(q32, k32, v32, go, D, m_e, l_e, casts=True)
    fwd = float(jnp.max(jnp.abs(o_e - o_r)) / jnp.max(jnp.abs(o_r)))
    grads = [float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(b))) for a, b in [(dq_e, dq_r), (dk_e, dk_r), (dv_e, dv_r)]]
    print(f"seed{seed}: vs fp32 oracle: fwd rel={fwd:.2e}  grads rel=" + ", ".join(f"{g:.2e}" for g in grads))

    # and the input-cast-only term: fp32 reference of *bf16-cast* inputs (no staging casts)
    def loss_inputcasts(q, k, v):
        o, _, _ = P.reference_attention(q, k, v)
        return jnp.sum(o * go)
    o_i, m_i, l_i = P.reference_attention(qb.astype(jnp.float32), kb.astype(jnp.float32), vb.astype(jnp.float32))
    dqi, dki, dvi = jax.grad(loss_inputcasts, argnums=(0, 1, 2))(qb.astype(jnp.float32), kb.astype(jnp.float32), vb.astype(jnp.float32))
    o_exact = P.reference_attention(q32, k32, v32)[0]
    fwd_in = float(jnp.max(jnp.abs(o_i - o_exact)) / jnp.max(jnp.abs(o_exact)))
    fwd_in_g = float(jnp.max(jnp.abs(qb.astype(jnp.float32)-q32)))  # trivial
    print(f"seed{seed}: input-cast term (fwd): {fwd_in:.2e}  bf16 eps = 2^-9 = {2**-9:.2e}")
