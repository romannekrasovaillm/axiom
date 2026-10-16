import sys, os
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("net/tests"))
import jax, jax.numpy as jnp
from conftest import small_config
from net.kernels import mla_persist as P
import net.mla as mla_mod

P.kernel = lambda q, k, v, *, bq=P.BQ, bk=P.BK: P.emulate_attention(q, k, v, bq=int(bq), bk=int(bk))
P.kernel_bwd = lambda q, k, v, do, D, m, l, *, bq=P.BQ, bk=P.BK: P.emulate_attention_bwd(q, k, v, do, D, m, l, bq=int(bq), bk=int(bk))
P.device_available = lambda: True

cfg = small_config()
B, T = 1, 256
params = mla_mod.init_mla(jax.random.key(3), cfg)
x = jax.random.normal(jax.random.key(4), (B, T, cfg.hidden)) * 0.3
go = jax.random.normal(jax.random.key(5), (B, T, cfg.hidden))

for mode in ("fp32", "bf16"):
    os.environ["AXIOM_COMPUTE_DTYPE"] = mode
    os.environ["AXIOM_MLA_PERSIST"] = "0"
    o_d = mla_mod.apply(params, cfg, x)
    os.environ["AXIOM_MLA_PERSIST"] = "1"
    o_p = mla_mod.apply(params, cfg, x)
    d = float(jnp.max(jnp.abs(o_p - o_d)))
    print(f"[{mode}] fwd: max_abs={d:.3e} max|o|={float(jnp.max(jnp.abs(o_d))):.3e} rel={d/float(jnp.max(jnp.abs(o_d))):.3e}")
    def loss(xx, flag):
        os.environ["AXIOM_MLA_PERSIST"] = flag
        return jnp.sum(mla_mod.apply(params, cfg, xx) * go)
    g_d = jax.grad(lambda xx: loss(xx, "0"))(x)
    g_p = jax.grad(lambda xx: loss(xx, "1"))(x)
    dg = float(jnp.max(jnp.abs(g_p - g_d)))
    print(f"[{mode}] grad dx: max_abs={dg:.3e} max|g|={float(jnp.max(jnp.abs(g_d))):.3e} rel={dg/float(jnp.max(jnp.abs(g_d))):.3e}")
    def ploss(p, flag):
        os.environ["AXIOM_MLA_PERSIST"] = flag
        return jnp.sum(mla_mod.apply(p, cfg, x) * go)
    gp_d = jax.grad(lambda p_: ploss(p_, "0"))(params)
    gp_p = jax.grad(lambda p_: ploss(p_, "1"))(params)
    worst = 0.0; nm_worst = ""
    for nm, a, b in zip(mla_mod.MLAParams._fields, gp_d, gp_p):
        den = float(jnp.max(jnp.abs(a)))
        if den == 0.0: continue
        r = float(jnp.max(jnp.abs(a - b))) / den
        if r > worst: worst, nm_worst = r, nm
    print(f"[{mode}] params grads: worst rel={worst:.3e} (d{nm_worst})")
