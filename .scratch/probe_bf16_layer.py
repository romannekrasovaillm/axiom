import sys, os
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("net/tests"))
import jax, jax.numpy as jnp, jax.random as jr
from conftest import small_config
from net import mla
from net.kernels import mla_persist as P

P.kernel = lambda q, k, v, **kw: P.emulate_attention(q, k, v, bq=int(kw.get("bq", 64)), bk=int(kw.get("bk", 64)))
P.kernel_bwd = lambda q, k, v, do, D, m, l, **kw: P.emulate_attention_bwd(q, k, v, do, D, m, l, bq=int(kw.get("bq", 64)), bk=int(kw.get("bk", 64)))
P.device_available = lambda: True

cfg = small_config(mla_head_dim=16)
params = mla.init_mla(jr.PRNGKey(3), cfg)
x = jr.normal(jr.PRNGKey(4), (1, 64, cfg.hidden)) * 0.3
os.environ["AXIOM_COMPUTE_DTYPE"] = "bf16"
os.environ["AXIOM_MLA_PERSIST"] = "0"
o = mla.apply(params, cfg, x)
print("dense bf16 layer OK", float(jnp.max(jnp.abs(o))))
os.environ["AXIOM_MLA_PERSIST"] = "1"
o2 = mla.apply(params, cfg, x)
print("persist bf16 layer OK", float(jnp.max(jnp.abs(o2))))
g = jax.grad(lambda xx: jnp.sum(mla.apply(params, cfg, xx) ** 2))(x)
print("grad OK", float(jnp.max(jnp.abs(g))))
