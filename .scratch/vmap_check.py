"""Проверка vmap(kahan_apply) vs per-leaf на идентичных входах."""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, "/home/roman/.arch-ml/worktrees/axiom-43e236d789188868/theseus-20261015183027")
import numpy as np
import jax
import jax.numpy as jnp
from net import master_dtype as md

n = 4
shape = (3, 5)
key = jax.random.PRNGKey(0)
masters = jax.random.normal(key, (n,) + shape).astype(jnp.bfloat16)
ups = jax.random.normal(jax.random.PRNGKey(1), (n,) + shape) * 1e-6
comps = (jax.random.normal(jax.random.PRNGKey(2), (n,) + shape) * 1e-3).astype(jnp.bfloat16)

stacked_m, stacked_c = jax.vmap(md.kahan_apply)(masters, ups, comps)
for i in range(n):
    m_i, c_i = md.kahan_apply(masters[i], ups[i], comps[i])
    if not np.array_equal(np.asarray(m_i), np.asarray(stacked_m[i])):
        print(i, "master DIFF", float(jnp.max(jnp.abs(m_i.astype(jnp.float32) - stacked_m[i].astype(jnp.float32)))))
    if not np.array_equal(np.asarray(c_i), np.asarray(stacked_c[i])):
        print(i, "comp DIFF", float(jnp.max(jnp.abs(c_i.astype(jnp.float32) - stacked_c[i].astype(jnp.float32)))))
print("vmap vs leaf kahan done")
# и «ручная» стаковая запись против vmap
m32 = masters.astype(jnp.float32)
y = ups.astype(jnp.float32) - comps.astype(jnp.float32)
nm = (m32 + y).astype(jnp.bfloat16)
nc = ((nm.astype(jnp.float32) - m32) - y).astype(jnp.bfloat16)
print("manual stacked master equal:", bool(jnp.all(nm == stacked_m)), "comp equal:", bool(jnp.all(nc == stacked_c)))
# jit-варианты (фьюжн в XLA может менять форму графа)
f = jax.jit(lambda a, b, c: jax.vmap(md.kahan_apply)(a, b, c))
mj, cj = f(masters, ups, comps)
print("jit vmap master equal:", bool(jnp.all(mj == stacked_m)), "comp equal:", bool(jnp.all(cj == stacked_c)))
# вместе со сложением цепочек (момент + применение) — как в фьюзере
def leg_like(m, g, c):
    m32i = m.astype(jnp.float32)
    mnew = 0.95 * m32i + 0.05 * g.astype(jnp.float32)
    updi = mnew * 1e-3
    return md.kahan_apply(m, updi, c)
leg = jax.jit(lambda a, b, c: jax.vmap(leg_like)(a, b, c))
mjl, cjl = leg(masters, ups, comps)
for i in range(n):
    mi, ci = leg_like(masters[i], ups[i], comps[i])
    if not (np.array_equal(np.asarray(mi), np.asarray(mjl[i])) and np.array_equal(np.asarray(ci), np.asarray(cjl[i]))):
        print(i, "leg DIFF", float(jnp.max(jnp.abs(ci.astype(jnp.float32) - cjl[i].astype(jnp.float32)))))
print("leg-like done")
