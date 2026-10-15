"""Отладка расхождения состояния (3 шага): какой лист/компонент."""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
ROOT = "/home/roman/.arch-ml/worktrees/axiom-43e236d789188868/theseus-20261015183027"
sys.path.insert(0, ROOT + "/net/tests"); sys.path.insert(0, ROOT)
import numpy as np
import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import jax.random as jr
from conftest import tiny_config
from net import model, optimizer

cfg = tiny_config()
base = model.init_params(jr.PRNGKey(0), cfg)
params = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), base)
grads = jax.tree_util.tree_map(
    lambda leaf: jr.normal(jr.PRNGKey(int(leaf.size) + 1), leaf.shape), params)
grads = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), grads)
ids = jr.randint(jr.PRNGKey(2), (2, 16), 0, cfg.vocab_size)
loss_fn = lambda p: model.compute_loss(p, cfg, ids, chunk_size=8)

comp = "kahan"
st_l = optimizer.init_state(params, master_dtype="bf16", master_comp=comp)
st_f = optimizer.init_state(params, master_dtype="bf16", master_comp=comp)
step_l = optimizer.make_step(cfg, master_dtype="bf16", master_comp=comp)
step_f = optimizer.make_step(cfg, master_dtype="bf16", master_comp=comp, fuse=True)
pl, pf = params, params
for i in range(3):
    ll, gl = jax.value_and_grad(loss_fn)(pl)
    lf, gf = jax.value_and_grad(loss_fn)(pf)
    print(f"step {i}: loss {float(ll)!r} vs {float(lf)!r}")
    pl, st_l = step_l(pl, gl, st_l, 1e-3)
    pf, st_f = step_f(pf, gf, st_f, 1e-3)
    dl = max((float(np.max(np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32))))
              for a, b in zip(jax.tree_util.tree_leaves(pl), jax.tree_util.tree_leaves(pf))
              if a is not None and b is not None and np.asarray(a).size), default=0.0)
    print(f"  max|dp|={dl:.3e}")

# найти листья состояния, где есть расхождение: сравнить по структуре путей
walk_l = jax.tree_util.tree_flatten_with_path(st_l)[0]
walk_f = jax.tree_util.tree_flatten_with_path(st_f)[0]
print("n state leaves:", len(walk_l), len(walk_f))
for (pl_, x), (pf_, y) in zip(walk_l, walk_f):
    if x is None or y is None:
        continue
    xa, ya = np.asarray(x, np.float32), np.asarray(y, np.float32)
    if not np.array_equal(xa, ya):
        d = float(np.max(np.abs(xa - ya))) if xa.size else 0.0
        rel = d / (float(np.max(np.abs(ya)) + 1e-30))
        print("DIFF", jax.tree_util.keystr(pf_), x.dtype, x.shape, f"max|d|={d:.3e} rel={rel:.3e}")
