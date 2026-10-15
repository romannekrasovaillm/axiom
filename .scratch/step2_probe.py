"""Шаг-2 с ОДНИМИ И ТЕМИ ЖЕ входами для обеих ног; ищем источник расхождения."""
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
grads0 = jax.tree_util.tree_map(
    lambda leaf: jr.normal(jr.PRNGKey(int(leaf.size) + 1), leaf.shape), params)
grads0 = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), grads0)
ids = jr.randint(jr.PRNGKey(2), (2, 16), 0, cfg.vocab_size)
loss_fn = lambda p: model.compute_loss(p, cfg, ids, chunk_size=8)

comp = "kahan"
st = optimizer.init_state(params, master_dtype="bf16", master_comp=comp)
step_l = optimizer.make_step(cfg, master_dtype="bf16", master_comp=comp)
step_f = optimizer.make_step(cfg, master_dtype="bf16", master_comp=comp, fuse=True)
lr = 1e-3
_, g1 = jax.value_and_grad(loss_fn)(params)
p1, st1 = step_l(params, g1, st, lr)
_, g2 = jax.value_and_grad(loss_fn)(p1)

a_p, a_s = step_l(p1, g2, st1, lr)
b_p, b_s = step_l(p1, g2, st1, lr)
print("leaf deterministic:", all(bool(jnp.all(x == y)) for x, y in zip(
    jax.tree_util.tree_leaves(a_s), jax.tree_util.tree_leaves(b_s))))
c_p, c_s = step_f(p1, g2, st1, lr)
wl = jax.tree_util.tree_flatten_with_path(a_s)[0]
wf = jax.tree_util.tree_flatten_with_path(c_s)[0]
n_diff = 0
for (path, x), (_, y) in zip(wl, wf):
    xs = jax.tree_util.keystr(path)
    if xs.endswith("]") and ("mlp." in xs or "attn." in xs):
        pass
    if not bool(jnp.all(x == y)):
        n_diff += 1
        if n_diff <= 12:
            d = float(jnp.max(jnp.abs(x.astype(jnp.float32) - y.astype(jnp.float32))))
            print("DIFF", xs, x.dtype, f"{d:.3e}")
print("total state diffs:", n_diff, "of", len(wl))
print("params equal:", all(bool(jnp.all(x == y)) for x, y in zip(
    jax.tree_util.tree_leaves(a_p), jax.tree_util.tree_leaves(c_p))))
