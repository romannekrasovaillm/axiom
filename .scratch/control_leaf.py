"""Контроль: leaf vs leaf (два независимых билда) и eager-многошаг, small cfg seed=11.

Если два независимых билда ОДНОГО и того же leaf-шага расходятся так же, как
leaf vs fused, то расхождение — свойство пересборки графа (XLA lowering), а не
фьюзинга.
"""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("NET_JAX_BACKEND", "cpu")
os.environ.setdefault("AXIOM_KDA_CC_IMPL", "jax")
ROOT = "/home/roman/.arch-ml/worktrees/axiom-43e236d789188868/theseus-20261015183027"
sys.path.insert(0, ROOT + "/net/tests"); sys.path.insert(0, ROOT)
import numpy as np
import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import jax.random as jr
from conftest import small_config
from net import model, optimizer

cfg = small_config()
seed = 11


def leg(steps, fuse, jit=True, build_tag=0):
    params = model.init_params(jr.PRNGKey(seed), cfg)
    grads = jax.tree_util.tree_map(
        lambda leaf: jr.normal(jr.PRNGKey(int(leaf.size) + seed + 1), leaf.shape), params)
    params = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), params)
    grads = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), grads)
    ids = jr.randint(jr.PRNGKey(2 + seed), (2, 16), 0, cfg.vocab_size)
    state = optimizer.init_state(params, master_dtype="bf16", master_comp="kahan")
    step = optimizer.make_step(cfg, master_dtype="bf16", master_comp="kahan", fuse=fuse, jit=jit)
    losses = []
    for _ in range(steps):
        loss, grads = jax.value_and_grad(
            lambda p: model.compute_loss(p, cfg, ids, chunk_size=8))(params)
        losses.append(float(loss))
        params, state = step(params, grads, state, 1e-3)
    return params, state, losses


def summary(a, b, tag):
    worst_p, n_p = 0.0, 0
    for x, y in zip(jax.tree_util.tree_leaves(a[0]), jax.tree_util.tree_leaves(b[0])):
        if x is None:
            continue
        d = np.abs(np.asarray(x, np.float64) - np.asarray(y, np.float64))
        if d.size:
            worst_p = max(worst_p, float(d.max()))
            n_p += int(np.count_nonzero(d))
    worst_s, n_s = 0.0, 0
    for x, y in zip(jax.tree_util.tree_leaves(a[1]), jax.tree_util.tree_leaves(b[1])):
        if x is None:
            continue
        d = np.abs(np.asarray(x, np.float64) - np.asarray(y, np.float64))
        if d.size:
            worst_s = max(worst_s, float(d.max()))
            n_s += int(np.count_nonzero(d))
    dloss = max(abs(p - q) for p, q in zip(a[2], b[2]))
    print(f"[{tag}] dloss={dloss:.3e} params max|d|={worst_p:.3e} (n={n_p}) "
          f"state max|d|={worst_s:.3e} (n={n_s})")


a = leg(3, fuse=False)
b = leg(3, fuse=False)      # тот же код, независимый билд
c = leg(3, fuse=True)
summary(a, b, "leaf vs leaf (два билда)")
summary(a, c, "leaf vs fused")

# eager 3 шага (медленно, но это ответ на «формулы тождественны?»)
d = leg(3, fuse=False, jit=False)
e = leg(3, fuse=True, jit=False)
summary(d, e, "eager leaf vs eager fused (3 шага)")
