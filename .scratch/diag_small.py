"""Диагностика расхождений на small_config (jit): 1/2/3 шага, детали по слотам."""
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


def inputs(seed=0):
    params = model.init_params(jr.PRNGKey(seed), cfg)
    grads = jax.tree_util.tree_map(
        lambda leaf: jr.normal(jr.PRNGKey(int(leaf.size) + seed + 1), leaf.shape), params)
    params = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), params)
    grads = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), grads)
    return params, grads


def report(a, b, tag):
    la = jax.tree_util.tree_flatten_with_path(a)[0]
    lb = dict((jax.tree_util.keystr(p), v) for p, v in jax.tree_util.tree_flatten_with_path(b)[0])
    worst = 0.0
    rows = []
    for path, x in la:
        y = lb[jax.tree_util.keystr(path)]
        if x is None or y is None:
            continue
        xa, ya = np.asarray(x), np.asarray(y)
        n = int(np.count_nonzero(xa != ya)) if xa.size else 0
        if n:
            d = float(np.max(np.abs(xa.astype(np.float64) - ya.astype(np.float64))))
            worst = max(worst, d)
            if len(rows) < 8:
                rows.append((jax.tree_util.keystr(path), xa.dtype.name, n, xa.size, d))
    print(f"[{tag}] worst={worst:.3e} n_leaves_diff={sum(1 for _ in rows)}")
    for r in rows:
        print("   ", r)


def leg(steps, seed, fuse, jit=True):
    params, grads = inputs(seed)
    ids = jr.randint(jr.PRNGKey(2 + seed), (2, 16), 0, cfg.vocab_size)
    state = optimizer.init_state(params, master_dtype="bf16", master_comp="kahan")
    step = optimizer.make_step(cfg, master_dtype="bf16", master_comp="kahan", fuse=fuse, jit=jit)
    losses = []
    snaps = []
    for _ in range(steps):
        loss, grads = jax.value_and_grad(
            lambda p: model.compute_loss(p, cfg, ids, chunk_size=8))(params)
        losses.append(float(loss))
        params, state = step(params, grads, state, 1e-3)
        snaps.append((params, state))
    return params, state, losses, snaps


for steps in (1, 3):
    pl, sl, ll, snap_l = leg(steps, 11, False)
    pf, sf, lf, snap_f = leg(steps, 11, True)
    print(f"== steps={steps} seed=11 dloss={max(abs(a-b) for a, b in zip(ll, lf)):.3e}")
    report(pl, pf, f"params steps={steps}")
    report(sl, sf, f"state steps={steps}")

pe_list = []
pl, sl, ll, _ = leg(1, 11, False, jit=False)
pf, sf, lf, _ = leg(1, 11, True, jit=False)
print("eager 1 step:", "loss equal:", ll == lf)
report(pl, pf, "eager params")
report(sl, sf, "eager state")
