"""Числа jit-паритета fused↔leaf под 0112-venv (для порогов теста_41)."""
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
from conftest import tiny_config
from net import model, optimizer

cfg = tiny_config()


def leg(steps, seed, fuse, comp="kahan", lr=1e-3):
    params = model.init_params(jr.PRNGKey(seed), cfg)
    grads = jax.tree_util.tree_map(
        lambda leaf: jr.normal(jr.PRNGKey(int(leaf.size) + seed + 1), leaf.shape), params)
    params = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), params)
    grads = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), grads)
    ids = jr.randint(jr.PRNGKey(2 + seed), (2, 16), 0, cfg.vocab_size)
    state = optimizer.init_state(params, master_dtype="bf16", master_comp=comp)
    step = optimizer.make_step(cfg, master_dtype="bf16", master_comp=comp, fuse=fuse)
    losses = []
    for _ in range(steps):
        loss, grads = jax.value_and_grad(
            lambda p: model.compute_loss(p, cfg, ids, chunk_size=8))(params)
        losses.append(float(loss))
        params, state = step(params, grads, state, lr)
    return params, state, losses


def cats(a, b):
    out = {}
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        if x is None or y is None:
            continue
        d = np.asarray(x).dtype.name
        xa, ya = np.asarray(x), np.asarray(y)
        if xa.size and not np.array_equal(xa, ya):
            k = out.setdefault(d, [0.0, 0])
            k[0] = max(k[0], float(np.max(np.abs(xa.astype(np.float64) - ya.astype(np.float64)))))
            k[1] += int(np.count_nonzero(xa != ya))
    return out


for seed in (0, 11):
    pl, sl, ll = leg(3, seed, False)
    pf, sf, lf = leg(3, seed, True)
    dp = cats(pl, pf)
    ds = cats(sl, sf)
    dloss = max(abs(a - b) for a, b in zip(ll, lf))
    print(f"seed={seed} dloss={dloss:.3e} params={dp} state={ds}")
