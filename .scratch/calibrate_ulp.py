"""Относительные меры расхождений мастеров (ulp-норма) и моментов — seeds 11/21."""
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


def ulp_bf16(x):
    x = np.asarray(x, np.float64)
    mag = np.abs(x)
    out = np.zeros_like(mag)
    nz = mag > 0
    e = np.floor(np.log2(np.where(nz, mag, 1.0)))
    out[nz] = np.exp2(e[nz] - 8)
    return out


def leg(steps, seed, fuse, lr=1e-3):
    params = model.init_params(jr.PRNGKey(seed), cfg)
    grads = jax.tree_util.tree_map(
        lambda leaf: jr.normal(jr.PRNGKey(int(leaf.size) + seed + 1), leaf.shape), params)
    params = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), params)
    grads = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), grads)
    ids = jr.randint(jr.PRNGKey(2 + seed), (2, 16), 0, cfg.vocab_size)
    state = optimizer.init_state(params, master_dtype="bf16", master_comp="kahan")
    step = optimizer.make_step(cfg, master_dtype="bf16", master_comp="kahan", fuse=fuse)
    losses = []
    for _ in range(steps):
        loss, grads = jax.value_and_grad(
            lambda p: model.compute_loss(p, cfg, ids, chunk_size=8))(params)
        losses.append(float(loss))
        params, state = step(params, grads, state, lr)
    return params, state, losses


for seed in (11, 21):
    pl, sl, ll = leg(3, seed, False)
    pf, sf, lf = leg(3, seed, True)
    worst_r = 0.0
    worst_abs = 0.0
    n = 0
    for x, y in zip(jax.tree_util.tree_leaves(pl), jax.tree_util.tree_leaves(pf)):
        if x is None:
            continue
        xa, ya = np.asarray(x, np.float64), np.asarray(y, np.float64)
        d = np.abs(xa - ya)
        if not d.size:
            continue
        ulp = np.maximum(ulp_bf16(xa), ulp_bf16(ya))
        r = np.where(ulp > 0, d / np.maximum(ulp, 1e-300), 0.0)
        worst_r = max(worst_r, float(r.max()))
        worst_abs = max(worst_abs, float(d.max()))
        n += int(np.count_nonzero(d))
    mom_abs, mom_rel, mom_n, mom_scale = 0.0, 0.0, 0, 0.0
    for x, y in zip(jax.tree_util.tree_leaves(sl), jax.tree_util.tree_leaves(sf)):
        if x is None or np.asarray(x).dtype.name != "float32":
            continue
        xa, ya = np.asarray(x, np.float64), np.asarray(y, np.float64)
        d = np.abs(xa - ya)
        mom_abs = max(mom_abs, float(d.max()) if d.size else 0.0)
        mom_scale = max(mom_scale, float(np.abs(xa).max()) if xa.size else 0.0)
        mom_n += int(np.count_nonzero(d))
    dloss = max(abs(a - b) for a, b in zip(ll, lf))
    print(f"seed={seed} dloss={dloss:.3e} | master: max|d|={worst_abs:.3e} "
          f"max ulp-ratio={worst_r:.2f} n_diff={n} | moments: max|d|={mom_abs:.3e} "
          f"scale={mom_scale:.3e} rel={mom_abs / max(mom_scale, 1e-30):.3e} n={mom_n}")
