"""Калибровка порогов паритета: seeds × 3 шага, масштабы расхождений (small cfg)."""
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
from conftest import small_config, tiny_config
from net import model, optimizer


def bf16_steps(a, b):
    """Число шагов сетки bf16 между a и b (знаковый-magnitude: |bits| разность)."""
    ia = np.asarray(a).view(np.uint16).astype(np.int64)
    ib = np.asarray(b).view(np.uint16).astype(np.int64)
    same_sign = (ia >> 15) == (ib >> 15)
    mag = np.abs(ia - ib)
    cross = (ia & 0x7FFF) + (ib & 0x7FFF)
    return np.where(same_sign, mag, cross)


def leg(cfg, steps, seed, fuse, lr=1e-3):
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


for cfg_name, cfg in (("small", small_config()), ("tiny", tiny_config())):
    for seed in (0, 11, 21):
        pl, sl, ll = leg(cfg, 3, seed, False)
        pf, sf, lf = leg(cfg, 3, seed, True)
        p_gap = 0
        p_n = 0
        for x, y in zip(jax.tree_util.tree_leaves(pl), jax.tree_util.tree_leaves(pf)):
            if x is None:
                continue
            g = bf16_steps(x, y)
            p_gap = max(p_gap, int(g.max()) if g.size else 0)
            p_n += int(np.count_nonzero(g))
        mom, mom_scale, mom_n = 0.0, 0.0, 0
        comp, comp_n = 0.0, 0
        for x, y in zip(jax.tree_util.tree_leaves(sl), jax.tree_util.tree_leaves(sf)):
            if x is None:
                continue
            xa, ya = np.asarray(x), np.asarray(y)
            d = np.abs(xa.astype(np.float64) - ya.astype(np.float64))
            if xa.dtype.name == "float32" and d.size:
                mom = max(mom, float(d.max()))
                mom_scale = max(mom_scale, float(np.abs(xa).max()))
                mom_n += int(np.count_nonzero(d))
            if xa.dtype.name == "bfloat16" and d.size:
                comp = max(comp, float(d.max()))
                comp_n += int(np.count_nonzero(d))
        dloss = max(abs(a - b) for a, b in zip(ll, lf))
        print(f"[{cfg_name} seed={seed}] dloss={dloss:.3e} master_gap(ulp bf16)={p_gap} "
              f"master_nelem_diff={p_n} moment_max={mom:.3e} (scale {mom_scale:.3e}, n={mom_n}) "
              f"comp_max={comp:.3e} n={comp_n}")
