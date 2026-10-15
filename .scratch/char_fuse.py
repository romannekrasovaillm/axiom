"""Характеризация расхождения fused vs leaf: eager vs jit, масштаб и точки."""
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
comp = "kahan"


def compare(a, b):
    """(max|d| по master/moments/comp, где diff) — раздельно по категориям."""
    la = jax.tree_util.tree_flatten_with_path(a)[0]
    lb = dict((jax.tree_util.keystr(p), v) for p, v in jax.tree_util.tree_flatten_with_path(b)[0])
    worst = {}
    ndiff = {}
    for path, x in la:
        y = lb[jax.tree_util.keystr(path)]
        if x is None or y is None:
            continue
        xs = jax.tree_util.keystr(path)
        if xs.endswith("[2]") or (xs.endswith("[1]") and "muon" in ""):
            pass
        cat = "other"
        if xs.endswith("[0]"):
            cat = "slot0"
        elif xs.endswith("[1]"):
            cat = "slot1"
        elif xs.endswith("[2]"):
            cat = "slot2"
        xa, ya = np.asarray(x, np.float32), np.asarray(y, np.float32)
        if xa.size and not np.array_equal(xa, ya):
            d = float(np.max(np.abs(xa - ya)))
            worst[cat] = max(worst.get(cat, 0.0), d)
            ndiff[cat] = ndiff.get(cat, 0) + int(np.sum(xa != ya))
    return worst, ndiff


def run_leg(steps=5, lr=1e-3, seed=0):
    base = model.init_params(jr.PRNGKey(seed), cfg)
    params = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), base)
    grads = jax.tree_util.tree_map(
        lambda leaf: jr.normal(jr.PRNGKey(int(leaf.size) + seed + 1), leaf.shape), params)
    grads = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), grads)
    ids = jr.randint(jr.PRNGKey(2 + seed), (2, 16), 0, cfg.vocab_size)
    loss_fn = lambda p: model.compute_loss(p, cfg, ids, chunk_size=8)
    st_l = optimizer.init_state(params, master_dtype="bf16", master_comp=comp)
    st_f = optimizer.init_state(params, master_dtype="bf16", master_comp=comp)
    step_l = optimizer.make_step(cfg, master_dtype="bf16", master_comp=comp)
    step_f = optimizer.make_step(cfg, master_dtype="bf16", master_comp=comp, fuse=True)
    pl, pf = params, params
    losses = ([], [])
    for _ in range(steps):
        ll, gl = jax.value_and_grad(loss_fn)(pl)
        lf, gf = jax.value_and_grad(loss_fn)(pf)
        losses[0].append(float(ll)); losses[1].append(float(lf))
        pl, st_l = step_l(pl, gl, st_l, lr)
        pf, st_f = step_f(pf, gf, st_f, lr)
    return params, pl, pf, st_l, st_f, losses


def maxdiff(a, b):
    out = 0.0
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        if x is None or y is None:
            continue
        xa, ya = np.asarray(x, np.float32), np.asarray(y, np.float32)
        if xa.size:
            out = max(out, float(np.max(np.abs(xa - ya))))
    return out


print("=== jit, 5 шагов, 3 сида ===")
for seed in (0, 3, 11):
    params, pl, pf, st_l, st_f, losses = run_leg(steps=5, seed=seed)
    w, nd = compare(st_l, st_f)
    print(f"seed={seed} dloss={max(abs(a-b) for a, b in zip(*losses)):.3e} "
          f"max|dp|={maxdiff(pl, pf):.3e} state: {w} ndiff={nd}")

print("=== eager (jit=False), 2 шага ===")
base = model.init_params(jr.PRNGKey(0), cfg)
params = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), base)
grads = jax.tree_util.tree_map(
    lambda leaf: jr.normal(jr.PRNGKey(int(leaf.size) + 1), leaf.shape), params)
grads = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), grads)
ids = jr.randint(jr.PRNGKey(2), (2, 16), 0, cfg.vocab_size)
loss_fn = lambda p: model.compute_loss(p, cfg, ids, chunk_size=8)
st_l = optimizer.init_state(params, master_dtype="bf16", master_comp=comp)
st_f = optimizer.init_state(params, master_dtype="bf16", master_comp=comp)
e_l = optimizer.make_step(cfg, master_dtype="bf16", master_comp=comp, jit=False)
e_f = optimizer.make_step(cfg, master_dtype="bf16", master_comp=comp, fuse=True, jit=False)
_, g1 = jax.value_and_grad(loss_fn)(params)
p1, st1 = e_l(params, g1, st_l, 1e-3)
p1f, st1f = e_f(params, g1, st_f, 1e-3)
print("eager step1: dp=%.3e ds=%.3e" % (maxdiff(p1, p1f), maxdiff(st1, st1f)))
w, nd = compare(st1, st1f)
print("  state cats:", w, nd)
