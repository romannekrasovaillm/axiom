"""Быстрая проверка вехи-11 (fuse) до тестов: паритет, census, гейты."""
import collections, os, sys, warnings
os.environ.setdefault("JAX_PLATFORMS", "cpu")
ROOT = "/home/roman/.arch-ml/worktrees/axiom-43e236d789188868/theseus-20261015183027"
sys.path.insert(0, ROOT + "/net/tests")
sys.path.insert(0, ROOT)

import numpy as np
import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import jax.random as jr
from conftest import tiny_config, small_config
from net import model, optimizer


def eqn_counts(fn, *args):
    closed = jax.make_jaxpr(fn)(*args)
    return collections.Counter(str(eqn.primitive) for eqn in closed.jaxpr.eqns)


def inputs(cfg, seed=0):
    params = model.init_params(jr.PRNGKey(seed), cfg)
    grads = jax.tree_util.tree_map(
        lambda leaf: jr.normal(jr.PRNGKey(int(leaf.size) + seed + 1), leaf.shape), params)
    grads = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), grads)
    params = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), params)
    return params, grads


def max_diff(a, b):
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    assert len(la) == len(lb)
    worst = 0.0
    for x, y in zip(la, lb):
        if x is None and y is None:
            continue
        xa = np.asarray(x, np.float32); ya = np.asarray(y, np.float32)
        worst = max(worst, float(np.max(np.abs(xa - ya))) if xa.size else 0.0)
    return worst


def main():
    cfg = tiny_config()
    # (1) гейты
    os.environ.pop(optimizer.OPT_FUSE_ENV, None)
    assert optimizer.opt_fuse_mode() is False
    os.environ[optimizer.OPT_FUSE_ENV] = "banana"
    try:
        optimizer.opt_fuse_mode()
        raise AssertionError("опечатка прошла")
    except ValueError as e:
        print("[gate] fail-closed ok:", str(e)[:60])
    os.environ[optimizer.OPT_FUSE_ENV] = "1"
    assert optimizer.opt_fuse_mode() is True
    assert optimizer.opt_fuse_mode("0") is False
    assert optimizer.opt_fuse_mode(True) is True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert optimizer.opt_fuse_mode("1", master_dtype="fp32") is False
        print("[gate] fp32-игнор:", str(caught[0].message)[:80])
    os.environ.pop(optimizer.OPT_FUSE_ENV, None)

    params, grads = inputs(cfg)

    for comp in ("kahan", "none", "sr"):
        state_c = optimizer.init_state(params, master_dtype="bf16", master_comp=comp)
        leaf = optimizer.make_step(cfg, master_dtype="bf16", master_comp=comp, jit=False)
        fused = optimizer.make_step(cfg, master_dtype="bf16", master_comp=comp, fuse=True, jit=False)
        p0, s0 = leaf(params, grads, state_c, 1e-3)
        p1, s1 = fused(params, grads, state_c, 1e-3)
        dp, ds = max_diff(p0, p1), max_diff(s0, s1)
        print(f"[parity one step] comp={comp}: max|dp|={dp:.3e} max|ds|={ds:.3e}")
        assert dp == 0.0 and ds == 0.0, f"не побитово: {dp} {ds}"

    # три шага мини-обучения
    ids = jr.randint(jr.PRNGKey(2), (2, 16), 0, cfg.vocab_size)
    loss_fn = lambda p: model.compute_loss(p, cfg, ids, chunk_size=8)
    for comp in ("kahan", "none"):
        st_l = optimizer.init_state(params, master_dtype="bf16", master_comp=comp)
        st_f = optimizer.init_state(params, master_dtype="bf16", master_comp=comp)
        step_l = optimizer.make_step(cfg, master_dtype="bf16", master_comp=comp)
        step_f = optimizer.make_step(cfg, master_dtype="bf16", master_comp=comp, fuse=True)
        pl, pf = params, params
        losses_l, losses_f = [], []
        for _ in range(3):
            ll, gl = jax.value_and_grad(loss_fn)(pl)
            lf, gf = jax.value_and_grad(loss_fn)(pf)
            losses_l.append(float(ll)); losses_f.append(float(lf))
            pl, st_l = step_l(pl, gl, st_l, 1e-3)
            pf, st_f = step_f(pf, gf, st_f, 1e-3)
        print(f"[parity 3 steps] comp={comp}: dloss={max(abs(a-b) for a,b in zip(losses_l,losses_f)):.3e} "
              f"max|dp|={max_diff(pl,pf):.3e} max|ds|={max_diff(st_l,st_f):.3e}")
        assert losses_l == losses_f and max_diff(pl, pf) == 0.0 and max_diff(st_l, st_f) == 0.0

    # census конвертов
    st = optimizer.init_state(params, master_dtype="bf16", master_comp="kahan")
    c_leaf = eqn_counts(optimizer.make_step(cfg, master_dtype="bf16", jit=False), params, grads, st, 1e-3)
    c_fuse = eqn_counts(optimizer.make_step(cfg, master_dtype="bf16", fuse=True, jit=False), params, grads, st, 1e-3)
    n_leaf, n_fuse = c_leaf["convert_element_type"], c_fuse["convert_element_type"]
    print(f"[census tiny] convert: leaf={n_leaf} fuse={n_fuse} | eqns: leaf={sum(c_leaf.values())} fuse={sum(c_fuse.values())}")
    assert n_fuse < n_leaf
    # fp32-путь: fuse игнорируется, дерево результат побитово как у leaf
    st32 = optimizer.init_state(jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), params))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p32f = optimizer.make_step(cfg, master_dtype="fp32", fuse=True, jit=False)(jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), params), grads, st32, 1e-3)[0]
    p32l = optimizer.make_step(cfg, master_dtype="fp32", jit=False)(jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), params), grads, st32, 1e-3)[0]
    assert max_diff(p32f, p32l) == 0.0
    print("[ok] все проверки прошли")


if __name__ == "__main__":
    main()
