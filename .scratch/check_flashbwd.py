"""Проверка проводки flash-bwd в net/mla.py (веха-12): smoke в реальных ветках."""
import sys
import os

sys.path.insert(0, "net/tests")
sys.path.insert(0, ".")
import conftest  # noqa: F401 — пины ADR-010 до import jax

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from net import mla
from net import model


def walk(jaxpr, out=None, depth=0):
    if out is None:
        out = []
    if depth > 25 or jaxpr is None:
        return out
    for eqn in jaxpr.eqns:
        for var in eqn.outvars:
            av = getattr(var, "aval", None)
            if av is not None:
                out.append((str(eqn.primitive.name), tuple(av.shape), str(av.dtype)))
        for value in eqn.params.values():
            items = (
                list(value.values()) if isinstance(value, dict)
                else list(value) if isinstance(value, (list, tuple))
                else [value]
            )
            for sub in items:
                inner = sub if hasattr(sub, "eqns") else getattr(sub, "jaxpr", None)
                if inner is not None and hasattr(inner, "eqns"):
                    walk(inner, out, depth + 1)
    return out


def graph_values(fn):
    return walk(jax.make_jaxpr(fn)().jaxpr)


class gate:
    def __init__(self, val):
        self.val = val

    def __enter__(self):
        self.saved = os.environ.get(mla.MLA_FLASH_BWD_ENV)
        if self.val is None:
            os.environ.pop(mla.MLA_FLASH_BWD_ENV, None)
        else:
            os.environ[mla.MLA_FLASH_BWD_ENV] = self.val

    def __exit__(self, *exc):
        if self.saved is None:
            os.environ.pop(mla.MLA_FLASH_BWD_ENV, None)
        else:
            os.environ[mla.MLA_FLASH_BWD_ENV] = self.saved


def main():
    cfg = conftest.small_config(mla_flash_bwd_tile=4)
    params = mla.init_mla(jr.PRNGKey(0), cfg)
    x = jr.normal(jr.PRNGKey(1), (2, 16, cfg.hidden))
    B, T, H = 2, 16, cfg.num_heads

    with gate(None):
        off = mla._dense_apply(params, cfg, x)
    with gate("1"):
        on = mla._dense_apply(params, cfg, x)
    print("fwd bitwise:", bool(jnp.array_equal(off, on)))

    def loss(p):
        return jnp.sum(mla._dense_apply(p, cfg, x) ** 2)

    with gate(None):
        v_off, g_off = jax.value_and_grad(loss)(params)
    with gate("1"):
        v_on, g_on = jax.value_and_grad(loss)(params)
    print("loss bitwise:", bool(jnp.array_equal(v_off, v_on)))
    leaves_o = jax.tree_util.tree_leaves(g_off)
    leaves_n = jax.tree_util.tree_leaves(g_on)
    scale = max([1e-30] + [float(np.max(np.abs(np.asarray(l)))) for l in leaves_o])
    rel = max(float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
              for a, b in zip(leaves_n, leaves_o)) / scale
    print(f"grad max_rel: {rel:.3e}")

    # прямая проба custom-операции
    q = jr.normal(jr.PRNGKey(2), (B, T, H, cfg.mla_head_dim))
    k = jr.normal(jr.PRNGKey(3), (B, T, H, cfg.mla_head_dim))
    v = jr.normal(jr.PRNGKey(4), (B, T, H, cfg.mla_head_dim))
    mask = mla._causal_mask(T)
    op = mla._attn_out_flashbwd_op(int(cfg.mla_flash_bwd_tile))

    def sc(q_, k_, v_):
        return jnp.sum(op(q_, k_, v_, mask) ** 2)

    _, pb = jax.vjp(sc, q, k, v)
    vals = walk(jax.make_jaxpr(lambda c: pb(c))(jnp.ones(())).jaxpr)
    full = [s for s in vals if len(s[1]) == 4 and s[1][2:] == (T, T) and "float" in s[2]]
    tile = [s for s in vals if len(s[1]) == 4 and s[1][2:] == (4, 4) and "float" in s[2]]
    print("bwd full-score float vars:", full)
    print("bwd tile-shaped float vars:", len(tile), tile[:2])

    # teeth: dense reference pullback
    def scr(q_, k_, v_):
        s = jnp.einsum("bthd,bshd->bhts", q_, k_)
        a = jax.nn.softmax(jnp.where(mask, s, jnp.finfo(jnp.float32).min), axis=-1)
        return jnp.sum(jnp.einsum("bhts,bshd->bthd", a, v_) ** 2)

    _, pbr = jax.vjp(scr, q, k, v)
    valsr = walk(jax.make_jaxpr(lambda c: pbr(c))(jnp.ones(())).jaxpr)
    fullr = [s for s in valsr if len(s[1]) == 4 and s[1][2:] == (T, T) and "float" in s[2]]
    print("dense ref full-score vars (teeth):", len(fullr))

    # whole-model pullback (policy none)
    cfg2 = conftest.small_config()
    params2 = model.init_params(jr.PRNGKey(0), cfg2)
    ids = jr.randint(jr.PRNGKey(2), (2, 16), 1, cfg2.vocab_size)

    def mloss(p):
        return model.compute_loss(p, cfg2, ids, 16, grad_ckpt_policy="none")

    with gate(None):
        _, pb2 = jax.vjp(mloss, params2)
        vals2 = walk(jax.make_jaxpr(lambda c: pb2(c))(jnp.ones(())).jaxpr)
    full2 = [s for s in vals2 if len(s[1]) == 4 and s[1][2:] == (16, 16) and "float" in s[2]]
    print("whole-model pullback OFF full-score vars:", len(full2))

    with gate("1"):
        _, pb3 = jax.vjp(mloss, params2)
        vals3 = walk(jax.make_jaxpr(lambda c: pb3(c))(jnp.ones(())).jaxpr)
    full3 = [s for s in vals3 if len(s[1]) == 4 and s[1][2:] == (16, 16) and "float" in s[2]]
    print("whole-model pullback ON full-score vars:", len(full3), full3[:4])


if __name__ == "__main__":
    main()
