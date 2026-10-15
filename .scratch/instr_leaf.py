"""Инструментовка: comp-лист b_alpha на входе шага-2 — где расходится."""
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
from net import model, optimizer, master_dtype as md

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
p1f, st1f = step_f(params, g1, st, lr)
print("params/state after step1 equal:", all(bool(jnp.all(a == b)) for a, b in zip(
    jax.tree_util.tree_leaves(p1), jax.tree_util.tree_leaves(p1f))))
p2, st2 = step_l(p1, g2, st1, lr)
p2f, st2f = step_f(p1f, g2, st1f, lr)


def slots(tree, needle):
    out = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        s = jax.tree_util.keystr(path)
        if needle in s:
            out[s[s.index(needle):]] = leaf
    return out

for prefix, group in (("layers[1].attn.b_alpha", "adamw_vector"),
                      ("layers[1].mlp.W_down", "muon_matrix")):
    p_leaf = slots(p1, prefix)["%s" % prefix]
    g_leaf = slots(g2, prefix)[prefix]
    s1 = slots(st1, prefix)
    s2l = slots(st2, prefix)
    s2f = slots(st2f, prefix)
    print(prefix, group, "keys:", sorted(s1))
    if group == "adamw_vector":
        m1, v1, c1 = s1[prefix + "[0]"], s1[prefix + "[1]"], s1[prefix + "[2]"]
        g32 = g_leaf.astype(jnp.float32)
        m = 0.9 * m1 + 0.1 * g32
        v = 0.95 * v1 + 0.05 * (g32 * g32)
        mh = m / 0.1
        vh = v / 0.05
        p32 = p_leaf.astype(jnp.float32)
        target = p32 * (1.0 - lr * cfg.weight_decay) - lr * mh / (jnp.sqrt(vh) + 1e-8)
        update = target - p32
        hand_p, hand_c = md.kahan_apply(p_leaf, update, c1)
        cl, cf = s2l[prefix + "[2]"], s2f[prefix + "[2]"]
    else:
        m1, c1 = s1[prefix + "[0]"], s1[prefix + "[1]"]
        g32 = g_leaf.astype(jnp.float32)
        m = 0.95 * m1 + 0.05 * g32
        upd = optimizer.newtonschulz5(m)
        p32 = p_leaf.astype(jnp.float32)
        target = p32 * (1.0 - lr * cfg.weight_decay) - lr * upd
        target = jnp.clip(target, -1.0, 1.0)
        update = target - p32
        hand_p, hand_c = md.kahan_apply(p_leaf, update, c1)
        cl, cf = s2l[prefix + "[1]"], s2f[prefix + "[1]"]
    print("  comp: leaf==hand:", bool(jnp.all(cl == hand_c)),
          "fused==hand:", bool(jnp.all(cf == hand_c)),
          "leaf==fused:", bool(jnp.all(cl == cf)))
    print("  hand:", np.asarray(hand_c, np.float32)[:5])
    print("  leaf:", np.asarray(cl, np.float32)[:5])
    print("  fused:", np.asarray(cf, np.float32)[:5])
    print("  leaf-fused diff:", np.asarray(cl.astype(jnp.float32) - cf.astype(jnp.float32))[:5])
    print("  update:", np.asarray(update, np.float32)[:5])
    print("  master equal leaf==fused:", bool(jnp.all(s2l[prefix + "[0]"] == s2f[prefix + "[0]"])))
