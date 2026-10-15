"""Census конверсий на геометрии l3-full (абстрактное дерево) — веха-11."""
import os, sys, collections
os.environ.setdefault("JAX_PLATFORMS", "cpu")
ROOT = "/home/roman/.arch-ml/worktrees/axiom-43e236d789188868/theseus-20261015183027"
sys.path.insert(0, ROOT + "/net/tests"); sys.path.insert(0, ROOT)
import numpy as np
import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
from net import model, optimizer
from net.config import load_config

cfg = load_config(ROOT + "/net/config.json")
shapes = jax.eval_shape(lambda key: model.init_params(key, cfg), jax.ShapeDtypeStruct((2,), jnp.uint32))
print("leaves:", len(jax.tree_util.tree_leaves(shapes)))
# bf16 master/grads (как в прогоне-лидере), состояние bf16-режима
states = jax.eval_shape(
    lambda p: optimizer.init_state(p, master_dtype="bf16", master_comp="kahan"),
    jax.tree_util.tree_map(lambda x: jax.ShapeDtypeStruct(x.shape, jnp.bfloat16), shapes),
)
tree_bf16 = jax.tree_util.tree_map(lambda x: jax.ShapeDtypeStruct(x.shape, jnp.bfloat16), shapes)
jaxpr_leaf = jax.make_jaxpr(
    optimizer.make_step(cfg, master_dtype="bf16", master_comp="kahan", jit=False),
)(tree_bf16, tree_bf16, states, 1e-3)
jaxpr_fuse = jax.make_jaxpr(
    optimizer.make_step(cfg, master_dtype="bf16", master_comp="kahan", fuse=True, jit=False),
)(tree_bf16, tree_bf16, states, 1e-3)
for name, jp in (("leaf", jaxpr_leaf), ("fused", jaxpr_fuse)):
    c = collections.Counter(str(e.primitive) for e in jp.jaxpr.eqns)
    print(f"[{name}] eqns={sum(c.values())} convert={c['convert_element_type']} "
          f"sort={c.get('sort', 0) + c.get('argsort', 0)} dot={c['dot_general']} "
          f"reduce={sum(v for k, v in c.items() if 'reduce' in k)}")
    print("   top:", c.most_common(8))
# сколько групп у фьюзинга (оценка: число различ. (group, shape, dtypes, state-sig))
import jax.tree_util as jtu
from net import master_dtype as md
sig = []
for path, leaf in jtu.tree_flatten_with_path(tree_bf16)[0]:
    name = optimizer.leaf_name(path[-1]) if path else ""
    g = optimizer.classify_leaf(name, leaf.ndim)
    st = None
    sig.append(g)
print("groups:", collections.Counter(sig))
