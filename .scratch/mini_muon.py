"""Мини-дерево: leaf vs fused vs hand для одного Muon-листа (без модели)."""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
ROOT = "/home/roman/.arch-ml/worktrees/axiom-43e236d789188868/theseus-20261015183027"
sys.path.insert(0, ROOT + "/net/tests"); sys.path.insert(0, ROOT)
import numpy as np
import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import jax.random as jr
from net import optimizer, master_dtype as md

cfg = type("C", (), {"num_heads": 2, "weight_decay": 0.1})()

p = jax.random.normal(jr.PRNGKey(0), (8, 4)).astype(jnp.bfloat16) * 0.02
g = jax.random.normal(jr.PRNGKey(1), (8, 4)).astype(jnp.bfloat16)
tree = {"W_down": p}
grads = {"W_down": g}
state = optimizer.init_state(tree, master_dtype="bf16", master_comp="kahan")
lr = 1e-3

st_l = optimizer.make_step(cfg, master_dtype="bf16", master_comp="kahan")
st_f = optimizer.make_step(cfg, master_dtype="bf16", master_comp="kahan", fuse=True)
p_l, s_l = st_l(tree, grads, state, lr)
p_f, s_f = st_f(tree, grads, state, lr)
print("params equal:", bool(jnp.all(p_l["W_down"] == p_f["W_down"])))
print("moment equal:", bool(jnp.all(s_l["W_down"][0] == s_f["W_down"][0])))
print("comp leaf:", np.asarray(s_l["W_down"][1], np.float32))
print("comp fused:", np.asarray(s_f["W_down"][1], np.float32))

# ручная арифметика
m = 0.95 * state["W_down"][0] + 0.05 * g.astype(jnp.float32)
upd = optimizer.newtonschulz5(m)
p32 = p.astype(jnp.float32)
target = jnp.clip(p32 * (1.0 - lr * 0.1) - lr * upd, -1.0, 1.0)
hand_p, hand_c = md.kahan_apply(p, target - p32, state["W_down"][1])
print("hand param equal leaf:", bool(jnp.all(hand_p == p_l["W_down"])), "fused:", bool(jnp.all(hand_p == p_f["W_down"])))
print("hand comp equal leaf:", bool(jnp.all(hand_c == s_l["W_down"][1])), "fused:", bool(jnp.all(hand_c == s_f["W_down"][1])))
print("hand comp:", np.asarray(hand_c, np.float32))
d_l = np.asarray(s_l["W_down"][1].astype(jnp.float32) - hand_c.astype(jnp.float32))
print("leaf-hand diff:", d_l)
d_f = np.asarray(s_f["W_down"][1].astype(jnp.float32) - hand_c.astype(jnp.float32))
print("fused-hand diff:", d_f)
# посмотреть update после клипа
print("update (clipped) sample:", np.asarray((target - p32), np.float32)[:2])
