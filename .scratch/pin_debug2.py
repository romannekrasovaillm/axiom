import sys, os
sys.path.insert(0, "net/tests"); sys.path.insert(0, ".")
import conftest  # noqa
import jax, jax.numpy as jnp, jax.random as jr
from net import mla, model


def walk(jaxpr, out=None, depth=0):
    if out is None:
        out = []
    if depth > 40 or jaxpr is None:
        return out
    for eqn in jaxpr.eqns:
        for var in eqn.outvars:
            av = getattr(var, "aval", None)
            if av is not None:
                out.append((str(eqn.primitive.name), tuple(getattr(av, "shape", ())), str(av.dtype)))
        for value in eqn.params.values():
            items = (list(value.values()) if isinstance(value, dict)
                     else list(value) if isinstance(value, (list, tuple)) else [value])
            for sub in items:
                inner = sub if hasattr(sub, "eqns") else getattr(sub, "jaxpr", None)
                if inner is not None and hasattr(inner, "eqns"):
                    walk(inner, out, depth + 1)
    return out


# --- 1) operator-level pin: коллизий нет по построению ---
B, T, H, dq, tile = 2, 8, 3, 5, 2
q = jr.normal(jr.PRNGKey(2), (B, T, H, dq))
k = jr.normal(jr.PRNGKey(3), (B, T, H, dq))
v = jr.normal(jr.PRNGKey(4), (B, T, H, dq))
mask = mla._causal_mask(T)
op = mla._attn_out_flashbwd_op(tile)
_, pb = jax.vjp(lambda q_, k_, v_: jnp.sum(op(q_, k_, v_, mask)), q, k, v)
vals = walk(jax.make_jaxpr(lambda c: pb(c))(jnp.ones(())).jaxpr)
score = [(p, s, d) for p, s, d in vals if len(s) == 4 and s[0] == B and s[1] == H and "float" in d]
print("operator pin: score-layout vars:", len(score))
print("  max dims:", sorted({(s[2], s[3]) for _, s, _ in score}))
print("  any full (T,T):", any(s[2] == T or s[3] == T for _, s, _ in score))
print("  any tile (tile,tile):", any(s[2] == tile and s[3] == tile for _, s, _ in score))

# --- 2) whole-model pin c tile=3 (H=4, T=12, dq=16) ---
cfg = conftest.small_config(mla_flash_bwd_tile=3)
params = model.init_params(jr.PRNGKey(0), cfg)
ids = jr.randint(jr.PRNGKey(2), (2, 12), 1, cfg.vocab_size)
Tw, Hw = 12, cfg.num_heads

def loss(p):
    return model.compute_loss(p, cfg, ids, 5, grad_ckpt_policy="none")

for g in (None, "1"):
    if g is None:
        os.environ.pop("AXIOM_MLA_FLASH_BWD", None)
    else:
        os.environ["AXIOM_MLA_FLASH_BWD"] = g
    _, pb2 = jax.vjp(loss, params)
    vals2 = walk(jax.make_jaxpr(lambda c: pb2(c))(jnp.ones(())).jaxpr)
    score2 = [(p, s, d) for p, s, d in vals2 if len(s) == 4 and s[0] == 2 and s[1] == Hw and "float" in d]
    print(f"model pin gate={g}: score-layout vars {len(score2)}; max dims:",
          sorted({(s[2], s[3]) for _, s, _ in score2})[-4:])
os.environ.pop("AXIOM_MLA_FLASH_BWD", None)
