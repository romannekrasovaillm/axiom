import sys, os
sys.path.insert(0, "net/tests"); sys.path.insert(0, ".")
import conftest  # noqa
import jax, jax.numpy as jnp, jax.random as jr
from net import mla

B, T, H, dq, tile = 2, 16, 4, 16, 4
q = jr.normal(jr.PRNGKey(2), (B, T, H, dq))
k = jr.normal(jr.PRNGKey(3), (B, T, H, dq))
v = jr.normal(jr.PRNGKey(4), (B, T, H, dq))
mask = mla._causal_mask(T)
op = mla._attn_out_flashbwd_op(tile)

_, pb = jax.vjp(lambda q_, k_, v_: jnp.sum(op(q_, k_, v_, mask)), q, k, v)
jp = jax.make_jaxpr(lambda c: pb(c))(jnp.ones(())).jaxpr

print("=== eqns with 4-D float outvars, leading (B,H) ===")
for i, eqn in enumerate(jp.eqns):
    for var in eqn.outvars:
        av = getattr(var, "aval", None)
        if av is None:
            continue
        sh = tuple(av.shape)
        if len(sh) == 4 and sh[0] == B and sh[1] == H and "float" in str(av.dtype):
            ins = [str(getattr(iv, "aval", "?").shape) if hasattr(getattr(iv, "aval", None), "shape") else "?" for iv in eqn.invars]
            print(f"#{i} {eqn.primitive.name:16s} out={sh} params={[kk for kk in eqn.params][:3]} in={ins}")
print("total eqns:", len(jp.eqns))
