"""TEMP diagnostic (deleted before commit): one-chunk tile-form debug."""

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402

f32, bf = jnp.float32, jnp.bfloat16


def main():
    C, dk, dv, tile = 32, 16, 8, 16
    key = jr.PRNGKey(0)
    k1, k2, k3, k4, k5 = jr.split(key, 5)
    q = jr.normal(k1, (C, dk)) * 0.5
    k = jr.normal(k2, (C, dk)) * 0.5
    do = jr.normal(k3, (C, dv))
    beta = jr.uniform(k4, (C,))
    alpha = jnp.exp(-1.2 * jr.uniform(k5, (C, dk)))
    # the forward pieces the adjoint consumes (fp32 exact forms here)
    la = jnp.log(alpha)
    G = jnp.cumsum(la, axis=0)
    tril = (jnp.arange(C)[:, None] >= jnp.arange(C)[None, :]).astype(f32)
    daqk = ((do @ jnp.zeros((dv, C)) + jr.normal(k3, (C, C)) * 0.3) * tril)
    # reference dr/dc
    E = jnp.exp(G[:, None, :] - G[None, :, :])          # (C, C, dk)
    dr_ref = jnp.einsum("ci,cid,id->cd", daqk, E, k)
    dc_ref = jnp.einsum("ci,cid,cd->id", daqk, E, q)
    # tile form
    ar = jnp.arange(C)
    dr = jnp.zeros((C, dk), f32)
    dc = jnp.zeros((C, dk), f32)
    for p in range(0, C, tile):
        rm = ((ar[:, None] >= p) & (ar[:, None] < p + tile)).astype(f32)
        dT = (daqk * rm).astype(f32)
        anchor = jnp.sum(G * (ar == p)[:, None].astype(f32), axis=0)
        rf = jnp.exp(jnp.minimum(G - anchor[None, :], 0.0))
        cf = jnp.exp(anchor[None, :] - G)
        dr = dr + rf * (dT @ (k * cf))
        dc = dc + cf * (dT.T @ (q * rf))
    print("dr ref max:", float(jnp.max(jnp.abs(dr_ref))), "mine:",
          float(jnp.max(jnp.abs(dr))))
    print("dr err:", float(jnp.max(jnp.abs(dr - dr_ref))))
    print("dc err:", float(jnp.max(jnp.abs(dc - dc_ref))))
    # diagnose: which (c,i) pairs contribute where
    for p in (0, 16):
        rm = ((ar[:, None] >= p) & (ar[:, None] < p + tile)).astype(f32)
        anchor = jnp.sum(G * (ar == p)[:, None].astype(f32), axis=0)
        rf = jnp.exp(jnp.minimum(G - anchor[None, :], 0.0))
        cf = jnp.exp(anchor[None, :] - G)
        Ehat = rf[:, None, :] * cf[None, :, :]           # (C, C, dk)
        print(f"tile {p}: max|Ehat-E| over tile rows="
              f"{float(jnp.max(jnp.abs((Ehat - E)[p:p + tile])))}")
        bad = jnp.unravel_index(jnp.argmax(jnp.abs(Ehat - E)), E.shape)
        print(f"   worst at {bad}: Ehat={float(Ehat[bad])} E={float(E[bad])}")


if __name__ == "__main__":
    main()
