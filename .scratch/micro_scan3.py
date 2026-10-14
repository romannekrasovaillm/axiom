#!/usr/bin/env python3
"""Micro: per-sub-layer checkpoints inside the scan body (model-like), and the
effect of prevent_cse on the stacked-carry copies in the differentiated scan.

Structure: body processes 2 "sub-layers"; each sub-layer is checkpointed
(saving its sources input, like the model's per-layer boundaries); the carry
(sources) is updated per sub-layer (.at[slot].set).
"""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import re
import jax
import jax.numpy as jnp
import jax.random as jr


def run(prevent_cse, slots=25, steps=5, b=2, t=64, h=32, mode="sub"):
    key = jr.PRNGKey(0)
    xs = jr.normal(key, (steps, 4, h, h))
    h0 = jr.normal(key, (b, t, h))
    src0 = jnp.zeros((slots, b, t, h))

    def sub_fn(hh, src, xx, s):
        mix = jnp.tanh(src[0]) * 0.1
        h2 = jnp.tanh(hh @ xx) * 0.5 + hh + mix
        src2 = src.at[s].set(h2 * 0.01)
        return h2, src2

    def body(carry, x):
        h, src = carry
        if mode == "sub":
            for s in range(4):
                f = lambda hh, ss, xx, _s=s: sub_fn(hh, ss, xx, _s)
                f = jax.checkpoint(f, prevent_cse=prevent_cse)
                h, src = f(h, src, x[s])
        else:
            for s in range(4):
                h, src = sub_fn(h, src, x[s], s)
        return (h, src), h.sum()

    if mode == "unit":
        body = jax.checkpoint(body, prevent_cse=prevent_cse)

    def loss(x):
        (h, src), ys = jax.lax.scan(body, (h0, src0), x)
        return h.sum() + src.sum() + ys.sum()

    hlo = jax.jit(jax.grad(loss)).lower(xs).compile().as_text()
    out = os.path.join(os.path.dirname(__file__), "out",
                       f"micro3-{mode}-pcse{int(prevent_cse)}.hlo.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write(hlo)
    rows = []
    for line in hlo.split('\n'):
        if ' while(' in line and '= (' in line:
            state = line.split(' = (', 1)[1]
            n_src = len(re.findall(r'f32\[25,2,64,32\]', state))
            n_5x25 = len(re.findall(r'f32\[5,25,2,64,32\]', state))
            n_5h = len(re.findall(r'f32\[5,2,64,32\]', state))
            tot = _state_bytes(state)
            rows.append((n_src, n_5x25, n_5h, tot, line.strip().split(' = ')[0][:24]))
    print(f"mode={mode} prevent_cse={prevent_cse}:", [(f"src={a}", f"stack25={b}", f"stackh={c}", f"total={d/2**20:.1f}MiB", e) for a, b, c, d, e in rows])


def _state_bytes(state):
    tot = 0
    for dt, dims in re.findall(r'(f32|bf16|s32)\[([0-9,]*)\]', state):
        n = 1
        for d in dims.split(','):
            if d:
                n *= int(d)
        tot += n * {'f32': 4, 'bf16': 2, 's32': 4}[dt]
    return tot


if __name__ == "__main__":
    run(True, mode="sub")
    run(True, mode="none")
    run(True, mode="unit")
