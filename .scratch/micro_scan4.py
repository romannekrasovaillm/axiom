#!/usr/bin/env python3
"""Micro: per-sub-layer checkpoints whose big carry input is the *unit's*
sources (shared, not mid-unit), with the mid-unit view rebuilt inside the
boundary from the per-sub-layer deltas.  Does the differentiated scan's state
collapse from 4 stacked sources copies to 1?
"""
import os
import re
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import jax.random as jr


def state_bytes(state):
    tot = 0
    for dt, dims in re.findall(r'(f32|bf16|s32)\[([0-9,]*)\]', state):
        n = 1
        for d in dims.split(','):
            if d:
                n *= int(d)
        tot += n * {'f32': 4, 'bf16': 2, 's32': 4}[dt]
    return tot


def run(mode, slots=25, steps=5, b=2, t=64, h=32):
    key = jr.PRNGKey(0)
    xs = jr.normal(key, (steps, 4, h, h))
    h0 = jr.normal(key, (b, t, h))
    src0 = jnp.zeros((slots, b, t, h))

    def body(carry, x):
        h, src_in = carry
        deltas = []

        def make_sub(s):
            def sub(hh, src0, prec, xx):
                src_s = src0
                for k, dd in enumerate(prec):
                    src_s = src_s.at[k].set(dd)
                mix = jnp.tanh(src_s[0]) * 0.1
                h2 = jnp.tanh(hh @ xx) * 0.5 + hh + mix
                return h2, h2 * 0.01
            return sub

        for s in range(4):
            f = jax.checkpoint(make_sub(s))
            h, d = f(h, src_in, tuple(deltas), x[s])
            deltas.append(d)
        src_out = src_in
        for k, dd in enumerate(deltas):
            src_out = src_out.at[k].set(dd)
        return (h, src_out), h.sum()

    def loss(x):
        (h, src), ys = jax.lax.scan(body, (h0, src0), x)
        return h.sum() + src.sum() + ys.sum()

    hlo = jax.jit(jax.grad(loss)).lower(xs).compile().as_text()
    out = os.path.join(os.path.dirname(__file__), "out", f"micro4-{mode}.hlo.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write(hlo)
    rows = []
    for line in hlo.split('\n'):
        if ' while(' in line and '= (' in line:
            state = line.split(' = (', 1)[1]
            n_src = len(re.findall(r'f32\[25,2,64,32\]', state))
            n_5x25 = len(re.findall(r'f32\[5,25,2,64,32\]', state))
            n_5h = len(re.findall(r'f32\[5,2,64,32\]', state))
            rows.append((f"src={n_src}", f"stack25={n_5x25}", f"stackh={n_5h}",
                         f"total={state_bytes(state)/2**20:.2f}MiB",
                         line.strip().split(' = ')[0][:24]))
    print(f"mode={mode}:", rows)


if __name__ == "__main__":
    run("rebuild")
