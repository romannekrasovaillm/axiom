#!/usr/bin/env python3
"""Micro-experiment: multiplicity of stacked source-carry copies in the
differentiated scan's while states, across body/checkpoint variants.

Body (AttnRes-like): reads a data-dependent statistic of the *previous*
sources slot; writes a new slot per step.  This forces per-step carries to be
saved by the transposed scan.
"""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import re
import jax
import jax.numpy as jnp
import jax.random as jr


def make(slots=25, steps=5, b=2, t=64, h=32, variant="plain", policy=None,
         prevent_cse=True):
    key = jr.PRNGKey(0)
    xs = jr.normal(key, (steps, h, h))
    h0 = jr.normal(key, (b, t, h))
    src0 = jnp.zeros((slots, b, t, h))

    def inner(hh, xx):
        return jnp.tanh(hh @ xx) * 0.5 + hh

    if variant == "ckpt_inner":
        def core(hh, xx):
            return jax.checkpoint(inner, prevent_cse=prevent_cse)(hh, xx)

        def body(carry, x):
            h, src = carry
            mix = jnp.tanh(src[0])
            h2 = core(h, x) + 0.1 * mix
            src2 = src.at[3].set(h2 * 0.01)
            return (h2, src2), h2.sum()
    else:
        def body(carry, x):
            h, src = carry
            mix = jnp.tanh(src[0])  # data-dependent read of the previous slot
            h2 = jnp.tanh(h @ x) * 0.5 + h + 0.1 * mix
            src2 = src.at[3].set(h2 * 0.01)
            return (h2, src2), h2.sum()

    if variant == "ckpt_body":
        b2 = jax.checkpoint(body, prevent_cse=prevent_cse)
    else:
        b2 = body

    def f(xs):
        return jax.lax.scan(b2, (h0, src0), xs)

    return f, xs


def run(variant, prevent_cse=True):
    f, xs = make(variant=variant, prevent_cse=prevent_cse)

    def loss(x):
        (h, src), ys = f(x)
        return h.sum() + src.sum() + ys.sum()

    hlo = jax.jit(jax.grad(loss)).lower(xs).compile().as_text()
    out = os.path.join(os.path.dirname(__file__), "out",
                       f"micro2-{variant}-pcse{int(prevent_cse)}.hlo.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write(hlo)
    counts = []
    for line in hlo.split('\n'):
        if ' while(' in line and '= (' in line:
            state = line.split(' = (', 1)[1]
            n25 = len(re.findall(r'f32\[25,2,64,32\]', state))
            n5x25 = len(re.findall(r'f32\[5,25,2,64,32\]', state))
            n5h = len(re.findall(r'f32\[5,2,64,32\]', state))
            counts.append((n25, n5x25, n5h, line.strip().split(' = ')[0][:30]))
    print(f"variant={variant} prevent_cse={prevent_cse}: (src, src_stacked, h_stacked) per while:", counts)


if __name__ == "__main__":
    for v in sys.argv[1:] or ["plain"]:
        run(v)
