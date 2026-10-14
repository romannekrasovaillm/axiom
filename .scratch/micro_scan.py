#!/usr/bin/env python3
"""Micro-experiment: how many stacked copies of a big scan carry does the
transposed scan (jax 0.11.2) keep in its loop state, and does an inner
jax.checkpoint change that?

Pattern mirrors the model's group-scan: carry = (h, sources), sources grows by
one slot per step; body computes h from h and per-step params.
"""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import re
import jax
import jax.numpy as jnp
import jax.random as jr


def make_case(variant: str, slots=25, steps=5, b=2, t=64, h=32):
    key = jr.PRNGKey(0)
    xs = jr.normal(key, (steps, h, h))
    h0 = jr.normal(key, (b, t, h))
    src0 = jnp.zeros((slots, b, t, h))

    def body(carry, x):
        h, src = carry
        if variant == "plain":
            h2 = jnp.tanh(h @ x) + h
        elif variant == "ckpt_inner":
            h2 = jax.checkpoint(lambda hh: jnp.tanh(hh @ x) * 0.5 + hh)(h)
        elif variant == "ckpt_body":
            h2 = jax.checkpoint(lambda hh: jnp.tanh(hh @ x) * 0.5 + hh)(h)
        else:
            raise ValueError(variant)
        src2 = src.at[0].set(h2 * 0.01)
        return (h2, src2), h2.sum()

    def f(xs):
        if variant == "ckpt_body":
            def b2(carry, x):
                return body(carry, x)
            return jax.lax.scan(jax.checkpoint(b2), (h0, src0), xs)
        return jax.lax.scan(body, (h0, src0), xs)

    return f, xs


def run(variant):
    f, xs = make_case(variant)

    def loss(x):
        (h, src), ys = f(x)
        return h.sum() + src.sum() + ys.sum()

    g = jax.jit(jax.grad(loss))(xs)
    hlo = jax.jit(jax.grad(loss)).lower(xs).compile().as_text()
    out = os.path.join(os.path.dirname(__file__), "out", f"micro-{variant}.hlo.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write(hlo)
    # count whiles with stacked copies of sources
    counts = []
    for line in hlo.split('\n'):
        if ' while(' in line and '= (' in line:
            state = line.split(' = (', 1)[1]
            n_src = len(re.findall(r'f32\[(?:5|25),2,64,32\]', state))
            n_src5 = len(re.findall(r'f32\[5,25,2,64,32\]', state))
            counts.append((n_src + n_src5, line.strip().split(' = ')[0][:40]))
    print(f"variant={variant}: whiles with sources-stacks:", counts)


if __name__ == "__main__":
    for v in sys.argv[1:] or ["plain", "ckpt_inner"]:
        run(v)
