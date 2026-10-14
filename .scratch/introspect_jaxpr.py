#!/usr/bin/env python3
"""Structural introspection of the primal and gradient jaxprs (small model)."""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("AXIOM_KDA_CC_IMPL", "jax")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import jax.random as jr

from dump_grad_jaxpr import build_cfg
import net.model as model


def describe(e, indent="  "):
    prim = e.primitive.name
    outs = ", ".join(str(v.aval) for v in e.outvars)
    line = f"{indent}{prim}: {outs}"
    if prim == "scan":
        # scan body jaxpr
        (body,) = [p for p in e.params.values() if hasattr(p, "jaxpr")]
        line += f" [body eqns={len(body.jaxpr.eqns)}]"
    return line


def main():
    cfg = build_cfg()
    key = jr.PRNGKey(0)
    params = model.init_params(key, cfg)
    ids = jr.randint(key, (2, 128), 0, cfg.vocab_size)

    def loss(p):
        return model.compute_loss(p, cfg, ids, chunk_size=16)

    print("======== PRIMAL (loss) top-level ========")
    primal = jax.make_jaxpr(loss)(params)
    counts = {}
    for e in primal.jaxpr.eqns:
        counts[e.primitive.name] = counts.get(e.primitive.name, 0) + 1
    print("primal top eqns:", len(primal.jaxpr.eqns), sorted(counts.items(), key=lambda kv: -kv[1])[:10])
    for i, e in enumerate(primal.jaxpr.eqns):
        if e.primitive.name in ("scan", "remat2", "custom_vjp_call_jaxpr"):
            print(describe(e), "  idx", i)
    print()
    print("======== GRADIENT top-level ========")
    grad = jax.make_jaxpr(jax.grad(loss))(params)
    counts = {}
    for e in grad.jaxpr.eqns:
        counts[e.primitive.name] = counts.get(e.primitive.name, 0) + 1
    print("grad top eqns:", len(grad.jaxpr.eqns), sorted(counts.items(), key=lambda kv: -kv[1])[:10])
    for i, e in enumerate(grad.jaxpr.eqns):
        if e.primitive.name in ("scan", "remat2", "custom_vjp_call_jaxpr"):
            print(describe(e), "  idx", i)
    # scan details in grad
    for e in grad.jaxpr.eqns:
        if e.primitive.name == "scan":
            print("---- scan params keys:", list(e.params.keys())[:20])
            break


if __name__ == "__main__":
    main()
