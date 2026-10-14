#!/usr/bin/env python3
"""Inspect the transposed group-scan's carry/xs structure in the grad jaxpr."""
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


def main():
    cfg = build_cfg()
    key = jr.PRNGKey(0)
    params = model.init_params(key, cfg)
    ids = jr.randint(key, (2, 128), 0, cfg.vocab_size)

    def loss(p):
        return model.compute_loss(p, cfg, ids, chunk_size=16)

    grad = jax.make_jaxpr(jax.grad(loss))(params)
    for i, e in enumerate(grad.jaxpr.eqns):
        if e.primitive.name != "scan":
            continue
        body = e.params["jaxpr"]
        ft_in, ft_out = e.params["ft_in"], e.params["ft_out"]
        length = e.params["length"]
        n_consts = len(ft_in.consts)
        n_carry = len(ft_in.carry)
        print(f"== scan idx {i}: length={length} consts={n_consts} carry={n_carry} "
              f"xs={ft_in.num_vars - n_consts - n_carry} body_eqns={len(body.jaxpr.eqns)}")
        print("   invars:", [str(v.aval) for v in body.jaxpr.invars][:6], "...")
        print("   carry avals:", [str(v.aval) for v in ft_in.carry])
        xs_avals = [str(v.aval) for v in body.jaxpr.invars[n_consts + n_carry:]]
        print("   xs count:", len(xs_avals))
        for j, a in enumerate(xs_avals[:24]):
            print(f"      xs[{j}]: {a}")
        print("   outs:", [str(v.aval) for v in body.jaxpr.outvars][:12])


if __name__ == "__main__":
    main()
