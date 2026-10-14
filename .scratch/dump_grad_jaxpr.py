#!/usr/bin/env python3
"""Dump the gradient jaxpr of the small model to inspect the backward structure."""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("AXIOM_KDA_CC_IMPL", "jax")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import jax.random as jr

from net.config import ModelConfig
import net.model as model


def build_cfg(**over):
    base = dict(
        vocab_size=128, hidden=32, num_layers=8, num_kda_layers=6, num_mla_layers=2,
        num_heads=2, head_dim=16, kda_dk=16, kda_dv=16, kda_decay_rank=16,
        kda_short_conv_kernel=4, kda_g_min=-5.0, mla_latent_dim=16, mla_head_dim=16,
        mlp_intermediate=64, siti_beta_gate=4.0, siti_beta_up=25.0,
        mtp_layers=1, mtp_loss_weight=0.1, attnres_blocks=1, attnres_block_size=4,
        vit_patch=14, vit_hidden=32, vit_depth=2, vit_heads=2, vit_mlp=64,
        image_size=56, moe_dense_layers=1, moe_latent_dim=16, moe_num_routed=4,
        moe_num_shared=1, moe_top_k=2, moe_expert_intermediate=8,
        moe_shared_intermediate=16, qb_weight=0.01, routing_seed=0,
        grad_ckpt_policy="per_layer", remat_policy="none",
        ce_chunk_tokens=64, kda_chunked_backward=True, kda_impl="chunked_cc",
        kda_wyut_chunk=16, scan_layers=True, attn_dense_reference=True,
    )
    base.update(over)
    return ModelConfig(**base)


def main():
    cfg = build_cfg()
    key = jr.PRNGKey(0)
    params = model.init_params(key, cfg)
    ids = jr.randint(key, (2, 128), 0, cfg.vocab_size)

    def loss(p):
        return model.compute_loss(p, cfg, ids, chunk_size=16)

    closed = jax.make_jaxpr(jax.grad(loss))(params)
    text = str(closed)
    out = os.path.join(os.path.dirname(__file__), "out", "small-grad.jaxpr.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(text)
    print("jaxpr chars:", len(text), "->", out)
    # summary of top-level eqns
    eqns = closed.jaxpr.eqns
    n_scan = sum(1 for e in eqns if e.primitive.name == "scan")
    n_remat = sum(1 for e in eqns if str(e.primitive.name).startswith("remat"))
    n_check = sum(1 for e in eqns if "checkpoint" in str(e.primitive.name))
    print(f"top-level eqns={len(eqns)} scans={n_scan} remat={n_remat} check={n_check}")
    prim_names = {}
    for e in eqns:
        prim_names[e.primitive.name] = prim_names.get(e.primitive.name, 0) + 1
    print("top prims:", sorted(prim_names.items(), key=lambda kv: -kv[1])[:15])


if __name__ == "__main__":
    main()
