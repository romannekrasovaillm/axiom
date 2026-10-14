#!/usr/bin/env python3
"""Small-scale reproduction of the B=4 group-scan backward memory structure.

Traces the l3-full *structure* on a small geometry: group-scan + per-layer
checkpoint + dense MLA oracle.  Lowers the value_and_grad of the loss and
reports XLA's memory analysis plus a shape census of the HLO.
"""
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


def build_cfg():
    cfg = ModelConfig(
        vocab_size=512, hidden=64, num_layers=12, num_kda_layers=9, num_mla_layers=3,
        num_heads=4, head_dim=16, kda_dk=16, kda_dv=16, kda_decay_rank=16,
        kda_short_conv_kernel=4, kda_g_min=-5.0, mla_latent_dim=32, mla_head_dim=16,
        mlp_intermediate=128, siti_beta_gate=4.0, siti_beta_up=25.0,
        mtp_layers=1, mtp_loss_weight=0.1, attnres_blocks=1, attnres_block_size=4,
        vit_patch=14, vit_hidden=32, vit_depth=2, vit_heads=2, vit_mlp=64,
        image_size=56, moe_dense_layers=1, moe_latent_dim=32, moe_num_routed=6,
        moe_num_shared=2, moe_top_k=2, moe_expert_intermediate=16,
        moe_shared_intermediate=32, qb_weight=0.01, routing_seed=0,
        grad_ckpt_policy="per_layer", remat_policy="none",
        ce_chunk_tokens=256, kda_chunked_backward=True, kda_impl="chunked_cc",
        kda_wyut_chunk=16, scan_layers=True, attn_dense_reference=True,
    )
    return cfg


def main(mode="current"):
    cfg = build_cfg()
    key = jr.PRNGKey(0)
    params = model.init_params(key, cfg)
    ids = jr.randint(key, (2, 512), 0, cfg.vocab_size)

    def loss(p):
        return model.compute_loss(p, cfg, ids, chunk_size=16)

    if mode == "fwd":
        g = jax.jit(loss)
    else:
        g = jax.jit(jax.value_and_grad(loss))
    lowered = g.lower(params)
    hlo = lowered.as_text(dialect="hlo")
    out = os.path.join(os.path.dirname(__file__), "out", f"small-{mode}.hlo.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(hlo)
    compiled = lowered.compile()
    exe = compiled.as_text()
    out_exe = os.path.join(os.path.dirname(__file__), "out", f"small-{mode}.exe.hlo.txt")
    with open(out_exe, "w") as f:
        f.write(exe)
    print("executable HLO bytes:", len(exe), "->", out_exe)
    ma = compiled.memory_analysis()
    print("HLO bytes:", len(hlo), "->", out)
    if ma is not None:
        print("memory_analysis:",
              "argument", ma.argument_size_in_bytes / 2**20,
              "output", ma.output_size_in_bytes / 2**20,
              "temp", ma.temp_size_in_bytes / 2**20,
              "alias", getattr(ma, "alias_size_in_bytes", 0) / 2**20,
              "peak (MiB)", (ma.temp_size_in_bytes + ma.argument_size_in_bytes
                             + ma.output_size_in_bytes) / 2**20)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "current")
