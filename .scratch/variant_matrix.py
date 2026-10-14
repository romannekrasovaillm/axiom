#!/usr/bin/env python3
"""Variant matrix: what drives the memory structure of the differentiated
group-scan at small scale.

For each variant: compile value_and_grad(compute_loss) on CPU, report XLA's
temp request, and count the stacked copies of the sources carry in the loop
states of the executable HLO.
"""
import os
import re
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
    base.update(over)
    return ModelConfig(**base)


VARIANTS = {
    "baseline": dict(),
    "no_attnres": dict(_use_attnres=False),
    "no_ckpt": dict(grad_ckpt_policy="none"),
    "unrolled": dict(scan_layers=False),
    "no_ce_chunk": dict(ce_chunk_tokens=0),
    "remat_dots": dict(remat_policy="dots_with_no_batch_dims_saveable"),
}


def run(name, over):
    over = dict(over)
    use_attnres = over.pop("_use_attnres", True)
    cfg = build_cfg(**over)
    key = jr.PRNGKey(0)
    params = model.init_params(key, cfg)
    ids = jr.randint(key, (2, 512), 0, cfg.vocab_size)

    def loss(p):
        return model.compute_loss(p, cfg, ids, chunk_size=16, use_attnres=use_attnres)

    lowered = jax.jit(jax.value_and_grad(loss)).lower(params)
    compiled = lowered.compile()
    ma = compiled.memory_analysis()
    exe = compiled.as_text()
    out = os.path.join(os.path.dirname(__file__), "out", f"var-{name}.exe.hlo.txt")
    open(out, "w").write(exe)
    # stacked sources = (steps, 13, B, T, h) shapes
    cnt_stack = len(re.findall(r'f32\[2,13,2,512,64\]', exe))
    cnt_src = len(re.findall(r'f32\[13,2,512,64\]', exe))
    state_copies = 0
    for line in exe.split('\n'):
        if ' while(' in line and '= (' in line:
            state = line.split(' = (', 1)[1]
            state_copies = max(state_copies, state.count('f32[2,13,2,512,64]'))
    temp = ma.temp_size_in_bytes / 2**20 if ma else float('nan')
    print(f"{name:12s} temp={temp:8.2f} MiB  stacked_src_defs={cnt_stack:4d}  src_defs={cnt_src:4d}  max_stack_in_state={state_copies}")
    sys.stdout.flush()
    return temp


if __name__ == "__main__":
    only = sys.argv[1:] or list(VARIANTS)
    for name in only:
        try:
            run(name, VARIANTS[name])
        except Exception as exc:  # noqa: BLE001
            print(f"{name:12s} FAILED: {type(exc).__name__}: {exc}")
