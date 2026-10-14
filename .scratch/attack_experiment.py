#!/usr/bin/env python3
"""Attack experiment: does an inner remat boundary around the MLA attention
core / a query-chunked attention core reduce the differentiated-scan peak?

Runs the same small config as variant_matrix and monkeypatches
``net.mla._dense_apply`` per variant (no repository edits).
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
import net.mla as mla
import net.compute_dtype as compute_dtype


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


_ORIG_DENSE = mla._dense_apply


def core_ckpt(params, cfg, x, mask=None):
    return jax.checkpoint(lambda p, xx, mm: _ORIG_DENSE(p, cfg, xx, mm))(params, x, mask)


def chunked_dense(chunk):
    """Query-chunked dense attention: no (B,H,T,T) tensor ever materialises.

    Exact causal-softmax math: queries in a chunk attend only to keys up to the
    chunk's end (the causal prefix); everything after is exactly zero in the
    masked softmax and its contribution to ``attn @ v`` is exactly zero.
    """

    def apply(params, cfg, x, mask=None):
        H, dq = cfg.num_heads, cfg.mla_head_dim
        x32 = x.astype(jnp.float32)
        c = compute_dtype.gemm(x32, params.W_c.astype(jnp.float32))
        q, k, v = mla._project(params, cfg, x32, c)
        T = x.shape[-2]
        scale = 1.0 / jnp.sqrt(dq)
        if mask is None:
            mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))

        def part(qc, mc, s_end):
            sc = compute_dtype.gemm_einsum("bthd,bshd->bhts", qc, k[:, :s_end]) * scale
            m = mc[:, :s_end]
            sc = jnp.where(m, sc, jnp.finfo(jnp.float32).min)
            at = jax.nn.softmax(sc, axis=-1)
            return compute_dtype.gemm_einsum("bhts,bshd->bthd", at, v[:, :s_end])

        outs = []
        for s in range(0, T, chunk):
            e = min(s + chunk, T)
            outs.append(jax.checkpoint(part)(q[:, s:e], mask[s:e], e))
        o = jnp.concatenate(outs, axis=-2).reshape(*x.shape[:-1], H * dq)
        gate = jax.nn.sigmoid(compute_dtype.gemm(x32, params.W_g.astype(jnp.float32)))
        return compute_dtype.gemm(gate * o, params.W_o.astype(jnp.float32))

    return apply


def run(name, patch):
    mla._dense_apply = patch
    cfg = build_cfg()
    key = jr.PRNGKey(0)
    params = model.init_params(key, cfg)
    ids = jr.randint(key, (2, 512), 0, cfg.vocab_size)

    def loss(p):
        return model.compute_loss(p, cfg, ids, chunk_size=16)

    lowered = jax.jit(jax.value_and_grad(loss)).lower(params)
    compiled = lowered.compile()
    ma = compiled.memory_analysis()
    exe = compiled.as_text()
    out = os.path.join(os.path.dirname(__file__), "out", f"attack-{name}.exe.hlo.txt")
    open(out, "w").write(exe)
    n_scores = len(re.findall(r'f32\[2,4,512,512\]', exe))
    n_small_scores = len(re.findall(r'f32\[2,4,(?:128|256),512\]', exe))
    temp = ma.temp_size_in_bytes / 2**20 if ma else float('nan')
    print(f"{name:16s} temp={temp:8.2f} MiB  (2,4,512,512)_defs={n_scores:4d}  chunked_score_defs={n_small_scores}")
    sys.stdout.flush()
    mla._dense_apply = _ORIG_DENSE
    return temp


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    run("baseline", _ORIG_DENSE)
    if which in ("all", "core"):
        run("core_ckpt", core_ckpt)
    if which in ("all", "chunk128"):
        run("chunk128", chunked_dense(128))
    if which in ("all", "chunk256"):
        run("chunk256", chunked_dense(256))
