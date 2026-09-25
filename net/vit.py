"""Vision encoder — ViT-S class (~22M, patch 14), native path into the LLM.

K3 trains MoonViT-V2 from scratch with next-token prediction and a native
multimodal objective (:608-617, :766-770).  The skeleton uses a ViT-S-class
encoder (patch 14) whose patch features are projected into ``hidden`` and
interleaved with text tokens — no post-hoc alignment stage.

Note: the learned 2D positional embeddings are kept (standard ViT).  NoPE
applies to the language backbone (KDA/MLA), not to the vision encoder's spatial
layout — recorded in ``config.json`` ``deviations``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import ModelConfig
from .norm import rms_norm


def _rand(key, shape, scale: float) -> jnp.ndarray:
    return jax.random.normal(key, shape) * scale


class ViTParams(NamedTuple):
    patch_embed: jnp.ndarray  # (patch*patch*3, vit_hidden)
    pos_embed: jnp.ndarray  # (num_patches, vit_hidden)
    blocks: tuple  # per-block pytrees (attn + mlp + norms)
    projector: jnp.ndarray  # (vit_hidden, hidden)
    norm: jnp.ndarray  # (vit_hidden,) final norm


def _init_block(key, cfg: ModelConfig):
    h = cfg.vit_hidden
    mlp = cfg.vit_mlp
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    return {
        "qkv": _rand(k1, (h, 3 * h), 0.02),
        "out": _rand(k2, (h, h), 0.02),
        "fc1": _rand(k3, (h, mlp), 0.02),
        "fc2": _rand(k4, (mlp, h), 0.02),
        "norm1": jnp.ones((h,)),
        "norm2": jnp.ones((h,)),
    }


def _attention(blk, cfg: ModelConfig, x: jnp.ndarray) -> jnp.ndarray:
    h = cfg.vit_hidden
    H = cfg.vit_heads
    qkv = x @ blk["qkv"]  # (B, N, 3h)
    q, k, v = jnp.split(qkv, 3, axis=-1)
    q = q.reshape(*x.shape[:-1], H, h // H)
    k = k.reshape(*x.shape[:-1], H, h // H)
    v = v.reshape(*x.shape[:-1], H, h // H)
    scale = 1.0 / jnp.sqrt(h // H)
    scores = jnp.einsum("bnhd,bmhd->bhnm", q, k) * scale
    attn = jax.nn.softmax(scores, axis=-1)
    o = jnp.einsum("bhnm,bmhd->bnhd", attn, v).reshape(*x.shape[:-1], h)
    return o @ blk["out"]


def _block(blk, cfg: ModelConfig, x: jnp.ndarray) -> jnp.ndarray:
    x = x + _attention(blk, cfg, rms_norm(x, blk["norm1"]))
    z = rms_norm(x, blk["norm2"])
    x = x + jax.nn.gelu(z @ blk["fc1"]) @ blk["fc2"]
    return x


def init_vit(key, cfg: ModelConfig) -> ViTParams:
    p = cfg.vit_patch
    h = cfg.vit_hidden
    n = (cfg.image_size // p) ** 2
    keys = jax.random.split(key, cfg.vit_depth + 3)
    blocks = tuple(_init_block(keys[i], cfg) for i in range(cfg.vit_depth))
    return ViTParams(
        patch_embed=_rand(keys[-3], (p * p * 3, h), 0.02),
        pos_embed=_rand(keys[-2], (n, h), 0.02),
        blocks=blocks,
        projector=_rand(keys[-1], (h, cfg.hidden), 0.02),
        norm=jnp.ones((h,)),
    )


def apply(params: ViTParams, cfg: ModelConfig, images: jnp.ndarray) -> jnp.ndarray:
    """Encode ``(B, C, H, W)`` images into ``(B, num_patches, hidden)`` features."""
    B = images.shape[0]
    p = cfg.vit_patch
    Hp, Wp = cfg.image_size // p, cfg.image_size // p
    x = images.reshape(B, 3, Hp, p, Wp, p)
    x = x.transpose(0, 2, 4, 1, 3, 5).reshape(B, Hp * Wp, p * p * 3)
    x = x @ params.patch_embed + params.pos_embed  # (B, N, vit_hidden)
    for blk in params.blocks:
        x = _block(blk, cfg, x)
    x = rms_norm(x, params.norm)
    return x @ params.projector  # (B, N, hidden)
