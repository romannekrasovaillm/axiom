"""MTP — multi-token prediction layer (kda-formulas.md / MODEL-L3-SKELETON.md 2.6).

Mirrors the structure of a backbone block (KDA attention + SiTU-GLU MLP) and
predicts the next token from a fusion of the backbone hidden state and the
next-token embedding, following the DeepSeek pattern (shared output head =
tied embedding).  The auxiliary MTP loss is computed in ``model.mtp_loss``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import ModelConfig
from . import kda as kda_mod
from . import mlp as mlp_mod
from .norm import rms_norm


def _rand(key, shape, scale: float) -> jnp.ndarray:
    return jax.random.normal(key, shape) * scale


class MTPParams(NamedTuple):
    W_f: jnp.ndarray  # (2*hidden, hidden) feature-fusion projection
    norm_in: jnp.ndarray  # (hidden,)
    kda: kda_mod.KDAParams
    norm_attn: jnp.ndarray  # (hidden,)
    mlp: mlp_mod.MLPParams
    norm_mlp: jnp.ndarray  # (hidden,)
    norm_out: jnp.ndarray  # (hidden,)


def init_mtp(key, cfg: ModelConfig) -> MTPParams:
    hid = cfg.hidden
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    return MTPParams(
        W_f=_rand(k1, (2 * hid, hid), 0.02),
        norm_in=jnp.ones((hid,)),
        kda=kda_mod.init_kda(k2, cfg),
        norm_attn=jnp.ones((hid,)),
        mlp=mlp_mod.init_mlp(k3, cfg),
        norm_mlp=jnp.ones((hid,)),
        norm_out=jnp.ones((hid,)),
    )


def apply(params: MTPParams, cfg: ModelConfig, h: jnp.ndarray, next_emb: jnp.ndarray, chunk_size: int) -> jnp.ndarray:
    """MTP block over ``(B, T, hidden)`` backbone hidden states.

    ``h`` is the backbone's last-layer hidden state; ``next_emb`` the embedding
    of the next token.  Returns ``(B, T, hidden)`` before the shared head.
    """
    B, T, hid = h.shape
    fused = jnp.concatenate([h, next_emb], axis=-1) @ params.W_f  # (B, T, hid)
    z = rms_norm(fused, params.norm_in)

    # KDA attention (chunked) vmap'd over the batch.
    attn = jax.vmap(lambda xb: kda_mod.apply_chunked(params.kda, cfg, xb, chunk_size))(z)
    z = z + attn
    z = z + mlp_mod.apply(params.mlp, cfg, rms_norm(z, params.norm_mlp))
    return rms_norm(z, params.norm_out)
