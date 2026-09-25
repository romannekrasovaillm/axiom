"""Model assembly — the L3 walking-skeleton network from scratch.

Composition (MODEL-L3-SKELETON.md section 1):
* tied embedding 160K x 1536
* 24 backbone layers, pattern [KDA, KDA, KDA, Gated-MLA] x 6 (18 KDA + 6 MLA)
* channel mixing: layer 0 — dense SiTU-GLU MLP, layers 1..23 — Stable LatentMoE
  (12 routed + 2 shared, top-2 dispatch, QB balancing, kda-formulas.md section 4)
* each layer: RMSNorm -> attention -> residual -> RMSNorm -> MLP/MoE -> residual
* AttnRes depth-mixing over prior layer deltas (additive, our decision)
* MTP (1 layer) with a shared (tied) output head
* ViT-S vision encoder (native path, projected into ``hidden``)
* NoPE: no positional embeddings anywhere in the backbone.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import ModelConfig, validate_config
from . import attnres as attnres_mod
from . import attn_sparse as attn_sparse_mod
from . import kda as kda_mod
from . import mla as mla_mod
from . import mlp as mlp_mod
from . import moe as moe_mod
from . import mtp as mtp_mod
from . import vit as vit_mod
from .norm import rms_norm


class BlockParams(NamedTuple):
    norm_attn: jnp.ndarray  # (hidden,)
    attn: kda_mod.KDAParams | mla_mod.MLAParams
    norm_mlp: jnp.ndarray  # (hidden,)
    mlp: mlp_mod.MLPParams | moe_mod.LatentMoEParams


class ModelParams(NamedTuple):
    embedding: jnp.ndarray  # (vocab, hidden)
    layers: tuple[BlockParams, ...]
    attnres: attnres_mod.AttnResParams
    norm_final: jnp.ndarray  # (hidden,)
    mtp: mtp_mod.MTPParams
    vit: vit_mod.ViTParams


def _layer_is_kda(index: int) -> bool:
    """KDA for the first three layers of each 4-layer group; MLA for the 4th."""
    return (index % 4) != 3


def _mla_ordinal(index: int) -> int:
    """Ordinal of an MLA layer among the MLA layers (they sit at ``index % 4 == 3``).

    Used to read the declared mode layout ``mla_layer_modes`` (ADR-012), which
    is indexed by MLA-layer order, not by backbone depth.
    """
    return index // 4


def _layer_is_dense(cfg: ModelConfig, index: int) -> bool:
    """The leading ``moe_dense_layers`` layers keep the dense SiTU-GLU MLP."""
    return index < cfg.moe_dense_layers


def dense_moe_split(cfg: ModelConfig) -> tuple[int, int]:
    """(dense, latent-MoE) layer counts — (1, 23) for the full skeleton."""
    return cfg.moe_dense_layers, cfg.num_layers - cfg.moe_dense_layers


def init_params(key, cfg: ModelConfig) -> ModelParams:
    validate_config(cfg)
    hid = cfg.hidden
    keys = jax.random.split(key, cfg.num_layers + 6)
    emb = jax.random.normal(keys[0], (cfg.vocab_size, hid)) * 0.02
    layers = []
    for i in range(cfg.num_layers):
        is_kda = _layer_is_kda(i)
        if is_kda:
            attn = kda_mod.init_kda(keys[i + 1], cfg)
        else:
            attn = mla_mod.init_mla(keys[i + 1], cfg)
        mlp = mlp_mod.init_mlp(keys[i + 1], cfg) if _layer_is_dense(cfg, i) else moe_mod.init_moe(keys[i + 1], cfg)
        layers.append(
            BlockParams(
                norm_attn=jnp.ones((hid,)),
                attn=attn,
                norm_mlp=jnp.ones((hid,)),
                mlp=mlp,
            )
        )
    return ModelParams(
        embedding=emb,
        layers=tuple(layers),
        attnres=attnres_mod.init_attnres(keys[-3], cfg),
        norm_final=jnp.ones((hid,)),
        mtp=mtp_mod.init_mtp(keys[-2], cfg),
        vit=vit_mod.init_vit(keys[-1], cfg),
    )


def _block_delta(
    block: BlockParams,
    is_kda: bool,
    cfg: ModelConfig,
    h: jnp.ndarray,
    chunk_size: int,
    collect_qb: bool = False,
    mode: str = "full",
    pool: attn_sparse_mod.CandidatePool | None = None,
):
    """The residual delta of one backbone layer (attention + MLP/MoE).

    Returns ``(delta, qb_loss, pool)``; ``qb_loss`` is None unless the layer is
    a LatentMoE block and ``collect_qb`` is set.  ``pool`` is the ADR-012
    candidate pool: an MLA layer in mode ``full`` replaces it, any other MLA
    layer consumes it unchanged, KDA layers pass it through untouched.
    """
    hn = rms_norm(h, block.norm_attn)
    if is_kda:
        attn_out = jax.vmap(lambda xb: kda_mod.apply_chunked(block.attn, cfg, xb, chunk_size))(hn)
    else:
        attn_out, pool = mla_mod.apply_with_pool(block.attn, cfg, hn, pool=pool, mode=mode)
    h1 = h + attn_out
    mlp_in = rms_norm(h1, block.norm_mlp)
    qb = None
    if isinstance(block.mlp, moe_mod.LatentMoEParams):
        if collect_qb:
            mlp_out, qb = moe_mod.apply(block.mlp, cfg, mlp_in, want_qb=True)
        else:
            mlp_out = moe_mod.apply(block.mlp, cfg, mlp_in)
    else:
        mlp_out = mlp_mod.apply(block.mlp, cfg, mlp_in)
    return attn_out + mlp_out, qb, pool


def forward(
    params: ModelParams,
    cfg: ModelConfig,
    input_ids: jnp.ndarray,
    chunk_size: int = 64,
    use_attnres: bool = True,
    return_hidden: bool = False,
    collect_qb: bool = False,
) -> jnp.ndarray | tuple:
    """Next-token logits for ``input_ids`` of shape (B, T).

    Returns ``(B, T, vocab)`` logits; ``return_hidden`` appends the backbone's
    final hidden state (used for MTP) and ``collect_qb`` appends the mean QB
    auxiliary loss over the LatentMoE layers (ours — see ``net/moe.py``).
    """
    emb = params.embedding[input_ids]  # (B, T, hidden)
    h = emb
    embed_src = emb
    layer_deltas = []
    qb_losses = []
    pool = None  # ADR-012 candidate pool, built by the first ``full`` MLA layer
    for i, block in enumerate(params.layers):
        is_kda = _layer_is_kda(i)
        mode = "full" if is_kda else mla_mod.layer_mode(cfg, _mla_ordinal(i))
        delta, qb, pool = _block_delta(block, is_kda, cfg, h, chunk_size, collect_qb, mode, pool)
        if qb is not None:
            qb_losses.append(qb)
        if use_attnres and i > 0:
            sources = jnp.stack([embed_src] + layer_deltas, axis=0)  # (N, B, T, hidden)
            corr = attnres_mod.apply_layer(params.attnres.w[i], sources)
        else:
            corr = 0.0
        h = h + delta + corr
        layer_deltas.append(delta)
    h = rms_norm(h, params.norm_final)
    logits = h @ params.embedding.T
    out: list = [logits]
    if return_hidden:
        out.append(h)
    if collect_qb:
        qb_mean = jnp.stack(qb_losses).mean() if qb_losses else jnp.zeros(())
        out.append(qb_mean)
    return out[0] if len(out) == 1 else tuple(out)


def _cross_entropy(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    logp = jax.nn.log_softmax(logits, axis=-1)
    nll = jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
    return -nll.mean()


def mtp_loss(
    params: ModelParams,
    cfg: ModelConfig,
    hidden: jnp.ndarray,
    input_ids: jnp.ndarray,
    chunk_size: int,
) -> jnp.ndarray:
    """Auxiliary MTP loss (predict token t+2 from position t, DeepSeek pattern).

    ``hidden`` is the backbone final hidden state (B, T, hidden); ``input_ids``
    (B, T) supplies the shifted next-token embeddings.  Returns a scalar CE.
    """
    next_emb = params.embedding[input_ids[:, 1:-1]]  # (B, T-2, hidden) — token at t+1
    h = hidden[:, :-2]  # (B, T-2, hidden)
    mtp_out = mtp_mod.apply(params.mtp, cfg, h, next_emb, chunk_size)  # (B, T-2, hidden)
    logits = mtp_out @ params.embedding.T  # (B, T-2, vocab)
    targets = input_ids[:, 2:]  # (B, T-2)
    return _cross_entropy(logits, targets)


def compute_loss(
    params: ModelParams,
    cfg: ModelConfig,
    input_ids: jnp.ndarray,
    chunk_size: int = 64,
    use_attnres: bool = True,
) -> jnp.ndarray:
    """Combined NTP + MTP auxiliary + QB load-balancing loss.

    The QB term (weight ``cfg.qb_weight``, mean over the LatentMoE layers) is
    ours: the paper's QB is a non-gradient bias update (Eq. 14) and defines no
    differentiable loss — see ``net/moe.py`` and ``config.json`` deviations.
    """
    logits, hidden, qb = forward(params, cfg, input_ids, chunk_size, use_attnres,
                                 return_hidden=True, collect_qb=True)
    ntp = _cross_entropy(logits[:, :-1], input_ids[:, 1:])
    aux = mtp_loss(params, cfg, hidden, input_ids, chunk_size)
    return ntp + cfg.mtp_loss_weight * aux + cfg.qb_weight * qb


def encode_vision(params: ModelParams, cfg: ModelConfig, images: jnp.ndarray) -> jnp.ndarray:
    """Encode ``(B, C, H, W)`` images into ``(B, P, hidden)`` features.

    The patch features are projected into the shared embedding space; the data
    pipeline interleaves them with text tokens (multiple images per sample are
    concatenated along the patch axis at that stage).
    """
    return vit_mod.apply(params.vit, cfg, images)


def param_count(cfg: ModelConfig) -> int:
    """Total parameter count from *shapes* only (no allocation).

    ``jax.eval_shape`` traces the initialiser so the full 160K-vocab embedding
    is never materialised.  The value is pinned in ``config.json`` and checked
    by the budget test.
    """
    key = jax.random.PRNGKey(0)
    shapes = jax.eval_shape(lambda k: init_params(k, cfg), key)
    return sum(
        math.prod(leaf.shape) for leaf in jax.tree_util.tree_leaves(shapes) if hasattr(leaf, "shape")
    )


def param_count_tree(cfg: ModelConfig) -> dict[str, int]:
    """Per-component parameter counts (embedding / layers / attnres / mtp / vit)."""
    key = jax.random.PRNGKey(0)
    shapes = jax.eval_shape(lambda k: init_params(k, cfg), key)

    def _size(tree):
        return sum(
            math.prod(x.shape) for x in jax.tree_util.tree_leaves(tree) if hasattr(x, "shape")
        )

    return {
        "embedding": _size(shapes.embedding),
        "layers": _size(shapes.layers),
        "attnres": _size(shapes.attnres),
        "mtp": _size(shapes.mtp),
        "vit": _size(shapes.vit),
    }


def active_param_count(cfg: ModelConfig) -> int:
    """Parameters activated per text token (MoE-style "activated params").

    Counts the backbone compute path: all attention, the dense layer-0 MLP,
    and for each LatentMoE layer the latent projections, router, the top-k
    routed experts actually dispatched to, and all shared experts — plus norms
    and AttnRes.  Excludes the tied embedding (a lookup, not per-token compute),
    the ViT (active on image tokens only) and the MTP auxiliary head
    (training-only, dropped at inference).  Pinned in ``config.json`` as
    ``active_params_per_token`` and checked by test_01/test_11.
    """
    key = jax.random.PRNGKey(0)
    shapes = jax.eval_shape(lambda k: init_params(k, cfg), key)

    def _size(tree):
        return sum(
            math.prod(x.shape) for x in jax.tree_util.tree_leaves(tree) if hasattr(x, "shape")
        )

    total = _size(shapes.attnres) + _size(shapes.norm_final)
    for i, block in enumerate(shapes.layers):
        total += _size(block.norm_attn) + _size(block.attn) + _size(block.norm_mlp)
        if _layer_is_dense(cfg, i):
            total += _size(block.mlp)
            continue
        inactive = 0
        for name in ("expert_g", "expert_u", "expert_d"):
            leaf = getattr(block.mlp, name)
            n_routed = leaf.shape[0]
            inactive += (n_routed - cfg.moe_top_k) * math.prod(leaf.shape[1:])
        total += _size(block.mlp) - inactive
    return total
