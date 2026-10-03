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

#: Per-layer remat policies of the backbone loop, declared in
#: ``net/config.json`` as ``grad_ckpt_policy`` (spine AD-9 / guard C-035 style:
#: the *file* is the switch, not an edit of this path).  ``none`` builds the
#: graph as before; ``per_layer`` wraps every backbone layer in
#: ``jax.checkpoint``, so the backward pass recomputes the layer from its inputs
#: (``h``, the candidate-pool carry and the previous layer deltas) instead of
#: keeping the layer's attention/MoE intermediates alive.  The coarse wrap of
#: the whole ``compute_loss`` in ``net/train_loop.py`` does **not** cut the
#: request (XLA then keeps the whole graph's intermediates on the recompute
#: boundary — D-8: 972 GiB at T=8192 even with ``--grad-checkpointing``), which
#: is why the boundary has to be per layer.
GRAD_CKPT_POLICIES = ("none", "per_layer")

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
    grad_ckpt_policy: str = "none",
    emit_logits: bool = True,
) -> jnp.ndarray | tuple:
    """Next-token logits for ``input_ids`` of shape (B, T).

    Returns ``(B, T, vocab)`` logits; ``return_hidden`` appends the backbone's
    final hidden state (used for MTP) and ``collect_qb`` appends the mean QB
    auxiliary loss over the LatentMoE layers (ours — see ``net/moe.py``).

    ``emit_logits=False`` skips the ``h @ embedding.T`` projection and returns
    only the requested extras — the chunked cross-entropy path of
    :func:`compute_loss` reads the hidden state and builds vocabulary-sized
    logits itself, one ``ce_chunk_tokens`` slice at a time, so the full
    ``(B, T, vocab)`` tensor is never materialised (D-8 remainder).  The
    default ``True`` keeps the projection exactly as before (bit-for-bit).

    ``grad_ckpt_policy`` (``none`` | ``per_layer``, see ``GRAD_CKPT_POLICIES``)
    selects the remat granularity of the backbone loop.  ``per_layer`` wraps
    each layer's transition in ``jax.checkpoint``; its live set is the layer
    input ``h``, the candidate-pool carry and the layer's own intermediates,
    so the backward pass recomputes the layer instead of retaining it.  With
    ``none`` the graph is built exactly as before (bit-for-bit).
    """
    if grad_ckpt_policy not in GRAD_CKPT_POLICIES:
        raise ValueError(
            f"неизвестная политика grad-checkpointing: {grad_ckpt_policy!r}; "
            f"ожидается одна из {GRAD_CKPT_POLICIES}"
        )
    remat = grad_ckpt_policy == "per_layer"
    emb = params.embedding[input_ids]  # (B, T, hidden)
    h = emb
    embed_src = emb
    layer_deltas: list[jnp.ndarray] = []
    qb_losses = []
    pool = None  # ADR-012 candidate pool, built by the first ``full`` MLA layer

    def _layer_step(
        block: BlockParams,
        attnres_w,
        h: jnp.ndarray,
        pool,
        prior_deltas: tuple,
        embed_src: jnp.ndarray,
        *,
        index: int,
        is_kda: bool,
        mode: str,
    ):
        """One backbone layer's transition — the remat unit.

        Takes the layer's parameters and the full live set (``h``, the pool
        carry, the previous deltas that AttnRes mixes) and returns the next
        ``h``, this layer's delta, its QB term and the updated pool.
        ``index``/``is_kda``/``mode`` are static and captured by the caller.
        """
        delta, qb, pool_out = _block_delta(
            block, is_kda, cfg, h, chunk_size, collect_qb, mode, pool
        )
        if use_attnres and index > 0:
            sources = jnp.stack([embed_src, *prior_deltas], axis=0)  # (N, B, T, hidden)
            corr = attnres_mod.apply_layer(attnres_w, sources)
        else:
            corr = 0.0
        return h + delta + corr, delta, qb, pool_out

    for i, block in enumerate(params.layers):
        is_kda = _layer_is_kda(i)
        mode = "full" if is_kda else mla_mod.layer_mode(cfg, _mla_ordinal(i))

        def step(
            block, attnres_w, h, pool, prior_deltas, embed_src,
            _i=i, _is_kda=is_kda, _mode=mode,
        ):
            return _layer_step(
                block, attnres_w, h, pool, prior_deltas, embed_src,
                index=_i, is_kda=_is_kda, mode=_mode,
            )

        if remat:
            step = jax.checkpoint(step)
        attnres_w = params.attnres.w[i] if use_attnres else None
        h, delta, qb, pool = step(
            block, attnres_w, h, pool, tuple(layer_deltas), embed_src
        )
        if qb is not None:
            qb_losses.append(qb)
        layer_deltas.append(delta)
    h = rms_norm(h, params.norm_final)
    out: list = []
    if emit_logits:
        out.append(h @ params.embedding.T)
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


def _chunked_cross_entropy(
    features: jnp.ndarray,
    targets: jnp.ndarray,
    embedding: jnp.ndarray,
    ce_chunk_tokens: int,
    loss_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Cross-entropy over vocabulary-sized logits built one T-slice at a time.

    ``features`` is (B, T', hidden) — the rows the tied output head is applied
    to (the backbone hidden state for NTP, the MTP output for the auxiliary
    head) — and ``embedding`` the (vocab, hidden) matrix, so a row's logits are
    ``f @ embedding.T``.  The naive :func:`_cross_entropy` materialises the
    whole (B, T', vocab) tensor; here it is built ``ce_chunk_tokens`` rows at a
    time and reduced as a global **sum** of per-token NLL over a global
    **count** of predicted tokens.  The global reduction is what keeps the
    value equal to the naive mean: a per-chunk mean of means would weight a
    short trailing chunk as much as a full one (the classic chunking bug).

    Each chunk's logits + NLL is wrapped in ``jax.checkpoint``, so the backward
    pass recomputes one chunk's ``(chunk, vocab)`` logits instead of retaining
    every chunk's; the live tensor is a single chunk's logits (~1 GiB at
    1024 x 262144 fp32) plus the (B, T', hidden) rows.

    ``loss_mask`` (B, T') excludes padding from numerator *and* denominator
    (its sum is the global count of non-pad tokens); ``None`` counts every row,
    matching the unmasked naive path.
    """
    _, total_rows = targets.shape
    if total_rows == 0 or ce_chunk_tokens <= 0:
        raise ValueError(
            "ce_chunk_tokens must be > 0 and targets non-empty for the chunked "
            f"path (got ce_chunk_tokens={ce_chunk_tokens!r}, rows={total_rows})"
        )
    n_chunks = -(-total_rows // ce_chunk_tokens)  # ceil

    def chunk_sum(f, t, emb, mask):
        logits = f @ emb.T
        logp = jax.nn.log_softmax(logits, axis=-1)
        nll = -jnp.take_along_axis(logp, t[..., None], axis=-1)[..., 0]  # (b, chunk)
        if mask is None:
            return jnp.sum(nll), jnp.array(nll.size, jnp.float32)
        m = mask.astype(nll.dtype)
        return jnp.sum(nll * m), jnp.sum(m)

    total_sum = jnp.zeros(())
    total_count = jnp.zeros(())
    for c in range(n_chunks):
        start = c * ce_chunk_tokens
        end = min(start + ce_chunk_tokens, total_rows)
        mask = None if loss_mask is None else loss_mask[:, start:end]
        part_sum, part_count = jax.checkpoint(chunk_sum)(
            features[:, start:end], targets[:, start:end], embedding, mask
        )
        total_sum = total_sum + part_sum
        total_count = total_count + part_count
    return total_sum / total_count


def mtp_loss(
    params: ModelParams,
    cfg: ModelConfig,
    hidden: jnp.ndarray,
    input_ids: jnp.ndarray,
    chunk_size: int,
    ce_chunk_tokens: int = 0,
) -> jnp.ndarray:
    """Auxiliary MTP loss (predict token t+2 from position t, DeepSeek pattern).

    ``hidden`` is the backbone final hidden state (B, T, hidden); ``input_ids``
    (B, T) supplies the shifted next-token embeddings.  Returns a scalar CE.

    ``ce_chunk_tokens > 0`` reduces the ``(B, T-2, vocab)`` head the same way
    as :func:`compute_loss` reduces the NTP head: the MTP projection stays full
    (its KDA recurrence is causal, so slicing its *input* T would change the
    result), only the vocabulary-sized output head is chunked.
    """
    next_emb = params.embedding[input_ids[:, 1:-1]]  # (B, T-2, hidden) — token at t+1
    h = hidden[:, :-2]  # (B, T-2, hidden)
    mtp_out = mtp_mod.apply(params.mtp, cfg, h, next_emb, chunk_size)  # (B, T-2, hidden)
    targets = input_ids[:, 2:]  # (B, T-2)
    if ce_chunk_tokens and ce_chunk_tokens > 0:
        return _chunked_cross_entropy(
            mtp_out, targets, params.embedding, ce_chunk_tokens
        )
    logits = mtp_out @ params.embedding.T  # (B, T-2, vocab)
    return _cross_entropy(logits, targets)


def loss_impl(cfg: ModelConfig) -> str:
    """Name of the cross-entropy implementation the config selects.

    ``chunked_ce`` when ``ce_chunk_tokens > 0`` (the memory-cut path),
    ``naive_ce`` otherwise (the whole-vocabulary mean).  Journaled by the run
    so a metrics file records which reduction produced the loss.
    """
    return "chunked_ce" if int(cfg.ce_chunk_tokens) > 0 else "naive_ce"


def compute_loss(
    params: ModelParams,
    cfg: ModelConfig,
    input_ids: jnp.ndarray,
    chunk_size: int = 64,
    use_attnres: bool = True,
    grad_ckpt_policy: str | None = None,
) -> jnp.ndarray:
    """Combined NTP + MTP auxiliary + QB load-balancing loss.

    The QB term (weight ``cfg.qb_weight``, mean over the LatentMoE layers) is
    ours: the paper's QB is a non-gradient bias update (Eq. 14) and defines no
    differentiable loss — see ``net/moe.py`` and ``config.json`` deviations.

    The backbone's remat granularity is read from ``cfg.grad_ckpt_policy``
    (declared in ``net/config.json``; pretrain ``l3-full`` carries
    ``per_layer``) unless ``grad_ckpt_policy`` overrides it for a call.  The
    per-layer checkpoint lives inside the layer loop of :func:`forward`,
    because a coarse wrap of this whole function does not reduce the
    activation request (D-8).

    The cross-entropy reduction is selected by ``cfg.ce_chunk_tokens`` (also
    declared in ``net/config.json``): ``0`` keeps the naive whole-vocabulary
    :func:`_cross_entropy`; a positive value routes both heads through
    :func:`_chunked_cross_entropy`, so the ``(B, T, vocab)`` logits are never
    materialised.  Both paths reduce to the same mean up to floating-point
    reduction order.
    """
    policy = cfg.grad_ckpt_policy if grad_ckpt_policy is None else grad_ckpt_policy
    ce_tokens = int(cfg.ce_chunk_tokens)
    if ce_tokens > 0:
        hidden, qb = forward(
            params, cfg, input_ids, chunk_size, use_attnres,
            return_hidden=True, collect_qb=True,
            grad_ckpt_policy=policy, emit_logits=False,
        )
        ntp = _chunked_cross_entropy(
            hidden[:, :-1], input_ids[:, 1:], params.embedding, ce_tokens
        )
    else:
        logits, hidden, qb = forward(
            params, cfg, input_ids, chunk_size, use_attnres,
            return_hidden=True, collect_qb=True,
            grad_ckpt_policy=policy,
        )
        ntp = _cross_entropy(logits[:, :-1], input_ids[:, 1:])
    aux = mtp_loss(params, cfg, hidden, input_ids, chunk_size, ce_chunk_tokens=ce_tokens)
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
