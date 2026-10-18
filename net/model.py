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

Diagnostic composition (VERIFICATION-LEG): a config may declare a leading
``dense-standard`` prefix (``layer_composition["dense-standard"]`` →
``cfg.dense_standard_layers``) — a standard block (dense attention through the
existing oracle of ``net/mla.py:_dense_apply`` + a dense SiTU-GLU MLP) that
separates pipeline bugs from the price of the KDA/MLA architecture.  The
production composition is the default and is not affected: with a zero-length
prefix :func:`layer_kind` reproduces the pinned ``[K,K,K,M]`` pattern exactly.
"""

from __future__ import annotations

import logging
import math
import os
from typing import NamedTuple

import jax
import jax.numpy as jnp

#: Journal line the group-scan emits when a declared switch cannot be honoured
#: (a single group, or a pool layout the scan cannot carry).  The fallback is
#: the unrolled path, so the graph stays correct — the line records *why* the
#: host-compile saving did not materialise.
_log = logging.getLogger(__name__)

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
#:
#: ``per_layer_block`` (ADR-054 Am.18, milestone-8) narrows the boundary scope from
#: the whole layer *transition* to the KDA/MLA+MoE *block* (the AttnRes carry
#: itself stays outside).  Rationale — the group-scan carry: under
#: ``per_layer`` every sub-layer boundary saves the whole ``sources`` buffer
#: (``num_layers + 1`` slots of ``(B, T, hidden)``, built for AttnRes) as one
#: of its inputs, and the differentiated ``lax.scan`` collects those
#: per-boundary saves into a stacked array *per sub-layer position*: 4
#: stacked copies of ``(units, L+1, B, T, hidden)`` at the pinned ``[K,K,K,M]``
#: layout (measured in the campaign HLO dump, ~47 GiB of the ~109 GiB B=4
#: floor).  The boundary exists to bound the (B, H, T, T) score transients,
#: and those stay inside the block — the AttnRes carry does not belong to it.
#:
#: Just moving the block-only boundary in and reading the carry outside it
#: does **not** cut the request: the masked read's softmax residuals (its
#: input plus the per-slot statistics) are then saved per sub-layer position,
#: 3 stacked copies per position = 12 in the measured loop state — worse than
#: ``per_layer``'s 4.  The boundary therefore also covers the AttnRes read,
#: but over a *rebuilt view*: its big input is the **unit-entry** carry — the
#: same tensor for all four sub-layer boundaries of a unit, hence a single
#: stacked save — and the mid-unit view is re-applied inside the boundary
#: from the unit's deltas (``.at`` data movement only, slot-for-slot the
#: carried buffer's values).  The correction add and the carry update stay
#: outside.  Measured on the scanned small-model reproduction (HLO loop
#: state, pinned by ``net/tests/test_37_per_layer_block.py``): 4 stacked
#: copies under ``per_layer`` → **1** under ``per_layer_block``.  The
#: arithmetic is unchanged (same values, same order at every arithmetic op)
#: — only *where* the backward recomputes.
GRAD_CKPT_POLICIES = ("none", "per_layer", "per_layer_block")

from .config import ModelConfig, validate_config
from .remat import remat_checkpoint, validate_remat_policy
from . import attnres as attnres_mod
from . import attn_sparse as attn_sparse_mod
from . import compute_dtype
from . import kda as kda_mod
from . import mla as mla_mod
from . import mlp as mlp_mod
from . import moe as moe_mod
from . import mtp as mtp_mod
from . import provenance
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


#: Layer kinds the backbone can build.  ``kda``/``mla`` are the pinned
#: ``[K,K,K,M]`` production pattern; ``dense-standard`` is the diagnostic block
#: of VERIFICATION-LEG (see :func:`layer_kind`).
LAYER_KINDS = ("kda", "mla", "dense-standard")


def _layer_is_kda(index: int) -> bool:
    """KDA for the first three layers of each 4-layer group; MLA for the 4th."""
    return (index % 4) != 3


def _mla_ordinal(index: int) -> int:
    """Ordinal of an MLA layer among the MLA layers (they sit at ``index % 4 == 3``).

    Used to read the declared mode layout ``mla_layer_modes`` (ADR-012), which
    is indexed by MLA-layer order, not by backbone depth.
    """
    return index // 4


def layer_kind(cfg: ModelConfig, index: int) -> str:
    """Kind of backbone layer ``index`` — one of :data:`LAYER_KINDS`.

    The leading ``cfg.dense_standard_layers`` layers are the *diagnostic*
    ``dense-standard`` blocks of the verification leg
    (``docs/specs/VERIFICATION-LEG.ru.md`` §«Конфигурация dense-reference»):
    a standard block — dense attention through the existing oracle
    (:func:`net.mla._dense_apply`) plus a dense SiTU-GLU MLP — used to separate
    pipeline bugs (data, packing, tokenisation, optimisation, resume, QAT) from
    the measured price of the KDA/MLA architecture.  The remaining tail keeps
    the pinned ``[K,K,K,M]`` pattern verbatim, indexed *relative to the prefix*,
    so the MLA ordinals of ``mla_layer_modes`` (ADR-012) stay aligned.

    ``dense_standard_layers = 0`` (the schema default, and every production
    config) returns exactly the pre-delta pattern: the business path is
    unchanged, and the regression is pinned by ``net/tests``.
    """
    prefix = int(cfg.dense_standard_layers)
    if index < prefix:
        return "dense-standard"
    return "mla" if not _layer_is_kda(index - prefix) else "kda"


def layer_kinds(cfg: ModelConfig) -> tuple[str, ...]:
    """Kind of every backbone layer, in depth order (see :func:`layer_kind`)."""
    return tuple(layer_kind(cfg, i) for i in range(cfg.num_layers))


def _mla_ordinal_at(cfg: ModelConfig, index: int) -> int:
    """MLA ordinal of backbone layer ``index`` (prefix-relative)."""
    return _mla_ordinal(index - int(cfg.dense_standard_layers))


def _layer_uses_dense_mlp(cfg: ModelConfig, index: int) -> bool:
    """Whether layer ``index`` keeps the dense SiTU-GLU MLP (no LatentMoE).

    A diagnostic ``dense-standard`` block is dense by definition (that is the
    point of the leg), whatever ``moe_dense_layers`` says; every other layer
    follows the pinned ``moe_dense_layers`` prefix.
    """
    return layer_kind(cfg, index) == "dense-standard" or _layer_is_dense(cfg, index)


def _layer_is_dense(cfg: ModelConfig, index: int) -> bool:
    """The leading ``moe_dense_layers`` layers keep the dense SiTU-GLU MLP."""
    return index < cfg.moe_dense_layers


def dense_moe_split(cfg: ModelConfig) -> tuple[int, int]:
    """(dense, latent-MoE) layer counts — (1, 23) for the full skeleton."""
    return cfg.moe_dense_layers, cfg.num_layers - cfg.moe_dense_layers


#: Layers per group-scan unit — one ``[KDA, KDA, KDA, MLA]`` block
#: (``net/config.py``'s ``num_layers % 4 == 0`` invariant).
_LAYERS_PER_UNIT = 4


def _group_scan_units(cfg: ModelConfig) -> tuple[int, str] | None:
    """Plan for the group-scan of the layer stack, or ``None`` to unroll.

    Returns ``(first_unit, mode)`` when the repeated tail groups can be run by
    one ``jax.lax.scan`` body: the leading ``first_unit`` units stay in python
    (they hold the one dense-MLP layer, ``net/model.py`` section 1) and the
    units ``first_unit .. num_layers/4 - 1`` share a body.  Every guard below
    that says "cannot" only *loses the host-compile saving* — the unrolled
    path is the same arithmetic — so a rejected plan is a journal warning, not
    an error.

    Guards (each is a structural premise of the scan, not a preference):

    * ``scan_layers`` off → the unrolled parity-reference path;
    * ``num_layers % 4`` → no whole ``[K,K,K,M]`` units to scan;
    * ``first_unit >= num_units`` → a single unit, nothing repeats;
    * heterogeneous ``mla_layer_modes`` over the scanned units → the shared
      body would have to branch on a per-iteration mode (a scan body cannot
      specialise per iteration without a host callback);
    * a live candidate pool (``not attn_dense_reference`` **and**
      ``mla_pool_size > 0``): the pool is a fixed-shape pytree carried across
      the scan, but ``sparse_union_attention`` flips it between ``None``
      (criterion-13 fast path, merged layer) and a built pool, so its
      *structure* is not stable across iterations.  Under the dense oracle the
      pool is never built (``apply_with_pool`` returns it untouched), which is
      exactly the pinned ``net/config.json`` layout;
    * a declared diagnostic ``dense_standard_layers`` prefix (VERIFICATION-LEG):
      the shared body below hard-codes the ``[K,K,K,M]`` sub-layer kinds, so it
      cannot reproduce a layer that is neither — the dense build runs unrolled.
    """
    if not bool(getattr(cfg, "scan_layers", False)):
        return None
    if int(getattr(cfg, "dense_standard_layers", 0)) > 0:
        _log.warning(
            "group-scan: %d diagnostic dense-standard layer(s) declared — the "
            "shared [K,K,K,M] body cannot reproduce them; falling back to the "
            "unrolled path",
            cfg.dense_standard_layers,
        )
        return None
    if int(cfg.num_layers) % _LAYERS_PER_UNIT != 0:
        return None
    n_units = int(cfg.num_layers) // _LAYERS_PER_UNIT
    first_unit = -(-int(cfg.moe_dense_layers) // _LAYERS_PER_UNIT)  # ceil
    if first_unit >= n_units:
        return None

    if not cfg.attn_dense_reference and int(cfg.mla_pool_size) > 0:
        _log.warning(
            "group-scan: candidate pool is live (attn_dense_reference=false, "
            "mla_pool_size=%d); the pool's pytree structure is not stable across "
            "scan iterations — falling back to the unrolled path",
            cfg.mla_pool_size,
        )
        return None

    modes = {mla_mod.layer_mode(cfg, u) for u in range(first_unit, n_units)}
    if len(modes) != 1:
        _log.warning(
            "group-scan: heterogeneous MLA modes over units %d..%d (%s) — "
            "falling back to the unrolled path",
            first_unit, n_units - 1, sorted(modes),
        )
        return None
    return first_unit, modes.pop()


def _stack_leaves(trees) -> object:
    """Stack a list of identical trees along a new leading axis (per leaf)."""
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *trees)


def init_params(key, cfg: ModelConfig) -> ModelParams:
    validate_config(cfg)
    hid = cfg.hidden
    keys = jax.random.split(key, cfg.num_layers + 6)
    emb = jax.random.normal(keys[0], (cfg.vocab_size, hid)) * 0.02
    layers = []
    for i in range(cfg.num_layers):
        kind = layer_kind(cfg, i)
        if kind == "kda":
            attn = kda_mod.init_kda(keys[i + 1], cfg)
        else:
            # ``mla`` and the diagnostic ``dense-standard`` block share the MLA
            # parameter tree: the dense-standard block reads it through the
            # existing dense oracle (``net/mla.py:_dense_apply``) instead of the
            # sparse+window path, so no second attention implementation is added.
            attn = mla_mod.init_mla(keys[i + 1], cfg)
        mlp = (
            mlp_mod.init_mlp(keys[i + 1], cfg)
            if _layer_uses_dense_mlp(cfg, i)
            else moe_mod.init_moe(keys[i + 1], cfg)
        )
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
    dense_standard: bool = False,
):
    """The residual delta of one backbone layer (attention + MLP/MoE).

    Returns ``(delta, qb_loss, pool)``; ``qb_loss`` is None unless the layer is
    a LatentMoE block and ``collect_qb`` is set.  ``pool`` is the ADR-012
    candidate pool: an MLA layer in mode ``full`` replaces it, any other MLA
    layer consumes it unchanged, KDA layers pass it through untouched.

    ``dense_standard`` marks the diagnostic VERIFICATION-LEG block: it runs the
    *dense* attention oracle (``net/mla.py:_dense_apply``) unconditionally —
    not through the ``attn_dense_reference`` switch, which is a property of the
    sparse path's A/B, not of this layer type — and passes the pool through
    untouched (like KDA, it neither builds nor consumes one).
    """
    hn = rms_norm(h, block.norm_attn)
    # ADR-054 Am.31 (цикл-19): границы вызовов компонентов — провенанс-метки
    # диагностики (``net/provenance.py``).  Гейт ``AXIOM_TRACE_MARKERS`` по
    # умолчанию выключен: ``mark`` делает прямой вызов, граф побитово прежний;
    # при ``=1`` имя scope (``probe_kda``/``probe_mla``/...) попадает в
    # ``op_name`` HLO и атрибутирует ядра/command buffer'ы трейса компоненту.
    # Оборачиваются только границы вызовов, не тела и не порядок операций.
    if dense_standard:
        attn_out = provenance.mark(
            "mla_dense", lambda: mla_mod._dense_apply(block.attn, cfg, hn)
        )
    elif is_kda:
        # ``cfg.kda_impl`` selects the KDA primitive (ADR-031 delta A):
        # ``chunked`` — the pre-delta associative-scan form (default), ``wyut``
        # — the WY/UT chunkwise form.  ``apply_kda`` keeps the branch in one
        # place so the two implementations stay swappable and cross-checked.
        attn_out = jax.vmap(
            lambda xb: provenance.mark(
                "kda",
                lambda xb: kda_mod.apply_kda(block.attn, cfg, xb, chunk_size),
                xb,
            )
        )(hn)
    else:
        attn_out, pool = provenance.mark(
            "mla",
            lambda: mla_mod.apply_with_pool(
                block.attn, cfg, hn, pool=pool, mode=mode
            ),
        )
    h1 = h + attn_out
    mlp_in = rms_norm(h1, block.norm_mlp)
    qb = None
    if isinstance(block.mlp, moe_mod.LatentMoEParams):
        if collect_qb:
            mlp_out, qb = provenance.mark(
                "moe",
                lambda: moe_mod.apply(block.mlp, cfg, mlp_in, want_qb=True),
            )
        else:
            mlp_out = provenance.mark(
                "moe", lambda: moe_mod.apply(block.mlp, cfg, mlp_in)
            )
    else:
        mlp_out = provenance.mark(
            "mlp", lambda: mlp_mod.apply(block.mlp, cfg, mlp_in)
        )
    return attn_out + mlp_out, qb, pool


def _layer_transition(
    block: BlockParams,
    attnres_w,
    h: jnp.ndarray,
    pool,
    prior_deltas: tuple,
    embed_src: jnp.ndarray,
    *,
    cfg: ModelConfig,
    chunk_size: int,
    collect_qb: bool,
    use_attnres: bool,
    index: int,
    is_kda: bool,
    mode: str,
    dense_standard: bool = False,
    block_remat_policy: str | None = None,
):
    """One backbone layer's transition, factored out of :func:`forward`.

    The residual delta plus the AttnRes correction over the *variable-length*
    prefix (embedding + every prior layer's delta) — the unrolled reference the
    group-scan reproduces with its masked fixed-shape form.  Returns
    ``(h_next, delta, qb, pool_next)``; ``qb`` is None unless the layer is a
    LatentMoE block and ``collect_qb`` is set.  ``dense_standard`` selects the
    diagnostic VERIFICATION-LEG block (see :func:`_block_delta`).

    ``block_remat_policy`` (``None`` = no boundary here) puts the
    :func:`net.remat.remat_checkpoint` boundary around the KDA/MLA+MoE *block*
    only, leaving the AttnRes correction outside — the ``per_layer_block``
    granularity (see :data:`GRAD_CKPT_POLICIES`) as composed by the unrolled
    path and by the scan's unrolled prefix.  The scanned backbone composes the
    same policy in :func:`_forward_group_scan`, where the boundary additionally
    covers the AttnRes read over the rebuilt view (that is what collapses the
    scan's stacked carry saves to one; see there).  Callers that ask for this
    must not wrap the whole transition again; with ``None`` the composition is
    the pre-delta one, bit-for-bit.
    """
    if block_remat_policy is None:
        delta, qb, pool_out = _block_delta(
            block, is_kda, cfg, h, chunk_size, collect_qb, mode, pool, dense_standard
        )
    else:
        def _block(block_, h_, pool_):
            return _block_delta(
                block_, is_kda, cfg, h_, chunk_size, collect_qb, mode, pool_,
                dense_standard,
            )

        delta, qb, pool_out = remat_checkpoint(_block, block_remat_policy)(
            block, h, pool
        )
    if use_attnres and index > 0:
        sources = jnp.stack([embed_src, *prior_deltas], axis=0)  # (N, B, T, hidden)
        corr = provenance.mark(
            "attnres", lambda: attnres_mod.apply_layer(attnres_w, sources)
        )
    else:
        corr = 0.0
    return h + delta + corr, delta, qb, pool_out


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
    remat_policy: str | None = None,
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

    ``grad_ckpt_policy`` (``none`` | ``per_layer`` | ``per_layer_block``, see
    ``GRAD_CKPT_POLICIES``) selects the remat granularity of the backbone loop.
    ``per_layer`` wraps each layer's transition in ``jax.checkpoint``; its live
    set is the layer input ``h``, the candidate-pool carry and the layer's own
    intermediates, so the backward pass recomputes the layer instead of
    retaining it.  ``per_layer_block`` narrows the scope to the KDA/MLA+MoE
    block: in the unrolled path the boundary wraps the block alone (the AttnRes
    correction stays outside); in the scanned backbone it wraps the block
    together with the AttnRes read over a view rebuilt from the shared
    unit-entry carry, so the differentiated scan keeps one stacked copy of the
    carry instead of one per sub-layer position; see ``GRAD_CKPT_POLICIES``.
    With ``none`` the graph is built exactly as before (bit-for-bit).

    ``remat_policy`` (``none`` | ``dots_saveable`` |
    ``dots_with_no_batch_dims_saveable``, ADR-049) selects *what stays saved*
    inside those per-layer boundaries; ``None`` reads ``cfg.remat_policy``.
    ``none`` applies ``jax.checkpoint`` without a policy, exactly as the code
    did before ADR-049 (bit-for-bit); the other two keep the outputs of the
    matmuls (all, or only the batch-free ones) instead of recomputing them.
    An unknown value raises here even when the boundaries are switched off, so
    a declaration can never be silently ignored.
    """
    if grad_ckpt_policy not in GRAD_CKPT_POLICIES:
        raise ValueError(
            f"неизвестная политика grad-checkpointing: {grad_ckpt_policy!r}; "
            f"ожидается одна из {GRAD_CKPT_POLICIES}"
        )
    remat = grad_ckpt_policy in ("per_layer", "per_layer_block")
    # ``per_layer_block``: the boundary sits inside ``_layer_transition``
    # (around the block), so the callers below must not wrap the transition.
    block_only = grad_ckpt_policy == "per_layer_block"
    remat_policy = cfg.remat_policy if remat_policy is None else remat_policy
    validate_remat_policy(remat_policy)  # fail-closed before any tracing

    # Group-scan of the repeated `[K,K,K,M]` tail groups (host-compile RAM /6):
    # declared by ``cfg.scan_layers`` and admitted only when the layout lets one
    # shared body reproduce the unrolled arithmetic (``_group_scan_units``).
    # ``None`` (the flag off, or an un-scanable layout) runs the loop below
    # verbatim — the parity reference.
    plan = _group_scan_units(cfg)
    if plan is not None:
        return _forward_group_scan(
            params,
            cfg,
            input_ids,
            chunk_size=chunk_size,
            use_attnres=use_attnres,
            return_hidden=return_hidden,
            collect_qb=collect_qb,
            remat=remat,
            block_only=block_only,
            remat_policy=remat_policy,
            emit_logits=emit_logits,
            first_unit=plan[0],
            mode=plan[1],
        )

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
        dense_standard: bool,
        block_remat_policy: str | None,
    ):
        """One backbone layer's transition — the remat unit.

        Takes the layer's parameters and the full live set (``h``, the pool
        carry, the previous deltas that AttnRes mixes) and returns the next
        ``h``, this layer's delta, its QB term and the updated pool.
        ``index``/``is_kda``/``mode``/``dense_standard`` are static and captured
        by the caller; ``block_remat_policy`` non-None moves the boundary
        inside :func:`_layer_transition` (the ``per_layer_block`` policy).
        """
        return _layer_transition(
            block, attnres_w, h, pool, prior_deltas, embed_src,
            cfg=cfg, chunk_size=chunk_size, collect_qb=collect_qb,
            use_attnres=use_attnres, index=index, is_kda=is_kda, mode=mode,
            dense_standard=dense_standard, block_remat_policy=block_remat_policy,
        )

    for i, block in enumerate(params.layers):
        kind = layer_kind(cfg, i)
        is_kda = kind == "kda"
        dense_standard = kind == "dense-standard"
        # Only the KDA/MLA pattern reads ``mla_layer_modes`` (ADR-012); a KDA or
        # a diagnostic dense-standard layer carries no selection mode at all.
        mode = (
            mla_mod.layer_mode(cfg, _mla_ordinal_at(cfg, i))
            if kind == "mla"
            else "full"
        )

        def step(
            block, attnres_w, h, pool, prior_deltas, embed_src,
            _i=i, _is_kda=is_kda, _mode=mode, _dense=dense_standard,
            _blk=remat_policy if block_only else None,
        ):
            return _layer_step(
                block, attnres_w, h, pool, prior_deltas, embed_src,
                index=_i, is_kda=_is_kda, mode=_mode, dense_standard=_dense,
                block_remat_policy=_blk,
            )

        if remat and not block_only:
            step = remat_checkpoint(step, remat_policy)
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
        out.append(compute_dtype.gemm(h, params.embedding.T))
    if return_hidden:
        out.append(h)
    if collect_qb:
        qb_mean = jnp.stack(qb_losses).mean() if qb_losses else jnp.zeros(())
        out.append(qb_mean)
    return out[0] if len(out) == 1 else tuple(out)


def _forward_group_scan(
    params: ModelParams,
    cfg: ModelConfig,
    input_ids: jnp.ndarray,
    *,
    chunk_size: int,
    use_attnres: bool,
    return_hidden: bool,
    collect_qb: bool,
    remat: bool,
    block_only: bool,
    remat_policy: str,
    emit_logits: bool,
    first_unit: int,
    mode: str,
) -> jnp.ndarray | tuple:
    """``forward`` with the repeated ``[K,K,K,M]`` tail run by one ``lax.scan``.

    Same arithmetic, same order, same parameters as the loop in :func:`forward`
    — the host-compile working set shrinks because XLA reuses one unit body's
    buffers across the scanned units instead of tracing a distinct region per
    layer (``tools/bench_block_merge.py`` measures the per-record cost flat in
    the layer count inside a scan).  The plan's guards (``_group_scan_units``)
    reject every layout the shared body cannot reproduce, so reaching here means
    the scanned units are uniform: all past the dense-MLP prefix (hence all
    LatentMoE), one MLA ``mode``, and a candidate pool whose pytree structure
    does not change across iterations (under the pinned ``attn_dense_reference``
    the pool is the ``None`` it started as).

    AttnRes is the one piece that is *not* a fixed-shape tensor in the unrolled
    loop: layer ``i`` mixes the embedding with every earlier layer's delta, an
    ``(i+1, B, T, hidden)`` stack rebuilt each step.  The scan carries a
    fixed-shape ``sources`` buffer of ``(num_layers+1, B, T, hidden)`` (slot 0
    the embedding, slot ``k+1`` layer ``k``'s delta) and reads the live prefix
    with :func:`net.attnres.apply_layer_masked`, which pushes the not-yet-written
    slots to ``-inf`` before the softmax — bit-for-bit the variable-length
    :func:`net.attnres.apply_layer` over the same prefix.

    ``remat``/``block_only`` compose the boundary of every sub-layer: ``remat``
    alone wraps the sub-layer transition (``per_layer``); ``block_only`` (with
    ``remat``) wraps the KDA/MLA+MoE block *together with the AttnRes read over
    a rebuilt view* — the ``per_layer_block`` granularity.  Each sub-layer
    boundary of a unit takes the same **unit-entry carry** as its big input
    (``src_in``) and re-applies the unit's deltas inside the boundary, so the
    differentiated scan collects **one** stacked copy of the ``sources``
    carry instead of one per sub-layer position; the correction add and the
    carry update stay outside (see ``GRAD_CKPT_POLICIES``).
    """
    n_units = int(cfg.num_layers) // _LAYERS_PER_UNIT
    units = list(range(first_unit, n_units))
    prefix_layers = first_unit * _LAYERS_PER_UNIT

    emb = params.embedding[input_ids]  # (B, T, hidden)
    h = emb
    b, t = input_ids.shape
    sources = jnp.zeros((cfg.num_layers + 1, b, t, cfg.hidden), dtype=emb.dtype)
    sources = sources.at[0].set(emb)
    pool = None  # ADR-012 candidate pool — ``None`` under the dense oracle
    qb_terms: list[jnp.ndarray] = []
    prefix_deltas: list[jnp.ndarray] = []

    for i in range(prefix_layers):
        block = params.layers[i]
        is_kda = _layer_is_kda(i)
        lm = "full" if is_kda else mla_mod.layer_mode(cfg, _mla_ordinal(i))

        def step(
            block, attnres_w, h, pool, prior_deltas, embed_src,
            _i=i, _is_kda=is_kda, _mode=lm,
            _blk=remat_policy if block_only else None,
        ):
            return _layer_transition(
                block, attnres_w, h, pool, prior_deltas, embed_src,
                cfg=cfg, chunk_size=chunk_size, collect_qb=collect_qb,
                use_attnres=use_attnres, index=_i, is_kda=_is_kda, mode=_mode,
                block_remat_policy=_blk,
            )

        if remat and not block_only:
            step = remat_checkpoint(step, remat_policy)
        attnres_w = params.attnres.w[i] if use_attnres else None
        h, delta, qb, pool = step(
            block, attnres_w, h, pool, tuple(prefix_deltas), emb
        )
        if qb is not None:
            qb_terms.append(qb)
        prefix_deltas.append(delta)
        sources = sources.at[i + 1].set(delta)

    def layer(u: int, s: int) -> BlockParams:
        return params.layers[u * _LAYERS_PER_UNIT + s]

    # One stack per sub-layer and field (a NamedTuple parameter tree cannot be
    # indexed on a stacked axis): norm vectors (U, 4, hidden), the three KDA
    # attention trees and the one MLA tree stacked over units, the four MLP
    # trees, and the unit ordinals for the AttnRes pseudo-query lookup.
    xs = (
        jnp.stack([jnp.stack([layer(u, s).norm_attn for s in range(4)]) for u in units]),
        jnp.stack([jnp.stack([layer(u, s).norm_mlp for s in range(4)]) for u in units]),
        _stack_leaves([layer(u, 0).attn for u in units]),
        _stack_leaves([layer(u, 1).attn for u in units]),
        _stack_leaves([layer(u, 2).attn for u in units]),
        _stack_leaves([layer(u, 3).attn for u in units]),
        tuple(_stack_leaves([layer(u, s).mlp for u in units]) for s in range(4)),
        jnp.asarray(units, dtype=jnp.int32),
    )

    def body(carry, unit):
        h, pool, sources, qb_sum = carry
        (
            norm_attn, norm_mlp, attn0, attn1, attn2, attn_mla, mlp, u,
        ) = unit
        # ``per_layer_block``: the unit-entry carry — the *same* buffer for all
        # sub-layer boundaries of this unit, hence a single save for the
        # differentiated scan (see ``GRAD_CKPT_POLICIES``).  Each boundary
        # rebuilds its mid-unit view from it plus the unit's deltas below.
        src_in = sources
        unit_deltas: list[jnp.ndarray] = []
        for s, attn_p in enumerate((attn0, attn1, attn2, attn_mla)):
            is_kda = s != _LAYERS_PER_UNIT - 1
            index = u * _LAYERS_PER_UNIT + s
            block = BlockParams(
                norm_attn=norm_attn[s], attn=attn_p,
                norm_mlp=norm_mlp[s], mlp=mlp[s],
            )

            def sub(
                block, attnres_w, h, pool, sources, deltas_u,
                _is_kda=is_kda, _index=index, _mode="full" if is_kda else mode,
            ):
                # ``per_layer_block`` (``block_only``): the boundary wraps the
                # block *and* the AttnRes read, but over a view rebuilt from
                # the shared unit-entry buffer ``src_in`` and the unit's
                # deltas — slot-for-slot the carried buffer's values, so the
                # arithmetic is unchanged, while the boundary's big input is
                # one loop-invariant buffer rather than a per-sub-layer version
                # of it.  The differentiated scan then keeps one stacked copy
                # of the carry, not one per sub-layer position (see
                # ``GRAD_CKPT_POLICIES``).  With ``per_layer``/``none``
                # ``deltas_u`` is empty and ``sources`` is the live carry:
                # the transition is wrapped (``remat``) or not, exactly as
                # before — never both boundaries at once.
                if block_only and use_attnres:
                    first_slot = _index - len(deltas_u) + 1
                    for k, delta_k in enumerate(deltas_u):
                        sources = sources.at[first_slot + k].set(delta_k)
                delta, qb, pool_out = _block_delta(
                    block, _is_kda, cfg, h, chunk_size, collect_qb, _mode, pool
                )
                if use_attnres:
                    corr = attnres_mod.apply_layer_masked(
                        attnres_w, sources, _index + 1
                    )
                else:
                    corr = 0.0
                return h + delta + corr, delta, qb, pool_out

            if remat:
                sub = remat_checkpoint(sub, remat_policy)
            attnres_w = (
                jnp.take(params.attnres.w, index, axis=0) if use_attnres else None
            )
            carried = src_in if block_only else sources
            # the deltas argument exists for the ``per_layer_block`` rebuild
            # only; passing it under ``per_layer``/``none`` would add unused
            # operands to the wrapped function and move the pre-delta graph
            # (empty tuple = no pytree leaves = no operands).
            deltas_arg = tuple(unit_deltas) if block_only else ()
            h, delta, qb, pool = sub(
                block, attnres_w, h, pool, carried, deltas_arg
            )
            sources = sources.at[index + 1].set(delta)
            unit_deltas.append(delta)
            if collect_qb and qb is not None:
                qb_sum = qb_sum + qb
        return (h, pool, sources, qb_sum), None

    carry = (h, pool, sources, jnp.zeros((), dtype=jnp.float32))
    (h, pool, sources, qb_sum), _ys = jax.lax.scan(body, carry, xs)

    h = rms_norm(h, params.norm_final)
    out: list = []
    if emit_logits:
        out.append(compute_dtype.gemm(h, params.embedding.T))
    if return_hidden:
        out.append(h)
    if collect_qb:
        # The unrolled loop means over every LatentMoE layer's QB term; the scan
        # split that set into the prefix terms (kept in python) and the scanned
        # sum, so rebuild the same numerator and divide by the same count.
        qb_total = qb_sum
        for term in qb_terms:
            qb_total = qb_total + term
        denom = max(int(cfg.num_layers) - int(cfg.moe_dense_layers), 1)
        out.append(qb_total / denom)
    return out[0] if len(out) == 1 else tuple(out)


def _cross_entropy(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    logp = jax.nn.log_softmax(logits, axis=-1)
    nll = jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
    return -nll.mean()


# --------------------------------------------------------------------------- #
# Веха-9 (ADR-054 Am.20, кампания MFU-55): bf16-хранение логитов CE в backward.
#
# Пол B=4 по HLO-дампу включает fp32-логиты чанка CE (4,4096,160000) — 9.77
# ГиБ × 71 вхождение в графе.  Гейт хранит логиты чанка для backward в bf16
# (каст при сохранении), арифметика logits→logsumexp/softmax остаётся fp32
# (апкаст при чтении).  Чтение fail-closed (форма ``AXIOM_ACCUM_JIT``/
# ``AXIOM_MASTER_DTYPE``): опечатка не должна тихо оставить fp32.
# --------------------------------------------------------------------------- #

#: Env-гейт bf16-хранения логитов чанкованного CE в backward (``0`` — дефолт).
CE_BF16_ENV = "AXIOM_CE_BF16"

#: Допустимые значения; неизвестное — ошибка (fail-closed, без догадок).
CE_BF16_VALUES = ("0", "1")
CE_BF16_OFF = "0"
CE_BF16_ON = "1"


class CeBf16Error(ValueError):
    """Неизвестное значение ``AXIOM_CE_BF16`` (fail-closed, без догадок)."""


def ce_bf16_mode(value: str | None = None) -> bool:
    """Хранение логитов CE-чанка в backward: ``False`` (дефолт) | ``True``.

    ``AXIOM_CE_BF16=1`` включает ветку :func:`_chunk_sum_bf16` в
    :func:`_chunked_cross_entropy`: логиты чанка, нужные backward, хранятся
    в bf16 (каст при сохранении), backward читает их с апкастом в fp32
    (softmax-арифметика fp32); forward-числа ветка не меняет.  Пустая/
    незаданная переменная = ``"0"``; значения — только ``"0"``/``"1"``;
    неизвестное — :class:`CeBf16Error`.  Явный аргумент сильнее окружения
    (форма ``accum_jit_mode``/``master_dtype_mode``).  Гейт читается на
    трейсе; вне чанкованного CE (``ce_chunk_tokens=0``) он инертен.
    """
    raw = value if value is not None else os.environ.get(CE_BF16_ENV)
    mode = (raw or "").strip().lower() or CE_BF16_OFF
    if mode not in CE_BF16_VALUES:
        raise CeBf16Error(
            f"{CE_BF16_ENV}={mode!r}: ожидается одно из {list(CE_BF16_VALUES)}"
        )
    return mode == CE_BF16_ON


@jax.custom_vjp
def _chunk_sum_bf16(
    f: jnp.ndarray,
    t: jnp.ndarray,
    emb: jnp.ndarray,
    mask: jnp.ndarray | None,
):
    """Один CE-чанк с bf16-хранением логитов для backward (веха-9).

    Forward — ровно выражения прежнего ``chunk_sum`` (:func:`_chunked_cross_entropy`):
    logits → ``log_softmax`` → gather NLL → глобальные sum/count; числа
    побитово те же.  Гейт ``AXIOM_CE_BF16=1`` включается только в backward:
    остаток — bf16-копия логитов (B,chunk,V); backward читает её с апкастом
    в fp32 и считает softmax-формулу ``grad_logits = (softmax(logits) −
    onehot(t))·g_sum`` (стандартная форма VJP log-softmax+gather) в fp32.

    Почему custom_vjp, а не каст в forward.  ``logits.astype(bf16).astype(f32)``
    в forward не срезает пол: transpose-правила ``log_softmax``/gather держат
    живыми fp32-копии (B,chunk,V) (трассировка backward-остатков показывает
    полный fp32-логит и до, и после каста).  Явный backward фиксирует состав
    остатков: ровно ``(bf16 logits, t, emb, f, mask)`` — 9.77→4.9 ГиБ на
    живой тензор чанка.

    Котангент счётчика: в числитель/знаменатель он входит только через
    ``mask``; маска — данные (паддинг), не параметр, и её котангент здесь
    объявлен нулевым (``None``), как и у ``t`` — форма остальных
    non-differentiable входов.

    Контракт backward по dtype (фикс-раунд вехи-9): возвращаемые котангенты
    несут dtype своих прималов (``f``/``emb``) — тот же результат, что даёт
    autodiff прежнего пути; при ``param_dtype=bfloat16`` это bf16-градиент
    эмбеддинга (не fp32), см. :func:`_chunk_sum_bf16_bwd`.
    """
    logits = compute_dtype.gemm(f, emb.T)
    logp = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(logp, t[..., None], axis=-1)[..., 0]
    if mask is None:
        return jnp.sum(nll), jnp.array(nll.size, jnp.float32)
    m = mask.astype(nll.dtype)
    return jnp.sum(nll * m), jnp.sum(m)


def _chunk_sum_bf16_fwd(f, t, emb, mask):
    logits = compute_dtype.gemm(f, emb.T)
    logp = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(logp, t[..., None], axis=-1)[..., 0]
    if mask is None:
        out = (jnp.sum(nll), jnp.array(nll.size, jnp.float32))
    else:
        m = mask.astype(nll.dtype)
        out = (jnp.sum(nll * m), jnp.sum(m))
    # Сохраняемый остаток — bf16-копия логитов (и мелкие t/emb/f/mask).
    return out, (logits.astype(jnp.bfloat16), t, emb, f, mask)


def _chunk_sum_bf16_bwd(res, g):
    logits_stored, t, emb, f, mask = res
    g_sum, _g_count = g
    logits = logits_stored.astype(jnp.float32)  # апкаст при чтении; дальше fp32
    p = jax.nn.softmax(logits, axis=-1)
    onehot = jax.nn.one_hot(t, logits.shape[-1], dtype=jnp.float32)
    grad_logits = (p - onehot) * g_sum
    if mask is not None:
        grad_logits = grad_logits * mask[..., None].astype(jnp.float32)
    grad_f = compute_dtype.gemm(grad_logits, emb)
    grad_emb = compute_dtype.gemm_einsum("btv,bth->vh", grad_logits, f)
    # Контракт backward (стенд GB10, фикс-раунд вехи-9): котангент обязан нести
    # dtype примала — ровно то, что даёт autodiff прежнего пути (сверено:
    # gate-off grad эмбеддинга bf16 — bf16).  Арифметика выше всегда fp32
    # (апкаст при чтении), но gemm_einsum градиента не знает о dtype emb, и без
    # каста при param_dtype=bfloat16 (рабочая копия bf16, fp32-мастер — режим
    # кампании) котангент emb шёл fp32 и падал при сложении со котангентом
    # второго потребителя того же тензора (``params.embedding[ids]`` в MTP,
    # bf16) в ``add_any``: AssertionError (float32[V,H], bfloat16[V,H]).
    return grad_f.astype(f.dtype), None, grad_emb.astype(emb.dtype), None


_chunk_sum_bf16.defvjp(_chunk_sum_bf16_fwd, _chunk_sum_bf16_bwd)


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

    Веха-9 (ADR-054 Am.20): гейт ``AXIOM_CE_BF16=1`` (:func:`ce_bf16_mode`)
    подменяет тело чанка на :func:`_chunk_sum_bf16` — forward-числа те же,
    но логиты, сохраняемые для backward, хранятся в bf16 (9.77→4.9 ГиБ на
    живой тензор чанка).  Дефолт 0 — путь ниже побитово прежний.

    This checkpoint is *not* one of the ADR-049 remat boundaries and keeps
    ``jax.checkpoint`` without a policy on purpose: it is a chunking mechanism
    whose whole point is to avoid retaining the vocabulary-sized matmul output
    (``f @ embedding.T`` is exactly what ``dots_saveable`` would keep), so a
    save-the-dots policy here would spend the memory this path exists to save.
    ADR-049 names the backbone layer (``net/model.py``) and the KDA scan bodies
    (``net/kda.py``) as the policy's consumers.
    """
    _, total_rows = targets.shape
    if total_rows == 0 or ce_chunk_tokens <= 0:
        raise ValueError(
            "ce_chunk_tokens must be > 0 and targets non-empty for the chunked "
            f"path (got ce_chunk_tokens={ce_chunk_tokens!r}, rows={total_rows})"
        )
    n_chunks = -(-total_rows // ce_chunk_tokens)  # ceil

    def chunk_sum(f, t, emb, mask):
        logits = compute_dtype.gemm(f, emb.T)
        logp = jax.nn.log_softmax(logits, axis=-1)
        nll = -jnp.take_along_axis(logp, t[..., None], axis=-1)[..., 0]  # (b, chunk)
        if mask is None:
            return jnp.sum(nll), jnp.array(nll.size, jnp.float32)
        m = mask.astype(nll.dtype)
        return jnp.sum(nll * m), jnp.sum(m)

    # Веха-9: гейт выбирает тело чанка (bf16-хранение логитов для backward);
    # дефолт 0 оставляет прежнее тело побитово.
    chunk_body = _chunk_sum_bf16 if ce_bf16_mode() else chunk_sum

    total_sum = jnp.zeros(())
    total_count = jnp.zeros(())
    for c in range(n_chunks):
        start = c * ce_chunk_tokens
        end = min(start + ce_chunk_tokens, total_rows)
        mask = None if loss_mask is None else loss_mask[:, start:end]
        part_sum, part_count = jax.checkpoint(chunk_body)(
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
    logits = compute_dtype.gemm(mtp_out, params.embedding.T)  # (B, T-2, vocab)
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
    remat_policy: str | None = None,
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

    ``remat_policy`` (ADR-049) selects what those per-layer boundaries keep
    saved; ``None`` reads ``cfg.remat_policy`` (schema default ``none``, i.e.
    the pre-ADR-049 graph bit-for-bit).  It changes no arithmetic — only which
    activations the backward pass recomputes instead of retaining.

    The cross-entropy reduction is selected by ``cfg.ce_chunk_tokens`` (also
    declared in ``net/config.json``): ``0`` keeps the naive whole-vocabulary
    :func:`_cross_entropy`; a positive value routes both heads through
    :func:`_chunked_cross_entropy`, so the ``(B, T, vocab)`` logits are never
    materialised.  Both paths reduce to the same mean up to floating-point
    reduction order.
    """
    policy = cfg.grad_ckpt_policy if grad_ckpt_policy is None else grad_ckpt_policy
    ce_tokens = int(cfg.ce_chunk_tokens)
    # ADR-054 Am.33 (цикл-21): тонкие провенанс-метки ``forward``/``ntp_ce``/
    # ``mtp`` — только границы вызовов (``net/provenance.py``).  Прежняя метка
    # ``probe_ce`` стояла вокруг всего ``compute_loss`` и через обратный проход
    # захватывала backward (37.5% шага), из-за чего CE-голова/lm_head не
    # отделялись; здесь каждый компонент получает своё имя.  При ``off``
    # (дефолт) ``mark`` — прямой вызов: те же объекты/формы/числа, ни одной
    # новой jit/checkpoint-границы; финальная сумма намеренно не обёрнута.
    if ce_tokens > 0:
        hidden, qb = provenance.mark(
            "forward",
            lambda: forward(
                params, cfg, input_ids, chunk_size, use_attnres,
                return_hidden=True, collect_qb=True,
                grad_ckpt_policy=policy, emit_logits=False,
                remat_policy=remat_policy,
            ),
        )
        ntp = provenance.mark(
            "ntp_ce",
            lambda: _chunked_cross_entropy(
                hidden[:, :-1], input_ids[:, 1:], params.embedding, ce_tokens
            ),
        )
    else:
        logits, hidden, qb = provenance.mark(
            "forward",
            lambda: forward(
                params, cfg, input_ids, chunk_size, use_attnres,
                return_hidden=True, collect_qb=True,
                grad_ckpt_policy=policy,
                remat_policy=remat_policy,
            ),
        )
        ntp = provenance.mark(
            "ntp_ce",
            lambda: _cross_entropy(logits[:, :-1], input_ids[:, 1:]),
        )
    aux = provenance.mark(
        "mtp",
        lambda: mtp_loss(
            params, cfg, hidden, input_ids, chunk_size, ce_chunk_tokens=ce_tokens
        ),
    )
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
        if _layer_uses_dense_mlp(cfg, i):
            total += _size(block.mlp)
            continue
        inactive = 0
        for name in ("expert_g", "expert_u", "expert_d"):
            leaf = getattr(block.mlp, name)
            n_routed = leaf.shape[0]
            inactive += (n_routed - cfg.moe_top_k) * math.prod(leaf.shape[1:])
        total += _size(block.mlp) - inactive
    return total
