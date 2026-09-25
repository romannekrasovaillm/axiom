"""Gated MLA — Multi-head Latent Attention with a full-rank output gate
(kda-formulas.md section 2), plus the long-context delta v1.5 (ADR-009).

NoPE: no positional encoding is applied to queries or keys (:353-357).  The
key/value representations are compressed into a low-dimensional latent
``c_t = W_c x_t`` and reconstructed through up-projections at attention time,
so only the latent is cached (:348-351).  The attention output is kept in FP32
during training to correct biased flash-attention rounding (:367-370) — the
skeleton computes attention in FP32 always.

Delta v1.5 (ADR-009, D1/D2/D3), all behind ``config.attn_dense_reference``:

* **D1** — a local sliding-window branch (``swa_window``) with its *own* K/V
  projections from the hidden state; the source's SWA has dedicated parameters
  (V4.1-Flash section 2.2), unlike the KDA sharing (see ``kda.py``).
* **D2** — sparse selection (CSA2-lite): a light indexer
  (``mla_index_heads x mla_index_dim``) scores the main-KV records from a query
  projected off the hidden, its keys are projected *from the latent* (as in
  CSA2, not from the hidden); attention runs over the union of the selected
  records and the window, duplicates removed by mask.
* **D3** — optional MXFP4 fake-quant of the latent ``c_t``
  (``qat_kv_enabled``, off at pretrain); the SWA KV stays FP8 (see
  ``quant.py``).

The dense causal path is kept verbatim as the oracle: it is what
``attn_dense_reference = True`` (the default) selects, and the reference for
criterion 13 and the A/B against the sparse path (ADR-009 D4).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import ModelConfig
from . import attn_sparse
from . import quant


def _rand(key, shape, scale: float) -> jnp.ndarray:
    return jax.random.normal(key, shape) * scale


class MLAParams(NamedTuple):
    W_c: jnp.ndarray  # (hidden, d_c)
    W_q: jnp.ndarray  # (hidden, H*dq)
    W_k_up: jnp.ndarray  # (d_c, H*dq)
    W_v_up: jnp.ndarray  # (d_c, H*dq)
    W_o: jnp.ndarray  # (H*dq, hidden)
    W_g: jnp.ndarray  # (hidden, hidden)
    # --- delta v1.5 (ADR-009) --------------------------------------------
    W_swa_k: jnp.ndarray  # (hidden, H*dq) — own SWA key projection (D1)
    W_swa_v: jnp.ndarray  # (hidden, H*dq) — own SWA value projection (D1)
    W_idx_q: jnp.ndarray  # (hidden, Hi*Di) — indexer queries from hidden (D2)
    W_idx_k: jnp.ndarray  # (d_c, Hi*Di) — indexer keys from the main KV (D2)
    swa_logit: jnp.ndarray  # () — learnable window share in the union softmax


def init_mla(key, cfg: ModelConfig) -> MLAParams:
    H, dq = cfg.num_heads, cfg.mla_head_dim
    hid, dc = cfg.hidden, cfg.mla_latent_dim
    di = cfg.mla_index_heads * cfg.mla_index_dim
    k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11 = jax.random.split(key, 11)
    # The indexer seed is folded in (like the MoE routing seed) so the
    # selection is reproducible from the pinned run manifest (spine AD-4).
    idx_key = jax.random.fold_in(k10, cfg.indexer_seed)
    return MLAParams(
        W_c=_rand(k1, (hid, dc), 0.02),
        W_q=_rand(k2, (hid, H * dq), 0.02),
        W_k_up=_rand(k3, (dc, H * dq), 0.02),
        W_v_up=_rand(k4, (dc, H * dq), 0.02),
        W_o=_rand(k5, (H * dq, hid), 0.02),
        W_g=_rand(k6, (hid, hid), 0.02),
        W_swa_k=_rand(k7, (hid, H * dq), 0.02),
        W_swa_v=_rand(k8, (hid, H * dq), 0.02),
        W_idx_q=_rand(k9, (hid, di), 0.02),
        W_idx_k=_rand(idx_key, (dc, di), 0.02),
        swa_logit=jnp.zeros(()),
    )


def _causal_mask(T: int) -> jnp.ndarray:
    return jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))


def _project(params: MLAParams, cfg: ModelConfig, x32: jnp.ndarray, c: jnp.ndarray):
    """Per-head q, k, v from the latent (shared by the dense and sparse paths)."""
    H, dq = cfg.num_heads, cfg.mla_head_dim
    q = (x32 @ params.W_q.astype(jnp.float32)).reshape(*x32.shape[:-1], H, dq)
    k = (c @ params.W_k_up.astype(jnp.float32)).reshape(*x32.shape[:-1], H, dq)
    v = (c @ params.W_v_up.astype(jnp.float32)).reshape(*x32.shape[:-1], H, dq)
    return q, k, v


def _dense_apply(params: MLAParams, cfg: ModelConfig, x: jnp.ndarray, mask: jnp.ndarray | None = None) -> jnp.ndarray:
    """Dense causal gated MLA over ``(B, T, hidden)`` (the oracle, ADR-009 D4)."""
    H, dq = cfg.num_heads, cfg.mla_head_dim
    x32 = x.astype(jnp.float32)
    c = x32 @ params.W_c.astype(jnp.float32)  # (B, T, d_c)
    q, k, v = _project(params, cfg, x32, c)

    scale = 1.0 / jnp.sqrt(dq)
    scores = jnp.einsum("bthd,bshd->bhts", q, k) * scale  # (B, H, T, T)
    if mask is None:
        mask = _causal_mask(x.shape[-2])
    scores = jnp.where(mask, scores, jnp.finfo(jnp.float32).min)
    attn = jax.nn.softmax(scores, axis=-1)  # (B, H, T, T)
    o = jnp.einsum("bhts,bshd->bthd", attn, v)  # (B, T, H, dq)
    o = o.reshape(*x.shape[:-1], H * dq)

    gate = jax.nn.sigmoid(x32 @ params.W_g.astype(jnp.float32))
    return (gate * o) @ params.W_o.astype(jnp.float32)


def layer_mode(cfg: ModelConfig, ordinal: int) -> str:
    """Selection mode of MLA layer ``ordinal`` (ADR-012, declared in the config).

    Layers past the declared layout default to ``full``, which is the ADR-009
    D2 behaviour and the exact fallback of a consumer without a pool.
    """
    modes = tuple(cfg.mla_layer_modes)
    return modes[ordinal] if ordinal < len(modes) else "full"


def wants_selection(cfg: ModelConfig) -> bool:
    """Whether a later MLA layer is in mode ``reuse``.

    Only then does the pool builder materialise its ``top_k`` indices for
    sharing: that array is ``O(T * top_k)`` and is pure overhead otherwise.
    """
    return any(mode == "reuse" for mode in tuple(cfg.mla_layer_modes))


def _sparse_apply(
    params: MLAParams,
    cfg: ModelConfig,
    x: jnp.ndarray,
    mode: str = "full",
    pool: attn_sparse.CandidatePool | None = None,
):
    """Sparse + sliding-window gated MLA (ADR-009 D1/D2/D3 + ADR-012 pool)."""
    H, dq = cfg.num_heads, cfg.mla_head_dim
    x32 = x.astype(jnp.float32)
    c = x32 @ params.W_c.astype(jnp.float32)  # (B, T, d_c)
    if cfg.qat_kv_enabled:
        # D3: fake-quant the cached latent in MXFP4, dequantise before attention.
        c = quant.fake_quant_mxfp4_latent(c)
    q, k, v = _project(params, cfg, x32, c)

    window = int(cfg.swa_window)
    if window > 0:
        ks = (x32 @ params.W_swa_k.astype(jnp.float32)).reshape(*x32.shape[:-1], H, dq)
        vs = (x32 @ params.W_swa_v.astype(jnp.float32)).reshape(*x32.shape[:-1], H, dq)
        # The SWA KV stays FP8 (source: the window is quantisation-sensitive).
        ks = quant.fake_quant_fp8_e4m3(ks)
        vs = quant.fake_quant_fp8_e4m3(vs)
    else:
        ks = vs = None

    # Mode ``reuse`` renders no scores at all (it takes the builder's selection
    # verbatim), so its indexer projections are not computed — the same
    # reasoning as the criterion-13 fast path: no work for a selection that is
    # not made here.
    if mode == "reuse" and pool is not None and pool.selection is not None:
        idx_q = idx_k = None
    else:
        idx_q = (x32 @ params.W_idx_q.astype(jnp.float32)).reshape(*x32.shape[:-1], cfg.mla_index_heads, cfg.mla_index_dim)
        idx_k = (c @ params.W_idx_k.astype(jnp.float32)).reshape(*x32.shape[:-1], cfg.mla_index_heads, cfg.mla_index_dim)

    o, pool_out = attn_sparse.sparse_union_attention(
        q, k, v, ks, vs, idx_q, idx_k,
        top_k=int(cfg.mla_top_k),
        window=window,
        window_bias=params.swa_logit,
        mode=mode,
        pool=pool,
        pool_block=int(cfg.mla_pool_block),
        pool_size=int(cfg.mla_pool_size),
        want_selection=wants_selection(cfg),
        fused=bool(cfg.attn_fused_assembly),
        exact_topk=bool(cfg.attn_topk_exact),
    )
    o = o.reshape(*x.shape[:-1], H * dq)
    gate = jax.nn.sigmoid(x32 @ params.W_g.astype(jnp.float32))
    return (gate * o) @ params.W_o.astype(jnp.float32), pool_out


def apply_with_pool(
    params: MLAParams,
    cfg: ModelConfig,
    x: jnp.ndarray,
    mask: jnp.ndarray | None = None,
    pool: attn_sparse.CandidatePool | None = None,
    mode: str = "full",
):
    """Gated MLA plus the shared candidate pool (ADR-012).

    Returns ``(out, pool_out)``: the layer output and the pool to hand to the
    next MLA layer.  The dense oracle ignores the pool (it is the reference,
    D4), so criterion 13 is unaffected by the pooling layout.
    """
    if cfg.attn_dense_reference:
        return _dense_apply(params, cfg, x, mask), pool
    return _sparse_apply(params, cfg, x, mode=mode, pool=pool)


def apply(params: MLAParams, cfg: ModelConfig, x: jnp.ndarray, mask: jnp.ndarray | None = None) -> jnp.ndarray:
    """Gated MLA over ``(B, T, hidden)`` (FP32 attention, Eq. 7).

    ``config.attn_dense_reference`` selects the dense oracle (default, D4);
    otherwise the v1.5 sparse + window path runs (ADR-009).  A single layer
    called this way is its own pool builder (mode ``full``): the sharing
    (ADR-012) is a property of the layer *stack*, threaded by ``model.forward``.
    """
    return apply_with_pool(params, cfg, x, mask)[0]


def topk_indices(
    params: MLAParams, cfg: ModelConfig, x: jnp.ndarray
) -> jnp.ndarray:
    """Deterministic indexer selection ``(B, T, min(mla_top_k, T))``.

    Mirrors the mode-``full`` selection inside :func:`_sparse_apply` so the
    determinism test (criterion 17) can read the indices directly — it is the
    selection the pool builder makes (ADR-012); a ``reindex`` layer instead
    re-scores the published pool, and ``reuse`` takes this one verbatim.
    """
    x32 = x.astype(jnp.float32)
    c = x32 @ params.W_c.astype(jnp.float32)
    if cfg.qat_kv_enabled:
        c = quant.fake_quant_mxfp4_latent(c)
    B, T = x.shape[:2]
    k_eff = min(int(cfg.mla_top_k), T)
    idx_q = (x32 @ params.W_idx_q.astype(jnp.float32)).reshape(B, T, cfg.mla_index_heads, cfg.mla_index_dim)
    idx_k = (c @ params.W_idx_k.astype(jnp.float32)).reshape(B, T, cfg.mla_index_heads, cfg.mla_index_dim)
    sc = jnp.einsum("bqhi,bshi->bqs", idx_q, idx_k) / cfg.mla_index_heads
    causal = jnp.arange(T)[None, :] <= jnp.arange(T)[:, None]
    sc = jnp.where(causal[None, :, :], sc, jnp.finfo(jnp.float32).min)
    if cfg.attn_topk_exact:
        # Same records as ``jax.lax.top_k``; see net/topk_exact.py for why the
        # exact selection is computed by tiling + merge instead of a full sort.
        return attn_sparse.topk_exact.topk_indices(sc, k_eff)
    return jax.lax.top_k(sc, k_eff)[1]


def reference_full_kv(params: MLAParams, cfg: ModelConfig, x: jnp.ndarray, mask: jnp.ndarray | None = None) -> jnp.ndarray:
    """Reference: the same attention with *uncompressed* keys/values.

    ``k = x @ (W_c W_k_up)`` and ``v = x @ (W_c W_v_up)`` — mathematically
    identical to the latent path, used as the parity oracle.
    """
    H, dq = cfg.num_heads, cfg.mla_head_dim
    x32 = x.astype(jnp.float32)
    W_k = (params.W_c @ params.W_k_up).astype(jnp.float32)
    W_v = (params.W_c @ params.W_v_up).astype(jnp.float32)
    q = (x32 @ params.W_q.astype(jnp.float32)).reshape(*x.shape[:-1], H, dq)
    k = (x32 @ W_k).reshape(*x.shape[:-1], H, dq)
    v = (x32 @ W_v).reshape(*x.shape[:-1], H, dq)
    scale = 1.0 / jnp.sqrt(dq)
    scores = jnp.einsum("bthd,bshd->bhts", q, k) * scale
    if mask is None:
        mask = _causal_mask(x.shape[-2])
    scores = jnp.where(mask, scores, jnp.finfo(jnp.float32).min)
    attn = jax.nn.softmax(scores, axis=-1)
    o = jnp.einsum("bhts,bshd->bthd", attn, v).reshape(*x.shape[:-1], H * dq)
    gate = jax.nn.sigmoid(x32 @ params.W_g.astype(jnp.float32))
    return (gate * o) @ params.W_o.astype(jnp.float32)
