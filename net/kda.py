"""KDA — Kimi Delta Attention (kda-formulas.md section 1).

Two equivalent forms, both implemented here and checked against each other in
the parity tests:

* **recurrent** — ``lax.scan`` over tokens, carrying the per-head state
  ``S in R^{dk x dv}`` and the ShortConv buffers (streaming, O(1) memory).
* **chunked** — ``lax.scan`` over chunks, parallel within each chunk via
  ``lax.associative_scan`` over the affine delta-rule transition monoid.

The recurrence (Eq. 1), parameterisation (Eq. 2), lower-bounded decay (Eq. 5)
and full-rank output gate (Eq. 6) follow the source verbatim.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import ModelConfig
from . import attn_sparse, quant
from .norm import headwise_rms_norm, l2_norm, swish
from .shortconv import short_conv, short_conv_step

DEFAULT_EPS = 1e-6


def _rand(key, shape, scale: float) -> jnp.ndarray:
    return jax.random.normal(key, shape) * scale


class KDAState(NamedTuple):
    """Carried state for one KDA layer (both forms share this structure)."""

    S: jnp.ndarray  # (heads, dk, dv)
    q_buf: jnp.ndarray  # (K-1, heads*dk)
    k_buf: jnp.ndarray  # (K-1, heads*dk)
    v_buf: jnp.ndarray  # (K-1, heads*dv)


class KDAParams(NamedTuple):
    W_q: jnp.ndarray
    W_k: jnp.ndarray
    W_v: jnp.ndarray
    W_o: jnp.ndarray
    W_g: jnp.ndarray
    W_beta: jnp.ndarray
    W_a_down: jnp.ndarray
    W_a_up: jnp.ndarray
    b_alpha: jnp.ndarray
    A: jnp.ndarray
    conv_q: jnp.ndarray
    conv_k: jnp.ndarray
    conv_v: jnp.ndarray
    # --- delta v1.5 (ADR-009 D1) -----------------------------------------
    swa_logit: jnp.ndarray  # () learnable window share
    # ``None`` unless ``swa_share_kda_projections`` is False (then dedicated
    # window projections/conv are allocated instead).  ``None`` is an empty
    # pytree node, so it costs no parameters and Orbax skips it (0-size arrays
    # are rejected by the checkpoint writer).
    W_swa_q: jnp.ndarray | None
    W_swa_k: jnp.ndarray | None
    W_swa_v: jnp.ndarray | None
    conv_swa_q: jnp.ndarray | None
    conv_swa_k: jnp.ndarray | None
    conv_swa_v: jnp.ndarray | None


def init_kda(key, cfg: ModelConfig) -> KDAParams:
    """Initialise a KDA layer's parameters.

    ``A`` (per-head log-scale) is initialised to zero per Eq. 5.  When
    ``swa_share_kda_projections`` is False (the rejected-by-default variant of
    ADR-009 D1) the window gets dedicated projections; otherwise those fields
    are zero-size and the window reuses the delta-rule q/k/v (0 matrix params).
    """
    H, dk, dv = cfg.num_heads, cfg.kda_dk, cfg.kda_dv
    hid = cfg.hidden
    r = cfg.kda_decay_rank
    K = cfg.kda_short_conv_kernel
    k1, k2, k3, k4, k5, k6, k7, k8, k9, kA, kc, ks = jax.random.split(key, 12)

    if cfg.swa_share_kda_projections:
        W_swa_q = W_swa_k = W_swa_v = None
        conv_swa_q = conv_swa_k = conv_swa_v = None
    else:
        ks1, ks2, ks3, ks4, ks5, ks6 = jax.random.split(ks, 6)
        W_swa_q = _rand(ks1, (hid, H * dk), 0.02)
        W_swa_k = _rand(ks2, (hid, H * dk), 0.02)
        W_swa_v = _rand(ks3, (hid, H * dv), 0.02)
        conv_swa_q = _rand(ks4, (K, H * dk), 0.1)
        conv_swa_k = _rand(ks5, (K, H * dk), 0.1)
        conv_swa_v = _rand(ks6, (K, H * dv), 0.1)

    return KDAParams(
        W_q=_rand(k1, (hid, H * dk), 0.02),
        W_k=_rand(k2, (hid, H * dk), 0.02),
        W_v=_rand(k3, (hid, H * dv), 0.02),
        W_o=_rand(k4, (H * dv, hid), 0.02),
        W_g=_rand(k5, (hid, hid), 0.02),
        W_beta=_rand(k6, (hid, H), 0.02),
        W_a_down=_rand(k7, (hid, r), 0.02),
        W_a_up=_rand(k8, (r, H * dk), 0.02),
        b_alpha=jnp.zeros((H * dk,)),
        A=jnp.zeros((H,)),
        conv_q=_rand(kc, (K, H * dk), 0.1),
        conv_k=_rand(k9, (K, H * dk), 0.1),
        conv_v=_rand(kA, (K, H * dv), 0.1),
        swa_logit=jnp.zeros(()),
        W_swa_q=W_swa_q,
        W_swa_k=W_swa_k,
        W_swa_v=W_swa_v,
        conv_swa_q=conv_swa_q,
        conv_swa_k=conv_swa_k,
        conv_swa_v=conv_swa_v,
    )


def init_state(cfg: ModelConfig) -> KDAState:
    H, dk, dv = cfg.num_heads, cfg.kda_dk, cfg.kda_dv
    K = cfg.kda_short_conv_kernel
    return KDAState(
        S=jnp.zeros((H, dk, dv)),
        q_buf=jnp.zeros((K - 1, H * dk)),
        k_buf=jnp.zeros((K - 1, H * dk)),
        v_buf=jnp.zeros((K - 1, H * dv)),
    )


# ---------------------------------------------------------------------------
# Shared per-token projections (used identically by both forms)
# ---------------------------------------------------------------------------


def _project(params: KDAParams, cfg: ModelConfig, x: jnp.ndarray) -> dict:
    """Project the (pre-normed) input into per-head q, k, v, beta, alpha, gate.

    ``x`` may have shape (..., hidden); all outputs keep the leading dims and
    expose a final ``(heads, d)`` axis.
    """
    H, dk, dv = cfg.num_heads, cfg.kda_dk, cfg.kda_dv
    qp = x @ params.W_q  # (..., H*dk)
    kp = x @ params.W_k  # (..., H*dk)
    vp = x @ params.W_v  # (..., H*dv)
    beta = jax.nn.sigmoid(x @ params.W_beta)  # (..., H)
    z = (x @ params.W_a_down) @ params.W_a_up + params.b_alpha  # (..., H*dk)
    z = z.reshape(*z.shape[:-1], H, dk)  # (..., H, dk)
    # lower-bounded log-decay (Eq. 5): g = g_min * sigmoid(exp(A) * z)
    g = cfg.kda_g_min * jax.nn.sigmoid(jnp.exp(params.A)[..., None] * z)  # (..., H, dk)
    alpha = jnp.exp(g)  # (..., H, dk), in (e^gmin, 1)
    gate = jax.nn.sigmoid(x @ params.W_g)  # (..., hid)
    return {
        "qp": qp,
        "kp": kp,
        "vp": vp,
        "beta": beta,
        "alpha": alpha,
        "gate": gate,
    }


def _postprocess(qp, kp, vp, cfg: ModelConfig) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """ShortConv -> Swish -> (L2Norm for q/k), then reshape to per-head."""
    H, dk, dv = cfg.num_heads, cfg.kda_dk, cfg.kda_dv
    q = swish(qp)
    k = swish(kp)
    v = swish(vp)
    q = l2_norm(q.reshape(*q.shape[:-1], H, dk))
    k = l2_norm(k.reshape(*k.shape[:-1], H, dk))
    v = v.reshape(*v.shape[:-1], H, dv)
    return q, k, v


def _output_gate(o: jnp.ndarray, gate: jnp.ndarray, params: KDAParams) -> jnp.ndarray:
    """Full-rank output gate (Eq. 6): W_o [sigmoid(W_g x) * RMSNorm(o)].

    ``o`` is the recurrent output in per-head space (..., H, dv); it is
    head-wise RMS-normalised, flattened, gated and projected to ``hidden``.
    """
    o = headwise_rms_norm(o)  # (..., H, dv)
    o = o.reshape(*o.shape[:-2], -1)  # (..., H*dv)
    return (gate * o) @ params.W_o


# ---------------------------------------------------------------------------
# Sliding-window branch (ADR-009 D1)
# ---------------------------------------------------------------------------


def _window_projection(params: KDAParams, cfg: ModelConfig, x: jnp.ndarray):
    """q/k/v (and the output gate) for the window branch.

    With ``swa_share_kda_projections=True`` (our decision, ADR-009 D1) the
    window reuses the q/k/v already computed by the linear layer of the
    delta rule — 0 new matrix parameters.  Otherwise dedicated projections are
    used (the source's shape, kept as the fallback).
    """
    if cfg.swa_share_kda_projections:
        proj = _project(params, cfg, x)
        qc = short_conv(proj["qp"], params.conv_q)
        kc = short_conv(proj["kp"], params.conv_k)
        vc = short_conv(proj["vp"], params.conv_v)
        gate = proj["gate"]
    else:
        qc = short_conv(x @ params.W_swa_q, params.conv_swa_q)
        kc = short_conv(x @ params.W_swa_k, params.conv_swa_k)
        vc = short_conv(x @ params.W_swa_v, params.conv_swa_v)
        gate = jax.nn.sigmoid(x @ params.W_g)
    q, k, v = _postprocess(qc, kc, vc, cfg)
    return q, k, v, gate


def window_output(params: KDAParams, cfg: ModelConfig, x: jnp.ndarray) -> jnp.ndarray:
    """Exact causal softmax over the last ``cfg.swa_window`` positions."""
    q, k, v, gate = _window_projection(params, cfg, x)
    # SWA KV stays FP8 (ADR-009 D3); the delta-rule copies above are untouched.
    kw = quant.fake_quant_fp8_e4m3(k)
    vw = quant.fake_quant_fp8_e4m3(v)
    o = attn_sparse.window_attention(q[None], kw[None], vw[None], int(cfg.swa_window))[0]
    return _output_gate(o, gate, params)


def _with_window(out: jnp.ndarray, params: KDAParams, cfg: ModelConfig, x: jnp.ndarray) -> jnp.ndarray:
    """Add the window branch to the recurrent output with a learnable share (D1).

    Disabled while the dense oracle is selected (``attn_dense_reference``, D4)
    or when the window is 0.
    """
    if cfg.attn_dense_reference or cfg.swa_window <= 0:
        return out
    return out + jax.nn.sigmoid(params.swa_logit) * window_output(params, cfg, x)


# ---------------------------------------------------------------------------
# Recurrent (streaming) form
# ---------------------------------------------------------------------------


def recurrent_step(
    params: KDAParams, cfg: ModelConfig, carry: KDAState, x: jnp.ndarray
) -> tuple[KDAState, jnp.ndarray]:
    """One token: update ShortConv buffers and delta-rule state, emit gated output."""
    H, dk, dv = cfg.num_heads, cfg.kda_dk, cfg.kda_dv
    proj = _project(params, cfg, x)
    qp, kp, vp = proj["qp"], proj["kp"], proj["vp"]

    qc, q_buf = short_conv_step(qp, params.conv_q, carry.q_buf)
    kc, k_buf = short_conv_step(kp, params.conv_k, carry.k_buf)
    vc, v_buf = short_conv_step(vp, params.conv_v, carry.v_buf)
    q, k, v = _postprocess(qc, kc, vc, cfg)

    beta = proj["beta"]  # (H,)
    alpha = proj["alpha"]  # (H, dk)
    S = _delta_step(carry.S, k, v, beta, alpha)
    o = jnp.einsum("hdv,hd->hv", S, q)  # (H, dv) = S^T q
    out = _output_gate(o, proj["gate"], params)
    return KDAState(S, q_buf, k_buf, v_buf), out


def _delta_step(S, k, v, beta, alpha):
    """S <- (I - beta k k^T) Diag(alpha) S + beta k v^T (Eq. 1).

    The forget term removes ``beta k (k^T Diag(alpha) S)``; the read
    ``k^T Diag(alpha) S = (alpha * k)^T S`` is against the *incoming* state S,
    not the decayed one.
    """
    decayed = alpha[..., None] * S  # Diag(alpha) S, row-wise channel decay
    ak = alpha * k  # (H, dk)
    read = jnp.einsum("hd,hdu->hu", ak, S)  # (H, dv) = (alpha*k)^T @ S
    forget = beta[:, None, None] * k[..., None] * read[:, None, :]  # (H, dk, dv)
    write = beta[:, None, None] * k[..., None] * v[:, None, :]  # (H, dk, dv)
    return decayed - forget + write


def apply_recurrent(params: KDAParams, cfg: ModelConfig, x: jnp.ndarray) -> jnp.ndarray:
    """Run the KDA attention over a sequence ``x`` of shape (T, hidden)."""
    carry0 = init_state(cfg)
    _, out = jax.lax.scan(lambda c, xt: recurrent_step(params, cfg, c, xt), carry0, x)
    return _with_window(out, params, cfg, x)  # (T, hidden)


# ---------------------------------------------------------------------------
# Chunked (parallel within chunk) form
# ---------------------------------------------------------------------------


def _delta_transitions(k, v, beta, alpha) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build the affine transition monoid elements ``(M, N)`` for one chunk.

    ``S_t = M_t @ S_{t-1} + N_t`` with
    ``M_t = Diag(alpha_t) - beta_t k_t (alpha_t * k_t)^T`` and ``N_t = beta_t k_t v_t^T``.
    """
    C = k.shape[0]
    D = k.shape[-1]  # dk
    eye = jnp.eye(D)
    M = alpha[..., None] * eye  # (C, H, dk, dk) diagonal part
    M = M - beta[..., None, None] * k[..., None] * (alpha * k)[..., None, :]
    N = beta[..., None, None] * k[..., None] * v[..., None, :]  # (C, H, dk, dv)
    return M, N


def _combine(a: tuple, b: tuple) -> tuple:
    """Compose two affine transitions: ``b after a`` => ``(b.M a.M, b.M a.N + b.N)``."""
    M1, N1 = a
    M2, N2 = b
    return M2 @ M1, M2 @ N1 + N2


def chunk_step(
    params: KDAParams, cfg: ModelConfig, carry: KDAState, x: jnp.ndarray
) -> tuple[KDAState, jnp.ndarray]:
    """One chunk of shape (C, hidden): parallel within the chunk via associative scan."""
    H, dk, dv = cfg.num_heads, cfg.kda_dk, cfg.kda_dv
    proj = _project(params, cfg, x)
    qp, kp, vp = proj["qp"], proj["kp"], proj["vp"]

    # ShortConv over the chunk with the incoming buffers (causal, exact).
    qc, q_buf = _short_conv_chunk(qp, params.conv_q, carry.q_buf)
    kc, k_buf = _short_conv_chunk(kp, params.conv_k, carry.k_buf)
    vc, v_buf = _short_conv_chunk(vp, params.conv_v, carry.v_buf)
    q, k, v = _postprocess(qc, kc, vc, cfg)

    beta = proj["beta"]  # (C, H)
    alpha = proj["alpha"]  # (C, H, dk)

    M, N = _delta_transitions(k, v, beta, alpha)  # (C, H, dk, dk), (C, H, dk, dv)
    P, Q = jax.lax.associative_scan(_combine, (M, N))  # prefix compositions S0 -> S_t

    # Output: o_t = (P_t S0 + Q_t)^T q_t = S0^T (P_t^T q_t) + (Q_t^T q_t)
    pq = jnp.einsum("chab,cha->chb", P, q)  # (C, H, dk) = P^T q
    inter = jnp.einsum("hdv,chd->chv", carry.S, pq)  # (C, H, dv) = S0^T (P^T q)
    intra = jnp.einsum("chuv,chu->chv", Q, q)  # (C, H, dv) = Q^T q
    o = inter + intra

    S_new = P[-1] @ carry.S + Q[-1]  # state after the chunk
    out = _output_gate(o, proj["gate"], params)
    return KDAState(S_new, q_buf, k_buf, v_buf), out


def _short_conv_chunk(x: jnp.ndarray, w: jnp.ndarray, buf: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Causal ShortConv over a chunk ``(C, D)`` given the incoming buffer."""
    k = w.shape[0]
    full = jnp.concatenate([buf, x], axis=0)  # (K-1+C, D)
    out = short_conv(full, w)[k - 1 :]  # (C, D)
    new_buf = full[x.shape[0] :]  # last K-1 inputs
    return out, new_buf


def apply_chunked(params: KDAParams, cfg: ModelConfig, x: jnp.ndarray, chunk_size: int) -> jnp.ndarray:
    """Run KDA attention over ``(T, hidden)`` in chunks of ``chunk_size``."""
    T = x.shape[0]
    C = chunk_size
    n_chunks = (T + C - 1) // C
    pad = n_chunks * C - T
    x_p = jnp.pad(x, ((0, pad), (0, 0))) if pad else x
    x_chunks = x_p.reshape(n_chunks, C, -1)
    carry0 = init_state(cfg)
    _, out = jax.lax.scan(lambda c, xc: chunk_step(params, cfg, c, xc), carry0, x_chunks)
    out = out.reshape(n_chunks * C, -1)
    return _with_window(out[:T], params, cfg, x)
