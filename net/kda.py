"""KDA — Kimi Delta Attention (kda-formulas.md section 1).

Equivalent forms, all implemented here and checked against each other in the
parity tests:

* **recurrent** — ``lax.scan`` over tokens, carrying the per-head state
  ``S in R^{dk x dv}`` and the ShortConv buffers (streaming, O(1) memory).
* **chunked** — ``lax.scan`` over chunks, parallel within each chunk via
  ``lax.associative_scan`` over the affine delta-rule transition monoid
  (materialises each token's ``(dk, dk)`` transition — the memory-bound form).
* **wyut** — WY representation + UT transform: the intra-chunk object is a
  ``C x C`` score matrix, the inter-chunk transfer a matmul (ADR-031 delta A).
* **chunked_cc** — the ADR-047 rewrite of the WY/UT form: the ``C x C`` scores
  are built tile-wise, so the ``(C, C, dk)`` decay-ratio tensor that dominated
  the ``wyut`` form's memory never exists.

The recurrence (Eq. 1), parameterisation (Eq. 2), lower-bounded decay (Eq. 5)
and full-rank output gate (Eq. 6) follow the source verbatim.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import ModelConfig
from .remat import DEFAULT_REMAT_POLICY, remat_checkpoint
from . import attn_sparse, compute_dtype, quant
from .norm import headwise_rms_norm, l2_norm, swish
from .shortconv import short_conv, short_conv_step

DEFAULT_EPS = 1e-6


def _remat_policy(cfg: ModelConfig) -> str:
    """Объявленная политика рематериализации (ADR-049) — ``none`` по умолчанию.

    ``getattr`` держит совместимость с cfg-подобными объектами, собранными в
    коде (тесты, приборы) до появления поля.  Молчаливым остаётся только
    *отсутствие* объявления, которое и означает ``none``; неизвестное значение
    падает в :func:`net.remat.remat_checkpoint` (fail-closed) — подмены политики
    не происходит.
    """
    return getattr(cfg, "remat_policy", DEFAULT_REMAT_POLICY)


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
    qp = compute_dtype.gemm(x, params.W_q)  # (..., H*dk)
    kp = compute_dtype.gemm(x, params.W_k)  # (..., H*dk)
    vp = compute_dtype.gemm(x, params.W_v)  # (..., H*dv)
    beta = jax.nn.sigmoid(compute_dtype.gemm(x, params.W_beta))  # (..., H)
    z = compute_dtype.gemm(
        compute_dtype.gemm(x, params.W_a_down), params.W_a_up
    ) + params.b_alpha  # (..., H*dk)
    z = z.reshape(*z.shape[:-1], H, dk)  # (..., H, dk)
    # lower-bounded log-decay (Eq. 5): g = g_min * sigmoid(exp(A) * z)
    g = cfg.kda_g_min * jax.nn.sigmoid(jnp.exp(params.A)[..., None] * z)  # (..., H, dk)
    alpha = jnp.exp(g)  # (..., H, dk), in (e^gmin, 1)
    gate = jax.nn.sigmoid(compute_dtype.gemm(x, params.W_g))  # (..., hid)
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
    return compute_dtype.gemm(gate * o, params.W_o)


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
        qc = short_conv(compute_dtype.gemm(x, params.W_swa_q), params.conv_swa_q)
        kc = short_conv(compute_dtype.gemm(x, params.W_swa_k), params.conv_swa_k)
        vc = short_conv(compute_dtype.gemm(x, params.W_swa_v), params.conv_swa_v)
        gate = jax.nn.sigmoid(compute_dtype.gemm(x, params.W_g))
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

    # State algebra, deliberately NOT gated by ``AXIOM_COMPUTE_DTYPE``: the
    # recurrent state is an accumulator, and the recipe (``net/compute_dtype.py``)
    # keeps accumulators fp32 — ``net/kda.py``'s ``_combine`` below is the same
    # decision on the associative-scan side.  The parameter/attention
    # contractions above (``_project``, ``_output_gate``) are gated.
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


# ---------------------------------------------------------------------------
# Chunked (WY representation + UT transform) form — ADR-031 delta A
# ---------------------------------------------------------------------------


def _log_cumulative_decay(alpha: jnp.ndarray) -> jnp.ndarray:
    """``G`` with ``G_{i} = log(prod_{r<=i} alpha_r)`` (Eq. 3's log-gamma)."""
    return jnp.cumsum(jnp.log(alpha), axis=0)


def _decay_ratio_exp(log_g: jnp.ndarray) -> jnp.ndarray:
    """Causal decay ratio ``exp(min(G_c - G_i, 0))`` for all ``(c, i)``.

    ``G`` is the cumulative log-decay (non-increasing along the chunk, since
    ``alpha <= 1``), so on the kept side ``c >= i`` the exponent is already
    ``<= 0`` and the clamp is inert.  The clamp only touches the masked
    ``c < i`` entries, where the source's ``Gamma_c / Gamma_i`` would overflow:
    this is the numerical-safety point of the whole form — ``Gamma`` underflows
    to zero within a 64-token chunk, so the reciprocal ``1/Gamma`` (the form the
    paper's Eq. 9 writes) is ``inf`` on real keys, not just on padding.

    Returns ``(H, C, C, dk)``: ``[h, c, i, d] = exp(min(G_c - G_i, 0))``.
    """
    g = log_g.transpose(1, 0, 2)  # (H, C, dk)
    diff = g[:, :, None, :] - g[:, None, :, :]  # (H, C, C, dk)
    return jnp.exp(jnp.minimum(diff, 0.0))


def _ut_solve(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """``T b`` for ``T = (I + L)^{-1}`` without ever forming ``T``.

    ``a = I + L`` is unit lower triangular in the last two axes and batched over
    heads; ``b`` is ``(H, C, d)``.  The WY/UT form needs the products
    ``T diag(beta) Gamma K`` and ``T diag(beta) V`` (Eq. 7), i.e. the solutions
    of ``(I + L) W = X`` and ``(I + L) U = V`` — the triangular system is solved
    directly (one batched TRSM for all heads) instead of inverting ``I + L``;
    both products go through :func:`_ut_solve_pair` so they share one call.
    ``jnp.linalg.inv`` factorises each head's matrix by LU with pivoting
    (``getrf`` + ``getri``), whose per-head kernels dominated the step profile;
    the unpivoted batched solve removes that family from the graph without
    touching the algebra.
    """
    return jax.lax.linalg.triangular_solve(a, b, left_side=True, lower=True)


def _ut_solve_pair(
    a: jnp.ndarray, xw: jnp.ndarray, vw: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(T xw, T vw)`` for ``T = (I + L)^{-1}`` in a **single** batched TRSM.

    Eq. 7 asks for two products against the *same* unit lower triangular
    ``a = I + L``: ``W = T diag(beta) (Gamma . K)`` and ``U = T diag(beta) V``.
    Written as two :func:`_ut_solve` calls they became two TRSM invocations per
    chunk-step, and G1 measured what XLA makes of a batched TRSM at this
    geometry: it does not fuse it into one large kernel but cuts it into
    ``batch_trsm_left_kernel<..., 64, 4, ...>`` tiles — ``nrhs`` sliced down to
    4 — 115 200 TRSM kernels plus 230 400 ``MakeBatchPointers`` in 120 s of the
    steady phase, GPU occupancy 3.5% (``evidence/mfu-55/G1/REPORT.md`` §2-3).

    The triangular solve is column-wise: the ``nrhs`` columns solved against one
    matrix share nothing but that matrix.  Concatenating the two right-hand sides
    along the last axis therefore changes neither the algebra nor which equation
    each column solves — only the tile blocking inside the kernel, and with it
    the last-ulp rounding (measured on CPU: bit-identical at the smoke geometry,
    ``max|delta|`` 4.8e-07 on a synthetic 32x32 system — three orders below the
    ``1e-5`` the ADR-040 Amendment allows the re-written forms against the
    pre-port tree).  What it does buy is half the
    solve calls and twice the ``nrhs`` the kernel is handed — the address G1 left
    open.  The split is pure indexing — ``[..., :dk]`` and ``[..., dk:]`` of the
    one solution are the two blocks ``T xw`` and ``T vw``, with no arithmetic in
    between; the pinning test checks them against the two separate solves.

    No second right-hand side is available here, so ``dk + dv`` is the widest
    ``nrhs`` one TRSM can be given without changing the algebra: the matrices are
    per-head, and folding the heads into a single matrix means a block-diagonal
    ``(H*C, H*C)`` system — ``H`` times the flops of the batched block-triangular
    one.  Option (2) of the task, a ``(C, H*d)`` right-hand side against one
    ``(C, C)`` matrix, is that same non-starter: the heads do not share ``I + L``
    (see the report; the batch axis over heads *is* the block structure, and it
    is already there).
    """
    dw = xw.shape[-1]
    wu = _ut_solve(a, jnp.concatenate([xw, vw], axis=-1))
    return wu[..., :dw], wu[..., dw:]


def wyut_chunk_step(
    params: KDAParams, cfg: ModelConfig, carry: KDAState, x: jnp.ndarray
) -> tuple[KDAState, jnp.ndarray]:
    """One chunk of ``(C, hidden)`` via the WY/UT chunkwise delta rule.

    Mirrors :func:`chunk_step` (same recurrence, same carry) but never
    materialises the per-token ``(dk, dk)`` transition matrices: the
    inter-chunk state transfer is a matmul, the intra-chunk correction is the
    ``C x C`` score matrix ``A = Tril((Gamma . Q)(K / Gamma)^T)``.
    """
    C = x.shape[0]
    proj = _project(params, cfg, x)
    qp, kp, vp = proj["qp"], proj["kp"], proj["vp"]

    qc, q_buf = _short_conv_chunk(qp, params.conv_q, carry.q_buf)
    kc, k_buf = _short_conv_chunk(kp, params.conv_k, carry.k_buf)
    vc, v_buf = _short_conv_chunk(vp, params.conv_v, carry.v_buf)
    q, k, v = _postprocess(qc, kc, vc, cfg)

    beta = proj["beta"]  # (C, H)
    alpha = proj["alpha"]  # (C, H, dk)

    log_g = _log_cumulative_decay(alpha)  # (C, H, dk) = log gamma^i
    e = _decay_ratio_exp(log_g)  # (H, C, C, dk)
    q, k, v = q.transpose(1, 0, 2), k.transpose(1, 0, 2), v.transpose(1, 0, 2)
    beta_t = beta.transpose(1, 0)  # (H, C)

    # Score matrices: A_{c,i} = q_c . diag(ratio) . k_i, and the same with k_c.
    aqk = jnp.einsum("hcid,hcd,hid->hci", e, q, k)  # (H, C, C)
    akk = jnp.einsum("hcid,hcd,hid->hci", e, k, k)  # (H, C, C)
    lower = jnp.tril(jnp.ones((C, C), dtype=bool))
    strict = jnp.tril(jnp.ones((C, C), dtype=bool), k=-1)
    aqk = jnp.where(lower, aqk, 0.0)
    l_mat = jnp.where(strict, akk, 0.0) * beta_t[:, :, None]  # L_{r,i}=beta_r Akk
    tri = jnp.eye(C, dtype=l_mat.dtype) + l_mat  # (I + L), unit lower triangular

    # W = T Diag(beta) (Gamma . K), U = T Diag(beta) V  (Eq. 7).  One TRSM for
    # both right-hand sides — same matrix, concatenated columns (see
    # :func:`_ut_solve_pair`); G1 measured the two-call form as a storm of
    # 115 200 small TRSM kernels per 120 s of the steady phase.
    gamma = jnp.exp(log_g).transpose(1, 0, 2)  # (H, C, dk)
    xw = (gamma * k) * beta_t[:, :, None]  # (H, C, dk)
    vw = v * beta_t[:, :, None]  # (H, C, dv)
    w, u = _ut_solve_pair(tri, xw, vw)

    s_in = carry.S  # (H, dk, dv)
    v_tilde = u - jnp.einsum("hcd,hde->hce", w, s_in)  # U - W S  (pseudo-value)

    # Output (Eq. 9): inter-chunk from S, intra-chunk from the score matrix.
    gamma_q = gamma * q  # (H, C, dk) = Gamma . Q
    inter = jnp.einsum("hcd,hde->hce", gamma_q, s_in)
    intra = jnp.einsum("hci,hie->hce", aqk, v_tilde)
    o = (inter + intra).transpose(1, 0, 2)  # (C, H, dv)

    # State transfer (Eq. 8): decay the carried state, absorb the chunk writes.
    gamma_c = gamma[:, -1, :]  # (H, dk) = gamma^C
    # gamma^{i+1->C} = gamma^C / gamma^i, evaluated as a log difference clamped
    # above at 0 (both factors are <= 1 for i < C, so the clamp is inert there).
    lam = jnp.exp(jnp.minimum(log_g[-1][None] - log_g, 0.0))  # (C, H, dk)
    lam = lam.transpose(1, 0, 2)  # (H, C, dk), all entries <= 1
    y = lam * k  # (H, C, dk)
    s_new = gamma_c[:, :, None] * s_in + jnp.einsum("hcd,hce->hde", y, v_tilde)

    out = _output_gate(o, proj["gate"], params)
    return KDAState(s_new, q_buf, k_buf, v_buf), out


def apply_wyut(
    params: KDAParams,
    cfg: ModelConfig,
    x: jnp.ndarray,
    chunk_size: int | None = None,
) -> jnp.ndarray:
    """KDA over ``(T, hidden)`` in chunks, WY representation + UT transform.

    ``chunk_size=None`` reads ``cfg.kda_wyut_chunk`` (the declarative default).
    Semantics are identical to :func:`apply_recurrent` / :func:`apply_chunked`
    (the same recurrence, Eq. 1); this is the memory-lean formulation —
    ``O(T)`` chunk carries and matmuls, no per-token ``(dk, dk)`` kit.
    """
    if chunk_size is None:
        chunk_size = cfg.kda_wyut_chunk
    T = x.shape[0]
    C = chunk_size
    n_chunks = (T + C - 1) // C
    pad = n_chunks * C - T
    x_p = jnp.pad(x, ((0, pad), (0, 0))) if pad else x
    x_chunks = x_p.reshape(n_chunks, C, -1)
    carry0 = init_state(cfg)

    _, out = jax.lax.scan(
        lambda c, xc: wyut_chunk_step(params, cfg, c, xc), carry0, x_chunks
    )
    out = out.reshape(n_chunks * C, -1)
    return _with_window(out[:T], params, cfg, x)


# ---------------------------------------------------------------------------
# Chunked (C x C intra-chunk, tile-wise decay factors) form — ADR-047
# ---------------------------------------------------------------------------

#: Largest row-tile width used to build the intra-chunk ``C x C`` scores.  The
#: column factor inside a tile is ``exp(G_p - G_i)`` (``p`` — the tile's first
#: position), i.e. at most ``exp(|g_min| * (tile - 1))``; the tile width is
#: bounded so that this never leaves the fp32/bf16 range (~3.4e38, ``ln`` ~ 88.7).
_CC_TILE_MAX = 16

#: Exponent budget the tile width has to fit into (``ln`` of the fp32 ceiling,
#: with a margin of one decimal order of magnitude).
_CC_EXP_BUDGET = 80.0


def _cc_tile(cfg: ModelConfig) -> int:
    """Row-tile width for the intra-chunk decay factors (``1 <= tile <= 16``)."""
    step = abs(float(cfg.kda_g_min))
    cap = int(_CC_EXP_BUDGET / step) if step > 0 else _CC_TILE_MAX
    return max(1, min(_CC_TILE_MAX, cap))


def _cc_scores(
    row: jnp.ndarray,
    col: jnp.ndarray,
    log_g: jnp.ndarray,
    tile: int,
    strict: bool,
) -> jnp.ndarray:
    """Decay-weighted ``C x C`` scores ``sum_d exp(G_c - G_i) row_c[d] col_i[d]``.

    ``row``/``col``/``log_g`` are ``(H, C, dk)`` (the per-head layout of
    :func:`wyut_chunk_step`); ``G`` is the chunk's cumulative log-decay, which is
    non-increasing along the position axis (``alpha <= 1`` per channel).

    The plain factorisation ``(row * exp(G)) (col * exp(-G))^T`` overflows:
    ``exp(-G)`` reaches ``e^320`` inside a 64-token chunk and the reciprocal of
    the underflowed ``exp(G)`` is ``inf``.  Here the chunk is cut into row tiles
    of ``tile`` positions and each tile measures its decay against its **own
    first position** ``p``, so that::

        exp(G_c - G_i) = exp(G_c - G_p) * exp(G_p - G_i)

    is a product of two representable factors — the row one is ``<= 1`` for
    every ``c`` of the tile, and the column one is ``<= 1`` for every position
    before the tile while inside the tile it is bounded by
    ``exp(|g_min| * (tile - 1))``: a constant of the model, not of the chunk
    width.  Nothing is ever divided by a decaying quantity, so real keys cannot
    produce ``inf``/``NaN`` (padding underflows to ``0``, where the contribution
    is negligible anyway).  Mask keeps ``c >= i`` (``strict``: ``c > i``).

    Returns ``(H, C, C)`` — the intra-chunk object itself, never ``(H, C, C, dk)``.
    """
    H, C, _ = row.shape
    out = jnp.zeros((H, C, C), dtype=row.dtype)
    for p in range(0, C, tile):
        b = min(tile, C - p)
        ref = log_g[:, p, :]  # (H, dk) = G_p
        rows = row[:, p : p + b, :] * jnp.exp(log_g[:, p : p + b, :] - ref[:, None, :])
        cols = col[:, : p + b, :] * jnp.exp(ref[:, None, :] - log_g[:, : p + b, :])
        blk = jnp.einsum("hbd,hjd->hbj", rows, cols)  # (H, b, p+b)
        row_ix = jnp.arange(p, p + b)[:, None]
        col_ix = jnp.arange(p + b)[None, :]
        keep = (row_ix > col_ix) if strict else (row_ix >= col_ix)
        out = out.at[:, p : p + b, : p + b].set(jnp.where(keep, blk, 0.0))
    return out


def cc_chunk_step(
    params: KDAParams, cfg: ModelConfig, carry: KDAState, x: jnp.ndarray
) -> tuple[KDAState, jnp.ndarray]:
    """One chunk of ``(C, hidden)``: ``C x C`` intra-chunk, ``dk x dv`` state.

    Same recurrence (Eq. 1), same WY/UT algebra and the same carry as
    :func:`wyut_chunk_step`; only the two score matrices are built tile-wise by
    :func:`_cc_scores`, so the ``(H, C, C, dk)`` decay-ratio tensor of the
    ``wyut`` form — the allocation that made it heavier than ``chunked`` — is
    never materialised.  The intra-chunk object is the ``C x C`` matrix itself:
    ``O(C^2)`` per head, on the tensor cores.

    Compute dtype (``AXIOM_COMPUTE_DTYPE``, see ``net/compute_dtype.py``): the
    parameter projections and the output gate below are the *same* gated helpers
    every other form uses (:func:`_project`, :func:`_output_gate`), so this form
    inherits the gate — bf16 GEMM operands with an fp32 accumulator — without a
    second implementation.  The intra-chunk algebra of :func:`_cc_scores` and of
    the WY/UT (``tri``/``w``/``u``/``v_tilde``) steps stays fp32 on purpose,
    the same decision the committed ``chunk_step`` records for the scan side: it
    is accumulator state algebra, and the recipe keeps accumulators fp32.  The
    ``wyut`` form's analogous score einsums are ungated for the same reason, so
    the two ADR-047 arms remain comparable.
    """
    C = x.shape[0]
    proj = _project(params, cfg, x)
    qp, kp, vp = proj["qp"], proj["kp"], proj["vp"]

    qc, q_buf = _short_conv_chunk(qp, params.conv_q, carry.q_buf)
    kc, k_buf = _short_conv_chunk(kp, params.conv_k, carry.k_buf)
    vc, v_buf = _short_conv_chunk(vp, params.conv_v, carry.v_buf)
    q, k, v = _postprocess(qc, kc, vc, cfg)

    beta = proj["beta"]  # (C, H)
    alpha = proj["alpha"]  # (C, H, dk)

    log_g = _log_cumulative_decay(alpha)  # (C, H, dk)
    q_t = q.transpose(1, 0, 2)  # (H, C, dk)
    k_t = k.transpose(1, 0, 2)
    v_t = v.transpose(1, 0, 2)
    log_g_t = log_g.transpose(1, 0, 2)
    beta_t = beta.transpose(1, 0)  # (H, C)

    tile = _cc_tile(cfg)
    aqk = _cc_scores(q_t, k_t, log_g_t, tile, strict=False)  # (H, C, C)
    akk = _cc_scores(k_t, k_t, log_g_t, tile, strict=True)  # (H, C, C)

    # L = strict_tril(diag(beta) Akk); the UT transform is applied as the solve
    # (I + L) W = X, never as a materialised T = (I + L)^{-1} (see :func:`_ut_solve`).
    l_mat = akk * beta_t[:, :, None]
    tri = jnp.eye(C, dtype=l_mat.dtype) + l_mat  # (I + L), unit lower triangular

    # W = T diag(beta) (Gamma . K), U = T diag(beta) V  (Eq. 7).  One TRSM for
    # both right-hand sides — same matrix, concatenated columns (see
    # :func:`_ut_solve_pair`).
    gamma = jnp.exp(log_g).transpose(1, 0, 2)  # (H, C, dk) = Gamma_c
    xw = (gamma * k_t) * beta_t[:, :, None]
    vw = v_t * beta_t[:, :, None]
    w, u = _ut_solve_pair(tri, xw, vw)

    s_in = carry.S  # (H, dk, dv)
    v_tilde = u - jnp.einsum("hcd,hde->hce", w, s_in)  # U - W S (pseudo-value)

    # Output (Eq. 9): inter-chunk from S, intra-chunk from the score matrix.
    gamma_q = gamma * q_t  # Gamma . Q
    inter = jnp.einsum("hcd,hde->hce", gamma_q, s_in)
    intra = jnp.einsum("hci,hie->hce", aqk, v_tilde)
    o = (inter + intra).transpose(1, 0, 2)  # (C, H, dv)

    # State transfer (Eq. 8): decay the carried state, absorb the chunk writes.
    gamma_c = gamma[:, -1, :]  # (H, dk) = Gamma^C
    lam = jnp.exp(jnp.minimum(log_g[-1][None] - log_g, 0.0)).transpose(1, 0, 2)
    y = lam * k_t
    s_new = gamma_c[:, :, None] * s_in + jnp.einsum("hcd,hce->hde", y, v_tilde)

    out = _output_gate(o, proj["gate"], params)
    return KDAState(s_new, q_buf, k_buf, v_buf), out


def apply_chunked_cc(
    params: KDAParams,
    cfg: ModelConfig,
    x: jnp.ndarray,
    chunk_size: int | None = None,
) -> jnp.ndarray:
    """KDA over ``(T, hidden)`` with a ``C x C`` intra-chunk matrix (ADR-047).

    The target structure of ADR-047 p. 2 — inside a chunk a masked ``C x C``
    attention-like matrix built by :func:`_cc_scores`; between chunks the compact
    ``dk x dv`` state, transferred by matmul.  No per-token ``(dk, dk)``
    transition matrix (``chunked``) and no ``(C, C, dk)`` decay-ratio tensor
    (``wyut``).  ``chunk_size=None`` reads ``cfg.kda_wyut_chunk`` — the same
    declarative width the WY/UT form uses.  Semantics are identical to
    :func:`apply_recurrent` (parity pinned by ``test_kda_chunked_cc.py``).
    """
    if chunk_size is None:
        chunk_size = cfg.kda_wyut_chunk
    T = x.shape[0]
    C = chunk_size
    n_chunks = (T + C - 1) // C
    pad = n_chunks * C - T
    x_p = jnp.pad(x, ((0, pad), (0, 0))) if pad else x
    x_chunks = x_p.reshape(n_chunks, C, -1)
    carry0 = init_state(cfg)

    def body(carry: KDAState, xc: jnp.ndarray):
        return cc_chunk_step(params, cfg, carry, xc)

    if cfg.kda_chunked_backward:
        # Recompute each chunk from its saved carry during backward instead of
        # retaining the per-chunk score matrices and tiles.  ADR-049: *what*
        # stays saved inside that boundary is the declared ``remat_policy``
        # (``none`` — the pre-ADR-049 bar, applied by ``remat_checkpoint``
        # without a ``policy`` argument).
        body = remat_checkpoint(body, _remat_policy(cfg))
    _, out = jax.lax.scan(body, carry0, x_chunks)
    out = out.reshape(n_chunks * C, -1)
    return _with_window(out[:T], params, cfg, x)


def apply_kda(
    params: KDAParams,
    cfg: ModelConfig,
    x: jnp.ndarray,
    chunk_size: int | None = None,
) -> jnp.ndarray:
    """Dispatch on ``cfg.kda_impl`` (``chunked`` — the regression reference).

    ``chunked`` is the pre-delta path, byte-for-byte as before; ``wyut`` is the
    ADR-031 delta-A form; ``chunked_cc`` is the ADR-047 tile-wise ``C x C`` form.
    ``chunked`` honours the caller's ``chunk_size``; the other two fall back to
    ``cfg.kda_wyut_chunk`` when it is ``None``.
    """
    if cfg.kda_impl == "wyut":
        return apply_wyut(params, cfg, x, chunk_size=chunk_size or cfg.kda_wyut_chunk)
    if cfg.kda_impl == "chunked_cc":
        return apply_chunked_cc(params, cfg, x, chunk_size=chunk_size or cfg.kda_wyut_chunk)
    return apply_chunked(params, cfg, x, chunk_size=chunk_size or cfg.kda_wyut_chunk)


def apply_chunked(
    params: KDAParams, cfg: ModelConfig, x: jnp.ndarray, chunk_size: int
) -> jnp.ndarray:
    """Run KDA attention over ``(T, hidden)`` in chunks of ``chunk_size``.

    ``cfg.kda_chunked_backward`` selects the backward-pass granularity of the
    chunk loop (D-8 remainder).  Off, the scan body is built exactly as before
    and the graph is bit-for-bit the pre-delta one.  On, the body is wrapped in
    ``jax.checkpoint``, so the affine-transition prefix ``P``/``Q`` of a chunk
    (``chunk_step``'s ``associative_scan`` — the O(T·H·dk·dk) kit) is recomputed
    from the chunk's carry during backward instead of being retained for every
    chunk.  This is the FLA-style chunked delta-rule backward; the forward pass
    and its numbers are unchanged either way, only what autodiff keeps differs.
    """
    T = x.shape[0]
    C = chunk_size
    n_chunks = (T + C - 1) // C
    pad = n_chunks * C - T
    x_p = jnp.pad(x, ((0, pad), (0, 0))) if pad else x
    x_chunks = x_p.reshape(n_chunks, C, -1)
    carry0 = init_state(cfg)

    def body(carry: KDAState, xc: jnp.ndarray):
        return chunk_step(params, cfg, carry, xc)

    if cfg.kda_chunked_backward:
        # Recompute each chunk's trajectory (M, N, P, Q and projections) from
        # its saved carry in the backward pass rather than retaining them.
        # ADR-049: the declared ``remat_policy`` chooses what stays saved
        # inside this boundary (``none`` reproduces the pre-ADR-049 graph).
        body = remat_checkpoint(body, _remat_policy(cfg))
    _, out = jax.lax.scan(body, carry0, x_chunks)
    out = out.reshape(n_chunks * C, -1)
    return _with_window(out[:T], params, cfg, x)
