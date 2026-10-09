"""Stable LatentMoE block (kda-formulas.md section 4, §2.3 :416-589).

LatentMoE separates the full model width from the routed-expert width: shared
experts are the full-width path, routed experts operate in a compact latent
space ``latent`` (0.5 x hidden, K3: 3584 = 0.5 x 7168).  At the skeleton scale
we keep the qualitative proportion (sparse top-k routing + shared experts) and
choose the quantitative expert count / sizes ourselves — recorded in
``config.json`` ``deviations``.

Forward (Eq. 11, :462-467):

    u = sum_{i in T_k(x)} p_i E_i^routed(W_down x)
    y = sum_{j=1}^{Ns} E_j^shared(x) + W_up RMSNorm(u)

Router (§4.3, :545-547) is deterministic: ``s_i = Sigmoid(W_r x_i)``,
``T_i = argtopk(s_i + b)``, and the mixing weights ``p_{i,j} = s_{i,j} /
sum_{r in T_i} s_{i,r}`` (a softmax-normalised sigmoid over the selected
experts — the "softmax-роутинг" of MODEL-L3-SKELETON.md section 1).  The bias
``b`` shifts dispatch but does not enter the mixture weights.

Execution: the routed term is *sparse on the hardware too* — the selected
experts are gathered into ragged groups and only they are computed
(:func:`_routed_experts_grouped`).  The pre-refactor spelling materialised every
routed expert and gathered the top-k afterwards, so the executed FLOPs were
``n_routed`` expert applications per token for ``top_k`` of useful work
(``tools/mfu_ladder.py:flops_moe_ffn`` "executed" vs "active").  Sparsity is
numerically free (a sparse ``ragged_dot`` is bit-identical to a dense one over
all experts); only the einsum -> ragged lowering moves the result, within the
tolerance pinned by ``net/tests/test_32_moe_grouped_topk.py``.

QB — global-batch quantile balancing (Eq. 14, :578-583):

    b_hat_j <- -quantile_{1-k/n}(s_{:,j} - alpha)    alpha_i: Top-(k+1) cutoff
    b       <- b_hat - mean(b_hat)

The update acts from the next step (causal); the bias is frozen at inference.
The practical estimator is a histogram of the marginals via all-reduce
(:592-598); here, on a single device, we compute the exact per-batch quantile.
The auxiliary loss below (``qb_aux_loss``) is our monitoring load-balancing
term (the paper's QB is auxiliary-free); its gradient is stopped when it is
integrated into the training loss — recorded as our decision in ``config.json``
``deviations``.
"""

from __future__ import annotations

import os
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import ModelConfig
from . import compute_dtype
from . import mlp as mlp_mod
from .norm import rms_norm


def _rand(key, shape, scale: float) -> jnp.ndarray:
    return jax.random.normal(key, shape) * scale


class LatentMoEParams(NamedTuple):
    W_down: jnp.ndarray  # (hidden, latent) down-projection W_↓
    W_up: jnp.ndarray  # (latent, hidden) up-projection W_↑
    expert_g: jnp.ndarray  # (n_routed, latent, expert_inter) routed gate branch
    expert_u: jnp.ndarray  # (n_routed, latent, expert_inter) routed up branch
    expert_d: jnp.ndarray  # (n_routed, expert_inter, latent) routed down branch
    shared_g: jnp.ndarray  # (n_shared, hidden, shared_inter) shared gate branch
    shared_u: jnp.ndarray  # (n_shared, hidden, shared_inter) shared up branch
    shared_d: jnp.ndarray  # (n_shared, shared_inter, hidden) shared down branch
    router_w: jnp.ndarray  # (hidden, n_routed) router weights
    router_b: jnp.ndarray  # (n_routed,) QB dispatch bias
    norm: jnp.ndarray  # (latent,) RMSNorm scale of the routed aggregation u


def init_moe(key, cfg: ModelConfig) -> LatentMoEParams:
    """Initialise a latent MoE layer.

    The router seed is folded from ``cfg.routing_seed`` (pinned in the run
    manifest) so routing is reproducible independently of the other weights.
    """
    hid, latent = cfg.hidden, cfg.moe_latent_dim
    nr, ns = cfg.moe_num_routed, cfg.moe_num_shared
    ei, si = cfg.moe_expert_intermediate, cfg.moe_shared_intermediate
    kd, ku, krg, kru, krd, ksg, ksu, ksd, kr = jax.random.split(key, 9)
    router_key = jax.random.fold_in(kr, cfg.routing_seed)
    return LatentMoEParams(
        W_down=_rand(kd, (hid, latent), 0.02),
        W_up=_rand(ku, (latent, hid), 0.02),
        expert_g=_rand(krg, (nr, latent, ei), 0.02),
        expert_u=_rand(kru, (nr, latent, ei), 0.02),
        expert_d=_rand(krd, (nr, ei, latent), 0.02),
        shared_g=_rand(ksg, (ns, hid, si), 0.02),
        shared_u=_rand(ksu, (ns, hid, si), 0.02),
        shared_d=_rand(ksd, (ns, si, hid), 0.02),
        router_w=_rand(router_key, (hid, nr), 0.02),
        router_b=jnp.zeros((nr,)),
        norm=jnp.ones((latent,)),
    )


def _flat(x: jnp.ndarray) -> jnp.ndarray:
    return x.reshape(-1, x.shape[-1])


def _scores(params: LatentMoEParams, x: jnp.ndarray) -> jnp.ndarray:
    """Unbiased router scores ``s = Sigmoid(W_r x)``, flattened ``(N, n)``."""
    xf = _flat(x)
    return jax.nn.sigmoid(compute_dtype.gemm(xf, params.router_w))


def _dispatch(params: LatentMoEParams, cfg: ModelConfig, x: jnp.ndarray):
    """Flattened router output: ``(s, topk_idx, sel_p, p_full)``.

    ``s``      — unbiased scores (N, n)
    ``topk_idx`` — selected expert indices (N, k)
    ``sel_p`` — normalised mixture weights of the selected experts (N, k)
    ``p_full`` — per-token mixture weights over all experts (N, n), zero for
                 experts not selected
    """
    s = _scores(params, x)
    n = cfg.moe_num_routed
    topk = jax.lax.top_k(s + params.router_b, cfg.moe_top_k)[1]  # (N, k)
    sel_s = jnp.take_along_axis(s, topk, axis=-1)  # (N, k)
    sel_p = sel_s / (jnp.sum(sel_s, axis=-1, keepdims=True) + 1e-8)  # (N, k)
    oh = jax.nn.one_hot(topk, n)  # (N, k, n)
    p_full = jnp.sum(oh * sel_p[..., None], axis=-2)  # (N, n)
    return s, topk, sel_p, p_full


def routing_indices(params: LatentMoEParams, cfg: ModelConfig, x: jnp.ndarray) -> jnp.ndarray:
    """Deterministic router: top-k routed-expert indices per token ``(..., k)``.

    Identical input and parameters give identical assignments (no PRNG at apply
    time).
    """
    *lead, _ = x.shape
    _, topk, _, _ = _dispatch(params, cfg, x)
    return topk.reshape(*lead, cfg.moe_top_k)


def load_fraction(params: LatentMoEParams, cfg: ModelConfig, x: jnp.ndarray) -> jnp.ndarray:
    """Dispatch fraction ``f_j`` of each routed expert over the batch.

    ``f_j = mean_i 1[j in T_i]``, so ``sum_j f_j = top_k``.
    """
    _, topk, _, _ = _dispatch(params, cfg, x)
    oh = jax.nn.one_hot(topk, cfg.moe_num_routed)  # (N, k, n)
    return jnp.sum(oh, axis=-2).mean(axis=0)  # (n,)


def qb_aux_loss(
    params: LatentMoEParams,
    cfg: ModelConfig,
    x: jnp.ndarray,
    routed: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> jnp.ndarray:
    """Differentiable load-balancing loss ``n * sum_j f_j * P_j`` (ours).

    ``f_j`` is the hard dispatch fraction and ``P_j`` the mean mixture weight of
    expert ``j`` over the batch.  At perfect balance ``f_j = k/n`` and
    ``P_j = 1/n``, giving a floor of ``k``.

    ``routed`` optionally supplies the already-computed ``(topk, p_full)`` pair
    from :func:`_dispatch`, so a caller that has just routed ``x`` (see
    :func:`apply`) does not pay for a second dispatch.
    """
    n = cfg.moe_num_routed
    if routed is None:
        _, topk, _, p_full = _dispatch(params, cfg, x)
    else:
        topk, p_full = routed
    oh = jax.nn.one_hot(topk, n)  # (N, k, n)
    f = jnp.sum(oh, axis=-2).mean(axis=0)  # (n,)
    p_mean = jnp.mean(p_full, axis=0)  # (n,)
    return n * jnp.sum(f * p_mean)


def qb_update_bias(params: LatentMoEParams, cfg: ModelConfig, x: jnp.ndarray) -> jnp.ndarray:
    """One QB bias update (Eq. 14, :578-583), applied from the next step.

    ``alpha_i`` is the Top-(k+1) cutoff of the *biased* scores ``s_i + b``; the
    new bias is the mean-centred negative ``(1 - k/n)``-quantile of
    ``s_{:,j} - alpha``.  Returns the next bias ``(n,)``.
    """
    n = cfg.moe_num_routed
    k = cfg.moe_top_k
    s = _scores(params, x)  # (N, n)
    biased = s + params.router_b  # (N, n)
    cutoff = jnp.sort(biased, axis=-1)[..., -(k + 1)]  # (N,) (k+1)-th largest
    d = s - cutoff[..., None]  # (N, n)
    q = 1.0 - k / n
    bhat = -jnp.quantile(d, q, axis=0)  # (n,)
    return bhat - jnp.mean(bhat)


def _ragged_gemm(lhs: jnp.ndarray, rhs: jnp.ndarray, group_sizes: jnp.ndarray) -> jnp.ndarray:
    """Ragged matmul at the compute-dtype boundary (``net/compute_dtype.py``).

    ``lhs`` is a batch of rows grouped by expert (``group_sizes[g]`` rows for
    expert ``g``) and ``rhs`` the ``(n_routed, k, n)`` expert weights.  The bf16
    mode casts the operands at the boundary and accumulates in fp32, mirroring
    :func:`compute_dtype.gemm_batched`.
    """
    if compute_dtype.is_bf16():
        return jax.lax.ragged_dot(
            compute_dtype.cast_in(lhs),
            compute_dtype.cast_in(rhs),
            group_sizes,
            preferred_element_type=jnp.float32,
        )
    return jax.lax.ragged_dot(lhs, rhs, group_sizes)


def _routed_experts_dense(params, cfg: ModelConfig, z, topk, sel_p):
    """Прежняя (до перф-правки) запись: einsum по ВСЕМ экспертам, потом gather top-k.

    Оставлена как **откатный путь** за флагом ``AXIOM_MOE_DISPATCH=dense``: она верна
    численно, но снова платит ``n_routed`` применений эксперта на токен вместо
    ``top_k`` (при ``n_routed=12, top_k=2`` — 6× лишней работы). Нужна как эталон и как
    средство отката без правки сигнатур: код возврата — один и тот же интерфейс.
    """
    g = compute_dtype.gemm_einsum("nl,elj->nej", z, params.expert_g)  # (N, n, ei)
    u_all = compute_dtype.gemm_einsum("nl,elj->nej", z, params.expert_u)
    a = mlp_mod.siti_glu((g, u_all), cfg.siti_beta_gate, cfg.siti_beta_up)
    if compute_dtype.is_bf16():
        e_all = compute_dtype.gemm_batched(
            a.transpose(1, 0, 2), params.expert_d).transpose(1, 0, 2)  # (N, n, latent)
    else:
        e_all = compute_dtype.gemm_einsum("nej,ejl->nel", a, params.expert_d)  # (N, n, latent)
    e = jnp.take_along_axis(e_all, topk[..., None], axis=1)  # (N, k, latent)
    return compute_dtype.gemm_einsum("nk,nkl->nl", sel_p, e)


def dispatch_mode() -> str:
    """Режим исполнения маршрутизированных экспертов: ``grouped`` (по умолчанию) | ``dense``.

    Переменная окружения ``AXIOM_MOE_DISPATCH`` читается **на каждом вызове**: флаг
    существует ровно для отката, поэтому он не кешируется и не требует перезапуска
    интерпретатора. Неизвестное значение — ошибка, а не тихий откат к умолчанию.
    """
    mode = (os.environ.get("AXIOM_MOE_DISPATCH") or "grouped").strip().lower()
    if mode not in ("grouped", "dense"):
        raise ValueError(
            f"AXIOM_MOE_DISPATCH={mode!r}: ожидается 'grouped' или 'dense'")
    return mode


def _routed_experts(params, cfg: ModelConfig, z, topk, sel_p):
    """Маршрутизированный микшер: групповая диспетчеризация top-k, либо dense по флагу."""
    if dispatch_mode() == "dense":
        return _routed_experts_dense(params, cfg, z, topk, sel_p)
    return _routed_experts_grouped(params, cfg, z, topk, sel_p)


def _routed_experts_grouped(params, cfg: ModelConfig, z, topk, sel_p):
    """Routed mix ``u`` computing **only the top-k selected experts** per token.

    Each token is gathered once per selected expert (``N * top_k`` rows), the rows
    are ordered by expert into ``n_routed`` ragged groups, the expert MLP runs on
    the groups, and the result is scattered back and mixed with ``sel_p``.  The
    hardware therefore executes ``top_k`` expert applications per token instead of
    ``n_routed`` — the algorithmic cost, not the dense one that
    ``tools/mfu_ladder.py:flops_moe_ffn`` reports as "active" vs "executed" (the
    bf16 L2 cell paid for all 12 experts to use 2: ``evidence/mfu-ladder/
    ladder-report-bf16.json``).

    The saving is numerically free: a sparse ``ragged_dot`` is *bit-identical* to
    a dense one over all experts, in both gate modes
    (``net/tests/test_32_moe_grouped_topk.py``).  What moves against the previous
    einsum spelling is only the lowering — an fp32 relative gap of ~1e-6, the
    tolerance the same test pins.
    """
    n_tokens = z.shape[0]
    k = cfg.moe_top_k
    n_routed = cfg.moe_num_routed

    expert_of = topk.reshape(-1)  # (N*k,) expert each assignment belongs to
    token_of = jnp.repeat(jnp.arange(n_tokens, dtype=expert_of.dtype), k)  # (N*k,)
    order = jnp.argsort(expert_of, stable=True)  # group the assignments by expert
    z_grouped = z[token_of[order]]  # (N*k, latent)
    group_sizes = jnp.bincount(expert_of, length=n_routed).astype(jnp.int32)

    h_g = _ragged_gemm(z_grouped, params.expert_g, group_sizes)
    h_u = _ragged_gemm(z_grouped, params.expert_u, group_sizes)
    a = mlp_mod.siti_glu((h_g, h_u), cfg.siti_beta_gate, cfg.siti_beta_up)
    e_grouped = _ragged_gemm(a, params.expert_d, group_sizes)  # (N*k, latent)

    # Undo the grouping and the token repeat, then mix over the k slots.
    e = jnp.zeros_like(e_grouped).at[order].set(e_grouped).reshape(n_tokens, k, -1)
    return compute_dtype.gemm_einsum("nk,nkl->nl", sel_p, e)


def apply(
    params: LatentMoEParams, cfg: ModelConfig, x: jnp.ndarray, want_qb: bool = False
):
    """Latent MoE over ``(..., hidden)`` -> ``(..., hidden)``.

    With ``want_qb=True`` returns ``(out, qb_aux_loss)`` instead.
    """
    *lead, hid = x.shape
    xf = _flat(x)  # (N, hidden)
    _, topk, sel_p, p_full = _dispatch(params, cfg, x)

    # routed experts in latent space
    z = compute_dtype.gemm(xf, params.W_down)  # (N, latent)
    # Expert application, computed only for the selected top-k experts per token
    # (see :func:`_routed_experts_grouped`).  The previous spelling materialised
    # all n_routed experts and gathered afterwards — the hardware paid 6x the
    # algorithmic cost at n_routed=12, top_k=2 (the MFU fix this replaces).
    u = _routed_experts(params, cfg, z, topk, sel_p)  # (N, latent)

    # shared experts (full-width path)
    sg = compute_dtype.gemm_einsum("nh,shj->nsj", xf, params.shared_g)  # (N, ns, si)
    su = compute_dtype.gemm_einsum("nh,shj->nsj", xf, params.shared_u)  # (N, ns, si)
    sa = mlp_mod.siti_glu((sg, su), cfg.siti_beta_gate, cfg.siti_beta_up)  # (N, ns, si)
    s_out = compute_dtype.gemm_einsum("nsj,sjh->nsh", sa, params.shared_d)  # (N, ns, hidden)
    shared_sum = jnp.sum(s_out, axis=1)  # (N, hidden)

    y = shared_sum + compute_dtype.gemm(rms_norm(u, params.norm), params.W_up)  # (N, hidden)
    out = y.reshape(*lead, hid)
    if want_qb:
        # The QB term is a monitoring load-balancing loss, not a gradient
        # signal: K3's QB is auxiliary-free (the bias update balances loads).
        # We stop the gradient so the router is trained only by the NTP/MTP
        # loss — a differentiable aux loss empirically destabilises the smoke.
        # Reuse the routing already computed above; a bare qb_aux_loss(x)
        # would pay for a second dispatch per MoE layer.
        qb = jax.lax.stop_gradient(qb_aux_loss(params, cfg, x, routed=(topk, p_full)))
        return out, qb
    return out
