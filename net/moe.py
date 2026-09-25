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

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import ModelConfig
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
    return jax.nn.sigmoid(xf @ params.router_w)


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
    z = xf @ params.W_down  # (N, latent)
    hg = jnp.einsum("nl,elj->nej", z, params.expert_g)  # (N, nr, ei)
    hu = jnp.einsum("nl,elj->nej", z, params.expert_u)  # (N, nr, ei)
    a = mlp_mod.siti_glu((hg, hu), cfg.siti_beta_gate, cfg.siti_beta_up)  # (N, nr, ei)
    e_out = jnp.einsum("nej,ejl->nel", a, params.expert_d)  # (N, nr, latent)
    sel_out = jnp.take_along_axis(e_out, topk[..., None], axis=1)  # (N, k, latent)
    u = jnp.einsum("nk,nkl->nl", sel_p, sel_out)  # (N, latent)

    # shared experts (full-width path)
    sg = jnp.einsum("nh,shj->nsj", xf, params.shared_g)  # (N, ns, si)
    su = jnp.einsum("nh,shj->nsj", xf, params.shared_u)  # (N, ns, si)
    sa = mlp_mod.siti_glu((sg, su), cfg.siti_beta_gate, cfg.siti_beta_up)  # (N, ns, si)
    s_out = jnp.einsum("nsj,sjh->nsh", sa, params.shared_d)  # (N, ns, hidden)
    shared_sum = jnp.sum(s_out, axis=1)  # (N, hidden)

    y = shared_sum + rms_norm(u, params.norm) @ params.W_up  # (N, hidden)
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
