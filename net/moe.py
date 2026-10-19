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

Counting-sort группировки (ADR-054 Am.23, веха-11)
--------------------------------------------------

``AXIOM_MOE_COUNTING=1`` (дефолт 0, fail-closed, читается на каждом вызове —
форма ``AXIOM_MOE_DISPATCH``) заменяет ``jnp.argsort(expert_of,
stable=True)`` в :func:`_routed_experts_grouped` на counting-sort
(:func:`_counting_sort_positions`: ``bincount`` + ``cumsum`` +
``scatter``/``gather``).  Стабильный порядок внутри эксперта — порядок
появления — тот же, что даёт stable argsort, поэтому раскладка (и выход MoE)
совпадает с прежней **побитово**, а CUB radix-sort (на GB10 —
``DeviceRadixSort`` с ``CopyOffsets``-эпизодом host-копии offsets, Am.2/Am.23)
не вызывается: детерминизм — по построению, семантика топ-2 не меняется.
"""

from __future__ import annotations

import functools
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


#: Router top-k selector: ``argmax`` (default) | ``jax`` (A/B reference).
#:
#: Read **once, at import** (unlike ``AXIOM_MOE_DISPATCH``, which is a hot
#: rollback and is read per call): the selector is part of the run's declared
#: execution recipe, not a switch flipped mid-process.  ``jax`` forces
#: ``jax.lax.top_k`` as the parity reference; ``argmax`` selects the iterative
#: selector inside its applicability window (:data:`ARGMAX_MAX_K` /
#: :data:`ARGMAX_MAX_N`) and falls back to ``jax.lax.top_k`` outside it.
TOPK_MODE_ENV = "AXIOM_MOE_TOPK"

#: Iterative-``argmax`` selector (the default; see :func:`_topk_argmax`).
ARGMAX_SELECTOR = "argmax"

#: ``jax.lax.top_k`` reference selector — the parity oracle for tests and A/B.
JAX_SELECTOR = "jax"

#: Declared selector values.
TOP_K_SELECTORS = (ARGMAX_SELECTOR, JAX_SELECTOR)

#: Applicability window of :func:`_topk_argmax`: the MoE router runs
#: ``top_k = moe_top_k`` (campaign: 2) over ``n = moe_num_routed`` (12) experts.
ARGMAX_MAX_K = 8
ARGMAX_MAX_N = 64


def _read_topk_mode(raw: str | None) -> str:
    """Parse ``AXIOM_MOE_TOPK`` (fail-closed: an unknown value is an error).

    A typo must not silently pick one of two execution paths whose *lowering*
    differs (the numerics are pinned equal by parity tests, but the reported
    execution is not) — so the unknown value refuses to guess, like
    ``net/compute_dtype.py``.
    """
    if raw is None:
        return ARGMAX_SELECTOR
    value = raw.strip().lower()
    if not value:
        return ARGMAX_SELECTOR
    if value not in TOP_K_SELECTORS:
        raise ValueError(
            f"{TOPK_MODE_ENV}={raw!r}: ожидается один из {list(TOP_K_SELECTORS)}")
    return value


#: The selector this process runs with, pinned at import time
#: (``AXIOM_MOE_TOPK``; see :data:`TOPK_MODE_ENV`).
TOPK_MODE = _read_topk_mode(os.environ.get(TOPK_MODE_ENV))


def _topk_argmax(scores: jnp.ndarray, k: int) -> jnp.ndarray:
    """Top-``k`` indices per row via ``k`` masked ``argmax`` steps — no sort.

    Why not ``jax.lax.top_k`` (ADR-054 Amendment 2, stage-2a): on GB10 every
    call lowers to the CUB ``DeviceRadixSort`` custom call, and its
    ``CopyOffsets`` step stages the ``(N+1)``-word offsets array host→device
    with a *pageable* ``cudaMemcpyAsync`` (gdb probe stacks of the stage-2a
    investigation: ``CubSortPairsExecute`` → ``CopyOffsets`` →
    ``cudaMemcpyAsync``).  The step profile attributed ~41 such 64 KiB pageable
    copies per step to the ``top_k`` custom calls; their stalls on the single
    host thread are the measured binding term of the ~14 s/step of MoE stalls
    (ADR-054 Am.2: submission contention — the *offsets round-trip* is the
    cost, not the device sort).  Iterative ``argmax`` + masking keeps the same
    top-k set with elementwise ops and reductions only, and emits no custom
    call.

    Semantics — aligned with ``jax.lax.top_k`` on the shapes it covers:

    * indices are returned in descending-score order; ties are broken by the
      **lowest index first** (``argmax`` returns the first maximum and the mask
      preserves that rule — the same order the CUB sort gives; pinned by
      ``net/tests/test_moe_topk_argmax.py``);
    * ``±inf`` entries are supported: when every still-unselected value is
      ``-inf``, an excluded position is indistinguishable from a free one *by
      value*, so that corner is resolved explicitly to the first free index;
    * ``NaN`` is not covered (the router does not produce it, and
      ``jax.lax.top_k`` does not define its order either);
    * requires ``k <= scores.shape[-1]`` (the caller guarantees it).

    Scope: small ``k``/``n`` only — see :func:`_topk_indices`, which selects
    this path for the router's regime (``k = 2`` over ``n = 12``) and falls back
    to ``jax.lax.top_k`` elsewhere.
    """
    n = scores.shape[-1]
    selected = jnp.zeros(scores.shape, dtype=bool)  # (N, n) picked positions
    work = scores
    picks = []
    for _ in range(k):
        chosen = jnp.argmax(work, axis=-1)  # (N,) first maximum (lowest index)
        # Picked positions are masked to -inf.  If all free entries are -inf as
        # well, argmax may return a picked position; resolve that explicitly to
        # the first free index — the order lax.top_k gives on -inf ties.
        exhausted = jnp.max(work, axis=-1) == -jnp.inf
        first_free = jnp.argmin(selected.astype(jnp.int32), axis=-1)  # (N,)
        chosen = jnp.where(exhausted, first_free, chosen)
        picks.append(chosen)
        one_hot = jax.nn.one_hot(chosen, n, dtype=bool)  # (N, n)
        selected = selected | one_hot
        work = jnp.where(one_hot, -jnp.inf, work)
    return jnp.stack(picks, axis=-1)  # (N, k), descending-score order


def _topk_indices(scores: jnp.ndarray, k: int) -> jnp.ndarray:
    """Top-``k`` indices per row: adaptive selector for the router.

    The branch is a pure function of ``k``/``n`` and the import-pinned
    :data:`TOPK_MODE` — never of values — so tracing under ``jit`` is
    deterministic and a re-run picks the same path:

    * ``TOPK_MODE=argmax`` (default) and ``0 < k <= ARGMAX_MAX_K`` and
      ``k <= n <= ARGMAX_MAX_N`` → :func:`_topk_argmax` (no custom call);
    * otherwise → ``jax.lax.top_k`` (kept as the reference and for large
      ``k``/``n``).
    """
    n = scores.shape[-1]
    if (
        TOPK_MODE == ARGMAX_SELECTOR
        and 0 < k <= ARGMAX_MAX_K
        and k <= n <= ARGMAX_MAX_N
    ):
        return _topk_argmax(scores, k)
    return jax.lax.top_k(scores, k)[1]


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
    topk = _topk_indices(s + params.router_b, cfg.moe_top_k)  # (N, k)
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


#: Env-гейт способа группировки назначений по экспертам: ``0`` (дефолт —
#: прежний ``argsort``) | ``1`` (counting-sort: bincount + cumsum + scatter).
COUNTING_MODE_ENV = "AXIOM_MOE_COUNTING"

#: Допустимые значения гейта; неизвестное — ошибка (fail-closed, форма
#: ``AXIOM_ACCUM_JIT``), а не тихий откат в дефолт.
COUNTING_MODES = ("0", "1")


def counting_sort_mode(value: str | None = None) -> bool:
    """Способ группировки маршрутизации: ``False`` (argsort, дефолт) | ``True``.

    ``AXIOM_MOE_COUNTING`` читается **на каждом вызове** (форма
    ``AXIOM_MOE_DISPATCH`` — флаг отката доступен без перезапуска).  Дефолт 0:
    прежний путь ``jnp.argsort(expert_of, stable=True)`` — побитово прежнее
    поведение.  Значение 1 включает counting-sort (bincount + cumsum +
    scatter): на GB10 ``argsort`` по малому числу различных ключей понижается в
    CUB radix-sort (``DeviceRadixSort`` + pageable-копии offsets), чей
    host-эпизод — измеренная цена шага (Am.23: CUB 270 ядер/шаг); counting-sort
    эмитит только elementwise/scan/scatter-операции.  Раскладка counting-sort
    **побитово совпадает** с прежней (стабильный порядок внутри эксперта =
    порядок появления — тот же, что у stable argsort), поэтому семантика топ-2
    маршрутизации не меняется, а детерминизм — по построению.  Неизвестное
    значение — :class:`ValueError` (fail-closed).
    """
    raw = value if value is not None else os.environ.get(COUNTING_MODE_ENV)
    mode = (raw or "").strip().lower() or "0"
    if mode not in COUNTING_MODES:
        raise ValueError(
            f"{COUNTING_MODE_ENV}={mode!r}: ожидается одно из {list(COUNTING_MODES)}")
    return mode == "1"


def _counting_sort_positions(expert_of: jnp.ndarray, group_sizes: jnp.ndarray) -> jnp.ndarray:
    """Стабильный counting-sort: позиция каждого назначения в порядке по экспертам.

    ``pos[i] = offsets[expert_of[i]] + rank_i``, где ``rank_i`` — число более
    ранних назначений того же эксперта, а ``offsets`` — начало блока эксперта
    (``cumsum(group_sizes) - group_sizes``).  Внутри эксперта порядок — порядок
    появления, то есть **тот же самый**, что у ``jnp.argsort(expert_of,
    stable=True)``: ``pos`` есть обратная перестановка к стабильному argsort,
    отсюда побитовое равенство раскладки (Am.23).  Никакого сорта/CUB:
    только bincount + cumsum + gather.
    """
    n_routed = group_sizes.shape[0]
    offsets = jnp.cumsum(group_sizes) - group_sizes  # (n,) начало блока эксперта
    one_hot = jax.nn.one_hot(expert_of, n_routed, dtype=jnp.int32)  # (N*k, n)
    ranks = jnp.cumsum(one_hot, axis=0) - 1  # ранг «по появлению» по каждому эксперту
    rank_of = jnp.take_along_axis(ranks, expert_of[:, None], axis=1)[:, 0]  # (N*k,)
    return offsets[expert_of] + rank_of  # (N*k,) позиция в упорядоченной раскладке


#: Env-гейт backward'а маршрутизированных экспертов (ADR-054 Am.52, цикл-30):
#: ``"jax"`` (дефолт — прежний autodiff, побитово) | ``"fused"``
#: (``jax.custom_vjp``: forward дословно прежний, backward — fused-кернел
#: ``net/kernels/moe_experts_bwd.py``).  Форма ``AXIOM_MOE_COUNTING``/
#: ``AXIOM_KDA_CC_IMPL``: читается на трейсе, неизвестное — ошибка.
MOE_BWD_ENV = "AXIOM_MOE_BWD"

#: Допустимые значения гейта; неизвестное — :class:`MoeBwdError` (fail-closed).
MOE_BWD_VALUES = ("jax", "fused")
MOE_BWD_JAX = "jax"
MOE_BWD_FUSED = "fused"


class MoeBwdError(ValueError):
    """Неизвестное значение ``AXIOM_MOE_BWD`` (fail-closed, без догадок)."""


def moe_bwd_mode(value: str | None = None) -> str:
    """Backward маршрутизированных экспертов: ``"jax"`` (дефолт) | ``"fused"``.

    ``AXIOM_MOE_BWD=fused`` оборачивает экспертную часть
    :func:`_routed_experts_grouped` в :func:`jax.custom_vjp`: forward-правило —
    дословно прежний путь (grouping + три ``ragged_dot`` + ``siti_glu``; числа
    те же), backward-правило — fused-кернел экспертного блока
    (:mod:`net.kernels.moe_experts_bwd`), а группировка/сборка строк и
    градиент маршрутизации (``sel_p``/``topk``) остаются в XLA.  ``jax`` —
    прежний autodiff, побитово и с прежним jaxpr (блокирующий критерий дельты).

    Пустая/незаданная переменная = ``"jax"``; значения — только
    ``"jax"``/``"fused"`` (регистр и окружающие пробелы нормализуются);
    неизвестное — :class:`MoeBwdError` (fail-closed: опечатка не должна тихо
    выдать прежний путь за fused-замер).  Явный аргумент сильнее окружения
    (форма :func:`net.model.ce_bwd_mode`).  Читается на трейсе, не кешируется —
    A/B идёт на одной ревизии.
    """
    raw = value if value is not None else os.environ.get(MOE_BWD_ENV)
    mode = (raw or "").strip().lower() or MOE_BWD_JAX
    if mode not in MOE_BWD_VALUES:
        raise MoeBwdError(
            f"{MOE_BWD_ENV}={mode!r}: ожидается одно из {list(MOE_BWD_VALUES)}"
        )
    return mode


def _grouped_layout(topk, n_tokens, k, n_routed, counting):
    """Раскладка назначений по экспертам — дословно прежний путь forward'а.

    Возвращает ``(expert_of, token_of, group_sizes, keys)``, где ``keys`` —
    ``("pos", pos)`` для counting-sort или ``("order", order)`` для stable
    argsort.  Вынесено из :func:`_routed_experts_grouped`, чтобы fused-forward
    считал ту же раскладку теми же выражениями (побитовое совпадение).
    """
    expert_of = topk.reshape(-1)
    token_of = jnp.repeat(jnp.arange(n_tokens, dtype=expert_of.dtype), k)
    group_sizes = jnp.bincount(expert_of, length=n_routed).astype(jnp.int32)
    if counting:
        pos = _counting_sort_positions(expert_of, group_sizes)
        return expert_of, token_of, group_sizes, ("pos", pos)
    order = jnp.argsort(expert_of, stable=True)
    return expert_of, token_of, group_sizes, ("order", order)


def _grouped_forward(wg, wu, wd, z, topk, sel_p, n_tokens, k, n_routed,
                     counting, beta_gate, beta_up):
    """Дословно прежний forward экспертной части + остатки для backward'а.

    Те же выражения, что в :func:`_routed_experts_grouped` (grouping, три
    ``_ragged_gemm``, ``siti_glu``, обратная перестановка и микс ``sel_p``):
    числа fused-forward совпадают с ``jax``-путём побитово.
    """
    expert_of, token_of, group_sizes, keys = _grouped_layout(
        topk, n_tokens, k, n_routed, counting)
    kind, key = keys
    if kind == "pos":
        z_grouped = jnp.zeros((expert_of.shape[0], z.shape[-1]), dtype=z.dtype).at[key].set(
            z[token_of]
        )
    else:
        z_grouped = z[token_of[key]]

    h_g = _ragged_gemm(z_grouped, wg, group_sizes)
    h_u = _ragged_gemm(z_grouped, wu, group_sizes)
    a = mlp_mod.siti_glu((h_g, h_u), beta_gate, beta_up)
    e_grouped = _ragged_gemm(a, wd, group_sizes)  # (N*k, latent)
    if kind == "pos":
        e = e_grouped[key].reshape(n_tokens, k, -1)
    else:
        e = jnp.zeros_like(e_grouped).at[key].set(e_grouped).reshape(n_tokens, k, -1)
    u = compute_dtype.gemm_einsum("nk,nkl->nl", sel_p, e)
    res = (wg, wu, wd, z, topk, sel_p, group_sizes, z_grouped, h_g, h_u, a, e,
           key)
    return u, res


def _grouped_backward(res, du, counting):
    """Fused backward экспертной части: кернел + XLA-группировка/сборка.

    Котангент выхода ``du`` (N, latent) переводится в котангент сгруппированных
    строк (``de_grouped``) *той же* перестановкой, что forward; fused-кернел
    экспертного блока возвращает градиенты весов (по экспертам) и
    ``dz_grouped``; ``dz`` собирается обратно по токенам, ``dsel_p`` —
    прежним XLA-путём (``∂u/∂sel_p`` через сохранённый ``e``).  ``topk`` —
    индексы: котангента нет (``None``).
    """
    from .kernels import moe_experts_bwd  # локальный импорт: нужен только под fused

    (wg, wu, wd, z, topk, sel_p, group_sizes, z_grouped, h_g, h_u, a, e,
     key) = res
    n_tokens, k, latent = e.shape
    rows = z_grouped.shape[0]
    dt = jnp.bfloat16

    # Котангент выхода -> котангент назначений -> сгруппированные строки.
    # u_nl = sum_k sel_p_nk · e_nkl  =>  de_nkl = du_nl · sel_p_nk.
    de = du[:, None, :] * sel_p[..., None]  # (N, k, latent)
    de_flat = de.reshape(rows, latent)
    if counting:
        de_grouped = jnp.zeros((rows, latent), dtype=de_flat.dtype).at[key].set(de_flat)
    else:
        de_grouped = de_flat[key]

    # Единственный fused-запуск: все градиенты экспертного блока.  Операнды
    # приводятся к объявленному bf16-пути mma; аккумуляторы — fp32.
    dwg, dwu, dwd, dz_grouped = moe_experts_bwd.kernel(
        z_grouped.astype(dt),
        h_g.astype(dt),
        h_u.astype(dt),
        a.astype(dt),
        de_grouped.astype(dt),
        wg.astype(dt),
        wu.astype(dt),
        wd.astype(dt),
        group_sizes,
    )

    # dz обратно по токенам (обратная перестановка forward'а) и сумма по k.
    if counting:
        dz_flat = dz_grouped[key]
    else:
        dz_flat = jnp.zeros_like(dz_grouped).at[key].set(dz_grouped)
    dz = jnp.sum(dz_flat.reshape(n_tokens, k, latent), axis=1)

    dsel_p = compute_dtype.gemm_einsum("nl,nkl->nk", du, e)
    return (
        dwg.astype(wg.dtype),
        dwu.astype(wu.dtype),
        dwd.astype(wd.dtype),
        dz.astype(z.dtype),
        None,
        dsel_p.astype(sel_p.dtype),
    )


@functools.lru_cache(maxsize=8)
def _grouped_fused_impl(n_tokens, k, n_routed, counting, beta_gate, beta_up):
    """``custom_vjp``-обёртка экспертной части, ключ — статическая геометрия.

    Фабрика кэшируется по статическим параметрам (объект ``custom_vjp``
    стабилен между шагами, jit-кеш не рвётся); гейт читается до входа сюда —
    на трейсе.  Правила резолвят :func:`net.kernels.moe_experts_bwd.kernel` на
    модуле в момент вызова, поэтому тесты подменяют шов, не трогая moe.py.
    """

    @jax.custom_vjp
    def experts(wg, wu, wd, z, topk, sel_p):
        u, _ = _grouped_forward(wg, wu, wd, z, topk, sel_p, n_tokens, k,
                                n_routed, counting, beta_gate, beta_up)
        return u

    def experts_fwd(wg, wu, wd, z, topk, sel_p):
        return _grouped_forward(wg, wu, wd, z, topk, sel_p, n_tokens, k,
                                n_routed, counting, beta_gate, beta_up)

    def experts_bwd(res, du):
        return _grouped_backward(res, du, counting)

    experts.defvjp(experts_fwd, experts_bwd)
    return experts


def _routed_experts(params, cfg: ModelConfig, z, topk, sel_p):
    """Маршрутизированный микшер: групповая диспетчеризация top-k, либо dense по флагу."""
    if dispatch_mode() == "dense":
        return _routed_experts_dense(params, cfg, z, topk, sel_p)
    return _routed_experts_grouped(params, cfg, z, topk, sel_p)


def _routed_experts_grouped(params, cfg: ModelConfig, z, topk, sel_p, *,
                            bwd: str | None = None):
    """Routed mix ``u`` computing **only the top-k selected experts** per token.

    Гейт ``AXIOM_MOE_BWD`` (``bwd`` сильнее env) выбирает backward: ``"jax"``
    (дефолт) — прежний autodiff побитово; ``"fused"`` — ``custom_vjp``, forward
    которого дословно повторяет путь ниже, а backward исполняет fused-кернел
    экспертного блока (:func:`_grouped_fused_impl`).  Forward не меняется.

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
    if moe_bwd_mode(bwd) == MOE_BWD_FUSED:
        impl = _grouped_fused_impl(
            z.shape[0], cfg.moe_top_k, cfg.moe_num_routed, counting_sort_mode(),
            cfg.siti_beta_gate, cfg.siti_beta_up)
        return impl(params.expert_g, params.expert_u, params.expert_d, z,
                    topk, sel_p)

    n_tokens = z.shape[0]
    k = cfg.moe_top_k
    n_routed = cfg.moe_num_routed

    expert_of = topk.reshape(-1)  # (N*k,) expert each assignment belongs to
    token_of = jnp.repeat(jnp.arange(n_tokens, dtype=expert_of.dtype), k)  # (N*k,)
    group_sizes = jnp.bincount(expert_of, length=n_routed).astype(jnp.int32)
    counting = counting_sort_mode()
    if counting:
        # Am.23: counting-sort вместо argsort — та же стабильная раскладка по
        # экспертам (порядок внутри эксперта = порядок появления), но без
        # CUB radix-sort: только bincount/cumsum/scatter.
        pos = _counting_sort_positions(expert_of, group_sizes)
        z_grouped = jnp.zeros((expert_of.shape[0], z.shape[-1]), dtype=z.dtype).at[pos].set(
            z[token_of]
        )  # (N*k, latent)
    else:
        order = jnp.argsort(expert_of, stable=True)  # group the assignments by expert
        z_grouped = z[token_of[order]]  # (N*k, latent)

    h_g = _ragged_gemm(z_grouped, params.expert_g, group_sizes)
    h_u = _ragged_gemm(z_grouped, params.expert_u, group_sizes)
    a = mlp_mod.siti_glu((h_g, h_u), cfg.siti_beta_gate, cfg.siti_beta_up)
    e_grouped = _ragged_gemm(a, params.expert_d, group_sizes)  # (N*k, latent)

    # Undo the grouping and the token repeat, then mix over the k slots.
    if counting:
        # Обратная перестановка — gather по тем же позициям counting-sort: для
        # назначения i результат лежит в e_grouped[pos[i]] (Am.23).
        e = e_grouped[pos].reshape(n_tokens, k, -1)
    else:
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
