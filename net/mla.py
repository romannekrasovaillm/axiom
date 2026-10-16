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

Compute dtype (``AXIOM_COMPUTE_DTYPE``, BF16-кампания фаза 1): every GEMM and
attention contraction of this module goes through ``net/compute_dtype.py``, whose
``fp32`` mode returns the caller's own expression unchanged (the oracle stays
byte-exact, ADR-009 D4) and whose ``bf16`` mode casts the *operands* to bf16
while the products accumulate and return in fp32.  Nothing else changes dtype:
the residual stream, the gate and the loss path stay fp32.
"""

from __future__ import annotations

import functools
import os
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import ModelConfig
from . import attn_sparse
from . import compute_dtype
from . import quant

# MFU-эксперимент 07.10 (директива «поднять MFU»): fused flash attention в
# dense-оракуле.  По умолчанию ВЫКЛ — pinned oracle байт-точен (ADR-009 D4);
# включение — только диагностические прогоны (VERIFICATION-LEG).
def _flash_dense() -> bool:
    """Whether the fused flash attention path of the dense oracle is selected.

    Read per call (not pinned at import) so a parity test can flip the switch
    in-process; the branch is resolved at trace time, so a compiled graph
    carries the value it was traced with.
    """
    return os.environ.get("AXIOM_MLA_DENSE_FLASH", "0") == "1"


# --------------------------------------------------------------------------- #
# Веха-9 (ADR-054 Am.20, кампания MFU-55): bf16-хранение скоров в backward.
#
# Пол B=4 по HLO-дампу — fp32 attention-scores (B,12,8192,8192) в backward
# (12 ГиБ × множество живых тензоров); flash лечит только forward.  Гейт
# хранит для backward bf16-копию softmax-выхода dense-пути, вычисления
# остаются fp32.  Чтение fail-closed (форма ``AXIOM_ACCUM_JIT``/
# ``AXIOM_MASTER_DTYPE``): опечатка не должна тихо оставить fp32 — замер ноги
# «bf16-хранение» стал бы замером прежнего пути.
# --------------------------------------------------------------------------- #

#: Env-гейт bf16-хранения скоров в backward (``0`` — дефолт, прежний путь).
MLA_BWD_BF16_ENV = "AXIOM_MLA_BWD_BF16"

#: Допустимые значения; неизвестное — ошибка (fail-closed, без догадок).
MLA_BWD_BF16_VALUES = ("0", "1")
MLA_BWD_BF16_OFF = "0"
MLA_BWD_BF16_ON = "1"


class MlaBwdBf16Error(ValueError):
    """Неизвестное значение ``AXIOM_MLA_BWD_BF16`` (fail-closed, без догадок)."""


def bwd_bf16_mode(value: str | None = None) -> bool:
    """Хранение скоров dense-пути в backward: ``False`` (дефолт) | ``True``.

    ``AXIOM_MLA_BWD_BF16=1`` включает ветку :func:`_attn_out_bf16stored`:
    softmax-выход (B,H,T,T), нужный backward, сохраняется в bf16 (каст при
    сохранении), сами softmax/матмулы и накопление идут в fp32, как раньше —
    forward-числа ветка не меняет (см. докстринг хелпера).  Пустая/незаданная
    переменная = ``"0"``; значения — только ``"0"``/``"1"``; неизвестное —
    :class:`MlaBwdBf16Error`.  Явный аргумент сильнее окружения (форма
    ``accum_jit_mode``/``master_dtype_mode``).  Читается на трейсе (как
    flash-переключатель), чтобы тест мог переключить гейт в процессе.
    """
    raw = value if value is not None else os.environ.get(MLA_BWD_BF16_ENV)
    mode = (raw or "").strip().lower() or MLA_BWD_BF16_OFF
    if mode not in MLA_BWD_BF16_VALUES:
        raise MlaBwdBf16Error(
            f"{MLA_BWD_BF16_ENV}={mode!r}: ожидается одно из "
            f"{list(MLA_BWD_BF16_VALUES)}"
        )
    return mode == MLA_BWD_BF16_ON


@jax.custom_vjp
def _attn_out_bf16stored(
    scores: jnp.ndarray, v: jnp.ndarray, mask: jnp.ndarray
) -> jnp.ndarray:
    """``softmax(where(mask, scores, −inf)) @ v`` с bf16-хранением для backward.

    Гейт ``AXIOM_MLA_BWD_BF16=1`` (веха-9, ADR-054 Am.20): тензор
    (B,H,T,T), который backward держит живым, — тот же самый softmax-выход,
    но в bf16.  Вычисления не меняются: маскирование, softmax и контракция
    идут в fp32 (``compute_dtype.gemm_einsum``), forward-значения побитово
    совпадают с прежним путём, backward читает bf16-копию с апкастом в fp32.

    Маска — аргумент (а не ``where`` снаружи) намеренно: transpose-правило
    ``select_n`` захватывает оба своих float-примитива как остатки, т.е.
    ``where(mask, scores, −inf)`` снаружи оставлял бы в графе лишний fp32
    (B,H,T,T) — тот же, что гейт призван срезать (измерено: pullback-константы
    несут fp32-скоры, потребитель — транспонированный ``_where``, в теле
    которого этот вход не используется).  Внутри custom-вызова маскирование
    живёт в моей формуле backward: ``grad_scores = where(mask, attn ⊙
    (grad_attn − rowsum), 0)``.

    Почему custom_vjp, а не каст в forward.  ``attn.astype(bf16).astype(f32)``
    в forward не срезает пол: transpose-правила внутренних примитивов
    softmax (числитель деления, выход exp, argmax-путь reduce_max) и самой
    контракции держат живыми **fp32**-тензоры (B,H,T,T) — трассировка
    линейного отображения показывает три fp32-копии и до, и после каста;
    bf16-каст добавляет лишнюю копию, ничего не убирая.  Явный backward
    фиксирует состав остатков: ровно ``(bf16 attn, v, mask)`` — 12→6 ГиБ на
    живой тензор и без остальных fp32-копий.

    Формула backward — стандартная для softmax (и совпадает с формулой,
    которую разворачивает autodiff):
    ``grad_scores = attn ⊙ (grad_attn − rowsum(grad_attn ⊙ attn))``, поверх —
    та же маска, что применялась к scores в forward.
    """
    masked = jnp.where(mask, scores, jnp.finfo(jnp.float32).min)
    attn = jax.nn.softmax(masked, axis=-1)
    return compute_dtype.gemm_einsum("bhts,bshd->bthd", attn, v)


def _attn_out_bf16stored_fwd(scores, v, mask):
    masked = jnp.where(mask, scores, jnp.finfo(jnp.float32).min)
    attn = jax.nn.softmax(masked, axis=-1)
    o = compute_dtype.gemm_einsum("bhts,bshd->bthd", attn, v)
    # Сохраняемый тензор — только bf16-копия softmax-выхода (и v, (B,T,H,dq);
    # mask — bool (B,H,T,T), байт на элемент).
    return o, (attn.astype(jnp.bfloat16), v, mask)


def _attn_out_bf16stored_bwd(res, g):
    attn_stored, v, mask = res
    attn = attn_stored.astype(jnp.float32)  # апкаст при чтении; дальше — fp32
    grad_attn = compute_dtype.gemm_einsum("bthd,bshd->bhts", g, v)
    rowsum = jnp.sum(grad_attn * attn, axis=-1, keepdims=True)
    grad_scores = attn * (grad_attn - rowsum)
    grad_scores = jnp.where(mask, grad_scores, 0.0)  # маска forward'а (transpose where)
    grad_v = compute_dtype.gemm_einsum("bhts,bthd->bshd", attn, g)
    # Контракт dtype (фикс-раунд вехи-9, форма CE-фикса): котангенты несут dtype
    # прималов.  Здесь оба примала границы — fp32 всегда: ``scores`` —
    # выход ``gemm_einsum`` (fp32-аккумулятор), ``v`` — выход ``_project``
    # (gemm-граница, fp32); bf16 живёт только в остатке ``attn_stored``.
    return grad_scores, grad_v, None


_attn_out_bf16stored.defvjp(_attn_out_bf16stored_fwd, _attn_out_bf16stored_bwd)


# --------------------------------------------------------------------------- #
# Веха-12 (ADR-054 Am.25, цикл-15 кампании MFU-55): flash-backward MLA —
# jax-ступень.  Тайловое переисчисление скоров в backward без хранения
# полной (B,H,T,T)-матрицы.
#
# Пол B=4 (Am.20, HLO-разбор): fp32 attention-scores (B,12,8192,8192) по 12 ГиБ
# живут в backward десятками вхождений (80+70+19); bf16-хранение (Am.20) срезает
# dtype, не снимает пол (plain-B=4 OOM 78,95 ГиБ).  Здесь backward вообще не
# хранит скоры: forward при обучении вычисляется плотно, как раньше (числа
# побитово те же), но для backward остаются только ``q``, ``k``, ``v``, маска,
# выход ``o`` и статистики softmax каждой строки (rowmax/rowsum, (B,H,T,1)) —
# а QK^T переисчитывается в backward по тайлам, как в flash-attention backward:
# внешний цикл по KV-тайлам, внутренний по Q-тайлам, живой скор-тайл ≤
# (B,H,tile,tile).
#
# Граница custom_vjp — ``(q, k, v, mask) -> o``, а не ``(scores, v, mask) -> o``
# (как у bf16-гейта Am.20): vjp границы обязан вернуть котангент формы входа, и
# граница по ``scores`` вернула бы ровно ту полную (B,H,T,T), ради снятия
# которой веха и делается.  Сдвинув границу за QK^T и softmax, backward отдаёт
# сразу ``(grad_q, grad_k, grad_v)`` — все три формы малы ((B,T,H,dq)), полной
# матрицы не существует ни в одном промежутке (пин — test_43: в jaxpr/StableHLO
# backward нет ни одной float-переменной формы (B,H,T,T), все скор-операции
# ≤ тайла).
#
# Формула backward — стандартная flash-attention: D_i = rowsum_s(do_i ⊙ o_i)
# (равно rowsum(dp ⊙ p), но не требует второго прохода), для пары тайлов
# p = exp(s_ij − rowmax_i)/rowsum_i (маска — 0), ds = p ⊙ (dp − D_i),
# dp = do_i V_j^т, grad_s = ds·scale; вклады копятся тайловыми матмулами
# (dq — по внешнему циклу, dk/dv — по внутреннему).  Чётность с прежним
# backward — в допуске вехи (≤1e-3; измерено ~1e-6 отн. в fp32-режиме).
#
# Гейт ``AXIOM_MLA_FLASH_BWD=1`` подменяет ТОЛЬКО backward dense-пути; forward
# не трогается (flash-fwd ``AXIOM_MLA_DENSE_FLASH`` — отдельная ветка, гейт
# туда не читается).  Дефолт ``0`` — побитово прежний граф.  Ширина тайла —
# конфиг ``ModelConfig.mla_flash_bwd_tile`` (дефолт 1024; память backward ∝
# тайлу, SMEM-бюджет GB10 99 КиБ/блок — ориентир будущей Pallas-ступени).
# Связка с ``AXIOM_MLA_BWD_BF16`` (Am.20): когда оба включены, исполняется
# тайловый путь — он хранит меньше (bf16-копии полной attn не заводит вовсе);
# bf16-гейт при этом остаётся фактическим значением в метрике, A/B-ноги
# различимы по журналу.  Стендовые числа B=4/B=8 — отдельная нога кампании;
# здесь пиннуется jax-уровень: числа, состав графа, паритет и nan-контроль.
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Веха-13 (ADR-054 Am.27, цикл-16 кампании MFU-55): персистентный MLA-блок —
# fwd+bwd внимания MLA одним запуском кернела на слой-проход (ступень-2).
#
# NVTX-атрибуция Am.27: блокирующие эпизоды шага — backward-структура while/
# remat (4,46 с одним спаном + 5 remat-спанов 150–300 мс) и тайл-цикл
# flash-bwd вехи-12 (~12 с/шаг хоста).  Гейт ``AXIOM_MLA_PERSIST=1`` проводит
# интервал внимания ``(q, k, v) -> o`` через персистентные кернелы
# (:mod:`net.kernels.mla_persist`, образец KDA-вех 3–4: grid=(B*H,), состояние
# в SMEM, один запуск на направление и слой-проход): forward-числа — тайловая
# online-softmax algebra в bf16-операндах мма (объявленный путь операндов,
# тот же, что у compute-dtype bf16), backward — персистентный кернел с
# переисчислением скоров, полной (B,H,T,T) не существует ни в одном
# промежутке; граф теряет и Python-тайл-цикл, и while/remat-структуру над
# вниманием.  Дефолт ``0`` — побитово прежний граф (провал-закрытый гейт,
# форма ``flash_bwd_mode``).  Приоритет при одновременном включении с
# ``AXIOM_MLA_FLASH_BWD``/``AXIOM_MLA_BWD_BF16``: исполняется персистентный
# путь (обе ноги-маркеры несут фактические значения гейтов — A/B-ноги
# различимы по журналу); конфликт с ``AXIOM_MLA_DENSE_FLASH`` (два разных
# внимания на один интервал) — ошибка, не тихий выбор.
# --------------------------------------------------------------------------- #

#: Env-гейт персистентных кернелов внимания MLA (``0`` — дефолт, прежний путь).
MLA_PERSIST_ENV = "AXIOM_MLA_PERSIST"

#: Допустимые значения; неизвестное — ошибка (fail-closed, без догадок).
MLA_PERSIST_VALUES = ("0", "1")
MLA_PERSIST_OFF = "0"
MLA_PERSIST_ON = "1"


class MlaPersistError(ValueError):
    """Неизвестное значение ``AXIOM_MLA_PERSIST`` / конфликт путей (fail-closed)."""


def persist_mode(value: str | None = None) -> bool:
    """Персистентный MLA-блок: ``False`` (дефолт) | ``True``.

    ``AXIOM_MLA_PERSIST=1`` проводит интервал внимания dense-пути через
    :func:`net.kernels.mla_persist.mla_attention` (персистентные fwd/bwd
    кернелы).  Пустая/незаданная переменная = ``"0"``; значения — только
    ``"0"``/``"1"``; неизвестное — :class:`MlaPersistError` (fail-closed:
    опечатка не должна тихо оставить прежний путь — замер ноги «персистентный
    блок» стал бы замером dense-пути).  Явный аргумент сильнее окружения
    (форма ``flash_bwd_mode``).  Читается на трейсе (как flash-переключатели),
    чтобы тест мог переключить гейт в процессе.
    """
    raw = value if value is not None else os.environ.get(MLA_PERSIST_ENV)
    mode = (raw or "").strip().lower() or MLA_PERSIST_OFF
    if mode not in MLA_PERSIST_VALUES:
        raise MlaPersistError(
            f"{MLA_PERSIST_ENV}={mode!r}: ожидается одно из "
            f"{list(MLA_PERSIST_VALUES)}"
        )
    return mode == MLA_PERSIST_ON


def _mla_persist_attention(q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """Интервал внимания через персистентные кернелы (ленивый импорт модуля).

    Лениво — чтобы ``net.mla`` не тянул pallas/mosaic на путях, где гейт
    выключен (форма ленивого импорта ``net.kda``).  Ветка — только плотный
    causal-интервал: явная маска (неслучайный паттерн внимания) кернелами не
    поддержана — ошибка на трейсе, не тихий откат (см. вызов).
    """
    from .kernels import mla_persist as mla_persist_kernels

    return mla_persist_kernels.mla_attention(q, k, v)


# --------------------------------------------------------------------------- #

#: Env-гейт тайлового backward dense-внимания MLA (``0`` — дефолт, прежний путь).
MLA_FLASH_BWD_ENV = "AXIOM_MLA_FLASH_BWD"

#: Допустимые значения; неизвестное — ошибка (fail-closed, без догадок).
MLA_FLASH_BWD_VALUES = ("0", "1")
MLA_FLASH_BWD_OFF = "0"
MLA_FLASH_BWD_ON = "1"


class MlaFlashBwdError(ValueError):
    """Неизвестное значение ``AXIOM_MLA_FLASH_BWD`` (fail-closed, без догадок)."""


def flash_bwd_mode(value: str | None = None) -> bool:
    """Тайловый backward dense-внимания: ``False`` (дефолт) | ``True``.

    ``AXIOM_MLA_FLASH_BWD=1`` подменяет backward dense-пути на переисчисление
    скоров по (KV,Q)-тайлам (:func:`_attn_out_flashbwd_bwd`); forward-числа
    ветка не меняет (плотный forward дословно), граф backward теряет полную
    (B,H,T,T).  Пустая/незаданная переменная = ``"0"``; значения — только
    ``"0"``/``"1"``; неизвестное — :class:`MlaFlashBwdError` (fail-closed:
    опечатка не должна тихо оставить прежний путь — замер ноги «flash-backward»
    стал бы замером dense-backward).  Явный аргумент сильнее окружения (форма
    ``bwd_bf16_mode``).  Читается на трейсе (как flash-переключатель), чтобы
    тест мог переключить гейт в процессе.
    """
    raw = value if value is not None else os.environ.get(MLA_FLASH_BWD_ENV)
    mode = (raw or "").strip().lower() or MLA_FLASH_BWD_OFF
    if mode not in MLA_FLASH_BWD_VALUES:
        raise MlaFlashBwdError(
            f"{MLA_FLASH_BWD_ENV}={mode!r}: ожидается одно из "
            f"{list(MLA_FLASH_BWD_VALUES)}"
        )
    return mode == MLA_FLASH_BWD_ON


def _attn_scale(dq: int) -> jnp.ndarray:
    """``1/sqrt(dq)`` — дословно то же выражение, что умножает скоры dense-пути.

    Вычисляется от статического ``dq`` (Python int), поэтому значение и его
    dtype (weak f32) в flash-ветке и в прежнем пути совпадают побитово.
    """
    return 1.0 / jnp.sqrt(dq)


def _attn_out_flashbwd_fwd(q, k, v, mask, tile):
    """Forward тайлового внимания: плотные числа дословно + статистики строк.

    Операции — ровно те же, что у прежнего dense-пути (``gemm_einsum`` скор,
    ``where``-маска, ``softmax``, ``gemm_einsum`` контракция), поэтому числа
    побитово те же; сверх этого считаются rowmax/rowsum каждой строки (B,H,T,1)
    — всё, что backward хранит о скорах.  ``tile`` — статическая ширина тайла
    (нёс в замыкании фабрики :func:`_attn_out_flashbwd_op` в forward не нужен;
    параметр — часть контракта fwd/bwd).
    """
    dh = q.shape[-1]
    scale = _attn_scale(dh)
    scores = compute_dtype.gemm_einsum("bthd,bshd->bhts", q, k) * scale  # (B,H,T,T)
    masked = jnp.where(mask, scores, jnp.finfo(jnp.float32).min)
    attn = jax.nn.softmax(masked, axis=-1)  # (B,H,T,T)
    o = compute_dtype.gemm_einsum("bhts,bshd->bthd", attn, v)  # (B,T,H,dq)
    rowmax = jnp.max(masked, axis=-1, keepdims=True)  # (B,H,T,1)
    rowsum = jnp.sum(jnp.exp(masked - rowmax), axis=-1, keepdims=True)
    return o, (q, k, v, mask, o, rowmax, rowsum)


def _attn_out_flashbwd_bwd(res, g, tile):
    """Backward по тайлам: скоры переисчисляются, полная (B,H,T,T) не живёт.

    Внешний цикл — KV-тайлы, внутренний — Q-тайлы (порядок flash-attention
    backward): для пары тайлов строится скор (B,H,ti,tj), восстанавливается
    ``p`` по статистикам forward, считается ``ds`` и три вклада (dq/dk/dv)
    тайловыми матмулами с теми же boundary-кастами ``compute_dtype``, что у
    forward-контракций.  ``mask`` индексируется по последним двум осям, поэтому
    принимаются и (T,S), и широковещательные формы.  Котангент маски — None.
    """
    q, k, v, mask, o, rowmax, rowsum = res
    B, T, H, dh = q.shape
    scale = _attn_scale(dh)
    # D_i = rowsum_s(g_i ⊙ o_i) — поточечная свёртка, не тайловая: нужна до
    # циклов (в flash-attention backward это замена rowsum(dp ⊙ p) без второго
    # прохода по тайлам).
    d = jnp.transpose(jnp.sum(g * o, axis=-1), (0, 2, 1))  # (B,H,T)
    dq_acc = jnp.zeros_like(q)
    dk_acc = jnp.zeros_like(k)
    dv_acc = jnp.zeros_like(v)
    for j0 in range(0, T, tile):  # внешний цикл: KV-тайлы
        j1 = min(j0 + tile, T)
        kj = k[:, j0:j1]
        vj = v[:, j0:j1]
        dk_j = jnp.zeros((B, j1 - j0, H, dh), jnp.float32)
        dv_j = jnp.zeros((B, j1 - j0, H, dh), jnp.float32)
        for i0 in range(0, T, tile):  # внутренний цикл: Q-тайлы
            i1 = min(i0 + tile, T)
            qi = q[:, i0:i1]
            do_i = g[:, i0:i1]
            # Переисчисление скора тайла (та же контракция и тот же scale).
            s = compute_dtype.gemm_einsum("bthd,bshd->bhts", qi, kj) * scale
            p = (
                jnp.where(mask[..., i0:i1, j0:j1], jnp.exp(s - rowmax[:, :, i0:i1, :]), 0.0)
                / rowsum[:, :, i0:i1, :]
            )  # (B,H,ti,tj) — вероятности строк, восстановленные по статистикам
            dp = compute_dtype.gemm_einsum("bthd,bshd->bhts", do_i, vj)
            grad_s = p * (dp - d[:, :, i0:i1, None]) * scale
            dq_acc = dq_acc.at[:, i0:i1, :, :].add(
                compute_dtype.gemm_einsum("bhts,bshd->bthd", grad_s, kj)
            )
            dk_j = dk_j + compute_dtype.gemm_einsum("bhts,bthd->bshd", grad_s, qi)
            dv_j = dv_j + compute_dtype.gemm_einsum("bhts,bthd->bshd", p, do_i)
        dk_acc = dk_acc.at[:, j0:j1, :, :].add(dk_j)
        dv_acc = dv_acc.at[:, j0:j1, :, :].add(dv_j)
    return dq_acc, dk_acc, dv_acc, None


@functools.lru_cache(maxsize=None)
def _attn_out_flashbwd_op(tile: int):
    """custom_vjp-внимание тайлового backward, по объекту на ширину тайла.

    Кэш по ``tile`` — чтобы трейсы с одной шириной делили один объект (форма
    стабильности для jit-кэшей), а разные ширины не переиспользовали чужой
    замыкающий тайл.  ``tile`` — статический Python int конфига.
    """

    @jax.custom_vjp
    def op(q, k, v, mask):
        return _attn_out_flashbwd_fwd(q, k, v, mask, tile)[0]

    op.defvjp(
        lambda q, k, v, mask: _attn_out_flashbwd_fwd(q, k, v, mask, tile),
        lambda res, g: _attn_out_flashbwd_bwd(res, g, tile),
    )
    return op


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
    q = compute_dtype.gemm(x32, params.W_q.astype(jnp.float32)).reshape(*x32.shape[:-1], H, dq)
    k = compute_dtype.gemm(c, params.W_k_up.astype(jnp.float32)).reshape(*x32.shape[:-1], H, dq)
    v = compute_dtype.gemm(c, params.W_v_up.astype(jnp.float32)).reshape(*x32.shape[:-1], H, dq)
    return q, k, v


def _dense_apply(params: MLAParams, cfg: ModelConfig, x: jnp.ndarray, mask: jnp.ndarray | None = None) -> jnp.ndarray:
    """Dense causal gated MLA over ``(B, T, hidden)`` (the oracle, ADR-009 D4).

    ``AXIOM_MLA_DENSE_FLASH=1`` switches attention to the fused flash path
    (``jax.nn.dot_product_attention``): no ``(B,H,T,T)`` materialisation —
    the MFU experiment of 07.10 (memory-bound kernels, ~50 GB/step of
    fp32 scores+attn traffic at l3-full b1).  Diagnostic only; default OFF
    keeps the oracle byte-exact.

    Layout note (re-test 08.10, BF16-кампания фаза 1): ``dot_product_attention``
    takes ``(B, T, N, H)`` — sequence *before* heads — while the projections here
    are ``(B, T, H, dq)``.  The pre-re-test code transposed to ``(B, H, T, dq)``
    and the kernel then read *heads* as the sequence axis, which is why the old
    "parity 0.0" reading did not survive a re-measurement on a fresh checkout:
    the shapes happened to round-trip, the arithmetic did not.  The operands are
    therefore passed in their native ``(B, T, H, dq)`` order and the mask in the
    kernel's ``(B, T, S)`` broadcast form.
    """
    H, dq = cfg.num_heads, cfg.mla_head_dim
    x32 = x.astype(jnp.float32)
    c = compute_dtype.gemm(x32, params.W_c.astype(jnp.float32))  # (B, T, d_c)
    q, k, v = _project(params, cfg, x32, c)

    if persist_mode():
        # Веха-13 (ADR-054 Am.27): персистентные кернелы внимания MLA —
        # один запуск fwd и один bwd на слой-проход.  Кернел реализует
        # causal-интервал; явная маска (иной паттерн внимания) — ошибка, не
        # тихий откат.  С ``AXIOM_MLA_DENSE_FLASH`` (альтернативное внимание
        # на тот же интервал) — тоже ошибка: два пути одновременно не
        # выбираются молча (Am.27 стоп-условие «конфликт — эскалация»).
        if _flash_dense():
            raise MlaPersistError(
                "AXIOM_MLA_PERSIST=1 и AXIOM_MLA_DENSE_FLASH=1 одновременно: "
                "два несовместимых пути внимания MLA на один интервал; выберите "
                "один (персистентный блок — ступень-2 кампании).")
        if mask is not None:
            raise MlaPersistError(
                "AXIOM_MLA_PERSIST=1: персистентный кернел реализует только "
                "причинную маску; явная маска внимания не поддержана — тихого "
                "отката на dense-путь нет.")
        o = _mla_persist_attention(q, k, v)
    elif _flash_dense():
        # The kernel takes the mask as ``(B, N, T, S)`` with ``N`` the query-head
        # count (``jax/_src/nn/functions.py``); the caller's ``(T, S)`` boolean
        # mask is broadcast to that shape over batch and heads.
        m = (
            None
            if mask is None
            else jnp.broadcast_to(mask[None, None], (x.shape[0], H) + mask.shape)
        )
        o = compute_dtype.cast_out(
            jax.nn.dot_product_attention(
                compute_dtype.cast_in(q),
                compute_dtype.cast_in(k),
                compute_dtype.cast_in(v),
                mask=m,
                is_causal=mask is None,
            )
        )
    else:
        scale = 1.0 / jnp.sqrt(dq)
        if mask is None:
            mask = _causal_mask(x.shape[-2])
        if flash_bwd_mode():
            # Веха-12 (ADR-054 Am.25): плотный forward (числа побитово те же —
            # те же операции) + тайловый backward: скоры переисчитываются по
            # (KV,Q)-тайлам, полная (B,H,T,T) в backward не живёт.  Приоритет
            # над bf16-хранением: тайловый путь не хранит attn вовсе.
            o = _attn_out_flashbwd_op(int(cfg.mla_flash_bwd_tile))(q, k, v, mask)
        elif bwd_bf16_mode():
            # Веха-9 (ADR-054 Am.20): softmax-выход хранится для backward в
            # bf16 — forward-числа побитово те же (маскирование живёт внутри,
            # чтобы transpose `where` не оставлял лишний fp32-остаток), а
            # backward читает bf16-копию (см. _attn_out_bf16stored).
            # Ветка не трогает flash-путь выше.
            scores = compute_dtype.gemm_einsum("bthd,bshd->bhts", q, k) * scale
            o = _attn_out_bf16stored(scores, v, mask)
        else:
            scores = compute_dtype.gemm_einsum("bthd,bshd->bhts", q, k) * scale  # (B, H, T, T)
            scores = jnp.where(mask, scores, jnp.finfo(jnp.float32).min)
            attn = jax.nn.softmax(scores, axis=-1)  # (B, H, T, T)
            o = compute_dtype.gemm_einsum("bhts,bshd->bthd", attn, v)  # (B, T, H, dq)
    o = o.reshape(*x.shape[:-1], H * dq)

    gate = jax.nn.sigmoid(compute_dtype.gemm(x32, params.W_g.astype(jnp.float32)))
    return compute_dtype.gemm(gate * o, params.W_o.astype(jnp.float32))


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
    c = compute_dtype.gemm(x32, params.W_c.astype(jnp.float32))  # (B, T, d_c)
    if cfg.qat_kv_enabled:
        # D3: fake-quant the cached latent in MXFP4, dequantise before attention.
        c = quant.fake_quant_mxfp4_latent(c)
    q, k, v = _project(params, cfg, x32, c)

    window = int(cfg.swa_window)
    if window > 0:
        ks = compute_dtype.gemm(x32, params.W_swa_k.astype(jnp.float32)).reshape(*x32.shape[:-1], H, dq)
        vs = compute_dtype.gemm(x32, params.W_swa_v.astype(jnp.float32)).reshape(*x32.shape[:-1], H, dq)
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
        idx_q = compute_dtype.gemm(x32, params.W_idx_q.astype(jnp.float32)).reshape(*x32.shape[:-1], cfg.mla_index_heads, cfg.mla_index_dim)
        idx_k = compute_dtype.gemm(c, params.W_idx_k.astype(jnp.float32)).reshape(*x32.shape[:-1], cfg.mla_index_heads, cfg.mla_index_dim)

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
    gate = jax.nn.sigmoid(compute_dtype.gemm(x32, params.W_g.astype(jnp.float32)))
    return compute_dtype.gemm(gate * o, params.W_o.astype(jnp.float32)), pool_out


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
    c = compute_dtype.gemm(x32, params.W_c.astype(jnp.float32))
    if cfg.qat_kv_enabled:
        c = quant.fake_quant_mxfp4_latent(c)
    B, T = x.shape[:2]
    k_eff = min(int(cfg.mla_top_k), T)
    idx_q = compute_dtype.gemm(x32, params.W_idx_q.astype(jnp.float32)).reshape(B, T, cfg.mla_index_heads, cfg.mla_index_dim)
    idx_k = compute_dtype.gemm(c, params.W_idx_k.astype(jnp.float32)).reshape(B, T, cfg.mla_index_heads, cfg.mla_index_dim)
    sc = compute_dtype.gemm_einsum("bqhi,bshi->bqs", idx_q, idx_k) / cfg.mla_index_heads
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
    W_k = compute_dtype.gemm(params.W_c, params.W_k_up).astype(jnp.float32)
    W_v = compute_dtype.gemm(params.W_c, params.W_v_up).astype(jnp.float32)
    q = compute_dtype.gemm(x32, params.W_q.astype(jnp.float32)).reshape(*x.shape[:-1], H, dq)
    k = compute_dtype.gemm(x32, W_k).reshape(*x.shape[:-1], H, dq)
    v = compute_dtype.gemm(x32, W_v).reshape(*x.shape[:-1], H, dq)
    scale = 1.0 / jnp.sqrt(dq)
    scores = compute_dtype.gemm_einsum("bthd,bshd->bhts", q, k) * scale
    if mask is None:
        mask = _causal_mask(x.shape[-2])
    scores = jnp.where(mask, scores, jnp.finfo(jnp.float32).min)
    attn = jax.nn.softmax(scores, axis=-1)
    o = compute_dtype.gemm_einsum("bhts,bshd->bthd", attn, v).reshape(*x.shape[:-1], H * dq)
    gate = jax.nn.sigmoid(compute_dtype.gemm(x32, params.W_g.astype(jnp.float32)))
    return compute_dtype.gemm(gate * o, params.W_o.astype(jnp.float32))
