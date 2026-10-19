"""Compute-dtype gate — ``AXIOM_COMPUTE_DTYPE`` (BF16 campaign, phase 1).

The gate selects the dtype of a **GEMM boundary** in the model package.  It is
deliberately *not* a parameter dtype (that axis already exists in the training
loop: ``net/train_loop.py``'s ``param_dtype`` casts a *working copy* of the
master tree) and it is not the checkpoint format or the metrics-journal schema,
neither of which this module can reach.

Modes
-----

``fp32`` (default)
    The pinned compute path.  :func:`gemm` / :func:`gemm_einsum` return the
    expression the caller had before the gate existed (``a @ b`` /
    ``jnp.einsum(eq, *ops)``), so the graph is bit-for-bit the pre-gate graph
    (baseline ``773e389``).  The oracle path of ADR-009 is byte-exact under the
    gate off by construction, not by tolerance.

``bf16``
    The Beam recipe: the **operands** of a dense matmul or an attention
    contraction are cast to bf16 at the boundary, the product **accumulates and
    returns in fp32** (``preferred_element_type=jnp.float32``).  Nothing else
    changes dtype — the residual stream, the normalisation, the activations,
    the loss path and the optimizer's master tree stay fp32.  Concretely:

    * *master weights stay fp32* — the gate never touches a parameter leaf; it
      casts the operand of a GEMM, and the weight a caller passes in is cast
      inside the boundary only.  ``optimizer.init_state`` (momentum/AdamW) and
      the gradient accumulation in ``net/train_loop.py`` therefore keep their
      fp32 dtype;
    * *residual connections stay fp32* — every block returns an fp32 tensor
      (the GEMM output dtype is forced to fp32), so ``h + delta + corr`` adds
      fp32 to fp32;
    * *casts happen only at the GEMM input boundary* — no other site in the
      model package introduces a dtype conversion, so the loss path cannot pick
      up a silent promotion or demotion.

Fail-closed
-----------

An unrecognised ``AXIOM_COMPUTE_DTYPE`` value raises :class:`ComputeDtypeError`
instead of silently running fp32.  A typo that quietly fell back to fp32 would
invalidate a bf16 campaign measurement; a typo that quietly ran bf16 would
perturb the pinned oracle.  Neither is acceptable, so the gate refuses to guess
(the opposite of ``net/tests/conftest.py``'s backend selector, whose fallback
direction is safe because both branches are exercised by the gate profile).

Not covered (declared, so the boundary is reviewable): ``net/vit.py`` (the
vision tower runs on image tokens, not on the pretrain compute axis).  The
MXFP4 QAT quantiser (``net/quant.py``), the checkpoint format and the journal
schema are out of scope by the task and are not referenced here.

FP8-нога проекций (цикл-26 кампании MFU-55, ADR-054 Am.45)
----------------------------------------------------------

Отдельный гейт ``AXIOM_GEMM_FP8`` (default ``0``) включает FP8-границу
**только** для плотных проекций/MLP — тех callsite-ов, которые передают
``gemm(..., fp8=True)``.  Остальные потребители :func:`gemm` (CE/lm_head,
MoE, oracle-пути) остаются на прежнем fp32/bf16-пути побитово: тело
``_chunked_cross_entropy`` и голова не трогаются, а измеренный минус FP8-CE
(Am.44) не смешивается с замером FP8-проекций.  Forward FP8-границы —
``matmul(q(a), q(b))·(s_a·s_b)`` (e4m3, per-tensor скейлы, fp32-аккум),
backward — STE: градиенты считаются по неквантованным операндам на обычной
bf16/fp32-границе, скейлы в котангенты не текут.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp

#: The gate's environment variable.
MODE_ENV = "AXIOM_COMPUTE_DTYPE"

#: ``fp32`` — the pinned compute path (default; the pre-gate graph verbatim).
FP32 = "fp32"

#: ``bf16`` — bf16 GEMM operands with an fp32 accumulator (the Beam recipe).
BF16 = "bf16"

#: Declared modes, in the order the report prints them.
MODES = (FP32, BF16)

#: Mode when the variable is unset (or empty).
DEFAULT_MODE = FP32


class ComputeDtypeError(ValueError):
    """Unrecognised ``AXIOM_COMPUTE_DTYPE`` value (fail-closed, no guessing)."""


def mode() -> str:
    """The gate's mode: ``fp32`` | ``bf16`` (read per call, ADR-4 determinism).

    Read on every call rather than pinned into an import-time constant so a
    test, a bench script or a sweep can select the mode in-process without
    reloading modules.  The branch is resolved when a JAX graph is **traced**
    (the same idiom as ``net/mla.py``'s flash switch), so a compiled graph
    carries the mode it was traced with.

    The caveat that follows is worth stating: ``jax.jit``'s cache is keyed on the
    function and the argument signatures, *not* on this variable, so flipping the
    gate under a live jitted function would reuse the graph traced under the old
    mode.  Entry points must therefore set the variable **before** the first
    trace — which is what they all do: the acceptance tests call the eager
    functions, ``net/train_loop.train`` builds a fresh ``jit`` per call, and the
    BF16 instruments (``tools/mfu_bf16_protocol.py``,
    ``tools/loss_parity_bf16.py``) run every cell in a process of its own.
    """
    raw = os.environ.get(MODE_ENV)
    if raw is None:
        return DEFAULT_MODE
    value = raw.strip().lower()
    if not value:
        return DEFAULT_MODE
    if value not in MODES:
        raise ComputeDtypeError(
            f"{MODE_ENV}={raw!r}: неизвестный режим вычислительного dtype; "
            f"ожидается один из {list(MODES)}"
        )
    return value


def is_bf16() -> bool:
    """True when the gate selects the bf16 GEMM boundary."""
    return mode() == BF16


def compute_dtype() -> jnp.dtype:
    """The dtype of a GEMM *operand* under the current gate (the boundary type)."""
    return jnp.bfloat16 if is_bf16() else jnp.float32


def cast_in(x: jnp.ndarray) -> jnp.ndarray:
    """Cast a GEMM *operand* to the boundary dtype (identity under ``fp32``).

    For boundaries that are not expressible as :func:`gemm` / :func:`gemm_einsum`
    — e.g. a fused attention kernel that takes (and returns) its operands
    directly.  The ``fp32`` branch returns the array untouched, so the gate-off
    graph keeps the caller's own expression.
    """
    return x.astype(jnp.bfloat16) if is_bf16() else x


def cast_out(x: jnp.ndarray) -> jnp.ndarray:
    """Cast a boundary *result* back to fp32 (identity under ``fp32``).

    A fused kernel run in bf16 returns bf16; leaving that in the residual stream
    would let the next ``fp32 * bf16`` promotion happen outside a boundary.  The
    result of every GEMM boundary is fp32 by contract, so this restores it
    explicitly rather than relying on promotion.
    """
    return x.astype(jnp.float32) if is_bf16() else x


# --------------------------------------------------------------------------- #
# FP8-нога плотных проекций (цикл-26 кампании MFU-55, ADR-054 Am.45).
#
# Гейт отдельный от ``AXIOM_COMPUTE_DTYPE``: проекционные GEMM под гейтом
# считаются e4m3-матмулом с per-tensor скейлами (fp32-аккум), backward —
# STE по неквантованным операндам.  Включение — только на объявленных
# callsite-ах через ``gemm(..., fp8=True)``; CE/lm_head, MoE и oracle-пути
# остаются на прежнем пути побитово (Am.44: FP8-CE измерен минусом).
# --------------------------------------------------------------------------- #

#: Env-гейт FP8-границы плотных проекций (``0`` — дефолт, прежний путь).
GEMM_FP8_ENV = "AXIOM_GEMM_FP8"

#: Допустимые значения; неизвестное — ошибка (fail-closed, без догадок).
GEMM_FP8_VALUES = ("0", "1")
GEMM_FP8_OFF = "0"
GEMM_FP8_ON = "1"


class GemmFp8Error(ValueError):
    """Неизвестное значение ``AXIOM_GEMM_FP8`` (fail-closed, без догадок)."""


def gemm_fp8_mode(value: str | None = None) -> bool:
    """FP8-граница проекционных GEMM: ``False`` (дефолт) | ``True``.

    ``AXIOM_GEMM_FP8=1`` направляет ``gemm(..., fp8=True)`` в
    :func:`gemm_fp8`: forward — e4m3-матмул с per-tensor скейлами
    (fp32-аккум), backward — bf16/fp32-граница по неквантованным операндам
    (STE).  Пустая/незаданная переменная = ``"0"``; значения — только
    ``"0"``/``"1"`` (пробелы нормализуются); неизвестное —
    :class:`GemmFp8Error`.  Явный аргумент сильнее окружения (форма
    :func:`net.mla.persist_mode`/``ce_fp8_mode``).  Читается на трейсе, не
    кэшируется — A/B идёт на одной ревизии.
    """
    raw = value if value is not None else os.environ.get(GEMM_FP8_ENV)
    mode = (raw or "").strip().lower() or GEMM_FP8_OFF
    if mode not in GEMM_FP8_VALUES:
        raise GemmFp8Error(
            f"{GEMM_FP8_ENV}={mode!r}: ожидается одно из {list(GEMM_FP8_VALUES)}"
        )
    return mode == GEMM_FP8_ON


def _fp8_matmul(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """FP8-матмул ``(…, M, K) @ (K, N)`` с fp32-аккумулятором — шов стенда.

    На GPU это ``jnp.matmul(..., preferred_element_type=jnp.float32)``.  На
    CPU-интерпретаторе fp8-dot может не lowering-иться, поэтому тесты
    подменяют шов деквантованным f32-матмулом
    (``a.astype(f32) @ b.astype(f32)``): значения e4m3 представимы в f32
    точно, численный эффект квантизации сохраняется.  Подмена — единственное
    место, где тесты трогают вычислительный путь.
    """
    return jnp.matmul(a, b, preferred_element_type=jnp.float32)


def _gemm_regular(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Прежняя граница :func:`gemm` (побитово): fp32 — ``a @ b``, bf16 — каст.

    Вынесена отдельно, чтобы backward FP8-ноги не переиспользовал публичный
    :func:`gemm` (тот под гейтом снова ушёл бы в :func:`gemm_fp8` — градиент
    градиента квантовался бы, а STE требует неквантованных операндов).
    """
    if mode() != BF16:
        return a @ b
    return jnp.matmul(
        a.astype(jnp.bfloat16),
        b.astype(jnp.bfloat16),
        preferred_element_type=jnp.float32,
    )


def _gemm_fp8_forward(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Forward FP8-границы: ``q(a)·q(b)·(s_a·s_b)`` в fp32 (e4m3-операнды)."""
    s_a = fp8_scale(a)
    s_b = fp8_scale(b)
    c = _fp8_matmul(quantize_e4m3(a, s_a), quantize_e4m3(b, s_b))
    return c * (s_a * s_b)


def _gemm_fp8_fwd(a: jnp.ndarray, b: jnp.ndarray):
    """Правило forward custom_vjp: выход и остаток ``(a, b)`` для STE-backward.

    Скейлы в остаток не кладутся: backward их не читает (константы, STE).
    """
    return _gemm_fp8_forward(a, b), (a, b)


@jax.custom_vjp
def gemm_fp8(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Плотный GEMM с FP8-forward и STE-backward (цикл-26, ADR-054 Am.45).

    Граница ``(a, b) -> c``.  Forward: per-tensor скейлы
    ``s_a = max|a|/448``, ``s_b = max|b|/448`` (fp32, данные-зависимые),
    операнды ``clip(a/s_a, ±448)`` кастуются в e4m3, произведение
    масштабируется обратно в fp32.  Backward (:func:`_gemm_fp8_bwd`):
    производная квантизации — единица (STE), скейлы — константы, поэтому
    градиенты считаются по **неквантованным** ``a``/``b`` на обычной
    bf16/fp32-границе.

    Поддерживается ``a`` с произвольными ведущими осями и 2-D ``b`` (веса
    проекций): backward сводит ведущие оси ``a``/``g`` в матрицу
    ``(prod(…), ·)``.  Вызывается из :func:`gemm` только при гейте on и
    float-операндах; прямой вызов минует гейт (тестовый шов).
    """
    return _gemm_fp8_forward(a, b)


def _gemm_fp8_bwd(res, g):
    """STE-backward FP8-границы: градиенты по неквантованным операндам.

    ``ga = g @ bᵀ``, ``gb = aᵀ @ g`` (ведущие оси ``a``/``g`` свёрнуты в
    строки) на :func:`_gemm_regular` — bf16-операнды/fp32-аккум под
    ``AXIOM_COMPUTE_DTYPE=bf16``, иначе fp32.  Скейлов в остатке нет:
    производная ``max|·|`` не нужна (STE).
    """
    a, b = res
    ga = _gemm_regular(g, jnp.swapaxes(b, -1, -2))
    a2 = a.reshape(-1, a.shape[-1])
    g2 = g.reshape(-1, g.shape[-1])
    gb = _gemm_regular(jnp.swapaxes(a2, -1, -2), g2)
    # Контракт backward: котангент несёт dtype своего примала.
    return ga.astype(a.dtype), gb.astype(b.dtype)


gemm_fp8.defvjp(_gemm_fp8_fwd, _gemm_fp8_bwd)


def gemm(a: jnp.ndarray, b: jnp.ndarray, *, fp8: bool = False) -> jnp.ndarray:
    """Dense matmul with the gate's boundary cast.

    ``fp32`` returns ``a @ b`` — the caller's own expression, character for
    character, so the gate-off graph is identical to the pre-gate one.  ``bf16``
    casts both operands at the boundary and accumulates in fp32.

    ``fp8=True`` (только объявленные проекционные callsite-ы) под
    ``AXIOM_GEMM_FP8=1`` и float-операндах с 2-D ``b`` направляет вызов в
    :func:`gemm_fp8`; при гейте off — прежний путь побитово.  Дефолт
    ``fp8=False`` оставляет CE/lm_head, MoE и oracle-пути нетронутыми.
    """
    enabled = gemm_fp8_mode()  # fail-closed: опечатка гейта — ошибка всегда
    if (
        fp8
        and enabled
        and b.ndim == 2
        and a.ndim >= 2
        and jnp.issubdtype(a.dtype, jnp.floating)
        and jnp.issubdtype(b.dtype, jnp.floating)
    ):
        return gemm_fp8(a, b)
    return _gemm_regular(a, b)


def gemm_einsum(equation: str, *operands: jnp.ndarray) -> jnp.ndarray:
    """``jnp.einsum`` with the gate's boundary cast (attention/scores contractions).

    Used where the contraction is expressed as an einsum rather than ``@`` — the
    attention score and value matmuls of the dense oracle and of the sparse
    union attention.  The fp32 mode returns the caller's own ``jnp.einsum``
    call back, so the gate-off numerics and reduction order are unchanged.
    """
    if mode() != BF16:
        return jnp.einsum(equation, *operands)
    cast = tuple(op.astype(jnp.bfloat16) for op in operands)
    return jnp.einsum(equation, *cast, preferred_element_type=jnp.float32)


#: Максимум представимого e4m3fn (FP8-нога проекций, ADR-054 Am.45).
E4M3_MAX = 448.0


def fp8_scale(x: jnp.ndarray) -> jnp.ndarray:
    """Per-tensor масштаб e4m3 для ``x``: ``max|x| / 448`` в fp32.

    Данные-зависимый скейл (per-tensor max) FP8-ноги проекций: делит операнд
    на его максимум так, чтобы весь диапазон e4m3 был использован.  Нулевой
    тензор получает скейл ``1/448`` (деление на ноль не возникает; нули
    квантизуются в нули).  Возвращается скаляр fp32; в backward FP8-границы
    скейл — константа (STE), производная через него не идёт.
    """
    amax = jnp.max(jnp.abs(x.astype(jnp.float32)))
    safe = jnp.where(amax > 0, amax, jnp.ones_like(amax))
    return (safe / E4M3_MAX).astype(jnp.float32)


def quantize_e4m3(x: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
    """Квантизация ``x`` в e4m3: ``clip(x/s, -448, 448)`` → ``float8_e4m3fn``.

    Скейл приходит из :func:`fp8_scale` (в custom_vjp backward читается как
    константа — STE).  Арифметика скейла — fp32; клип обязателен: e4m3fn не
    имеет inf, значение за 448 стало бы NaN.
    """
    scaled = jnp.clip(
        x.astype(jnp.float32) / scale, -E4M3_MAX, E4M3_MAX
    )
    return scaled.astype(jnp.float8_e4m3fn)


def gemm_batched(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Batched matmul ``(..., M, K) @ (..., K, N)`` with the boundary cast.

    Exists because *some* einsum patterns that mix a batch axis into one operand
    only — the MoE expert application ``"nej,ejl->nel"`` is the case in this
    repo — lower to a ``dot_general`` the **CPU** backend refuses in
    bf16-with-fp32-accumulate ("Unsupported element type for DotThunk::Execute:
    BF16 x BF16 = F32"), while the equivalent ``jnp.matmul`` runs.  The math is
    the caller's; only the spelling differs, so the fp32 branch is the plain
    ``jnp.matmul`` and the gate-off graph stays the pre-gate one.
    """
    if mode() != BF16:
        return jnp.matmul(a, b)
    return jnp.matmul(
        a.astype(jnp.bfloat16),
        b.astype(jnp.bfloat16),
        preferred_element_type=jnp.float32,
    )
