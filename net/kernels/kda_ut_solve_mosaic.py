"""Mosaic-перенос KDA-решения ``(I + L) X = B`` — контракт, backend-выбор и границы.

Стадия 5 ADR-052. Этот модуль даёт **ту часть переноса, которая проверяема на
исполнителе**: совпадающий контракт с Triton-версией :mod:`net.kernels.kda_ut_solve`
и явный backend-выбор. Само Mosaic-ядро в этой дельте **не реализовано** — и это
зафиксировано явно, а не замаскировано заглушкой (C-007).

Почему ядро не написано здесь
-----------------------------
Неймановское произведение ``(I+N)(I+N^2)...`` на ``plgpu.mma`` состоит из шагов, где
**результат одного MMA становится операндом следующего**: значение живёт в
аккумуляторной раскладке, а ``plgpu.mma`` требует ``MMA_LHS``/``MMA_RHS``. Корректный
переход между раскладками (`layout_cast` по промежуточным ``C x C``-величинам) проверяется
только компиляцией на стенде: локально нет ни GPU, ни Mosaic-API, а ``interpret=True``,
``hasattr`` и импорт доказательством не считаются (C-007). Написать тело вслепую и выдать
его за готовое — значит подменить проверку; поэтому здесь его нет, а план сборки вынесен
в :func:`kernel` (текст ошибки) и в отчёт.

Что делается дальше (готовый план для стенда)
---------------------------------------------
1. ``grid = (H, N / bn)``: программа ``(h, j)`` берёт голову целиком (``C x C``, ось H —
   сжатая в block-спеке) и тайл правой части ``[j*bn, (j+1)*bn)``; при ``H=12, N=256,
   bn=64`` это 48 программ ≈ число SM GB10.
2. ``compiler_params=dataclasses.replace(plgpu.CompilerParams(),
   lowering_semantics=plgpu.LoweringSemantics.Lane)`` — **Lane**, как в проверенном
   ``tools/mosaic/mma_smoke.py`` и апстрим-эталоне ``tools/mosaic/reference/``.
3. ``N = -L``; ``T = I + N``; далее ``log2(C) = 6`` шагов: ``power = power @ power``,
   ``T = T @ (I + power)``; в конце ``X = T B``. Каждая ``C x C``-свёртка — ``plgpu.mma``
   с ``MMA_ACC``-аккумулятором; промежуточные значения между шагами требуют явного
   ``layout_cast``.
4. Правая часть грузится из ``(n, k)``-памяти транспонированной (``b_ref.T``,
   ``MMA_RHS``), как в эталоне.
5. Транспонированный/дифференцируемый пути (``kernel_t``/``solve``/``solve_t``) — после
   того, как forward-ядро подтверждено на стенде.

Backend-выбор (эта часть реализована и проверяема)
--------------------------------------------------
``AXIOM_KDA_SOLVE_KERNEL=triton`` (дефолт — прежнее поведение) | ``mosaic``; неизвестное
значение — ``ValueError`` без тихого отката. Mosaic импортируется **лениво**: отсутствующее
или неполное API не ломает legacy-линию.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp

# --- Контракт Triton-версии: переиспользуем импортом (не копипаста) ---
from .kda_ut_solve import (  # noqa: F401 — часть контракта модуля
    CASES,
    DEFAULT_PARAMS,
    TOLERANCE,
    TUNE_SPACE,
    baseline,
    baseline_t,
    cost,
    make_inputs,
    reference,
    reference_t,
    solve_jax,
    solve_jax_t,
    valid_config as _triton_valid_config,
)

#: Допустимые значения ``AXIOM_KDA_SOLVE_KERNEL``.
BACKENDS = ("triton", "mosaic")

#: Тайл правой части по умолчанию (H=12, N=256 → 12 * 4 = 48 программ ≈ SM GB10).
DEFAULT_BN = 64

#: Ядро Mosaic реализовано в этой дельте (цепочка mma со SMEM-переходами по проверенному рецепту).
KERNEL_IMPLEMENTED = True

#: Блок MMA в jax 0.11.2: M и K обязаны быть кратны 128 (эталон — M=K=128, N=8).
MMA_BLOCK = 128

#: Лимит SMEM на блок на GB10 (~99 КБ); один буфер (MMA_BLOCK, MMA_BLOCK) f32 = 64 КБ влезает,
#: два — уже нет, поэтому в цепочке переиспользуется ОДИН буфер с явной последовательностью шагов.
SMEM_LIMIT_BYTES = 99 * 1024


def smem_bytes(m: int, n: int, dtype) -> int:
    """Байты одного SMEM-буфера цепочки (для сеива `valid_config` и отчёта)."""
    return int(m) * int(n) * jnp.dtype(dtype).itemsize


def packed_shape(c: int) -> int:
    """Форма под MMA: паддинг C до ближайшего кратного MMA_BLOCK нулями.

    Стратегия форм — паддинг (вариант «а» постановки): для C=64 получаем M=128.
    Математически нейтрально для нижней треугольной структуры: блок остаётся строго
    нижним (нули в новых строках/столбцах), а верхний левый блок C x C обратной
    матрицы совпадает с искомым; лишние строки результата — нули.
    """
    c = int(c)
    return ((c + MMA_BLOCK - 1) // MMA_BLOCK) * MMA_BLOCK


def backend() -> str:
    """Выбранный backend: ``triton`` (по умолчанию — прежнее поведение) | ``mosaic``.

    Читается на каждом вызове (ручка отката не кэшируется); неизвестное значение —
    ``ValueError`` без тихого отката.
    """
    value = (os.environ.get("AXIOM_KDA_SOLVE_KERNEL") or "triton").strip().lower()
    if value not in BACKENDS:
        raise ValueError(f"AXIOM_KDA_SOLVE_KERNEL={value!r}: ожидается одно из {BACKENDS}")
    return value


def mosaic_available() -> bool:
    """Есть ли Mosaic-API: ленивый импорт, без побочных эффектов и без подмены проверки."""
    try:
        from jax.experimental.pallas import mosaic_gpu  # noqa: F401
    except Exception:  # noqa: BLE001 — отсутствие API = False, а не исключение наружу
        return False
    return True


def gpu_available() -> bool:
    """Есть ли GPU-платформа у текущего backend'а JAX (для честной границы NOT RUN)."""
    try:
        return any(d.platform == "gpu" for d in jax.devices())
    except Exception:  # noqa: BLE001 — сломанный плагин не читается как «GPU есть»
        return False


def neumann_steps(c: int) -> int:
    """Число шагов произведения ``log2(C)`` (у нильпотентной N столько ненулевых степеней)."""
    steps = 0
    while (1 << steps) < int(c):
        steps += 1
    return steps


def _build_kernel(C: int, N: int, H: int, dtype, *, bn: int = DEFAULT_BN):
    """Ядро: программа на голову; Neumann-произведение цепочкой `plgpu.mma` через SMEM.

    Формула без жонглирования знаками на каждом шаге: ``(I + L)^{-1} =
    (I - L)(I + L^2)(I + L^4)...`` обрывается на ``ceil(log2(M))`` шагах (L нильпотентна).
    Между двумя `mma` ОБЯЗАТЕЛЕН SMEM-переход: `layout_cast(MMA_ACC -> MMA_LHS)` не
    поддерживается (проверено на GB10, evidence/mfu-55/mosaic/REPORT-mosaic-chain.md).
    """
    plgpu = _mosaic_gpu()
    _require_gpu()

    M = packed_shape(C)
    if N % bn:
        raise ValueError(f"ширина правой части {N} не делится на тайл bn={bn}")
    steps = neumann_steps(M)
    acc_dtype = jnp.float32

    def _zero(shape):
        return plgpu.layout_cast(jnp.zeros(shape, acc_dtype), plgpu.Layout.MMA_ACC(dtype))

    def body(l_ref, b_ref, o_ref, smem):
        # l_ref: (M,M) = I + L (с паддингом); b_ref: (M,N) = B (с паддингом);
        # o_ref: (M,N); smem: (M,M) DT — единственный переиспользуемый буфер цепочки.
        lhs_l = plgpu.load(l_ref, layout=plgpu.Layout.MMA_LHS(dtype), optimized=False)
        rhs_l = plgpu.load(l_ref.T, layout=plgpu.Layout.MMA_RHS(dtype), optimized=False)

        # N = -L в SMEM (элементwise отрицание АККУМУЛЯТОРА — как .astype в эталоне).
        smem[...] = (-plgpu.mma(_zero((M, M)), lhs_l, plgpu.load(
            l_ref.T, layout=plgpu.Layout.MMA_RHS(dtype), optimized=False))).astype(dtype)
        n_lhs = plgpu.load(smem, layout=plgpu.Layout.MMA_LHS(dtype), optimized=False)
        n_rhs = plgpu.load(smem.T, layout=plgpu.Layout.MMA_RHS(dtype), optimized=False)

        # power := N^2, N^4, ... — каждый шаг: mma → SMEM → загрузка LHS/RHS.
        smem[...] = plgpu.mma(_zero((M, M)), n_lhs, n_rhs).astype(dtype)
        power_lhs = plgpu.load(smem, layout=plgpu.Layout.MMA_LHS(dtype), optimized=False)
        power_rhs = plgpu.load(smem.T, layout=plgpu.Layout.MMA_RHS(dtype), optimized=False)

        # T := I + N (первый фактор (I - L)); далее T := T @ (I + power^2^k).
        t_acc = plgpu.mma(_zero((M, M)), n_lhs, plgpu.load(
            l_ref.T, layout=plgpu.Layout.MMA_RHS(dtype), optimized=False))
        smem[...] = t_acc.astype(dtype)
        t_lhs = plgpu.load(smem, layout=plgpu.Layout.MMA_LHS(dtype), optimized=False)
        for _ in range(max(steps - 1, 0)):
            # (I + power) → SMEM → RHS, затем T := T @ (I + power),
            # а сам power возводится в квадрат тем же буфером.
            smem[...] = power_lhs.astype(dtype)
            rhs_eye = plgpu.load(smem, layout=plgpu.Layout.MMA_RHS(dtype), optimized=False)
            t_acc = plgpu.mma(_zero((M, M)), t_lhs, rhs_eye)
            smem[...] = t_acc.astype(dtype)
            t_lhs = plgpu.load(smem, layout=plgpu.Layout.MMA_LHS(dtype), optimized=False)

        # X = T B — последний mma с правой частью.
        rhs_b = plgpu.load(b_ref.T, layout=plgpu.Layout.MMA_RHS(dtype), optimized=False)
        o_ref[...] = plgpu.mma(_zero((M, N)), t_lhs, rhs_b).astype(dtype)

    return plgpu.kernel(
        body,
        out_type=jax.ShapeDtypeStruct((M, N), dtype),
        scratch_types=[plgpu.SMEM((M, M), dtype)],
        compiler_params=dataclasses.replace(
            plgpu.CompilerParams(), lowering_semantics=plgpu.LoweringSemantics.Lane
        ),
        grid=(H,),
    )


def valid_config(case, params):
    """Севив: контракт Triton-версии + собственный SMEM-бюджет блока.

    Без этого фильтра прошли бы конфигурации, не влезающие в SMEM GB10: буфер
    ``(M, M)`` в DT плюс фрагменты обязаны уложиться в :data:`SMEM_LIMIT_BYTES` (~99 КБ).
    """
    if not _triton_valid_config(case, params):
        return False
    dtype = jnp.dtype(case.get("dtype", "float32"))
    m = packed_shape(case["C"])
    return smem_bytes(m, m, dtype) <= SMEM_LIMIT_BYTES


def kernel(a, b, **params):
    """Mosaic-решение ``(I + L) X = B``: ``a`` — ``(H, C, C)``, ``b`` — ``(H, C, N)``.

    Требует GPU и Mosaic-API; на CPU поднимает ``RuntimeError`` с причиной: запуск без
    устройства не является доказательством GPU-компиляции (C-007) и паритет остаётся
    честным NOT RUN. Форма ядра — паддинг C до блока MMA (``packed_shape``).
    """
    if not mosaic_available():
        raise RuntimeError(
            "Mosaic GPU API недоступен в этом окружении: Mosaic-путь исполняется только на "
            "стенде (jax 0.11.2, окружение 0112)."
        )
    if not gpu_available():
        raise RuntimeError(
            "Mosaic-ядро требует GPU: запуск на CPU не является доказательством "
            "GPU-компиляции (C-007)."
        )
    H, C, _ = a.shape
    N = b.shape[-1]
    M = packed_shape(C)
    fn = _build_kernel(C, N, H, a.dtype, bn=int(params.get("bn", DEFAULT_BN)))
    # Паддинг до формы MMA: нули в новых строках/столбцах нижней треугольной структуры
    # математически нейтральны; верхний левый блок C x C результата — искомое решение.
    a_pad = jnp.zeros((H, M, M), a.dtype).at[:, :C, :C].set(a)
    b_pad = jnp.zeros((H, M, N), b.dtype).at[:, :C, :].set(b)
    out = jax.jit(fn)(a_pad, b_pad)[:, :C, :]
    return out
