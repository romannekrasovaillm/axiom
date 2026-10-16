"""Mosaic-перенос KDA-решения ``(I + L) X = B`` — контракт, backend-выбор и границы.

Стадия 5 ADR-052. Модуль даёт: совпадающий контракт с Triton-версией
:mod:`net.kernels.kda_ut_solve`, явный backend-выбор и само Mosaic-ядро —
Neumann-цепочку ``plgpu.mma`` через SMEM (:func:`_build_kernel`).

Контракт сетки 0.11.2 (рецепт проверен на GB10, ``evidence/mfu-55/mosaic/REPORT-mosaic-chain.md``)
-------------------------------------------------------------------------------------------------
``grid=(H,)`` требует ``grid_names`` той же длины (иначе ``plgpu.kernel`` падает
``ValueError`` уже при создании ядра — ``Mesh``), а рефы внутри ``body`` — **глобальные**
массивы с осью головы (``(H, M, M)``/``(H, N, M)``): ``plgpu.mma`` принимает только
2D-операнды. Поэтому одна программа на голову, голова выбирается осью
``jax.lax.axis_index("head")`` и срезом ``ref.at[h]`` — внутри программы всё 2D.

Три требования раскладок ``plgpu.mma`` (рецепт: ``evidence/mfu-55/mosaic/REPORT-rhs-layout.md``)
-----------------------------------------------------------------------------------------------
``plgpu.mma(acc, a, b)`` считает ``acc + a @ b`` и проверяет ровно эти три раскладки:

* **LHS** — логический ``(m, k)`` с ``k``-контигуальной памятью: row-major источник
  ``(m, k)`` грузится **без транспозиции** (``plgpu.load(l_ref.at[h], MMA_LHS)``).
* **RHS** — логический ``(k, n)``, но **память обязана быть ``k``-контигуальной**: исходный
  массив должен лежать как ``(n, k)`` и грузиться через ``.T``
  (``plgpu.load(bt_ref.at[h].T, MMA_RHS)``). Именно поэтому правая часть задачи
  ``B: (H, C, D) = (k, n)`` подаётся в ядро **транспонированной в памяти** —
  ``b.transpose(0, 2, 1)`` даёт ``(H, D, C) = (n, k)``. Простое ``.T`` от row-major
  ``(k, n)`` даёт несовместимые strides (``UnsupportedTransferError``); не то ``.T``
  (лишняя/пропущенная транспозиция) — ``Incompatible shapes`` (дефект №4).
* **ACC** — логический ``(m, n)`` (``MMA_ACC``), инициализируется ``layout_cast``.

Для квадратных ``(M, M)`` промежуточных матриц цепочки транспозиция в памяти не нужна, но
операнд RHS всё равно грузится ``.T`` от row-major буфера — тогда ``k``-контигуальность
выполняется автоматически (проверено на GB10, ``REPORT-mosaic-chain.md``).

Что проверяемо локально (без устройства) и как
----------------------------------------------
``net/tests/test_kda_mosaic_parity.py`` исполняет сборку и **трассировку** ядра на
CPU-буферах: реальная машинерия ``plgpu.kernel``/mpmd + примитивы с контрактом 0.11.2
(``mma``/``load``/``layout_cast``/``Layout.MMA_*`` подменены шимами — их нет или они
другие в локальном jax 0.10.2). Трассировка ловит GPU-путевые классы: пропущенный
``grid_names`` (создание ядра), не-2D операнд ``plgpu.mma`` (``too many values to
unpack`` из ``_mma_abstract_eval``) и неверную раскладку RHS (``k`` не совпал — как
``Incompatible shapes`` на стенде); статически то же ловят AST-проверки вызова и
транспозиции RHS.

Дальше (стенд, не здесь)
------------------------
Прогон ``AXIOM_KDA_SOLVE_KERNEL=mosaic`` на GB10: компиляция sm_121, IR/PTX = MMA,
численный паритет с Triton-версией, время и детерминизм.

Backend-выбор
-------------
``AXIOM_KDA_SOLVE_KERNEL=triton`` (дефолт — прежнее поведение) | ``mosaic``; неизвестное
значение — ``ValueError`` без тихого отката. Mosaic импортируется **лениво**:
отсутствующее или неполное API не ломает legacy-линию.
"""

from __future__ import annotations

import dataclasses
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


#: Кэш ленивого импорта Mosaic-API (``None`` — ещё не импортировали).
_MOSAIC_GPU = None


def _mosaic_gpu():
    """Ленивый геттер Mosaic-API: импорт ровно один раз, далее — из кэша.

    Возвращает модуль ``jax.experimental.pallas.mosaic_gpu``. Отсутствующее или
    сломанное API — понятная ошибка с причиной; **тихого отката на Triton нет**
    (backend выбирается явно через ``AXIOM_KDA_SOLVE_KERNEL`` и под капотом не
    подменяется — C-007).
    """
    global _MOSAIC_GPU
    if _MOSAIC_GPU is None:
        try:
            from jax.experimental.pallas import mosaic_gpu as plgpu
        except Exception as exc:  # noqa: BLE001 — отсутствие API = ошибка с причиной
            raise RuntimeError(
                "Mosaic GPU API недоступен в этом окружении: не удалось импортировать "
                f"jax.experimental.pallas.mosaic_gpu ({exc!r}). Mosaic-путь исполняется "
                "только на стенде (jax 0.11.2, окружение 0112); откат на Triton не "
                "выполняется — backend выбирается явно."
            ) from exc
        _MOSAIC_GPU = plgpu
    return _MOSAIC_GPU


def _require_gpu() -> None:
    """Guard: сборка Mosaic-ядра без устройства бессмысленна (C-007).

    Единый источник ошибки и для :func:`kernel`, и для :func:`_build_kernel`, чтобы
    причина отказа совпадала на обоих входах.
    """
    if not gpu_available():
        raise RuntimeError(
            "Mosaic-ядро требует GPU: запуск на CPU не является доказательством "
            "GPU-компиляции (C-007)."
        )


def neumann_steps(c: int) -> int:
    """Число шагов произведения ``log2(C)`` (у нильпотентной N столько ненулевых степеней)."""
    steps = 0
    while (1 << steps) < int(c):
        steps += 1
    return steps


def _build_kernel(C: int, N: int, H: int, dtype, *, bn: int = DEFAULT_BN):
    """Ядро: одна программа на голову; Neumann-произведение цепочкой `plgpu.mma` через SMEM.

    Контракт 0.11.2 (проверен на GB10, evidence/mfu-55/mosaic/REPORT-mosaic-chain.md):
    ``grid=(H,)`` обязан нести ``grid_names``; рефы внутри ``body`` — глобальные массивы
    с осью головы, а ``plgpu.mma`` требует 2D-операнды, поэтому голова выбирается осью
    ``jax.lax.axis_index("head")`` и срезом ``ref.at[h]`` — внутри программы всё 2D.

    Раскладка RHS (``evidence/mfu-55/mosaic/REPORT-rhs-layout.md``): правая часть подаётся
    транспонированной в памяти (``bt_ref: (H, N, M)``, оси ``(n, k)``) и грузится через
    ``.T`` — иначе память RHS не ``k``-контигуальна и ``plgpu.mma`` падает
    ``UnsupportedTransferError``. LHS — без транспозиции.

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

    def body(l_ref, bt_ref, o_ref, smem):
        # l_ref: (H, M, M) = I + L (паддинг C->M по оси k); bt_ref: (H, N, M) = правая часть
        # В ПАМЯТИ (оси (n, k) = (D, C->M)); o_ref: (H, M, N) — выход всех голов;
        # smem: (M, M) DT — буфер цепочки на программу.
        # Рефы видны целиком (глобальные), программа берёт свою голову осью сетки:
        # срез по `ref.at[h]` — единственный способ сделать операнды mma 2D.
        h = jax.lax.axis_index("head")
        l_h = l_ref.at[h]   # (M, M): LHS (m, k) — k-контигуальна, транспозиция не нужна
        o_h = o_ref.at[h]   # (M, N); bt_ref — RHS в памяти (H, N, M) = (n, k), грузится `.T`
        lhs_l = plgpu.load(l_h, layout=plgpu.Layout.MMA_LHS(dtype), optimized=False)
        rhs_l = plgpu.load(l_h.T, layout=plgpu.Layout.MMA_RHS(dtype), optimized=False)

        # N = -L в SMEM (элементwise отрицание АККУМУЛЯТОРА — как .astype в эталоне).
        smem[...] = (-plgpu.mma(_zero((M, M)), lhs_l, plgpu.load(
            l_h.T, layout=plgpu.Layout.MMA_RHS(dtype), optimized=False))).astype(dtype)
        n_lhs = plgpu.load(smem, layout=plgpu.Layout.MMA_LHS(dtype), optimized=False)
        n_rhs = plgpu.load(smem.T, layout=plgpu.Layout.MMA_RHS(dtype), optimized=False)

        # power := N^2, N^4, ... — каждый шаг: mma → SMEM → загрузка LHS/RHS.
        smem[...] = plgpu.mma(_zero((M, M)), n_lhs, n_rhs).astype(dtype)
        power_lhs = plgpu.load(smem, layout=plgpu.Layout.MMA_LHS(dtype), optimized=False)
        power_rhs = plgpu.load(smem.T, layout=plgpu.Layout.MMA_RHS(dtype), optimized=False)

        # T := I + N (первый фактор (I - L)); далее T := T @ (I + power^2^k).
        t_acc = plgpu.mma(_zero((M, M)), n_lhs, plgpu.load(
            l_h.T, layout=plgpu.Layout.MMA_RHS(dtype), optimized=False))
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

        # X = T B — последний mma с правой частью. Источник RHS лежит в памяти (n, k),
        # поэтому `.T` даёт логический (k, n) с k-контигуальной памятью (рецепт GB10).
        rhs_b = plgpu.load(bt_ref.at[h].T, layout=plgpu.Layout.MMA_RHS(dtype), optimized=False)
        o_h[...] = plgpu.mma(_zero((M, N)), t_lhs, rhs_b).astype(dtype)

    return plgpu.kernel(
        body,
        # Выход — все головы сразу (каждая программа пишет свой срез o_ref.at[h]).
        out_type=jax.ShapeDtypeStruct((H, M, N), dtype),
        scratch_types=[plgpu.SMEM((M, M), dtype)],
        compiler_params=dataclasses.replace(
            plgpu.CompilerParams(), lowering_semantics=plgpu.LoweringSemantics.Lane
        ),
        grid=(H,),
        grid_names=("head",),
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
    честным NOT RUN. Форма ядра — паддинг C до блока MMA (``packed_shape``); ядро
    возвращает ``(H, M, N)``, наружу отдаётся ``(H, C, N)``.

    Внутри ``b`` подаётся **транспонированной в памяти** — ``b.transpose(0, 2, 1)``
    (``(H, C, N) -> (H, N, C)``): только так ``bt_ref.at[h].T`` даёт ``k``-контигуальную
    память RHS, которую требует ``plgpu.mma`` (рецепт GB10,
    ``evidence/mfu-55/mosaic/REPORT-rhs-layout.md``). Копия на входе — цена раскладки,
    математику не меняет.
    """
    if not mosaic_available():
        raise RuntimeError(
            "Mosaic GPU API недоступен в этом окружении: Mosaic-путь исполняется только на "
            "стенде (jax 0.11.2, окружение 0112)."
        )
    _require_gpu()
    H, C, _ = a.shape
    N = b.shape[-1]
    M = packed_shape(C)
    fn = _build_kernel(C, N, H, a.dtype, bn=int(params.get("bn", DEFAULT_BN)))
    # Паддинг до формы MMA: нули в новых строках/столбцах нижней треугольной структуры
    # математически нейтральны; верхний левый блок C x C результата — искомое решение.
    a_pad = jnp.zeros((H, M, M), a.dtype).at[:, :C, :C].set(a)
    # Правая часть уходит в ядро транспонированной: память (H, N, C) = (n, k), паддинг
    # C->M применяется к оси k (последней). Без транспозиции последний RHS-операнд
    # оказался бы (n, k) вместо (k, n) — `plgpu.mma` падает `Incompatible shapes`
    # (дефект №4), а `.T` от row-major (k, n) — `UnsupportedTransferError` (дефект №5).
    bt = b.transpose(0, 2, 1)  # (H, C, N) -> (H, N, C)
    bt_pad = jnp.zeros((H, N, M), b.dtype).at[:, :, :C].set(bt)
    out = jax.jit(fn)(a_pad, bt_pad)[:, :C, :]
    return out
