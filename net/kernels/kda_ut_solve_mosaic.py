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
    valid_config,
)

#: Допустимые значения ``AXIOM_KDA_SOLVE_KERNEL``.
BACKENDS = ("triton", "mosaic")

#: Тайл правой части по умолчанию (H=12, N=256 → 12 * 4 = 48 программ ≈ SM GB10).
DEFAULT_BN = 64

#: Ядро Mosaic в этой дельте не реализовано — модуль объявляет это, а не притворяется.
KERNEL_IMPLEMENTED = False


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


def kernel(a, b, **params):
    """Mosaic-ядро ``(I + L) X = B`` — в этой дельте НЕ реализовано.

    Вызов поднимает ``RuntimeError`` с причиной и планом сборки, вместо того чтобы
    подставить CPU-путь, ``interpret=True`` или фиктивную арифметику. Так паритет
    остаётся честным «NOT RUN», а не «прошло» без устройства (C-007).
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
    raise RuntimeError(
        "Mosaic-ядро не реализовано в этой дельте (KERNEL_IMPLEMENTED=False): план — grid "
        "(H, N/bn) со сжатой осью головы, Lane-семантика, неймановское произведение из "
        "log2(C) шагов plgpu.mma и финальное X = T B; промежуточные C x C-величины требуют "
        "явного layout_cast между MMA_LHS/RHS и MMA_ACC — это проверяется только "
        "компиляцией на стенде. Транспонированный/дифференцируемый пути — следующая дельта."
    )
