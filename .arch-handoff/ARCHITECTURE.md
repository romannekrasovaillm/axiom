# Архитектурный контекст (epic-context)

Собран: 2026-10-10T09:52:19.486152894+00:00

Источники:
- /home/roman/axiom/evidence/mfu-55/mosaic/REPORT-mosaic-chain.md
- /home/roman/axiom/net/kernels/kda_ut_solve_mosaic.py
- /home/roman/axiom/docs/adr/ADR-052-kandidatnaya-liniya-jax-0-11-2-mosaic-gpu-ryadom-s-legacy-0-10-2-izolirovannoe-okruzhenie-capability-gate-staged-rollout.md

<!-- источник: /home/roman/axiom/evidence/mfu-55/mosaic/REPORT-mosaic-chain.md -->

# Mosaic: рабочий паттерн цепочки MMA-умножений (GB10, jax 0.11.2)

**Цель:** перенести KDA-решение `(I+L)X=B` (Neumann-произведение = **цепочка** умножений) на Mosaic. Одиночный `plgpu.mma` уже работал (`correctness-pass`), но цепочка упиралась в раскладки.

## Три барьера, снятые по очереди (все — на стенде)

| # | Симптом | Причина | Решение |
|---|---|---|---|
| 1 | `AttributeError` без сообщения на стадии компиляции | `kernel_fn.lower(...)` — у объекта `plgpu.kernel` нет метода `lower` | `jax.jit(kernel_fn).lower(...)` |
| 2 | `NotImplementedError: Cannot convert from TiledLayout(…warp_dims=(-7,)) to TiledLayout(…warp_dims=(-7, Replicated(times=1)))` | `layout_cast` **из `MMA_ACC` в `MMA_LHS` не поддерживается** — результат `mma` нельзя напрямую подать операндом | переход **через SMEM**: `smem[...] = p.astype(DT)` → `plgpu.load(smem, layout=MMA_LHS(DT), optimized=False)` |
| 3 | `AttributeError: 'ShapeDtypeStruct' object has no attribute 'get_ref_aval'` | `scratch_types` ожидает ref-типы, а не `ShapeDtypeStruct`; и это **список** | `scratch_types=[plgpu.SMEM((M, N), DT)]`, тело `body(in_ref, out_ref, smem)` |

## Рабочий рецепт (проверен на GB10)

```python
def body(L_ref, o_ref, smem):                      # scratch_types=[plgpu.SMEM((M,N),DT)]
    acc = plgpu.layout_cast(jnp.zeros((M,N), ACC), plgpu.Layout.MMA_ACC(DT))
    a   = plgpu.load(L_ref,   layout=plgpu.Layout.MMA_LHS(DT), optimized=False)
    b   = plgpu.load(L_ref.T, layout=plgpu.Layout.MMA_RHS(DT), optimized=False)   # RHS = (n,k)
    p   = plgpu.mma(acc, a, b)                     # ACC-раскладка, f32
    smem[...] = p.astype(DT)                       # ВЫГРУЗКА в SMEM — обязательный шаг цепочки
    p_lhs = plgpu.load(smem, layout=plgpu.Layout.MMA_LHS(DT), optimized=False)
    acc2  = plgpu.layout_cast(jnp.zeros((M,N), ACC), plgpu.Layout.MMA_ACC(DT))
    p2    = plgpu.mma(acc2, p_lhs, b)
    o_ref[...] = p2.astype(DT)

kernel = plgpu.kernel(body, out_type=..., scratch_types=[plgpu.SMEM((M,N), DT)],
                      compiler_params=plgpu.CompilerParams(
                          lowering_semantics=plgpu.LoweringSemantics.Lane))   # Lane, не Warpgroup
```

## Следствие для переноса

1. Для Neumann-произведения нужен **SMEM-буфер на каждый промежуточный результат** (C×C или C×(dk+dv)) — либо один переиспользуемый, но с явной синхронизацией шагов (барьер/commit).
2. Точность цепочки ограничена bf16 на каждом шаге выгрузки (в прототипе rel ≈ 1e-2 относительно максимума) — для KDA это вопрос численной политики: либо аккумулятор в fp32 и выгрузка только там, где требует раскладка, либо переход на fp32-операнды, если `MMA_LHS/RHS` их поддержат на CC12.1 (не проверено).
3. Раскладки RHS: правая часть подаётся в памяти `(n, k)` и грузится с `b_ref.T`.

<!-- источник: /home/roman/axiom/net/kernels/kda_ut_solve_mosaic.py -->

"""Mosaic-перенос KDA-решения ``(I + L) X = B`` — контракт, backend-выбор и границы.

Стадия 5 ADR-052. Этот модуль даёт **ту часть переноса, которая проверяема на
исполнителе**: совпадающий контракт с Triton-версией :mod:`net.kernels.kda_ut_solve`
и явный backend-выбор. Само Mosaic-ядро в этой дельте **не реализовано** — и это
зафиксировано явно, а не замаскировано заглушкой (C-007).

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

<!-- источник: /home/roman/axiom/docs/adr/ADR-052-kandidatnaya-liniya-jax-0-11-2-mosaic-gpu-ryadom-s-legacy-0-10-2-izolirovannoe-okruzhenie-capability-gate-staged-rollout.md -->

---
id: ADR-052
title: "Кандидатная линия JAX 0.11.2 (Mosaic GPU) рядом с legacy 0.10.2: изолированное окружение, capability gate, staged rollout"
status: Proposed
date: "2026-10-10"
depends_on: [ADR-041, ADR-050]
affects: [CMP-002, NFR-002]
spec_files: [ARCHITECTURE-SPINE.md]
---

# ADR-052. Кандидатная линия JAX 0.11.2 рядом с legacy 0.10.2

- Date: 2026-10-10
- Status: Proposed
- Маршрут значимости: Standard (смена стека исполнения — существенный NFR воспроизводимости; финансовое влияние через потолок MFU)

## Context

**Где мы.** Кампания MFU-55 измеренно упёрлась в потолок текущего стека: **377 ток/с = 1.16% MFU**, узкое место — **~6 000 запусков CUDA-графов на шаг** (33 418 `cuGraphLaunch` за 120 с = 91.86 с, 77% времени, при 1.04 с всей GPU-работы). Флагами это не управляется (`min_graph_size`, `command_buffer` — нулевой эффект), батч `B=2` не компилируется (`INTERNAL: Failed to get configs`). Ядра — не узкое место: собственный Pallas-кернел убрал 115 200 TRSM-ядер (493 173 → 9 333 запусков) и **не дал ни миллисекунды**.

**Что даёт смена линии.** На legacy-стеке (jax/jaxlib 0.10.2) Mosaic-путь тензорных ядер **недоступен**: капы показывают `mgpu_mma_bf16`/`mgpu_mma_fp8e4m3` = FAIL (`Layout.MMA_ACC` отсутствует), рабочий только Triton (`triton_dot`/

> **Контекст усечён** до 6000 символов; полные тексты — в файлах-источниках (см. MANIFEST.json).
