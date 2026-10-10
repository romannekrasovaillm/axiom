# Архитектурный контекст (epic-context)

Собран: 2026-10-10T13:04:42.789957465+00:00

Источники:
- /home/roman/axiom/evidence/mfu-55/mosaic/REPORT-rhs-layout.md
- /home/roman/axiom/evidence/mfu-55/mosaic/REPORT-diagnostics.md
- /home/roman/axiom/net/kernels/kda_ut_solve_mosaic.py

<!-- источник: /home/roman/axiom/evidence/mfu-55/mosaic/REPORT-rhs-layout.md -->

# Mosaic: раскладка RHS — решение открытого дефекта (GB10, jax 0.11.2)

**Дефект:** `UnsupportedTransferError: Tiled strides must be a multiple of the vector length, except for the load vectorized dimension` при загрузке правой части в `plgpu.mma`.

## Причина (установлена экспериментом)

`plgpu.mma(acc, a, b)` требует:
- `a` — `MMA_LHS`, форма `(m, k)`;
- `b` — `MMA_RHS`, форма `(k, n)`, **причём память RHS обязана быть `k`-контигуальной**, то есть исходный массив должен лежать в порядке **`(n, k)`** и подаваться через `.T`.

Проверено на GB10 (три варианта, один и тот же (M,K)=(128,128)):

## Рецепт (рабочий, проверен)

```python
def body(l_ref, bt_ref, o_ref, smem):        # l_ref (H,M,K); bt_ref (H,N,K) — ПРАВАЯ ЧАСТЬ ТРАНСПОНИРОВАНА в памяти
    h = jax.lax.axis_index("head")
    acc = plgpu.layout_cast(jnp.zeros((M, N), ACC), plgpu.Layout.MMA_ACC(DT))
    a   = plgpu.load(l_ref.at[h],    layout=plgpu.Layout.MMA_LHS(DT), optimized=False)
    b   = plgpu.load(bt_ref.at[h].T, layout=plgpu.Layout.MMA_RHS(DT), optimized=False)   # .T → (K,N), k-контигуально
    o_ref.at[h][...] = plgpu.mma(acc, a, b).astype(DT)

kernel = plgpu.kernel(body, out_type=ShapeDtypeStruct((H, M, N), DT),
                      scratch_types=[plgpu.SMEM((M, M), DT)],
                      compiler_params=plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Lane),
                      grid=(H,), grid_names=("head",))
# вызов: kernel(L, B.transpose(0, 2, 1))   ← B в памяти (H, C, D) → (H, D, C)
```

## Что это значит для KDA-ядра

В модели правая часть `B` имеет форму `(H, C, dk+dv)` — то есть `(k=C, n=D)`. Для Mosaic её нужно подавать **транспонированной в памяти**: `(H, D, C)`. Дополнительная копия на входе — цена раскладки; альтернатива (если профиль покажет, что копия дорога) — держать правую часть в таком виде уже в `make_inputs`.

<!-- источник: /home/roman/axiom/evidence/mfu-55/mosaic/REPORT-diagnostics.md -->

# Mosaic-ядро KDA: журнал диагностики GPU-пути (5 итераций, 10.10.2026)

Все дефекты выявлены **только исполнением/трассировкой на GB10** (jax 0.11.2, CC12.1). CPU-тесты их не ловят по построению.

| # | Симптом | Класс | Статус |
|---|---|---|---|
| 1 | `NameError: _mosaic_gpu` (+ `_require_gpu`, `dataclasses`) | имя не определено на GPU-пути | ✅ исправлено (тесты с fake-Mosaic) |
| 2 | `ValueError: grid_names must have the same length as grid` | создание ядра (Python) | ✅ исправлено (`grid_names=("head",)`) |
| 3 | `_mma_abstract_eval: too many values to unpack` | `plgpu.mma` требует 2D-операндов | ✅ исправлено (`axis_index("head")` + `ref.at[h]`) |
| 4 | `Incompatible shapes: lhs=(128,128), rhs=(256,128), acc=(128,256)` | лишняя транспозиция `b_h.T` (MMA_RHS ждёт `(k,n)`) | ✅ исправлено (на стенде, диагностически) |
| 5 | `UnsupportedTransferError: Tiled strides must be a multiple of the vector length` | **транспорт раскладок**: `plgpu.load(l_h.T, layout=MMA_RHS)` — транспонирование рефа даёт strides `(1, M)`, несовместимые с векторизованной загрузкой | ❌ ОТКРЫТО |

## Суть пятого дефекта

Для `rhs` нужен реф формы `(k, n)` с `k`-контигуальной памятью. В апстрим-тесте это выполнялось через `b_ref.T` при `b` формы `(n, k)` (маленький `n=8`) — у нас же `(m,k)`/`(k,n)` матрицы размера 128/256, и простое транспонирование рефа даёт strides, которые векторизованный transfer не принимает.

**Кандидаты решения (для следующей итерации):**
1. Подавать RHS не транспонированием рефа, а копией в SMEM с нужной раскладкой (`plgpu.copy_gmem_to_smem` + `wait_gmem_to_smem`, как в апстрим-тесте для TMA-путей);
2. Либо хранить правую часть в памяти уже в `(n, k)`-порядке (переупаковка в `make_inputs`), чтобы `.T` давал `k`-контигуальный доступ;
3. Либо `optimized=True` для этой загрузки (проверить, примет ли transfer с крупным выравниванием `M=128`, `N=256`).

## Честная оценка стоимости

5 итераций «дефект → прогон харнесса (~15–40 мин) → GPU-проверка (~10 мин)», и **ни один дефект не был виден на CPU** — то есть канал «исполнитель без GPU» структурно неэффективен для Mosaic-ядра: каждый цикл требует живого стенда. Ожидаемое число оставшихся итераций — 3–6 (транспорт раскладок, SMEM-бюджет, точность цепочки, детерминизм).

## Что предлагается (решение за архитектором/владельцем)

1. **Сначала перемерить кампанию на 0.11.2** (дёшево, ~30 мин стенда): прогнать 4–6 шагов l3-full в `~/venv-axiom-0112` и снять `cuGraphLaunch`, tok/s, MFU против потолка 0.10.2 (**377 ток/с = 1.16%**; 33 418 `cuGraphLaunch` = 77% времени). Если новый XLA сам снимает барьер диспетчеризации — ценность Mosaic-кернела вторична, и усилия перераспределяются.
2. **Mosaic-ядро продолжить после этого** — либо силами исполнителя с доступом к стенду (итерации станут минутами), либо довести диагностикой архитектора (5 дефектов уже так закрыты).

<!-- источник: /home/roman/axiom/net/kernels/kda_ut_solve_mosaic.py -->

"""Mosaic-перенос KDA-решения ``(I + L) X = B`` — контракт, backend-выбор и границы.

Стадия 5 ADR-052. Модуль даёт: совпадающий контракт с Triton-версией
:mod:`net.kernels.kda_ut_solve`, явный backend-выбор и само Mosaic-ядро —
Neumann-цепочку ``plgpu.mma`` через SMEM (:func:`_build_kernel`).

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

#: Допустимые зн

> **Контекст усечён** до 6000 символов; полные тексты — в файлах-источниках (см. MANIFEST.json).
