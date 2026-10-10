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

**Проверка:** `compile_stage: ok`, `status: run-ok`, `max_abs 0.0079` на цепочке `(L@L)@L` (M=K=N=128, bf16, L = tril(ones,-1)*0.01). Артефакты: `neumann_chain_proto.py`, `neumann_chain_proto.json`.

## Следствие для переноса

1. Для Neumann-произведения нужен **SMEM-буфер на каждый промежуточный результат** (C×C или C×(dk+dv)) — либо один переиспользуемый, но с явной синхронизацией шагов (барьер/commit).
2. Точность цепочки ограничена bf16 на каждом шаге выгрузки (в прототипе rel ≈ 1e-2 относительно максимума) — для KDA это вопрос численной политики: либо аккумулятор в fp32 и выгрузка только там, где требует раскладка, либо переход на fp32-операнды, если `MMA_LHS/RHS` их поддержат на CC12.1 (не проверено).
3. Раскладки RHS: правая часть подаётся в памяти `(n, k)` и грузится с `b_ref.T`.
