# Mosaic: раскладка RHS — решение открытого дефекта (GB10, jax 0.11.2)

**Дефект:** `UnsupportedTransferError: Tiled strides must be a multiple of the vector length, except for the load vectorized dimension` при загрузке правой части в `plgpu.mma`.

## Причина (установлена экспериментом)

`plgpu.mma(acc, a, b)` требует:
- `a` — `MMA_LHS`, форма `(m, k)`;
- `b` — `MMA_RHS`, форма `(k, n)`, **причём память RHS обязана быть `k`-контигуальной**, то есть исходный массив должен лежать в порядке **`(n, k)`** и подаваться через `.T`.

Проверено на GB10 (три варианта, один и тот же (M,K)=(128,128)):

| Как подавали RHS | Результат |
|---|---|
| `b_ref.at[h]` где `b` в памяти `(k, n)` = `(128, N)` | ❌ `UnsupportedTransferError` (strides не векторизуются) |
| `b_ref.at[h].T` где `b` в памяти `(k, n)` | ❌ `ValueError: Incompatible shapes … rhs=(N, 128)` (mma видит k=N) |
| **`b_ref.at[h].T` где `b` в памяти `(n, k)` = `(N, 128)`** | ✅ **ok для N=8, 64, 256**, `max_abs 6.1e-05` |

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

Артефакт: `mma_rhs_layout_proto.py` + `.json` (три N, все ok).

## Что это значит для KDA-ядра

В модели правая часть `B` имеет форму `(H, C, dk+dv)` — то есть `(k=C, n=D)`. Для Mosaic её нужно подавать **транспонированной в памяти**: `(H, D, C)`. Дополнительная копия на входе — цена раскладки; альтернатива (если профиль покажет, что копия дорога) — держать правую часть в таком виде уже в `make_inputs`.
