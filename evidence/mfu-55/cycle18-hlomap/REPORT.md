# Цикл-18 кампании MFU-55 — карта kernel→HLO и скрининг XLA-флагов

- Дата: 18.10.2026, стенд GB10 (`spark-44c3`), JAX 0.11.2 / XLA 0.11.2, nsys 2025.3.2.
- Основание: ADR-054 Am.29 п.1 (карта kernel→HLO до фьюзинга), Am.30.
- Клетка: `l3-full` B=4 plain, persist-on (`AXIOM_MLA_PERSIST=1`,
  `AXIOM_KDA_CC_IMPL=persistent`), bf16-compute, `per_layer_block`, CE 2048.
- Стенд освобождён владельцем на неделю (18–25.10.2026); сервисы llama остановлены
  на прогоны. Кодовых правок цикл не делает.

## 1. HLO-дамп и карта kernel→HLO

Дамп: `~/axiom-run/xladump-c18` (2421 файл, 1.1 ГБ; свежий JAX-кэш — иначе
компиляция не пишет HLO). Ключевой модуль —
`module_0508.jit_loss_fn.sm_12.1a_gpu_after_optimizations.txt` (109 МБ).

### 1.1. Что такое `command_buffer_1229` / `_985`

| буфер | GPU/шаг | графов | ядер | op_name | вердикт |
|---|---:|---:|---:|---|---|
| `command_buffer_1229` | 7.09 с | 256 | 40 192 | `transpose(jvp(vmap()))/while/body/eval_jaxpr/.../checkpoint/...` | **reverse-scan бэкварда KDA** (jnp) |
| `command_buffer_985` | 2.66 с | 256 | 17 664 | `jvp(vmap())/while/body/eval_jaxpr/...` | forward-mode jvp того же скана |

Доказательство формы: `%while.203` carry содержит
`f32[256,4,32,1536]` (256 чанков × 32 токена × hidden), `f32[256,4,12,128,128]`
(per-chunk score-матрицы), `f32[256,4,3,1536]` — ровно 256 чанков KDA (C=32).
Ядра `input_reduce_fusion_17` несут `op_name=.../checkpoint/chuv,chu->chv/dot_general`
— линейно-вниманиевая контракция KDA; `input_add_reduce_fusion` —
`checkpoint/reduce_sum`.

### 1.2. Топ-ядра хвоста и их конфигурация

| ядро | форма | конфиг | op_name (сокр.) |
|---|---|---|---|
| `gemm_fusion_dot_1221` | f32[768,128,128] | Triton nested, 4 warps, 4 стадии, тайл 1×64×128 | `.../checkpoint/dot_general` |
| `gemm_fusion_dot_1231` | f32[768,128,128] | Triton nested, 4 warps, **1 стадия**, тайл 1×64×32 | `jvp(vmap())/while/body/.../dot_general` |
| `gemm_fusion_dot_1239` | f32[720,128,128] | 4 warps, 1 стадия, 1×64×32 | то же |
| `gemm_fusion_dot_1232` | f32[384,128,256] | 2 warps, 4 стадии, 1×64×64 | `jvp(vmap())/.../dot_general` |
| `gemm_fusion_dot_1238` | f32[336,128,256] | 4 warps, 3 стадии, 1×64×128 | то же |
| `input_reduce_fusion_17` | f32[4,32,12,128] | native reduce | `checkpoint/chuv,chu->chv/dot_general` |
| `input_add_reduce_fusion` | f32[4,32,12,128] + f32[4,32,12,128,128] | native | `checkpoint/reduce_sum` |

Итог: хвост — **мелкие f32 Triton nested GEMM** (тайлы 128×32/64/128, 1–4 стадии),
собранные в 512 графов по 69–157 ядер. Это документированный follow-up
`net/kernels/kda_cc_fused_persist.py`: forward — один персистентный кернел,
backward — jnp reverse scan по чанкам; **variant B** (персистентный transposed
backward) не реализован.

## 2. Скрининг XLA-флагов GEMM-пути

Клетки по 3 шага (стационар n=2), тот же стек; каждый флаг — свой свежий кэш.

| клетка | флаги | ток/с | шаг, с | Δ |
|---|---|---:|---:|---:|
| ctrl | `--xla_gpu_autotune_level=1` | 840.8 | 38.979 | — |
| no_triton | `--xla_gpu_enable_triton_gemm=false` | 806.1 | 40.649 | **−4.1%** (минус №24) |
| gemm_v2 | `--xla_gpu_experimental_gemm_fusion_v2=true` | 838.8 | 39.068 | −0.2% |
| warp_spec | `--xla_gpu_experimental_enable_triton_warp_specialization=true --xla_gpu_triton_gemm_any=true` | 842.2 | 38.915 | +0.2% (шум) |
| ragged_gg / ragged_ws / v2_ws | `use_ragged_dot_grouped_gemm` и комбинации | вторая батарея | — | см. summary.txt |

Вывод: выбор GEMM-эмиттера шаг не двигает; цена — в структуре reverse-scan
(512 графов, 57.9k ядер), а не в конфигурации отдельного GEMM.

## 3. Решение (Am.30)

Реализовать **variant B — персистентный transposed-backward кернел KDA**
(один запуск на слой-проход, adjoint-состояние в SMEM, обратный проход по чанкам
внутри программы), гейт `AXIOM_KDA_CC_IMPL=persistent_bwd` (fail-closed, дефолт
прежний). Forward не трогается, jnp reverse scan остаётся оракулом и откатом.
Ожидание: снятие 7.09 с (1229) и части 985 ⇒ шаг 32–35 с, G1 достижим.
Приёмка: CPU-паритет ≤1e-3, 10-шаговый nan-гейт, стендовый A/B (10 шагов, B=4
persist-on), шаг ≤35.5 с.

## 4. Артефакты

- Дамп: `~/axiom-run/xladump-c18` (на стенде, 1.1 ГБ, не в git; воспроизводится
  `tools/c18-dump.sh`, ~25 мин).
- Метрики/журнал клетки — `evidence/mfu-55/cycle18-hlomap/`.
- Клетки флагов — `evidence/mfu-55/cycle18-flags*/` (metrics.jsonl, journal.json,
  run.log, summary.txt).
- Инструменты: `tools/c18-dump.sh`, `tools/c18_map_kernels.py`,
  `tools/c18-flags.sh`, `tools/c18-flags2.sh`, `tools/c18_cmdbuf_kernels.py`.

## 5. Ограничения

- Имена фьюженов в дампе и nsys совпадают (проверено на 1221/1231/1239/…), но
  привязка «буфер → фаза шага» опирается на NVTX-хронологию: 985/1229 идут в
  начале стационарного шага, до per-layer последовательности и больших
  backward-буферов; точная граница forward/backward по хосту не размечена.
- Вторая батарея флагов (ragged_gg и комбинации) на момент отчёта ещё шла;
  её результат — в `cycle18-flags2/summary.txt`, решение Am.30 он не меняет,
  если выигрыш <3%.
