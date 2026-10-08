# nsys-профиль l3-full — сводки (stage-2, MFU-кампания)

**Провенанс.** Прогон 08.10.2026, стенд GB10 (spark-44c3), архитектор:
`nsys profile --stats=true -o evidence/mfu-profile/nsys-pretrain-l3full python tools/pretrain_run.py
--model-preset l3-full --grad-checkpointing --steps 6 --total-steps 6 --seq-len 8192 --batch-size 1
--shard-root /home/roman/axiom-run/raw-v2 --tokens-root /home/roman/gb10-shared/datasets/axiom-pretrain-l3/tokens-v2 --streams W`

Носители (остаются на стенде, в git не влезают): `nsys-pretrain-l3full.nsys-rep` (605 МБ),
`nsys-pretrain-l3full.sqlite` (1.5 ГБ), `nsys-pretrain-l3full.log` (170 МБ). Здесь — извлечения.

## Сводные числа (SQL по sqlite)

- Окно трейса: **1201 с**; запусков ядер: **590 689**; суммарное ядровое время: **527.8 с**
  (включая ~720 с фазы компиляции/автотюна в начале — там занятость 40–100%).
- **Стационарная фаза (последние ~4 шага): ядровая занятость ~6%** — ~5 с ядер на шаг 82 с;
  ~54–72 тыс. запусков ядер на шаг.
- Хост-API (см. `host-api-top.txt`): **93 437 `cuStreamSynchronize` = 239 с блокировки хоста**;
  740 716 `cuLaunchHostFunc`; 1.52 млн `cuEventRecord`; 1.07 млн `cuStreamWaitEvent`;
  165 тыс. HtoD + 58 тыс. DtoH копий; **3.6 млн `cuCtxSetCurrent`** — хост-медиация исполнения.

## Семейства ядер в стационарной фазе (см. `kern-top.txt`)

| Ядро | Запусков | Время |
|---|---|---|
| `loop_multiply_fusion` | 66 936 | 2 089 мс |
| `wrapped_add` | 39 708 | 1 691 мс |
| `gemm_fusion_dot_general_1` | 29 100 | 1 686 мс |
| `wrapped_transpose` | 20 412 | 1 045 мс |
| **`cutlass_80_simt_sgemm` (fp32 SIMT GEMM)** | 6 270 | **5 169 мс — треть ядрового времени** |

`loop_*`-префикс = фузии внутри while-цикла (KDA chunked scan: 128 чанков × 18 слоёв).
fp32-SIMT sgemm — кандидат: маты моноида KDA (128×128, fp32) — **[ТРЕБУЕТ ПРОВЕРКИ]**: проверить
под гейтом bf16, исчезает ли SIMT-класс (клетка L3-KDA лестницы).

## Воспроизводимость

```sql
-- топ ядер
SELECT s.value, COUNT(*), ROUND(SUM(k.end-k.start)/1e9,1) s
FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName = s.id GROUP BY 1 ORDER BY 3 DESC LIMIT 25;
-- хост-API
SELECT s.value, COUNT(*), ROUND(SUM(r.end-r.start)/1e6,0) ms
FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId = s.id GROUP BY 1 ORDER BY 2 DESC LIMIT 16;
-- занятость по 120-с окнам
SELECT CAST((start-(SELECT MIN(start) FROM CUPTI_ACTIVITY_KIND_KERNEL))/1.2e11 AS INT) bin,
       ROUND(SUM(end-start)/1e9,1) kernel_s FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY 1 ORDER BY 1;
```

**Вывод (для кампании).** Шаг l3-full: ~82 с wall при ~5 с ядер — потери не в арифметике,
а в хост-медиации исполнения (sync/event/context-трафик) и в микро-GEMM-лавине KDA-chunked;
отдельный сток — fp32 SIMT sgemm. Все быстрые флаги (command buffers, autotune level, TF32,
chunk size, dtype, flash, ckpt, allocator) проверены — ноль эффекта (`/home/roman/axiom-run/mfu-exp.log`).
