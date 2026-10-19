# Цикл-30 — GPU-исполнение fused MoE-backward: `dz_grouped` НЕ проходит паритет

- Стенд GB10 (sm_121), jax 0.11.2, `check_moe_bwd.py` (архитектор; вариант
  check_kernel.py для tuple-выходов): ядро исполнено (jax.jit), сверка с
  `reference_bwd`, детерминизм (два запуска).
- case0 (rows=2048, latent=768, inter=384, n_exp=12): rel по выходам
  `(dwg, dwu, dwd, dz_grouped)` = **9.50e-04, 7.89e-04, 5.34e-07, 1.28e+00**;
  det=True.
- Первые три выхода — в допуске; **`dz_grouped` — 128 % ошибки**: ошибка
  трансрипции `dz = dhg·wgᵀ + dhu·wuᵀ` в кернеле (ориентация mma/сложение/запись
  выхода), т.к. CPU-паритет test_54 проверяет шов-эмуляцию, а не само ядро.
- `valid_config`/lowering: OK 2/0.
