# Цикл-36 — fused dispatch/scatter: МИНУС (−50 %)
A/B B=4, AXIOM_GEMM_FP8=1 + AXIOM_MOE_BWD=fused + AXIOM_MOE_DISPATCH=fused, 10 шагов:
стационар **691.6 ток/с, шаг 47.381 с** (база 1389.6 / 23.581) — шаг вырос вдвое.
Лосс в допуске, nan 0. GPU-паритет всех трёх fused-кернелов достигнут
(dispatch rel 0, combine 0/1.3e-7, scatter 0 — Am.59/60), но исполнение дороже XLA:
per-expert кернелы (12 программ × 3 кернела × 24 слоя) добавляют запуски/сериализацию.
Вывод: XLA-часть `sec_bwd_moe` (~3.6 с) — **не** dispatch/scatter (память там мс);
это per-layer remat-recompute форварда и routing/elementwise. Гейт остаётся jax.
