---
id: NFR-004
type: nfr
title: "Инференс целевой модели на DGX Spark"
status: "proposed"
verification: "замер на Spark: размещение 4-бит, TTFT/ITL, деградация против bf16 в среде"
measure: "модель L1 (30B/3B активных) исполняется в 128 ГБ unified в 4-бит с запасом под контекст; дельта качества от квантизации ≤ 2 п.п."
affects: [CMP-004]
---
