---
id: ADR-009
type: adr
title: "Импорт KV/длинноконтекстной дисциплины DeepSeek V4.1-Flash в скелет L3 (SWA-ветка, sparse-выбор в MLA, FP4 латентного KV)"
status: "accepted"
affects: [CMP-002, CMP-004]
source: "docs/adr/ADR-009-import-kv-dlinnokontekstnoy-discipliny-deepseek-v4-1-flash-v-skelet-l3-swa-vetka-v-kazhdom-sloe-sparse-vybor-v-mla-sloyah-fp4-na-latentnom-keshe.md"
---

Д1: SWA-ветка `n_win=128` в каждом слое (на MLA — своими проекциями, на KDA — шаринг уже вычисленных q/k/v). Д2: CSA2-lite — лёгкий indexer → top_k в 6 MLA-слоях, без компрессии последовательности и без кросс-слойных режимов. Д3: MXFP4 QAT латентного KV-кэша (E2M1 + E4M3 scale на 16 каналов, без global scale), SWA KV остаётся FP8. Д4: плотный путь — оракул за флагом, A/B при равном бюджете. Д5: train-aware симуляция sparse/окна во всех стадиях, включая пост-тренинг. Основание — измеренный провал критерия №6 (`net/tests/test_06_nope_extrapolation.py`) и квадратичный compute-WALL на MLA-слоях при 1M (NFR-005). Отложено с условиями возврата: Engram, DSpark вместо MTP, Sinkhorn-balanced update. Отвергнуто: CED, кросс-слойное переиспользование KV, mHC, увеличение доли MLA.
