---
id: ADR-018
type: adr
title: "Block-wise token merging в MLA-слоях скелета (заимствование Step-5), дефолт off до замера 64K"
status: "accepted"
affects: [CMP-002, NFR-005]
source: "docs/adr/ADR-018-block-wise-token-merging-v-mla-sloyah-skeleta-zaimstvovanie-step-5.md"
---

Block-wise token merging — декларативная механика MLA-слоев (net/config.json mla_block_merge +
deviations; источник Step-5-Preview model card). Реализация за флагом, плотный путь — оракул.
Гейт: замер на гэпе 64K по критериям 1–7 ADR-009 (recall не хуже baseline, цена ниже измеримо);
до зелёного замера флаг off. Стражи: C-034, C-035.
