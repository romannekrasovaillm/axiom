---
id: ADR-050
type: adr
title: "Кампания MFU-55 на GB10: предмет и конвенция числителя, гейты G0-G4, рычаги по профиль-доказанным адресам"
status: "proposed"
affects: [CMP-002, CMP-004, NFR-003]
source: "docs/adr/ADR-050-kampaniya-mfu-55-na-gb10-predmet-i-konvenciya-chislitelya-geyty-g0-g4-rychagi-po-profil-dokazannym-adresam.md"
---

Директива владельца (09.10.2026): поднять MFU на GB10 до 55%. **Предмет:** 55% от измеренного
bf16-пика 98.2 TFLOPS = 54.0 TFLOPS; базовая конвенция числителя — `6·N_active·tokens`
(3.014 GFLOP/токен, `mfu_params_only=true`), для MoE — по активным параметрам; снят дрейф
конвенции в 2.4× (ранее в пине 7.2 GFLOP/токен ⇒ «7.5k ток/с» вместо **17 917**). Цель — цель,
не пин: `C-046` и пин аренды 800 ток/с (2.46%) не меняются; `verdict_50pct` пересматривается
амендментом только по факту G3.

**Гейты:** G0 диагностика (закрыт) → G1 launch-bound (ядер/шаг ≤2 000, синхронизаций ≤5 000,
занятость ≥40%, **MFU ≥3% = 977 ток/с**) → G2 тензорные ядра (bf16/fp32 ≥2×, **≥10% = 3 258**) →
G3 батч токенов + реальный MoE-dispatch (**≥20% = 6 515**) → G4 **≥55% = 17 917**.

**G0 — результат:** на актуальной линии (`arch/mfu-int`, `kda_impl=chunked_cc`) шаг — шторм мелких
ядер: 493 173 запуска за 120 с стационара при занятости GPU 5.1%, доминируют LU/TRSM внутри чанков
KDA (`getrf_panel`, `batch_trsm_left_kernel`, `LuPivotsToPermutation` — по 34 560); синхронизации
`cuStreamSynchronize` 113 895 (236.8 с) и `cuEventSynchronize` 22 399 (237.1 с) за окно.
Baseline кампании: **532.8 ток/с, шаг 15.38 с, MFU 1.64%** (CE-чанк 4096); рядом remat
`dots_with_no_batch` 468.3, `none` 406.7. Артефакты: `evidence/mfu-55/G0b/REPORT.md`,
`G0/REPORT.md` (невалидный прогон на снятой форме `chunked` — свидетельство дрейфа линий).
