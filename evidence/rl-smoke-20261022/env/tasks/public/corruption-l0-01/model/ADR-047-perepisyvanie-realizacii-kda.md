---
id: ADR-047
type: adr
title: "Переписывание реализации KDA-слоя: математика delta-rule сохраняется, реализация заменяется (внутри чанка C×C, между чанками состояние dk×dv) как единственный рычаг MFU модели"
status: "accepted"
affects: [CMP-002, NFR-003]
source: "docs/adr/ADR-047-perepisyvanie-realizacii-kda-sloya-matematika-delta-rule-sohranyaetsya-realizaciya-zamenyaetsya-vnutri-chanka-c-c-matrica-mezhdu-chankami-sostoyanie-dk-dv-kak-edinstvennyy-rychag-mfu-modeli.md"
---

KDA-слой переписывается (форма `chunked_cc`): математика delta-rule сохраняется, но внутри чанка считается C×C-матрица вместо материализации dk×dk на каждый токен, между чанками переносится состояние dk×dv. Мотив — единственный рычаг MFU модели: старая форма давала 2.1% MFU на слое при ~56% времени шага. Wyut-форма отвергнута фактом (97.9 с vs 85.5, OOM 46.81 ГиБ). Критерии: паритет с оракулом, MFU L3a ≥ 20%, память не выше chunked. Измеренный эффект: 260.2 ток/с против 99.0 (×2.63), шаг 31.5–33.0 с против 82.7–85.5, loss сопоставим (13.42 vs 13.41), паритет подтверждён 85 тестами.
