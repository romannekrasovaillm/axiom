---
id: ADR-045
type: adr
title: "Трактовка рангов MFU-лестницы: ceiling (L0–L2) не является тренировочным свидетельством (L3–L4); 50% MFU к тренировке не пинится"
status: "accepted"
affects: [CMP-002, NFR-003]
source: "docs/adr/ADR-045-traktovka-rangov-mfu-lestnicy-ceiling-l0-l2-ne-yavlyaetsya-trenirovochnym-svidetelstvom-l3-l4-50-mfu-k-trenirovke-ne-pinitsya.md"
---

Ранги лестницы MFU разделяются по доказательной силе: ceiling-ранги L0–L2 (чистый matmul, плотные блоки) показывают потолок XLA-пути, но НЕ являются тренировочным свидетельством; тренировочные выводы делаются только по L3–L4 (слои/шаг). Цель «50% MFU» к тренировке не пинится (не достижима как тренировочный показатель на этом классе железа); пиннутся ступени с измеримым механизмом.
