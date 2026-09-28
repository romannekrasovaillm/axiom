---
id: ADR-016
type: adr
title: "Скоуп гейтов: правила кейса не применяются к снапшоту задачи (workspace)"
status: "accepted"
affects: [CMP-006]
source: "docs/adr/ADR-016-skoup-geytov-pravila-keysa-ne-primenyayutsya-k-snapshotu-zadachi-workspace.md"
---

Правила CONSTRAINTS кейса не применяются к workspace-снапшоту задачи агента (иначе стражи меряют
снапшот, а не кейс). Граница скоупа фиксируется в каждой rule-формулировке.
