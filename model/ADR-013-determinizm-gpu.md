---
id: ADR-013
type: adr
title: "Детерминизм вычислений на GPU для A5: xla_gpu_deterministic_ops в гейтовом профиле"
status: "accepted"
affects: [CMP-002, NFR-002]
source: "docs/adr/ADR-013-determinizm-vychisleniy-na-gpu-dlya-a5-xla-gpu-deterministic-ops-v-geytovom-profile.md"
---

Гейтовый профиль A4/A5 включает --xla_gpu_deterministic_ops (выставляется до создания XLA-клиента);
вердиника повторных прогонов побитово сверяются (workspace_sha256). Страж: C-042 (runtime-пиннинг).
Уточнение 27.09: вторая причина недетерминизма — байткод в workspace-снапшоте (вычищается executor'ом);
ADR-013 остаётся в силе, формулировка расширена.
