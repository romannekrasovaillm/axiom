---
id: ADR-040
type: adr
title: "Политика compute-dtype (bf16-гейт): инфраструктура принята, включение отложено до parity-вердикта и stage-2"
status: "accepted"
affects: [CMP-002, NFR-002]
source: "docs/adr/ADR-040-politika-compute-dtype-bf16-geyt-infrastruktura-prinyata-vklyuchenie-otlozheno-do-parity-verdikta-i-stage-2.md"
---

Гейт `AXIOM_COMPUTE_DTYPE` принят как инфраструктура: дефолт fp32 байт-точен по построению (оракул ADR-009 незыблем), bf16-режим — bf16-операнды GEMM с накоплением в fp32 (рецепт Beam). Включение в претрейн отложено за двумя гейтами: (а) parity-нога зелёная (ΔBPB ≤ +0.05 на длинном горизонте), (б) stage-2 профиль показывает выигрыш в целевых фазах. Матрица кампании (9 клеток): bf16 даёт ×0.98–1.16 на шаге — dtype измеренно НЕ является рычагом MFU (граф memory/dispatch-bound). Отклонение parity-ноги (поток W против микса 85/15 ADR-021) принято осознанно: парная идентичность данных — контролирующее требование dtype-A/B.
