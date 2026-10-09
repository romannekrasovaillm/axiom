---
id: ADR-043
type: adr
title: "Трейсы Spine как поток T в корпусе обучения токенизатора: строгий отбор сессионных jsonl, обязательный скраб секретов, ограничение доли"
status: "accepted"
affects: [CMP-001]
source: "docs/adr/ADR-043-treysy-spine-kak-potok-t-v-korpuse-obucheniya-tokenizatora-strogiy-otbor-sessionnyh-jsonl-obyazatelnyy-skrab-sekretov-ogranichenie-doli.md"
---

Сессионные трейсы Spine включаются в корпус обучения токенизатора отдельным потоком T: строгий отбор jsonl (только валидные сессии), обязательный скраб секретов перед включением, явное ограничение доли потока в миксе.
