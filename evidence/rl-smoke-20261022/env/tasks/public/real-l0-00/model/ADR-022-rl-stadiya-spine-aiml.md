---
id: ADR-022
type: adr
title: "RL-стадия исполняется на Spine AI/ML как среда и верификатор; рецепт — онлайн-синтез сред по FrogNano + практики Poolside Laguna"
status: "accepted"
affects: [CMP-003, CMP-006]
source: "docs/adr/ADR-022-rl-stadiya-na-spine-aiml-onlayn-sintez-sred-po-frognano-i-praktiki-laguna.md"
---

RL-стадия ведётся на собственном контуре Spine AI/ML (arch-ml CLI) как среда и верификатор: ось победы — задачи с механическим fitness-гейтом (AD-1/AD-2/ADR-006). Рецепт — онлайн-синтез сред раундами под текущий чекпойнт (FrogNano: ~1500 синтетических SWE-сред, двусторонняя исполняемая валидация fail-to-pass, маскирование промпта/выводов инструментов в loss, штраф за длину, вердикт — тесты над финальным состоянием) плюс практики Poolside Laguna. Среда v1 спроектирована поверх верификаторов arch-ml (docs/specs/ENVIRONMENT-V1.md); GB10 нагружается по всем стадиям претрейна и пострейна. Директива владельца 04.10.2026.
