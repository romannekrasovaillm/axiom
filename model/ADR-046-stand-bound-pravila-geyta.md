---
id: ADR-046
type: adr
title: "Stand-bound правила гейта (CUDA/.locks): явный SKIP с обязательной отметкой вместо ложного FAIL, прогон на GB10 как условие приёмки"
status: "accepted"
affects: [CMP-002, NFR-002]
source: "docs/adr/ADR-046-stand-bound-pravila-geyta-cuda-locks-yavnyy-skip-s-obyazatelnoy-otmetkoy-vmesto-lozhnogo-fail-progon-na-gb10-kak-uslovie-priyomki.md"
---

Правила гейта, зависящие от стенда (наличие CUDA, доступ к `~/gb10-shared/.locks`), на машинах без стенда дают явный SKIP с обязательной отметкой, а не ложный FAIL; истинная приёмка таких правил — прогон на GB10 (условие приёмки). Права `.locks` починены (755); режим доступа — операторский долг.
