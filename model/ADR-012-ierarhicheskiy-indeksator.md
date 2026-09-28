---
id: ADR-012
type: adr
title: "Иерархический разреженный индексатор: общий пул кандидатов (import 2-3-2) в скелет L3"
status: "accepted"
affects: [CMP-002, NFR-005]
source: "docs/adr/ADR-012-ierarhicheskiy-razrezhennyy-indeksator-obschiy-pul-kandidatov-import-2-3-2-istochnika-v-skelet-l3.md"
---

Иерархический разреженный индексатор с общим пулом кандидатов (композиция трёх источников
2-3-2) входит в скелет L3 декларативно через net/config.json. Отвергнуты: per-layer пулы,
отказ от индексатора, отложенная интеграция. Стражи: C-034 (решение до реализации),
C-035 (декларативность механики).
