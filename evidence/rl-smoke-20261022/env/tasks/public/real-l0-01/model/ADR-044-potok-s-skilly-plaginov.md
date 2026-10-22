---
id: ADR-044
type: adr
title: "Поток S: скиллы и методики библиотеки плагинов в корпусе обучения токенизатора (27 МБ md, 70% кириллицы)"
status: "accepted"
affects: [CMP-001]
source: "docs/adr/ADR-044-potok-s-skilly-i-metodiki-biblioteki-plaginov-v-korpuse-obucheniya-tokenizatora-27-mb-md-70-kirillicy.md"
---

Библиотека плагинов (скиллы и методики, ~27 МБ markdown, ~70% кириллицы) включается в корпус обучения токенизатора отдельным потоком S — чтобы токенизатор видел доменный русскоязычный технический регистр контура.
