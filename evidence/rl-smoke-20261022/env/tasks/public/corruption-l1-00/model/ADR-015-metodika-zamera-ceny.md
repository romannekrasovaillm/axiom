---
id: ADR-015
type: adr
title: "Методика замера цены (критерий 16): пиннинг частот, чередование ног, медиана с разбросом, запрет вердикта по шуму"
status: "accepted"
affects: [CMP-002, NFR-005]
source: "docs/adr/ADR-015-metodika-zamera-ceny-kriteriy-16-pinning-chastot-cheredovanie-nog-mediana-s-razbrosom-zapret-verdikta-po-shumu.md"
---

Замер цены механик длинного контекста (критерий 16): пиннинг частот GPU, чередование ног A/B,
медиана с разбросом (запрет вердикта по шуму), отчёт с p50/p90. Отвергнуты: однократный замер,
вердикт по среднему без разброса.
