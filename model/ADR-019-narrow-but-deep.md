---
id: ADR-019
type: adr
title: "Narrow-but-deep — гипотеза пропорций для L1-конфига (заимствование Step-5)"
status: "accepted"
affects: [CMP-002]
source: "docs/adr/ADR-019-narrow-but-deep-gipoteza-proporcij-l1-zaimstvovanie-step-5.md"
---

В пакет выбора L1 внесены две альтернативы пропорций при равном param_budget: A «широко-неглубокая»
(масштабирование базы K3) и B «narrow but deep» (глубже-уже, ориентир Step-5). Сравнение — по оси
победы на лестнице среды, механический вердикт. Риск B — router collapse на глубоких MoE-стеках.
