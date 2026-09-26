---
id: ADR-006
type: adr
title: "Домен оси победы — архитектурные гейты собственного контура"
status: "accepted"
affects: [CMP-005, CMP-006, CAP-001]
source: "docs/adr/ADR-006-domen-osi-pobedy-arhitekturnye-geyty-sobstvennogo-kontura.md"
---

Задачи — архитектурные правки кейсов собственного контура; вердикт — детерминированные гейты arch-ml (fitness/trace/spine + тесты) без деградации. Источники: реальные кейсы, синтетическая порча чистых кейсов, holdout (ротация, обучение по нему запрещено). Precondition запуска RL — кальбровочный прогон K3 (прохождение не >90% и не <10%). Версия набора гейтов пиннится в run manifest.
