---
id: ADR-051
type: adr
title: "Spine Bank (arch-be 0.3.16) как внешний контур архитектурного управления и контроля дрейфа кейса Axiom"
status: "proposed"
affects: [CMP-002, CMP-003, CMP-005]
source: "docs/adr/ADR-051-spine-bank-arch-be-0-3-16-kak-vneshniy-kontur-arhitekturnogo-upravleniya-i-kontrolya-dreyfa-keysa-axiom.md"
---

Продуктовый контур `arch-ml 0.5.3` дополняется банковским `arch-be 0.3.16` (Spine Bank) как
**внешним независимым барьером**: единый гейт (fitness + delta_guard + rule_weakened + spine_lint
+ trace + model_drift + arch_drift + secrets + nfr), дрейф «модель ↔ код», SSOT-аудит флота
worktree, измерение зубьев правил, регистр ложных срабатываний, метрика доверия 1–5.
Продуктовые прогоны не затрагиваются (`arch-be` — только контроль); версии пинуются решением.

**Первый прогон (09.10.2026) нашёл:** 5 рёбер `declared-edge-unused` (CMP-003/004/006), 1211 warn
`secret_literal` (ложные — git-sha в машинных фактах), `trust` 2/5 (40 правил без владельца),
`expiry` — 0 из 53, 51 файл дрейфа в 28 worktree (`CONSTRAINTS.yaml` — 17 версий).

**Закрыто:** волна W1+W3 — карточки жизненного цикла 53 правил (`owner` 53/53, `expiry` 53/53,
`effort_hours` = 57.75 чел.-ч), регистр FP, зубья 30/53; `trust` **2/5 → 3/5**; реестр долгов
инвариантов `[PENDING-EVIDENCE]` (P-1…P-6 с триггерами A4/A5/INT-001) вынесен в
`docs/OPEN-QUESTIONS.md`. Открыто: 5 рёбер, шум `secrets`, SSOT флота, ступени 4–5 доверия
(только на GB10 с evidence-бандлом). Аудит: `docs/reviews/SPINE-BANK-AUDIT-2026-10-09.ru.md`;
дельта `changes/w1-registry-hygiene/`.

Номер 049 занят решением о ремaтериализации (ADR-049, линия `arch/mfu-int`) — настоящий документ
переномерован при переносе и стоит **ADR-051**.

**Пересмотр (expiry):** безусловно 2027-01-31; триггеры — смена версии `arch-be`/`arch-ml`, появление единого гейта в ML-редакции (прогон `trust` + сверка вердиктов), закрытие волн W2/W4/W5.
**Оценка рубриками (09.10.2026):** `adr_quality` 4.60/5.
**Долг контура:** 36 принятых ADR без отчётов рубрик (`rubric handover` 36/36) → `decision_quality` в гейте включается после покрытия (R-1 в `docs/OPEN-QUESTIONS.md`).
