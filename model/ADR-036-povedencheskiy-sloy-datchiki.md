---
id: ADR-036
type: adr
title: "Поведенческий слой Spine: датчики, факты, утверждения, классы вердикта"
status: "proposed"
affects: [CMP-001, CMP-002, CMP-006]
source: "docs/adr/ADR-036-povedencheskiy-sloy-datchiki-fakty.md"
---

Механизм закрытия разрыва 85/15 (documentary vs поведенческие коммиты):
**утверждение → датчик → факт → предикат → вердикт**. Датчик пишет запись факта
в `evidence/facts/<S-id>.jsonl` (пин предмета + цепочка `prev_sha256`), вердикт
выносит предикат утверждения `model/claims.yaml` через `tools/check_claims.py`.
Классы: `pass | fail | unverified`; `unverified` не открывает ни деньги, ни
стадии (preflight аренды, `--require-verified`). Смета — утверждение, а не факт;
`config.json` — декларация, факт конфигурации даёт S-001. Новые правила C-047
(structural, песочница) и C-048 (behavioural, репо). Реестр инцидентов с полем
«какой датчик увидел бы предвестник». Атом `drift_config_value` (L1–L3) и H-слой
для Task Spec v2. Датчики аддитивны; Task Spec v1 воспроизводится побайтово.
