---
id: ADR-058
type: adr
title: "Манифест прогона environment v2: ось победы, baseline K3 и запрет фиктивного пиннинга"
status: "accepted"
affects: [CMP-005, CMP-006, NFR-002]
source: "docs/adr/ADR-058-manifest-progona-environment-v2-os-pobedy.md"
---

Run manifest среды не нёс ни оси победы, ни baseline, а поле пиннинга снапшота заполнялось
`sha256_text("stub-model")` при отсутствии `--model-snapshot` (`env/main.py:76`), поэтому
заглушка проходила как «запинненная модель», и заявление о превосходстве (AD-1) было
непроверяемо. Отдельно: `victory_axis` лежит в `DECISION-PACKAGE.json#/choice/victory_axis`,
а правило адресовало несуществующий top-level путь, страж C-013 проверял подстроку.

**Решение:** схема `environment-v1/manifest-v2` — единственная доказательная форма:
обязательные `victory_axis` (копия `choice.victory_axis`, сверка по нормализованной строке),
`baseline.{model, snapshot_sha256}`, `model.is_stub`; `is_stub: true` доказательством не
является; v1 валиден только с `allow_legacy=True` (исторический аудит). Стражи: C-013
(structural — путь и единственность оси), C-055 (ось+baseline в манифесте, заглушка
отклоняется), C-056 (RL-precondition: калибровка не-заглушки в коридоре [10%, 90%];
known-red до прогона K3 на GB10 — ADR-056).
