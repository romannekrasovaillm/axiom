---
id: ADR-039
type: adr
title: "Каталог свойств, генераторы кандидатов и проверка проверок"
status: "proposed"
affects: [CMP-001, CMP-002, CMP-005, CMP-006]
source: "docs/adr/ADR-039-katalog-svoystv-generatory-proverka-proverok.md"
---

Поведенческие проверки универсализуются на четырёх слоях вокруг доменной
привязки. Каталог `tools/properties/**` реализует двенадцать видов свойств
(`declared_equals_actual`, `bounds`, `conservation`, `identity`, `determinism`,
`reversibility`, `differential`, `metamorphic`, `monotonic_trend`, `liveness`,
`safety`, `freshness`), описанных в `model/properties.yaml`; утверждения
`model/claims.yaml` ссылаются на шаблон (`property`+`params`), существующие
стражи сопоставлены шаблонам в `model/rule-properties.yaml` (`full | wrapped |
not_applicable`) и сверяются теневым прогоном (S-038 `guard_shadow_agreement`).
Пять генераторов (`generate_declared|observed|telemetry|incidents`, `proposer`)
предлагают привязки в `model/candidates.yaml`; принимает кандидата только
архитектор, `proposer` вне пути вердикта. Мутационный исполнитель
`tools/properties/mutate.py` и датчики S-039/S-040 считают мутационный счёт
(гейтовые утверждения обязаны убивать все неэквивалентные мутанты), матрица
возмущений `tools/properties/chaos.py` и S-041 `detection_matrix` проверяют, что
стражи вообще ловят порчу. Новое правило C-052 (`structural`, песочница):
`check_claims --verify-properties`.
