---
id: ADR-038
type: adr
title: "Приоритизация и защита поведенческого слоя"
status: "proposed"
affects: [CMP-001, CMP-002, CMP-005, CMP-006]
source: "docs/adr/ADR-038-prioritizaciya-i-zashchita-povedencheskogo-sloya.md"
---

Слой ADR-037 становится устойчивым: факт прослеживается до сырья (`raw` в
реестре датчиков, `raw_ref` в записи факта, `probe --raw` ловит подмену истории),
протокол экспортёра (`describe`/`collect`) отделён от домена и вынесен в пакеты
(`tools/sensors/packs/**`), приоритеты считаются по ставкам (`stake`, `--queue`),
а гейты защищены от трёх ловушек: fail-open стражей (аудит S-034, правило K5),
прокси-метрик (`level`, `e2e_pair`, строка «прокси-расхождение») и шума железа
(окна/допуски с происхождением, S-035 `noise_baseline`, S-037 `flapping`).
Открывающие гейты собраны в `model/opening-gates.yaml` и проходятся единым
`tools/preflight.py` (`pass` только если все требования `pass`); `unverified` и
`flapping` дверь не открывают. Наследование прозаических правил — реестр
`model/rule-lineage.yaml` (30 documentary-правил: `successor | none | pending`).
Новое правило C-051 (`structural`, песочница): `check_claims --verify-layer`.
