# Дельта: model-adr-049-051
- Route: Standard
- Created: 2026-10-09
- Владелец: архитектор. Основание: обязательство вести архитектурную модель (trace_check: ADR → CMP 100%); ADR-049 (remat, линия `arch/mfu-int`), ADR-050 (кампания MFU-55), ADR-051 (Spine Bank).

## Проблема

Модель кейса отстала от решений: в `model/` отсутствовали сущности **ADR-049** (политика ремaтериализации — принято 09.10 в линии `arch/mfu-int`, в модель не внесено), **ADR-050** (кампания MFU-55) и **ADR-051** (Spine Bank). При этом решения уже действуют в коде и гейтах (remat-политика измерена: 468.6 против 407.2 ток/с; кампания MFU идёт на стенде; контур Bank дал `trust` 3/5). Дрейф «решение ↔ модель» — тот класс, который ловит `trace check`; модель обязана следовать за решениями, а не отставать.

## ADDED

- `model/ADR-049-politika-rematerializacii-…md` — сущность ADR-049 (remat-политика): раскладка шага (backward 73%, forward 21%, оптимизатор 6%), матрица политик на GB10 (`none` 407.2, `dots_with_no_batch` 468.6, `dots_saveable` OOM), `affects: [CMP-002, NFR-003]`.
- `model/ADR-050-kampaniya-mfu-55-…md` — предмет и конвенция числителя (`6·N_active` = 3.014 GFLOP/токен ⇒ 55% = 17 917 ток/с), гейты G0–G4 (977/3 258/6 515/17 917), результат G0 (493 173 ядра/120 с, занятость 5.1%, LU/TRSM чанков KDA), baseline 532.8 ток/с (1.64%), `affects: [CMP-002, CMP-004, NFR-003]`.
- `model/ADR-051-spine-bank-…md` — Spine Bank как внешний контур управления и контроля дрейфа, результаты W1+W3 (`trust` 2/5 → 3/5), `affects: [CMP-002, CMP-003, CMP-005]`.
- `docs/adr/ADR-049-politika-rematerializacii-…md` — перенос текста решения из линии `arch/mfu-int` (без него модель ссылалась бы в пустоту; артефакт уже действует в коде).

## MODIFIED

- Ничего в существующих сущностях не менялось: ADR-047/048 уже описывают KDA и оптимизатор, ADR-050 — их продолжение.

## REMOVED

- Ничего.

## План отката

`git revert` коммита дельты (правки только в `model/` и один перенесённый `docs/adr/`), либо снятие трёх файлов модели. Признак регрессии: `arch-ml model validate model` или `trace check` перестали быть PASS.

## Критерии приёмки

- [ ] `arch-ml model validate model` — PASS (ссылочная целостность, 0 error).
- [ ] `arch-ml trace check .` — звено ADR → CMP 100%, сирот нет (было 47/47, станет 50/50).
- [ ] `arch-ml control spine ARCHITECTURE-SPINE.md` — без новых находок.
- [ ] `arch-be gate --route standard` — `model_validate`, `trace_check`, `model_drift` без новых error.
