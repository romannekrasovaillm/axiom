# Дельта: w1-registry-hygiene
- Route: Standard
- Created: 2026-10-09
- Владелец: архитектор. Основание: ADR-049 (Spine Bank как контур управления и контроля дрейфа), аудит `docs/reviews/SPINE-BANK-AUDIT-2026-10-09.ru.md` §3, §6.

## Проблема

Реестр правил кейса (53 правила, C-001…C-053) не сопровождается: `arch-be trust` даёт 2/5, ступень «Правила сопровождаются» заблокирована — **у 40 из 53 правил нет владельца**, `expiry` не заполнен ни у одного, `effort_hours` пуст (`arch-be control rules-report`, 09.10.2026). Это антипаттерн 5 Spine Bank («правило без срока жизни»): правило без владельца и даты пересмотра живёт до первого дедлайна, а стоимость сопровождения реестра не видна. Плюс единый гейт тонет в шуме: 1211 warn `secret_literal` в машинном журнале фактов `evidence/facts/S-036.jsonl` (git-sha под шаблоном `hex-token`) — ложные срабатывания не помечены, регистр FP пуст (`arch-be digest`).

## ADDED

- Карточные поля `owner` / `expiry` / `effort_hours` у всех 53 правил `CONSTRAINTS.yaml` (Binds: реестр правил ↔ `rules-report` ↔ `trust`).
- Регистр ложных срабатываний `evidence/fp-register.md` с пометкой FP по `secret_literal` в машинных фактах датчика S-036.

## MODIFIED

- `CONSTRAINTS.yaml`: у 40 правил без владельца добавляется `owner: архитектор`; всем 53 — `expiry` (2027-01-31; правила активных кампаний ADR-032/040/047/048 — 2026-12-31) и `effort_hours` (первичная оценка: command_succeeds 2.0; content-правила 0.5; file_exists 0.25). Существующие 13 `owner` не перезаписываются.
- Копия реестра в пакете `.arch-handoff/CONSTRAINTS.yaml` синхронизируется с корневой (иначе единый гейт даёт `registry_diverged`, T-02).

## REMOVED

- Ничего. Правила и их формулировки не меняются — дельта добавляет метаданные жизненного цикла.

## План отката

`git revert` коммита дельты (правки только в `CONSTRAINTS.yaml` и `evidence/fp-register.md`); бэкап исходного реестра — `/tmp/axiom-CONSTRAINTS.yaml.bak-20261009`. Признак регрессии: `arch-be control check .` перестал быть PASS или `rules-report` теряет разбор карточек.

## Критерии приёмки

- [ ] `arch-be control rules-report .` — в колонках Owner/Expiry нет пустых ячеек; суммарный `effort_hours` печатается.
- [ ] `arch-be trust .` — ступень «Правила сопровождаются» больше не блокируется формулировкой «у 40 правил нет владельца».
- [ ] `arch-be gate --repo . --route standard` — `fitness` без новых error; вердикт не хуже прежнего (INCOMPLETE по ресурсу GB10).
- [ ] `arch-ml control check` / `fitness_check` — 53 правила, новых нарушений нет.
- [ ] `arch-be digest --repo .` — доля FP по `secrets` перестаёт считаться нулевой (регистр непуст).

## Не входит в дельту (следующие шаги)

- `arch-be rules teeth --save` (измерение зубьев; требует копии кейса ~14 ГБ — диск заполнен на 91%, отдельным шагом).
- Исполняемые правила для AD-1/AD-3/AD-5/AD-6/AD-13 из шаблонов Bank (`rules template apply`) — отдельная дельта (W1-бис).
