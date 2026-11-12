---
id: ADR-058
title: "Манифест прогона environment v2: ось победы, baseline K3 и запрет фиктивного пиннинга"
status: Accepted
date: "2026-11-11"
depends_on: [ADR-002, ADR-004, ADR-006, ADR-014, ADR-037, ADR-056]
affects: [env/schemas.py, env/manifest.py, env/main.py, tools/check_victory_axis.py, tools/check_rl_precondition.py, CONSTRAINTS.yaml, docs/specs/ENVIRONMENT-V1.md]
spec_files: [ARCHITECTURE-SPINE.md, model/AD-1-os-pobedy-odna.md]
---

# ADR-058. Манифест прогона environment v2: ось победы, baseline K3 и запрет фиктивного пиннинга

- Date: 2026-11-11
- Status: Accepted (2026-11-11; дельты приняты приёмочной командой. Амендмент B
  от 2026-11-11: precondition декомпозирован — C-056 «среда откалибрована»
  достижимо на GB10, C-057 «K3-baseline» — долг P-7b с путём C-048 → смета AD-8 →
  аренда INT-001/API; на GB10 K3 невозможен по памяти)
- Маршрут: Standard (3 триггера: api_contract_change, data_contract_change, significant_nfr)
- Модель-автор: spine-arch

## Context

AD-1 требует: заявление о превосходстве допускается только по прогону стенда с
**пиннутыми снапшотами обеих моделей** и равным бюджетом, ось победы объявлена
одна и до запуска обучения. Механическая проверка этого инварианта отсутствует
(долг P-1 в `docs/OPEN-QUESTIONS.md`), и за время долга накопились предметные
дефекты контракта прогона:

1. **Фиктивный пиннинг.** `env/main.py:76` пишет в манифест
   `sha256_text("stub-model")` при отсутствии `--model-snapshot`, а
   `--model-base` имеет `default="stub"` (`env/main.py:192`). Валидатор
   (`env/schemas.py:164–169`) и `PINNING_PATHS` (`env/schemas.py:213–221`)
   принимают это как **полный** пиннинг: хеш несуществующего снапшота
   неотличим от хеша весов.
2. **Нет оси и baseline в контракте.** Схема §8
   (`docs/specs/ENVIRONMENT-V1.md:119–140`) описывает одну модель
   (`model.{snapshot_sha256, base}`) и вердикт; полей `victory_axis` и
   `baseline.*` нет. `victory_axis` не читается ни одним модулем `env/`,
   `tools/`, `net/` (grep: 0 вхождений в `.py`).
3. **Путь поля в правиле неверен.** Rule AD-1 (`ARCHITECTURE-SPINE.md:17`,
   `model/AD-1-os-pobedy-odna.md:12`) адресует
   `DECISION-PACKAGE.json.victory_axis`; фактически ключ лежит внутри
   `choice` (`docs/DECISION-PACKAGE.json:16, :23`), верхнеуровневого поля нет
   (`json.load(...).get("victory_axis") is None`).
4. **Страж проверяет подстроку.** C-013 — `must_contain` с
   `pattern: "\"victory_axis\""` (`CONSTRAINTS.yaml:175–186`), поэтому
   неверный путь, вложенность и второй пакет решения без оси
   (`docs/DECISION-PACKAGE-V41-FLASH.json`, gate A3, `victory_axis` — 0
   вхождений) остаются зелёными.

K3-калибровка (precondition RL, `ENVIRONMENT-V1.md:181`: pass-rate ∈ [10%, 90%])
ни разу не исполнялась: в `evidence/` лежат только отчёты заглушки
(`calibration-stub.json`, pass_rate 0.0, `ready: false` — честный fail-closed)
и обкатка Qwen3-4B (ADR-033); отчёта с `model = k3` нет.

## Decision

Вводится схема **`environment-v1/manifest-v2`** как единственная доказательная
форма run manifest. Решение одно и состоит из пяти частей:

1. **Обязательные поля пиннинга v2** (`env/schemas.py::PINNING_PATHS`):
   `schema_version`, `gates_version.{arch_ml_build, constraints_sha256,
   hidden_constraints_sha256, excluded_infra_rules}`, `workspace_sha256`,
   `model.{snapshot_sha256, base}`, **`baseline.{model, snapshot_sha256}`**,
   **`victory_axis`**. `missing_pinning()` по умолчанию требует v2; для
   исторического аудита — явный `allow_legacy=True` (v1-набор), и только он.
2. **Заглушка — не доказательство.** Поле `model.is_stub: bool` обязательно;
   `is_stub: true` даёт ошибку пиннинга. `env/main.py` требует явные
   `--model-snapshot` и `--model-base`; отладочная заглушка возможна только
   с `--allow-stub`, и `dump_manifest` для неё падает (fail-closed, как
   `ready: false` у calibration-stub).
3. **Ось сверяется с пакетом решения.** `victory_axis` манифеста сравнивается
   (детерминированно, по нормализованной строке) с
   `docs/DECISION-PACKAGE.json#/choice/victory_axis`; расхождение — ошибка
   `victory_axis: не совпадает с пакетом решения`.
4. **Стражи предметные.** C-013 переводится в `kind: structural` и вызывает
   `tools/check_victory_axis.py --selftest`: проверка **пути**
   `choice.victory_axis`, непустоты, отсутствия конкурирующего top-level
   `victory_axis`, совпадения ссылок в спайне и `model/AD-1`, плюс мутанты
   (удаление поля, перенос на top-level, подмена значения — красные).
   Новое C-055 вызывает `tools/check_manifest_v2.py --selftest` (схема v2:
   заглушка и манифест без оси/baseline отклоняются).
5. **RL-precondition проверяется механически — двумя разными правилами
   (амендмент B, 2026-11-11).** Это два разных обязательства, и раньше они были
   слиты в одно невыполнимое:

   - **C-056 «среда откалибрована»** (`tools/check_rl_precondition.py --part env`):
     в `evidence/` есть отчёт калибровки **не-заглушки** (своя модель/чекпойнт) с
     `pass_rate ∈ [0.10, 0.90]`, полным пиннингом и ссылкой из
     `evidence/rl-precondition.json`. Достижимо на GB10 (прогон `jaxlm`-адаптера).
     Блокирует **RL-расход**.
   - **C-057 «K3-baseline»** (`--part k3`): отчёт `model = k3` с тем же коридором,
     пиннингом и ссылкой из `evidence/k3-baseline.json`. На GB10 **невозможен**:
     2.8T параметров ≈ 1,4 ТБ в FP4 против 128 ГБ unified. Путь: C-048 (KPI
     800 ток/с) → смета AD-8 → аренда INT-001 (мультиузловой кластер) либо
     API для публичного/обезличенного набора. Блокирует **заявление о
     превосходстве** (AD-1), но не обучение.

Носители решения: `env/schemas.py`, `env/manifest.py`, `env/main.py`,
`tools/check_victory_axis.py`, `tools/check_manifest_v2.py`,
`tools/check_rl_precondition.py`, `docs/specs/ENVIRONMENT-V1.md` §8,
`CONSTRAINTS.yaml` (C-013, C-055, C-056), спайн AD-1 и `model/AD-1`.

## Alternatives Considered

| Вариант | Плюсы | Минусы |
|---------|-------|--------|
| **Статус-кво: ось только в прозе пакета, страж-подстрока C-013** | ничего не меняется, исторические манифесты валидны | фиктивный снапшот проходит как доказательство; неверный путь поля невидим; заявление о превосходстве непроверяемо (дефекты 1–4 выше) |
| **Только усилить страж C-013 (structural), не меняя манифест** | минимальная цена, ловит дефект пути поля | не ловит ни фиктивный пиннинг, ни отсутствие baseline: ось проверяется сама по себе, а не в связке с прогоном — Prevent AD-1 («сравнение на равном бюджете») остаётся вне контроля |
| **Обязать baseline лишь для сравнительных прогонов (`comparison: true`)** | не ломает обычные прогоны среды | «сравнительный» — флаг в том же манифесте, который пишет прогон: подделывается одним булевым; вердикт pass-rate без baseline не интерпретируем (неясно, на какой лесенке он получен) |
| **Принять решение: схема manifest-v2 с обязательными `victory_axis` + `baseline.*` + `is_stub` (выбран)** | доказательство становится проверяемым машинно; P-1 закрывается поведенчески, а не прозой | несовместимость с v1: существующие манифесты теряют статус доказательства без перезаписи |

## Consequences

### Positive

- Заявление о превосходстве становится проверяемым: манифест несёт ось, baseline
  и запрет заглушки; `missing_pinning` краснеет на манифесте без `victory_axis`
  и `baseline.snapshot_sha256`.
- Закрывается класс дефекта «зелёный гейт на надписи» (AD-10): C-013 перестаёт
  быть подстрочным.
- K3-калибровка перестаёт быть обещанием: её отсутствие — красное правило с
  владельцем и триггером, а не пробел в прозе.

### Negative

- **Несовместимость схемы:** манифесты v1 (включая
  `evidence/a4-skeleton-run-manifest.json`) без `allow_legacy=True` дают ошибку.
  A4-гейт остаётся закрытым на своём основании (покрытие стадий), но из
  v1-манифеста нельзя делать заявления о превосходстве; для A5-дрейфа нужна
  перезапись v2 или явная пометка legacy в отчёте.
- **Ручной труд на каждом прогоне:** `--model-snapshot`, `--baseline-model`,
  `--baseline-snapshot` обязательны; автоматизация появится только вместе с
  оркестратором сравнительного стенда.
- **C-056 красное до прогона своей модели на GB10** (достижимо: `jaxlm`-адаптер
  на стенде), **C-057 — долг до K3-baseline** (недостижим на GB10: 2.8T ≈ 1,4 ТБ
  FP4 против 128 ГБ; путь через C-048 → смету → аренду/API). Цена честности:
  второй красный виден, но не блокирует обучение — только заявление о превосходстве.
- **Риск ложного Comfort:** сравнение оси по нормализованной строке — не
  семантическое; переформулировка оси в пакете без переформулировки в манифесте
  даст красный гейт (это желаемое поведение), обратное — нет.

## Амендмент B (2026-11-11): декомпозиция precondition

Прежняя формулировка «known-red до прогона K3 **на GB10**» была ошибочной: K3 —
2.8T параметров (≈1,4 ТБ даже в FP4, ≥17 узлов H100-80 ГБ только под веса) против
128 ГБ unified на GB10. Ошибка возникла из композиции §10 `ENVIRONMENT-V1`
(«прогон K3», машина не указана) и решения владельца «живые прогоны только на
spark-44c3» (`docs/OPEN-QUESTIONS.md:7`).

Решением владельца (2026-11-11, вариант B) precondition разделён:
**C-056** — калибровка среды на GB10 (блокирует RL-расход, достижимо);
**C-057** — K3-baseline (блокирует заявление о превосходстве, долг P-7b,
путь: C-048 → смета AD-8 → аренда INT-001/API). Вечный красный с недостижимым
триггером недопустим (AD-10).

## Реализация (факт, 2026-11-11)

- Код: `env/axis.py`, `env/schemas.py`, `env/manifest.py`, `env/main.py`,
  `tools/check_victory_axis.py`, `tools/check_manifest_v2.py`,
  `tools/check_rl_precondition.py`; `EXCLUDED_INFRA_RULES` = 22 (C-055/C-056
  инфраструктурные, C-013 — `infra: false`, исполняется в воркспейсе).
- Приёмка: `pytest env/tests -k "manifest or schemas or r1_workspace_ruleset"`
  — 47 passed, 1 skipped; `test_stub_branch_matches_frozen_baseline` зелёный
  после перегенерации фикстуры `env/tests/fixtures/calibrate-stub-baseline.json`
  (pass_rate 0.45 — штатное поведение заглушки с `restore_probability=0.5`,
  это НЕ регрессия вердикта; `evidence/calibration-stub.json` с pass_rate 0.0 —
  другой артефакт, отчёт заглушки на пуле `env/data`).
- C-056: `python3 tools/check_rl_precondition.py` → exit 1, «K3-калибровка не
  найдена» — единственное намеренно красное правило (P-7).

## Reversibility

**costly.** Код и правила обратимы (правка трёх модулей и трёх правил), но
после появления первых манифестов v2 откат означает потерю сопоставимости
доказательств: манифесты придётся либо переписывать, либо помечать legacy, а
заявления, сделанные на них, — пересматривать. Данные (evidence) не
переписываются: v1-артефакты остаются как есть.

## Допущения

```yaml
assumptions:
  - id: AS-1
    statement: "Пакет решения хранит ось на пути choice.victory_axis (не top-level)"
    kind: measurable
    source: docs/DECISION-PACKAGE.json#/choice/victory_axis
    threshold: { op: "exists", value: true }
    review_by: 2027-01-31
  - id: AS-2
    statement: "Калибровка СРЕДЫ (не-заглушка) исполняется на GB10 до открытия RL-расхода"
    kind: qualitative
    source: "владелец, 2026-11-11: живые прогоны только на стенде spark-44c3; C-056"
    review_by: 2027-01-31
  - id: AS-4
    statement: "K3-baseline достигается только через аренду INT-001 (мультиузловой кластер ≥1,4 ТБ FP4) или API для публичного набора; на GB10 невозможен"
    kind: qualitative
    source: "ADR-058 амендмент B: 2.8T/104B против 128 ГБ unified; C-048 блокирует аренду до KPI 800 ток/с"
    review_by: 2027-01-31
  - id: AS-3
    statement: "Сравнительный стенд K3 использует тот же контракт манифеста (§8), а не отдельный"
    kind: qualitative
    source: "ADR-002 п.8, ENVIRONMENT-V1.md:12"
    review_by: 2027-01-31
```

## References

- `ARCHITECTURE-SPINE.md:13–19` (AD-1), `model/AD-1-os-pobedy-odna.md`
- `docs/specs/ENVIRONMENT-V1.md` §8 (run manifest), §10 (калибровка)
- `env/schemas.py:137–236`, `env/manifest.py`, `env/main.py:76,192`
- `CONSTRAINTS.yaml` C-013 (перевод в structural), C-055, C-056, C-038
- `docs/OPEN-QUESTIONS.md` P-1 (долг AD-1), P-2 (AD-3)
- ADR-004 (пиннинг), ADR-014 (манифест A4), ADR-037 (поведенческий слой), ADR-056 (known-red)
