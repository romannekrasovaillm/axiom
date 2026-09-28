# Карточка датасета `axiom-domain-ds-v1` — ЧЕРНОВИК

- **Статус:** `v1-draft`, собрана 2026-09-28T10:29:22Z
- **Решение:** ADR-020 (дельта-1..3), ADR-004 (карточка + хеш), ADR-005 (SFT-стадия)
- **Расположение:** `<HOME>/gb10-shared/datasets/axiom-domain-ds-v1/` (C-032/C-033: корпуса на gb10-shared); в репозитории — симлинк `data/datasets/axiom-domain-ds-v1` на этот каталог
- **Инструменты сборки:** `tools/axiom_ds/build.py` (E), `tools/axiom_ds/cpt_serialize.py` (K/D/S), `tools/axiom_ds/card.py` (эта карточка)

**Почему черновик** (проверки карточки, а не оценка):

- allowlist корпуса (22) не совпадает с константой (31): корпус собран до расширения дельты-3 — нужна пересборка K/D/S
- в allowlist константы, но не в прогоне корпуса: agents, credit-assignment, gb10, kat, lambert-rl, memory-systems, reasoning, rl-training, safety-eval — нужна пересборка K/D/S

## 1. Состав и измеренные доли

| Компонент | Что это | Записей | Файлов | approx-токенов | Доля | Основа доли |
|---|---|---|---|---|---|---|
| E | агентные эпизоды сессий (класс исхода — механический) | 2 042 | 1 393 | 60 284 039 | 53.0 % | manifest |
| K | концепты из статей (карточки библиотеки) | 96 217 | 107 916 | 44 848 799 | 39.4 % | chars//4 (манифест без пофайловых токенов) |
| D | дистилляты статей | 5 658 | 5 658 | 8 171 390 | 7.2 % | chars//4 (манифест без пофайловых токенов) |
| S | процедурные скиллы (SKILL.md плагинов домена) | 263 | 263 | 413 059 | 0.4 % | chars//4 (манифест без пофайловых токенов) |
| **Итого** |  |  |  | 113 717 287 | 100.0 % |  |

Доли считаются по измеренным компонентам (E, K, D, S); измерены все компоненты, доли в сумме 100 %.
Мера объёма — `approx_tokens = max(1, len(text) // 4)` (ADR-021), общая для E и K/D/S; упаковка в 8K-последовательности — при CPT-лупе.

## 2. Шард-файлы и хеши

| Компонент | Файл | Записей | Байт | sha256 (пересчёт с диска) | Сверка с манифестом |
|---|---|---|---|---|---|
| D+K+S | cpt-kds-v0.1-00000.jsonl.zst | 102 138 | 79 193 928 | eecba497f7c33199… | совпал |
| E | <HOME>/gb10-shared/datasets/axiom-domain-ds-v1/episodes-v1.jsonl | 2 042 | 331 231 314 | 7467209ed292ca81… | совпал |

Полные хеши — в машинной карточке (`axiom-domain-ds-v1-card.json`). Сверка пересчитывает sha256 по файлам на диске: расхождение — сигнал подмены/дрейфа шарда, а не «шум отчёта».

## 3. Происхождение: источники и правила отбора

- артефакты, из которых собрана карточка: манифест CPT `<HOME>/gb10-shared/datasets/axiom-domain-ds-v1/manifest-cpt.json`, отчёт CPT `<HOME>/gb10-shared/datasets/axiom-domain-ds-v1/cpt-kds-v0.1-report.json`, артефакт эпизодов `<HOME>/gb10-shared/datasets/axiom-domain-ds-v1/episodes-v1-report.json`

**E** — агентные эпизоды сессий (класс исхода — механический)

- источники: `<HOME>/.claude/projects`
- артефакт прогона: `<HOME>/gb10-shared/datasets/axiom-domain-ds-v1/episodes-v1-report.json` (2026-09-28T10:22:43Z)
- класс исхода — механический: контракт сессии либо парный harness-отчёт; классы `verified-complete`, `verified-partial`, `verified-failed`, `unverified`
- в SFT-компонент: `verified-complete`, `verified-partial`; `verified-partial` — статус `partial` плюс зелёный сьют в последних 5 tool-результатах (`\b(?:\d+\s+)?passed\b` при отсутствии `\b(?:FAILED|ERROR)\b`)
- скраб и deny-list — до эпизодизации: значения секретов до карточки не доходят, в отчёте только счётчики
- состав по классам (записано): `unverified` 1 918, `verified-complete` 120, `verified-failed` 3, `verified-partial` 1

**K** — концепты из статей (карточки библиотеки)

- источники: `<HOME>/library/concepts`
- k_types: `algorithmic`, `algorithmic_primitive`, `architectural_component`, `benchmark`, `dataset`, `design_proposition`, `hypothesis`, `task`
- k_levels: `α`, `β`, `γ`
- k_title_prefix: True
- min_text_chars: 32
- containment: True
- containment_threshold: 0.8
- состав источника по типам карточек: `algorithmic_primitive` 49 046, `architectural_component` 42 070, `benchmark` 13 416, `hypothesis` 2 911, `task` 352, `design_proposition` 8, `dataset` 1
- уровни карточек: `α` 52 923, `β` 43 905, `unknown` 10 466, `γ` 505, `advanced` 2, `meta` 2, `core` 1

**D** — дистилляты статей

- источники: `<HOME>/library/distillate`
- d_subdirs: `2_статьи`, `3_блоги`
- min_text_chars: 32
- containment: True
- containment_threshold: 0.8

**S** — процедурные скиллы (SKILL.md плагинов домена)

- источники: `<HOME>/experiments/agents/0710-ariadna/plugins`
- s_plugins: `agent-harness`, `agent-harnesses`, `agent-infra`, `agentic-rl`, `arch-core`, `arch-distilled`, `aws-builders`, `cpt`, `data-curator`, `dka`, `document-tools`, `effort`, `frontier-intelligence`, `frontier-lab`, `kimi`, `laguna`, `misc-tools`, `patterns-integration`, `patterns-resilience`, `pretrain`, `tui-agent-skills`, `verification`
- skill_filename: SKILL.md
- min_text_chars: 32
- containment: True
- containment_threshold: 0.8
- записано по плагинам: `laguna` 43, `agent-harnesses` 40, `kimi` 26, `agent-infra` 25, `frontier-lab` 22, `data-curator` 18, `effort` 16, `tui-agent-skills` 14, `agentic-rl` 11, `pretrain` 10, `frontier-intelligence` 8, `document-tools` 6, `misc-tools` 6, `agent-harness` 5, `dka` 5, `cpt` 4, `verification` 4

## 4. Решения по составу

- **Allowlist плагинов:** решение владельца 2026-09-28 (ADR-020, дельта-3); база 22 + расширение дельты-3 9 = 31 плагинов.
- **Legacy-корпус** `~/gb10-shared/datasets/cpt_corpus_full.txt` — доля в v1 0%: txt-склейка прошлых корпусов (30.8 % пересечения типов с K/D/S): в v1 не входит — состав v1 целиком собирается сериализатором K/D/S (ADR-020, дельта-3)
- **Контейнмент D→K** — монитор пересечения D↔K (не удаляющая ступень); премиса вложения «карточка K внутри дистиллята D» опровергнута измерением 28.09.2026 (0 пар на 96 217 × 5 658 при пороге 0.55): на боевом корпусе ступень — no-op, оставлена наблюдателем
- **Микс при упаковке** — гипотеза долей при упаковке CPT/SFT — в микс-декларации; в карточке только измеренные доли компонент (`docs/datasets/axiom-domain-ds-v1-mix.md`).

## 5. Плагины S: таблица allowlist

| Плагин | В allowlist | В прогоне корпуса | Файлов | Записей | approx-токенов |
|---|---|---|---|---|---|
| agent-harness | да | да | 5 | 5 | — |
| agent-harnesses | да | да | 40 | 40 | — |
| agent-infra | да | да | 25 | 25 | — |
| agentic-rl | да | да | 11 | 11 | — |
| agents **+Δ3** | да | нет | 22 | 0 | — |
| arch-core | да | да | 0 | 0 | — |
| arch-distilled | да | да | 0 | 0 | — |
| aws-builders | да | да | 0 | 0 | — |
| cpt | да | да | 4 | 4 | — |
| credit-assignment **+Δ3** | да | нет | 26 | 0 | — |
| data-curator | да | да | 18 | 18 | — |
| dka | да | да | 5 | 5 | — |
| document-tools | да | да | 6 | 6 | — |
| effort | да | да | 16 | 16 | — |
| frontier-intelligence | да | да | 8 | 8 | — |
| frontier-lab | да | да | 22 | 22 | — |
| gb10 **+Δ3** | да | нет | 25 | 0 | — |
| kat **+Δ3** | да | нет | 21 | 0 | — |
| kimi | да | да | 26 | 26 | — |
| laguna | да | да | 43 | 43 | — |
| lambert-rl **+Δ3** | да | нет | 23 | 0 | — |
| memory-systems **+Δ3** | да | нет | 40 | 0 | — |
| misc-tools | да | да | 6 | 6 | — |
| patterns-integration | да | да | 0 | 0 | — |
| patterns-resilience | да | да | 0 | 0 | — |
| pretrain | да | да | 10 | 10 | — |
| reasoning **+Δ3** | да | нет | 25 | 0 | — |
| rl-training **+Δ3** | да | нет | 45 | 0 | — |
| safety-eval **+Δ3** | да | нет | 27 | 0 | — |
| tui-agent-skills | да | да | 14 | 14 | — |
| verification | да | да | 4 | 4 | — |

Вне allowlist: 163 плагинов / 1 080 файлов SKILL.md — в корпус не входят (чужой домен контура).

Столбец approx-токенов заполняется для шардов, собранных после дельты-3: в артефакте прогона `<HOME>/gb10-shared/datasets/axiom-domain-ds-v1/manifest-cpt.json` пофайловых токенов ещё нет — стоят прочерки, а не нули.

## 6. Скраб и дедуп (счётчики)

| Контур | Замен скраба | Точных дублей | Near-дублей | Оставлено |
|---|---|---|---|---|
| E (эпизоды) | 1 865 | 202 | 179 | 2 042 |
| K/D/S (CPT-корпус) | 0 | 9 765 | 108 | 102 138 |

Правила скраба и их счётчики (по шаблонам):

| Правило | Замен |
|---|---|
| aws | 45 |
| bearer | 23 |
| env_assignment | 637 |
| github | 18 |
| hex_env | 722 |
| jwt | 12 |
| openai | 92 |
| private_key | 52 |
| private_key_dangling | 49 |
| private_key_marker | 5 |
| slack | 6 |
| url_password | 204 |

Дедуп K/D/S — общий пул дедупа K∪D∪S; E дедуплицируется отдельно (пересечения D↔K, S↔D↔K ожидаемы); шум E-контура в пул K/D/S не попадает.

## 7. Что не сделано и открытые вопросы

- allowlist корпуса (22) не совпадает с константой (31): корпус собран до расширения дельты-3 — нужна пересборка K/D/S
- в allowlist константы, но не в прогоне корпуса: agents, credit-assignment, gb10, kat, lambert-rl, memory-systems, reasoning, rl-training, safety-eval — нужна пересборка K/D/S
- финальные доли микса (K+D+S домен / E-эпизоды / публичный претрейн-текст) определяются при CPT-лупе — не карточкой
- токенизация и упаковка 8K — вне этого пайплайна (`net/data.py`)

## 8. Воспроизведение

```bash
# тесты пайплайнов (синтетика, без сети)
python -m pytest tools/tests/test_axiom_ds.py tools/tests/test_cpt_serialize.py tools/tests/test_axiom_ds_card.py -q

# карточка из артефактов прогонов
python -m axiom_ds.card build-card

# пересборка компоненты E (сессии → эпизоды)
python -m axiom_ds.build --source ~/.claude/projects \
    --out <HOME>/gb10-shared/datasets/axiom-domain-ds-v1/episodes-v1.jsonl \
    --report <HOME>/gb10-shared/datasets/axiom-domain-ds-v1/episodes-v1-report.json \
    --sft-out <HOME>/gb10-shared/datasets/axiom-domain-ds-v1/episodes-v1-sft.jsonl

# пересборка CPT-корпуса K/D/S (полный прогон, ~40 мин)
python -m axiom_ds.cpt_serialize build-cpt --restart
```

Приватные корпуса в git не попадают: в репозитории — эта карточка и симлинк `data/datasets/axiom-domain-ds-v1` на `~/gb10-shared` (C-032/C-033, AD-6).
