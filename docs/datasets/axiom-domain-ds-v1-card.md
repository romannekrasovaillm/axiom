# Карточка датасета `axiom-domain-ds-v1` — финал

- **Статус:** `v1`, собрана 2026-09-28T11:28:09Z
- **Решение:** ADR-020 (дельта-1..3), ADR-004 (карточка + хеш), ADR-005 (SFT-стадия)
- **Расположение:** `<HOME>/gb10-shared/datasets/axiom-domain-ds-v1/` (C-032/C-033: корпуса на gb10-shared); в репозитории — симлинк `data/datasets/axiom-domain-ds-v1` на этот каталог
- **Инструменты сборки:** `tools/axiom_ds/build.py` (E), `tools/axiom_ds/cpt_serialize.py` (K/D/S), `tools/axiom_ds/card.py` (эта карточка)

## 1. Состав и измеренные доли

| Компонент | Что это | Записей | Файлов | approx-токенов | Доля | Основа доли |
|---|---|---|---|---|---|---|
| E | агентные эпизоды сессий (класс исхода — механический) | 2 043 | 1 394 | 60 409 494 | 52.9 % | manifest |
| K | концепты из статей (карточки библиотеки) | 96 217 | 107 916 | 44 812 494 | 39.2 % | manifest |
| D | дистилляты статей | 5 658 | 5 658 | 8 169 289 | 7.1 % | manifest |
| S | процедурные скиллы (SKILL.md плагинов домена) | 520 | 520 | 834 142 | 0.7 % | manifest |
| **Итого** |  |  |  | 114 225 419 | 100.0 % |  |

Доли считаются по измеренным компонентам (E, K, D, S); измерены все компоненты, доли в сумме 100 %.
Мера объёма — `approx_tokens = max(1, len(text) // 4)` (ADR-021), общая для E и K/D/S; упаковка в 8K-последовательности — при CPT-лупе.

## 2. Шард-файлы и хеши

| Компонент | Файл | Записей | Байт | sha256 (пересчёт с диска) | Сверка с манифестом |
|---|---|---|---|---|---|
| D+K+S | cpt-kds-v0.1-00000.jsonl.zst | 102 395 | 79 987 774 | 286f0a29e1d6f5cd… | совпал |
| E | <HOME>/gb10-shared/datasets/axiom-domain-ds-v1/episodes-v1.jsonl | 2 043 | 331 887 836 | 2bd5508c72e3ecfb… | совпал |

Полные хеши — в машинной карточке (`axiom-domain-ds-v1-card.json`). Сверка пересчитывает sha256 по файлам на диске: расхождение — сигнал подмены/дрейфа шарда, а не «шум отчёта».

## 3. Происхождение: источники и правила отбора

- артефакты, из которых собрана карточка: манифест CPT `<HOME>/gb10-shared/datasets/axiom-domain-ds-v1/manifest-cpt.json`, отчёт CPT `<HOME>/gb10-shared/datasets/axiom-domain-ds-v1/cpt-kds-v0.1-report.json`, артефакт эпизодов `<HOME>/gb10-shared/datasets/axiom-domain-ds-v1/episodes-v1-report.json`

**E** — агентные эпизоды сессий (класс исхода — механический)

- источники: `<HOME>/.claude/projects`
- артефакт прогона: `<HOME>/gb10-shared/datasets/axiom-domain-ds-v1/episodes-v1-report.json` (2026-09-28T11:00:11Z)
- класс исхода — механический: контракт сессии либо парный harness-отчёт; классы `verified-complete`, `verified-partial`, `verified-failed`, `unverified`
- в SFT-компонент: `verified-complete`, `verified-partial`; `verified-partial` — статус `partial` плюс зелёный сьют в последних 5 tool-результатах (`\b(?:\d+\s+)?passed\b` при отсутствии `(?i)\b(?:failed|error)s?\b`)
- скраб и deny-list — до эпизодизации: значения секретов до карточки не доходят, в отчёте только счётчики
- SFT-компонент: `verified-complete` 122 + `verified-partial` (флаг `partial-green`) 0 — 0.0 % компоненты (0.0 % всех записанных E); negative-пул `verified-failed` 3
- состав по классам (записано): `unverified` 1 918, `verified-complete` 122, `verified-failed` 3

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
- s_plugins: `agent-harness`, `agent-harnesses`, `agent-infra`, `agentic-rl`, `agents`, `arch-core`, `arch-distilled`, `aws-builders`, `cpt`, `credit-assignment`, `data-curator`, `dka`, `document-tools`, `effort`, `frontier-intelligence`, `frontier-lab`, `gb10`, `kat`, `kimi`, `laguna`, `lambert-rl`, `memory-systems`, `misc-tools`, `patterns-integration`, `patterns-resilience`, `pretrain`, `reasoning`, `rl-training`, `safety-eval`, `tui-agent-skills`, `verification`
- skill_filename: SKILL.md
- min_text_chars: 32
- containment: True
- containment_threshold: 0.8
- записано по плагинам: `rl-training` 45, `laguna` 43, `agent-harnesses` 41, `memory-systems` 40, `safety-eval` 29, `credit-assignment` 26, `kimi` 26, `agent-infra` 25, `gb10` 25, `reasoning` 25, `lambert-rl` 23, `agents` 22, `frontier-lab` 22, `kat` 21, `data-curator` 18, `effort` 16, `tui-agent-skills` 14, `agentic-rl` 11, `pretrain` 10, `frontier-intelligence` 8, `document-tools` 6, `misc-tools` 6, `agent-harness` 5, `dka` 5, `cpt` 4, `verification` 4

## 4. Решения по составу

- **Allowlist плагинов:** решение владельца 2026-09-28 (ADR-020, дельта-3); база 22 + расширение дельты-3 9 = 31 плагинов.
- **Legacy-корпус** `~/gb10-shared/datasets/cpt_corpus_full.txt` — доля в v1 0%: txt-склейка прошлых корпусов (30.8 % пересечения типов с K/D/S): в v1 не входит — состав v1 целиком собирается сериализатором K/D/S (ADR-020, дельта-3)
- **Контейнмент D→K** — монитор пересечения D↔K (не удаляющая ступень); премиса вложения «карточка K внутри дистиллята D» опровергнута измерением 28.09.2026: на боевом корпусе при пороге окна 0.55 ступень сняла 0 записей — она работает наблюдателем (no-op) и оставлена в коде как непрерывный монитор пересечения. Прогон: контейнеров 5 658, проверок 96 217, снято 0 (порог 0.8, окно 0.55). Замер пересечения по парам «одна статья»: max оконный Jaccard 0.3524 при пороге 0.55, пар ≥ порога — 0 (294 пар; ADR-020 (дельта-2/3): диагностика контейнмента D↔K 28.09.2026 — 294 реальные пары «одна статья»; боевой прогон дельты-2 (96 217 K × 5 658 D) пар ≥ порога не дал)
- **Микс при упаковке** — гипотеза долей при упаковке CPT/SFT — в микс-декларации; в карточке только измеренные доли компонент (`docs/datasets/axiom-domain-ds-v1-mix.md`).

## 5. Плагины S: таблица allowlist

| Плагин | В allowlist | В прогоне корпуса | Файлов | Записей | approx-токенов |
|---|---|---|---|---|---|
| agent-harness | да | да | 5 | 5 | 6 674 |
| agent-harnesses | да | да | 41 | 41 | 76 122 |
| agent-infra | да | да | 25 | 25 | 54 754 |
| agentic-rl | да | да | 11 | 11 | 16 989 |
| agents **+Δ3** | да | да | 22 | 22 | 36 260 |
| arch-core | да | да | 0 | 0 | 0 |
| arch-distilled | да | да | 0 | 0 | 0 |
| aws-builders | да | да | 0 | 0 | 0 |
| cpt | да | да | 4 | 4 | 7 752 |
| credit-assignment **+Δ3** | да | да | 26 | 26 | 44 141 |
| data-curator | да | да | 18 | 18 | 22 480 |
| dka | да | да | 5 | 5 | 9 236 |
| document-tools | да | да | 6 | 6 | 10 197 |
| effort | да | да | 16 | 16 | 25 082 |
| frontier-intelligence | да | да | 8 | 8 | 15 702 |
| frontier-lab | да | да | 22 | 22 | 37 563 |
| gb10 **+Δ3** | да | да | 25 | 25 | 39 758 |
| kat **+Δ3** | да | да | 21 | 21 | 22 347 |
| kimi | да | да | 26 | 26 | 31 953 |
| laguna | да | да | 43 | 43 | 55 164 |
| lambert-rl **+Δ3** | да | да | 23 | 23 | 32 019 |
| memory-systems **+Δ3** | да | да | 40 | 40 | 71 094 |
| misc-tools | да | да | 6 | 6 | 12 487 |
| patterns-integration | да | да | 0 | 0 | 0 |
| patterns-resilience | да | да | 0 | 0 | 0 |
| pretrain | да | да | 10 | 10 | 15 891 |
| reasoning **+Δ3** | да | да | 25 | 25 | 33 185 |
| rl-training **+Δ3** | да | да | 45 | 45 | 91 211 |
| safety-eval **+Δ3** | да | да | 29 | 29 | 49 410 |
| tui-agent-skills | да | да | 14 | 14 | 9 996 |
| verification | да | да | 4 | 4 | 6 675 |

Вне allowlist: 163 плагинов / 1 081 файлов SKILL.md — в корпус не входят (чужой домен контура).

## 6. Скраб и дедуп (счётчики)

| Контур | Замен скраба | Точных дублей | Near-дублей | Оставлено |
|---|---|---|---|---|
| E (эпизоды) | 1 905 | 202 | 179 | 2 043 |
| K/D/S (CPT-корпус) | 1 | 9 765 | 108 | 102 395 |

Правила скраба и их счётчики (по шаблонам):

| Правило | Замен |
|---|---|
| aws | 49 |
| bearer | 25 |
| env_assignment | 636 |
| github | 22 |
| hex_env | 733 |
| jwt | 16 |
| openai | 96 |
| private_key | 56 |
| private_key_dangling | 51 |
| private_key_marker | 5 |
| slack | 8 |
| url_password | 209 |

Дедуп K/D/S — общий пул дедупа K∪D∪S; E дедуплицируется отдельно (пересечения D↔K, S↔D↔K ожидаемы); шум E-контура в пул K/D/S не попадает.

Near-dup LSH-коллизии — известное поведение ступени (K/D/S: 108, E: 179); проверка повторного дедупа выхода: записей 2 042, точных дублей снято 0, near-дублей снято 0 (источник: дельта-3 прогон 28.09.2026).

## 7. Что не сделано и открытые вопросы

- проверки карточки пройдены; состав зафиксирован
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
