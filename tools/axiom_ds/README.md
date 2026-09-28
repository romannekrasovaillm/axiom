# `axiom_ds` — пайплайн датасета `axiom-domain-ds-v1` (ADR-020, дельты-1..3)

Компонент **E** датасета — сессии кодовых агентов владельца → верифицированные
агентные эпизоды; компоненты **K/D/S** — доменный CPT-корпус из библиотеки
концептов, дистиллятов и скиллов. Сырьё и выход — вне репозитория (AD-6,
C-032/C-033); в git живут код, тесты, карточка датасета и симлинк.

## Порядок ступеней (строгий)

```
deny-list → скраб → эпизодизация → верификация → дедуп → запись jsonl   (E)
фильтр источника → скраб → <HOME> → дедуп K∪D∪S → контейнмент → шард     (K/D/S)
карточка датасета ← манифест CPT + отчёт эпизодов                        (карточка)
```

| Модуль | Ступень | Роль |
|---|---|---|
| `scrub.py` | 1 | deny-list каталогов/файлов-ключей + замена секретов на `<REDACTED>` |
| `episodes.py` | 2 | сессия → эпизоды по границам tool-циклов |
| `verify.py` | 3 | механический класс исхода (контракт сессии или отчёт турникета) |
| `harness.py` | 3 | индекс отчётов турникета (`result.json`, `*.log`) для привязки по времени |
| `dedup.py` | 4 | точные дубли (sha256) + near-dup (MinHash 64×16×4, Jaccard ≥ 0.8) |
| `build.py` | 5 | CLI эпизодов (E), потоковый прогон, числовой отчёт |
| `cpt_serialize.py` | 5 | CLI CPT-корпуса K/D/S: фильтры, дедуп, контейнмент, шарды, манифест |
| `card.py` | 6 | карточка датасета: доли, sha256 шардов, происхождение, решения |

Ступень 1 стоит **до всего**: запретный путь не открывается вовсе (не «прочитать
и вычистить»), а скраб применяется к каждому событию до попадания в ходы.

## Запуск

```bash
# проба на 20 сессиях
python -m axiom_ds.build \
    --source ~/.claude/projects \
    --limit 20 \
    --out /tmp/axiom-ds-probe.jsonl \
    --report /tmp/axiom-ds-probe-report.json

# боевой прогон (по умолчанию пишет в gb10-shared — так задумано C-033)
python -m axiom_ds.build \
    --source ~/.claude/projects \
    --out ~/gb10-shared/datasets/axiom-domain-ds-v1/episodes-v1.jsonl \
    --report ~/gb10-shared/datasets/axiom-domain-ds-v1/episodes-v1-report.json \
    --sft-out  ~/gb10-shared/datasets/axiom-domain-ds-v1/episodes-v1-sft.jsonl \
    --negative-out ~/gb10-shared/datasets/axiom-domain-ds-v1/episodes-v1-negative.jsonl
```

Выход вне `~/gb10-shared`, `/tmp`, `/var/tmp` CLI отклоняет (код возврата 2) —
обход только явным `--allow-any-out`.

## Формат эпизода (jsonl, одна строка = один эпизод, после скраба)

```json
{"id": "ep-<sha256(session‖index)[:16]>",
 "source_session": "<sessionId сессии>",
 "class": "verified-complete | verified-partial | verified-failed | unverified",
 "turns": [{"role": "user|assistant|tool",
            "kind": "text|tool_call|tool_result",
            "content": "…"}],
 "started_at": "2026-09-01T10:00:00.000Z",
 "ended_at":   "2026-09-01T10:05:00.000Z",
 "evidence": "in-session-contract | harness-report | partial-green | none",
 "verification": "partial-green | null"}
```

`content` хода `tool_call` — JSON-строка `{"name": …, "input": …}`: без аргументов
агентная траектория теряет само действие.

`verification` — флаг допуска в SFT-компонент: несут только эпизоды класса
`verified-partial` (частичный исход с механическим подтверждением), у остальных
`null` — их допуск определяет `class`.

## Классы верификации (fail-closed)

* `verified-complete` — идут в SFT-ядро (`--sft-out`);
* `verified-partial` — `partial` **плюс** механический признак зелёного сьюта в
  последних 5 tool-результатах («N passed»/«passed» при отсутствии
  «FAILED»/«ERROR»); в SFT-ядро допускаются **с флагом** `verification:
  partial-green` (дельта-3, ADR-020);
* `verified-failed` — negative-пул RL (`--negative-out`);
* `unverified` — в SFT не попадает, считается счётчиком (в том числе `partial`
  без механического подтверждения).

Противоречие источников трактуется **против** эпизода: если контракт сессии
говорит `complete`, а парный harness-отчёт — `blocked`, класс `verified-failed`
(ложный `verified-complete` отправил бы в SFT эпизод без подтверждённого исхода).

## Отчёт

Числовой, без содержимого: сессии/события/эпизоды, `by_class` (корпус до дедупа)
и `by_class_written` (состав датасета), `sft_ready` (полные) и `sft_partial_ready`
(частичные с флагом), `negative_ready`, объём (`chars`, `approx_tokens` — мера
`chars // 4`, как у CPT-компонента), `out_bytes`/`out_sha256` выхода,
`generated_at`, снятые дубли, `redactions` по правилам, `harness`, длительность.

## Карточка датасета (дельта-3)

```bash
python -m axiom_ds.card build-card                 # в docs/datasets/, из артефактов прогонов
python -m axiom_ds.card build-card --episodes <путь к отчёту/манифесту E>
```

Карточка собирается **из артефактов**: манифеста CPT-корпуса (`manifest-cpt.json`),
отчёта эпизодов (`episodes-v1-report.json`) и — при наличии — отчёта CPT-прогона.
Числа берутся оттуда, а sha256 шардов **пересчитывается с диска**: расхождение —
причина черновика, а не «шум отчёта». Статус `v1` ставится только когда сошлись все
проверки (артефакты на месте, хеши сверены, allowlist корпуса совпадает с
константой, объёмы измерены); иначе `v1-draft` со списком причин.

Файлы: `docs/datasets/axiom-domain-ds-v1-card.md` (человекочитаемая) и
`...-card.json` (машинная). Гипотеза долей при упаковке — отдельно, в
`docs/datasets/axiom-domain-ds-v1-mix.md`; в карточке — только измеренные доли
компонент.

## CPT-корпус K/D/S

```bash
python -m axiom_ds.cpt_serialize build-cpt --probe --limit-files 300   # проба в /tmp
python -m axiom_ds.cpt_serialize build-cpt --restart                   # боевой прогон
```

`--restart` снимает **только сменные артефакты прогона**: шарды данных
`<prefix>-*.jsonl*` и манифест. Курируемые файлы рядом (заметки `*.md`, отчёты
человека) остаются — прогон не вправе их снести (дельта-3).

Граница скраба (дельта-3): правило `hex_env` (длинное hex-значение в контексте
присваивания) **не применяется** к ключам с hash-именами — подстроки
`sha256`/`sha1`/`md5`/`hash`/`salt`/`checksum` в имени ключа. Значения таких ключей
— хеши и соли доменных данных, их замена портит корпус, а не защищает. Для
credential-имён (`key`/`token`/`secret`/`password` и родственных) правило работает
как прежде. Список подстрок — константа `scrub.HASH_KEY_MARKERS`.

## Спот-аудит скраба

Наивный `grep -cE "sk-|AKIA|BEGIN.*PRIVATE"` даёт ложные срабатывания на
обычных словах (`di**sk-**sizing`, `ta**sk-**text`) — он не является проверкой
утечки. Точный аудит:

```bash
grep -cE '\bsk-[A-Za-z0-9_-]{16,}|\bAKIA[0-9A-Z]{16}\b|-----BEGIN[^-]*PRIVATE KEY-----' out.jsonl   # → 0
grep -cE '\bgh[pousr]_[A-Za-z0-9]{20,}|\bxox[baprs]-[A-Za-z0-9-]{10,}|\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.' out.jsonl   # → 0
```

## Тесты

```bash
python -m pytest tools/tests/test_axiom_ds.py tools/tests/test_cpt_serialize.py \
    tools/tests/test_axiom_ds_card.py -q
```

Только синтетические фикстуры (`T-s1`, `T-s3` — границы скраба, `T-s2` — deny-list,
`T-e1` — эпизодизация, `T-v1..T-v4` — классы верификации, `T-d1`, `T-d2` — дедуп,
`T-b1`, `T-b2` — CLI; в `test_cpt_serialize.py` — `T-k1..T-p1` корпуса K/D/S;
в `test_axiom_ds_card.py` — `T-card1..T-card6` карточки): реальные сессии, реальная
библиотека и реальные секреты в тесты не входят.
