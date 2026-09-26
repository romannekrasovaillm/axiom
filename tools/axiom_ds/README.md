# `axiom_ds` — пайплайн агентных эпизодов (ADR-020, дельта-1)

Компонент **E** датасета `axiom-domain-ds-v1`: сессии кодовых агентов владельца →
верифицированные агентные эпизоды. Сырьё и выход — вне репозитория (AD-6,
C-032/C-033); в git живут только код и тесты.

## Порядок ступеней (строгий)

```
deny-list → скраб → эпизодизация → верификация → дедуп → запись jsonl
```

| Модуль | Ступень | Роль |
|---|---|---|
| `scrub.py` | 1 | deny-list каталогов/файлов-ключей + замена секретов на `<REDACTED>` |
| `episodes.py` | 2 | сессия → эпизоды по границам tool-циклов |
| `verify.py` | 3 | механический класс исхода (контракт сессии или отчёт турникета) |
| `harness.py` | 3 | индекс отчётов турникета (`result.json`, `*.log`) для привязки по времени |
| `dedup.py` | 4 | точные дубли (sha256) + near-dup (MinHash 64×16×4, Jaccard ≥ 0.8) |
| `build.py` | 5 | CLI, потоковый прогон, числовой отчёт |

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
 "class": "verified-complete | verified-failed | unverified",
 "turns": [{"role": "user|assistant|tool",
            "kind": "text|tool_call|tool_result",
            "content": "…"}],
 "started_at": "2026-09-01T10:00:00.000Z",
 "ended_at":   "2026-09-01T10:05:00.000Z",
 "evidence": "in-session-contract | harness-report | none"}
```

`content` хода `tool_call` — JSON-строка `{"name": …, "input": …}`: без аргументов
агентная траектория теряет само действие.

## Классы верификации (fail-closed)

* `verified-complete` — **только они** идут в SFT-ядро (`--sft-out`);
* `verified-failed` — negative-пул RL (`--negative-out`);
* `unverified` — в SFT не попадает, считается счётчиком (включая `partial`).

Противоречие источников трактуется **против** эпизода: если контракт сессии
говорит `complete`, а парный harness-отчёт — `blocked`, класс `verified-failed`
(ложный `verified-complete` отправил бы в SFT эпизод без подтверждённого исхода).

## Отчёт

Числовой, без содержимого: сессии/события/эпизоды, `by_class` (корпус до дедупа)
и `by_class_written` (состав датасета), `sft_ready`, `negative_ready`, снятые
дубли, `redactions` по правилам, `harness`, длительность.

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
python -m pytest tools/tests/test_axiom_ds.py -q
```

Только синтетические фикстуры (`T-s1`, `T-s2`, `T-e1`, `T-v1..T-v3`, `T-d1`,
`T-d2`, `T-b1`, `T-b2`): реальные сессии и реальные секреты в тесты не входят.
