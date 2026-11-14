# C-056: блокер калибровки среды и состояние исполнения

- Дата: 2026-11-14
- Правило: C-056 (ADR-058 Am.B, §10 ENVIRONMENT-V1)
- Проверка: `python3 tools/check_rl_precondition.py --part env`
- Статус: **в очереди на стенде**, гейт остаётся красным до годного отчёта

## Симптом (инцидент 13.11.2026)

Калибровка среды запускалась в свободное окно GPU (`axiom-run/calib-env-20261112-005655`)
и падала на первом форварде после восстановления чекпоинта:

```
error: persistent kernel: configuration (C=64, H=12, dk=128, dv=128, dtype=bfloat16)
is not among the live cases of the sieve (mma tiling rules and the GB10 SMEM floor --
see valid_config); no silent fallback
```

## Причина

Не качество модели и не чекпоинт — **несовпадение runtime-режима**:

| параметр | живое обучение | скрипт калибровки (было) |
|---|---|---|
| `AXIOM_KDA_CC_IMPL` | `jax` | не задан → дефолт `persistent` |
| `kda_wyut_chunk` | 32 (`--chunk-size 32`) | 64 (`net/config.json`) |

Persistent-ядро (`net/kernels/kda_cc_fused_persist.py::valid_config`) отвергает
`C=64` по SMEM-бюджету GB10 (135 424 B > 101 376 B); живой вариант — `C=32`.
Отказ намеренный («no silent fallback»), поэтому падение корректное — ошибочен
был режим запуска, а не ядро.

## Исправление

Скрипт `~/.scratch/calib-env-queue.sh` на стенде (бэкап `.bak-20261114`):

1. `export AXIOM_KDA_CC_IMPL=jax AXIOM_KDA_SOLVE=jax` — тот же режим, что у обучения;
2. `--config .scratch/config-calib-chunk32.json` — копия `net/config.json` с
   `kda_wyut_chunk: 32`;
3. чекпоинт и токенизатор выбираются согласованно: финальный
   `long-jax2-ab14050/.../step-00050000` (+ `tokenizer-v2`), иначе пилот
   `pilot-l3full-endurance/step-00000900` (+ старый токенизатор).

Проверено: `bash -n` — OK; все пути на стенде существуют; конфиг читается
(`kda_wyut_chunk=32`).

## Состояние

Очередь запущена 2026-11-14 07:48Z (PID 1653626), ждёт освобождения GPU по
AD-7/C-040 (опрос 300 с). Лог: `~/axiom-run/calib-env-20261114-044840/queue.log`.
Отчёт появится как `calibration-<model>.json` в том же каталоге.

**Ограничение по существу:** коридор `[10%, 90%]` требует модели, решающей
агентные задачи (20 публичных задач, бюджет до 10 ч). База на ~0,7B токенов
обучения ожидаемо даст pass_rate ≈ 0 — отчёт будет **честным замером, но
не годным** для C-056. Правило закрывается только после SFT-чекпоинта,
проходящего коридор; скрипт к этому готов.

## Ссылки

- `tools/check_rl_precondition.py` — критерий (готовность, пиннинг, коридор).
- `changes/` кейса — контракт калибровки; ADR-058 Am.B.
- Инцидент-лог: `~/axiom-run/calib-env-20261112-005655/queue.log:365-367`.
