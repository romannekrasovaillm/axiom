# Задача для кодового харнесса

ЗАДАЧА (перевести интеграцию кернела на custom_vjp-путь: `K.solve` вместо прямого `K.kernel`).

ФАКТ. В `net/kda.py` кернел KDA вызывается **напрямую** (`net.kernels.kda_ut_solve.kernel`) при `AXIOM_KDA_SOLVE=pallas`. Прогон обучения на GB10 падает:
```
[pretrain] ОШИБКА цикла: Linearization failed to produce known values for all output primals.
  This is typically caused by attempting to differentiate a function that uses an operation with no defined JVP
```
Причина: Pallas-ядро JVP не даёт. В модуле кернела теперь есть **`custom_vjp`-обёртка**: `solve(a, b, ...)` (forward — кернел, backward — аналитический adjoint через транспонированный кернел `kernel_t`), а также `solve_t`. То есть дифференцируемый путь уже реализован и проверен на CPU (gradcheck 9.5e-7; 113 KDA-тестов зелёные) — интеграция просто ещё не переключена.

ЧТО СДЕЛАТЬ.
1. В `net/kda.py` в ветке `AXIOM_KDA_SOLVE=pallas` (функция `_ut_solve_pair`) заменить прямой вызов `kernel(...)` на **`solve(...)`** из `net.kernels.kda_ut_solve` (custom_vjp-путь). Сигнатуры: `solve(a, b)` → решение `(I+L)X=B`; правую часть по-прежнему подавать **конкатенированной** (dk+dv) одним вызовом, как сейчас.
2. Сохранить: дефолт `jax` (прежнее поведение), ошибку при неизвестном значении флага, паритет и допуски.
3. Обновить/дополнить тест `net/tests/test_kda_pallas_flag.py`: при `AXIOM_KDA_SOLVE=pallas` используется именно дифференцируемый путь (`solve`), и **градиент протекает** — тест должен брать `jax.grad` от функции, использующей `_ut_solve_pair`, и проверять, что градиенты конечны и совпадают с дефолтным (`jax`) путём в пределах допуска (на малых формах, CPU).
4. В отчёте — что осталось измерить на GB10: проходит ли обучение end-to-end при `AXIOM_KDA_SOLVE=pallas`; число ядер `batch_trsm_left_kernel`/`MakeBatchPointers` (115 200/230 400 → ?); tok/s, занятость, MFU.

ОГРАНИЧЕНИЯ. Зона: `net/kda.py`, `net/tests/*`. Не трогать `net/kernels/*` (кернел не менять), `net/config.json`, математику, публичные сигнатуры. GPU-прогоны запрещены. Один рычаг — один коммит.

ПРОВЕРКА (ПК): `/home/roman/venv-axiom/bin/python -m pytest -q net/tests/test_kda_chunked_cc.py net/tests/test_kda_wyut.py net/tests/test_kda_dtype_gate_parity.py net/tests/test_kda_pallas_flag.py` → все зелёные.

РЕЗУЛЬТАТ. `.arch-handoff/result.json` + ФИНАЛЬНЫЙ git-commit. В assumptions — что именно заменено и как проверено протекание градиента; в open_questions — список для GPU-замера.

## Границы (scope)

Машинный контракт границ — `MANIFEST.json.scope` (хэш `scope_hash`); изменение вне границ ловится гейтом, а не обсуждается постфактум.

- **Можно писать:** `net/kda.py`, `net/tests/*`
- **Нельзя писать (сильнее allow):** `model/`, `ARCHITECTURE-SPINE.md`, `CONSTRAINTS.yaml`
- **Можно запускать:** `python3 -m pytest*`
- **Сеть:** сети нет (детерминированный узел)

## План отката

Откат: `git reset --hard 7f69bd1` (baseline — последний коммит до работы исполнителя; вся его работа приходит одним коммитом поверх).
Сигналы отката: провал fitness-гейта (`arch-ml control check`), непустой `conflicts_with_prior_decisions`, статус `blocked`.
Владелец решения об откате — solution-архитектор; исполнитель откат не выполняет и не маскирует проблему обходным редизайном.
Обратимость: полная — единая точка изменений, коммит исполнителя.

## Финализация (обязательно)

Результат забирается из git, поэтому перед финальным ответом зафиксируй работу коммитом:

```bash
git add -A -- . ':!.arch-handoff'
git commit -m "<кратко: что реализовано>"
git status --short   # пусто, кроме .arch-handoff/
```

- Коммитится код и тесты; служебный каталог `.arch-handoff/` в коммит не входит.
- Работа без коммита считается невыполненной: оркестратор увидит её только через git log.

## Контракт результата

Финальный ответ обязан завершаться JSON-объектом (после него — ни символа); тот же JSON запиши файлом `.arch-handoff/result.json` (файл переживает обрыв вывода):

Схема объекта: `{"status": "complete|partial|blocked", "assumptions": [], "open_questions": [], "conflicts_with_prior_decisions": []}`

- `status`: `complete` — выполнено полностью; `partial` — частично; `blocked` — заблокировано.
- `assumptions`: допущения, принятые при реализации.
- `open_questions`: вопросы к архитектору.
- `conflicts_with_prior_decisions`: расхождения с принятыми ранее решениями (ADR, spine).

Архитектурный контекст — `ARCHITECTURE.md`, ограничения — `CONSTRAINTS.yaml`, рубрика приёмки — `RUBRIC.yaml` (при наличии).

## Чеклист перед финальным ответом

- [ ] `SPEC.md` (контракты интерфейсов: входы/выходы, структуры данных, границы ошибок, критерии верификации) заполнен архитектором — сверь реализацию с ним; расхождения фиксируй в `conflicts_with_prior_decisions`, а не молчаливым отступлением.
