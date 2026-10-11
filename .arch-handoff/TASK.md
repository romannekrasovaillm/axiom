# Задача для кодового харнесса

DONATE-дельта ADR-054 (стадия-1 «исполнение шага»): устранить ~28k DtoD-копий/шаг (31.5 ГиБ) и связанную диспетчерскую нагрузку, НЕ меняя арифметику.

КОНТЕКСТ (измерено, ADR-053 Am.3 / ADR-054): профиль c64_B2 на GB10 — GPU-ядра 1.55 с из 34 с шага; ~420k runtime-API-вызовов/шаг; ~28k DtoD-копий/шаг (~31.5 ГиБ ≈ размер деревьев params/grads/state). Причина: jit-функции без donate — каждый вызов копирует буферы. Премисса donate уже есть в коде: `params = working_params(master)` пересоздаётся каждый шаг (net/train_loop.py:2999), значит буфер params после grad_fn мёртв.

ЧТО СДЕЛАТЬ.

1. `net/train_loop.py` (функция ~2860-3000):
   a. Рядом с `grad_fn = jax.jit(jax.value_and_grad(loss_fn))` (сейчас :2873) добавить
      `grad_fn_donated = jax.jit(jax.value_and_grad(loss_fn), donate_argnums=(0,))`.
   b. В ветке `len(group) == 1` (:2976) вызывать `grad_fn_donated(params, group[0])` — буфер params
      после этого вызова в итерации не читается (профайлер ниже получает НОВЫЙ params из :2999; сверь
      все использования params между вызовом и концом итерации — если найдётся читатель donated-буфера,
      оставь недонированный вызов и запиши это в assumptions).
   c. Ветку накопления (len(group) > 1, :2984-2995) НЕ трогать: params нужен на каждом микровызове.
   d. `step_fn = optimizer.make_step(...)` (:2874) — передать `donate=not train_config.phase_profile`
      (см. п.2): при живом профайлере grads читается ПОСЛЕ step_fn (profiler.measure, :3011-3021) —
      donate там запрещён.
2. `net/optimizer.py` `make_step(...)` (:217): новый kw-аргумент `donate: bool = False` (дефолт False —
   обратная совместимость всех остальных вызовов); финал:
   `return jax.jit(step, donate_argnums=(0, 1, 2)) if (jit and donate) else (jax.jit(step) if jit else step)`.
   Docstring: donate — контракт владения буферами master/grads/state (аргументы потребляются);
   арифметика идентична.
3. Тесты (новый файл `net/tests/test_donate_step.py`, CPU):
   - паритет: N=3 шага мини-конфига (существующий smoke-конфиг из соседних тестов) с donate=True и
     donate=False → loss-последовательности и финальные params allclose (atol как в существующих
     паритет-тестах); это доказывает «donate не меняет математику»;
   - повторные вызовы не падают (buffer-reuse): 3+ последовательных шага donate-пути без
     RuntimeError «donated buffer»;
   - accum-ветка (micro-batch 2, accum_tokens) на donate-пути не использует grad_fn_donated
     (проверка исходника или поведения);
   - phase_profile=True → make_step получил donate=False (spy/инспекция флага).
4. Docstrings: в train_loop у grad_fn_donated — правило «donate только когда буфер мёртв после вызова;
   params пересоздаётся из master каждый шаг (:2999) — премисса».

ОГРАНИЧЕНИЯ. Только `net/train_loop.py`, `net/optimizer.py`, `net/tests/*`. Математику (loss_fn,
compute_loss, оптимизатор-арифметику), конфиги и остальные модули не трогать. GPU-прогоны не запускать
(их делает архитектор на стенде). JAX на этом узле CPU-only — тесты обязаны идти на CPU.

ПРОВЕРКА (ПК): `/home/roman/venv-axiom/bin/python -m pytest -q net/tests/test_donate_step.py` зелёные;
плюс регресс существующих: `... -m pytest -q net/tests -k "train or optimizer or step"` (CPU-подмножество).

РЕЗУЛЬТАТ: `.arch-handoff/result.json` + финальный git-commit. В assumptions — все найденные читатели
буферов; в open_questions — что перемерить архитектору (A/B c64_B2, nsys DtoD/шаг).

## Границы (scope)

Машинный контракт границ — `MANIFEST.json.scope` (хэш `scope_hash`); изменение вне границ ловится гейтом, а не обсуждается постфактум.

- **Можно писать:** `net/train_loop.py`, `net/optimizer.py`, `net/tests/*`
- **Нельзя писать (сильнее allow):** `model/`, `ARCHITECTURE-SPINE.md`, `CONSTRAINTS.yaml`, `net/config.json`
- **Можно запускать:** `python3 -m pytest*`, `/home/roman/venv-axiom/bin/python -m pytest*`
- **Сеть:** сети нет (детерминированный узел)

## План отката

Откат: `git reset --hard 06a1498` (baseline — последний коммит до работы исполнителя; вся его работа приходит одним коммитом поверх).
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

### Самопроверка через MCP (маршрут Standard/Critical — обязательна)

Перед финальным коммитом вызови через MCP-сервер `arch-spine` (подключён per-run, `.arch-handoff/mcp.json`) ровно эти проверки и добейся `passed=true`:

- `fitness_check` — `{"repo": "<корень прогона>"}`;
- `scope_check` — `{"repo": "<корень прогона>"}`.

Их вызовы пишутся в MCP-журнал прогона (`.arch-handoff/mcp-journal.jsonl`) — это evidence «исполнитель проверял себя». Отсутствие записей = непроверенное, а не проверенное: на приёмке такой прогон не засчитывается (INCOMPLETE). `handoff_status` показывает обязательные проверки маршрута, `contract_validate` — пройдёт ли контракт результата.

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
