# Задача для кодового харнесса

ЗАДАЧА (Mosaic MMA smoke для JAX 0.11.2 на GB10 — довести до компиляции и корректности).

КОНТЕКСТ. Кандидатная линия JAX 0.11.2 поднята (ADR-052), Mosaic ALU подтверждён (`correctness-pass`). Публичный MMA-API есть: `jax.experimental.pallas.mosaic_gpu.mma(acc, a, b)` — «Computes `acc + a @ b` synchronously using **Ampere MMA instructions**»; есть layout'ы `plgpu.Layout.MMA_ACC / MMA_LHS / MMA_RHS` (на legacy 0.10.2 их не было). Мой smoke-кернел (по мотивам апстрим-теста) падает:
```
VerificationError: 'mosaic_gpu.mma' op operand #1 must be vector of A type supported by the `a` and `b` operands
  of the synchronous `mma` op values of ranks 2, but got 'vector<128x128xf32>'
  %99 = "mosaic_gpu.mma"(%93, %95, %98) : (vector<128x8xf32>, vector<128x128xf32>, vector<128x8xf32>) -> vector<128x8xf32>
```
То есть `plgpu.load(a_ref, layout=plgpu.Layout.MMA_LHS(dtype), optimized=False)` при загрузке **из GMEM** не даёт нужный тип/layout — операнд остаётся f32. Апстрим-тест `tests/pallas/mosaic_gpu_test.py::test_mma` (JAX tag `jax-v0.11.2`) грузит операнды иначе — там операнды попадают в SMEM (путь `copy_gmem_to_smem` + `wait_gmem_to_smem`, либо SMEM-рефы), и `plgpu.mma` получает правильные layout'ы.

ЧТО СДЕЛАТЬ.
1. Достать эталон: тест `tests/pallas/mosaic_gpu_test.py` из тега `jax-v0.11.2` (сеть разрешена через прокси; файл ~10k строк, нужен тест `test_mma` и его обвязка `self.kernel`/fixtures) — понять точный путь загрузки операндов для `plgpu.mma`.
2. Написать `tools/mosaic/mma_smoke.py` — самодостаточный скрипт: M → K → N (взять реальные формы из задачи: K=128, N=8..128 как в тесте, dtype bf16), входы готовятся на numpy, эталон — numpy/jnp в fp32. Скрипт обязан: скомпилировать и **исполнить на устройстве** (не `interpret`, не CPU), сверить результат с эталоном с явным допуском (обосновать число), записать JSON-отчёт (status/error/shape/dtype/max_abs/rel/секунды) в указанный путь. Аргументы: `--output PATH`.
3. Добавить `tools/tests/test_mma_smoke_contract.py` — CPU-проверки контракта скрипта (наличие аргумента `--output`, структура JSON при прогоне на CPU/`interpret` или корректный отказ), чтобы не гонять GPU из CI.
4. В отчёте — точный путь API (какие функции/аргументы), почему предыдущий вариант не работал (root cause), и что осталось проверить на стенде (компиляция под sm_121, IR/PTX, детерминизм, несколько K-tile).

ОГРАНИЧЕНИЯ. Зона: `tools/mosaic/*`, `tools/tests/*`. НЕ трогать `net/`, `net/kernels/`, `net/config.json`, `tools/pretrain_run.py`. GPU-прогоны делаются только архитектором на GB10 (в задаче их запускать не нужно; если репозиторий исполняется на CPU-хосте — используй `interpret`/`JAX_PLATFORMS=cpu` только для контрактных тестов, а факт GPU-компиляции не объявляй).
- Не подменять проверку: `hasattr`, импорт, CPU-прогон и `interpret=True` НЕ являются доказательством GPU-компиляции.
- Не выдумывать API (`mgpu_mma_` не использовать как dependency, если он внутренний): опираться на публичный `plgpu.mma`.

ПРОВЕРКА (ПК): `/home/roman/venv-axiom/bin/python -m pytest -q tools/tests/test_mma_smoke_contract.py`.

РЕЗУЛЬТАТ. `.arch-handoff/result.json` + ФИНАЛЬНЫЙ git-commit. В assumptions — точный путь загрузки операндов и обоснование допуска; в open_questions — что должен проверить архитектор на GB10 (компиляция под sm_121, IR/PTX/SASS, детерминизм, расширение на K-tile и целевые формы кампании).

## Границы (scope)

Машинный контракт границ — `MANIFEST.json.scope` (хэш `scope_hash`); изменение вне границ ловится гейтом, а не обсуждается постфактум.

- **Можно писать:** `tools/mosaic/*`, `tools/tests/*`
- **Нельзя писать (сильнее allow):** `model/`, `ARCHITECTURE-SPINE.md`, `CONSTRAINTS.yaml`
- **Можно запускать:** `python3 -m pytest*`, `python3 tools/*`
- **Сеть:** только локальный прокси-эндпоинт (канон ADR-050)

## План отката

Откат: `git reset --hard 9916ac3` (baseline — последний коммит до работы исполнителя; вся его работа приходит одним коммитом поверх).
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
