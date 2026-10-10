# Задача для кодового харнесса

ЗАДАЧА (Mosaic-перенос одного кернела: KDA-решение (I+L)X = B — стадия 5 ADR-052, parity-перенос).

КОНТЕКСТ (доказано на GB10, jax 0.11.2, окружение ~/venv-axiom-0112):
- Mosaic ALU — `correctness-pass`; **Mosaic MMA — `correctness-pass`** на трёх кейсах bf16 (128x128x8 max_abs 0.0313; 128x64x8 0.0183; 256x128x8 0.0335). См. `evidence/mfu-55/env/REPORT-jax-0112.md`;
- рабочий путь MMA (найден по апстрим-эталону, референсы лежат в `tools/mosaic/reference/_test_mma_harness.py` и `_test_mma_case.py`):
  `plgpu.kernel(..., compiler_params=dataclasses.replace(plgpu.CompilerParams(), lowering_semantics=plgpu.LoweringSemantics.Lane))` (**Lane**, не Warpgroup);
  `plgpu.load(a_ref, layout=plgpu.Layout.MMA_LHS(dtype), optimized=False)`; правая часть — `plgpu.load(b_ref.T, layout=plgpu.Layout.MMA_RHS(dtype), optimized=False)` (RHS в памяти `(n,k)`);
  аккумулятор `plgpu.layout_cast(jnp.zeros((m,n), acc_f32), plgpu.Layout.MMA_ACC(dtype))`; умножение `plgpu.mma(acc, a, b)` (Ampere MMA).
- Переносимый кернел — существующий `net/kernels/kda_ut_solve.py` (Triton-путь): решает `(I+L)·X = B` для `(H=12, C=64, dk+dv=256)`, где `I+L` — единичная нижняя треугольная; реализация — неймановское произведение (≈11 `dot_general`), пин точности f32 `DotAlgorithmPreset.F32_F32_F32` для fp32-пути.

ЧАСТЬ 1 (мелкий фикс, обязателен). В `tools/mosaic/mma_smoke.py` стадия компиляции вызывает `kernel_fn.lower(a, b)` — у объекта `plgpu.kernel` метода `lower` нет (отсюда ложный `blocked`/`AttributeError` без сообщения). Исправить на `jax.jit(kernel_fn).lower(a, b)` (для `plgpu.kernel`-объекта jit-обёртка — работающий путь; проверено на GB10 архитектором). Заодно не глотать traceback: в `error` писать `traceback.format_exc()[-2000:]`, а не только `str(exc)`. Добавить кейс с `N=16` (в текущем наборе меняются K и M, но не N).

ЧАСТЬ 2 (перенос). Создать `net/kernels/kda_ut_solve_mosaic.py` — Mosaic-реализация того же решения, с **тем же контрактом модуля**, что у Triton-версии (`CASES/make_inputs/reference/baseline/kernel/cost/TUNE_SPACE/valid_config`, плюс `kernel_t`/`solve`/`solve_t` если переносишь и транспонированный/дифференцируемый путь):
1. Neumann-итерации через `plgpu.mma` (Lane-семантика), операнды — через `MMA_LHS`/`MMA_RHS`, аккумулятор — `MMA_ACC`; учти, что `C=64` (малая матрица) и `H=12` — подбери grid/раскладку так, чтобы каждая программа решала свою голову/чанк (не делай один гигантский блок).
2. Явный backend-выбор НЕ ломает текущий путь: Triton остаётся дефолтом (`AXIOM_KDA_SOLVE_KERNEL=triton`), Mosaic включается `=mosaic`; неизвестное значение → `ValueError` (без тихого отката). Ленивый импорт Mosaic, чтобы отсутствие/сломанный API не ломал legacy-линию.
3. Паритет: на CPU/`interpret` сравнить Mosaic-реализацию с `reference` (jnp) в пределах обоснованного допуска; если CPU-путь для Mosaic недоступен — зафиксировать это как NOT RUN и оставить архитектору GPU-проверку (не выдавать «прошло» без устройства).
4. Тест `net/tests/test_kda_mosaic_parity.py`: контракт (константы `CASES`, сигнатуры), поведение при `mosaic` без Mosaic-API (честная ошибка), и что `triton`-путь не изменился.

ОГРАНИЧЕНИЯ. Зона: `net/kernels/*`, `tools/mosaic/*`, `tools/tests/*`. НЕ трогать `net/kda.py`, `net/config.json`, `tools/pretrain_run.py` (интеграция в граф — отдельная дельта после GPU-проверки). GPU-прогоны в задаче НЕ запускать — их делает архитектор в `~/venv-axiom-0112` на GB10. Не подменять проверку (`hasattr`, импорт, CPU-прогон, `interpret` не доказывают GPU-корректность; Mosaic-путь обязан быть GPU-проверяемым).

ПРОВЕРКА (ПК): `/home/roman/venv-axiom/bin/python -m pytest -q net/tests/test_kda_mosaic_parity.py tools/tests/test_mma_smoke_contract.py`.

РЕЗУЛЬТАТ. `.arch-handoff/result.json` + ФИНАЛЬНЫЙ git-commit. В assumptions — раскладка grid/блоков и допуск паритета с числом; в open_questions — что проверить архитектору на GB10 (компиляция под sm_121, IR/PTX — что это действительно MMA, а не скалярный цикл, время против Triton-версии и против XLA `triangular_solve`, детерминизм).

## Границы (scope)

Машинный контракт границ — `MANIFEST.json.scope` (хэш `scope_hash`); изменение вне границ ловится гейтом, а не обсуждается постфактум.

- **Можно писать:** `net/kernels/*`, `tools/mosaic/*`, `tools/tests/*`
- **Нельзя писать (сильнее allow):** `model/`, `ARCHITECTURE-SPINE.md`, `CONSTRAINTS.yaml`
- **Можно запускать:** `python3 -m pytest*`, `python3 tools/*`
- **Сеть:** сети нет (детерминированный узел)

## План отката

Откат: `git reset --hard b4e3d57` (baseline — последний коммит до работы исполнителя; вся его работа приходит одним коммитом поверх).
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
