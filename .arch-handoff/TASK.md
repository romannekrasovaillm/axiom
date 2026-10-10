# Задача для кодового харнесса

ЗАДАЧА (Mosaic MMA smoke для JAX 0.11.2 / GB10 — довести до компиляции и корректности).

ВАЖНО ПРО СЕТЬ: сеть в прогоне ОТКЛЮЧЕНА (`net: none`). Эталонный тест JAX 0.11.2 уже вырезан в репозиторий локально — скачивать ничего не нужно:
- `tools/mosaic/reference/_test_mma_harness.py` — обвязка апстрим-теста (`class PallasTest`: `LOWERING_SEMANTICS`, метод `kernel()`, который вызывает настоящий `plgpu.kernel` с `compiler_params=dataclasses.replace(plgpu.CompilerParams(), lowering_semantics=self.LOWERING_SEMANTICS)`);
- `tools/mosaic/reference/_test_mma_case.py` — сам `test_mma` (формы M=K=128, N=8; dtype bf16/fp16/fp8/int8; `acc_dtype=float32`), ключевые строки:
```python
acc = plgpu.layout_cast(jnp.zeros((m, n), acc_dtype), plgpu.Layout.MMA_ACC(dtype))
a = plgpu.load(a_ref, layout=plgpu.Layout.MMA_LHS(dtype), optimized=False)
b = plgpu.load(b_ref.T, layout=plgpu.Layout.MMA_RHS(dtype), optimized=False)
o_ref[...] = plgpu.mma(acc, a, b)
```

СОСТОЯНИЕ (мои проверки на GB10, `~/venv-axiom-0112`, jax 0.11.2):
- `plgpu.mma(acc, a, b)` существует и документирован как «Computes acc + a @ b synchronously using Ampere MMA instructions»;
- layout'ы `plgpu.Layout.MMA_ACC / MMA_LHS / MMA_RHS` существуют;
- `plgpu.kernel` имеет сигнатуру: `(body, *, out_type, scratch_types, compiler_params, grid, grid_names, cluster, cluster_names, num_threads, thread_name)`;
- мой smoke (`~/axiom-run/mosaic-mma-smoke.py`, копия в `evidence/mfu-55/env/mosaic-mma-smoke.py`) падает:
  `VerificationError: 'mosaic_gpu.mma' op operand #1 must be vector of A type supported by the 'a' and 'b' operands … got 'vector<128x128xf32>'`
  (то есть A приходит f32, а не bf16 → layout не применён при загрузке **из GMEM** без правильных compiler_params/пути загрузки);
- `plgpu.DimensionSemantics` в модуле `mosaic_gpu` НЕ экспортируется (ищи в `jax.experimental.pallas` или во внутреннем модуле — но не превращай внутренний путь в публичный контракт без обоснования).

ЧТО СДЕЛАТЬ.
1. Воспроизвести путь из эталона: вызвать `plgpu.kernel` **с явными `compiler_params`** (в первую очередь разобраться с `lowering_semantics` и, если требуется, `dimension_semantics` для `grid`), и загрузить операнды так, как это делает тест (в т.ч. `b_ref.T` для `MMA_RHS`). Входные массивы — numpy, приводить к dtype через `jnp.asarray(x, dtype=jnp.dtype("bfloat16"))` (не `ndarray.astype(jnp.bfloat16)`).
2. Написать `tools/mosaic/mma_smoke.py`: самодостаточный скрипт с `--output PATH`; M=128, K=128, N=8 (плюс опционально ещё 1–2 N/K-комбинации); эталон — fp32 в numpy/jnp; печатает и пишет JSON: `status` ∈ {`correctness-pass`,`correctness-fail`,`blocked`}, `shape`, `dtype`, `max_abs`, `rel`, `seconds`, `error_type`, `error`. Скрипт обязан различать compile-стадию и run-стадию.
3. Добавить `tools/tests/test_mma_smoke_contract.py` — CPU-контрактные проверки (CLI `--output`, структура JSON, корректный `blocked` без GPU), чтобы CI не требовал GPU.
4. В отчёте — точный root cause предыдущего падения (какой параметр/путь загрузки требуется), и что осталось проверить на GB10 (компиляция под sm_121, IR/PTX/SASS, детерминизм, несколько K-tile).

ОГРАНИЧЕНИЯ. Зона: `tools/mosaic/*`, `tools/tests/*`. НЕ трогать `net/`, `net/kernels/`, `net/config.json`, `tools/pretrain_run.py`. GPU-прогоны в задаче НЕ запускать (их выполняет архитектор на GB10) — но скрипт должен быть готов к запуску на стенде в этом окружении: `/home/roman/venv-axiom-0112/bin/python tools/mosaic/mma_smoke.py --output <path>`. **Не подменять проверку**: `hasattr`, импорт, CPU-прогон и `interpret=True` не являются доказательством GPU-компиляции; не выдумывать API (`mgpu_mma_` внутренний — не dependency).

ПРОВЕРКА (ПК): `/home/roman/venv-axiom/bin/python -m pytest -q tools/tests/test_mma_smoke_contract.py`.

РЕЗУЛЬТАТ. `.arch-handoff/result.json` + ФИНАЛЬНЫЙ git-commit. В assumptions — точный путь загрузки и параметры compiler_params; в open_questions — что проверить на стенде.

## Границы (scope)

Машинный контракт границ — `MANIFEST.json.scope` (хэш `scope_hash`); изменение вне границ ловится гейтом, а не обсуждается постфактум.

- **Можно писать:** `tools/mosaic/*`, `tools/tests/*`
- **Нельзя писать (сильнее allow):** `model/`, `ARCHITECTURE-SPINE.md`, `CONSTRAINTS.yaml`
- **Можно запускать:** `python3 -m pytest*`, `python3 tools/*`
- **Сеть:** сети нет (детерминированный узел)

## План отката

Откат: `git reset --hard da44935` (baseline — последний коммит до работы исполнителя; вся его работа приходит одним коммитом поверх).
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
