# Задача для кодового харнесса

ЗАДАЧА (первый собственный Pallas-кернел кампании MFU-55: батчевое треугольное решение для KDA).

КОНТЕКСТ. Профиль стационарной фазы (nsys, GB10) показал: батчевое `jax.lax.linalg.triangular_solve` XLA исполняет как **115 200 отдельных ядер** `batch_trsm_left_kernel<...,64,4,...>` + **230 400 служебных `MakeBatchPointers`** за 120 с, занятость GPU 3.5%. Алгебра: внутри чанка KDA решается `(I+L)·X = B`, где `I+L` — единичная нижняя треугольная `(H, C, C)`, правая часть `B` — `(H, C, dk+dv)`; H=12, C=64, dk=dv=128, hidden=1536 (`net/config.json`, `net/kda.py:_ut_solve_pair`). Кернел оправдан по канону скилла `pallas-gb10-kernel`: нестандартная операция, которую XLA не собирает в крупное ядро.

ВАЖНО ПРО ОКРУЖЕНИЕ (снято probe_gb10.py, файл `~/axiom-run/gb10_caps_full.json` на стенде): jax/jaxlib **0.10.2**, CUDA 13.0, GB10 sm_121. Путь **Mosaic GPU MMA недоступен** (`mgpu_mma_bf16`/`fp8` FAIL: `Layout.MMA_ACC` отсутствует в этой версии JAX), зато **`triton_dot` и `triton_elementwise` — PASS**. Поэтому первый кернел пишется на **Pallas-Triton** (`pl.pallas_call` с `pltpu`/`pl` Triton-путём), НЕ на Mosaic GPU. Потолки машины (для roofline): copy 131.2 ГБ/с, fp32 43.0 TFLOPS, bf16 79.6 TFLOPS.

ЧТО СДЕЛАТЬ (контракт модуля — обязателен, см. скилл `pallas-gb10-kernel`):
1. Создать `net/kernels/kda_ut_solve.py` — модуль кернела со строго этим контрактом:
   ```python
   CASES = [dict(H=12, C=64, dk=128, dv=128, dtype="float32"), dict(... bf16 ...)]
   def make_inputs(key, **case) -> tuple            # a=(H,C,C) единичная нижняя треугольная, b=(H,C,dk+dv)
   def kernel(a, b, **params) -> jax.Array          # решение (I+L) X = B, батчево по H
   def reference(a, b) -> jax.Array                 # эталон: jnp.linalg.solve в f32 (или явная подстановка)
   def baseline(a, b) -> jax.Array                  # jax.lax.linalg.triangular_solve — то, что заменяем
   def cost(**case) -> dict(flops=..., bytes=...)   # минимально необходимые флопы/байты
   TUNE_SPACE = dict(...)                           # осмысленные десятки конфигураций
   def valid_config(case, params) -> bool           # отсев по SMEM/делимости до компиляции
   ```
   Формы — ТОЛЬКО реальные из задачи (H=12, C=64, dk=dv=128); не подменять квадратными 4096³.
2. Провести статический lowering: `python3 ~/axiom-run/skills/pallas-gb10-kernel/scripts/lower_check.py net/kernels/kda_ut_solve.py` (на ПК это работает без GPU) — исправлять, пока все CASES не OK.
3. Добавить тест `net/tests/test_kernel_kda_ut_solve.py`: сверка `kernel` с `reference` на CPU (JAX_PLATFORMS=cpu) в пределах разумного atol (обосновать числом), проверка формы выхода, отсев невалидных конфигураций `valid_config`.
4. В отчёте — статическая оценка: сколько ядер/запусков порождает XLA-версия против кернела (по форме и числу блоков), как выбиралось `TUNE_SPACE`.

ОГРАНИЧЕНИЯ.
- Зона: `net/kernels/*` (новый каталог) и `net/tests/*`. НЕ трогать `net/kda.py`, `net/config.json`, другие файлы: интеграция кернела в KDA — отдельная дельта после GPU-проверки.
- GPU-прогоны НЕ запускать: `check_kernel.py` и `bench.py`/`autotune.py` требуют GB10 — их выполняет архитектор на стенде (там же лежат капы). На ПК — только `lower_check.py` и CPU-тесты.
- Никаких «оптимизаций для красоты»: один кернел, одна задача.
- Если на jax 0.10.2 Triton-путь Pallas имеет иные ограничения, чем описано в скилле (API 0.11) — зафиксируй это в open_questions с конкретной ошибкой, не подменяя решение.

ПРОВЕРКА (на ПК, этим интерпретатором): `/home/roman/venv-axiom/bin/python -m pytest -q net/tests/test_kernel_kda_ut_solve.py` и `lower_check.py` по всем CASES.

РЕЗУЛЬТАТ. `.arch-handoff/result.json` + ФИНАЛЬНЫЙ git-commit. В assumptions — параметры по умолчанию и обоснование atol; в open_questions — что должен измерить архитектор на GB10 (время против XLA-`triangular_solve`, ГБ/с и % от потолка 131.2 ГБ/с, лучшие параметры тюна).

## Границы (scope)

Машинный контракт границ — `MANIFEST.json.scope` (хэш `scope_hash`); изменение вне границ ловится гейтом, а не обсуждается постфактум.

- **Можно писать:** `net/kernels/*`, `net/tests/*`
- **Нельзя писать (сильнее allow):** `model/`, `ARCHITECTURE-SPINE.md`, `CONSTRAINTS.yaml`
- **Можно запускать:** `python3 -m pytest*`, `python3 net/kernels/*`
- **Сеть:** сети нет (детерминированный узел)

## План отката

Откат: `git reset --hard ad7b400` (baseline — последний коммит до работы исполнителя; вся его работа приходит одним коммитом поверх).
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
