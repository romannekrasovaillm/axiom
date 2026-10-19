# Задача для кодового харнесса

Цикл-28 кампании MFU-55 (ADR-054 Am.49): (1) memory-lean `bwd_moe`-нога phase-profile (checkpoint) — сейчас `sec_bwd_moe`=null из-за OOM при B=4; (2) fail-closed валидация дормантных FP8-гейтов (`AXIOM_FP8_ROW_SCALE` не проверяется, пока `AXIOM_GEMM_FP8_BWD=0`). Численные пути не меняются; фазовая нога — диагностика.

ОКРУЖЕНИЕ: CPU-интерпретатор `/home/roman/venv-axiom-0112cpu-py312/bin/python` (jax 0.11.2, CPU); `NET_JAX_BACKEND=cpu JAX_PLATFORMS=cpu`; по одному файлу pytest с heartbeat; полный pytest не гонять.

1) `net/train_loop.py`, `PhaseProfiler.__init__`: нога `bwd_moe` — обернуть `_phase_moe_stack` в `jax.checkpoint` внутри grad-лямбды:
   `jax.jit(jax.grad(lambda p, ids: jnp.sum(jax.checkpoint(lambda pp: _phase_moe_stack(pp, cfg, ids))(p))))`
   (ремат слоя: память ≤ одного слоя вместо всех MoE-слоёв). Комментарий: время ноги включает пересчёт и подаётся как **ориентир**, не партиция (согласовано с `bwd_reconciliation`). Остальные ноги не трогать.
2) `net/compute_dtype.py`: все FP8-гейты валидируются на каждом трейсе независимо от прочих: `AXIOM_FP8_ROW_SCALE` — вызывать парсер/валидатор в `gemm_fp8`/`gemm` даже когда `AXIOM_GEMM_FP8_BWD=0` и даже когда `AXIOM_GEMM_FP8=0`? Достаточно: при любом вызове `gemm_fp8`/`gemm` и при сборке fp8-пути опечатка ROW-гейта → ошибка (fail-closed). Не менять поведение корректных значений.
3) Тесты `net/tests/test_52_phase_bwd_moe.py` (CPU):
   a. `PhaseProfiler` с малой конфигурацией: нога `bwd_moe` компилируется и исполняется; результат — pytree градиентов той же формы, что params (не None); проверка, что `jax.checkpoint` в графе (по lower/jaxpr — remat-примитив присутствует).
   b. Гейты: `AXIOM_FP8_ROW_SCALE=мусор` → ошибка и при `AXIOM_GEMM_FP8_BWD=0`, и при `AXIOM_GEMM_FP8=0` (если путь gemm вызывается); корректные 0/1 — без ошибок; регистр/пробелы.
   c. Регресс: `test_50_gemm_fp8`, `test_51_gemm_fp8_bwd` (не ломать семантику).
4) Прогони по одному файлу: `test_52_phase_bwd_moe.py`, затем `test_50_gemm_fp8`, `test_51_gemm_fp8_bwd`, `test_46_mtp_kda_impl`.
5) Ограничения: без unsafe; `net/config.json` не трогать; численные пути — без изменений; стиль русских docstring.

JSON-контракт результата обязателен (status/assumptions/open_questions). Финальный git-коммит обязателен.

## Границы (scope)

Машинный контракт границ — `MANIFEST.json.scope` (хэш `scope_hash`); изменение вне границ ловится гейтом, а не обсуждается постфактум.

- **Можно писать:** `net/**`
- **Нельзя писать (сильнее allow):** `model/`, `ARCHITECTURE-SPINE.md`, `CONSTRAINTS.yaml`
- **Можно запускать:** `python3`, `python`, `pytest`, `/home/roman/venv-axiom-0112cpu-py312/bin/python*`
- **Сеть:** сети нет (детерминированный узел)

## План отката

Откат: git reset --hard <baseline_commit из .arch-handoff/MANIFEST.json> (диагностика/валидация; численные пути не меняются). Признак регрессии: phase-profile ломается, гейты не валидируются, новые красные.

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
