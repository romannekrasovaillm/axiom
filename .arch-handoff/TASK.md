# Задача для кодового харнесса

Диагностическая инструментация провенанса (ADR-054 Am.31, цикл-19): gated name scopes, чтобы HLO/nsys-трейс клетки B=4 показал, какому компоненту принадлежат command_buffer_1229/985 (хвост 9.75 с GPU, 512 графов, 57.9k ядер). Гейт по умолчанию ВЫКЛЮЧЕН и не меняет численные операции.

ОКРУЖЕНИЕ: CPU-интерпретатор — `/home/roman/venv-axiom-0112cpu-py312/bin/python` (jax 0.11.2, CPU). Все прогоны — с `NET_JAX_BACKEND=cpu JAX_PLATFORMS=cpu`; по одному файлу pytest с heartbeat (не дольше 30 с тишины), полный pytest не гонять.

1) Новый модуль `net/provenance.py` (стиль — как `net/mla.py`, persist_mode):
   - `TRACE_MARKERS_ENV = "AXIOM_TRACE_MARKERS"`; значения только "0"|"1"; пустая/незаданная = "0"; неизвестное → `TraceMarkersError` (fail-closed, без тихого отката).
   - `def markers_mode(value: str | None = None) -> bool` — читает env на трейсе (явный аргумент сильнее env), нормализует регистр/пробелы.
   - `def mark(name: str, fn, *args, **kwargs)`: при off — `return fn(*args, **kwargs)` (прямой вызов, граф побитово прежний, никакого named_call); при on — `return jax.named_call(fn, name=f"probe_{name}")(*args, **kwargs)`.
   - Русский docstring: зачем (name scope попадает в op_name HLO как `probe_<name>`), почему off не меняет граф, почему fail-closed.
2) `net/model.py` — обернуть ТОЛЬКО границы вызовов (не тела, не порядок операций):
   - `_block_delta`: KDA (`kda_mod.apply_kda` внутри `jax.vmap`) → `provenance.mark("kda", lambda xb: kda_mod.apply_kda(block.attn, cfg, xb, chunk_size), xb)`; MLA `mla_mod.apply_with_pool` → `provenance.mark("mla", lambda: mla_mod.apply_with_pool(block.attn, cfg, hn, pool=pool, mode=mode))` (вернуть кортеж); `dense_standard` ветку → `provenance.mark("mla_dense", ...)`; MoE `moe_mod.apply` → `provenance.mark("moe", ...)` в ОБОИХ call-site (want_qb=True/False); dense MLP `mlp_mod.apply` → `provenance.mark("mlp", ...)`.
   - `_layer_transition`: `attnres_mod.apply_layer` → `provenance.mark("attnres", lambda: attnres_mod.apply_layer(attnres_w, sources))`.
   - При off — те же объекты/формы/числа; никаких новых jit/checkpoint границ.
3) `net/train_loop.py` — обернуть внешний вызов CE/loss-вычисления одним `provenance.mark("ce", lambda: ...)` в самом внешнем месте (не внутри чанкового цикла и не в горячем внутреннем скане); в отчёте указать файл:строка и почему выбрано это место.
4) Тесты `net/tests/test_45_trace_markers.py` (CPU):
   a. `markers_mode`: пусто/0/1/неизвестное/регистр/пробелы; явный аргумент сильнее env.
   b. off: `mark` вызывает fn напрямую (счётчик вызовов), без named_call; результат равен прямому вызову.
   c. on: `jax.jit(lambda x: mark("kda", lambda: jnp.sum(x * 2))).lower(...).as_text()` содержит `probe_kda`; off — не содержит.
   d. Побитовый паритет на маленькой модели (tiny/l1-подобный конфиг): выходы loss/grad при off и on равны `jnp.array_equal`.
   e. Неизвестное значение env → `TraceMarkersError` до компиляции/исполнения.
5) Прогони по одному файлу: `test_45_trace_markers.py`, затем регресс затронутого: test_24_grad_checkpoint, test_31_remat_policy, test_37_per_layer_block, test_39, test_02_kda_parity, test_kda_chunked_cc.
6) Ограничения: без unsafe; существующие тесты не ломать; `net/config.json` не трогать; гейт default off; никаких изменений численных операций при off (блокирующий критерий).

JSON-контракт результата обязателен (status/assumptions/open_questions). Финальный git-коммит обязателен.

## Границы (scope)

Машинный контракт границ — `MANIFEST.json.scope` (хэш `scope_hash`); изменение вне границ ловится гейтом, а не обсуждается постфактум.

- **Можно писать:** `net/**`
- **Нельзя писать (сильнее allow):** `model/`, `ARCHITECTURE-SPINE.md`, `CONSTRAINTS.yaml`
- **Можно запускать:** `python3`, `python`, `pytest`, `/home/roman/venv-axiom-0112cpu-py312/bin/python*`
- **Сеть:** сети нет (детерминированный узел)

## План отката

Откат: git reset --hard <baseline_commit из .arch-handoff/MANIFEST.json> (гейт и модуль диагностические; численные пути при off не менялись). Признак регрессии: паритет off/on не побитовый или новые красные в регрессе.

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
