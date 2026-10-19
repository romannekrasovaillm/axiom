# Задача для кодового харнесса

Цикл-26 кампании MFU-55 (ADR-054 Am.45): full-step FP8 GEMM для плотных проекций/MLP с STE — смена класса к 55 %. Мотив: FP8 CE in situ дал минус (−1.6 %, Am.44) — изолированные 85 TFLOPS не переносятся; проверяем FP8 на тяжёлых плотных GEMM шага (проекции attention, MLP), а не на CE. Гейт default 0 (побитово bf16/fp32-путь).

ОКРУЖЕНИЕ: CPU-интерпретатор `/home/roman/venv-axiom-0112cpu-py312/bin/python` (jax 0.11.2, CPU); `NET_JAX_BACKEND=cpu JAX_PLATFORMS=cpu`; по одному файлу pytest с heartbeat; полный pytest не гонять. fp8-матмул на CPU может не lowering-иться — шов `_fp8_matmul(a, b)` (на GPU — `jnp.matmul(..., preferred_element_type=jnp.float32)`), тесты подменяют деквантованным f32-матмулом.

1) `net/compute_dtype.py` — гейт `AXIOM_GEMM_FP8` (форма `mode()`/`persist_mode`): `"0"` (дефолт) | `"1"`; пусто/не задано = "0"; неизвестное → `GemmFp8Error` (fail-closed); читается на трейсе; явный аргумент сильнее env.
   - `def gemm_fp8(a, b)`: `jax.custom_vjp` с границей `(a, b) -> c`; forward: `s_a=max|a|/448`, `s_b=max|b|/448` (f32, данные-зависимые), `qa=clip(a/s_a,±448).astype(e4m3)`, `qb=...`, `c=_fp8_matmul(qa,qb)*(s_a*s_b)` (f32-аккум);
     backward: `ga = compute_dtype.gemm(g, b)` и `gb = compute_dtype.gemm(a.T-сторона, g)` — bf16-операнды/f32-аккум по **неквантованным** a/b (STE; производная квантизации = I; скейлы — константы).
   - `def gemm(a, b)`: при гейте on и `a,b` float — `gemm_fp8(a,b)`, иначе прежний bf16/fp32-путь (побитово); не менять семантику fp32/bf16 режимов.
2) Callsite-ы (только эти; CE НЕ трогать): `net/kda.py:_project` (все проекции q/k/v/gate/alpha/beta), `net/mla.py` — проекции латента/выхода (найти `compute_dtype.gemm`/`gemm_einsum` в проекционных путях), `net/mlp.py:apply` (gate/up/down, кроме SwiGLU-элементвайза). Если callsite использует `gemm_einsum`/`gemm_batched` — добавить аналогичный fp8-путь только там, где это чистый GEMM (не attention-скоры).
3) Тесты `net/tests/test_50_gemm_fp8.py` (CPU, шов `_fp8_matmul`):
   a. гейт: пусто/0/1/регистр/пробелы; неизвестное → `GemmFp8Error`; явный аргумент сильнее env.
   b. off: `compute_dtype.gemm` и все callsite-ы побитово равны ревизии до дельты.
   c. on: forward `c` == bf16-эталон в scale-relative (документировать фактический max|d|/scale; ожидание ≤2e-2), градиенты a/b == bf16-эталон в scale-relative ≤2e-2 (STE).
   d. композит: маленькая модель (KDA/MLA/MLP) при gate=on: лосс == off в допуске; 2-шаговый прогон без nan.
   e. скейлы: нулевой/вырожденный тензор (max=0) не даёт nan; скейлы не текут в градиенты.
4) Прогони по одному файлу: `test_50_gemm_fp8.py`, затем регресс: `test_02_kda_parity`, `test_31_mla_dense_compute_dtype`, `test_29_compute_dtype` (унаследованный красный — зафиксировать), `test_46_mtp_kda_impl`, `test_47_ce_bwd`.
5) Ограничения: без unsafe; `net/config.json` не трогать; gate default 0; при off — побитово (блокирующий критерий); CE не трогать; стиль русских docstring.

JSON-контракт результата обязателен (status/assumptions/open_questions). Финальный git-коммит обязателен.

## Границы (scope)

Машинный контракт границ — `MANIFEST.json.scope` (хэш `scope_hash`); изменение вне границ ловится гейтом, а не обсуждается постфактум.

- **Можно писать:** `net/**`
- **Нельзя писать (сильнее allow):** `model/`, `ARCHITECTURE-SPINE.md`, `CONSTRAINTS.yaml`
- **Можно запускать:** `python3`, `python`, `pytest`, `/home/roman/venv-axiom-0112cpu-py312/bin/python*`
- **Сеть:** сети нет (детерминированный узел)

## План отката

Откат: git reset --hard <baseline_commit из .arch-handoff/MANIFEST.json> (гейт default 0 — побитово прежний путь). Признак регрессии: off-путь перестал быть побитовым, паритет on вне допуска, nan на 10 шагах или новые красные.

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
