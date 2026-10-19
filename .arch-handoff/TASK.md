# Задача для кодового харнесса

Цикл-27 кампании MFU-55 (ADR-054 Am.47): FP8 в **backward** GEMM-helper (градиентные матмулы). Forward-fp8 уже дал +8.8 % (1200.9 ток/с, Am.46); backward шага 22.7 с — крупнейший оставшийся рычаг того же класса. Под-гейт default 0 (bf16-backward — побитово текущему пути).

ОКРУЖЕНИЕ: CPU-интерпретатор `/home/roman/venv-axiom-0112cpu-py312/bin/python` (jax 0.11.2, CPU); `NET_JAX_BACKEND=cpu JAX_PLATFORMS=cpu`; по одному файлу pytest с heartbeat; полный pytest не гонять. fp8-матмул на CPU может не lowering-иться — шов `_fp8_matmul(a,b)` (на GPU — `jnp.matmul(..., preferred_element_type=jnp.float32)`), тесты подменяют деквантованным f32-матмулом.

1) `net/compute_dtype.py` — под-гейты (форма `mode()`/`persist_mode`, fail-closed, читаются на трейсе, явный аргумент сильнее env):
   - `AXIOM_GEMM_FP8_BWD=0|1` (default 0: текущий backward в bf16 через `compute_dtype.gemm`, побитово); `1`: градиентные матмулы `ga = g @ bᵀ`, `gb = aᵀ @ g` в fp8 (e4m3, per-tensor скейлы, f32-аккум) со STE;
   - `AXIOM_FP8_ROW_SCALE=0|1` (default 0: per-tensor; `1`: per-row для `g`/`a` — `(q(a/s_row)@qb)*s_row[:,None]`, аккуратно с транспонированием `gb`; per-tensor для второго операнда). Per-block (128×128) НЕ делать — нужен отдельный кернел (отложено).
2) `gemm_fp8` (в `compute_dtype.py`): backward-правило расширить; при под-гейте off — прежний bf16-путь дословно; при on — fp8-матмулы с теми же скейлами/STE; `dtype` котангентов — как у прималов; нулевой/вырожденный тензор (max=0) не даёт nan (скейл-эпсилон).
3) Тесты `net/tests/test_51_gemm_fp8_bwd.py` (CPU, шов `_fp8_matmul`):
   a. под-гейты: пусто/0/1/регистр/пробелы; неизвестное → ошибка; явный аргумент сильнее env.
   b. off (AXIOM_GEMM_FP8=1, BWD=0): градиенты побитово равны ревизии до дельты (bf16-backward).
   c. on: forward == прежнему fp8-forward (побитово/в допуске); градиенты a/b == bf16-эталону в scale-relative (начать с 5e-2; фактический max|d|/scale задокументировать); per-tensor и per-row варианты; нулевой тензор.
   d. композит: маленькая модель (KDA/MLA/MLP) при FP8+BWD=on: 2-шаговый прогон без nan, лосс в допуске 5e-2 к off-ноге.
   e. STE-границы: скейлы не текут в градиенты; нет нулевых градиентов.
4) Прогони по одному файлу: `test_51_gemm_fp8_bwd.py`, затем регресс: `test_50_gemm_fp8`, `test_02_kda_parity`, `test_31_mla_dense_compute_dtype`, `test_46_mtp_kda_impl`, `test_47_ce_bwd`.
5) Ограничения: без unsafe; `net/config.json` не трогать; под-гейт default 0; при off — побитово (блокирующий критерий); CE/MoE не трогать; стиль русских docstring.

JSON-контракт результата обязателен (status/assumptions/open_questions). Финальный git-коммит обязателен.

## Границы (scope)

Машинный контракт границ — `MANIFEST.json.scope` (хэш `scope_hash`); изменение вне границ ловится гейтом, а не обсуждается постфактум.

- **Можно писать:** `net/**`
- **Нельзя писать (сильнее allow):** `model/`, `ARCHITECTURE-SPINE.md`, `CONSTRAINTS.yaml`
- **Можно запускать:** `python3`, `python`, `pytest`, `/home/roman/venv-axiom-0112cpu-py312/bin/python*`
- **Сеть:** сети нет (детерминированный узел)

## План отката

Откат: git reset --hard <baseline_commit из .arch-handoff/MANIFEST.json> (под-гейт default 0 — bf16-backward, побитово текущему FP8-forward-only пути). Признак регрессии: off-путь перестал быть побитовым, паритет on вне допуска, nan/дрейф лосса на 10 шагах или новые красные.

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
