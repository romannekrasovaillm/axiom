# Задача для кодового харнесса

Цикл-35 кампании MFU-55 (ADR-054 Am.58): fused dispatch/scatter MoE. Измерено (Am.56/57): `sec_bwd_moe` 7.144 с = fused-кернел 3.55 с + **XLA-часть ~3.6 с** (группировка/gather-scatter/routing-грады + per-layer remat-recompute; память ни при чём). Замена XLA-последовательности одним Mosaic-кернелом на чанк токенов; гейт default `jax` (побитово).

ОКРУЖЕНИЕ: CPU-интерпретатор `/home/roman/venv-axiom-0112cpu-py312/bin/python`; `NET_JAX_BACKEND=cpu JAX_PLATFORMS=cpu`; по одному файлу pytest. GPU-only кернел; на CPU fused отказывает типизированно (тесты подменяют шов). Реальный GPU-чек — за архитектором (стенд); до него паритет не заявлять.

1) `net/moe.py` — гейт `AXIOM_MOE_DISPATCH` (форма `counting_sort_mode`/`persist_mode`): `"jax"` (дефолт — текущая XLA-последовательность, побитово) | `"fused"`; пусто/не задано = "jax"; неизвестное → `MoeDispatchError` (fail-closed); читается на трейсе.
2) Новый модуль `net/kernels/moe_dispatch.py` (Mosaic GPU, контракт pallas-gb10-kernel: `CASES/make_inputs/reference/kernel/cost/TUNE_SPACE/valid_config/device_available`):
   - **dispatch**: входы `z (tokens,768)` (латентные токены), `topk_idx (tokens,k)` (k=2), `sel_p (tokens,k)`; выход `z_grouped (rows,768)` (rows=tokens*k, в **каноническом порядке**: стабильно по эксперту, внутри эксперта — по порядку появления токена, как forward), `group_sizes (12,)`, и (для backward) `assign_rows`/`row_of_token` (позиция каждой (token,k) пары в группировке);
   - **combine/scatter**: входы `de_grouped (rows,768)`, `row_of_token`, `sel_p`; выходы `dz (tokens,768)` = Σ_k sel_p·de (порядок суммирования k фиксирован — детерминизм), `dsel_p (tokens,k)` = <de, e_k>;
   - всё per-expert (grid=(12,)) + префиксные суммы `group_sizes`; SMEM ≤99 КиБ; bf16/fp32-аккум; `rows % bm == 0` fail-closed (как в moe_experts_bwd); пустая группа — пустой диапазон без делений;
   - jnp-`reference_dispatch`/`reference_combine` для CPU-паритета.
3) `net/moe.py`: при gate="fused" использовать fused dispatch/combine в forward и в backward-пути (`de_grouped`/`dz`/`dsel_p`), сохраняя порядок/канон; при off — прежний код дословно.
4) Тесты `net/tests/test_55_moe_dispatch.py` (CPU, шов):
   a. гейт: пусто/jax/fused/регистр/пробелы; неизвестное → `MoeDispatchError`; явный аргумент сильнее env;
   b. off: output/градиенты побитово равны ревизии до дельты;
   c. on (шов = jnp-reference): `z_grouped` совпадает с текущим (порядок!), `group_sizes`; `dz`/`dsel_p` == autodiff off в scale-relative ≤2e-2; детерминизм (два запуска);
   d. пустые/неполные группы; k=1 и k=2; `valid_config`/`device_available=False` — типизированный отказ;
   e. lower_check-адаптер `net/tests/moe_dispatch_lower.py`.
5) Прогони: `test_55_moe_dispatch.py`, затем `test_32_moe_grouped_topk`, `test_54_moe_bwd_fused`, `test_46_mtp_kda_impl`.
6) Ограничения: без unsafe; `net/config.json` не трогать; gate default `jax`; при off — побитово (блокирующий критерий); порядок групп/токенов — канон forward; стиль русских docstring.
JSON-контракт результата обязателен. Финальный git-коммит обязателен.

## Границы (scope)

Машинный контракт границ — `MANIFEST.json.scope` (хэш `scope_hash`); изменение вне границ ловится гейтом, а не обсуждается постфактум.

- **Можно писать:** `net/**`
- **Нельзя писать (сильнее allow):** `model/`, `ARCHITECTURE-SPINE.md`, `CONSTRAINTS.yaml`
- **Можно запускать:** `python3`, `python`, `pytest`, `/home/roman/venv-axiom-0112cpu-py312/bin/python*`
- **Сеть:** сети нет (детерминированный узел)

## План отката

Откат: git reset --hard <baseline_commit из .arch-handoff/MANIFEST.json> (гейт default `jax` — побитово прежний путь). Признак регрессии: off-путь перестал быть побитовым, порядок/паритет dispatch вне допуска, новые красные.

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
