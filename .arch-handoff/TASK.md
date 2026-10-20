# Задача для кодового харнесса

Цикл-36 кампании MFU-55 (ADR-054 Am.59): починить fused dispatch-путь MoE. Стендовый A/B `AXIOM_MOE_DISPATCH=fused` (вместе с `AXIOM_GEMM_FP8=1` и `AXIOM_MOE_BWD=fused`) упал на первом шаге: `sum requires ndarray or scalar arguments, got <class 'tuple'> at position 0`. Dispatch-кернел проверен на GPU (bitwise exact, Am.59), подозреваемые — `moe_dispatch.combine`/`scatter` (`_combine_run`/`_scatter_run`): `jnp.sum` на tuple/распаковка. CPU-тесты `test_55` 46 passed, но **модельный композит с fused-гейтом не проверялся** — это обязательный гейт цикла-36.

ОКРУЖЕНИЕ: CPU-интерпретатор `/home/roman/venv-axiom-0112cpu-py312/bin/python`; `NET_JAX_BACKEND=cpu JAX_PLATFORMS=cpu`; по одному файлу pytest. GPU-чек — архитектор (стенд); до него паритет не заявлять.

1) Диагностика/фикс `net/kernels/moe_dispatch.py`:
   - Найти `sum(...)`/возврат tuple в `_combine_run`/`_scatter_run` (и их обвязке `combine`/`scatter`); привести к ndarray: `de_grouped`, `dsel_p`, `dz` — по одному массиву; распаковку — явную.
   - Проверить формы/оси: `de_grouped[row] = du[token]·sel_p[token,slot]` (rows×latent); `dsel_p[token,slot] = <du[token], e_grouped[row]>` (tokens×k, fp32-аккум); `dz = Σ_k dz_grouped.reshape(tokens,k,latent)` по фиксированной оси 1.
   - Если fused combine/scatter починить не удаётся без риска — **допустимый компромисс**: в fused-пути оставить только dispatch-кернел, а combine/scatter вести jnp-`reference_combine`/`reference_scatter` (XLA), честно указав это в docstring/отчёте; гейт остаётся, семантика та же.
2) **Обязательный модельный композит-тест** `net/tests/test_56_moe_compose.py` (CPU, швы = jnp-reference и/или подмена kernel-швов):
   - `moe.apply` с `AXIOM_MOE_DISPATCH=fused` (и отдельно с `AXIOM_MOE_BWD=fused`): forward-выход и градиенты по параметрам/x == off-путь в scale-relative ≤2e-2 (для dispatch-only — побитово/≤1e-6, т.к. это перестановка);
   - прогон `compute_loss`/`jax.grad` на малой модели (l1/tiny-подобной) с fused-гейтом — **без исключений**, форма градиентов;
   - регресс: off-путь побитово прежний.
3) Прогони: `test_56_moe_compose.py`, `test_55_moe_dispatch.py`, `test_32_moe_grouped_topk`, `test_54_moe_bwd_fused`.
4) Ограничения: без unsafe; `net/config.json` не трогать; default `jax`; при off — побитово; порядок/канон forward сохранить; стиль русских docstring.
5) В отчёте явно: реальный GPU-чек и стендовый A/B — за архитектором; если выбран компромисс dispatch-only, назвать его и ожидаемый потолок (снимается только часть XLA-части).
JSON-контракт результата обязателен. Финальный git-коммит обязателен.

## Границы (scope)

Машинный контракт границ — `MANIFEST.json.scope` (хэш `scope_hash`); изменение вне границ ловится гейтом, а не обсуждается постфактум.

- **Можно писать:** `net/**`
- **Нельзя писать (сильнее allow):** `model/`, `ARCHITECTURE-SPINE.md`, `CONSTRAINTS.yaml`
- **Можно запускать:** `python3`, `python`, `pytest`, `/home/roman/venv-axiom-0112cpu-py312/bin/python*`
- **Сеть:** сети нет (детерминированный узел)

## План отката

Откат: git reset --hard <baseline_commit из .arch-handoff/MANIFEST.json> (гейт `AXIOM_MOE_DISPATCH` default `jax` — побитово прежний путь). Признак регрессии: off-путь перестал быть побитовым, модельный композит не совпадает, новые красные.

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
