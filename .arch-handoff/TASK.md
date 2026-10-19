# Задача для кодового харнесса

Цикл-29 кампании MFU-55 (ADR-054 Am.51): диагностика `bwd_moe`-ноги phase-profile. Сейчас `sec_bwd_moe`=null, а `PhaseProfiler.attempt` глотает исключение — причина неизвестна. Сделать причину видимой и попробовать per-layer remat внутри `_phase_moe_stack`. Численные пути шага не меняются (только диагностическая нога).

ОКРУЖЕНИЕ: CPU-интерпретатор `/home/roman/venv-axiom-0112cpu-py312/bin/python` (jax 0.11.2, CPU); `NET_JAX_BACKEND=cpu JAX_PLATFORMS=cpu`; по одному файлу pytest с heartbeat; полный pytest не гонять.

1) `net/train_loop.py`, `PhaseProfiler`:
   - `self._errors: dict[str, str] = {}` (инициализация в `__init__`);
   - в `attempt(field, thunk, ...)`: `except Exception as exc: self._errors[field] = f"{type(exc).__name__}: {str(exc)[:200]}"; return None, None` (сохранить форму возврата);
   - в `measure(...)`: добавить в запись поле `phase_errors` = `dict(self._errors)` (после прогона; пустой dict, если ошибок нет); поле аддитивное, схему `pretrain-metrics/v1` не ломает; обновить константу/докстринг полей phase-profile.
2) `_phase_moe_stack(params, cfg, input_ids, *, remat=False)`: при `remat=True` каждый слой — `jax.checkpoint(lambda hh, b=block: hh + <mlp/moe>(b))(h)` (per-layer remat, по образцу модели); forward-нога (`sec_moe`) вызывает без remat (прежнее поведение); `bwd_moe`-нога — `jax.jit(jax.grad(lambda p, ids: jnp.sum(_phase_moe_stack(p, cfg, ids, remat=True))))` (внешний `jax.checkpoint` из Am.49 убрать, чтобы не дублировать; если он был — заменить).
3) Тесты `net/tests/test_53_bwd_moe_diag.py` (CPU):
   a. `attempt`-ошибка: нога с искусственным исключением → `phase_errors["<field>"]` содержит имя/текст, поле времени null, метрики не падают;
   b. `remat=True`: на малой конфигурации `_phase_moe_stack` с remat даёт те же значения, что без remat (побитово/в допуске), в jaxpr/lower присутствует remat-примитив на слой; `bwd_moe`-нога компилируется и возвращает градиенты той же формы, что params;
   c. forward-нога `sec_moe` не изменилась (побитово к ревизии до дельты);
   d. `phase_errors` присутствует в записи phase-profile (пустой dict при успехе).
4) Прогони по одному файлу: `test_53_bwd_moe_diag.py`, затем `test_52_phase_bwd_moe`, `test_46_mtp_kda_impl`.
5) Ограничения: без unsafe; `net/config.json` не трогать; численные пути шага — без изменений; стиль русских docstring.

JSON-контракт результата обязателен (status/assumptions/open_questions). Финальный git-коммит обязателен.

## Границы (scope)

Машинный контракт границ — `MANIFEST.json.scope` (хэш `scope_hash`); изменение вне границ ловится гейтом, а не обсуждается постфактум.

- **Можно писать:** `net/**`
- **Нельзя писать (сильнее allow):** `model/`, `ARCHITECTURE-SPINE.md`, `CONSTRAINTS.yaml`
- **Можно запускать:** `python3`, `python`, `pytest`, `/home/roman/venv-axiom-0112cpu-py312/bin/python*`
- **Сеть:** сети нет (детерминированный узел)

## План отката

Откат: git reset --hard <baseline_commit из .arch-handoff/MANIFEST.json> (диагностика; численные пути не меняются). Признак регрессии: phase-profile ломается, phase_errors не появляется, новые красные.

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
