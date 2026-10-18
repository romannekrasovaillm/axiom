# Задача для кодового харнесса

Цикл-21 кампании MFU-55 (ADR-054 Am.33, подготовка G2): тонкая провенанс-карта. Сейчас `probe_ce` стоит на всём `compute_loss` (net/train_loop.py:3342) и через `transpose(jvp(probe_ce))` захватывает backward (37.5% шага), из-за чего CE-голова/lm_head не отделимы. Задача — заменить грубую метку тонкими, гейт `AXIOM_TRACE_MARKERS` (default off, побитово прежний граф), без изменения численных операций.

ОКРУЖЕНИЕ: CPU-интерпретатор `/home/roman/venv-axiom-0112cpu-py312/bin/python` (jax 0.11.2, CPU); `NET_JAX_BACKEND=cpu JAX_PLATFORMS=cpu`; по одному файлу pytest с heartbeat; полный pytest не гонять.

1) `net/train_loop.py`: убрать внешнюю метку `provenance.mark("ce", ...)` вокруг `model.compute_loss` — вернуть прямой вызов `model.compute_loss(params, cfg, batch, chunk_size=train_config.chunk_size)` (комментарий заменить: тонкие метки живут в `net/model.compute_loss`; внешняя метка захватывала backward). Импорт `provenance` оставить для метки оптимизатора (п.3).
2) `net/model.py`, `compute_loss` — тонкие метки (только границы вызовов):
   - `forward(...)` в обеих ветках (`ce_tokens>0` и else) → `provenance.mark("forward", lambda: forward(...))` (вернуть тот же результат/кортеж);
   - NTP-CE: `_chunked_cross_entropy(hidden[:, :-1], input_ids[:, 1:], params.embedding, ce_tokens)` и `_cross_entropy(logits[:, :-1], input_ids[:, 1:])` → `provenance.mark("ntp_ce", lambda: ...)`;
   - `mtp_loss(params, cfg, hidden, input_ids, chunk_size, ce_chunk_tokens=ce_tokens)` → `provenance.mark("mtp", lambda: ...)`;
   - финальную сумму (`ntp + cfg.mtp_loss_weight*aux + cfg.qb_weight*qb`) не оборачивать.
   - При off — те же объекты/формы/числа; никаких новых jit/checkpoint границ.
3) `net/train_loop.py`: пометить шаг оптимизатора в ОБОИХ call-site (`master, state = step_fn(...)` и `master, state, params = step_fn(...)`) как `provenance.mark("opt", lambda: step_fn(...))` (распаковать кортеж). Это метка границы; если jit-граница не пронесёт имя внутрь модуля — в отчёте так и сказать (атрибуция оптимизатора дополнительно пойдёт по модулю `jit_step` в дампе).
4) Тесты `net/tests/test_45_trace_markers.py` — обновить/добавить:
   a. при `AXIOM_TRACE_MARKERS=1` lowered HLO маленькой модели/`compute_loss` содержит `probe_forward`, `probe_ntp_ce`, `probe_mtp`; `probe_ce` больше НЕ появляется как внешний loss-scope (если старый тест его ждал — поправить);
   b. gate off — побитовый паритет `compute_loss`/градиентов (существующий тест расширить на новые call-site);
   c. метка `probe_opt` присутствует в lowered HLO шага оптимизатора (если jit-граница не даёт — задокументировать и проверить хотя бы наличие вызова `mark` с `opt`);
   d. существующие тесты gate off/on и `MtpKdaImplError` не ломать.
5) Прогони по одному файлу: `test_45_trace_markers.py`, затем регресс: `test_24_grad_checkpoint`, `test_31_remat_policy`, `test_37_per_layer_block`, `test_02_kda_parity`, `test_46_mtp_kda_impl`.
6) Ограничения: без unsafe; `net/config.json` не трогать; гейт default off; при off — побитово прежние числа (блокирующий критерий); стиль русских docstring.

JSON-контракт результата обязателен (status/assumptions/open_questions). Финальный git-коммит обязателен.

## Границы (scope)

Машинный контракт границ — `MANIFEST.json.scope` (хэш `scope_hash`); изменение вне границ ловится гейтом, а не обсуждается постфактум.

- **Можно писать:** `net/**`
- **Нельзя писать (сильнее allow):** `model/`, `ARCHITECTURE-SPINE.md`, `CONSTRAINTS.yaml`
- **Можно запускать:** `python3`, `python`, `pytest`, `/home/roman/venv-axiom-0112cpu-py312/bin/python*`
- **Сеть:** сети нет (детерминированный узел)

## План отката

Откат: git reset --hard <baseline_commit из .arch-handoff/MANIFEST.json> (метки диагностические; при off численные пути и граф побитово прежние). Признак регрессии: паритет off/on не побитовый или новые красные.

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
