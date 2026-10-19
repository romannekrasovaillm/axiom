# Задача для кодового харнесса

Цикл-23 кампании MFU-55 (ADR-054 Am.35/36): fused backward для chunked-CE (lm_head/softmax-CE). Пофазовый профиль: `sec_bwd_ce` 2.267 с при ~14 % пика, `sec_ce` (fwd) 0.886 с — CE/lm_head неэффективен; оба CE-головы (NTP и MTP) идут через `_chunked_cross_entropy` (net/model.py), сейчас backward — autodiff через per-chunk `jax.checkpoint`. Задача — явный `jax.custom_vjp` с fused backward, гейт default off (побитово прежний путь), без изменения forward-чисел.

ОКРУЖЕНИЕ: CPU-интерпретатор `/home/roman/venv-axiom-0112cpu-py312/bin/python` (jax 0.11.2, CPU); `NET_JAX_BACKEND=cpu JAX_PLATFORMS=cpu`; по одному файлу pytest с heartbeat; полный pytest не гонять.

1) `net/model.py` — гейт `AXIOM_CE_BWD` (форма `ce_bf16_mode`/`persist_mode`): значения `"jax"` (дефолт — текущий autodiff-путь, побитово) и `"fused"`; пусто/не задано = `"jax"`; неизвестное → `CeBwdError` (fail-closed); читается на трейсе; явный аргумент сильнее env.
2) `_chunked_cross_entropy` — при gate="fused" обернуть ВЫЧИСЛЕНИЕ ЛОССА в `jax.custom_vjp` с границей `(features, embedding) -> loss` (targets/loss_mask — недифференцируемые статические/целочисленные аргументы; для них градиент не нужен, но функция должна работать и с `loss_mask`):
   - forward-правило: дословно текущий chunked-цикл (глобальная сумма NLL / глобальный счётчик; bf16-ветка `ce_bf16_mode` сохраняется), числа те же;
   - backward-правило (dL — котангент скаляра): для каждого чанка пересчитать `logits = features_chunk @ embedding.T` (bf16-операнды, fp32-аккум — как в forward), `p = softmax(logits)`, `dlogits = (p - onehot(target)) * (dL / global_count)`; `dembedding += dlogits.T @ features_chunk`; `dfeatures_chunk = dlogits @ embedding`; чанки аккумулируются. Полные `(chunk, vocab)` для backward не сохранять (пересчёт; никакого `jax.checkpoint` внутри fused-ветки), живой тензор ≤ одного чанка.
   - маска `loss_mask`: исключённые строки не дают вклада ни в числитель, ни в знаменатель (как в forward).
3) Численная политика: fused-backward сверяется с autodiff того же bf16-пути; допуск — scale-relative ≤2e-2 (кампанийный допуск CE bf16) с документированием фактического max|d|/scale; forward при gate="fused" обязан совпадать с gate="jax" (тот же цикл).
4) Тесты `net/tests/test_47_ce_bwd.py` (CPU):
   a. гейт: пусто/jax/fused/регистр/пробелы; неизвестное → `CeBwdError` до трейса; явный аргумент сильнее env.
   b. off: `_chunked_cross_entropy` (и градиенты по features/embedding) побитово равны ревизии до дельты (прямой вызов текущей реализации).
   c. on: forward-лосс побитово равен off; градиенты features/embedding == autodiff off в scale-relative ≤2e-2; несколько сидов/форм, включая хвостовой неполный чанк и `loss_mask` (если используется).
   d. память: живой тензор чанка (форма) ≤ (chunk_tokens, vocab) — пин на jaxpr/`lower().as_text()` по отсутствию полного (B*T, vocab) буфера.
   e. Композит: маленькая модель с `compute_loss` (NTP+MTP) при gate=fused: лосс == off (побитово), градиенты параметров в допуске.
5) Прогони по одному файлу: `test_47_ce_bwd.py`, затем регресс: `test_29_compute_dtype` (известный унаследованный красный — зафиксировать), `test_30`, `test_36`, `test_39`, `test_46_mtp_kda_impl`.
6) Ограничения: без unsafe; `net/config.json` не трогать; gate default `"jax"`; при off — побитово прежние числа и граф (блокирующий критерий); стиль русских docstring.

JSON-контракт результата обязателен (status/assumptions/open_questions). Финальный git-коммит обязателен.

## Границы (scope)

Машинный контракт границ — `MANIFEST.json.scope` (хэш `scope_hash`); изменение вне границ ловится гейтом, а не обсуждается постфактум.

- **Можно писать:** `net/**`
- **Нельзя писать (сильнее allow):** `model/`, `ARCHITECTURE-SPINE.md`, `CONSTRAINTS.yaml`
- **Можно запускать:** `python3`, `python`, `pytest`, `/home/roman/venv-axiom-0112cpu-py312/bin/python*`
- **Сеть:** сети нет (детерминированный узел)

## План отката

Откат: git reset --hard <baseline_commit из .arch-handoff/MANIFEST.json> (гейт default `jax` — побитово прежний путь). Признак регрессии: off-путь перестал быть побитовым или новые красные.

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
