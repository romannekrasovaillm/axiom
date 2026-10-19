# Задача для кодового харнесса

Цикл-25 кампании MFU-55 (ADR-054 Am.43): FP8 CE/lm_head через custom_vjp + STE — FP8-нога к 55 %. Forward-логиты в fp8 e4m3 (численно годен: loss rel 5e-5), backward в bf16/fp32 по НЕквантованным операндам (наивный fp8-autodiff в JAX ломает градиенты — каст fp32->fp8 не даёт корректного градиента; измерено dFeat rel=1.0). Гейт default 0 (побитово прежний bf16-путь).

ОКРУЖЕНИЕ: CPU-интерпретатор `/home/roman/venv-axiom-0112cpu-py312/bin/python` (jax 0.11.2, CPU); `NET_JAX_BACKEND=cpu JAX_PLATFORMS=cpu`; по одному файлу pytest с heartbeat; полный pytest не гонять. ВАЖНО: fp8-матмул на CPU-бэкенде может не lowering-иться — предусмотреть шов `_fp8_matmul(a, b)` (на GPU — `jnp.matmul(..., preferred_element_type=jnp.float32)`), который тесты подменяют на деквантованный f32-матмул (`a.astype(f32) @ b.astype(f32)`); численный эффект квантизации сохраняется.

1) `net/model.py` — гейт `AXIOM_CE_FP8` (форма `ce_bf16_mode`/`persist_mode`): значения `"0"` (дефолт) и `"1"`; пусто/не задано = "0"; неизвестное → `CeFp8Error` (fail-closed); читается на трейсе; явный аргумент сильнее env.
2) `_chunked_cross_entropy` — при гейте on: `jax.custom_vjp` с границей `(features, embedding) -> loss` (targets/loss_mask — недифференцируемые аргументы, котангент None):
   - forward (на чанк, глобальные сумма NLL / счётчик, loss_mask в числителе и знаменателе): `a = q(features_chunk, s_f)`, `b = q(embedding.T, s_e)` — квантизация e4m3: `clip(x/s, -448, 448).astype(jnp.float8_e4m3fn)`, per-tensor `s_f=max|features_chunk|/448`, `s_e=max|embedding|/448` (f32, данные-зависимые, STE: в backward — константы); `logits = _fp8_matmul(a, b) * (s_f*s_e)`; softmax/logsumexp/NLL — в fp32;
   - backward: пересчитать лог-иты чанка тем же fp8-путём (без сохранения полных (chunk,vocab)), `p=softmax(logits)` (f32), `dlogits=(p-onehot(target))*(dL/global_count)`, `dembedding += compute_dtype.gemm(dlogits.T, features_chunk)` и `dfeatures_chunk = compute_dtype.gemm(dlogits, embedding)` — bf16-операнды/f32-аккум по НЕквантованным операндам; loss_mask исключает строки из обоих;
   - при gate off — дословно прежний код (побитово).
3) `net/compute_dtype.py` при необходимости — helper для fp8-квантизации/скейла (не менять поведение bf16/fp32-режимов).
4) Тесты `net/tests/test_49_ce_fp8.py` (CPU, со швом `_fp8_matmul`):
   a. гейт: пусто/0/1/регистр/пробелы; неизвестное → `CeFp8Error`; явный аргумент сильнее env.
   b. off: loss и градиенты features/embedding побитово равны ревизии до дельты.
   c. on: forward-loss == off в допуске (ожидание ≤2e-2, фактически ~5e-5); градиенты == bf16-эталон в scale-relative ≤2e-2 (фактический max|d|/scale задокументировать); хвостовой неполный чанк; loss_mask.
   d. память: в jaxpr/lower нет полного (B,T,vocab); живой тензор ≤ чанка; в backward нет удержания логитов между чанками.
   e. STE: градиент квантизации не даёт нулей/мусора (сравнение с bf16-эталоном — зубы); fp8-скейлы не течут в градиенты.
5) Прогони по одному файлу: `test_49_ce_fp8.py`, затем регресс: `test_29_compute_dtype` (унаследованный красный — зафиксировать), `test_30_mla_flash_parity`, `test_36_accum_jit`, `test_39_bwd_bf16_storage`, `test_46_mtp_kda_impl`.
6) Ограничения: без unsafe; `net/config.json` не трогать; gate default 0; при off — побитово (блокирующий критерий); стиль русских docstring.

JSON-контракт результата обязателен (status/assumptions/open_questions). Финальный git-коммит обязателен.

## Границы (scope)

Машинный контракт границ — `MANIFEST.json.scope` (хэш `scope_hash`); изменение вне границ ловится гейтом, а не обсуждается постфактум.

- **Можно писать:** `net/**`
- **Нельзя писать (сильнее allow):** `model/`, `ARCHITECTURE-SPINE.md`, `CONSTRAINTS.yaml`
- **Можно запускать:** `python3`, `python`, `pytest`, `/home/roman/venv-axiom-0112cpu-py312/bin/python*`
- **Сеть:** сети нет (детерминированный узел)

## План отката

Откат: git reset --hard <baseline_commit из .arch-handoff/MANIFEST.json> (гейт default 0 — побитово прежний путь). Признак регрессии: off-путь перестал быть побитовым, паритет on вне допуска или новые красные.

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
