# Задача для кодового харнесса

Цикл-20 кампании MFU-55 (ADR-054 Am.32): перевести KDA MTP-слоя на кампанийный путь `chunked_cc` под гейтом, чтобы снять 256-графовый jnp-хвост (9.75 с GPU/шаг, 512 графов, 57.9k ядер — провенанс-карта цикла-19). Гейт по умолчанию — ПРЕЖНИЙ jnp-путь (побитово), новый путь — opt-in.

ОКРУЖЕНИЕ: CPU-интерпретатор `/home/roman/venv-axiom-0112cpu-py312/bin/python` (jax 0.11.2, CPU); все прогоны `NET_JAX_BACKEND=cpu JAX_PLATFORMS=cpu`; персистентный KDA-кернел на CPU отказывает — в тестах нового пути выставлять `AXIOM_KDA_CC_IMPL=jax` (форма существующих тестов) или подменять швы. По одному файлу pytest с heartbeat; полный pytest не гонять.

1) `net/mtp.py` — гейт и маршрутизация:
   - `MTP_KDA_IMPL_ENV = "AXIOM_MTP_KDA_IMPL"`; значения только `"chunked"` (дефолт, текущий `apply_chunked`) и `"campaign"` (`kda_mod.apply_kda(params.kda, cfg, xb, chunk_size)` — диспетчеризация по `cfg.kda_impl` = `chunked_cc`); пусто/не задано = `"chunked"`; неизвестное → `MtpKdaImplError` (fail-closed, форма `net.mla.persist_mode`); явный аргумент сильнее env; читается на трейсе.
   - В `apply(...)`: `attn = jax.vmap(lambda xb: _kda_interval(params.kda, cfg, xb, chunk_size))(z)`, где `_kda_interval` по гейту вызывает `apply_kda` или `apply_chunked`. При `"chunked"` — побитово прежний вызов/граф; никаких других изменений в `mtp.apply`.
2) Тесты `net/tests/test_46_mtp_kda_impl.py` (CPU):
   a. гейт: пусто/`chunked`/`campaign`/регистр/пробелы; неизвестное → `MtpKdaImplError` до трейса; явный аргумент сильнее env.
   b. off: `mtp.apply` при гейте off == прямой вызов прежней ветки `kda_mod.apply_chunked` (побитово; сравнивать outputs и градиенты по параметрам MTP).
   c. on: при `AXIOM_MTP_KDA_IMPL=campaign` и `AXIOM_KDA_CC_IMPL=jax` MTP-выход и градиенты == off в scale-relative допуске ≤2e-2 (кампанийный допуск bf16-операндов KDA; задокументировать фактический max|d|/scale в docstring).
   d. гейт не меняет другие call-site (в `net/model.py` KDA не тронут) — проверить, что `apply_kda` вызывается только из `_kda_interval`.
   e. nan/конечность: выход конечен на нескольких сидах.
3) Прогони по одному файлу: `test_46_mtp_kda_impl.py`, затем регресс: `test_kda_chunked_cc.py`, `test_kda_cc_fused.py`, `test_kda_cc_fused_persist.py` (если долго — только CPU-сечения), `test_02_kda_parity.py`, а также любой существующий тест модели/пресета, который затрагивает MTP (найди grep'ом `mtp`).
4) Ограничения: без unsafe; `net/config.json` не трогать; гейт default `"chunked"`; при off — побитово прежние числа и граф (блокирующий критерий); существующие тесты не ломать; стиль русских docstring.

JSON-контракт результата обязателен (status/assumptions/open_questions). Финальный git-коммит обязателен.

## Границы (scope)

Машинный контракт границ — `MANIFEST.json.scope` (хэш `scope_hash`); изменение вне границ ловится гейтом, а не обсуждается постфактум.

- **Можно писать:** `net/**`
- **Нельзя писать (сильнее allow):** `model/`, `ARCHITECTURE-SPINE.md`, `CONSTRAINTS.yaml`
- **Можно запускать:** `python3`, `python`, `pytest`, `/home/roman/venv-axiom-0112cpu-py312/bin/python*`
- **Сеть:** сети нет (детерминированный узел)

## План отката

Откат: git reset --hard <baseline_commit из .arch-handoff/MANIFEST.json> (гейт по умолчанию оставляет прежний jnp-путь; численные пути при off не менялись). Признак регрессии: MTP-выход при off перестал быть побитовым или новые красные.

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
