# Задача для кодового харнесса

Цикл-24 кампании MFU-55 (ADR-054 Am.34/38): fused MoE-кернел (forward, ступень 1) — главный измеренный адрес G2/G3. Провенанс: MoE forward 4.13 с при ~0.3 TFLOP (~0.07 TFLOPS) — доминирует overhead/мелкие кернелы, не GEMM. Цель ступени 1: один персистентный Mosaic-кернел на эксперта, фьюзящий gate/up GEMM + SwiGLU (siti_glu) + down GEMM для сгруппированных строк; группировка/сборка (sort/scatter/mix) остаются XLA; backward — прежний autodiff (ступень 2 — отдельная дельта). Гейт default off (побитово прежний путь).

ОКРУЖЕНИЕ: CPU-интерпретатор `/home/roman/venv-axiom-0112cpu-py312/bin/python` (jax 0.11.2, CPU); `NET_JAX_BACKEND=cpu JAX_PLATFORMS=cpu`; по одному файлу pytest с heartbeat; полный pytest не гонять.

1) `net/moe.py` — гейт `AXIOM_MOE_IMPL` (форма `counting_sort_mode`/`persist_mode`): значения `"ragged"` (дефолт — текущий `lax.ragged_dot`-путь, побитово) и `"fused"`; пусто/не задано = `"ragged"`; неизвестное → `MoeImplError` (fail-closed); читается на трейсе; явный аргумент сильнее env.
2) Новый модуль `net/kernels/moe_experts_fused.py` (Mosaic GPU, sm_121 — образец `net/kernels/kda_cc_fused_persist.py`):
   - `CASES`, `make_inputs`, `reference`, `kernel`, `cost`, `TUNE_SPACE`, `valid_config`, `device_available` по контракту pallas-gb10-kernel;
   - `kernel(z_grouped (rows, latent=768), group_sizes (12,), wg (12,768,384), wu (12,768,384), wd (12,384,768)) -> e_grouped (rows, 768)`:
     каждый эксперт — отдельная программа (grid=(n_routed,)), строки группы читаются из z_grouped по префиксным суммам group_sizes; gate/up — `plgpu.mma` bf16-операнды/fp32-аккум (768x384), SwiGLU в теле (siti-семантика — прочитать `net/mlp.py:siti_glu` и повторить дословно), down (384x768); веса эксперта стримятся из GMEM (SMEM ≤99 КиБ/блок — тайлинг), выход — в e_grouped по тем же строкам. Один запуск на forward всего MoE-блока;
   - `valid_config`: латент 768/expert_inter 384/12 экспертов — живые формы; sieve отвергает то, что не влезает в SMEM/tiling (без тихого отката);
   - GPU-only: на CPU `device_available=False` → режим `fused` отказывает типизированной ошибкой (тесты подменяют шов, форма KDA/MLA).
3) `net/moe.py`: при gate="fused" заменить три `_ragged_gemm`+`siti_glu` на вызов `moe_experts_fused.kernel` (grouping и scatter/mix — вне кернела, как сейчас); при off — прежний код дословно.
4) Тесты `net/tests/test_48_moe_fused.py` (CPU):
   a. гейт: пусто/ragged/fused/регистр/пробелы; неизвестное → `MoeImplError`; явный аргумент сильнее env.
   b. off: MoE output/градиенты побитово равны ревизии до дельты (прямой вызов).
   c. on (с подменой kernel-шва на jnp-реализацию declared-нумерики): forward == off в scale-relative ≤2e-2 (bf16-операнды), SwiGLU-семантика совпадает; группировка/порядок строк сохраняются.
   d. `valid_config` отвергает нежизнеспособные формы; `device_available=False` → типизированная ошибка в fused-режиме.
   e. lower_check-адаптер (`net/tests/moe_fused_lower.py`, форма `kda_persist_bwd_lower.py`) — если доступен статический lowering.
5) Прогони по одному файлу: `test_48_moe_fused.py`, затем регресс: `test_32_moe_grouped_topk`, `test_46_mtp_kda_impl`, `test_33/34` (MoE-тесты — найти grep'ом `moe`), `test_02_kda_parity`.
6) Ограничения: без unsafe; `net/config.json` не трогать; gate default `ragged`; при off — побитово прежний путь (блокирующий критерий); стиль русских docstring.

JSON-контракт результата обязателен (status/assumptions/open_questions). Финальный git-коммит обязателен.

## Границы (scope)

Машинный контракт границ — `MANIFEST.json.scope` (хэш `scope_hash`); изменение вне границ ловится гейтом, а не обсуждается постфактум.

- **Можно писать:** `net/**`
- **Нельзя писать (сильнее allow):** `model/`, `ARCHITECTURE-SPINE.md`, `CONSTRAINTS.yaml`
- **Можно запускать:** `python3`, `python`, `pytest`, `/home/roman/venv-axiom-0112cpu-py312/bin/python*`
- **Сеть:** сети нет (детерминированный узел)

## План отката

Откат: git reset --hard <baseline_commit из .arch-handoff/MANIFEST.json> (гейт default `ragged` — побитово прежний путь). Признак регрессии: off-путь перестал быть побитовым, паритет MoE вне допуска или новые красные.

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
