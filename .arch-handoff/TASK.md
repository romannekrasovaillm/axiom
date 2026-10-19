# Задача для кодового харнесса

Цикл-30 кампании MFU-55 (ADR-054 Am.52): **fused MoE backward** — главный измеренный адрес. `sec_bwd_moe` = 10.235 с (49 % полного backward 20.98, 38 % шага 27.1) при ~0.6 TFLOP ≈ 0.06 TFLOPS: overhead/мелкие кернелы градиента экспертов. Forward остаётся ragged (fused-forward измерен минусом); fused — только backward. Гейт default `jax` (побитово).

ОКРУЖЕНИЕ: CPU-интерпретатор `/home/roman/venv-axiom-0112cpu-py312/bin/python` (jax 0.11.2, CPU); `NET_JAX_BACKEND=cpu JAX_PLATFORMS=cpu`; по одному файлу pytest с heartbeat; полный pytest не гонять. GPU-only кернел; на CPU fused-режим отказывает типизированно (тесты подменяют шов, форма KDA/MLA).

1) `net/moe.py` — гейт `AXIOM_MOE_BWD` (форма `counting_sort_mode`/`persist_mode`): `"jax"` (дефолт — прежний autodiff, побитово) | `"fused"`; пусто/не задано = "jax"; неизвестное → `MoeBwdError` (fail-closed); читается на трейсе.
2) `_routed_experts_grouped`: при gate="fused" — `jax.custom_vjp` вокруг экспертной части: forward — ДОСЛОВНО прежний путь (grouping + три `_ragged_gemm` + `siti_glu`; числа те же); backward — fused-кернел + XLA для группировки/сборки:
   - входы кернела: `z_grouped (rows,768)`, `h_g (rows,384)`, `h_u (rows,384)`, `a (rows,384)`, `de_grouped (rows,768)`, веса `wg (12,768,384)`, `wu (12,768,384)`, `wd (12,384,768)`, `group_sizes (12,)`;
   - на эксперта (grid=(12,), SMEM ≤99 КиБ, bf16-операнды/fp32-аккум): `dwd = aᵀ·de` (384×768); `da = de·wdᵀ` (rows×384); siti-adjoint дословно (`net/mlp.py:siti_glu`: `gate=βg·tanh(hg/βg)·sigmoid(hg)`, `up=βu·tanh(hu/βu)`; `dhg = da·up·d(hg-side)`, `dhu = da·gate·sech²(hu/βu)`, где `d(hg-side)=sech²(hg/βg)·sigmoid(hg)+tanh(hg/βg)·sigmoid(hg)(1−sigmoid(hg))`); `dwg = zᵀ·dhg`, `dwu = zᵀ·dhu`; `dz_grouped = dhg·wgᵀ + dhu·wuᵀ`;
   - выходы: `dwg,dwu,dwd` (по экспертам) и `dz_grouped`; XLA — `de_grouped` из котангента выхода (обратная перестановка/сборка, как в forward), `dz` обратно по токенам, градиенты `sel_p`/`topk` (routing) — прежним XLA-путём.
   - новый модуль `net/kernels/moe_experts_bwd.py` (Mosaic GPU): `CASES/make_inputs/reference/kernel/cost/TUNE_SPACE/valid_config/device_available`, плюс jnp-`reference_bwd` (адъюнкт экспертного блока) для CPU-паритета.
3) Тесты `net/tests/test_54_moe_bwd_fused.py` (CPU, шов kernel):
   a. гейт: пусто/jax/fused/регистр/пробелы; неизвестное → `MoeBwdError`; явный аргумент сильнее env;
   b. off: MoE output/градиенты побитово равны ревизии до дельты;
   c. on (шов = jnp-reference_bwd): градиенты z_grouped/весов == autodiff off в scale-relative ≤2e-2 (bf16); порядок групп/строк сохранён; особые случаи: пустой эксперт (group_size=0), неполная группа;
   d. `valid_config`/`device_available=False` → отказ типизированно;
   e. lower_check-адаптер `net/tests/moe_experts_bwd_lower.py` (форма `kda_persist_bwd_lower.py`).
4) Прогони по одному файлу: `test_54_moe_bwd_fused.py`, затем `test_32_moe_grouped_topk`, `test_46_mtp_kda_impl`, `test_47_ce_bwd`, `test_50_gemm_fp8`.
5) Ограничения: без unsafe; `net/config.json` не трогать; gate default `jax`; при off — побитово (блокирующий критерий); forward не менять; стиль русских docstring.

JSON-контракт результата обязателен (status/assumptions/open_questions). Финальный git-коммит обязателен.

## Границы (scope)

Машинный контракт границ — `MANIFEST.json.scope` (хэш `scope_hash`); изменение вне границ ловится гейтом, а не обсуждается постфактум.

- **Можно писать:** `net/**`
- **Нельзя писать (сильнее allow):** `model/`, `ARCHITECTURE-SPINE.md`, `CONSTRAINTS.yaml`
- **Можно запускать:** `python3`, `python`, `pytest`, `/home/roman/venv-axiom-0112cpu-py312/bin/python*`
- **Сеть:** сети нет (детерминированный узел)

## План отката

Откат: git reset --hard <baseline_commit из .arch-handoff/MANIFEST.json> (гейт default `jax` — побитово прежний путь). Признак регрессии: off-путь перестал быть побитовым, паритет MoE-backward вне допуска, новые красные.

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
