# Задача для кодового харнесса

ЗАДАЧА (довести Mosaic-ядро KDA-решения: реализовать тело `kernel` по проверенному рецепту).

СОСТОЯНИЕ. Каркас `net/kernels/kda_ut_solve_mosaic.py` есть (контракт переиспользует Triton-версию импортом, backend override `AXIOM_KDA_SOLVE_KERNEL`), но тело не реализовано (`KERNEL_IMPLEMENTED=False`). Это было честно: неизвестен был переход раскладок между последовательными `plgpu.mma`. **Теперь он найден и проверен на GB10** — см. `evidence/mfu-55/mosaic/REPORT-mosaic-chain.md`:

```python
def body(L_ref, o_ref, smem):                       # scratch_types=[plgpu.SMEM((M,N),DT)]
    acc = plgpu.layout_cast(jnp.zeros((M,N), ACC), plgpu.Layout.MMA_ACC(DT))
    a   = plgpu.load(L_ref,   layout=plgpu.Layout.MMA_LHS(DT), optimized=False)
    b   = plgpu.load(L_ref.T, layout=plgpu.Layout.MMA_RHS(DT), optimized=False)   # RHS = (n,k)
    p   = plgpu.mma(acc, a, b)
    smem[...] = p.astype(DT)                        # ВЫГРУЗКА в SMEM — обязательный шаг цепочки
    p_lhs = plgpu.load(smem, layout=plgpu.Layout.MMA_LHS(DT), optimized=False)
    acc2  = plgpu.layout_cast(jnp.zeros((M,N), ACC), plgpu.Layout.MMA_ACC(DT))
    p2    = plgpu.mma(acc2, p_lhs, b)               # цепочка продолжается так же
    o_ref[...] = p2.astype(DT)
```
Проверено: `compile_stage ok`, `run-ok`, `max_abs 0.0079` на цепочке `(L@L)@L` (M=K=N=128, bf16). **Ключевые факты:** `layout_cast(MMA_ACC → MMA_LHS)` НЕ поддерживается — переход только через SMEM; `scratch_types` — список `plgpu.SMEM(shape, dtype)`; `lowering_semantics=Lane`; у `plgpu.kernel`-объекта нет `.lower` (нужен `jax.jit(fn).lower`).

ЧТО СДЕЛАТЬ.
1. Реализовать тело `kernel` (Neumann-произведение для `(I+L)·X = B`): цепочка `plgpu.mma` с SMEM-переходами между шагами, `S = ceil(log2(C))` шагов; результат — `X`.
2. **Формы.** Задача: H=12, C=64, dk+dv=256. MMA в 0.11.2 требует форм, кратных блокам (эталон использует M=K=128, N=8). Выбери и **обоснуй** в отчёте один из путей: (а) паддинг `C=64 → 128` нулями (математически нейтрально для нижней треугольной структуры, но требует аккуратности с `(I+L)`); (б) батчевание по головам так, чтобы m-размерность набиралась из голов; (в) иной. Укажи, как это влияет на память SMEM (лимит GB10 ~99 КБ на блок).
3. Сохранить контракт: `CASES` — реальные формы (H=12, C=64, dk=dv=128), `make_inputs/reference/baseline/cost/TUNE_SPACE/valid_config`; `kernel()` вызывается из общего диспетчера, дефолт остаётся Triton.
4. Тест `net/tests/test_kda_mosaic_parity.py` дополнить: (а) `kernel` не поднимает `KERNEL_IMPLEMENTED`-ошибку, а требует GPU (честный отказ на CPU без подмены); (б) структура вызова (scratch/семантика) соответствует рецепту; (в) `valid_config` отсеивает конфигурации, не влезающие в SMEM.
5. CPU-паритет: если Mosaic на CPU недоступен — так и написать (NOT RUN), не выдавая проверку за пройденную; но `reference`-путь (jnp) обязан быть сверен с `solve_jax` в пределах допуска.

ОГРАНИЧЕНИЯ. Зона: `net/kernels/*`, `net/tests/*`. НЕ трогать `net/kda.py`, `net/config.json`, `tools/pretrain_run.py`. GPU-прогоны НЕ запускать (их делает архитектор в `~/venv-axiom-0112`); код обязан быть исполняемым на стенде командой:
`/home/roman/venv-axiom-0112/bin/python -c "…kernel…"`. Не подменять проверку (`hasattr`/импорт/`interpret` — не доказательство).

ПРОВЕРКА (ПК): `/home/roman/venv-axiom/bin/python -m pytest -q net/tests/test_kda_mosaic_parity.py`.

РЕЗУЛЬТАТ. `.arch-handoff/result.json` + ФИНАЛЬНЫЙ git-commit. В assumptions — выбранная стратегия форм (с обоснованием) и оценка SMEM; в open_questions — что проверить на GB10 (компиляция под sm_121, IR/PTX: действительно ли MMA, время против Triton и XLA, детерминизм, точность цепочки).

## Границы (scope)

Машинный контракт границ — `MANIFEST.json.scope` (хэш `scope_hash`); изменение вне границ ловится гейтом, а не обсуждается постфактум.

- **Можно писать:** `net/kernels/*`, `net/tests/*`
- **Нельзя писать (сильнее allow):** `model/`, `ARCHITECTURE-SPINE.md`, `CONSTRAINTS.yaml`
- **Можно запускать:** `python3 -m pytest*`
- **Сеть:** сети нет (детерминированный узел)

## План отката

Откат: `git reset --hard cb2e45d` (baseline — последний коммит до работы исполнителя; вся его работа приходит одним коммитом поверх).
Сигналы отката: провал fitness-гейта (`arch-ml control check`), непустой `conflicts_with_prior_decisions`, статус `blocked`.
Владелец решения об откате — solution-архитектор; исполнитель откат не выполняет и не маскирует проблему обходным редизайном.
Обратимость: полная — единая точка изменений, коммит исполнителя.

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
