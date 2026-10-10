# Задача для кодового харнесса

ЗАДАЧА (Mosaic-ядро KDA: применить найденный рецепт раскладки RHS — последний блокирующий дефект закрыт).

СОСТОЯНИЕ. Четыре дефекта закрыты (NameError ×3, `grid_names`, 2D-операнды). Пятый — транспорт раскладок — **решён экспериментом на GB10**, рецепт в `evidence/mfu-55/mosaic/REPORT-rhs-layout.md`:

**`plgpu.mma(acc, a, b)` требует, чтобы память RHS была `k`-контигуальной**: массив правой части надо подавать в порядке `(n, k)` и грузить через `.T`. Проверено для `N=8/64/256` — все `ok`, `max_abs 6.1e-05` (`mma_rhs_layout_proto.py`).

Рабочий паттерн (проверен на GB10 целиком):
```python
def body(l_ref, bt_ref, o_ref, smem):     # l_ref (H,M,K); bt_ref (H,N,K) — B ТРАНСПОНИРОВАНА в памяти
    h = jax.lax.axis_index("head")
    acc = plgpu.layout_cast(jnp.zeros((M, N), ACC), plgpu.Layout.MMA_ACC(DT))
    a   = plgpu.load(l_ref.at[h],    layout=plgpu.Layout.MMA_LHS(DT), optimized=False)
    b   = plgpu.load(bt_ref.at[h].T, layout=plgpu.Layout.MMA_RHS(DT), optimized=False)
    o_ref.at[h][...] = plgpu.mma(acc, a, b).astype(DT)

plgpu.kernel(body, out_type=ShapeDtypeStruct((H,M,N), DT),
             scratch_types=[plgpu.SMEM((M,M), DT)],
             compiler_params=plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Lane),
             grid=(H,), grid_names=("head",))
# вызов: kernel(L, B.transpose(0, 2, 1))
```

ЧТО СДЕЛАТЬ.
1. Применить рецепт в `net/kernels/kda_ut_solve_mosaic.py`: правая часть `(H, C, D)` подаётся в ядро **транспонированной** `(H, D, C)` (в `kernel()` — `b.transpose(0, 2, 1)`), внутри — `plgpu.load(bt_ref.at[h].T, layout=MMA_RHS(...))`; LHS — `l_ref.at[h]` без транспозиции; выход `(H, M, N)` через `o_ref.at[h][...]`.
2. Сохранить Neumann-цепочку и SMEM-переход между шагами (проверен: результата одного `mma` нельзя подать операндом напрямую, только через SMEM: `smem[...] = p.astype(DT)` → `plgpu.load(smem, layout=MMA_LHS/RHS, optimized=False)`). Для цепочки промежуточные матрицы `(M, M)` квадратные — там транспозиция в памяти не нужна, но **операнд RHS всё равно должен быть `k`-контигуален** (для квадратной `(M,M)` это выполняется при `.T` от row-major, как в прототипе).
3. Математику, паддинг C→128 и семантику (дефолт backend `triton`) не менять.
4. Тесты: дополнить `net/tests/test_kda_mosaic_parity.py` проверкой, что ядро вызывается с **транспонированной** правой частью (`b.transpose(0,2,1)`) и что в `body` RHS грузится через `.T` (статическая проверка исходника или spy на `plgpu.load`); сохранить требование «трассировка/создание ядра проходит без GPU» из прошлой дельты.
5. Обновить docstring модуля: зафиксировать три требования раскладок (LHS `(m,k)` без транспозиции; RHS — память `(n,k)` + `.T`; ACC `(m,n)`), чтобы следующая правка не наступила на те же грабли.

ОГРАНИЧЕНИЯ. Зона: `net/kernels/*`, `net/tests/*`. GPU-прогоны не запускать (их делает архитектор). `net/kda.py`, `net/config.json` не трогать.

ПРОВЕРКА (ПК): `/home/roman/venv-axiom/bin/python -m pytest -q net/tests/test_kda_mosaic_parity.py` → все зелёные.

РЕЗУЛЬТАТ. `.arch-handoff/result.json` + ФИНАЛЬНЫЙ git-commit. В assumptions — как реализована транспозиция входа и почему; в open_questions — что осталось измерить на GB10 (паритет bf16/fp32, IR/PTX = MMA, время против Triton и XLA, детерминизм, SMEM-бюджет).

## Границы (scope)

Машинный контракт границ — `MANIFEST.json.scope` (хэш `scope_hash`); изменение вне границ ловится гейтом, а не обсуждается постфактум.

- **Можно писать:** `net/kernels/*`, `net/tests/*`
- **Нельзя писать (сильнее allow):** `model/`, `ARCHITECTURE-SPINE.md`, `CONSTRAINTS.yaml`
- **Можно запускать:** `python3 -m pytest*`
- **Сеть:** сети нет (детерминированный узел)

## План отката

Откат: `git reset --hard 893c0e8` (baseline — последний коммит до работы исполнителя; вся его работа приходит одним коммитом поверх).
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
