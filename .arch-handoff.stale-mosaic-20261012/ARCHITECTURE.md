# Архитектурный контекст (epic-context)

Собран: 2026-10-11T08:17:25.376719054+00:00

Источники:
- /home/roman/.arch-ml/worktrees/axiom-43e236d789188868/mfu55-step-exec/docs/adr/ADR-054-ispolnenie-shaga-donate-argumenty-jit-ustranenie-dtod-kopiy-fyuzing-put-g1-g4.md
- /home/roman/.arch-ml/worktrees/axiom-43e236d789188868/mfu55-step-exec/docs/adr/ADR-053-kampaniya-krupnyy-shag-rost-mfu-cherez-ukrupnenie-raboty-batch-tokenov-gradientnye-tehniki-potolok-pamyati-izmeren.md
- /home/roman/.arch-ml/worktrees/axiom-43e236d789188868/mfu55-step-exec/ARCHITECTURE-SPINE.md

<!-- источник: /home/roman/.arch-ml/worktrees/axiom-43e236d789188868/mfu55-step-exec/docs/adr/ADR-054-ispolnenie-shaga-donate-argumenty-jit-ustranenie-dtod-kopiy-fyuzing-put-g1-g4.md -->

---
id: ADR-054
title: "Исполнение шага: donate-аргументы jit (устранение ~28k DtoD-копий/шаг и ~420k API-вызовов), стадия-2 — фьюзинг; путь G1→G4 к 55% MFU"
status: Proposed
date: "2026-10-11"
depends_on: [ADR-050, ADR-053]
affects: [net/train_loop.py, net/optimizer.py, CMP-002]
spec_files: [ARCHITECTURE-SPINE.md]
---

# ADR-054. Исполнение шага (donate) и фьюзинг

- Date: 2026-10-11
- Status: Proposed
- Маршрут: Standard (significant_nfr; financial_impact — ток/с определяет календарь 20B-прогона)

## Context

Все флаговые рычаги исчерпаны измерением (ADR-053 Am.3): лидер `c64_B2` = 478.7 ток/с (1.469% MFU),
S1 (3%) не взят, разрыв ×2.04; до 55% (17 917 ток/с) ×37. Профиль лидера (nsys, стационар):

- GPU-ядра **1.55 с из 34 с шага (4.6%)**; ~92k запусков ядер/шаг; ~28k DtoD-копий/шаг (**31.5 ГиБ**);
  ~420k runtime-API-вызовов/шаг — хост-поток насыщен (165 с API на 157 с окна);
- причина копий в коде: `grad_fn = jax.jit(jax.value_and_grad(loss_fn))` без donate
  (`net/train_loop.py:2873`), `step_fn = jax.jit(step)` без donate (`net/optimizer.py:366`) —
  каждый вызов таскает деревья params/grads/state через копии буферов; премисса donate уже есть:
  `params = working_params(master)` пересоздаётся каждый шаг (`net/train_loop.py:2999`);
- ядра без диспетчеризации дали бы ~10 570 ток/с ≈ 32% MFU — рамка Am.2 (≈16k ≈ 50%) достижима
  при устранении исполнения-барьера; 55% требует и укрупнения форм (B≥4, заблокирован HLO-полом
  109.38 GiB — отдельная диагностика).

## Decision

1. **Стадия-1 (эта дельта, код через handoff): donate-аргументы.**
   - `grad_fn_donated = jax.jit(jax.value_and_grad(loss_fn), donate_argnums=(0,))` — используется
     в ветке одного микробатча (`len(group) == 1`, основной путь кампании); в accum-ветке params
     нужен на каждом микровызове — там остаётся недонированный `grad_fn`;
   - `optimizer.make_step(..., donate: bool = True)`: `jax.jit(step, donate_argnums=(0, 1, 2))`
     (master, grads, state потребляются); при `phase_profile` — `donate=False` (profiler
     читает `grads` после `step_fn`);
   - арифметика не меняется: donate — контракт владения буферами, не математика; паритет —
     существующие тесты + новый тест «шаг до/после donate: loss и params allclose» на CPU.
2. **Стадия-2 (после ремера стадии-1, отдельной дельтой): фьюзинг узлов графа** — Pallas-кернел
   KDA-чанка (один запуск на слой-проход вместо микро-узлов; скилл `pallas-gb10-kernel`), батчинг
   NS5-итераций; приоритет по остаточному профилю.
3. **Диагностика параллельно (архитектор, не код):** HLO-дамп B=4 — какой оп требует 88.91 ГиБ
   (пол 109.38 GiB не срезается рематом, Am.3); закрывает/переоткрывает лестницу батча B=4/8 для G3.
4. **Гейты (та же лестница ADR-050):** G1 ≥ 3% (977 ток/с) при `ядер/шаг ≤ 2000, синхронизаций ≤ 5000`;
   далее G2 ≥ 10%, G3 ≥ 20% (нужен B ≥ 4), G4 55%. Приёмка стадии-1: A/B на c64_B2 —
   DtoD/шаг < 2k (с ~28k), ток/с растёт (ожидание консервативное: шаг 34 → ≤ 25 с); loss-паритет.
5. **Честность цели:** 55% не пинится как порог (ADR-045/050 п.3); «строгий выход на 55» =
   движение по гейтам с числом на каждом, фиксация потолка измерением, если он ниже.

## Alternatives Considered

| Вариант | Плюсы | Минусы |
|---|---|---|
| **A. Donate + затем фьюзинг по профилю** (принято) | Дешёво, адрес доказан (31.5 ГиБ DtoD = размер деревьев), арифметика не меняется | Не гарантирует S1 сам по себе; требует аккуратности с accum-веткой и profiler |
| B. Сразу Pallas-фьюзинг KDA | Убирает микро-ядра в корне | Дорого, рискованно, а donate может снять большую часть барьера — профиль после стадии-1 переназначит приоритет |
| C. Смена знаменателя (FP8/MXFP4) | Кратный множитель | Отдельное решение (ADR-050 п.5е), ломает сопоставимость; барьер исполнения не убирает |
| D. Аренда внешнего GPU | Обход потолка GB10 | Не решает задачу владельца (свой бокс); ADR-050 требует сначала свой потолок |

## Consequences



### Positive

- Прямой адрес измеренного барьера (DtoD = 43% API-времени); ожидание — кратное снижение
  диспетчерской нагрузки и рост ток/с без изменения математики;
- обратимость тотальная (флаг `donate`, дефолт безопасен), паритет — существующие тесты.

### Negative

- Donate с багом = « donated buffer reused» — ловится тестами на повторных шагах;
- если DtoD-копии порождаются не jit-вызовами, а самим графом (например, prelayout-копиями),
  стадия-1 даст меньше ожидаемого — тогда профиль стадии-1 переназначит стадию-2;
- 55% остаётся за пределами гарантии: LPDDR5x-баланс и зрелость XLA на sm_121 — риск зафиксировать
  потолок ниже цели (тогда развилка FP8/аренда — отдельный ADR, ADR-050 п.4).

## Reversibility

**reversible.** Флаг `donate` (default True, откат — False одной строкой); код изолирован в
worktree `arch/mfu55-step-exec`; веса/данные не затрагиваются.

## References

- Профиль: `evidence/mfu-55/bigstep/sweep2/profile-best/` (sqlite-запросы в ADR-053 Am.3);
- ADR-053 Am.3 (вердикты ночи-2), ADR-050 (лестница гейтов), ADR-041 (память), ADR-045 (рамка пина).

<!-- источник: /home/roman/.arch-ml/worktrees/axiom-43e236d789188868/mfu55-step-exec/docs/adr/ADR-053-kampaniya-krupnyy-shag-rost-mfu-cherez-ukrupnenie-rab

> **Контекст усечён** до 6000 символов; полные тексты — в файлах-источниках (см. MANIFEST.json).
