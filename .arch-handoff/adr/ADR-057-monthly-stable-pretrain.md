---
id: ADR-057
title: "Месячная программа Axiom: стабильный continued pretrain на GB10 (NaN-гард, LR-sweep, длинный прогон, midtrain/eval)"
status: Accepted
date: "2026-10-23"
depends_on: [ADR-041, ADR-050, ADR-055, ADR-056]
affects: [evidence, tools, ARCHITECTURE-SPINE.md]
spec_files: [ARCHITECTURE-SPINE.md]
---

# ADR-057. Месячная программа стабильного pretrain

- Date: 2026-10-23
- Status: Accepted (владелец: бюджет месяц, приоритет — «Стабильный pretrain»)
- Маршрут: Standard (significant_nfr — стабильность/качество; financial_impact — месяц GB10)

## Context

Ночной continued pretrain (22–23.10.2026) дал **NaN на step 206** при LR 1e-2:
шаги 201–205 конечны, 206 — NaN; сегментный оркестратор не имел NaN-гарда,
продолжил до 1700, а `keep-last 3` удалил последний конечный чекпоинт step200.
Safe rerun с LR 1e-3 и per-segment гардом дал 100 конечных шагов
(loss 11.799→9.355), midtrain 10 шагов и eval, но горизонт мал.

Владелец дал бюджет **месяц** и выбрал приоритет «Стабильный pretrain».

## Decision

1. **Phase 0 — устойчивый runner.** Перед длинными прогонами:
   - `--stop-file` + `--stop-check-every 1`; watchdog читает metrics после
     каждого шага и при NaN/Inf касается STOP;
   - последний finite-чекпоинт **пинится** в отдельный каталог после каждого
     сегмента; ретенция active не может его удалить;
   - сегменты 50–100 шагов, `checkpoint-every` = длине сегмента;
   - при NaN/failure active-сегмент карантинируется, midtrain/eval на NaN не
     запускается; vLLM восстанавливается автоматически.
2. **Phase 1 — LR-sweep.** 5e-4 / 1e-3 / 2e-3, по 300 шагов от одного
   finite-базиса (demo step10), B=2, W. Гейт: все шаги finite, loss снижается,
   tok/s в ожидаемом диапазоне; выбирается максимальный LR без NaN.
3. **Phase 2 — длинный continued pretrain** выбранным LR, ≥10 000 шагов,
   сегментами, с чекпоинтами и периодическим eval (domain/W). Гейт: нет NaN,
   loss-тренд, eval не деградирует; иначе — стоп и разбор.
4. **Phase 3 — midtrain** по принятой политике ADR-055 (10 шагов LR 1e-3,
   K=1.0, чекпоинт step20) от лучшего pretrain-чекпоинта; eval domain/W.
5. **Phase 4 — full SFT/RL** — после стабильного pretrain, отдельными дельтами
   (реальный токенизатор 160k, eval-набор CL-017, K3-калибровка).
6. **Гейты и отчётность:** каждый сегмент пишет STATUS; итог — REPORT с
   first_nan (если был), pinned step, eval. Owner-gate — на переход Phase 2→3.

## Alternatives Considered

| Вариант | Плюсы | Минусы |
|---|---|---|
| **A. Watchdog + LR-sweep + длинный pretrain** (принято) | Устраняет причину ночного инцидента; выбор LR по измерению | Требует времени на sweep |
| B. Сразу длинный pretrain на LR 1e-3 | Быстрее к модели | Нет доказательства, что 1e-3 — лучший безопасный LR |
| C. Сначала full SFT/RL | Пост-тренировка быстрее | Стоит на нестабильных/слабых весах |
| D. Только eval-бенчмарк | Дешёво | Не создаёт лучшую модель |

## Consequences

### Positive
- NaN не может молча испортить длинный прогон; последний finite всегда закреплён.
- LR выбирается измерением, а не догадкой.
- Длинный pretrain даёт более содержательный чекпоинт для midtrain/SFT/RL.

### Negative
- Sweep и длинный прогон занимают дни; GPU занят, vLLM периодически остановлен.
- Pinned-копии увеличивают расход shared-диска (контроль `keep-last` + пин).
- Даже 10k шагов — небольшой бюджет токенов для 1B-модели; это continued run,
  не полноценный pretrain.

## Reversibility

**Обратимо.** Все стадии — отдельные прогоны/чекпоинты; откат — вернуться к
предыдущему pinned/демо-чекпоинту. Данные и веса не мигрируют. ADR можно
пометить Superseded при смене приоритета.

## Evidence

- Ночной инцидент: `~/gb10-shared/runs/axiom-overnight-20261022/OVERNIGHT-REPORT.md`.
- Safe rerun: `~/gb10-shared/runs/axiom-safe-rerun-20261023/SAFE-RERUN-REPORT.md`,
  eval domain 14.220→13.044, W 8.456→8.530; finite step100/110.
- Скилл `gb10-mfu-campaign-loop`: правило NaN-гарда и пина.
