# Дельта: jax-0112-candidate
- Route: Standard
- Created: 2026-10-10
- Владелец: архитектор. Основание: ADR-052 (кандидатная линия JAX 0.11.2), измеренный потолок 0.10.2 (1.16% MFU, барьер `cuGraphLaunch`).

## Проблема

Кампания MFU-55 упёрлась в потолок стека 0.10.2: 377 ток/с (1.16% MFU), узкое место — ~6 000 запусков CUDA-графов на шаг (не ядра, не память), флагами не управляется. Mosaic-путь тензорных ядер на 0.10.2 недоступен (`Layout.MMA_ACC` отсутствует). Нужна проверка кандидатной линии JAX 0.11.2 (публичный `pallas.mosaic_gpu.mma`, Ampere MMA) — без риска для рабочей линии.

## ADDED

- Модель: сущность `model/ADR-052-…md` (решение о кандидатной линии, capability gate, staged rollout, rollback).
- Окружение: `/home/roman/venv-axiom-0112` (пин `jax==0.11.2`, `jaxlib==0.11.2` + GPU plugin/PJRT фактических версий) — изолированно, legacy `venv-axiom` не изменяется.
- (Далее по стадиям) `mosaic-0112-probe` report, `mosaic-0112-kernel` smoke (ALU), `mosaic-0112-mma` smoke (MMA, CC12.1) — с явными статусами DONE/BLOCKED/NOT RUN.

## MODIFIED

- Ничего в коде кейса: кандидатная линия не меняет `net/`, `tools/`, конфиги и прогоны кампании.

## REMOVED

- Ничего.

## План отката

Удаление `/home/roman/venv-axiom-0112` и артефактов кандидата; рабочая линия (`venv-axiom`, 0.10.2) не затронута. Признак регрессии — любой прогон кампании, начатый в неверном окружении (объявлять окружение в каждом замере явно).

## Критерии приёмки

- [ ] `pip check` чист, inventory зафиксирован (jax/jaxlib/plugin/PJRT/runtime фактических версий).
- [ ] JAX GPU работает: `jax.devices()` — GB10, JIT-операция с `block_until_ready` даёт корректный результат.
- [ ] `mosaic-0112-probe` report сохранён в `evidence/mfu-55/env/`.
- [ ] Mosaic ALU smoke: компиляция и исполнение (не `interpret`), численная сверка.
- [ ] Mosaic MMA на CC12.1: статус `compile-pass`/`blocked` с evidence (IR/PTX/SASS), а не по `hasattr`.
- [ ] Lock/манифест обеих линий сохранён; rollback проверен (переключение окружения).
