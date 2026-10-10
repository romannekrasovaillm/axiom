---
id: ADR-052
type: adr
title: "Кандидатная линия JAX 0.11.2 (Mosaic GPU) рядом с legacy 0.10.2: изолированное окружение, capability gate, staged rollout"
status: "proposed"
affects: [CMP-002, NFR-002]
source: "docs/adr/ADR-052-kandidatnaya-liniya-jax-0-11-2-mosaic-gpu-ryadom-s-legacy-0-10-2-izolirovannoe-okruzhenie-capability-gate-staged-rollout.md"
---

Кампания упёрлась в потолок стека 0.10.2: **377 ток/с = 1.16% MFU**, узкое место — **~6 000 запусков CUDA-графов на шаг** (33 418 `cuGraphLaunch` за 120 с = 77% времени при 1.04 с GPU-работы); флаги (`min_graph_size`, `command_buffer`) эффекта не дают, `B=2` не компилируется. Собственный Pallas-кернел убрал 115 200 TRSM-ядер и не дал времени — то есть ядра не узкое место. Mosaic MMA на 0.10.2 недоступен (`Layout.MMA_ACC`).

**Решение:** кандидатная линия — **отдельное окружение** `/home/roman/venv-axiom-0112` (`jax==0.11.2`, `jaxlib==0.11.2` + GPU plugin/PJRT фактических версий с wheel-хешами); legacy `venv-axiom` (0.10.2) не изменяется и служит точкой отката. **Capability gate вместо проверки версии**: JAX GPU / Mosaic ALU / Mosaic MMA — раздельные статусы, MMA подтверждается только компиляцией под CC12.1 с IR/PTX/SASS и численной сверкой; `hasattr`, CPU-прогон и `interpret` не являются доказательством. **Staged rollout** по `mosaic-gb10-migrate`: baseline-инвентаризация → adapter (legacy default, ленивый импорт Mosaic) → non-MMA части → MMA (после доказательства) → parity → performance → switch → retire. Числа кампании на новой линии **перемеряются** (правило сопоставимости). Nightly/latest не используются.

Скиллы-опоры: `mosaic-gb10/{migrate,probe,pin,kernel,mma,tune}`, `jax-mosaic-0112/{jax-0112-env,mosaic-0112-probe,mosaic-0112-kernel,mosaic-0112-mma,mosaic-0112-tune,mosaic-0112-migrate,jax-0112-transforms}`, `gb10-jax-env`, `pallas-gb10-*`.
