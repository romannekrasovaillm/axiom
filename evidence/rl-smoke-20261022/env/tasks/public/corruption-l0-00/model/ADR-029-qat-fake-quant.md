---
id: ADR-029
type: adr
title: "QAT MXFP4: собственный fake-quant STE; AQT не принимается"
status: "accepted"
affects: [CMP-002, CMP-003]
source: "docs/adr/ADR-029-qat-mxfp4-sobstvennyy-fake-quant.md"
---

QAT-тренировка (со стадии SFT, ADR-005 п. 7) — собственный pure-JAX fake-quant
STE в net/quant.py: веса MXFP4 (блок 32), латентный KV по ADR-009 D3
(E2M1 + E4M3/16ch), FP8 E4M3 для SWA KV; претрейн — без QAT. AQT отклонён:
нет OCP MXFP4-microscaling семантики ADR-009 D3, вендор-триггер, TPU-центричность,
для fake-quant-тренировки ускорения не даёт. Реальный квантованный инференс —
отдельное решение стадии инференса. Отработано смоуками SFT (ПК + GB10,
qat_weights: on, 200 шагов).
