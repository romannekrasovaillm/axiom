# Axiom safe rerun — итог (LR 1e-3 + NaN guard)

- metrics-seg-0.jsonl: steps 11..50; finite=40/40; loss 11.79904842376709 -> 10.428465843200684
- metrics-seg-1.jsonl: steps 51..100; finite=50/50; loss 10.284464836120605 -> 9.355151176452637
- metrics-seg-2.jsonl: steps 101..114; finite=14/14; loss 9.387964248657227 -> 9.411521911621094
- midtrain: 101..110; loss 13.596001625061035 -> 11.84277057647705
- eval:
  - pretrain: domain 14.22038402557373, W 8.455506898462772
  - midtrain: domain 13.04444244702657, W 8.52990360558033

Pretrain остановлен на pinned finite (segment2 прерван ради бюджета).
