---
id: ADR-021
type: adr
title: "Претрейн-микс L3: 17B FineWeb-Edu + 3B The Stack + 1B decay-фаза"
status: "accepted"
affects: [CMP-002]
source: "docs/adr/ADR-021-pretreyn-miks-l3-veb-kod-annealing-17b-fweb-3b-stack-1b-decay.md"
---

Микс претрейна L3: W ~17B FineWeb-Edu + C ~3B The Stack (v1-dedup дефолт; v2 флагом; fallback
codeparrot-clean) + Q ~1B decay на сужённом качественном миксе. Все шард-файлы — публичные,
с карточкой и sha256 (ADR-004). Приватные компоненты ADR-020 в претрейн не входят (AD-6).
