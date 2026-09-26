---
id: ADR-008
type: adr
title: "Стек реализации модели — JAX / MaxText (supersede ADR-007)"
status: "accepted"
affects: [CMP-002, CMP-003, CMP-004]
source: "docs/adr/ADR-008-stek-realizacii-modeli-jax-maxtext-supersede-adr-007.md"
---

База — форк MaxText (JAX/Flax NNX): шардирование, MTP, Muon/MuonClip, Tunix SFT + GRPO/GSPO, Orbax-чекпойнты; K2-семейство поддержано. KDA — собственная реализация (referенс JAX/scan на скелете → кернелы Pallas к L1, замер MFU). QAT MXFP4 — своя fake-quant либо AQT [ТРЕБУЕТ ПРОВЕРКИ]. Кандидат отката — ADR-007 (PyTorch/FLA, проработан).
