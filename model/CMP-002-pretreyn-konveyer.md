---
id: CMP-002
type: cmp
title: "Претрейн-конвейер и аренда GPU"
status: "designed"
depends_on: [CMP-001, INT-001]
code_roots: [net/model.py, net/train_loop.py, net/optimizer.py, net/checkpoint.py, net/config.py, net/kda.py, net/mla.py, net/moe.py, net/mlp.py, net/mtp.py, net/attn_sparse.py, net/attnres.py, net/shortconv.py, net/norm.py, net/quant.py, net/remat.py, net/compute_dtype.py, net/master_dtype.py, net/model_ladder.py, net/curriculum.py, net/vit.py, net/infer.py, net/provenance.py, net/topk_exact.py, net/gpu_smoke.py, net/kernels, tools/pretrain_run.py, tools/run_a4_pipeline.py]
implements: [AD-4, AD-8, NFR-003]
---

Стадийный прогон: **L3-скелет первым** (8 GPU × 12 ч, ≈$200) — проверка всего конвейера. Переход к L1 (64 GPU × 14 сут) и далее — только по отдельному решению владельца о масштабном классе. Стек — JAX + форк MaxText (ADR-008): шардирование/MTP/Muon/Tunix-GRPO из MaxText, KDA — собственная реализация (referенс → кернелы Pallas к L1 с замером MFU), QAT — своя fake-quant/AQT; смета валидна только при достойном MFU — замер обязателен перед L1. Чекпойнты Orbax и финальные веса — в каноническое хранилище `~/gb10-shared` (решение владельца 2026-09-12); в рабочих каталогах — симлинки, копии весов запрещены (C-032); хеш пиннинга — по содержимому (AD-4).
