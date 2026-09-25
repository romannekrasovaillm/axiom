# axiom

**An LLM built from scratch on JAX — full pretrain + posttrain pipeline, single-box first.**

> TL;DR (RU): собственная нейросеть с нуля на JAX — претрейн, SFT, RL, агентные способности и свой
> конвейер обучения. Не адаптация чужих весов. Скелет **L3** (~1B MoE / 20B токенов) обучается
> целиком на одной машине (NVIDIA GB10 / DGX Spark). Архитектура и дисциплина — по опубликованным
> практикам фронтир-лабораторий; каждое решение зафиксировано ADR и проверяется механическими
> fitness-правилами (`CONSTRAINTS.yaml`), без LLM-судей.

## What is here

- **`net/`** — the model, in JAX: hybrid **KDA** linear attention (DeltaNet-family with short
  conv + decay gates) + **MLA** layers + **AttnRes** + **LatentMoE** (latent 0.5×hidden,
  12 routed + 2 shared experts, top-2, aux-free QB monitoring) + **SiTU-GLU**, **MTP-1** head,
  minimal vision path (ViT). BF16 pretrain, Muon + Newton–Schulz (Polar Express) optimizer.
- **`env/`** — the RL environment: task generation, verifier, reward (binary, mechanical), net executor.
- **`tools/`** — training/eval harness: SFT/RL smoke runs, precision pinning checks, A4 pipeline,
  manifest and budget gates.
- **`docs/adr/`** — architecture decision records (RU): scale class, JAX/MaxText stack (ADR-008),
  determinism & accuracy pinning, RL arena, post-training, fitness-gate domain.
- **`docs/specs/`** — model skeleton spec, environment spec, stage deltas.
- **`docs/research/`** — kernel/delta research notes from actual runs.
- **`ARCHITECTURE-SPINE.md`** — the invariants (AD-1…AD-8): mechanical verdicts, single run contract,
  privacy, GB10 resource limits, cost ceiling.
- **`CONSTRAINTS.yaml`** — 31+ mechanical fitness rules (C-001…C-031) checked without any LLM.
- **`docs/CASE-PASSPORT.ru.md`** — the original project passport: goals, scale decisions, budgets.

## Architecture at a glance

```
tokens ─┬─ KDA (18 layers)  short-conv + decay gates, 1M-context friendly, SWA window 128
        ├─ MLA (6 layers)   latent 512, top-k 512 index heads
        ├─ AttnRes blocks   every 12 layers
        └─ LatentMoE        latent 768, 12 routed (top-2) + 2 shared, QB monitoring
head:   tied embeddings, MTP-1 (weight 0.1), vocab 160k
vision: ViT patch 14, depth 12 → fused into the trunk
train:  BF16, Muon (Newton–Schulz, Polar Express) + AdamW, QAT-ready (MXFP4 fake-quant)
```

Why KDA: a 1M context does not fit into 128 GB of a single GB10 with a classic KV cache at any
quantization — see ADR-001/ADR-003.

## Frontier practices, attributed

Design decisions cite published reports and keep a fidelity matrix (`docs/FIDELITY-TO-K3.md`):
KDA/AttnRes/LatentMoE skeleton follows the published Kimi K3 design; router/bias/routed-scale
from DeepSeek-V3; Muon/Newton–Schulz precision and MTP self-speculative decoding from StepFun;
MoE routing pathologies, async CISPO recipe and eval-integrity practice from Poolside Laguna.
Our novelty lives in the data pipeline, post-training and the harness — not in re-deriving
published kernels.

## Quickstart

```bash
# pure-python harness tests (no JAX needed)
python3 -m pytest tools/tests -q

# model tests need JAX (CPU is enough for most)
python3 -m pip install -r env/requirements.txt
python3 -m pytest net/tests -q
```

`venv` note: tests and smoke scripts expect a JAX-enabled interpreter (see `net/README.md` for
CUDA `LD_LIBRARY_PATH` details).

## Status

Skeleton L3 pretrain pipeline: walking skeleton complete, staged training in progress.
Scale class L1 (≈$44k) is an open decision after skeleton results.

## License

MIT — see [LICENSE](LICENSE).
