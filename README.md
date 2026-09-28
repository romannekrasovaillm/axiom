<div align="center">
  <img src="docs/assets/banner.svg" alt="AXIOM — LLM from scratch on JAX" width="100%"/>

  [![A4 gate](https://img.shields.io/badge/A4_gate-4%2F5_stages_executed-ff9f1c?style=flat-square)](docs/RESULTS-2026-09-27.ru.md)
  [![tests](https://img.shields.io/badge/tests-374_passed-2ea44f?style=flat-square)](net/tests/)
  [![data](https://img.shields.io/badge/data-101k%20domain%20records-8250df?style=flat-square)](docs/datasets/axiom-domain-ds-v1-card.md)
  [![JAX](https://img.shields.io/badge/JAX-0.10.2-2b6cb0?style=flat-square)](https://github.com/jax-ml/jax)
  [![Python](https://img.shields.io/badge/Python-3.11-3776ab?style=flat-square)](https://www.python.org/)
  [![license](https://img.shields.io/badge/license-MIT-2ea44f?style=flat-square)](LICENSE)
  [![reports](https://img.shields.io/badge/reports-live_on_Pages-8250df?style=flat-square)](https://romannekrasovaillm.github.io/axiom/)

  **RU:** собственная нейросеть с нуля на JAX — претрейн, SFT, RL, агентные способности и свой
  конвейер обучения. Не адаптация чужих весов. Скелет **L3** (~1B MoE / 20B токенов) обучается
  целиком на одной машине (NVIDIA GB10 / DGX Spark). Каждое решение зафиксировано ADR и проверяется
  механическими fitness-правилами (`CONSTRAINTS.yaml`) — без LLM-судей.
</div>

## ✨ Why this is interesting

- **Built from scratch, verified mechanically.** Hybrid **KDA** linear attention + **MLA** + **LatentMoE** + **MTP-1**, BF16 pretrain, Muon/Newton–Schulz — and every architectural claim is pinned by a behavioural fitness rule, not prose.
- **A dishonest green gate is worse than an honest red one.** The A4 run manifest carries the *composition* of pipeline stages and a *computed* `pipeline_complete` — a partial run cannot close the gate, forge hashes are rejected by construction.
- **The full pipeline, proven end-to-end on a single box** (≈$200 of compute): checkpoint → inference → RL environment → SFT → RL, with 374 acceptance tests, deterministic XLA profiling and run manifests with sha256 pinning.
- **Live architecture-as-code**: interactive [Archify diagrams](https://romannekrasovaillm.github.io/axiom/) rendered from a typed JSON IR — pipeline status and the data map update with every push.
- **A curated domain dataset**: verified agent episodes (mechanical verdicts, fail-closed), 101k records of concepts/distillates/skills from a private library — scrubbed, deduplicated, carded.

## 📊 Live reports (GitHub Pages)

- [Pipeline A4: stages → manifest → verify](https://romannekrasovaillm.github.io/axiom/diagrams/a4-gate-flow.html) — interactive Archify diagram
- [Data map: pretrain (public) vs post-train (private)](https://romannekrasovaillm.github.io/axiom/diagrams/axiom-data-map.html)
- [Results report (RU)](https://romannekrasovaillm.github.io/axiom/RESULTS-2026-09-27.ru.html) · [Open questions & forks](https://romannekrasovaillm.github.io/axiom/OPEN-QUESTIONS.md)

## What is here

- **`net/`** — the model, in JAX: hybrid **KDA** linear attention (DeltaNet-family with short
  conv + decay gates) + **MLA** layers + **AttnRes** + **LatentMoE** (latent 0.5×hidden,
  12 routed + 2 shared experts, top-2, aux-free QB monitoring) + **SiTU-GLU**, **MTP-1** head,
  minimal vision path (ViT). BF16 pretrain, Muon + Newton–Schulz (Polar Express) optimizer.
- **`env/`** — the RL environment: task generation, verifier, reward (binary, mechanical), net executor.
- **`tools/`** — training/eval harness: SFT/RL smoke runs, precision pinning checks, A4 pipeline,
  manifest and budget gates, **dataset pipelines** (`tools/prep_pretrain/`, `tools/axiom_ds/`).
- **`docs/adr/`** — architecture decision records (RU): scale class, JAX/MaxText stack (ADR-008),
  determinism & accuracy pinning, RL arena, post-training, **long-context mechanics (ADR-009/012/018),
  domain dataset (ADR-020), pretrain mix (ADR-021), compute stack (ADR-017)**.
- **`docs/specs/`** — model skeleton spec, environment spec, stage deltas.
- **`docs/research/`** — kernel/delta research notes from actual runs.
- **`docs/datasets/`** — dataset cards (machine-readable + human): composition, sha256, provenance.
- **`ARCHITECTURE-SPINE.md`** — the invariants (AD-1…AD-11): mechanical verdicts, single run contract,
  snapshot pinning, privacy, GB10 resource limits, cost ceiling, subject-matter guards.

Design decisions cite published reports and keep a fidelity matrix (`docs/FIDELITY-TO-K3.md`):
KDA/AttnRes/LatentMoE skeleton follows the published Kimi K3 design; router/bias/routed-scale
from DeepSeek-V3; Muon/Newton–Schulz precision and MTP self-speculative decoding from StepFun;
MoE routing pathologies, async CISPO recipe and eval-integrity practice from Poolside Laguna.
Our novelty lives in the data pipeline, post-training and the harness — not in re-deriving
published kernels.

## 🚀 Quickstart

```bash
# pure-python harness tests (no JAX needed)
python3 -m pytest tools/tests -q

# model tests need JAX (CPU is enough for most)
python3 -m pip install -r env/requirements.txt
python3 -m pytest net/tests -q
```

`venv` note: tests and smoke scripts expect a JAX-enabled interpreter (see `net/README.md` for
CUDA `LD_LIBRARY_PATH` details).

## 🤝 How to help

Pick a fork from [`docs/OPEN-QUESTIONS.md`](docs/OPEN-QUESTIONS.md) — every item names the owner,
the options and what it unblocks:

- **Operator on GB10/DGX Spark** — run the smoke pipeline + `spark_inference` on the stand; closes the A4 gate (O-1).
- **Dataset curation** — review the 1 331 skills outside the domain allowlist (V-2); legacy corpus conversion (V-3).
- **Pretrain loop** — the streaming loader + resume path (V-4) before the 20B-token run (≈$200).
- **Long-context measurement** — 64K gap benchmarks for the block-wise token merging flag (D-4).
- **Docs & diagrams** — keep the live Pages and the fidelity matrix honest.

Rules of engagement: no weakening of fitness rules (anti-weakening, ADR-011), no fabricated
evidence (fail-closed everywhere), every decision as an ADR **before** implementation.

## Status

Skeleton L3 pretrain pipeline: walking skeleton complete (4/5 A4 stages executed with
mechanical evidence), staged training in progress. Scale class L1 (≈$44k) is an open decision
after skeleton results. Pretrain corpus W/C shards downloading; domain dataset v1 carded.

## License

MIT — see [LICENSE](LICENSE).
