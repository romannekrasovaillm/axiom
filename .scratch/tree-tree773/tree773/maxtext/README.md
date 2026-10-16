# MaxText integration (Phase 2)

ADR-008 pins the stack to JAX with a MaxText fork as the training backbone.  At
the walking-skeleton stage the core layers are implemented in pure JAX under
`net/` (no Flax/NNX dependency) so the KDA/Gated-MLA/SiTU-GLU mechanics can be
verified cheaply against the oracles before they are ported to the fork.  This
directory records the mapping; the fork itself (vendoring
`github.com/google/maxtext`) is not committed into the working tree — only the
porting contract below is.

## Layer -> MaxText mapping

| Our module (`net/`) | MaxText target | Notes |
|---|---|---|
| `kda.KDAParams` | custom `nnx.Module` (new) | KDA is not in MaxText; our JAX reference + the chunked form become the reference for a Pallas kernel at L1 |
| `mla.MLAParams` | MaxText K2 MLA module | gate added per K3 (full-rank `W_g`), NoPE (no RoPE) |
| `mlp.MLPParams` (SiTU-GLU) | custom activation in the MLP module | dense at skeleton; LatentMoE is the L1 form |
| `attnres.AttnResParams` | custom depth-mixing hook | additive AttnRes, full form (L=24) |
| `mtp.MTPParams` | MaxText MTP layer | MaxText has MTP since 07/2025; shared tied head |
| `vit.ViTParams` | MaxText native multimodal vision tower | ViT-S minimum in the skeleton |
| `optimizer.make_step` | MaxText Muon / MuonClip | per-head orthogonalisation is our patch |
| `quant.fake_quant_mxfp4` | QAT hook on the SFT stage | our JAX fake-quant (AQT unverified, ADR-008) |
| `checkpoint` | Orbax (already used) | manifest hash + symlink to gb10-shared |

## What stays in pure JAX for the skeleton

* Per-head Muon + weight clipping, cosine + 1% warmup, weight decay 0.1
  (`optimizer.py`) — ported into the fork's optimizer registration.
* Context curriculum 8K -> 64K, 90/10 split (`curriculum.py`).
* QAT fake-quant MXFP4 with a pretrain-off flag (`quant.py`).
* Byte-level BPE 160K tokenizer (`tokenizer.py`).
* Data cards + shard hashes + symlink mounting (`data.py`).

## What the fork adds (deferred to the full run)

Sharding (TP/PP/EP/CP) via MaxText configuration rather than our code, the
Grain data loader over the 4-domain shards, and the Pallas KDA kernels — the
skeleton tolerates low MFU per ADR-008 (Negative).
