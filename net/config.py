"""Model configuration for the L3 walking skeleton.

The config is the single source of truth for the skeleton's architecture and
is pinned in ``net/config.json`` (see MODEL-L3-SKELETON.md section 1).  The
parameter budget test reads the *actual* parameter count back from here, so
``ModelConfig`` must stay in sync with the layer implementations.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

#: Key of the declared block-wise token merging switch (ADR-018) in the config.
BLOCK_MERGE_KEY = "mla_block_merge"

#: The case's declarative config — the file the reader below reads.  It is the
#: *switch* (spine AD-9: "конфиг первичен"): turning a declared mechanism on is
#: an edit of this file, not of the code path that consumes it.
CONFIG_PATH = Path(__file__).with_name("config.json")


@dataclass(frozen=True)
class BlockMergeConfig:
    """Block-wise token merging of the MLA history (ADR-018, Step-5 borrowing).

    ``enabled`` is the declarative switch and ``block`` the number of records
    aggregated into one merged record (mean pooling, see
    :func:`net.attn_sparse.mean_blocks`): the indexer scores one merged record
    per block and attention reads the merged records instead of the raw ones.

    Pinned **off** until the 64K gap measurement is green (ADR-018 p. 3: the
    mechanism is accepted through criteria 1-7 of ADR-009 and only then does the
    default flip).  Both fields are numbers, not code: ``net/attn_sparse.py``
    reads them (guard C-035), so turning the mechanism on is a config edit.
    """

    enabled: bool = False
    block: int = 16


@dataclass(frozen=True)
class ModelConfig:
    # --- tokens / embeddings -------------------------------------------------
    vocab_size: int = 160_000
    hidden: int = 1536
    tie_embeddings: bool = True

    # --- layer composition (3:1 ratio, MODEL-L3-SKELETON.md section 1) ------
    num_layers: int = 24
    num_kda_layers: int = 18
    num_mla_layers: int = 6
    num_heads: int = 12
    head_dim: int = 128  # dk == dv == head_dim for the skeleton

    # --- KDA (kda-formulas.md section 1) ------------------------------------
    kda_dk: int = 128
    kda_dv: int = 128
    kda_decay_rank: int = 128          # low-rank decay-logit projection
    kda_short_conv_kernel: int = 4     # ShortConv depthwise kernel size
    kda_g_min: float = -5.0            # lower bound of the log-decay (Eq. 5)

    # --- Gated MLA (kda-formulas.md section 2) ------------------------------
    mla_latent_dim: int = 512          # compressed KV latent c_t = W_c x_t
    mla_head_dim: int = 128
    mla_top_k: int = 512               # sparse selection width (min(512, T))
    mla_index_heads: int = 1           # light indexer heads (scaled to 12 Q heads)
    mla_index_dim: int = 128           # light indexer head dim

    # --- long-context delta v1.5 (ADR-009: SWA window, sparse, FP4 latent) ---
    swa_window: int = 128              # local sliding-window size (V4.1-Flash 2.2)
    swa_share_kda_projections: bool = True  # reuse KDA q/k/v for the window (ours)
    qat_kv_enabled: bool = False       # FP4 fake-quant of the MLA latent (SFT+)
    attn_dense_reference: bool = True  # dense oracle / A-B switch (D4; A4 gate)
    indexer_seed: int = 0              # pinned determinism seed of the indexer

    # --- hierarchical sparse indexer (ADR-012: shared candidate pool) --------
    # Level 1 (mode ``full``) publishes a pool of ``mla_pool_size`` record blocks
    # of ``mla_pool_block`` records; level 2 (``reindex``/``reuse``) selects its
    # ``top_k`` inside that pool instead of over the whole causal prefix.  The
    # layout is declared per MLA layer, in layer order.  ``mla_pool_size = 0``
    # disables the pool (every MLA layer then behaves as ADR-009 D2 ``full``).
    mla_pool_block: int = 64           # records per pool block (V4.1-Flash 2.3.2)
    mla_pool_size: int = 0             # m: blocks kept in the pool (0 = pool off)
    mla_layer_modes: tuple[str, ...] = ()  # per MLA layer: full | reindex | reuse

    # --- block-wise token merging (ADR-018: Step-5-Preview model card) --------
    # The history is aggregated into blocks of ``block`` records before the
    # sparse selection: the indexer scores one merged record per block and
    # attention reads the merged records, so the selection stops paying per
    # record.  Declared as ``mla_block_merge`` in ``net/config.json`` (spine
    # AD-9, guard C-035) and read by ``net/attn_sparse.py`` — the *file* is the
    # switch, so enabling it is not a code edit.  Pinned off until the 64K gap
    # measurement is green (ADR-018 p. 3); with the flag off the unmerged path
    # is bit-for-bit unchanged.
    mla_block_merge: BlockMergeConfig = BlockMergeConfig()

    # --- sparse assembly (criterion 16 delta: fused gather + attention) ------
    # ``True`` assembles the union attention without materialising the gathered
    # ``(B, Q, k_eff, H, D)`` keys/values: the duplicate mask is a scatter+lookup
    # in a per-position mark rather than a ``(Q, W, k_eff)`` compare, and the
    # softmax denominator is applied to each branch's gathered values in place
    # instead of to a concatenated ``(B, Q, k_eff + W, H, D)`` tensor.  Same
    # arithmetic, same selection, same widths — the flag only selects between the
    # fused assembly and the verbatim unfused one, so the two can be A/B-timed
    # and cross-checked (``net/tests/test_13_*``).
    attn_fused_assembly: bool = True

    # --- exact top-k selection (criterion 16 delta: tiled merge) -------------
    # ``True`` selects the sparse records with ``net/topk_exact.topk_indices``:
    # the same records as ``jax.lax.top_k`` (whose GPU lowering is a full bitonic
    # sort of the whole row plus an iota payload), computed by tiling the row,
    # sorting the tiles and merging them pairwise while keeping the top k at each
    # merge.  Exact by construction — no approximation, no threshold, no change
    # to ``mla_top_k``, the window, the pool layout or the selection semantics.
    # Pinned to ``False``: the tiled kernel is *exact but not faster* on this
    # backend — in the MLA stack it is ~15% slower than ``jax.lax.top_k``
    # (the tiling multiplies the sort/marshalling kernels per scan step while
    # the selection is not on the stack's critical path), see
    # ``docs/research/topk-exact-delta-2026-09-13.md``.  ``jax.lax.top_k`` stays
    # the pinned leg; ``True`` is the A/B leg that the delta's tests exercise.
    attn_topk_exact: bool = False

    # --- SiTU-GLU MLP (kda-formulas.md section 4.2) -------------------------
    mlp_intermediate: int = 4096       # dense layer-0 MLP
    siti_beta_gate: float = 4.0        # beta_1, gate branch
    siti_beta_up: float = 25.0         # beta_2, up branch

    # --- Stable LatentMoE (kda-formulas.md section 4, MODEL-L3-SKELETON.md 1) -
    moe_dense_layers: int = 1          # leading dense SiTU-GLU layers (K3: 1 of 93)
    moe_latent_dim: int = 768          # latent width l = 0.5 x hidden (as 3584/7168)
    moe_num_routed: int = 12           # routed experts (base 896 — scale-down, ours)
    moe_num_shared: int = 2            # shared experts Ns (base: 2, :472)
    moe_top_k: int = 2                 # active routed experts per token (base 16)
    moe_expert_intermediate: int = 384  # SiTU-GLU intermediate of a routed expert
    moe_shared_intermediate: int = 384  # SiTU-GLU intermediate of the shared path
    qb_weight: float = 0.01            # weight of the QB auxiliary loss (ours)
    routing_seed: int = 0              # pinned seed for routing determinism tests

    # --- MTP (kda-formulas.md / MODEL-L3-SKELETON.md section 2.6) -----------
    mtp_layers: int = 1
    mtp_loss_weight: float = 0.1

    # --- AttnRes (kda-formulas.md section 3) --------------------------------
    attnres_blocks: int = 2            # N blocks
    attnres_block_size: int = 12       # S = L/N layers per block

    # --- vision (ViT-S class, ~22M, patch 14) --------------------------------
    vit_patch: int = 14
    vit_hidden: int = 384
    vit_depth: int = 12
    vit_heads: int = 6
    vit_mlp: int = 1536
    image_size: int = 224

    # --- precision / curriculum ---------------------------------------------
    pretrain_dtype: str = "bf16"
    context_curriculum: tuple[int, ...] = (8192, 65536)
    curriculum_split: tuple[float, float] = (0.90, 0.10)  # 90/10 token budget

    # --- optimizer (MODEL-L3-SKELETON.md section 3) --------------------------
    weight_decay: float = 0.1
    warmup_ratio: float = 0.01
    lr_schedule: str = "cosine"

    # QAT is enabled for the SFT stage only (MODEL-L3-SKELETON.md section 3).
    qat_enabled: bool = False

    # --- grad-checkpointing granularity (D-8; spine AD-9/C-035 form) ---------
    # ``none`` builds the training graph as before; ``per_layer`` wraps every
    # backbone layer of ``net.model.forward`` in ``jax.checkpoint`` so the
    # backward pass recomputes the layer from its inputs instead of keeping its
    # attention/MoE intermediates (the coarse wrap of the whole ``compute_loss``
    # in ``net/train_loop.py`` does not cut the request — D-8, 972 GiB at
    # T=8192).  Declared in ``net/config.json`` as ``grad_ckpt_policy``; the
    # pretrain preset ``l3-full`` carries ``per_layer``.  Read by ``net/model.py``
    # (``compute_loss`` -> ``forward``); the schema default stays ``none`` so a
    # config built in code (tests, smokes) keeps the pre-D-8 graph.
    grad_ckpt_policy: str = "none"

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, ensure_ascii=False)


def default_config() -> ModelConfig:
    return ModelConfig()


def validate_config(cfg: ModelConfig) -> None:
    """Check structural invariants the layers rely on.

    The KDA output gate multiplies the per-token gate (``hidden``) by the
    flattened recurrent output (``H * dv``), and MLA concatenates ``H * dq``
    heads back to ``hidden``, so ``num_heads * head_dim == hidden`` and the
    per-head dims must equal ``head_dim``.
    """
    assert cfg.num_heads * cfg.head_dim == cfg.hidden, "num_heads * head_dim must equal hidden"
    assert cfg.kda_dk == cfg.kda_dv == cfg.head_dim, "kda_dk == kda_dv == head_dim"
    assert cfg.mla_head_dim == cfg.head_dim, "mla_head_dim == head_dim"
    assert cfg.num_kda_layers + cfg.num_mla_layers == cfg.num_layers
    assert cfg.num_layers % 4 == 0, "layers must form whole [K,K,K,M] blocks"
    assert 1 <= cfg.moe_dense_layers <= cfg.num_layers
    assert 1 <= cfg.moe_top_k <= cfg.moe_num_routed
    assert cfg.moe_latent_dim > 0 and cfg.moe_expert_intermediate > 0
    assert cfg.moe_num_shared >= 1 and cfg.moe_shared_intermediate > 0
    assert cfg.swa_window >= 0, "swa_window must be >= 0 (0 disables the window branch)"
    assert cfg.mla_top_k >= 1, "mla_top_k must be >= 1"
    assert cfg.mla_index_heads >= 1 and cfg.mla_index_dim > 0

    # Grad-checkpointing granularity (D-8): the declared policy has to be one
    # the backbone loop implements — an unknown value would silently fall back
    # to no remat, which is exactly the OOM the field exists to prevent.
    assert cfg.grad_ckpt_policy in ("none", "per_layer"), (
        "grad_ckpt_policy must be none|per_layer, got "
        f"{cfg.grad_ckpt_policy!r}"
    )

    # Hierarchical pool (ADR-012): the modes are declared, and a consumer
    # (``reindex``/``reuse``) can only run after a builder (``full``).
    modes = tuple(cfg.mla_layer_modes)
    assert cfg.mla_pool_block >= 1, "mla_pool_block must be >= 1"
    assert cfg.mla_pool_size >= 0, "mla_pool_size must be >= 0 (0 disables the pool)"
    assert all(mode in ("full", "reindex", "reuse") for mode in modes), (
        f"mla_layer_modes entries must be full|reindex|reuse, got {modes}"
    )
    assert len(modes) <= cfg.num_mla_layers, (
        f"mla_layer_modes declares {len(modes)} modes for {cfg.num_mla_layers} MLA layers"
    )
    if cfg.mla_pool_size > 0 and any(mode != "full" for mode in modes):
        assert modes[0] == "full", (
            "the first MLA layer must be 'full': it is the pool builder the "
            "reindex/reuse layers consume (ADR-012)"
        )

    # Block-wise token merging (ADR-018): the block is a width, so it is a
    # positive integer; and when the mechanism is on, the window branch must
    # cover the query's own partially filled block — the eligible blocks are the
    # complete ones *before* the query, so a window narrower than the block
    # would leave up to ``block - 1`` records before it unattended.
    merge = cfg.mla_block_merge
    assert isinstance(merge, BlockMergeConfig), (
        "mla_block_merge must be a BlockMergeConfig (enabled/block), got "
        f"{type(merge).__name__}"
    )
    assert merge.block >= 1, "mla_block_merge.block must be >= 1 (records per block)"
    if merge.enabled:
        assert cfg.swa_window >= merge.block, (
            f"mla_block_merge: swa_window={cfg.swa_window} must cover the block "
            f"({merge.block}) — the intra-block tail is the window's job (ADR-018)"
        )


@lru_cache(maxsize=8)
def _read_declared(path: str, mtime_ns: int, size: int) -> dict:
    """Parse a declared config, cached by ``(path, mtime, size)``.

    The cache key carries the file's stat, so any edit invalidates the entry: a
    test, or the architect flipping the switch, is never served a stale value.
    A broken or absent file reads as ``{}`` — the caller decides what that means.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def declared_block_merge(path: str | Path | None = None) -> int:
    """Records per merged block declared in the config (``0`` = mechanism off).

    ADR-018 declares block-wise token merging as ``mla_block_merge:
    {enabled, block}`` in ``net/config.json``; this reader is what makes the
    *declaration* the switch (spine AD-9, guard C-035) instead of an edit of the
    attention path.  ``net/attn_sparse.py`` consumes it as the default width of
    its ``block_merge`` parameter, so the flag is flipped in the file — with the
    flag off the unmerged path is bit-for-bit what it was.

    The reader lives here, in the schema module, rather than next to the
    attention primitives: this module imports no numerics, so the guard that
    compares declaration and code can run on a plain ``python3`` (no jax), and
    "the config is primary" has a single reader to point at.  It never guesses a
    width: an absent file, an absent field, ``enabled: false``, a missing or
    non-integer ``block``, or ``block < 1`` all read as ``0``.  A declaration
    that *says* it is enabled but carries a broken width is reported as a broken
    declaration by the guard, and ``validate_config`` asserts the same for a
    config loaded through the schema.
    """
    candidate = Path(path) if path is not None else CONFIG_PATH
    try:
        stat = candidate.stat()
    except OSError:
        return 0
    declared = _read_declared(str(candidate), stat.st_mtime_ns, stat.st_size).get(
        BLOCK_MERGE_KEY
    )
    if not isinstance(declared, dict) or not bool(declared.get("enabled", False)):
        return 0
    block = declared.get("block")
    if not isinstance(block, int) or isinstance(block, bool) or block < 1:
        return 0
    return block


def load_config(path: str | Path = "config.json") -> ModelConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    known = {fld.name for fld in dataclasses.fields(ModelConfig)}
    filtered = {k: v for k, v in data.items() if k in known}
    # JSON has no tuples: the declared per-layer mode layout comes back as a
    # list, pin it back to the immutable form the layers read.
    for name in ("context_curriculum", "curriculum_split", "mla_layer_modes"):
        if name in filtered and isinstance(filtered[name], list):
            filtered[name] = tuple(filtered[name])
    # JSON has no nested records either: ``mla_block_merge`` is declared as an
    # object (ADR-018), and the reader (``net/attn_sparse.py``) reads the file,
    # so both the schema object and the raw declaration must agree.  A malformed
    # declaration raises instead of falling back to the default width: silently
    # substituting a block size is exactly the guessing the flag exists to avoid.
    declared_merge = filtered.get("mla_block_merge")
    if isinstance(declared_merge, dict):
        enabled = declared_merge.get("enabled", False)
        block = declared_merge.get("block", BlockMergeConfig.block)
        if not isinstance(enabled, bool):
            raise ValueError(f"mla_block_merge.enabled must be true/false, got {enabled!r}")
        if not isinstance(block, int) or isinstance(block, bool):
            raise ValueError(f"mla_block_merge.block must be an integer, got {block!r}")
        filtered["mla_block_merge"] = BlockMergeConfig(enabled=enabled, block=block)
    return ModelConfig(**filtered)


def save_config(cfg: ModelConfig, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(cfg.to_json())
