"""Integrity tests for the model ladder (``net/model_ladder.py``, ADR-036).

The ladder is the single carrier of the four prototyping rungs, and its value
rests on one property: **architectural parity** — a low rung must differ from
the M3 target only in *size*, never in *mechanism*.  This module is what turns
that claim into a checked property:

* **T-ML-1** the registry carries exactly the four rungs M0..M3, in climb order;
* **T-ML-2** the low rungs are the *frozen* smoke presets of
  ``net/tests/conftest.py`` (production code may not import a test module, so
  the ladder carries its own copies and this test pins them together field by
  field); the target rung's lazy factory is callable and returns the full
  vocab config without counting its (heavy) parameters;
* **T-ML-3** every rung builds and no feature is switched off — KDA, MLA,
  LatentMoE (routed *and* shared experts), MTP, SiTi gates, AttnRes;
* **T-ML-4** the KDA:MLA composition is 3:1 on every rung (whole ``[K,K,K,M]``
  blocks, the shape ``validate_config`` pins);
* **T-ML-5** the pinned parameter estimate is re-measured, the honest count is
  in the rung's corridor, and an independent dim-based estimate agrees with the
  shape count (jax) — a drift of a factory dimension turns this red;
* **T-ML-6** an unknown rung name fails loudly and tells the caller what the
  ladder carries.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1]
CASE_DIR = TOOLS_DIR.parent
for _p in (str(TOOLS_DIR), str(CASE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from net.config import ModelConfig  # noqa: E402
from net.model_ladder import MODEL_LADDER, get_rung  # noqa: E402

#: Rungs whose parameter count is cheap to measure (the 1.01B target is not:
#: the test only checks that its factory builds, per the delta's contract).
NUMERIC_RUNGS = ("tiny", "small", "mid")


def _build(name: str) -> ModelConfig:
    """Build the rung's config through its factory (no overrides)."""
    return get_rung(name).factory()


def _assert_same_config(lhs: ModelConfig, rhs: ModelConfig, label: str) -> None:
    """Field-by-field equality with a diagnostic naming the first drift."""
    for f in dataclasses.fields(lhs):
        lv, rv = getattr(lhs, f.name), getattr(rhs, f.name)
        assert lv == rv, f"{label}: field {f.name!r} drifted: ladder={lv!r} canon={rv!r}"


def _layer_is_kda(index: int) -> bool:
    """The pinned ``[K,K,K,M]`` pattern: every fourth layer is MLA."""
    return index % 4 != 3


def _estimate_params(cfg: ModelConfig) -> int:
    """Independent parameter count from the config's dimensions.

    Hand-summed from the layer initialisers (``net/kda.py``, ``net/mla.py``,
    ``net/moe.py``, ``net/mlp.py``, ``net/mtp.py``, ``net/attnres.py``,
    ``net/vit.py``).  It is deliberately independent of ``net.model.param_count``
    (which traces the real initialiser through ``jax.eval_shape``): the two
    agreeing is the cross-check, and the estimate alone keeps T-ML-5 meaningful
    in an environment without jax.
    """
    from net.model import layer_kind

    hid = cfg.hidden
    heads, dk, dv = cfg.num_heads, cfg.kda_dk, cfg.kda_dv
    kernel, rank = cfg.kda_short_conv_kernel, cfg.kda_decay_rank
    latent, head_dim = cfg.mla_latent_dim, cfg.mla_head_dim
    index = cfg.mla_index_heads * cfg.mla_index_dim

    def kda_layer() -> int:
        n = hid * heads * dk * 2 + hid * heads * dv  # W_q, W_k, W_v
        n += heads * dv * hid + hid * hid  # W_o, W_g
        n += hid * heads + hid * rank + rank * heads * dk  # W_beta, W_a_down, W_a_up
        n += heads * dk + heads  # b_alpha, A
        n += kernel * heads * dk * 2 + kernel * heads * dv + 1  # conv_q/k/v, swa_logit
        if not cfg.swa_share_kda_projections:
            n += hid * heads * dk * 2 + hid * heads * dv
            n += kernel * heads * dk * 2 + kernel * heads * dv
        return n

    def mla_layer() -> int:
        n = hid * latent + hid * heads * head_dim + latent * heads * head_dim * 2
        n += heads * head_dim * hid + hid * hid  # W_o, W_g
        n += hid * heads * head_dim * 2  # W_swa_k, W_swa_v
        n += hid * index + latent * index + 1  # W_idx_q, W_idx_k, swa_logit
        return n

    def dense_mlp() -> int:
        return hid * cfg.mlp_intermediate * 2 + cfg.mlp_intermediate * hid

    def moe_layer() -> int:
        nr, ns = cfg.moe_num_routed, cfg.moe_num_shared
        lat, ei = cfg.moe_latent_dim, cfg.moe_expert_intermediate
        si = cfg.moe_shared_intermediate
        n = hid * lat + lat * hid  # W_down, W_up
        n += nr * lat * ei * 2 + nr * ei * lat  # expert_g/u/d
        n += ns * hid * si * 2 + ns * si * hid  # shared_g/u/d
        n += hid * nr + nr + lat  # router_w, router_b, norm
        return n

    total = cfg.vocab_size * hid  # tied embedding
    for i in range(cfg.num_layers):
        kind = layer_kind(cfg, i)
        total += 2 * hid  # norm_attn, norm_mlp
        total += kda_layer() if kind == "kda" else mla_layer()
        # ``net.model._layer_uses_dense_mlp``: a diagnostic dense-standard block
        # is dense by definition, every other layer follows the pinned prefix
        # (the *absolute* layer index, not the prefix-relative one).
        dense = kind == "dense-standard" or i < cfg.moe_dense_layers
        total += dense_mlp() if dense else moe_layer()
    total += cfg.num_layers * hid  # AttnRes pseudo-queries
    total += hid  # norm_final
    total += 2 * hid * hid + hid + kda_layer() + hid  # MTP W_f, norm_in, KDA
    total += dense_mlp() + hid + hid  # MTP MLP, norm_mlp, norm_out
    patch, vit_hidden, vit_mlp = cfg.vit_patch, cfg.vit_hidden, cfg.vit_mlp
    total += patch * patch * 3 * vit_hidden
    total += (cfg.image_size // patch) ** 2 * vit_hidden  # pos_embed
    total += cfg.vit_depth * (vit_hidden * 3 * vit_hidden + vit_hidden * vit_hidden)
    total += cfg.vit_depth * (vit_hidden * vit_mlp + vit_mlp * vit_hidden + 2 * vit_hidden)
    total += vit_hidden * hid + vit_hidden  # projector, norm
    return total


def _measure_params(cfg: ModelConfig) -> int | None:
    """The shape count (``net.model.param_count``), or ``None`` without jax."""
    try:
        from net.model import param_count
    except Exception:  # noqa: BLE001 — jax absent: the estimate carries the test
        return None
    return int(param_count(cfg))


# --------------------------------------------------------------------------- #
# T-ML-1: the ladder carries exactly the four rungs, in climb order.
# --------------------------------------------------------------------------- #


def test_ml1_registry_carries_four_rungs_in_climb_order() -> None:
    assert [r.name for r in MODEL_LADDER] == ["tiny", "small", "mid", "l3-full"]
    assert [r.level for r in MODEL_LADDER] == ["M0", "M1", "M2", "M3"]
    assert len({r.name for r in MODEL_LADDER}) == 4


def test_ml1_every_rung_declares_its_contract_fields() -> None:
    for rung in MODEL_LADDER:
        assert rung.budget_hint and rung.purpose, rung.name
        assert callable(rung.factory), rung.name
        assert rung.params_estimate is None or rung.params_estimate > 0, rung.name
        if rung.param_corridor is not None:
            lo, hi = rung.param_corridor
            assert 0 < lo < hi, rung.name


# --------------------------------------------------------------------------- #
# T-ML-2: the low rungs are the frozen conftest presets; M3 builds lazily.
# --------------------------------------------------------------------------- #


def test_ml2_tiny_and_small_match_the_frozen_conftest_presets() -> None:
    from net.tests import conftest

    _assert_same_config(_build("tiny"), conftest.tiny_config(), "tiny")
    _assert_same_config(_build("small"), conftest.small_config(), "small")


def test_ml2_l3_full_factory_builds_the_target_config_without_counting() -> None:
    cfg = _build("l3-full")
    assert isinstance(cfg, ModelConfig)
    # The target preset's vocabulary is the full tokenizer one — the smoke
    # presets substitute a tiny vocab, this is the config as declared.
    assert cfg.vocab_size > 1000
    assert (cfg.num_kda_layers, cfg.num_mla_layers) == (18, 6)
    assert cfg.num_layers == 24


def test_ml2_registry_import_does_not_pull_the_sft_runner() -> None:
    """The lazy contract: importing the registry must not import the runner.

    A fresh interpreter is used so the answer cannot depend on whichever test
    ran first — importing ``net.model_ladder`` alone must leave
    ``tools.run_sft_smoke`` out of ``sys.modules`` (the runner pulls the stage
    machinery: journal, tokenizer, checkpointing).
    """
    import subprocess

    code = (
        "import sys; sys.path.insert(0, %r); import net.model_ladder; "
        "print('tools.run_sft_smoke' in sys.modules)"
    ) % str(CASE_DIR)
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False", out.stdout + out.stderr
    # And the lazy factory really does produce the runner's config.
    assert get_rung("l3-full").factory().hidden == 1536


# --------------------------------------------------------------------------- #
# T-ML-3: each rung builds with the full feature set (no feature switched off).
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", [r.name for r in MODEL_LADDER])
def test_ml3_every_rung_builds_with_all_features_on(name: str) -> None:
    from net.config import validate_config

    cfg = _build(name)
    validate_config(cfg)
    assert cfg.num_kda_layers > 0, "KDA attention disabled"
    assert cfg.num_mla_layers > 0, "MLA attention disabled"
    assert cfg.moe_num_routed > 0, "routed experts disabled"
    assert cfg.moe_num_shared > 0, "shared experts disabled"
    assert cfg.moe_top_k == 2, "MoE topology is pinned at top-2"
    assert cfg.mtp_layers > 0, "MTP head disabled"
    assert cfg.attnres_blocks > 0, "AttnRes disabled"
    assert cfg.siti_beta_gate > 0 and cfg.siti_beta_up > 0, "SiTi gates disabled"
    assert cfg.mlp_intermediate > 0 and cfg.moe_latent_dim > 0
    assert cfg.dense_standard_layers == 0, "a rung is business path, not the leg"


# --------------------------------------------------------------------------- #
# T-ML-4: the KDA:MLA composition is 3:1 on every rung.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", [r.name for r in MODEL_LADDER])
def test_ml4_composition_is_three_to_one_on_every_rung(name: str) -> None:
    cfg = _build(name)
    assert cfg.num_kda_layers == 3 * cfg.num_mla_layers, (
        f"{name}: composition {cfg.num_kda_layers} KDA : {cfg.num_mla_layers} MLA "
        "is not 3:1"
    )
    assert cfg.num_kda_layers + cfg.num_mla_layers == cfg.num_layers
    assert cfg.num_layers % 4 == 0, "the tail must be whole [K,K,K,M] blocks"


# --------------------------------------------------------------------------- #
# T-ML-5: the parameter estimate is honest and in the rung's corridor.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", NUMERIC_RUNGS)
def test_ml5_parameter_count_is_measured_and_in_the_corridor(name: str) -> None:
    rung = get_rung(name)
    cfg = _build(name)

    estimated = _estimate_params(cfg)
    measured = _measure_params(cfg)
    if measured is not None:
        assert measured == estimated, (
            f"{name}: the dim-based estimate {estimated:,} disagrees with the "
            f"shape count {measured:,} — a layer initialiser changed shape"
        )

    count = measured if measured is not None else estimated
    assert count == rung.params_estimate, (
        f"{name}: the pinned estimate {rung.params_estimate:,} is stale; the "
        f"factory now counts {count:,}"
    )
    lo, hi = rung.param_corridor
    assert lo <= count <= hi, (
        f"{name}: {count:,} params outside the corridor [{lo:,}, {hi:,}]"
    )


def test_ml5_the_ladder_is_a_monotone_climb_in_scale() -> None:
    counts = [_estimate_params(_build(n)) for n in NUMERIC_RUNGS]
    assert counts == sorted(counts) and len(set(counts)) == len(counts)
    # M3's pinned target is far above the top prototype rung — the ladder is
    # about paying for the question, not about approaching the target cheaply.
    l3 = get_rung("l3-full")
    assert l3.params_estimate > counts[-1]
    assert l3.param_corridor is None, "the target rung is outside corridor control"


def test_ml5_mid_lands_in_the_declared_m2_band() -> None:
    count = _estimate_params(_build("mid"))
    assert 30_000_000 <= count <= 60_000_000, (
        f"mid={count:,} left the ~30-60M band ADR-036 declares for M2"
    )


# --------------------------------------------------------------------------- #
# T-ML-6: an unknown rung name fails loudly and lists what the ladder carries.
# --------------------------------------------------------------------------- #


def test_ml6_unknown_rung_raises_key_error_listing_the_ladder() -> None:
    with pytest.raises(KeyError) as excinfo:
        get_rung("nonsense")
    message = str(excinfo.value)
    for rung in MODEL_LADDER:
        assert rung.name in message
    assert "nonsense" in message
    assert "unknown ladder rung" in message
