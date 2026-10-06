"""VERIFICATION-LEG revision (E-4.1) — the diagnostic ``dense-standard`` layer.

The verification leg separates *pipeline* bugs from the *price of the KDA/MLA
architecture* by running the same pretrain pipeline on a GPT-2-class **dense**
config (``docs/specs/VERIFICATION-LEG.ru.md``, decision of 06.10 on escalation
E-4).  The dense arm needs a layer type the skeleton does not have: a standard
block — dense attention through the **existing** oracle of
``net/mla.py:_dense_apply`` plus a dense SiTU-GLU MLP — declared as
``layer_composition["dense-standard"] = N`` and built by ``net/model.py``.

What this module pins mechanically:

* **the business path is unchanged** — a config without the declared prefix
  (``net/config.json``, the pilot) builds the pinned ``[K,K,K,M]`` composition
  layer for layer, with the declared parameter counts;
* **both legs exist and are the two files the spec names** — ``config-dense124m``
  is 12 dense-standard layers, ``config-arch124m`` is 9 KDA + 3 MLA at the same
  width/depth;
* **the layer actually runs** — a forward on a mini sequence (CPU) produces
  finite logits/loss, and the dense-standard block ignores the
  ``attn_dense_reference`` switch by construction (it is selected by layer kind,
  so the arm cannot silently become sparse);
* **the declaration is guarded** — a malformed/mismatching ``layer_composition``
  is an error, not a silent fallback to the production composition.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import pytest

from net import model
from net.config import ModelConfig, load_config, validate_config
from net.kda import KDAParams
from net.mla import MLAParams
from net.mlp import MLPParams

NET_DIR = Path(__file__).resolve().parents[1]
DENSE = NET_DIR / "config-dense124m.json"
ARCH = NET_DIR / "config-arch124m.json"
PILOT = NET_DIR / "config.json"


def _pattern(n: int) -> tuple[str, ...]:
    """The pinned ``[K,K,K,M]`` pattern for ``n`` layers (production shape)."""
    return tuple("mla" if i % 4 == 3 else "kda" for i in range(n))


def _declared(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _mini(cfg: ModelConfig, **overrides) -> ModelConfig:
    """The declared config shrunk to CPU-smoke sizes, composition untouched.

    The composition (``dense_standard_layers``, ``num_layers``, the
    ``[K,K,K,M]`` tail) is what is under test and is never overridden here; only
    the widths are reduced so the forward runs in seconds on a CPU.  The
    *declared* file's own shape count is still checked separately
    (``tools/tests/test_config_dense124m.py`` / ``test_config_arch124m.py``).
    """
    shrink = dict(
        vocab_size=256,
        hidden=64,
        num_heads=4,
        head_dim=16,
        kda_dk=16,
        kda_dv=16,
        kda_decay_rank=16,
        mla_latent_dim=32,
        mla_head_dim=16,
        mlp_intermediate=128,
        moe_latent_dim=32,
        vit_hidden=32,
        vit_depth=2,
        vit_heads=2,
        vit_mlp=64,
        image_size=56,
        attnres_block_size=2,
    )
    shrink.update(overrides)
    return dataclasses.replace(cfg, **shrink)


# ---------------------------------------------------------------------------
# business path: a config without the prefix builds exactly what it did before
# ---------------------------------------------------------------------------


def test_default_config_keeps_the_pinned_pattern() -> None:
    cfg = ModelConfig()
    assert cfg.dense_standard_layers == 0
    assert model.layer_kinds(cfg) == _pattern(cfg.num_layers)
    kinds = model.layer_kinds(cfg)
    assert kinds.count("kda") == 18
    assert kinds.count("mla") == 6
    assert kinds.count("dense-standard") == 0


def test_layer_kind_matches_the_legacy_helpers() -> None:
    """``layer_kind`` is the single definition; the old helpers still agree."""
    cfg = ModelConfig()
    for i in range(cfg.num_layers):
        assert model.layer_kind(cfg, i) == ("kda" if model._layer_is_kda(i) else "mla")


def test_pilot_config_builds_with_unchanged_composition() -> None:
    cfg = load_config(PILOT)
    validate_config(cfg)
    assert cfg.dense_standard_layers == 0
    kinds = model.layer_kinds(cfg)
    assert kinds == _pattern(24)
    assert (kinds.count("kda"), kinds.count("mla")) == (18, 6)
    # "Builds" is the shape-only trace: init_params runs, and the count is the
    # one pinned in the pilot config (no allocation at 1B scale).
    declared = _declared(PILOT)
    assert model.param_count(cfg) == declared["actual_param_count"]
    assert model.active_param_count(cfg) == declared["active_params_per_token"]


def test_moe_split_unchanged_for_the_pilot() -> None:
    assert model.dense_moe_split(ModelConfig()) == (1, 23)


# ---------------------------------------------------------------------------
# the two arms the spec names
# ---------------------------------------------------------------------------


def test_dense_config_is_all_dense_standard() -> None:
    cfg = load_config(DENSE)
    validate_config(cfg)
    data = _declared(DENSE)
    assert cfg.dense_standard_layers == 12
    assert model.layer_kinds(cfg) == ("dense-standard",) * 12
    assert (cfg.num_kda_layers, cfg.num_mla_layers) == (0, 0)
    assert (cfg.moe_dense_layers, cfg.dense_standard_layers) == (12, 12)
    # Spec values (VERIFICATION-LEG.ru.md, "Конфигурация dense-reference").
    assert cfg.num_layers == 12
    assert cfg.num_heads == 12
    assert cfg.hidden == cfg.num_heads * cfg.head_dim == 768
    assert cfg.head_dim == 64
    assert cfg.mlp_intermediate == 3072
    assert cfg.vocab_size == 160000
    assert cfg.attn_dense_reference is True
    # The sparse-path declarations are off: no MLA layer is built at all.
    assert tuple(cfg.mla_layer_modes) == ()
    assert cfg.mla_pool_size == 0
    assert cfg.mla_block_merge.enabled is False
    assert data["layer_composition"]["dense-standard"] == 12


def test_arch_config_is_the_pilot_composition_at_124m_width() -> None:
    cfg = load_config(ARCH)
    validate_config(cfg)
    assert cfg.dense_standard_layers == 0
    assert model.layer_kinds(cfg) == _pattern(12)
    kinds = model.layer_kinds(cfg)
    assert (kinds.count("kda"), kinds.count("mla")) == (9, 3)
    # Same width/depth as the dense arm — the A/B differs in composition only.
    dense = load_config(DENSE)
    assert (cfg.num_layers, cfg.hidden, cfg.num_heads, cfg.head_dim) == (
        dense.num_layers, dense.hidden, dense.num_heads, dense.head_dim
    )
    assert cfg.mlp_intermediate == dense.mlp_intermediate
    assert cfg.vocab_size == dense.vocab_size
    # ``tokenizer_hash`` is declaration metadata (not a schema field) — compare
    # the two files' pins so the A/B cannot be tokenized differently.
    assert _declared(ARCH)["tokenizer_hash"] == _declared(DENSE)["tokenizer_hash"]


# ---------------------------------------------------------------------------
# the dense-standard block actually runs
# ---------------------------------------------------------------------------


def _run_forward(cfg: ModelConfig, *, T: int = 8, batch: int = 2, seed: int = 0):
    params = model.init_params(jr.PRNGKey(seed), cfg)
    ids = jr.randint(jr.PRNGKey(seed + 1), (batch, T), 0, cfg.vocab_size)
    logits = model.forward(params, cfg, ids, chunk_size=4)
    return params, ids, logits


def test_dense_config_builds_12_dense_blocks_and_forwards() -> None:
    cfg = _mini(load_config(DENSE))
    validate_config(cfg)
    params, _ids, logits = _run_forward(cfg)

    assert len(params.layers) == 12
    # Every layer is a dense-standard block: the dense oracle's parameter tree
    # plus a dense SiTU-GLU MLP — no KDA, no LatentMoE anywhere.
    assert all(isinstance(block.attn, MLAParams) for block in params.layers)
    assert all(isinstance(block.mlp, MLPParams) for block in params.layers)
    assert not any(isinstance(block.attn, KDAParams) for block in params.layers)

    assert logits.shape == (2, 8, cfg.vocab_size)
    assert bool(jnp.isfinite(logits).all())


def test_dense_config_loss_is_finite() -> None:
    cfg = _mini(load_config(DENSE))
    params, ids, _ = _run_forward(cfg)
    loss = model.compute_loss(params, cfg, ids, chunk_size=4)
    assert bool(jnp.isfinite(loss))
    assert float(loss) > 0.0


def test_dense_standard_ignores_the_sparse_oracle_switch() -> None:
    """The block is dense by layer kind, not by ``attn_dense_reference``.

    Both flag values must give bit-identical logits for the same parameters: if
    the switch leaked into this layer type, the dense arm could silently run the
    sparse+window path and the leg's verdict would be about a different model.
    """
    on = _mini(load_config(DENSE))
    off = dataclasses.replace(on, attn_dense_reference=False)
    params = model.init_params(jr.PRNGKey(3), on)
    ids = jr.randint(jr.PRNGKey(4), (1, 6), 0, on.vocab_size)
    a = model.forward(params, on, ids, chunk_size=4)
    b = model.forward(params, off, ids, chunk_size=4)
    assert bool(jnp.array_equal(a, b))


def test_dense_config_forward_with_scan_declared_still_runs() -> None:
    """The group-scan cannot carry a dense-standard stack — and says so.

    ``_group_scan_units`` rejects the composition (journal line, not a wrong
    graph), so even a config that declares ``scan_layers = true`` runs the
    unrolled path with the same numbers.
    """
    cfg = dataclasses.replace(_mini(load_config(DENSE)), scan_layers=True)
    assert model._group_scan_units(cfg) is None
    params, ids, logits = _run_forward(cfg)
    plain = dataclasses.replace(cfg, scan_layers=False)
    assert bool(jnp.allclose(logits, model.forward(params, plain, ids, chunk_size=4)))


def test_arch_config_builds_and_forwards() -> None:
    cfg = _mini(load_config(ARCH))
    validate_config(cfg)
    params, _ids, logits = _run_forward(cfg)

    kinds = [type(block.attn).__name__ for block in params.layers]
    assert kinds.count("KDAParams") == 9
    assert kinds.count("MLAParams") == 3
    assert logits.shape == (2, 8, cfg.vocab_size)
    assert bool(jnp.isfinite(logits).all())


# ---------------------------------------------------------------------------
# the declaration is guarded: no silent fallback to the production pattern
# ---------------------------------------------------------------------------


def _write_config(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_declared_dense_standard_is_read_from_layer_composition(tmp_path) -> None:
    path = _write_config(tmp_path, {
        "num_layers": 12, "num_kda_layers": 0, "num_mla_layers": 0,
        "layer_composition": {"dense-standard": 12, "kda": 0, "mla": 0},
    })
    cfg = load_config(path)
    assert cfg.dense_standard_layers == 12
    assert model.layer_kinds(cfg) == ("dense-standard",) * 12


def test_explicit_schema_field_wins_over_the_composition_object(tmp_path) -> None:
    path = _write_config(tmp_path, {
        "dense_standard_layers": 4,
        "num_layers": 24, "num_kda_layers": 15, "num_mla_layers": 5,
        "layer_composition": {"dense-standard": 12, "kda": 15, "mla": 5},
    })
    cfg = load_config(path)
    assert cfg.dense_standard_layers == 4
    validate_config(cfg)  # tail is 20 layers = five whole [K,K,K,M] units


@pytest.mark.parametrize("bad", ["12", -1, True, 1.5, None])
def test_malformed_dense_standard_declaration_raises(tmp_path, bad) -> None:
    payload = {"num_layers": 12, "num_kda_layers": 0, "num_mla_layers": 0,
               "layer_composition": {"dense-standard": bad}}
    if bad is None:
        # ``null`` is read as "not declared" — that is the explicit-absent case,
        # and the conservative answer is the production composition (0), which
        # validate_config then rejects for this layer count.
        cfg = load_config(_write_config(tmp_path, payload))
        assert cfg.dense_standard_layers == 0
        with pytest.raises(AssertionError):
            validate_config(cfg)
        return
    with pytest.raises(ValueError):
        load_config(_write_config(tmp_path, payload))


def test_mismatching_declared_kda_counts_raise(tmp_path) -> None:
    path = _write_config(tmp_path, {
        "num_layers": 12, "num_kda_layers": 9, "num_mla_layers": 3,
        "layer_composition": {"kda": 8, "mla": 3},
    })
    with pytest.raises(ValueError, match="disagrees with num_kda_layers"):
        load_config(path)


def test_non_object_layer_composition_raises(tmp_path) -> None:
    path = _write_config(tmp_path, {
        "num_layers": 12, "num_kda_layers": 9, "num_mla_layers": 3,
        "layer_composition": [9, 3],
    })
    with pytest.raises(ValueError):
        load_config(path)


def test_validate_rejects_broken_dense_standard_prefixes() -> None:
    base = dict(num_layers=24, num_kda_layers=18, num_mla_layers=6)

    # A prefix longer than the backbone, or negative, or not an integer.
    for bad in (25, -1, 1.5, True):
        with pytest.raises(AssertionError):
            validate_config(ModelConfig(**base, dense_standard_layers=bad))

    # A tail that is not a whole number of [K,K,K,M] units.
    with pytest.raises(AssertionError):
        validate_config(ModelConfig(**base, dense_standard_layers=2))

    # KDA/MLA counts that do not describe the tail.
    with pytest.raises(AssertionError):
        validate_config(
            ModelConfig(num_layers=24, num_kda_layers=14, num_mla_layers=6,
                        dense_standard_layers=4)
        )

    # The tail pattern is fixed 3:1 — a 17/7 split of 24 layers is not it.
    with pytest.raises(AssertionError):
        validate_config(ModelConfig(num_layers=24, num_kda_layers=17, num_mla_layers=7))

    # Well-formed mixed and pure-dense builds pass.
    validate_config(
        ModelConfig(num_layers=24, num_kda_layers=15, num_mla_layers=5,
                    dense_standard_layers=4)
    )
    validate_config(
        ModelConfig(num_layers=12, num_kda_layers=0, num_mla_layers=0,
                    dense_standard_layers=12)
    )
