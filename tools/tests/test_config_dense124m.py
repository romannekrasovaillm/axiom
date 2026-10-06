"""Contract test for ``net/config-dense124m.json`` (verification leg, dense arm).

The dense config is declarative *data* for ``docs/specs/VERIFICATION-LEG.ru.md``:
the skeleton scaled to GPT-2-class width with **every** layer of the diagnostic
``dense-standard`` type (dense attention through the existing oracle
``net/mla.py:_dense_apply`` + a dense SiTU-GLU MLP), declared as
``layer_composition["dense-standard"] = 12`` and built by ``net/model.py``.

This test is the mechanical proof of what the leg depends on:

* the file loads through the real schema (``net.config.load_config``) and passes
  ``validate_config`` — the config is usable by the runner, not merely JSON;
* the fields the spec names are exactly the declared ones, the tokenizer pin is a
  byte-for-byte copy of ``net/config.json``, and the layer composition is the
  declared dense-standard one (not the production KDA/MLA tail);
* the declared parameter counts match the live shape count (a declared number
  without a guard is a liability).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1]
CASE_DIR = TOOLS_DIR.parent
for _p in (str(TOOLS_DIR), str(CASE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DENSE = CASE_DIR / "net" / "config-dense124m.json"
BASE = CASE_DIR / "net" / "config.json"


def _dense_json() -> dict:
    return json.loads(DENSE.read_text(encoding="utf-8"))


def test_dense_config_loads_and_validates() -> None:
    from net.config import load_config, validate_config

    cfg = load_config(DENSE)  # raises on a malformed nested record
    validate_config(cfg)  # raises on a broken schema invariant


def test_spec_values_are_declared() -> None:
    cfg = _dense_json()
    assert cfg["num_layers"] == 12
    assert cfg["num_heads"] == 12
    assert cfg["head_dim"] == 64
    assert cfg["hidden"] == cfg["num_heads"] * cfg["head_dim"] == 768
    assert cfg["mlp_intermediate"] == 3072
    assert cfg["vocab_size"] == 160000
    assert cfg["attn_dense_reference"] is True
    # Structural values the schema asserts must stay consistent in the file.
    assert cfg["kda_dk"] == cfg["kda_dv"] == cfg["head_dim"]
    assert cfg["mla_head_dim"] == cfg["head_dim"]
    assert cfg["num_layers"] % 4 == 0
    assert len(cfg["mla_layer_modes"]) == cfg["num_mla_layers"]


def test_dense_arm_is_the_dense_standard_composition() -> None:
    """The dense arm declares the diagnostic layer type, not the KDA/MLA tail."""
    from net.config import load_config
    from net.model import layer_kinds

    data = _dense_json()
    cfg = load_config(DENSE)
    assert data["layer_composition"]["dense-standard"] == 12
    assert cfg.dense_standard_layers == 12
    assert (cfg.num_kda_layers, cfg.num_mla_layers) == (0, 0)
    assert cfg.moe_dense_layers == 12
    assert layer_kinds(cfg) == ("dense-standard",) * 12
    # Sparse-path mechanisms are declared off: no MLA layer is built at all.
    assert data["mla_layer_modes"] == []
    assert data["mla_pool_size"] == 0
    assert data["mla_block_merge"]["enabled"] is False


def test_tokenizer_hash_is_copied_verbatim() -> None:
    assert _dense_json()["tokenizer_hash"] == json.loads(BASE.read_text(encoding="utf-8"))["tokenizer_hash"]


def test_declared_param_count_matches_live_shapes() -> None:
    """The declared counts are measured, not asserted from the skeleton."""
    pytest.importorskip("jax")
    from net.config import load_config
    from net.model import active_param_count, param_count

    cfg = load_config(DENSE)
    declared = _dense_json()
    assert declared["actual_param_count"] == param_count(cfg)
    assert declared["active_params_per_token"] == active_param_count(cfg)
    assert declared["active_params_per_token"] < declared["actual_param_count"]
