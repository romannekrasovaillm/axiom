"""Contract test for ``net/config-arch124m.json`` (verification leg, arch arm).

The arch config is the *architecture* arm of the leg's internal A/B
(``docs/specs/VERIFICATION-LEG.ru.md``, "Метрика и эталон" (i)): the same width
and depth as the dense arm (12 layers, 12 heads, d_model 768, mlp 3072) with the
pilot's KDA/MLA composition scaled by the declared 3:1 ratio — 9 KDA + 3 MLA.

It is checked the same way as the dense arm: the file must load through the real
schema, pass ``validate_config``, declare exactly the composition the layers
build, copy the tokenizer pin verbatim, and carry parameter counts that match the
live shape count.
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

ARCH = CASE_DIR / "net" / "config-arch124m.json"
DENSE = CASE_DIR / "net" / "config-dense124m.json"
BASE = CASE_DIR / "net" / "config.json"


def _arch_json() -> dict:
    return json.loads(ARCH.read_text(encoding="utf-8"))


def test_arch_config_loads_and_validates() -> None:
    from net.config import load_config, validate_config

    cfg = load_config(ARCH)
    validate_config(cfg)


def test_arch_arm_is_the_pilot_composition_at_124m_width() -> None:
    from net.config import load_config
    from net.model import layer_kinds

    data = _arch_json()
    cfg = load_config(ARCH)
    kinds = layer_kinds(cfg)
    assert cfg.dense_standard_layers == 0
    assert data["layer_composition"] == {"kda": 9, "mla": 3, "dense-standard": 0, "ratio": "3:1"}
    assert (kinds.count("kda"), kinds.count("mla")) == (9, 3)
    assert (cfg.num_kda_layers, cfg.num_mla_layers) == (9, 3)
    assert cfg.num_layers % 4 == 0 and cfg.num_layers == 12
    assert cfg.num_heads == 12 and cfg.hidden == 768 and cfg.head_dim == 64
    assert cfg.mlp_intermediate == 3072


def test_arms_share_width_depth_and_tokenizer() -> None:
    """The A/B differs in composition, not in width/depth/tokenizer."""
    arch = _arch_json()
    dense = json.loads(DENSE.read_text(encoding="utf-8"))
    for field in ("num_layers", "num_heads", "head_dim", "hidden",
                  "mlp_intermediate", "vocab_size", "tokenizer_hash"):
        assert arch[field] == dense[field], field
    assert arch["layer_composition"]["kda"] == 9
    assert dense["layer_composition"]["dense-standard"] == 12


def test_tokenizer_hash_is_copied_verbatim() -> None:
    assert _arch_json()["tokenizer_hash"] == json.loads(BASE.read_text(encoding="utf-8"))["tokenizer_hash"]


def test_declared_param_count_matches_live_shapes() -> None:
    pytest.importorskip("jax")
    from net.config import load_config
    from net.model import active_param_count, param_count

    cfg = load_config(ARCH)
    declared = _arch_json()
    assert declared["actual_param_count"] == param_count(cfg)
    assert declared["active_params_per_token"] == active_param_count(cfg)
    assert declared["active_params_per_token"] < declared["actual_param_count"]
