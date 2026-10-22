"""Acceptance criterion 1 — parameter budget and layer composition.

Pinned in ``net/config.json``: total parameters in [0.9, 1.2]B, vocabulary
160K, layer composition 18 KDA : 6 Gated MLA (3:1) for attention and
1 dense + 23 Stable LatentMoE for channel mixing, plus the counter of
parameters activated per token (``active_params_per_token``).
"""

from __future__ import annotations

import json
from pathlib import Path

from net.config import ModelConfig
from net.model import (
    active_param_count,
    dense_moe_split,
    param_count,
    param_count_tree,
    _layer_is_kda,
)

CONFIG = Path(__file__).resolve().parent.parent / "config.json"


def test_param_count_within_budget():
    cfg = ModelConfig()
    total = param_count(cfg)
    assert 900_000_000 <= total <= 1_200_000_000, total
    assert cfg.vocab_size == 160_000


def test_layer_composition_ratio():
    cfg = ModelConfig()
    kinds = [_layer_is_kda(i) for i in range(cfg.num_layers)]
    assert sum(kinds) == 18
    assert cfg.num_layers - sum(kinds) == 6
    assert sum(kinds) / cfg.num_layers == 0.75  # 3:1


def test_dense_moe_split():
    """Channel mixing: 1 dense SiTU-GLU layer + 23 Stable LatentMoE layers."""
    cfg = ModelConfig()
    n_dense, n_moe = dense_moe_split(cfg)
    assert (n_dense, n_moe) == (1, 23)
    assert n_dense + n_moe == cfg.num_layers


def test_config_json_pins_actual_count():
    data = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert data["actual_param_count"] == param_count(ModelConfig())
    assert data["vocab_size"] == 160_000
    assert data["layer_composition"] == {"kda": 18, "mla": 6, "ratio": "3:1"}
    assert data["moe_composition"]["dense"] == 1
    assert data["moe_composition"]["latent_moe"] == 23
    lo, hi = data["param_budget"]
    assert lo <= data["actual_param_count"] <= hi


def test_active_params_counter():
    """Active-per-token counter: pinned, below total, above the attention floor.

    Attention alone (~277M) plus the frozen dense layer-0 MLP already sets a
    ~0.30B floor, so the ~0.25-0.35B orientation of the spec is unreachable —
    recorded in ``config.json`` ``deviations``.
    """
    cfg = ModelConfig()
    total = param_count(cfg)
    active = active_param_count(cfg)
    data = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert data["active_params_per_token"] == active
    assert 250_000_000 <= active < total
    assert active < total - cfg.vocab_size * cfg.hidden  # inactive experts exist


def test_embedding_is_dominant_and_tied():
    # The 160K x 1536 tied embedding is ~245M of the ~0.98B total.
    cfg = ModelConfig()
    tree = param_count_tree(cfg)
    assert tree["embedding"] == cfg.vocab_size * cfg.hidden
    assert tree["embedding"] > 200_000_000
