"""SiTU-GLU MLP (kda-formulas.md section 4.2, Eq. 12).

Dense channel-mixing of the leading layer(s) of the backbone (layer 0, as the
base's 1 dense of 93) and of the MTP block.  Layers 1..23 use the Stable
LatentMoE of ``net/moe.py``; the SiTU-GLU activation itself (``siti_glu``) is
shared by the dense MLP, the routed experts and the shared experts.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import ModelConfig
from . import compute_dtype


def _rand(key, shape, scale: float) -> jnp.ndarray:
    return jax.random.normal(key, shape) * scale


class MLPParams(NamedTuple):
    W_g: jnp.ndarray  # (hidden, intermediate)
    W_u: jnp.ndarray  # (hidden, intermediate)
    W_down: jnp.ndarray  # (intermediate, hidden)


def init_mlp(key, cfg: ModelConfig) -> MLPParams:
    hid, inter = cfg.hidden, cfg.mlp_intermediate
    k1, k2, k3 = jax.random.split(key, 3)
    return MLPParams(
        W_g=_rand(k1, (hid, inter), 0.02),
        W_u=_rand(k2, (hid, inter), 0.02),
        W_down=_rand(k3, (inter, hid), 0.02),
    )


def siti_glu(x: jnp.ndarray, beta_gate: float, beta_up: float) -> jnp.ndarray:
    """Sigmoid-Tanh-Unit GLU activation (Eq. 12).

    ``SiTU-GLU(h_g, h_u) = [beta_gate tanh(h_g/beta_gate) * sigmoid(h_g)] * [beta_up tanh(h_u/beta_up)]``.
    """
    gate = beta_gate * jnp.tanh(x[0] / beta_gate) * jax.nn.sigmoid(x[0])
    up = beta_up * jnp.tanh(x[1] / beta_up)
    return gate * up


def apply(params: MLPParams, cfg: ModelConfig, x: jnp.ndarray) -> jnp.ndarray:
    """SiTU-GLU MLP over ``(..., hidden)``.

    The three projections are the dense MLP's GEMMs; the activation between them
    is element-wise and stays in the input dtype (fp32 under both gate modes,
    because a boundary returns fp32 — ``AXIOM_COMPUTE_DTYPE``, see
    ``net/compute_dtype.py``).
    """
    h_g = compute_dtype.gemm(x, params.W_g)
    h_u = compute_dtype.gemm(x, params.W_u)
    a = siti_glu((h_g, h_u), cfg.siti_beta_gate, cfg.siti_beta_up)
    return compute_dtype.gemm(a, params.W_down)
