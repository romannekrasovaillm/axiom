"""AttnRes — Attention Residuals (kda-formulas.md section 3, Eq. 8-9).

Each layer retrieves representations from all preceding layers through a
learned pseudo-query ``w_l`` and a softmax kernel ``phi(q, k) = exp(q^T RMSNorm(k))``
(:386), rather than accumulating residuals uniformly.  The skeleton uses the
*full* form (Eq. 8-9): with L = 24 layers the O(L^2 d) cost is affordable, which
is the regime the report itself assigns to the full form ("network depth is
modest (L < 100)", :393).

Integration choice (our decision, recorded in ``config.json`` ``deviations``):
AttnRes is *additive* — ``h_l = h_{l-1} + f_l + r_l`` where ``f_l`` is the layer's
own residual delta and ``r_l`` the AttnRes weighted sum over the embedding and
prior layer deltas.  The source vectors follow Eq. 8 verbatim:
``k_i = v_i = h_embed`` (i = 0) and ``f_i(h_i)`` (1 <= i <= l-1).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import ModelConfig
from .norm import headwise_rms_norm


class AttnResParams(NamedTuple):
    w: jnp.ndarray  # (num_layers, hidden) learned pseudo-queries


def init_attnres(key, cfg: ModelConfig) -> AttnResParams:
    return AttnResParams(w=jax.random.normal(key, (cfg.num_layers, cfg.hidden)) * 0.02)


def apply_layer(w_l: jnp.ndarray, sources: jnp.ndarray) -> jnp.ndarray:
    """Depth attention for one layer over ``sources`` of shape (N, B, T, hidden).

    ``sources[0]`` is the token embedding, ``sources[1:]`` the prior layer
    deltas.  Returns ``(B, T, hidden)``.
    """
    keys = headwise_rms_norm(sources)  # RMSNorm(k), Eq. 8
    scores = jnp.einsum("d,nbtd->nbt", w_l, keys)  # (N, B, T)
    weights = jax.nn.softmax(scores, axis=0)  # (N, B, T)
    return jnp.einsum("nbt,nbtd->btd", weights, sources)  # (B, T, hidden)
