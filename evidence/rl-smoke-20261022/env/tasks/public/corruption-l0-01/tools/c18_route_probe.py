#!/usr/bin/env python3
"""c18_route_probe.py — какая ветка KDA реально трассируется на клетке persist.

Вопрос цикла-18: в HLO клетки persist-on остался jnp-скан чанков
(`chuv,chu->chv`, 256 графов). Проверяем на CPU (без GPU): при
AXIOM_KDA_CC_IMPL=persistent функция net.kda.cc_chunk_step не должна
трассироваться вовсе (её заменяет персистентный кернел). Счётчик — на обёртке
cc_chunk_step; оба кернел-шва подменяются jnp-репликой (форма теста
test_kda_cc_fused_persist). Печатает число трассировок cc_chunk_step для
«jax» и «persistent».

Запуск: NET_JAX_BACKEND=cpu JAX_PLATFORMS=cpu python3 tools/c18_route_probe.py
"""
import os
import sys

sys.path.insert(0, "/home/roman/axiom")

import jax
import jax.numpy as jnp

from net import kda
from net.config import ModelConfig
from net.kernels import kda_cc_fused_persist as persist

H, T, DK, DV = 2, 64, 128, 128
cfg = ModelConfig(
    hidden=H * DV, num_heads=H, head_dim=128,
    kda_dk=DK, kda_dv=DV, kda_decay_rank=128, kda_short_conv_kernel=4,
    kda_g_min=-5.0, kda_wyut_chunk=32, kda_impl="chunked_cc",
)
params = kda.init_kda(jax.random.PRNGKey(0), cfg)
x = jax.random.normal(jax.random.PRNGKey(1), (T, cfg.hidden))

counts = {"cc_chunk_step": 0}
_orig = kda.cc_chunk_step


def _wrapped(*a, **kw):
    counts["cc_chunk_step"] += 1
    return _orig(*a, **kw)


kda.cc_chunk_step = _wrapped

# kernel seams: the declared-numerics substitutions (CPU test form)
persist.device_available = lambda: True
persist.kernel = lambda q, k, v, alpha, beta32, S, *, tile=16, chunk=32: (
    persist.reference(q, k, v, alpha, beta32, S, tile=tile, chunk=chunk))


def _bwd_stub(q, k, v, alpha, beta32, S, do, dS, *, tile=16, chunk=32,
              debug_outputs=False):
    z = jnp.zeros_like
    return z(q), z(k), z(v), z(alpha), z(beta32), z(S)


persist.kernel_bwd = _bwd_stub

for impl in ("jax", "persistent"):
    os.environ["AXIOM_KDA_CC_IMPL"] = impl
    counts["cc_chunk_step"] = 0
    fn = lambda xb: jnp.sum(kda.apply_chunked_cc(params, cfg, xb, chunk_size=32))
    try:
        out = jax.jit(jax.grad(fn))(x)
        finite = bool(jnp.all(jnp.isfinite(out)))
        print(f"{impl:10s} cc_chunk_step traced={counts['cc_chunk_step']:3d} "
              f"grad_finite={finite}")
    except Exception as exc:  # noqa: BLE001 — диагностический зонд
        print(f"{impl:10s} ERROR {type(exc).__name__}: {exc}")
