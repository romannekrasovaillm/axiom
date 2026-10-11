"""Прототипы 16: явный lax.broadcast_in_dim для row-вектора (путь правила broadcast)."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental.pallas import mosaic_gpu as plgpu

DT = jnp.bfloat16
C, DK = 64, 128
CP = plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Lane)
WG = plgpu.Layout.WGMMA
L1 = plgpu.Layout.MMA_ACC(DT, m_warps=1)


def _body(cast, explicit, L):
    def body(xg, bg, og, x_s, b_s, o_s):
        h = jax.lax.axis_index("head")
        x_s[...] = xg.at[h][...]
        b_s[...] = bg.at[h][...]
        xv = plgpu.load(x_s, layout=L, optimized=False).astype(jnp.float32)
        bv = plgpu.load(b_s, layout=L, optimized=False).astype(jnp.float32)
        r = jnp.sum(bv, axis=1)
        if cast:
            r = plgpu.layout_cast(r, L.reduce(1))
        if explicit:
            bf = jax.lax.broadcast_in_dim(r, (C, DK), (0,))
        else:
            bf = r[:, None]
        o_s[...] = (xv * bf).astype(DT)
        og.at[h][...] = o_s[...]
    return body


def _mk(cast, explicit, L):
    body = _body(cast, explicit, L)
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, DK), DT),
                       scratch_types=[plgpu.SMEM((C, DK), DT), plgpu.SMEM((C, 32), jnp.float32),
                                      plgpu.SMEM((C, DK), DT)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run


def caseZ4(x, b32):  # explicit + cast, WGMMA-раскладка
    return _mk(True, True, WG)(x, b32)


def caseZ5(x, b32):  # explicit без cast, WGMMA
    return _mk(False, True, WG)(x, b32)


def caseZ6(x, b32):  # explicit + cast, ACC(m_warps=1)
    return _mk(True, True, L1)(x, b32)


def caseZ7(x, b32):  # без explicit, только cast, WGMMA (эталон падения)
    return _mk(True, False, WG)(x, b32)


def case_inputs(name):
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (4, C, DK)).astype(DT)
    b32 = jax.random.normal(key, (4, C, 32)).astype(jnp.float32)
    return {"caseZ4": (x, b32), "caseZ5": (x, b32), "caseZ6": (x, b32), "caseZ7": (x, b32)}.get(name)
