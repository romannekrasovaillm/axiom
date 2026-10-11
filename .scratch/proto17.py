"""Прототипы 17: GMEM offset load/store, SMEM offset store, column-vector, multi-out."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.experimental import pallas as pl

DT = jnp.bfloat16
C, DK, DV, H = 64, 128, 128, 4
CP = plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Lane)
WG = plgpu.Layout.WGMMA


# P3a: GMEM offset load (срез по колонкам головы)
def caseP3a(x):
    def body(xg, og, o_s):
        h = jax.lax.axis_index("head")
        frag = plgpu.load(xg.at[pl.ds(0, C), pl.ds(h * DK, DK)], layout=WG, optimized=False)
        o_s[...] = frag
        og.at[h][...] = o_s[...]
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((H, C, DK), DT),
                       scratch_types=[plgpu.SMEM((C, DK), DT)],
                       compiler_params=CP, grid=(H,), grid_names=("head",))
    return run(x)


# P3b: GMEM offset store
def caseP3b(x):
    def body(xg, og, x_s):
        h = jax.lax.axis_index("head")
        x_s[...] = xg.at[h][...]
        og.at[pl.ds(0, C), pl.ds(h * DV, DV)][...] = x_s[...]
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((C, H * DV), DT),
                       scratch_types=[plgpu.SMEM((C, DV), DT)],
                       compiler_params=CP, grid=(H,), grid_names=("head",))
    return run(x)


# P4: SMEM offset store (паттерн stage для конва)
def caseP4(x3, x):
    def body(xg3, xg, og, s_s):
        h = jax.lax.axis_index("head")
        s_s.at[pl.ds(3, C), pl.ds(0, DK)][...] = xg.at[h][...]
        s_s.at[pl.ds(0, 3), pl.ds(0, DK)][...] = xg3.at[h][...]
        og.at[h][...] = s_s.at[pl.ds(3, C), pl.ds(0, DK)][...]
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((H, C, DK), DT),
                       scratch_types=[plgpu.SMEM((C + 3, DK), DT)],
                       compiler_params=CP, grid=(H,), grid_names=("head",))
    return run(x3, x)


# P5: column-vector: reduce(0) -> broadcast (1,) -> умножение
def caseP5(x, w2):
    def body(xg, wg, og, x_s, w_s, o_s):
        h = jax.lax.axis_index("head")
        x_s[...] = xg.at[h][...]
        w_s[...] = wg.at[h][...]
        xv = plgpu.load(x_s, layout=WG, optimized=False).astype(jnp.float32)
        wv2 = plgpu.load(w_s, layout=WG, optimized=False).astype(jnp.float32)
        wv = jnp.sum(wv2, axis=0)
        wb = jax.lax.broadcast_in_dim(wv, (C, DK), (1,))
        o_s[...] = (xv * wb).astype(DT)
        og.at[h][...] = o_s[...]
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((H, C, DK), DT),
                       scratch_types=[plgpu.SMEM((C, DK), DT), plgpu.SMEM((2, DK), jnp.float32),
                                      plgpu.SMEM((C, DK), DT)],
                       compiler_params=CP, grid=(H,), grid_names=("head",))
    return run(x, w2)


# P10: два выхода
def caseP10(x):
    def body(xg, o1g, o2g, x_s):
        h = jax.lax.axis_index("head")
        x_s[...] = xg.at[h][...]
        o1g.at[h][...] = x_s[...]
        o2g.at[h][...] = jnp.sum(x_s[...].astype(jnp.float32), axis=1)
    run = plgpu.kernel(body, out_type=(jax.ShapeDtypeStruct((H, C, DK), DT),
                                       jax.ShapeDtypeStruct((H, C), jnp.float32)),
                       scratch_types=[plgpu.SMEM((C, DK), DT)],
                       compiler_params=CP, grid=(H,), grid_names=("head",))
    return run(x)


def case_inputs(name):
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (H, C, DK)).astype(DT)
    x3 = jax.random.normal(key, (H, 3, DK)).astype(DT)
    xflat = jax.random.normal(key, (C, H * DK)).astype(DT)
    w2 = jax.random.normal(key, (H, 2, DK)).astype(jnp.float32)
    return {"caseP3a": (xflat,), "caseP3b": (x,), "caseP4": (x3, x),
            "caseP5": (x, w2), "caseP10": (x,)}.get(name)
