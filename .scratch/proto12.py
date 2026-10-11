"""Прототипы 12: WGMMA-раскладка как универсальная для (C,·) + broadcast 32->DK."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.experimental import pallas as pl

DT = jnp.bfloat16
C, DK = 64, 128
CP = plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Lane)
WG = plgpu.Layout.WGMMA


# Z1: (C,32) WGMMA -> broadcast_to (C,DK) c подсказкой WGMMA -> умножение
def caseZ1(x, b32):
    def body(xg, bg, og, x_s, b_s, o_s):
        h = jax.lax.axis_index("head")
        x_s[...] = xg.at[h][...]
        b_s[...] = bg.at[h][...]
        xv = plgpu.load(x_s, layout=WG, optimized=False)
        bv = plgpu.load(b_s, layout=WG, optimized=False)
        bf = plgpu.layout_cast(jnp.broadcast_to(bv, (C, DK)), WG)
        o_s[...] = (xv.astype(jnp.float32) * bf).astype(DT)
        og.at[h][...] = o_s[...]
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, DK), DT),
                       scratch_types=[plgpu.SMEM((C, DK), DT), plgpu.SMEM((C, 32), jnp.float32),
                                      plgpu.SMEM((C, DK), DT)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(x, b32)


# Z2: то же, но broadcast на (C,C)
def caseZ2(a, b32):
    def body(ag, bg, og, a_s, b_s, o_s):
        h = jax.lax.axis_index("head")
        a_s[...] = ag.at[h][...]
        b_s[...] = bg.at[h][...]
        av = plgpu.load(a_s, layout=WG, optimized=False)
        bv = plgpu.load(b_s, layout=WG, optimized=False)
        bf = plgpu.layout_cast(jnp.broadcast_to(bv, (C, C)), WG)
        o_s[...] = (av.astype(jnp.float32) * bf).astype(DT)
        og.at[h][...] = o_s[...]
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, C), DT),
                       scratch_types=[plgpu.SMEM((C, C), DT), plgpu.SMEM((C, 32), jnp.float32),
                                      plgpu.SMEM((C, C), DT)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(a, b32)


# Z3: (C,) из редукции WGMMA-значения -> [:, None] -> broadcast
def caseZ3(x, b32):
    def body(xg, bg, og, x_s, b_s, o_s):
        h = jax.lax.axis_index("head")
        x_s[...] = xg.at[h][...]
        b_s[...] = bg.at[h][...]
        xv = plgpu.load(x_s, layout=WG, optimized=False)
        bv = plgpu.load(b_s, layout=WG, optimized=False)
        r = plgpu.layout_cast(jnp.sum(bv, axis=1), WG.reduce(1))
        o_s[...] = (xv.astype(jnp.float32) * r[:, None]).astype(DT)
        og.at[h][...] = o_s[...]
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, DK), DT),
                       scratch_types=[plgpu.SMEM((C, DK), DT), plgpu.SMEM((C, 32), jnp.float32),
                                      plgpu.SMEM((C, DK), DT)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(x, b32)


def case_inputs(name):
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (4, C, DK)).astype(DT)
    a = jax.random.normal(key, (4, C, C)).astype(DT)
    b32 = jax.random.normal(key, (4, C, 32)).astype(jnp.float32)
    return {"caseZ1": (x, b32), "caseZ2": (a, b32), "caseZ3": (x, b32)}.get(name)
