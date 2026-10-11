"""Прототипы 7: beta-скаляры через WGMMA-layout и построчный broadcast."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental.pallas import mosaic_gpu as plgpu

DT = jnp.bfloat16
C, DK = 64, 128
CP = plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Lane)
WG = plgpu.Layout.WGMMA


# G1: beta8 -> WGMMA load -> sum -> broadcast (C,C) -> умножение
def caseG1(beta8, a):
    def body(bg, ag, og, b_s, a_s):
        h = jax.lax.axis_index("head")
        b_s[...] = bg.at[h][...]
        a_s[...] = ag.at[h][...]
        bv = plgpu.load(b_s, layout=WG, optimized=False)
        r = jnp.sum(bv, axis=1)
        av = plgpu.load(a_s, layout=WG, optimized=False)
        og.at[h][...] = av * r[:, None]
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, C), jnp.float32),
                       scratch_types=[plgpu.SMEM((C, 8), jnp.float32), plgpu.SMEM((C, C), jnp.float32)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(beta8, a)


# G2: beta8 -> WGMMA -> sum -> broadcast на (C,DK) bf16
def caseG2(x, beta8):
    def body(xg, bg, og, x_s, b_s):
        h = jax.lax.axis_index("head")
        x_s[...] = xg.at[h][...]
        b_s[...] = bg.at[h][...]
        bv = plgpu.load(b_s, layout=WG, optimized=False)
        r = jnp.sum(bv, axis=1)
        xv = plgpu.load(x_s, layout=WG, optimized=False)
        og.at[h][...] = (xv.astype(jnp.float32) * r[:, None]).astype(DT)
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, DK), DT),
                       scratch_types=[plgpu.SMEM((C, DK), DT), plgpu.SMEM((C, 8), jnp.float32)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(x, beta8)


# G3: beta8 -> WGMMA -> sum -> broadcast, матрица читается обычным ref-чтением
def caseG3(x, beta8):
    def body(xg, bg, og, x_s, b_s):
        h = jax.lax.axis_index("head")
        x_s[...] = xg.at[h][...]
        b_s[...] = bg.at[h][...]
        bv = plgpu.load(b_s, layout=WG, optimized=False)
        r = jnp.sum(bv, axis=1)
        og.at[h][...] = (x_s[...].astype(jnp.float32) * r[:, None]).astype(DT)
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, DK), DT),
                       scratch_types=[plgpu.SMEM((C, DK), DT), plgpu.SMEM((C, 8), jnp.float32)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(x, beta8)


# G4: beta8 (bf16) -> WGMMA -> sum -> broadcast на (C,DK) bf16 (все bf16)
def caseG4(x, beta8):
    def body(xg, bg, og, x_s, b_s):
        h = jax.lax.axis_index("head")
        x_s[...] = xg.at[h][...]
        b_s[...] = bg.at[h][...]
        bv = plgpu.load(b_s, layout=WG, optimized=False)
        r = jnp.sum(bv, axis=1)
        xv = plgpu.load(x_s, layout=WG, optimized=False)
        og.at[h][...] = xv * r[:, None]
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, DK), DT),
                       scratch_types=[plgpu.SMEM((C, DK), DT), plgpu.SMEM((C, 8), DT)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(x, beta8)


def case_inputs(name):
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (4, C, DK)).astype(DT)
    a = jax.random.normal(key, (4, C, C)).astype(jnp.float32)
    beta8 = jax.random.normal(key, (4, C, 8)).astype(jnp.float32)
    beta8b = beta8.astype(DT)
    return {"caseG1": (beta8, a), "caseG2": (x, beta8), "caseG3": (x, beta8),
            "caseG4": (x, beta8b)}.get(name)
