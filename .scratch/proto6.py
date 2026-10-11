"""Прототипы 6: reduced-layout broadcast, размеры 1-D буферов, iota-маска."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental.pallas import mosaic_gpu as plgpu

DT = jnp.bfloat16
C, DK = 64, 128
CP = plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Lane)
ACC32 = plgpu.Layout.MMA_ACC(jnp.float32)
ACC32_R1 = plgpu.Layout.MMA_ACC(jnp.float32).reduce(1)  # построчный скаляр


# E1: beta8 -> ACC load -> sum -> reduced layout -> broadcast (C,C)
def caseE1(beta8, a):
    def body(bg, ag, og, b_s, a_s):
        h = jax.lax.axis_index("head")
        b_s[...] = bg.at[h][...]
        a_s[...] = ag.at[h][...]
        bv = plgpu.load(b_s, layout=ACC32, optimized=False)
        r = plgpu.layout_cast(jnp.sum(bv, axis=1), ACC32_R1)
        og.at[h][...] = a_s[...] * r[:, None]
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, C), jnp.float32),
                       scratch_types=[plgpu.SMEM((C, 8), jnp.float32), plgpu.SMEM((C, C), jnp.float32)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(beta8, a)


# E2: то же -> умножение bf16 (C,DK)
def caseE2(x, beta8):
    def body(xg, bg, og, x_s, b_s):
        h = jax.lax.axis_index("head")
        x_s[...] = xg.at[h][...]
        b_s[...] = bg.at[h][...]
        bv = plgpu.load(b_s, layout=ACC32, optimized=False)
        r = plgpu.layout_cast(jnp.sum(bv, axis=1), ACC32_R1)
        og.at[h][...] = (x_s[...].astype(jnp.float32) * r[:, None]).astype(DT)
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, DK), DT),
                       scratch_types=[plgpu.SMEM((C, DK), DT), plgpu.SMEM((C, 8), jnp.float32)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(x, beta8)


# E3: маска-вход bf16, умножение на (C,C) f32
def caseE3(a, mask):
    def body(ag, mg, og, a_s, m_s):
        h = jax.lax.axis_index("head")
        a_s[...] = ag.at[h][...]
        m_s[...] = mg[...]
        og.at[h][...] = jnp.where(m_s[...] > 0.5, a_s[...], 0.0)
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, C), jnp.float32),
                       scratch_types=[plgpu.SMEM((C, C), jnp.float32), plgpu.SMEM((C, C), DT)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(a, mask)


# E4: буферы (C,2) и (C,4) f32
def caseE4(b2):
    def body(bg, og, b_s):
        h = jax.lax.axis_index("head")
        b_s[...] = bg.at[h][...]
        og.at[h][...] = b_s[...] + 1.0
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, 2), jnp.float32),
                       scratch_types=[plgpu.SMEM((C, 2), jnp.float32)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(b2)


# E5: маска через broadcasted_iota + WG_STRIDED
def caseE5(a):
    def body(ag, og, a_s):
        h = jax.lax.axis_index("head")
        a_s[...] = ag.at[h][...]
        ridx = plgpu.layout_cast(plgpu.broadcasted_iota(jnp.int32, (C, C), 0),
                                 plgpu.Layout.WG_STRIDED((C, C), 2))
        cidx = plgpu.layout_cast(plgpu.broadcasted_iota(jnp.int32, (C, C), 1),
                                 plgpu.Layout.WG_STRIDED((C, C), 2))
        og.at[h][...] = jnp.where(ridx >= cidx, a_s[...], 0.0)
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, C), jnp.float32),
                       scratch_types=[plgpu.SMEM((C, C), jnp.float32)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(a)


def case_inputs(name):
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (4, C, DK)).astype(DT)
    a = jax.random.normal(key, (4, C, C)).astype(jnp.float32)
    beta8 = jax.random.normal(key, (4, C, 8)).astype(jnp.float32)
    b2 = jax.random.normal(key, (4, C, 2)).astype(jnp.float32)
    mask = jnp.tril(jnp.ones((C, C), DT))
    return {"caseE1": (beta8, a), "caseE2": (x, beta8), "caseE3": (a, mask),
            "caseE4": (b2,), "caseE5": (a,)}.get(name)
