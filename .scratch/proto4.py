"""Прототипы 4: beta через (C,8)+reduce, маски (iota / GMEM-вход)."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental.pallas import mosaic_gpu as plgpu

DT = jnp.bfloat16
C, DK = 64, 128
CP = plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Lane)
ACC32 = plgpu.Layout.MMA_ACC(jnp.float32)


# C1: beta8 (C,8) -> ACC-загрузка -> sum(axis=1) -> broadcast -> (C,C)
def caseC1(beta8, a):
    def body(bg, ag, og, b_s, a_s):
        h = jax.lax.axis_index("head")
        b_s[...] = bg.at[h][...]
        a_s[...] = ag.at[h][...]
        bv = plgpu.load(b_s, layout=ACC32, optimized=False)
        r = jnp.sum(bv, axis=1)
        og.at[h][...] = a_s[...] * r[:, None]
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, C), jnp.float32),
                       scratch_types=[plgpu.SMEM((C, 8), jnp.float32), plgpu.SMEM((C, C), jnp.float32)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(beta8, a)


# C2: beta8 plain read -> sum(axis=1)
def caseC2(beta8):
    def body(bg, og, b_s):
        h = jax.lax.axis_index("head")
        b_s[...] = bg.at[h][...]
        r = jnp.sum(b_s[...], axis=1)
        og.at[h][...] = r
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C), jnp.float32),
                       scratch_types=[plgpu.SMEM((C, 8), jnp.float32)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(beta8)


# C3: маски через broadcasted_iota внутри ядра
def caseC3(a):
    def body(ag, og, a_s):
        h = jax.lax.axis_index("head")
        a_s[...] = ag.at[h][...]
        ridx = plgpu.broadcasted_iota(jnp.int32, (C, C), 0)
        cidx = plgpu.broadcasted_iota(jnp.int32, (C, C), 1)
        mask = plgpu.layout_cast((ridx >= cidx).astype(DT), plgpu.Layout.MMA_ACC(DT))
        og.at[h][...] = jnp.where(mask, a_s[...], 0.0)
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, C), jnp.float32),
                       scratch_types=[plgpu.SMEM((C, C), jnp.float32)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(a)


# C4: маска как GMEM-вход (C,C) f32 (по программе — общая, но грузим срез)
def caseC4(a, mask):
    def body(ag, mg, og, a_s, m_s):
        h = jax.lax.axis_index("head")
        a_s[...] = ag.at[h][...]
        m_s[...] = mg[...]
        og.at[h][...] = jnp.where(m_s[...] > 0.5, a_s[...], 0.0)
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, C), jnp.float32),
                       scratch_types=[plgpu.SMEM((C, C), jnp.float32), plgpu.SMEM((C, C), jnp.float32)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(a, mask)


# C5: 1-D буфер (128,) f32 — граница «multiple of 128»
def caseC5(v):
    def body(vg, og, v_s):
        h = jax.lax.axis_index("head")
        v_s[...] = vg.at[h][...]
        og.at[h][...] = v_s[...] * 2.0
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, 128), jnp.float32),
                       scratch_types=[plgpu.SMEM((128,), jnp.float32)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(v)


# C6: broadcast (C,) -> (C,DK) умножение bf16
def caseC6(x, beta):
    def body(xg, bg, og, x_s, b_s):
        h = jax.lax.axis_index("head")
        x_s[...] = xg.at[h][...]
        b_s[...] = bg.at[h][...]
        bv = plgpu.load(b_s, layout=plgpu.Layout.MMA_ACC(jnp.float32), optimized=False)
        og.at[h][...] = (x_s[...].astype(jnp.float32) * bv[:, None]).astype(DT)
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, DK), DT),
                       scratch_types=[plgpu.SMEM((C, DK), DT), plgpu.SMEM((C, 8), jnp.float32)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(x, beta)


def case_inputs(name):
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (4, C, DK)).astype(DT)
    a = jax.random.normal(key, (4, C, C)).astype(jnp.float32)
    beta8 = jax.random.normal(key, (4, C, 8)).astype(jnp.float32)
    v = jax.random.normal(key, (4, 128)).astype(jnp.float32)
    mask = jnp.tril(jnp.ones((C, C), jnp.float32))
    return {"caseC1": (beta8, a), "caseC2": (beta8,), "caseC3": (a,),
            "caseC4": (a, mask), "caseC5": (v,), "caseC6": (x, beta8)}.get(name)
