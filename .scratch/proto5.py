"""Прототипы 5: layout_cast над broadcast, маски-константы, iota с раскладкой."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental.pallas import mosaic_gpu as plgpu

DT = jnp.bfloat16
C, DK = 64, 128
CP = plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Lane)
ACC32 = plgpu.Layout.MMA_ACC(jnp.float32)
ACCDT = plgpu.Layout.MMA_ACC(DT)
MASK = jnp.tril(jnp.ones((C, C), DT))


# D1: beta8 -> reduce -> layout_cast(broadcast) -> (C,C)
def caseD1(beta8, a):
    def body(bg, ag, og, b_s, a_s):
        h = jax.lax.axis_index("head")
        b_s[...] = bg.at[h][...]
        a_s[...] = ag.at[h][...]
        bv = plgpu.load(b_s, layout=ACC32, optimized=False)
        r = jnp.sum(bv, axis=1)
        bf = plgpu.layout_cast(jnp.broadcast_to(r[:, None], (C, C)), ACC32)
        og.at[h][...] = a_s[...] * bf
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, C), jnp.float32),
                       scratch_types=[plgpu.SMEM((C, 8), jnp.float32), plgpu.SMEM((C, C), jnp.float32)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(beta8, a)


# D2: beta8 -> reduce -> layout_cast(broadcast) -> умножение bf16 (C,DK)
def caseD2(x, beta8):
    def body(xg, bg, og, x_s, b_s):
        h = jax.lax.axis_index("head")
        x_s[...] = xg.at[h][...]
        b_s[...] = bg.at[h][...]
        bv = plgpu.load(b_s, layout=ACC32, optimized=False)
        r = jnp.sum(bv, axis=1)
        bf = plgpu.layout_cast(jnp.broadcast_to(r[:, None], (C, DK)), ACC32)
        og.at[h][...] = (x_s[...].astype(jnp.float32) * bf).astype(DT)
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, DK), DT),
                       scratch_types=[plgpu.SMEM((C, DK), DT), plgpu.SMEM((C, 8), jnp.float32)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(x, beta8)


# D3: маска-константа, закрытая в теле ядра
def caseD3(a):
    def body(ag, og, a_s, m_s):
        h = jax.lax.axis_index("head")
        a_s[...] = ag.at[h][...]
        m_s[...] = MASK
        og.at[h][...] = jnp.where(m_s[...] > 0.5, a_s[...], 0.0)
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, C), jnp.float32),
                       scratch_types=[plgpu.SMEM((C, C), jnp.float32), plgpu.SMEM((C, C), DT)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(a)


# D4: broadcasted_iota с layout_cast
def caseD4(a):
    def body(ag, og, a_s):
        h = jax.lax.axis_index("head")
        a_s[...] = ag.at[h][...]
        ridx = plgpu.layout_cast(
            plgpu.broadcasted_iota(jnp.int32, (C, C), 0),
            plgpu.Layout.WG_STRIDED((C, C), jnp.int32))
        cidx = plgpu.layout_cast(
            plgpu.broadcasted_iota(jnp.int32, (C, C), 1),
            plgpu.Layout.WG_STRIDED((C, C), jnp.int32))
        og.at[h][...] = jnp.where(ridx >= cidx, a_s[...], 0.0)
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, C), jnp.float32),
                       scratch_types=[plgpu.SMEM((C, C), jnp.float32)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(a)


# D5: beta понижение через SMEM-выражение с явной целью записи
def caseD5(x, beta8):
    def body(xg, bg, og, x_s, b_s):
        h = jax.lax.axis_index("head")
        x_s[...] = xg.at[h][...]
        b_s[...] = bg.at[h][...]
        bv = plgpu.load(b_s, layout=ACC32, optimized=False)
        r = jnp.sum(bv, axis=1)
        x_s[...] = (x_s[...].astype(jnp.float32) * r[:, None]).astype(DT)
        og.at[h][...] = x_s[...]
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, DK), DT),
                       scratch_types=[plgpu.SMEM((C, DK), DT), plgpu.SMEM((C, 8), jnp.float32)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(x, beta8)


def case_inputs(name):
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (4, C, DK)).astype(DT)
    a = jax.random.normal(key, (4, C, C)).astype(jnp.float32)
    beta8 = jax.random.normal(key, (4, C, 8)).astype(jnp.float32)
    return {"caseD1": (beta8, a), "caseD2": (x, beta8), "caseD3": (a,),
            "caseD4": (a,), "caseD5": (x, beta8)}.get(name)
