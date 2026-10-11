"""Прототипы 11: мини-паттерн rmsnorm (reduce+broadcast), ref-срезы через .at."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.experimental import pallas as pl

DT = jnp.bfloat16
C, DK = 64, 128
CP = plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Lane)
L1 = plgpu.Layout.MMA_ACC(DT, m_warps=1)


# Y1: 16 строк — как в шаблоне rmsnorm
def caseY1(x, b):
    def body(xg, bg, og, x_s, b_s, o_s):
        h = jax.lax.axis_index("head")
        x_s[...] = xg.at[h][...]
        b_s[...] = bg.at[h][...]
        xv = plgpu.load(x_s, layout=L1, optimized=False).astype(jnp.float32)
        bv = plgpu.load(b_s, layout=L1, optimized=False).astype(jnp.float32)
        r = jnp.sum(bv, axis=1)
        o_s[...] = (xv * r[:, None]).astype(DT)
        og.at[h][...] = o_s[...]
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, 16, DK), DT),
                       scratch_types=[plgpu.SMEM((16, DK), DT), plgpu.SMEM((16, 32), jnp.float32),
                                      plgpu.SMEM((16, DK), DT)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(x, b)


# Y2: 64 строки
def caseY2(x, b):
    def body(xg, bg, og, x_s, b_s, o_s):
        h = jax.lax.axis_index("head")
        x_s[...] = xg.at[h][...]
        b_s[...] = bg.at[h][...]
        xv = plgpu.load(x_s, layout=L1, optimized=False).astype(jnp.float32)
        bv = plgpu.load(b_s, layout=L1, optimized=False).astype(jnp.float32)
        r = jnp.sum(bv, axis=1)
        o_s[...] = (xv * r[:, None]).astype(DT)
        og.at[h][...] = o_s[...]
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, DK), DT),
                       scratch_types=[plgpu.SMEM((C, DK), DT), plgpu.SMEM((C, 32), jnp.float32),
                                      plgpu.SMEM((C, DK), DT)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(x, b)


# Y3: с явным layout_cast на r
def caseY3(x, b):
    def body(xg, bg, og, x_s, b_s, o_s):
        h = jax.lax.axis_index("head")
        x_s[...] = xg.at[h][...]
        b_s[...] = bg.at[h][...]
        xv = plgpu.load(x_s, layout=L1, optimized=False).astype(jnp.float32)
        bv = plgpu.load(b_s, layout=L1, optimized=False).astype(jnp.float32)
        r = plgpu.layout_cast(jnp.sum(bv, axis=1), L1.reduce(1))
        o_s[...] = (xv * r[:, None]).astype(DT)
        og.at[h][...] = o_s[...]
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, DK), DT),
                       scratch_types=[plgpu.SMEM((C, DK), DT), plgpu.SMEM((C, 32), jnp.float32),
                                      plgpu.SMEM((C, DK), DT)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(x, b)


# Y5: смещённая загрузка SMEM через .at[ds]
def caseY5(x4):
    def body(xg, og, x_s):
        h = jax.lax.axis_index("head")
        x_s[...] = xg.at[h][...]
        frag = plgpu.load(x_s.at[pl.ds(3, C), pl.ds(0, DK)], layout=L1, optimized=False)
        og.at[h][...] = frag
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, DK), DT),
                       scratch_types=[plgpu.SMEM((C + 3, DK), DT)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(x4)


# Y6: строчный GMEM-срез по голове через .at
def caseY6(x):
    def body(xg, og, x_s):
        h = jax.lax.axis_index("head")
        frag = plgpu.load(xg.at[pl.ds(0, C), h, pl.ds(0, DK)], layout=L1, optimized=False)
        x_s[...] = frag
        og[pl.ds(0, C), h, pl.ds(0, DK)][...] = x_s[...]
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((C, 4, DK), DT),
                       scratch_types=[plgpu.SMEM((C, DK), DT)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(x)


def case_inputs(name):
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (4, C, DK)).astype(DT)
    x16 = jax.random.normal(key, (4, 16, DK)).astype(DT)
    b = jax.random.normal(key, (4, C, 32)).astype(jnp.float32)
    b16 = jax.random.normal(key, (4, 16, 32)).astype(jnp.float32)
    x4 = jax.random.normal(key, (4, C + 3, DK)).astype(DT)
    return {"caseY1": (x16, b16), "caseY2": (x, b), "caseY3": (x, b),
            "caseY5": (x4,), "caseY6": (x,)}.get(name)
