"""Прототипы 8: фазовые примитивы (срезы SMEM, транспозы, редукции, sel-mma, multi-out)."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental.pallas import mosaic_gpu as plgpu

DT = jnp.bfloat16
C, DK, DV = 64, 128, 128
CP = plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Lane)
L1 = plgpu.Layout.MMA_ACC(DT, m_warps=1)


# T1: статические срезы SMEM (конв-паттерн)
def caseT1(x):
    def body(xg, og, x_s, y_s):
        h = jax.lax.axis_index("head")
        x_s[...] = xg.at[h][...]
        y_s[...] = x_s[3:3 + C, :] + x_s[2:2 + C, :]
        og.at[h][...] = y_s[...]
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, DK), DT),
                       scratch_types=[plgpu.SMEM((C + 3, DK), DT), plgpu.SMEM((C, DK), DT)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(x)


# T2: GMEM -> фрагмент с явной раскладкой -> SMEM
def caseT2(x):
    def body(xg, og, x_s):
        h = jax.lax.axis_index("head")
        frag = plgpu.load(xg.at[h], layout=L1, optimized=False)
        x_s[...] = frag
        og.at[h][...] = x_s[...]
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, DK), DT),
                       scratch_types=[plgpu.SMEM((C, DK), DT)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(x)


# T3: RHS из (DK,C)-буфера, заполненного транспонированной записью
def caseT3(x):
    def body(xg, og, x_s, xt_s):
        h = jax.lax.axis_index("head")
        x_s[...] = xg.at[h][...]
        frag = plgpu.load(x_s, layout=L1, optimized=False)
        xt_s.transpose((1, 0))[...] = frag
        lhs = plgpu.load(x_s, layout=plgpu.Layout.MMA_LHS(DT), optimized=False)
        rhs = plgpu.load(xt_s.T, layout=plgpu.Layout.MMA_RHS(DT), optimized=False)
        og.at[h][...] = plgpu.mma(_acc((C, C)), lhs, rhs)
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, C), jnp.float32),
                       scratch_types=[plgpu.SMEM((C, DK), DT), plgpu.SMEM((DK, C), DT)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(x)


# T4: редукция по оси 0 (колоночные скаляры) на ACC-значении
def caseT4(x):
    def body(xg, og, x_s):
        h = jax.lax.axis_index("head")
        x_s[...] = xg.at[h][...]
        xv = plgpu.load(x_s, layout=L1, optimized=False).astype(jnp.float32)
        og.at[h][...] = jnp.sum(xv, axis=0)
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, DK), jnp.float32),
                       scratch_types=[plgpu.SMEM((C, DK), DT)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(x)


# T5: sel-mma: (C,C)-селектор последней строки × (C,DK)-матрица
def caseT5(g):
    def body(gg, og, g_s, gt_s, sel_s):
        h = jax.lax.axis_index("head")
        g_s[...] = gg.at[h][...]
        frag = plgpu.load(g_s, layout=L1, optimized=False)
        gt_s.transpose((1, 0))[...] = frag
        ridx = plgpu.layout_cast(plgpu.broadcasted_iota(jnp.int32, (C, C), 0),
                                 plgpu.Layout.WG_STRIDED((C, C), 2))
        cidx = plgpu.layout_cast(plgpu.broadcasted_iota(jnp.int32, (C, C), 1),
                                 plgpu.Layout.WG_STRIDED((C, C), 2))
        sel_s[...] = (cidx == (C - 1)).astype(DT)
        lhs = plgpu.load(sel_s, layout=plgpu.Layout.MMA_LHS(DT), optimized=False)
        rhs = plgpu.load(gt_s.T, layout=plgpu.Layout.MMA_RHS(DT), optimized=False)
        og.at[h][...] = plgpu.mma(_acc((C, DK)), lhs, rhs)
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, DK), jnp.float32),
                       scratch_types=[plgpu.SMEM((C, DK), DT), plgpu.SMEM((DK, C), DT),
                                      plgpu.SMEM((C, C), DT)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(g)


# T6: два выхода
def caseT6(x):
    def body(xg, o1g, o2g, x_s):
        h = jax.lax.axis_index("head")
        x_s[...] = xg.at[h][...]
        o1g.at[h][...] = x_s[...]
        o2g.at[h][...] = jnp.sum(x_s[...].astype(jnp.float32), axis=1)
    run = plgpu.kernel(body, out_type=(jax.ShapeDtypeStruct((4, C, DK), DT),
                                       jax.ShapeDtypeStruct((4, C), jnp.float32)),
                       scratch_types=[plgpu.SMEM((C, DK), DT)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(x)


# T7: маска поверх ACC-значения (где+layout)
def caseT7(a):
    def body(ag, og, a_s):
        h = jax.lax.axis_index("head")
        a_s[...] = ag.at[h][...]
        av = plgpu.load(a_s, layout=L1, optimized=False).astype(jnp.float32)
        ridx = plgpu.layout_cast(plgpu.broadcasted_iota(jnp.int32, (C, C), 0),
                                 plgpu.Layout.WG_STRIDED((C, C), 2))
        cidx = plgpu.layout_cast(plgpu.broadcasted_iota(jnp.int32, (C, C), 1),
                                 plgpu.Layout.WG_STRIDED((C, C), 2))
        og.at[h][...] = jnp.where(ridx >= cidx, av, 0.0)
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, C), jnp.float32),
                       scratch_types=[plgpu.SMEM((C, C), DT)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(a)


def _acc(shape):
    return plgpu.layout_cast(jnp.zeros(shape, jnp.float32), plgpu.Layout.MMA_ACC(DT))


def case_inputs(name):
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (4, C, DK)).astype(DT)
    a = jax.random.normal(key, (4, C, C)).astype(DT)
    return {"caseT1": (x,), "caseT2": (x,), "caseT3": (x,), "caseT4": (x,),
            "caseT5": (x,), "caseT6": (x,), "caseT7": (a,)}.get(name)
