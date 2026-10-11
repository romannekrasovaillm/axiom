"""Прототипы 19: layout_cast WGMMA->ACC, multi-out со strided-записью, reuse op-буфера."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.experimental import pallas as pl

DT = jnp.bfloat16
C, DK, DV, H = 64, 128, 128, 4
CP = plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Lane)
WG = plgpu.Layout.WGMMA
ACC = plgpu.Layout.MMA_ACC(DT)


# R1: layout_cast(WGMMA-значение -> ACC) и mma с ним как acc
def caseR1(s, y, v):
    def body(sg, yg, vg, og, s_s, y_s, op_s):
        h = jax.lax.axis_index("head")
        sv = plgpu.load(sg.at[h], layout=WG, optimized=False)
        sv = sv * 2.0
        acc0 = plgpu.layout_cast(sv, ACC)
        y_s[...] = yg.at[h][...]
        vv = plgpu.load(vg.at[h], layout=WG, optimized=False)
        op_s.transpose((1, 0))[...] = vv.astype(DT)
        lhs = plgpu.load(y_s, layout=plgpu.Layout.MMA_LHS(DT), optimized=False)
        rhs = plgpu.load(op_s.T, layout=plgpu.Layout.MMA_RHS(DT), optimized=False)
        og.at[h][...] = plgpu.mma(acc0, lhs, rhs)
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((H, DK, DV), jnp.float32),
                       scratch_types=[plgpu.SMEM((DK, DV), jnp.float32),
                                      plgpu.SMEM((DK, C), DT), plgpu.SMEM((DV, C), DT)],
                       compiler_params=CP, grid=(H,), grid_names=("head",))
    return run(s, y, v)


# R2: несколько выходов + strided-запись в (C, H*DV)
def caseR2(x):
    def body(xg, o1g, o2g, x_s):
        h = jax.lax.axis_index("head")
        x_s[...] = xg.at[h][...]
        o1g.at[pl.ds(0, C), pl.ds(h * DV, DV)][...] = x_s[...].astype(jnp.float32)
        o2g.at[h][...] = jnp.zeros((3, 4), DT)
    run = plgpu.kernel(body, out_type=(jax.ShapeDtypeStruct((C, H * DV), jnp.float32),
                                       jax.ShapeDtypeStruct((H, 3, 4), DT)),
                       scratch_types=[plgpu.SMEM((C, DV), DT)],
                       compiler_params=CP, grid=(H,), grid_names=("head",))
    return run(x)


# R3: reuse одного op-буфера: маска -> операнд -> маска -> операнд -> mma
def caseR3(x, y):
    def body(xg, yg, og, op_s, pat_s):
        h = jax.lax.axis_index("head")
        ridx = plgpu.layout_cast(plgpu.broadcasted_iota(jnp.int32, (C, DK), 0),
                                 plgpu.Layout.WG_STRIDED((C, DK), 2))
        xv = plgpu.load(xg.at[h], layout=WG, optimized=False)
        yv = plgpu.load(xg.at[h], layout=WG, optimized=False)
        op_s[...] = (ridx >= 16).astype(DT)
        mk = plgpu.load(op_s, layout=WG, optimized=False).astype(jnp.float32)
        op_s[...] = (xv * mk).astype(DT)
        lhs = plgpu.load(op_s, layout=plgpu.Layout.MMA_LHS(DT), optimized=False)
        op_s[...] = (ridx >= 32).astype(DT)
        mk2 = plgpu.load(op_s, layout=WG, optimized=False).astype(jnp.float32)
        op_s[...] = (yv * mk2).astype(DT)
        rhs = plgpu.load(op_s.T, layout=plgpu.Layout.MMA_RHS(DT), optimized=False)
        og.at[h][...] = plgpu.mma(_acc((C, C)), lhs, rhs)
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((H, C, C), jnp.float32),
                       scratch_types=[plgpu.SMEM((C, DK), DT), plgpu.SMEM((C, DK), DT)],
                       compiler_params=CP, grid=(H,), grid_names=("head",))
    return run(x, y)


def _acc(shape):
    return plgpu.layout_cast(jnp.zeros(shape, jnp.float32), ACC)


def case_inputs(name):
    key = jax.random.PRNGKey(0)
    s = jax.random.normal(key, (H, DK, DV)).astype(jnp.float32)
    y = jax.random.normal(key, (H, DK, C)).astype(DT)
    v = jax.random.normal(key, (H, C, DV)).astype(DT)
    x = jax.random.normal(key, (H, C, DV)).astype(DT)
    return {"caseR1": (s, y, v), "caseR2": (x,), "caseR3": (x, x)}.get(name)
