"""Прототипы 9: конв через смещённые SMEM-загрузки + веса (dk,) в reduce(0)."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.experimental import pallas as pl

DT = jnp.bfloat16
C, DK = 64, 128
CP = plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Lane)
L1 = plgpu.Layout.MMA_ACC(DT, m_warps=1)


# U1: смещённая SMEM-загрузка (по строкам), 4 тапа, веса (DK,) в reduce(0)
def caseU1(x4, w):
    def body(xg, wg, og, x_s, w_s, o_s):
        h = jax.lax.axis_index("head")
        x_s[...] = xg.at[h][...]
        w_s[...] = wg.at[h][...]
        acc = None
        for j in range(4):
            xv = plgpu.load(x_s.at[pl.ds(3 - j, C), :], layout=L1, optimized=False).astype(jnp.float32)
            wv = plgpu.load(w_s.at[pl.ds(j, 1), :], layout=L1.reduce(0), optimized=False).astype(jnp.float32)
            term = xv * wv[None, :]
            acc = term if acc is None else acc + term
        o_s[...] = acc.astype(DT)
        og.at[h][...] = o_s[...]
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, DK), DT),
                       scratch_types=[plgpu.SMEM((C + 3, DK), DT), plgpu.SMEM((4, DK), DT),
                                      plgpu.SMEM((C, DK), DT)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(x4, w)


# U2: то же, но веса загружены как (4,DK) с L1 и срез строки через load(ref.at[j])
def caseU2(x4, w):
    def body(xg, wg, og, x_s, w_s, o_s):
        h = jax.lax.axis_index("head")
        x_s[...] = xg.at[h][...]
        w_s[...] = wg.at[h][...]
        acc = None
        for j in range(4):
            xv = plgpu.load(x_s.at[pl.ds(3 - j, C), :], layout=L1, optimized=False).astype(jnp.float32)
            wf = plgpu.load(w_s.at[j, :], layout=L1.reduce(0), optimized=False).astype(jnp.float32)
            term = xv * wf[None, :]
            acc = term if acc is None else acc + term
        o_s[...] = acc.astype(DT)
        og.at[h][...] = o_s[...]
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, DK), DT),
                       scratch_types=[plgpu.SMEM((C + 3, DK), DT), plgpu.SMEM((4, DK), DT),
                                      plgpu.SMEM((C, DK), DT)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(x4, w)


# U3: (C,) вектор через редукцию (C,8)->сумма -> [:, None] на ACC-значении
def caseU3(x, b8):
    def body(xg, bg, og, x_s, b_s):
        h = jax.lax.axis_index("head")
        x_s[...] = xg.at[h][...]
        b_s[...] = bg.at[h][...]
        xv = plgpu.load(x_s, layout=L1, optimized=False).astype(jnp.float32)
        bv = plgpu.load(b_s, layout=L1, optimized=False).astype(jnp.float32)
        r = jnp.sum(bv, axis=1)
        og.at[h][...] = (xv * r[:, None]).astype(DT)
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, DK), DT),
                       scratch_types=[plgpu.SMEM((C, DK), DT), plgpu.SMEM((C, 8), jnp.float32)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(x, b8)


# U4: (C,8) ACC-загрузка: подходит ли L1 (m_warps=1, тайл (16,32)) для n=8?
def caseU4(b8):
    def body(bg, og, b_s):
        h = jax.lax.axis_index("head")
        b_s[...] = bg.at[h][...]
        bv = plgpu.load(b_s, layout=L1, optimized=False).astype(jnp.float32)
        og.at[h][...] = jnp.sum(bv, axis=1)
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C), jnp.float32),
                       scratch_types=[plgpu.SMEM((C, 8), jnp.float32)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(b8)


def case_inputs(name):
    key = jax.random.PRNGKey(0)
    x4 = jax.random.normal(key, (4, C + 3, DK)).astype(DT)
    w = jax.random.normal(key, (4, 4, DK)).astype(DT)
    b8 = jax.random.normal(key, (4, C, 8)).astype(jnp.float32)
    x = jax.random.normal(key, (4, C, DK)).astype(DT)
    return {"caseU1": (x4, w), "caseU2": (x4, w), "caseU3": (x, b8), "caseU4": (b8,)}.get(name)
