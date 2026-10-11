"""Прототипы 10: head-major I/O, смещённые SMEM-загрузки, beta32, cumsum-цепочка."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.experimental import pallas as pl

DT = jnp.bfloat16
C, DK, DV = 64, 128, 128
CP = plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Lane)
L4 = plgpu.Layout.MMA_ACC(DT)          # m_warps=4: тайл (64,8)
L1 = plgpu.Layout.MMA_ACC(DT, m_warps=1)  # тайл (16,32)


# X1: срезы ref по оси головы (stride) — GMEM (C,H,DK)
def caseX1(x):
    def body(xg, og, x_s):
        h = jax.lax.axis_index("head")
        frag = plgpu.load(xg[pl.ds(0, C), h, pl.ds(0, DK)], layout=L4, optimized=False)
        x_s[...] = frag
        og[pl.ds(0, C), h, pl.ds(0, DK)][...] = x_s[...]
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((C, 4, DK), DT),
                       scratch_types=[plgpu.SMEM((C, DK), DT)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(x)


# X2: смещённые SMEM-загрузки для конва (паддированный буфер)
def caseX2(x4, w):
    def body(xg, wg, og, x_s, w_s):
        h = jax.lax.axis_index("head")
        x_s[...] = xg.at[h][...]
        w_s[...] = wg.at[h][...]
        acc = None
        for j in range(4):
            xv = plgpu.load(x_s[pl.ds(3 - j, C), pl.ds(0, DK)], layout=L4, optimized=False).astype(jnp.float32)
            wv = plgpu.load(w_s[pl.ds(j, 1), pl.ds(0, DK)], layout=L4.reduce(0), optimized=False).astype(jnp.float32)
            term = xv * wv[None, :]
            acc = term if acc is None else acc + term
        og.at[h][...] = acc.astype(DT)
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, DK), DT),
                       scratch_types=[plgpu.SMEM((C + 3, DK), DT), plgpu.SMEM((4, DK), DT)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(x4, w)


# X3: beta32 (C,32) L4 -> reduce -> роу-скейл (C,DK)
def caseX3(x, b32):
    def body(xg, bg, og, x_s, b_s):
        h = jax.lax.axis_index("head")
        x_s[...] = xg.at[h][...]
        b_s[...] = bg.at[h][...]
        xv = plgpu.load(x_s, layout=L4, optimized=False).astype(jnp.float32)
        bv = plgpu.load(b_s, layout=L4, optimized=False).astype(jnp.float32)
        r = jnp.sum(bv, axis=1)
        og.at[h][...] = (xv * r[:, None]).astype(DT)
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, DK), DT),
                       scratch_types=[plgpu.SMEM((C, DK), DT), plgpu.SMEM((C, 32), jnp.float32)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(x, b32)


# X4: cumsum: log(alpha) -> транспонированная запись (DK,C) -> mma с tril-маской
def caseX4(a):
    def body(ag, og, la_s, la_t_s, l_s):
        h = jax.lax.axis_index("head")
        la_s[...] = ag.at[h][...]
        frag = plgpu.load(la_s, layout=L4, optimized=False).astype(jnp.float32)
        la_t_s.transpose((1, 0))[...] = frag.astype(DT)
        ridx = plgpu.layout_cast(plgpu.broadcasted_iota(jnp.int32, (C, C), 0),
                                 plgpu.Layout.WG_STRIDED((C, C), 2))
        cidx = plgpu.layout_cast(plgpu.broadcasted_iota(jnp.int32, (C, C), 1),
                                 plgpu.Layout.WG_STRIDED((C, C), 2))
        l_s[...] = (ridx >= cidx).astype(DT)
        lhs = plgpu.load(l_s, layout=plgpu.Layout.MMA_LHS(DT), optimized=False)
        rhs = plgpu.load(la_t_s.T, layout=plgpu.Layout.MMA_RHS(DT), optimized=False)
        acc = plgpu.mma(_acc((C, DK)), lhs, rhs)
        og.at[h][...] = acc
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, DK), jnp.float32),
                       scratch_types=[plgpu.SMEM((C, DK), DT), plgpu.SMEM((DK, C), DT),
                                      plgpu.SMEM((C, C), DT)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(a)


# X6: фрагмент -> транспонированная запись (DK,C) -> RHS -> mma T@B
def caseX6(t, b):
    def body(tg, bg, og, t_s, bt_s):
        h = jax.lax.axis_index("head")
        t_s[...] = tg.at[h][...]
        frag = plgpu.load(bg.at[h][...], layout=L4, optimized=False)
        bt_s.transpose((1, 0))[...] = frag.astype(DT)
        lhs = plgpu.load(t_s, layout=plgpu.Layout.MMA_LHS(DT), optimized=False)
        rhs = plgpu.load(bt_s.T, layout=plgpu.Layout.MMA_RHS(DT), optimized=False)
        og.at[h][...] = plgpu.mma(_acc((C, C)), lhs, rhs)
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, C), jnp.float32),
                       scratch_types=[plgpu.SMEM((C, C), DT), plgpu.SMEM((DK, C), DT)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(t, b)


def _acc(shape):
    return plgpu.layout_cast(jnp.zeros(shape, jnp.float32), plgpu.Layout.MMA_ACC(DT))


def case_inputs(name):
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (4, C, DK)).astype(DT)
    x4 = jax.random.normal(key, (4, C + 3, DK)).astype(DT)
    w = jax.random.normal(key, (4, 4, DK)).astype(DT)
    b32 = jax.random.normal(key, (4, C, 32)).astype(jnp.float32)
    t = jnp.tril(jnp.ones((4, C, C)), -1).astype(DT)
    return {"caseX1": (x,), "caseX2": (x4, w), "caseX3": (x, b32),
            "caseX4": (x,), "caseX6": (t, x)}.get(name)
