"""Прототипы 18: reduce(0)+col-broadcast, маски через SMEM, anchor-mma."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental.pallas import mosaic_gpu as plgpu

DT = jnp.bfloat16
C, DK = 64, 128
CP = plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Lane)
WG = plgpu.Layout.WGMMA


# Q1: reduce(0) -> (DK,) -> broadcast (1,) -> умножение (C,DK)
def caseQ1(x, w):
    def body(xg, wg, og, x_s, w_s, o_s):
        h = jax.lax.axis_index("head")
        x_s[...] = xg.at[h][...]
        w_s[...] = wg.at[h][...]
        xv = plgpu.load(x_s, layout=WG, optimized=False).astype(jnp.float32)
        wv2 = plgpu.load(w_s, layout=WG, optimized=False).astype(jnp.float32)
        wv = jnp.sum(wv2, axis=0)
        wb = jax.lax.broadcast_in_dim(wv, (C, DK), (1,))
        o_s[...] = (xv * wb).astype(DT)
        og.at[h][...] = o_s[...]
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, DK), DT),
                       scratch_types=[plgpu.SMEM((C, DK), DT), plgpu.SMEM((C, DK), jnp.float32),
                                      plgpu.SMEM((C, DK), DT)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(x, w)


# Q2: маска-iota -> SMEM -> load(WGMMA) -> умножение на WGMMA-значение
def caseQ2(x):
    def body(xg, og, x_s, mk_s, o_s):
        h = jax.lax.axis_index("head")
        x_s[...] = xg.at[h][...]
        ridx = plgpu.layout_cast(plgpu.broadcasted_iota(jnp.int32, (C, DK), 0),
                                 plgpu.Layout.WG_STRIDED((C, DK), 2))
        mk_s[...] = (ridx >= 16).astype(DT)
        xv = plgpu.load(x_s, layout=WG, optimized=False).astype(jnp.float32)
        mk = plgpu.load(mk_s, layout=WG, optimized=False).astype(jnp.float32)
        o_s[...] = (xv * mk).astype(DT)
        og.at[h][...] = o_s[...]
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, DK), DT),
                       scratch_types=[plgpu.SMEM((C, DK), DT), plgpu.SMEM((C, DK), DT),
                                      plgpu.SMEM((C, DK), DT)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(x)


# Q3: транспонированная запись (DK,C) из WGMMA-значения + RHS-загрузка + mma
def caseQ3(t, la):
    def body(tg, lg, og, t_s, la_t_s):
        h = jax.lax.axis_index("head")
        t_s[...] = tg.at[h][...]
        lav = plgpu.load(lg.at[h], layout=WG, optimized=False)
        la_t_s.transpose((1, 0))[...] = lav.astype(DT)
        lhs = plgpu.load(t_s, layout=plgpu.Layout.MMA_LHS(DT), optimized=False)
        rhs = plgpu.load(la_t_s.T, layout=plgpu.Layout.MMA_RHS(DT), optimized=False)
        og.at[h][...] = plgpu.mma(_acc((C, DK)), lhs, rhs)
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, DK), jnp.float32),
                       scratch_types=[plgpu.SMEM((C, C), DT), plgpu.SMEM((DK, C), DT)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(t, la)


# Q4: anchor-mma: Lst (C,C)-паттерн на LHS, la_t на RHS -> (C,DK) все строки = G_p
def caseQ4(lst, la):
    def body(lg, ag, og, lst_s, la_t_s):
        h = jax.lax.axis_index("head")
        lst_s[...] = lg.at[h][...]
        lav = plgpu.load(ag.at[h], layout=WG, optimized=False)
        la_t_s.transpose((1, 0))[...] = lav.astype(DT)
        lhs = plgpu.load(lst_s, layout=plgpu.Layout.MMA_LHS(DT), optimized=False)
        rhs = plgpu.load(la_t_s.T, layout=plgpu.Layout.MMA_RHS(DT), optimized=False)
        og.at[h][...] = plgpu.mma(_acc((C, DK)), lhs, rhs)
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, DK), jnp.float32),
                       scratch_types=[plgpu.SMEM((C, C), DT), plgpu.SMEM((DK, C), DT)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(lst, la)


def _acc(shape):
    return plgpu.layout_cast(jnp.zeros(shape, jnp.float32), plgpu.Layout.MMA_ACC(DT))


def case_inputs(name):
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (4, C, DK)).astype(DT)
    w = jax.random.normal(key, (4, C, DK)).astype(jnp.float32)
    t = jnp.tril(jnp.ones((4, C, C)), -1).astype(DT)
    la = jax.random.normal(key, (4, C, DK)).astype(DT)
    lst = jnp.tril(jnp.ones((4, C, C))).astype(DT)
    return {"caseQ1": (x, w), "caseQ2": (x,), "caseQ3": (t, la), "caseQ4": (lst, la)}.get(name)
