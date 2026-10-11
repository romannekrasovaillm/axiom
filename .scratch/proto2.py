"""Прототипы 2: вызовы ядер + вопросы раскладок load(.T)/LHS/RHS, SMEM-транспозы."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental.pallas import mosaic_gpu as plgpu

DT = jnp.bfloat16
C, DK = 64, 128
CP = plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Lane)


def _acc(shape):
    return plgpu.layout_cast(jnp.zeros(shape, jnp.float32), plgpu.Layout.MMA_ACC(DT))


# V1: LHS из row-major (C,DK) + RHS через .T
def caseV1(x, y):
    def body(xg, yg, og, x_s, y_s):
        h = jax.lax.axis_index("head")
        x_s[...] = xg.at[h][...]
        y_s[...] = yg.at[h][...]
        lhs = plgpu.load(x_s, layout=plgpu.Layout.MMA_LHS(DT), optimized=False)
        rhs = plgpu.load(y_s.T, layout=plgpu.Layout.MMA_RHS(DT), optimized=False)
        og.at[h][...] = plgpu.mma(_acc((C, C)), lhs, rhs)
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, C), jnp.float32),
                       scratch_types=[plgpu.SMEM((C, DK), DT), plgpu.SMEM((C, DK), DT)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(x, y)


# V3: LHS из транспонированного вида (DK,C)
def caseV3(x, y):
    def body(xg, yg, og, x_s, y_s):
        h = jax.lax.axis_index("head")
        x_s[...] = xg.at[h][...]
        y_s[...] = yg.at[h][...]
        lhs = plgpu.load(x_s.T, layout=plgpu.Layout.MMA_LHS(DT), optimized=False)
        rhs = plgpu.load(y_s.T, layout=plgpu.Layout.MMA_RHS(DT), optimized=False)
        og.at[h][...] = plgpu.mma(_acc((DK, C)), lhs, rhs)
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, DK, C), jnp.float32),
                       scratch_types=[plgpu.SMEM((C, DK), DT), plgpu.SMEM((C, DK), DT)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(x, y)


# V4: RHS из row-major (C,DK) (без .T)
def caseV4(x, y):
    def body(xg, yg, og, x_s, y_s):
        h = jax.lax.axis_index("head")
        x_s[...] = xg.at[h][...]
        y_s[...] = yg.at[h][...]
        lhs = plgpu.load(x_s, layout=plgpu.Layout.MMA_LHS(DT), optimized=False)
        rhs = plgpu.load(y_s, layout=plgpu.Layout.MMA_RHS(DT), optimized=False)
        og.at[h][...] = plgpu.mma(_acc((C, C)), lhs, rhs)
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, C), jnp.float32),
                       scratch_types=[plgpu.SMEM((C, DK), DT), plgpu.SMEM((C, DK), DT)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(x, y)


# V2: RHS через .T от явного row-major (n,k) буфера (DK-row x C-col подан с T-видом)
def caseV2(x):
    def body(xg, og, x_s, xt_s):
        h = jax.lax.axis_index("head")
        x_s[...] = xg.at[h][...]
        # xt_s (DK, C): заполняем транспонированной записью из фрагмента x_s
        frag = plgpu.load(x_s, layout=plgpu.Layout.MMA_LHS(DT), optimized=False)
        xt_s.transpose((1, 0))[...] = frag
        lhs = plgpu.load(x_s, layout=plgpu.Layout.MMA_LHS(DT), optimized=False)
        rhs = plgpu.load(xt_s.T, layout=plgpu.Layout.MMA_RHS(DT), optimized=False)
        og.at[h][...] = plgpu.mma(_acc((C, C)), lhs, rhs)
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, C), jnp.float32),
                       scratch_types=[plgpu.SMEM((C, DK), DT), plgpu.SMEM((DK, C), DT)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(x)


# V5: транспонированная запись фрагмента в SMEM (без mma)
def caseV5(x):
    def body(xg, og, x_s, xt_s):
        h = jax.lax.axis_index("head")
        x_s[...] = xg.at[h][...]
        frag = plgpu.load(x_s, layout=plgpu.Layout.MMA_LHS(DT), optimized=False)
        xt_s.transpose((1, 0))[...] = frag
        og.at[h][...] = xt_s[...]
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, DK, C), DT),
                       scratch_types=[plgpu.SMEM((C, DK), DT), plgpu.SMEM((DK, C), DT)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(x)


# V6: построчная редукция на SMEM-значении (l2norm)
def caseV6(x):
    def body(xg, og, x_s):
        h = jax.lax.axis_index("head")
        x_s[...] = xg.at[h][...]
        xf = x_s[...].astype(jnp.float32)
        r = jnp.sum(xf * xf, axis=1)
        og.at[h][...] = (xf * (1.0 / jnp.sqrt(r + 1e-6))[:, None]).astype(DT)
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, DK), DT),
                       scratch_types=[plgpu.SMEM((C, DK), DT)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(x)


# V7: broadcast вектора (C,) по строкам в (C,C)
def caseV7(beta, a):
    def body(bg, ag, og, b_s, a_s):
        h = jax.lax.axis_index("head")
        b_s[...] = bg.at[h][...]
        a_s[...] = ag.at[h][...]
        bv = b_s[...].astype(jnp.float32)
        og.at[h][...] = a_s[...] * bv[:, None]
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, C), jnp.float32),
                       scratch_types=[plgpu.SMEM((C,), jnp.float32), plgpu.SMEM((C, C), jnp.float32)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(beta, a)


# V8: cumsum через нижнетреугольный mma с транспонированной записью RHS
def caseV8(la):
    def body(lag, og, la_s, la_t_s, l_s):
        h = jax.lax.axis_index("head")
        la_s[...] = lag.at[h][...]
        frag = plgpu.load(la_s, layout=plgpu.Layout.MMA_LHS(DT), optimized=False)
        la_t_s.transpose((1, 0))[...] = frag
        l_s[...] = jnp.tril(jnp.ones((C, C), DT))
        lhs = plgpu.load(l_s, layout=plgpu.Layout.MMA_LHS(DT), optimized=False)
        rhs = plgpu.load(la_t_s.T, layout=plgpu.Layout.MMA_RHS(DT), optimized=False)
        og.at[h][...] = plgpu.mma(_acc((C, DK)), lhs, rhs)
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, DK), jnp.float32),
                       scratch_types=[plgpu.SMEM((C, DK), DT), plgpu.SMEM((DK, C), DT),
                                      plgpu.SMEM((C, C), DT)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(la)


# V9: mma с bm=64, bn=128, k=128 (inter: (C,DK)@(DK,DV))
def caseV9(q, s):
    def body(qg, sg, og, q_s, s_t_s):
        h = jax.lax.axis_index("head")
        q_s[...] = qg.at[h][...]
        # s приходит (DK,DV); для RHS нужна память (n,k)=(DV,DK) — кладём транспонированной записью
        sf = plgpu.load(sg.at[h], layout=plgpu.Layout.MMA_LHS(DT), optimized=False)
        s_t_s.transpose((1, 0))[...] = sf
        lhs = plgpu.load(q_s, layout=plgpu.Layout.MMA_LHS(DT), optimized=False)
        rhs = plgpu.load(s_t_s.T, layout=plgpu.Layout.MMA_RHS(DT), optimized=False)
        og.at[h][...] = plgpu.mma(_acc((C, 128)), lhs, rhs)
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, 128), jnp.float32),
                       scratch_types=[plgpu.SMEM((C, DK), DT), plgpu.SMEM((128, DK), DT)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(q, s)


def case_inputs(name):
    key = jax.random.PRNGKey(0)
    x = (jax.random.normal(key, (4, C, DK))).astype(DT)
    beta = jax.random.normal(key, (4, C)).astype(jnp.float32)
    a = jax.random.normal(key, (4, C, C)).astype(jnp.float32)
    return {
        "caseV1": (x, x), "caseV3": (x, x), "caseV4": (x, x),
        "caseV2": (x,), "caseV5": (x,), "caseV6": (x,), "caseV7": (beta, a),
        "caseV8": (x,), "caseV9": (x, x),
    }.get(name)
