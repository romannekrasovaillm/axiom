"""Прототипы примитивов для fused-кернела KDA-чанка (lowering-проверка под sm_121)."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental.pallas import mosaic_gpu as plgpu

DT = jnp.bfloat16
C, DK, DV = 64, 128, 128


def _acc(shape):
    return plgpu.layout_cast(jnp.zeros(shape, jnp.float32), plgpu.Layout.MMA_ACC(DT))


# --- 1. scores: mma(64x64, k=128) из SMEM-буферов ---------------------------------
def case1_scores(q, k, mask):
    def body(qg, kg, og, q_s, k_s, a_s):
        h = jax.lax.axis_index("head")
        plgpu.copy_gmem_to_smem(qg.at[h], q_s, plgpu.Barrier()) if False else None
        # прямые записи в SMEM (инференс раскладок)
        q_s[...] = qg.at[h]
        k_s[...] = kg.at[h]
        lhs = plgpu.load(q_s, layout=plgpu.Layout.MMA_LHS(DT), optimized=False)
        rhs = plgpu.load(k_s.T, layout=plgpu.Layout.MMA_RHS(DT), optimized=False)
        acc = plgpu.mma(_acc((C, C)), lhs, rhs)
        a_s[...] = jnp.where(mask, acc, 0.0)
        og.at[h][...] = a_s[...]
    return plgpu.kernel(
        body,
        out_type=jax.ShapeDtypeStruct((2, C, C), jnp.float32),
        scratch_types=[plgpu.SMEM((C, DK), DT), plgpu.SMEM((C, DK), DT), plgpu.SMEM((C, C), jnp.float32)],
        compiler_params=plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Lane),
        grid=(2,), grid_names=("head",),
    )


def case_inputs(name):
    key = jax.random.PRNGKey(0)
    if name == "case1_scores":
        q = jax.random.normal(key, (2, C, DK)).astype(DT)
        k = jax.random.normal(key, (2, C, DK)).astype(DT)
        mask = jnp.tril(jnp.ones((C, C), bool))
        return (q, k, mask)
    if name == "case2_smem_tw":
        x = jax.random.normal(key, (C, DK)).astype(DT)
        return (x,)
    if name == "case3_lhs_t":
        x = jax.random.normal(key, (C, DK)).astype(DT)
        return (x,)
    if name == "case4_rowred":
        x = jax.random.normal(key, (C, DK)).astype(DT)
        return (x,)
    if name == "case5_cumsum":
        la = jax.random.normal(key, (C, DK)).astype(DT)
        return (la,)
    if name == "case7_bcast":
        beta = jax.random.normal(key, (C,)).astype(jnp.float32)
        a = jax.random.normal(key, (C, C)).astype(jnp.float32)
        return (beta, a)
    return None


# --- 2. транспонированная запись в SMEM из фрагмента ------------------------------
def case2_smem_tw(x):
    def body(xg, og, x_s, y_s):
        h = jax.lax.axis_index("head")
        x_s[...] = xg.at[h]
        frag = plgpu.load(x_s, layout=plgpu.Layout.MMA_LHS(DT), optimized=False)
        y_s.transpose((1, 0))[...] = frag
        og.at[h][...] = y_s[...]
    return plgpu.kernel(
        body,
        out_type=jax.ShapeDtypeStruct((2, DK, C), DT),
        scratch_types=[plgpu.SMEM((C, DK), DT), plgpu.SMEM((DK, C), DT)],
        compiler_params=plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Lane),
        grid=(2,), grid_names=("head",),
    )


# --- 3. загрузка LHS из транспонированного вида SMEM -----------------------------
def case3_lhs_t(x):
    def body(xg, og, x_s, y_s):
        h = jax.lax.axis_index("head")
        x_s[...] = xg.at[h]
        lhs = plgpu.load(x_s.transpose((1, 0)), layout=plgpu.Layout.MMA_LHS(DT), optimized=False)
        rhs = plgpu.load(x_s.transpose((1, 0)), layout=plgpu.Layout.MMA_RHS(DT), optimized=False)
        acc = plgpu.mma(_acc((DK, C)), lhs, rhs)
        y_s[...] = acc
        og.at[h][...] = y_s[...]
    return plgpu.kernel(
        body,
        out_type=jax.ShapeDtypeStruct((2, DK, C), jnp.float32),
        scratch_types=[plgpu.SMEM((C, DK), DT), plgpu.SMEM((DK, C), jnp.float32)],
        compiler_params=plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Lane),
        grid=(2,), grid_names=("head",),
    )


# --- 4. построчная редукция (l2norm) на фрагменте --------------------------------
def case4_rowred(x):
    def body(xg, og, x_s, y_s):
        h = jax.lax.axis_index("head")
        x_s[...] = xg.at[h]
        xf = x_s[...].astype(jnp.float32)
        r = jnp.sum(xf * xf, axis=1)  # (C,)
        y_s[...] = (xf * (1.0 / jnp.sqrt(r + 1e-6))[:, None]).astype(DT)
        og.at[h][...] = y_s[...]
    return plgpu.kernel(
        body,
        out_type=jax.ShapeDtypeStruct((2, C, DK), DT),
        scratch_types=[plgpu.SMEM((C, DK), DT), plgpu.SMEM((C, DK), DT)],
        compiler_params=plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Lane),
        grid=(2,), grid_names=("head",),
    )


# --- 5. cumsum через нижнетреугольный mma (RHS транспонирован в памяти) ----------
def case5_cumsum(la):
    def body(lag, og, la_s, la_t_s, l_s, g_s):
        h = jax.lax.axis_index("head")
        la_s[...] = lag.at[h]
        frag = plgpu.load(la_s, layout=plgpu.Layout.MMA_LHS(DT), optimized=False)
        la_t_s.transpose((1, 0))[...] = frag
        # L (C,C) нижняя треугольная из единиц
        l_s[...] = jnp.tril(jnp.ones((C, C), DT))
        lhs = plgpu.load(l_s, layout=plgpu.Layout.MMA_LHS(DT), optimized=False)
        rhs = plgpu.load(la_t_s.T, layout=plgpu.Layout.MMA_RHS(DT), optimized=False)
        acc = plgpu.mma(_acc((C, DK)), lhs, rhs)
        g_s[...] = acc
        og.at[h][...] = g_s[...]
    return plgpu.kernel(
        body,
        out_type=jax.ShapeDtypeStruct((2, C, DK), jnp.float32),
        scratch_types=[plgpu.SMEM((C, DK), DT), plgpu.SMEM((DK, C), DT),
                       plgpu.SMEM((C, C), DT), plgpu.SMEM((C, DK), jnp.float32)],
        compiler_params=plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Lane),
        grid=(2,), grid_names=("head",),
    )


# --- 7. broadcast вектора по строкам SMEM -----------------------------------------
def case7_bcast(beta, a):
    def body(bg, ag, og, b_s, a_s, o_s):
        h = jax.lax.axis_index("head")
        b_s[...] = bg.at[h]
        a_s[...] = ag.at[h]
        bv = b_s[...].astype(jnp.float32)
        o_s[...] = a_s[...] * bv[:, None]
        og.at[h][...] = o_s[...]
    return plgpu.kernel(
        body,
        out_type=jax.ShapeDtypeStruct((2, C, C), jnp.float32),
        scratch_types=[plgpu.SMEM((C,), jnp.float32), plgpu.SMEM((C, C), jnp.float32),
                       plgpu.SMEM((C, C), jnp.float32)],
        compiler_params=plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Lane),
        grid=(2,), grid_names=("head",),
    )
