"""Прототипы 3: beta-вектор, редукции, иота-маски."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental.pallas import mosaic_gpu as plgpu

DT = jnp.bfloat16
C, DK = 64, 128
CP = plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Lane)


# B1: beta (C,) -> broadcast по строкам в (C,C)
def caseB1(beta, a):
    def body(bg, ag, og, b_s, a_s):
        h = jax.lax.axis_index("head")
        b_s[...] = bg.at[h][...]
        a_s[...] = ag.at[h][...]
        og.at[h][...] = a_s[...] * b_s[...][:, None]
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, C), jnp.float32),
                       scratch_types=[plgpu.SMEM((C,), jnp.float32), plgpu.SMEM((C, C), jnp.float32)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(beta, a)


# B2: beta упакована в (C,8): чтение/запись
def caseB2(beta8):
    def body(bg, og, b_s):
        h = jax.lax.axis_index("head")
        b_s[...] = bg.at[h][...]
        og.at[h][...] = b_s[...] * 2.0
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, 8), jnp.float32),
                       scratch_types=[plgpu.SMEM((C, 8), jnp.float32)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(beta8)


# B3: (C,) чтение и запись обратно
def caseB3(beta):
    def body(bg, og, b_s):
        h = jax.lax.axis_index("head")
        b_s[...] = bg.at[h][...]
        og.at[h][...] = b_s[...]
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C), jnp.float32),
                       scratch_types=[plgpu.SMEM((C,), jnp.float32)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(beta)


# B4: l2norm через mma: sq @ ones -> (C,8), затем broadcast (C,8)->(C,DK)
def caseB4(x):
    def body(xg, og, x_s, one_s):
        h = jax.lax.axis_index("head")
        x_s[...] = xg.at[h][...]
        one_s[...] = jnp.ones((DK, 8), DT)
        sq = (x_s[...].astype(jnp.float32) ** 2).astype(DT)
        lhs = plgpu.load(sq, layout=None, optimized=False) if False else sq
        # упрощённый teст: только редукция через mma невозможна без layout; проверим broadcast
        og.at[h][...] = jnp.broadcast_to(jnp.sum(x_s[...].astype(jnp.float32), axis=1)[:, None], (C, DK)).astype(DT)
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, DK), DT),
                       scratch_types=[plgpu.SMEM((C, DK), DT), plgpu.SMEM((DK, 8), DT)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(x)


# B5: загрузка в MMA_ACC-раскладку + построчная редукция
def caseB5(x):
    def body(xg, og, x_s):
        h = jax.lax.axis_index("head")
        x_s[...] = xg.at[h][...]
        xv = plgpu.load(x_s, layout=plgpu.Layout.MMA_ACC(DT), optimized=False)
        r = jnp.sum(xv.astype(jnp.float32) ** 2, axis=1)
        og.at[h][...] = r
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C), jnp.float32),
                       scratch_types=[plgpu.SMEM((C, DK), DT)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(x)


# B6: маска через layout_cast от константы (iota-класс)
def caseB6(a):
    def body(ag, og, a_s):
        h = jax.lax.axis_index("head")
        a_s[...] = ag.at[h][...]
        mask = plgpu.layout_cast(jnp.tril(jnp.ones((C, C), DT)), plgpu.Layout.MMA_ACC(DT))
        og.at[h][...] = jnp.where(mask, a_s[...], 0.0)
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, C), jnp.float32),
                       scratch_types=[plgpu.SMEM((C, C), jnp.float32)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(a)


def case_inputs(name):
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (4, C, DK)).astype(DT)
    beta = jax.random.normal(key, (4, C)).astype(jnp.float32)
    beta8 = jax.random.normal(key, (4, C, 8)).astype(jnp.float32)
    a = jax.random.normal(key, (4, C, C)).astype(jnp.float32)
    return {"caseB1": (beta, a), "caseB2": (beta8,), "caseB3": (beta,),
            "caseB4": (x,), "caseB5": (x,), "caseB6": (a,)}.get(name)
