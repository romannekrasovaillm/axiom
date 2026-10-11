"""Прототипы 14: isoляция broadcast — константа с reduce-раскладкой vs reduction."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.experimental import pallas as pl

DT = jnp.bfloat16
C, DK = 64, 128
CP = plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Lane)
L1 = plgpu.Layout.MMA_ACC(DT, m_warps=1)


# W1: broadcast константы с reduce-раскладкой (не из редукции)
def caseW1(x):
    def body(xg, og, x_s, o_s):
        h = jax.lax.axis_index("head")
        x_s[...] = xg.at[h][...]
        xv = plgpu.load(x_s, layout=L1, optimized=False).astype(jnp.float32)
        r = plgpu.layout_cast(jnp.ones((C,), jnp.float32), L1.reduce(1))
        o_s[...] = (xv * r[:, None]).astype(DT)
        og.at[h][...] = o_s[...]
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, DK), DT),
                       scratch_types=[plgpu.SMEM((C, DK), DT), plgpu.SMEM((C, DK), DT)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(x)


# W2: то же, но вектор приходит из GMEM-загрузки формы (C,) — проверим и такой путь
def caseW2(x, b1):
    def body(xg, bg, og, x_s, b_s, o_s):
        h = jax.lax.axis_index("head")
        x_s[...] = xg.at[h][...]
        b_s[...] = bg.at[h][...]
        xv = plgpu.load(x_s, layout=L1, optimized=False).astype(jnp.float32)
        r = plgpu.load(b_s, layout=L1.reduce(1), optimized=False).astype(jnp.float32)
        o_s[...] = (xv * r[:, None]).astype(DT)
        og.at[h][...] = o_s[...]
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, DK), DT),
                       scratch_types=[plgpu.SMEM((C, DK), DT), plgpu.SMEM((C, 32), jnp.float32),
                                      plgpu.SMEM((C, DK), DT)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(x, b1)


# W3: редукция + явный layout_cast до rsqrt (как в шаблоне: rsqrt(ss*...) )
def caseW3(x, b32):
    def body(xg, bg, og, x_s, b_s, o_s):
        h = jax.lax.axis_index("head")
        x_s[...] = xg.at[h][...]
        b_s[...] = bg.at[h][...]
        xv = plgpu.load(x_s, layout=L1, optimized=False).astype(jnp.float32)
        bv = plgpu.load(b_s, layout=L1, optimized=False).astype(jnp.float32)
        ss = plgpu.layout_cast(jnp.sum(bv, axis=1), L1.reduce(1))
        inv = jax.lax.rsqrt(ss * (1.0 / DK) + 1e-6)
        o_s[...] = (xv * inv[:, None]).astype(DT)
        og.at[h][...] = o_s[...]
    run = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((4, C, DK), DT),
                       scratch_types=[plgpu.SMEM((C, DK), DT), plgpu.SMEM((C, 32), jnp.float32),
                                      plgpu.SMEM((C, DK), DT)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(x, b32)


def case_inputs(name):
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (4, C, DK)).astype(DT)
    b32 = jax.random.normal(key, (4, C, 32)).astype(jnp.float32)
    b1 = jax.random.normal(key, (4, C, 32)).astype(jnp.float32)[:, :, :]
    return {"caseW1": (x,), "caseW2": (x, b1), "caseW3": (x, b32)}.get(name)
