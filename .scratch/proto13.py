"""Прототипы 13: isоляция условия работы broadcast — pipeline vs plain, carry."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.experimental import pallas as pl

DT = jnp.bfloat16
C, DK = 64, 128
CP = plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Lane)
L1 = plgpu.Layout.MMA_ACC(DT, m_warps=1)


# Y11: как Y1, но внутри emit_pipeline-тела
def caseY11(x, b32):
    def body(_, x_s, b_s, o_s, inv):
        xv = plgpu.load(x_s, layout=L1, optimized=False).astype(jnp.float32)
        bv = plgpu.load(b_s, layout=L1, optimized=False).astype(jnp.float32)
        r = jnp.sum(bv, axis=1)
        o_s[...] = (xv * r[:, None]).astype(o_s.dtype)
        return inv
    def run(xg, bg, og):
        h = jax.lax.axis_index("head")
        xspec = plgpu.BlockSpec((C, DK), lambda i: (h, 0))
        bspec = plgpu.BlockSpec((C, 32), lambda i: (h, 0))
        ospec = plgpu.BlockSpec((C, DK), lambda i: (h, 0))
        acc0 = plgpu.layout_cast(jnp.zeros((C,), jnp.float32), L1.reduce(1))
        plgpu.emit_pipeline(body, grid=(1,), in_specs=[xspec, bspec], out_specs=[ospec],
                            max_concurrent_steps=1, init_carry=acc0)(xg, bg, og)
    run = plgpu.kernel(run, out_type=jax.ShapeDtypeStruct((4, C, DK), DT),
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(x, b32)


# Y12: r считается вне тела (init_carry), тело только умножает
def caseY12(x, b32):
    def body(_, x_s, o_s, inv):
        xv = plgpu.load(x_s, layout=L1, optimized=False).astype(jnp.float32)
        o_s[...] = (xv * inv[:, None]).astype(o_s.dtype)
        return inv
    def run(xg, bg, og, b_s):
        h = jax.lax.axis_index("head")
        b_s[...] = bg.at[h][...]
        bv = plgpu.load(b_s, layout=L1, optimized=False).astype(jnp.float32)
        r = plgpu.layout_cast(jnp.sum(bv, axis=1), L1.reduce(1))
        xspec = plgpu.BlockSpec((C, DK), lambda i: (h, 0))
        ospec = plgpu.BlockSpec((C, DK), lambda i: (h, 0))
        plgpu.emit_pipeline(body, grid=(1,), in_specs=[xspec], out_specs=[ospec],
                            max_concurrent_steps=1, init_carry=r)(xg, og)
    run = plgpu.kernel(run, out_type=jax.ShapeDtypeStruct((4, C, DK), DT),
                       scratch_types=[plgpu.SMEM((C, 32), jnp.float32)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(x, b32)


# Y13: plain-тело, но r считается вне и передаётся аргументом run_scoped? — просто
# проверим: plain-тело, r через init_carry-подобный приём (замыкание запрещено).
def caseY13(x, b32):
    def mk(r):
        def body(xg, og, x_s):
            h = jax.lax.axis_index("head")
            xv = plgpu.load(x_s, layout=L1, optimized=False).astype(jnp.float32)
            og.at[h][...] = (xv * r[:, None]).astype(DT)
        return body
    # r вычислим статически на хосте (не через ядро) — это тест broadcast-условия
    r = jnp.arange(C, dtype=jnp.float32)
    run = plgpu.kernel(mk(plgpu.layout_cast(r, L1.reduce(1))),
                       out_type=jax.ShapeDtypeStruct((4, C, DK), DT),
                       scratch_types=[plgpu.SMEM((C, DK), DT)],
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(x)


def case_inputs(name):
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (4, C, DK)).astype(DT)
    b32 = jax.random.normal(key, (4, C, 32)).astype(jnp.float32)
    return {"caseY11": (x, b32), "caseY12": (x, b32), "caseY13": (x, b32)}.get(name)
