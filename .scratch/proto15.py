"""Прототипы 15: pipeline-контекст с (None, C, DK)-блоками — где broadcast работает."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.experimental import pallas as pl

DT = jnp.bfloat16
C, DK = 64, 128
CP = plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Lane)
L1 = plgpu.Layout.MMA_ACC(DT, m_warps=1)


def caseY20(x, b32):
    def body(_, x_s, b_s, o_s, inv):
        xv = plgpu.load(x_s, layout=L1, optimized=False).astype(jnp.float32)
        bv = plgpu.load(b_s, layout=L1, optimized=False).astype(jnp.float32)
        r = jnp.sum(bv, axis=1)
        o_s[...] = (xv * r[:, None]).astype(o_s.dtype)
        return inv
    def run(xg, bg, og):
        h = jax.lax.axis_index("head")
        xspec = plgpu.BlockSpec((None, C, DK), lambda i: (h, 0, 0))
        bspec = plgpu.BlockSpec((None, C, 32), lambda i: (h, 0, 0))
        ospec = plgpu.BlockSpec((None, C, DK), lambda i: (h, 0, 0))
        acc0 = plgpu.layout_cast(jnp.zeros((C,), jnp.float32), L1.reduce(1))
        plgpu.emit_pipeline(body, grid=(1,), in_specs=[xspec, bspec], out_specs=[ospec],
                            max_concurrent_steps=1, init_carry=acc0)(xg, bg, og)
    run = plgpu.kernel(run, out_type=jax.ShapeDtypeStruct((4, C, DK), DT),
                       compiler_params=CP, grid=(4,), grid_names=("head",))
    return run(x, b32)


def case_inputs(name):
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (4, C, DK)).astype(DT)
    b32 = jax.random.normal(key, (4, C, 32)).astype(jnp.float32)
    return {"caseY20": (x, b32)}.get(name)
