import sys, traceback
sys.path.insert(0, "/home/roman/experiments/agents/0710-ariadna/plugins/gb10/skills/pallas-gb10-kernel/scripts")
import lower_check
lower_check.patch_for_gb10()
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

dt = jnp.bfloat16; f32 = jnp.float32; B, D, C = 64, 128, 64
L = plgpu.Layout.MMA_LHS(dt, m_warps=4); R = plgpu.Layout.MMA_RHS(dt, m_warps=4)
A = plgpu.Layout.MMA_ACC(dt, m_warps=4)
_l = lambda ref, layout: plgpu.load(ref, layout=layout, optimized=False)
_zero = lambda shape: plgpu.layout_cast(jnp.zeros(shape, f32), A)

def va():
    """ACC -> GMEM store into ds() slice (KDA style); GMEM -> ACC load back."""
    def body(qg, kg, dqg, k_s):
        qv = _l(qg.at[pl.ds(0, B), pl.ds(0, D)], L)
        acc = plgpu.mma(_zero((B, C)), qv, _l(kg, R))
        dqg.at[pl.ds(64, B), pl.ds(0, C)][...] = acc        # store (no prior write)
        old = _l(dqg.at[pl.ds(0, B), pl.ds(0, C)], A)       # load (separate region)
        dqg.at[pl.ds(128, B), pl.ds(0, C)][...] = acc + old
    kern = plgpu.kernel(body, out_type=(),
                        scratch_types=[plgpu.SMEM((C, D), dt)],
                        grid=(1,), grid_names=("t",))
    return jax.jit(lambda q,k,dq: kern(q,k,dq)).trace(
        jax.ShapeDtypeStruct((256, D), dt), jax.ShapeDtypeStruct((D, C), dt),
        jax.ShapeDtypeStruct((256, C), f32)).lower(lowering_platforms=("cuda",))

def vb():
    """GMEM (B,) f32 -> reduce(1) load."""
    def body(qg, mg, og, k_s):
        qv = _l(qg.at[pl.ds(0, B), pl.ds(0, D)], L)
        acc = plgpu.mma(_zero((B, C)), qv, _l(k_s.T, R))
        mv = _l(mg.at[pl.ds(0, B)], A.reduce(1))
        rb = plgpu.layout_cast(jax.lax.broadcast_in_dim(mv, (B, C), (0,)), A)
        og[...] = acc * rb
    kern = plgpu.kernel(body, out_type=(jax.ShapeDtypeStruct((B, C), f32),),
                        scratch_types=[plgpu.SMEM((C, D), dt)],
                        grid=(1,), grid_names=("t",))
    return jax.jit(lambda q,m: kern(q,m)).trace(
        jax.ShapeDtypeStruct((256, D), dt), jax.ShapeDtypeStruct((128,), f32)
    ).lower(lowering_platforms=("cuda",))

for name, f in [("va", va), ("vb", vb)]:
    try:
        f(); print(name, "OK")
    except Exception as e:
        print(name, "FAIL:", type(e).__name__, str(e).strip().splitlines()[-1][:200])
