import sys
sys.path.insert(0, "/home/roman/experiments/agents/0710-ariadna/plugins/gb10/skills/pallas-gb10-kernel/scripts")
import lower_check
lower_check.patch_for_gb10()
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
dt = jnp.bfloat16; f32 = jnp.float32; Bq = Bk = 64
L = plgpu.Layout.MMA_LHS(dt, m_warps=4); R = plgpu.Layout.MMA_RHS(dt, m_warps=4)
A = plgpu.Layout.MMA_ACC(dt, m_warps=4)
_l = lambda ref, layout: plgpu.load(ref, layout=layout, optimized=False)
_zero = lambda shape: plgpu.layout_cast(jnp.zeros(shape, f32), A)
def lower(fn, *shapes):
    return jax.jit(fn).trace(*shapes).lower(lowering_platforms=("cuda",)).as_text()

def variant_iota_direct():
    def body(qg, kg, og, k_s):
        s = plgpu.mma(_zero((Bq, Bk)), _l(qg.at[0:Bq, 0:128], L), _l(k_s.T, R))
        ridx = plgpu.broadcasted_iota(jnp.int32, (Bq, Bk), 0)
        cidx = plgpu.broadcasted_iota(jnp.int32, (Bq, Bk), 1)
        mk = plgpu.layout_cast((ridx >= cidx).astype(f32), A)
        sm = jnp.where(mk > 0, s, jnp.finfo(f32).min)
        og[...] = sm
    kern = plgpu.kernel(body, out_type=(jax.ShapeDtypeStruct((Bq, Bk), f32),),
                        scratch_types=[plgpu.SMEM((Bk, 128), dt)], grid=(1,), grid_names=("t",))
    return lower(lambda q, k: kern(q, k), jax.ShapeDtypeStruct((Bq, 128), dt), jax.ShapeDtypeStruct((Bk, 128), dt))

def variant_iota_strided_then_cast():
    def body(qg, kg, og, k_s):
        s = plgpu.mma(_zero((Bq, Bk)), _l(qg.at[0:Bq, 0:128], L), _l(k_s.T, R))
        g = plgpu.Layout.WG_STRIDED((Bq, Bk), 2)
        ridx = plgpu.layout_cast(plgpu.broadcasted_iota(jnp.int32, (Bq, Bk), 0), g)
        cidx = plgpu.layout_cast(plgpu.broadcasted_iota(jnp.int32, (Bq, Bk), 1), g)
        mk = plgpu.layout_cast((ridx >= cidx).astype(f32), A)
        sm = jnp.where(mk > 0, s, jnp.finfo(f32).min)
        og[...] = sm
    kern = plgpu.kernel(body, out_type=(jax.ShapeDtypeStruct((Bq, Bk), f32),),
                        scratch_types=[plgpu.SMEM((Bk, 128), dt)], grid=(1,), grid_names=("t",))
    return lower(lambda q, k: kern(q, k), jax.ShapeDtypeStruct((Bq, 128), dt), jax.ShapeDtypeStruct((Bk, 128), dt))

for nm, f in [("iota_direct", variant_iota_direct), ("strided_then_cast", variant_iota_strided_then_cast)]:
    try:
        f(); print(nm, "OK")
    except Exception as e:
        print(nm, "FAIL:", type(e).__name__, str(e).strip().splitlines()[-1][:150])
