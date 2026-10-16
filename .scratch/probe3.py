"""Probes round 3: ACC ops via the SMEM-mask idiom; GMEM loads isolated."""
import sys
sys.path.insert(0, "/home/roman/experiments/agents/0710-ariadna/plugins/gb10/skills/pallas-gb10-kernel/scripts")
import lower_check
lower_check.patch_for_gb10()
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

dt = jnp.bfloat16
f32 = jnp.float32
B, D, C = 64, 128, 64
L = plgpu.Layout.MMA_LHS(dt, m_warps=4)
R = plgpu.Layout.MMA_RHS(dt, m_warps=4)
A = plgpu.Layout.MMA_ACC(dt, m_warps=4)

def _l(ref, layout):
    return plgpu.load(ref, layout=layout, optimized=False)

def _zero(shape):
    return plgpu.layout_cast(jnp.zeros(shape, f32), A)

def _iota(shape, axis):
    return plgpu.layout_cast(plgpu.broadcasted_iota(jnp.int32, shape, axis),
                             plgpu.Layout.WG_STRIDED(shape, 2))

def lower(fn, *shapes):
    return jax.jit(fn).trace(*shapes).lower(lowering_platforms=("cuda",)).as_text()

def probe3_acc_ops_masked():
    def body(qg, kg, og, k_s, pat_s):
        s = plgpu.mma(_zero((B, C)), _l(qg, L), _l(k_s.T, R))
        ridx, cidx = _iota((B, C), 0), _iota((B, C), 1)
        pat_s[...] = (ridx >= cidx).astype(dt)
        mk = _l(pat_s, A).astype(f32)
        sm = jnp.where(mk > 0, s, jnp.finfo(f32).min)
        m = jnp.max(sm, axis=1)
        mb = plgpu.layout_cast(jax.lax.broadcast_in_dim(m, (B, C), (0,)), A)
        p = jnp.exp(sm - mb)
        l = jnp.sum(p, axis=1)
        lb = plgpu.layout_cast(jax.lax.broadcast_in_dim(l, (B, C), (0,)), A)
        o = plgpu.layout_cast(p / lb, A)
        og[...] = o
    kern = plgpu.kernel(body, out_type=(jax.ShapeDtypeStruct((B, C), f32),),
                        scratch_types=[plgpu.SMEM((C, D), dt), plgpu.SMEM((B, C), dt)],
                        grid=(1,), grid_names=("t",))
    return lower(lambda q, k: kern(q, k),
                 jax.ShapeDtypeStruct((B, D), dt), jax.ShapeDtypeStruct((C, D), dt))

def probe4a_gmem_acc_rmw():
    def body(qg, kg, dqg, og, k_s):
        acc = _l(dqg.at[0:B, 0:C], A)
        acc = plgpu.mma(acc, _l(qg, L), _l(kg, R))
        dqg.at[0:B, 0:C][...] = acc
        og[...] = acc
    kern = plgpu.kernel(body, out_type=(jax.ShapeDtypeStruct((B, C), f32),),
                        scratch_types=[plgpu.SMEM((C, D), dt)],
                        grid=(1,), grid_names=("t",))
    return lower(lambda q, k, dq: kern(q, k, dq),
                 jax.ShapeDtypeStruct((B, D), dt),
                 jax.ShapeDtypeStruct((D, C), dt),
                 jax.ShapeDtypeStruct((B, C), f32))

def probe4b_gmem_reduce_load():
    def body(qg, kg, mg, og, k_s):
        acc = plgpu.mma(_zero((B, C)), _l(qg, L), _l(kg, R))
        mv = _l(mg.at[0:B], A.reduce(1))     # (B,) f32 GMEM with reduce layout
        rb = plgpu.layout_cast(jax.lax.broadcast_in_dim(mv, (B, C), (0,)), A)
        og[...] = acc * rb
    kern = plgpu.kernel(body, out_type=(jax.ShapeDtypeStruct((B, C), f32),),
                        scratch_types=[plgpu.SMEM((C, D), dt)],
                        grid=(1,), grid_names=("t",))
    return lower(lambda q, k, m: kern(q, k, m),
                 jax.ShapeDtypeStruct((B, D), dt),
                 jax.ShapeDtypeStruct((D, C), dt),
                 jax.ShapeDtypeStruct((B,), f32))

def probe4c_gmem_acc_load():
    def body(qg, kg, dqg, og, k_s):
        acc = plgpu.mma(_zero((B, C)), _l(qg, L), _l(kg, R))
        old = _l(dqg.at[0:B, 0:C], A)        # pure GMEM -> ACC load
        og[...] = acc + old
    kern = plgpu.kernel(body, out_type=(jax.ShapeDtypeStruct((B, C), f32),),
                        scratch_types=[plgpu.SMEM((C, D), dt)],
                        grid=(1,), grid_names=("t",))
    return lower(lambda q, k, dq: kern(q, k, dq),
                 jax.ShapeDtypeStruct((B, D), dt),
                 jax.ShapeDtypeStruct((D, C), dt),
                 jax.ShapeDtypeStruct((B, C), f32))

def probe4d_gmem_acc_store():
    def body(qg, kg, dqg, og, k_s):
        acc = plgpu.mma(_zero((B, C)), _l(qg, L), _l(kg, R))
        dqg.at[0:B, 0:C][...] = acc           # pure ACC -> GMEM store
        og[...] = acc
    kern = plgpu.kernel(body, out_type=(jax.ShapeDtypeStruct((B, C), f32),),
                        scratch_types=[plgpu.SMEM((C, D), dt)],
                        grid=(1,), grid_names=("t",))
    return lower(lambda q, k, dq: kern(q, k, dq),
                 jax.ShapeDtypeStruct((B, D), dt),
                 jax.ShapeDtypeStruct((D, C), dt),
                 jax.ShapeDtypeStruct((B, C), f32))

def probe8_full_fwd_tile():
    """The fwd inner tile, all together: S, mask, softmax-stats, O update."""
    def body(qg, kg, vg, og, k_s, v_t, p_s, pat_s, st_s):
        acc_o = _zero((B, D))
        qv = _l(qg, L)
        s = plgpu.mma(_zero((B, C)), qv, _l(k_s.T, R))
        ridx, cidx = _iota((B, C), 0), _iota((B, C), 1)
        pat_s[...] = (ridx >= cidx).astype(dt)
        mk = _l(pat_s, A).astype(f32)
        sm = jnp.where(mk > 0, s, jnp.finfo(f32).min)
        m = jnp.max(sm, axis=1)                       # (B,) reduce(1)
        mb = plgpu.layout_cast(jax.lax.broadcast_in_dim(m, (B, C), (0,)), A)
        p = jnp.exp(sm - mb)
        p_s[...] = p.astype(dt)
        acc_o = plgpu.mma(acc_o, _l(p_s, L), _l(v_t.T, R))
        l = jnp.sum(p, axis=1)
        st_s[...] = l
        og.at[0:B, 0:D][...] = acc_o / plgpu.layout_cast(
            jax.lax.broadcast_in_dim(l, (B, D), (0,)), A)
    kern = plgpu.kernel(body, out_type=(jax.ShapeDtypeStruct((B, D), f32),),
                        scratch_types=[plgpu.SMEM((C, D), dt), plgpu.SMEM((D, C), dt),
                                       plgpu.SMEM((B, C), dt), plgpu.SMEM((B, C), dt),
                                       plgpu.SMEM((B,), f32)],
                        grid=(1,), grid_names=("t",))
    return lower(lambda q, k, v: kern(q, k, v),
                 jax.ShapeDtypeStruct((B, D), dt), jax.ShapeDtypeStruct((C, D), dt),
                 jax.ShapeDtypeStruct((D, C), dt))

PROBES = [probe3_acc_ops_masked, probe4a_gmem_acc_rmw, probe4b_gmem_reduce_load,
          probe4c_gmem_acc_load, probe4d_gmem_acc_store, probe8_full_fwd_tile]

if __name__ == "__main__":
    for pr in PROBES:
        try:
            txt = pr()
            print(f"{pr.__name__}: OK, chars={len(txt)}")
        except Exception as e:
            msg = str(e).strip().splitlines()[-1] if str(e).strip() else repr(e)
            print(f"{pr.__name__}: FAIL {type(e).__name__}: {msg[:300]}")
