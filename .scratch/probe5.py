import sys
sys.path.insert(0, "/home/roman/experiments/agents/0710-ariadna/plugins/gb10/skills/pallas-gb10-kernel/scripts")
import lower_check
lower_check.patch_for_gb10()
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

dt = jnp.bfloat16; f32 = jnp.float32; Bq, Bk, D = 64, 64, 128
L = plgpu.Layout.MMA_LHS(dt, m_warps=4); R = plgpu.Layout.MMA_RHS(dt, m_warps=4)
A = plgpu.Layout.MMA_ACC(dt, m_warps=4)
_l = lambda ref, layout: plgpu.load(ref, layout=layout, optimized=False)
_zero = lambda shape: plgpu.layout_cast(jnp.zeros(shape, f32), A)
def _iota(shape, axis):
    return plgpu.layout_cast(plgpu.broadcasted_iota(jnp.int32, shape, axis),
                             plgpu.Layout.WG_STRIDED(shape, 2))

def probe_mini_fwd():
    nT = 256
    def body(qg, kg, vg, og, mg_, lg_, k_s, v_s, p_s, pat_s):
        prog = jax.lax.axis_index("prog")
        b, h = prog // 2, prog % 2
        nq = nT // Bq
        ridx, cidx = _iota((Bq, Bk), 0), _iota((Bq, Bk), 1)
        def _outer(i):
            qv = _l(qg.at[b, pl.ds(i * Bq, Bq), pl.ds(h * D, D)], L)
            acc0 = _zero((Bq, D))
            m0 = plgpu.layout_cast(jnp.full((Bq,), jnp.finfo(f32).min, f32), A.reduce(1))
            l0 = plgpu.layout_cast(jnp.zeros((Bq,), f32), A.reduce(1))
            def _inner(j, carry):
                acc, m, l = carry
                k_s[...] = _l(kg.at[b, pl.ds(j * Bk, Bk), pl.ds(h * D, D)], A).astype(dt)
                v_s[...] = _l(vg.at[b, pl.ds(j * Bk, Bk), pl.ds(h * D, D)], A).astype(dt)
                s = plgpu.mma(_zero((Bq, Bk)), qv, _l(k_s.T, R))
                pat_s[...] = (ridx >= cidx).astype(dt)
                mk = _l(pat_s, A).astype(f32)
                sm = jnp.where(mk > 0, s, jnp.finfo(f32).min)
                mnew = jnp.maximum(m, jnp.max(sm, axis=1))
                alpha = jnp.exp(m - mnew)
                mb = plgpu.layout_cast(jax.lax.broadcast_in_dim(mnew, (Bq, Bk), (0,)), A)
                p = jnp.exp(sm - mb)
                l = l * alpha + jnp.sum(p, axis=1)
                p_s[...] = p.astype(dt)
                ab = plgpu.layout_cast(jax.lax.broadcast_in_dim(alpha, (Bq, D), (0,)), A)
                acc = plgpu.layout_cast(acc * ab, A)
                acc = plgpu.mma(acc, _l(p_s, L), _l(v_s, R))   # natural (k,n) RHS
                return acc, mnew, l
            acc, m, l = pl.loop(0, nq, init_carry=(acc0, m0, l0))(_inner)
            lb2 = plgpu.layout_cast(jax.lax.broadcast_in_dim(l, (Bq, D), (0,)), A)
            og.at[b, pl.ds(i * Bq, Bq), pl.ds(h * D, D)][...] = acc / lb2
            mg_.at[b, h, pl.ds(i * Bq, Bq)][...] = m
            lg_.at[b, h, pl.ds(i * Bq, Bq)][...] = l
        pl.loop(0, nq)(_outer)
    kern = plgpu.kernel(body,
        out_type=(jax.ShapeDtypeStruct((2, nT, 2 * D), f32),
                  jax.ShapeDtypeStruct((2, 2, nT), f32),
                  jax.ShapeDtypeStruct((2, 2, nT), f32)),
        scratch_types=[plgpu.SMEM((Bk, D), dt), plgpu.SMEM((Bk, D), dt),
                       plgpu.SMEM((Bq, Bk), dt), plgpu.SMEM((Bq, Bk), dt)],
        grid=(4,), grid_names=("prog",))
    return jax.jit(lambda q,k,v: kern(q,k,v)).trace(
        jax.ShapeDtypeStruct((2, nT, 2*D), dt), jax.ShapeDtypeStruct((2, nT, 2*D), dt),
        jax.ShapeDtypeStruct((2, nT, 2*D), dt)).lower(lowering_platforms=("cuda",)).as_text()

try:
    txt = probe_mini_fwd()
    import re
    print("mini_fwd OK, chars=", len(txt), "for-loops:", len(re.findall(r"scf\.for", txt)))
except Exception as e:
    print("mini_fwd FAIL:", type(e).__name__, str(e).strip().splitlines()[-1][:300])
