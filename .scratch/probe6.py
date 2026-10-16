import sys
sys.path.insert(0, "/home/roman/experiments/agents/0710-ariadna/plugins/gb10/skills/pallas-gb10-kernel/scripts")
import lower_check
lower_check.patch_for_gb10()
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

dt = jnp.bfloat16; f32 = jnp.float32; Bq, Bk, D, H = 64, 64, 128, 2
L = plgpu.Layout.MMA_LHS(dt, m_warps=4); R = plgpu.Layout.MMA_RHS(dt, m_warps=4)
A = plgpu.Layout.MMA_ACC(dt, m_warps=4)
_l = lambda ref, layout: plgpu.load(ref, layout=layout, optimized=False)
_zero = lambda shape: plgpu.layout_cast(jnp.zeros(shape, f32), A)
def _iota(shape, axis):
    return plgpu.layout_cast(plgpu.broadcasted_iota(jnp.int32, shape, axis),
                             plgpu.Layout.WG_STRIDED(shape, 2))

def probe_4d_fwd():
    nT = 256
    def body(qg, kg, vg, og, mg, lg, k_s, v_s, p_s, pat_s):
        prog = jax.lax.axis_index("program")
        b, h = prog // H, prog % H
        nq = nT // Bq
        ridx, cidx = _iota((Bq, Bk), 0), _iota((Bq, Bk), 1)
        pat_s[...] = (ridx >= cidx).astype(dt)
        def _step(qv, j, carry, diag):
            acc, m, l = carry
            k_s[...] = _l(kg.at[b, pl.ds(j * Bk, Bk), h, pl.ds(0, D)], A).astype(dt)
            v_s[...] = _l(vg.at[b, pl.ds(j * Bk, Bk), h, pl.ds(0, D)], A).astype(dt)
            s = plgpu.mma(_zero((Bq, Bk)), qv, _l(k_s.T, R))
            s = plgpu.layout_cast(s * 0.088, A)
            if diag:
                mk = _l(pat_s, A).astype(f32)
                sm = jnp.where(mk > 0, s, jnp.finfo(f32).min)
            else:
                sm = s
            mnew = jnp.maximum(m, jnp.max(sm, axis=1))
            alpha = jnp.exp(m - mnew)
            mb = plgpu.layout_cast(jax.lax.broadcast_in_dim(mnew, (Bq, Bk), (0,)), A)
            p = jnp.exp(sm - mb)
            l = l * alpha + jnp.sum(p, axis=1)
            p_s[...] = p.astype(dt)
            ab = plgpu.layout_cast(jax.lax.broadcast_in_dim(alpha, (Bq, D), (0,)), A)
            acc = plgpu.layout_cast(acc * ab, A)
            acc = plgpu.mma(acc, _l(p_s, L), _l(v_s, R))
            return acc, mnew, l
        def _tile(i):
            qv = _l(qg.at[b, pl.ds(i * Bq, Bq), h, pl.ds(0, D)], L)
            acc0 = _zero((Bq, D))
            m0 = plgpu.layout_cast(jnp.full((Bq,), jnp.finfo(f32).min, f32), A.reduce(1))
            l0 = plgpu.layout_cast(jnp.zeros((Bq,), f32), A.reduce(1))
            def _inner(j, carry):
                return _step(qv, j, carry, False)
            acc, m, l = pl.loop(0, i, init_carry=(acc0, m0, l0))(_inner)
            acc, m, l = _step(qv, i, (acc, m, l), True)
            lb = plgpu.layout_cast(jax.lax.broadcast_in_dim(l, (Bq, D), (0,)), A)
            og.at[b, pl.ds(i * Bq, Bq), h, pl.ds(0, D)][...] = acc / lb
            mg.at[b, h, pl.ds(i * Bq, Bq)][...] = m
            lg.at[b, h, pl.ds(i * Bq, Bq)][...] = l
        pl.loop(0, nq)(_tile)
    kern = plgpu.kernel(body,
        out_type=(jax.ShapeDtypeStruct((2, nT, H, D), f32),
                  jax.ShapeDtypeStruct((2, H, nT), f32),
                  jax.ShapeDtypeStruct((2, H, nT), f32)),
        scratch_types=[plgpu.SMEM((Bk, D), dt), plgpu.SMEM((Bk, D), dt),
                       plgpu.SMEM((Bq, Bk), dt), plgpu.SMEM((Bq, Bk), dt)],
        grid=(2 * H,), grid_names=("program",))
    return jax.jit(lambda q,k,v: kern(q,k,v)).trace(
        jax.ShapeDtypeStruct((2, nT, H, D), dt), jax.ShapeDtypeStruct((2, nT, H, D), dt),
        jax.ShapeDtypeStruct((2, nT, H, D), dt)).lower(lowering_platforms=("cuda",)).as_text()

def probe_4d_bwd():
    nT = 256
    def body(qg, kg, vg, dog, Dg, mg, lg, dqg, dkg, dvg,
             k_s, v_s, q_s, do_s, p_s, ds_s, pat_s):
        prog = jax.lax.axis_index("program")
        b, h = prog // H, prog % H
        nk = nT // Bk
        ridx, cidx = _iota((Bq, Bk), 0), _iota((Bq, Bk), 1)
        pat_s[...] = (ridx >= cidx).astype(dt)
        def _step(i, j, carry, diag):
            dk_acc, dv_acc = carry
            q_s[...] = _l(qg.at[b, pl.ds(i * Bq, Bq), h, pl.ds(0, D)], A).astype(dt)
            do_s[...] = _l(dog.at[b, pl.ds(i * Bq, Bq), h, pl.ds(0, D)], A).astype(dt)
            s = plgpu.mma(_zero((Bq, Bk)), _l(q_s, L), _l(k_s.T, R))
            s = plgpu.layout_cast(s * 0.088, A)
            if diag:
                mk = _l(pat_s, A).astype(f32)
                sm = jnp.where(mk > 0, s, jnp.finfo(f32).min)
            else:
                sm = s
            mi = _l(mg.at[b, h, pl.ds(i * Bq, Bq)], A.reduce(1))
            li = _l(lg.at[b, h, pl.ds(i * Bq, Bq)], A.reduce(1))
            Di = _l(Dg.at[b, h, pl.ds(i * Bq, Bq)], A.reduce(1))
            inv_l = 1.0 / li
            mb = plgpu.layout_cast(jax.lax.broadcast_in_dim(mi, (Bq, Bk), (0,)), A)
            lb = plgpu.layout_cast(jax.lax.broadcast_in_dim(inv_l, (Bq, Bk), (0,)), A)
            p = jnp.exp(sm - mb) * lb
            dp = plgpu.mma(_zero((Bq, Bk)), _l(do_s, L), _l(v_s.T, R))
            db = plgpu.layout_cast(jax.lax.broadcast_in_dim(Di, (Bq, Bk), (0,)), A)
            ds = jnp.where(mk > 0, p * (dp - db) * 0.088, 0.0) if diag else p * (dp - db) * 0.088
            p_s[...] = p.astype(dt)
            ds_s[...] = ds.astype(dt)
            dk_acc = plgpu.mma(dk_acc, _l(ds_s.T, L), _l(q_s, R))
            dv_acc = plgpu.mma(dv_acc, _l(p_s.T, L), _l(do_s, R))
            accq = _l(dqg.at[b, pl.ds(i * Bq, Bq), h, pl.ds(0, D)], A)
            accq = plgpu.mma(accq, _l(ds_s, L), _l(k_s, R))
            dqg.at[b, pl.ds(i * Bq, Bq), h, pl.ds(0, D)][...] = accq
            return dk_acc, dv_acc
        def _outer(j):
            k_s[...] = _l(kg.at[b, pl.ds(j * Bk, Bk), h, pl.ds(0, D)], A).astype(dt)
            v_s[...] = _l(vg.at[b, pl.ds(j * Bk, Bk), h, pl.ds(0, D)], A).astype(dt)
            dk0 = _zero((Bk, D)); dv0 = _zero((Bk, D))
            dk_acc, dv_acc = _step(j, j, (dk0, dv0), True)
            def _inner(i, carry):
                return _step(i, j, carry, False)
            dk_acc, dv_acc = pl.loop(j + 1, nk, init_carry=(dk_acc, dv_acc))(_inner)
            dkg.at[b, pl.ds(j * Bk, Bk), h, pl.ds(0, D)][...] = dk_acc
            dvg.at[b, pl.ds(j * Bk, Bk), h, pl.ds(0, D)][...] = dv_acc
        pl.loop(0, nk)(_outer)
    kern = plgpu.kernel(body,
        out_type=(jax.ShapeDtypeStruct((2, nT, H, D), f32),
                  jax.ShapeDtypeStruct((2, nT, H, D), f32),
                  jax.ShapeDtypeStruct((2, nT, H, D), f32)),
        scratch_types=[plgpu.SMEM((Bk, D), dt), plgpu.SMEM((Bk, D), dt),
                       plgpu.SMEM((Bq, D), dt), plgpu.SMEM((Bq, D), dt),
                       plgpu.SMEM((Bq, Bk), dt), plgpu.SMEM((Bq, Bk), dt),
                       plgpu.SMEM((Bq, Bk), dt)],
        grid=(2 * H,), grid_names=("program",))
    return jax.jit(lambda q,k,v,do,D,m,l: kern(q,k,v,do,D,m,l)).trace(
        jax.ShapeDtypeStruct((2, nT, H, D), dt), jax.ShapeDtypeStruct((2, nT, H, D), dt),
        jax.ShapeDtypeStruct((2, nT, H, D), dt), jax.ShapeDtypeStruct((2, nT, H, D), dt),
        jax.ShapeDtypeStruct((2, H, nT), f32), jax.ShapeDtypeStruct((2, H, nT), f32),
        jax.ShapeDtypeStruct((2, H, nT), f32)).lower(lowering_platforms=("cuda",)).as_text()

for name, f in [("4d_fwd", probe_4d_fwd), ("4d_bwd", probe_4d_bwd)]:
    try:
        txt = f()
        import re
        smem = sorted(set(int(x) for x in re.findall(r"smem_bytes=(\d+)", txt)))
        print(name, "OK chars=", len(txt), "smem_bytes seen:", smem[:4])
    except Exception as e:
        print(name, "FAIL:", type(e).__name__, str(e).strip().splitlines()[-1][:400])
