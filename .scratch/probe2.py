"""Probes round 2: isolate the uncertain idioms (wiring fixed)."""
import sys

sys.path.insert(0, "/home/roman/experiments/agents/0710-ariadna/plugins/gb10/skills/pallas-gb10-kernel/scripts")
import lower_check  # noqa: E402

lower_check.patch_for_gb10()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.experimental import pallas as pl  # noqa: E402
from jax.experimental.pallas import mosaic_gpu as plgpu  # noqa: E402

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


def lower(fn, *shapes):
    return jax.jit(fn).trace(*shapes).lower(lowering_platforms=("cuda",)).as_text()


def probe2_lhs_transposed():
    """LHS from a .T view of an (n,k) SMEM buffer (non-square)."""

    def body(qg, og, ds_s, q_s):
        # dk-like: a (C,B) LHS from ds_s (B,C) .T view; b (B,D) RHS from q_s (B,D)
        acc = plgpu.mma(_zero((C, D)), _l(ds_s.T, L), _l(q_s, R))
        og.at[0:C, 0:D][...] = acc

    kern = plgpu.kernel(body,
                        out_type=(jax.ShapeDtypeStruct((C, D), f32),),
                        scratch_types=[plgpu.SMEM((B, C), dt), plgpu.SMEM((B, D), dt)],
                        grid=(1,), grid_names=("t",))
    return lower(lambda q: kern(q), jax.ShapeDtypeStruct((B, D), dt))


def probe3_acc_ops():
    """ACC: where/exp/max(axis=1)/sum(axis=1)/broadcast/mul/div."""

    def body(qg, og, k_s):
        s = plgpu.mma(_zero((B, C)), _l(qg, L), _l(k_s.T, R))
        ridx = plgpu.layout_cast(plgpu.broadcasted_iota(jnp.int32, (B, C), 0),
                                 plgpu.Layout.WG_STRIDED((B, C), 2))
        cidx = plgpu.layout_cast(plgpu.broadcasted_iota(jnp.int32, (B, C), 1),
                                 plgpu.Layout.WG_STRIDED((B, C), 2))
        keep = ridx >= cidx
        sm = jnp.where(keep, s, jnp.finfo(f32).min)
        m = jnp.max(sm, axis=1)
        p = jnp.exp(sm - m[:, None])
        l = jnp.sum(p, axis=1)
        rb = plgpu.layout_cast(jax.lax.broadcast_in_dim(l, (B, C), (0,)), A)
        o = plgpu.layout_cast(p / rb, A)
        og[...] = o

    kern = plgpu.kernel(body,
                        out_type=(jax.ShapeDtypeStruct((B, C), f32),),
                        scratch_types=[plgpu.SMEM((C, D), dt)],
                        grid=(1,), grid_names=("t",))
    return lower(lambda q: kern(q), jax.ShapeDtypeStruct((B, D), dt))


def probe4_gmem_rmw():
    """GMEM -> ACC load, mma, store back; plus f32 GMEM load with reduce layout."""

    def body(qg, kg, dqg, mg, og, k_s):
        acc = _l(dqg.at[0:B, 0:C], A)
        acc = plgpu.mma(acc, _l(qg, L), _l(kg, R))
        dqg.at[0:B, 0:C][...] = acc
        mv = _l(mg.at[0:B], A.reduce(1))       # (B,) fp32 from GMEM with reduce layout
        rb = plgpu.layout_cast(jax.lax.broadcast_in_dim(mv, (B, C), (0,)), A)
        og[...] = acc * rb

    kern = plgpu.kernel(body,
                        out_type=(jax.ShapeDtypeStruct((B, C), f32),),
                        scratch_types=[plgpu.SMEM((C, D), dt)],
                        grid=(1,), grid_names=("t",))
    return lower(lambda q, k, dq, m: kern(q, k, dq, m),
                 jax.ShapeDtypeStruct((B, D), dt),
                 jax.ShapeDtypeStruct((D, C), dt),
                 jax.ShapeDtypeStruct((B, C), f32),
                 jax.ShapeDtypeStruct((B,), f32))


def probe5a_smem_transpose_rhs():
    """O += p @ v: b (k=C, n=D) from a (D,C) buffer .T view (the fwd v-transpose)."""

    def body(qg, kg, og, p_s, v_t):
        s = plgpu.mma(_zero((B, C)), _l(qg, L), _l(kg.T, R))
        p_s[...] = jnp.exp(s * 0.1).astype(dt)
        o = plgpu.mma(_zero((B, D)), _l(p_s, L), _l(v_t.T, R))
        og[...] = o

    kern = plgpu.kernel(body,
                        out_type=(jax.ShapeDtypeStruct((B, D), f32),),
                        scratch_types=[plgpu.SMEM((B, C), dt), plgpu.SMEM((D, C), dt)],
                        grid=(1,), grid_names=("t",))
    return lower(lambda q, k: kern(q, k),
                 jax.ShapeDtypeStruct((B, D), dt), jax.ShapeDtypeStruct((C, D), dt))


def probe5b_smem_transpose_lhs():
    """dv += p^T @ do: a (C,B) from p_s (B,C) .T; b (B,D) from do_s natural (k,n)."""

    def body(qg, dog, og, p_s, do_s):
        s = plgpu.mma(_zero((B, C)), _l(qg, L), _l(qg, R))
        p_s[...] = jnp.exp(s * 0.1).astype(dt)
        dv = plgpu.mma(_zero((C, D)), _l(p_s.T, L), _l(do_s, R))
        og.at[0:C, 0:D][...] = dv

    kern = plgpu.kernel(body,
                        out_type=(jax.ShapeDtypeStruct((C, D), f32),),
                        scratch_types=[plgpu.SMEM((B, C), dt), plgpu.SMEM((B, D), dt)],
                        grid=(1,), grid_names=("t",))
    return lower(lambda q, d: kern(q, d),
                 jax.ShapeDtypeStruct((B, D), dt), jax.ShapeDtypeStruct((B, D), dt))


def probe6_pl_loop():
    """Dynamic pl.loop, dynamic offsets, carry through init (no carry: closure)."""

    def body(qg, og, k_s):
        n = qg.shape[0] // B

        def _iter(i):
            qv = _l(qg.at[pl.ds(i * B, B), 0:D], L)
            s = plgpu.mma(_zero((B, C)), qv, _l(k_s.T, R))
            og.at[pl.ds(i * B, B), 0:C][...] = s

        pl.loop(0, n)(_iter)

    kern = plgpu.kernel(body,
                        out_type=(jax.ShapeDtypeStruct((4 * B, C), f32),),
                        scratch_types=[plgpu.SMEM((C, D), dt)],
                        grid=(1,), grid_names=("t",))
    return lower(lambda q: kern(q), jax.ShapeDtypeStruct((4 * B, D), dt))


def probe7_zero_acc_repeat():
    """Two sequential mma chains + a loop-carried ACC value (register carry)."""
    def body(qg, og, k_s):
        acc = _zero((B, C))

        def _iter(i, carry):
            qv = _l(qg.at[pl.ds(i * B, B), 0:D], L)
            return plgpu.mma(carry, qv, _l(k_s.T, R))

        out = pl.loop(0, 4, init_carry=acc)(_iter)
        og[...] = out

    kern = plgpu.kernel(body,
                        out_type=(jax.ShapeDtypeStruct((B, C), f32),),
                        scratch_types=[plgpu.SMEM((C, D), dt)],
                        grid=(1,), grid_names=("t",))
    return lower(lambda q: kern(q), jax.ShapeDtypeStruct((4 * B, D), dt))


PROBES = [probe2_lhs_transposed, probe3_acc_ops, probe4_gmem_rmw,
          probe5a_smem_transpose_rhs, probe5b_smem_transpose_lhs,
          probe6_pl_loop, probe7_zero_acc_repeat]

if __name__ == "__main__":
    for pr in PROBES:
        try:
            txt = pr()
            print(f"{pr.__name__}: OK, chars={len(txt)}")
        except Exception as e:
            msg = str(e).strip().splitlines()[-1] if str(e).strip() else repr(e)
            print(f"{pr.__name__}: FAIL {type(e).__name__}: {msg[:400]}")
