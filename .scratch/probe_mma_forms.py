"""Probes: which SMEM/ACC/GMEM operand forms lower under sm_121 (CPU node).

Each probe is lowered separately; failures are reported per probe.
"""
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
    jfn = jax.jit(fn)
    return jfn.trace(*shapes).lower(lowering_platforms=("cuda",)).as_text()


def probe1_rhs_natural():
    """RHS loaded from a natural (k,n) SMEM buffer (no .T view)."""

    def body(qg, kg, og, k_s):
        qv = _l(qg, L)                     # (B,D) LHS: m=B, k=D
        kv = _l(k_s, R)                    # k_s (B,D): k=B? n=D -> probe intent: acc (B,D) += qv (B,D) @ ??
        acc = plgpu.mma(_zero((B, C)), _l(qg, L), _l(k_s.T, R))
        og[...] = acc.astype(f32)

    kern = plgpu.kernel(body,
                        out_type=(jax.ShapeDtypeStruct((B, C), f32),),
                        scratch_types=[plgpu.SMEM((C, D), dt)],
                        grid=(1,), grid_names=("t",))
    return lower(lambda q, k: kern(q, k),
                 jax.ShapeDtypeStruct((B, D), dt), jax.ShapeDtypeStruct((C, D), dt))


def probe2_rhs_transposed_view():
    """RHS from a .T view of an (n,k) buffer; LHS from a .T view (non-square)."""

    def body(qg, kg, og, ds_s):
        # score: a=(B,D) LHS natural, b=(k=D, n=C) from buffer (C,D) .T
        s = plgpu.mma(_zero((B, C)), _l(qg, L), _l(kg.T, R))
        # dk-like: a=(C,B) LHS from ds_s (B,C) .T view; b=qg (B,D) as RHS (k,n)=(B,D)
        dk = plgpu.mma(_zero((C, D)), _l(ds_s.T, L), _l(qg, R))
        og[...] = (s.astype(f32) + dk.astype(f32))

    kern = plgpu.kernel(body,
                        out_type=(jax.ShapeDtypeStruct((B, C), f32),),
                        scratch_types=[plgpu.SMEM((C, D), dt), plgpu.SMEM((B, C), dt)],
                        grid=(1,), grid_names=("t",))
    return lower(lambda q, k: kern(q, k),
                 jax.ShapeDtypeStruct((B, D), dt), jax.ShapeDtypeStruct((C, D), dt))


def probe3_acc_ops():
    """ACC value: max/exp/where/sum(axis=1)/broadcast/multiply family."""

    def body(qg, kg, og, pat_s):
        s = plgpu.mma(_zero((B, C)), _l(qg, L), _l(kg.T, R))
        ridx = plgpu.layout_cast(plgpu.broadcasted_iota(jnp.int32, (B, C), 0),
                                 plgpu.Layout.WG_STRIDED((B, C), 2))
        cidx = plgpu.layout_cast(plgpu.broadcasted_iota(jnp.int32, (B, C), 1),
                                 plgpu.Layout.WG_STRIDED((B, C), 2))
        keep = ridx >= cidx
        sm = jnp.where(keep, s, jnp.finfo(f32).min)
        m = jnp.max(sm, axis=1)            # (B,)
        p = jnp.exp(sm - m[:, None])
        l = jnp.sum(p, axis=1)
        rb = plgpu.layout_cast(jax.lax.broadcast_in_dim(l, (B, C), (0,)), A)
        o = plgpu.layout_cast(p / rb, A)
        og[...] = o.astype(f32)

    kern = plgpu.kernel(body,
                        out_type=(jax.ShapeDtypeStruct((B, C), f32),),
                        scratch_types=[plgpu.SMEM((C, D), dt), plgpu.SMEM((B, C), dt)],
                        grid=(1,), grid_names=("t",))
    return lower(lambda q, k: kern(q, k),
                 jax.ShapeDtypeStruct((B, D), dt), jax.ShapeDtypeStruct((C, D), dt))


def probe4_gmem_rmw():
    """GMEM read-modify-write: load GMEM tile into ACC, mma, store back."""

    def body(qg, kg, dqg, og, ds_s):
        acc = _l(dqg.at[0:B, 0:C], A)          # GMEM -> ACC load
        acc = plgpu.mma(acc, _l(qg, L), _l(kg, R))   # a (B,D), b (D,C) natural rhs
        dqg.at[0:B, 0:C][...] = acc            # ACC -> GMEM store
        og[...] = acc.astype(f32)

    kern = plgpu.kernel(body,
                        out_type=(jax.ShapeDtypeStruct((B, C), f32),),
                        scratch_types=[plgpu.SMEM((C, D), dt), plgpu.SMEM((B, C), dt)],
                        grid=(1,), grid_names=("t",))
    return lower(lambda q, k, dq: kern(q, k, dq),
                 jax.ShapeDtypeStruct((B, D), dt),
                 jax.ShapeDtypeStruct((D, C), dt),
                 jax.ShapeDtypeStruct((B, C), f32))


def probe5_acc_to_smem_roundtrip():
    """Store ACC (B,C) f32->bf16 into SMEM, reload as LHS natural and .T."""

    def body(qg, kg, vg, og, p_s, v_t):
        s = plgpu.mma(_zero((B, C)), _l(qg, L), _l(kg.T, R))
        p = jnp.exp(s * 0.1)
        p_s[...] = p.astype(dt)                      # ACC -> SMEM bf16 (B,C)
        # O += p @ v: b (k=C, n=D) from buffer v_t (D,C) .T view
        o1 = plgpu.mma(_zero((B, D)), _l(p_s, L), _l(v_t.T, R))
        # and the transposed use: a (C,B) LHS from p_s.T, b (B,D) RHS from qg natural
        o2 = plgpu.mma(_zero((C, D)), _l(p_s.T, L), _l(qg, R))
        og.at[0:B, 0:D][...] = o1
        og.at[B:B + C, 0:D][...] = o2.astype(f32)

    kern = plgpu.kernel(body,
                        out_type=(jax.ShapeDtypeStruct((B + C, D), f32),),
                        scratch_types=[plgpu.SMEM((B, C), dt), plgpu.SMEM((D, C), dt)],
                        grid=(1,), grid_names=("t",))
    return lower(lambda q, k, v: kern(q, k, v),
                 jax.ShapeDtypeStruct((B, D), dt), jax.ShapeDtypeStruct((C, D), dt),
                 jax.ShapeDtypeStruct((D, D), dt))


def probe6_pl_loop_dynamic():
    """Dynamic pl.loop + dynamic ds offsets + stats vectors in f32 SMEM."""

    def body(qg, og, m_s):
        n = qg.shape[0] // B

        def _iter(i):
            qv = _l(qg.at[pl.ds(i * B, B), 0:D], L)
            og.at[pl.ds(i * B, B), 0:D][...] = qv.astype(f32)
            m_s[...] = jnp.zeros((B,), f32)

        pl.loop(0, n)(_iter)

    kern = plgpu.kernel(body,
                        out_type=(jax.ShapeDtypeStruct((4 * B, D), f32),),
                        scratch_types=[plgpu.SMEM((B,), f32)],
                        grid=(1,), grid_names=("t",))
    return lower(lambda q: kern(q), jax.ShapeDtypeStruct((4 * B, D), dt))


PROBES = [probe1_rhs_natural, probe2_rhs_transposed_view, probe3_acc_ops,
          probe4_gmem_rmw, probe5_acc_to_smem_roundtrip, probe6_pl_loop_dynamic]


if __name__ == "__main__":
    import traceback
    for pr in PROBES:
        try:
            txt = pr()
            import re
            smem = re.findall(r"smem_bytes=(\d+)", txt)
            print(f"{pr.__name__}: OK, chars={len(txt)} smem={smem[:5]}")
        except Exception as e:
            msg = str(e).strip().splitlines()[-1] if str(e).strip() else repr(e)
            print(f"{pr.__name__}: FAIL {type(e).__name__}: {msg[:300]}")
