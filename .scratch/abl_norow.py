"""Row-wise RMSNorm on GB10 via Pallas Mosaic GPU: y = x * rsqrt(mean(x^2) + eps) * w.

Pattern: row reduction + broadcast (RMSNorm/LayerNorm/softmax-style), any hidden
size (multiple of bn).
  * persistent grid: `programs` CUDA blocks, each loops over row blocks of 16 rows
  * pass 1: emit_pipeline over column chunks, carry = per-row sum of squares
  * pass 2: emit_pipeline over the same chunks, normalize + scale, TMA store.
    The re-read of a 16 x n row block (n=8192 bf16: 256 KB per block,
    ~12 MB for 48 blocks) mostly hits the 24 MB L2, so HBM traffic stays ~2 passes.

Layout notes (learned by lowering for sm_121, JAX 0.11.2):
  * values are loaded with explicit register layouts: rows use
    Layout.MMA_ACC(bf16, m_warps=1) (16-row tile), per-row scalars use
    .reduce(1), the weight vector uses .reduce(0) -> broadcasts are free.
  * values that cross an emit_pipeline boundary must go through `init_carry`
    (a closed-over array loses its layout and inference fails).
  * a value in MMA_ACC layout cannot be stored into a *tiled* SMEM ref: the
    output BlockSpec is left untiled; the input ones use TilingTransform((8, 64)).
  * every TMA box dimension must be <= 256 elements (including the 1-D weight).

Validated by lowering only; run check_kernel.py on the device.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

SMEM_LIMIT = 99 * 1024
BM = 16  # row tile fixed by the register layout below


def _store(ref, value):
    """`plgpu.store(..., optimized=False)` on newer JAX, plain assignment on 0.11.2."""
    st = getattr(plgpu, "store", None)
    if st is not None:
        st(ref, value, optimized=False)
    else:
        ref[...] = value


@functools.partial(jax.jit, static_argnames=("bn", "stages", "programs", "eps"))
def kernel(x, w, *, bn: int = 256, stages: int = 2, programs: int = 48, eps: float = 1e-6):
    m, n = x.shape
    if m % BM or n % bn or bn % 64 or bn > 256:
        raise ValueError(f"need m % {BM} == 0, n % bn == 0, bn in (64, 128, 192, 256); got {x.shape}, bn={bn}")
    row_blocks = m // BM
    programs = min(programs, row_blocks)
    L = plgpu.Layout.MMA_ACC(jnp.bfloat16, m_warps=1)  # layout only; works for any dtype values
    tile = (plgpu.TilingTransform((8, 64)),)
    w_tile = (plgpu.TilingTransform((min(bn, 128),)),)

    @plgpu.kernel(out_type=jax.ShapeDtypeStruct((m, n), x.dtype), grid=(programs,), grid_names=("p",))
    def run(x_g, w_g, o_g):
        p = jax.lax.axis_index("p")

        @pl.loop(p, row_blocks, step=programs)
        def _(rb):
            def ss_body(_, x_s, acc):
                xv = plgpu.load(x_s, layout=L, optimized=False).astype(jnp.float32)
                return acc + jnp.sum(xv * xv, axis=1)

            acc0 = plgpu.layout_cast(jnp.zeros((BM,), jnp.float32), L.reduce(1))
            ss = plgpu.emit_pipeline(
                ss_body, grid=(n // bn,),
                in_specs=[plgpu.BlockSpec((BM, bn), lambda c: (rb, c), transforms=tile)],
                max_concurrent_steps=stages, init_carry=acc0,
            )(x_g)
            inv = jax.lax.rsqrt(ss * (1.0 / n) + eps)

            def out_body(_, x_s, w_s, o_s, inv):
                xv = plgpu.load(x_s, layout=L, optimized=False).astype(jnp.float32)
                wv = plgpu.load(w_s, layout=L.reduce(0), optimized=False).astype(jnp.float32)
                _store(o_s, (xv * wv[None, :]).astype(o_s.dtype))
                return inv

            plgpu.emit_pipeline(
                out_body, grid=(n // bn,),
                in_specs=[plgpu.BlockSpec((BM, bn), lambda c: (rb, c), transforms=tile),
                          plgpu.BlockSpec((bn,), lambda c: (c,), transforms=w_tile)],
                out_specs=[plgpu.BlockSpec((BM, bn), lambda c: (rb, c))],
                max_concurrent_steps=stages, init_carry=inv,
            )(x_g, w_g, o_g)

    return run(x, w)


def reference(x, w, eps: float = 1e-6):
    xf = x.astype(jnp.float32)
    inv = jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return (xf * inv * w.astype(jnp.float32)).astype(x.dtype)


# ----------------------------- contract ------------------------------------

CASES = [
    dict(m=4096, n=1024, dtype="bfloat16"),
    dict(m=4096, n=4096, dtype="bfloat16"),
    dict(m=2048, n=8192, dtype="bfloat16"),
]
TOLERANCE = {"bfloat16": (3e-2, 2e-2), "float16": (3e-3, 3e-3), "float32": (1e-5, 1e-5)}
TUNE_SPACE = dict(bn=[128, 256], stages=[2, 3, 4], programs=[48, 96, 144])


def make_inputs(key, m, n, dtype):
    k1, k2 = jax.random.split(key)
    dt = jnp.dtype(dtype)
    return (jax.random.normal(k1, (m, n), jnp.float32).astype(dt),
            (1.0 + 0.1 * jax.random.normal(k2, (n,), jnp.float32)).astype(dt))


def cost(m, n, dtype):
    b = jnp.dtype(dtype).itemsize
    return dict(flops=4 * m * n, bytes=2 * m * n * b + n * b)  # ideal: read x once, write y


def valid_config(case, params):
    b = jnp.dtype(case["dtype"]).itemsize
    bn, st = params["bn"], params["stages"]
    if case["n"] % bn or case["m"] % BM:
        return False
    pass2 = st * (2 * BM * bn * b + bn * b)
    return pass2 <= SMEM_LIMIT - 1024


if __name__ == "__main__":
    args = make_inputs(jax.random.key(0), **CASES[0])
    err = jnp.max(jnp.abs(kernel(*args).astype(jnp.float32) - reference(*args).astype(jnp.float32)))
    print("max abs err:", float(err))
