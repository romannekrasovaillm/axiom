"""Mosaic MMA smoke для GB10 (CC12.1), JAX 0.11.2 — по эталону tests/pallas/mosaic_gpu_test.py::test_mma."""
import functools, json, sys, time
import jax, jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

M = K = 128
N = 8
DTYPE = jnp.dtype(jnp.bfloat16)
ACC = jnp.float32


@functools.partial(plgpu.kernel, out_type=jax.ShapeDtypeStruct((M, N), ACC))
def kernel(a_ref, b_ref, o_ref):
    acc = plgpu.layout_cast(jnp.zeros((M, N), ACC), plgpu.Layout.MMA_ACC(DTYPE))
    a = plgpu.load(a_ref, layout=plgpu.Layout.MMA_LHS(DTYPE), optimized=False)
    b = plgpu.load(b_ref.T, layout=plgpu.Layout.MMA_RHS(DTYPE), optimized=False)
    o_ref[...] = plgpu.mma(acc, a, b)


def main():
    rep = {"jax": jax.__version__, "device": str(jax.devices()[0]), "cc": str(jax.devices()[0].compute_capability),
           "shape": {"m": M, "k": K, "n": N, "dtype": str(DTYPE)}}
    rng = np.random.default_rng(0)
    a = rng.random((M, K)).astype(np.float32) - 0.5
    b = rng.random((N, K)).astype(np.float32) - 0.5
    ref = a.astype(np.float32) @ b.T.astype(np.float32)
    try:
        t0 = time.time()
        out = kernel(a.astype(np.float32), b.astype(np.float32))
        jax.block_until_ready(out)
        dt = time.time() - t0
        max_abs = float(jnp.max(jnp.abs(out - ref)))
        denom = float(jnp.max(jnp.abs(ref))) or 1.0
        rep.update({
            "status": "correctness-pass" if max_abs / denom < 0.05 else "correctness-fail",
            "max_abs_vs_ref": max_abs, "rel": max_abs / denom, "seconds": round(dt, 4),
            "note": "plgpu.kernel + plgpu.load(MMA_LHS/RHS) + plgpu.mma(MMA_ACC) на устройстве",
        })
    except Exception as e:
        rep.update({"status": "blocked", "error_type": type(e).__name__, "error": str(e)[:600]})
    json.dump(rep, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False)[:700])


main()
