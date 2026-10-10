"""Прототип 2: переход MMA_ACC -> SMEM -> MMA_LHS для цепочки mma."""
import dataclasses, json, sys, time
import jax, jax.numpy as jnp
from jax.experimental.pallas import mosaic_gpu as plgpu

M = N = K = 128
DT = jnp.dtype(jnp.bfloat16)
ACC = jnp.float32


def body(L_ref, o_ref, smem):
    acc = plgpu.layout_cast(jnp.zeros((M, N), ACC), plgpu.Layout.MMA_ACC(DT))
    a = plgpu.load(L_ref, layout=plgpu.Layout.MMA_LHS(DT), optimized=False)
    b = plgpu.load(L_ref.T, layout=plgpu.Layout.MMA_RHS(DT), optimized=False)
    p = plgpu.mma(acc, a, b)                 # p = L @ L  (ACC-раскладка, f32)
    smem[...] = p.astype(DT)                 # выгружаем результат в SMEM
    p_lhs = plgpu.load(smem, layout=plgpu.Layout.MMA_LHS(DT), optimized=False)
    acc2 = plgpu.layout_cast(jnp.zeros((M, N), ACC), plgpu.Layout.MMA_ACC(DT))
    p2 = plgpu.mma(acc2, p_lhs, b)           # (L@L) @ L
    o_ref[...] = p2.astype(DT)


def main():
    rep = {"jax": jax.__version__, "shape": [M, K, N]}
    L = (jnp.tril(jnp.ones((M, K)), -1) * 0.01).astype(DT)
    ref = (L.astype(jnp.float32) @ L.astype(jnp.float32) @ L.astype(jnp.float32))
    try:
        fn = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((M, N), DT),
                          scratch_types=[plgpu.SMEM((M, N), DT)],
                          compiler_params=plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Lane))
        t0 = time.time()
        jax.jit(fn).lower(jnp.zeros((M, K), DT)).compile()
        rep["compile_stage"] = "ok"
        out = jax.block_until_ready(fn(jnp.zeros((M, K), DT)))
        max_abs = float(jnp.max(jnp.abs(out.astype(jnp.float32) - ref.astype(jnp.float32))))
        rep.update({"status": "run-ok", "max_abs": max_abs, "rel": max_abs / float(jnp.max(ref)), "seconds": round(time.time()-t0, 3)})
    except Exception as e:
        import traceback
        rep.update({"status": "blocked", "error_type": type(e).__name__, "error": traceback.format_exc()[-1200:]})
    json.dump(rep, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "error"}, ensure_ascii=False)[:300])
    if rep.get("error"): print("---", rep["error"][-900:])


main()
