"""Чистый тест: (M,K)@(K,N) через plgpu.mma для разных N, головы срезом, как в KDA-ядре."""
import json, sys, traceback
import jax, jax.numpy as jnp
from jax.experimental.pallas import mosaic_gpu as plgpu

H, M, K = 12, 128, 128
DT = jnp.dtype(jnp.bfloat16)
ACC = jnp.float32


def make(N):
    def body(l_ref, b_ref, o_ref, smem):     # l_ref (H,M,K), b_ref (H,N,K) [RHS в (n,k)!], o_ref (H,M,N)
        h = jax.lax.axis_index("head")
        acc = plgpu.layout_cast(jnp.zeros((M, N), ACC), plgpu.Layout.MMA_ACC(DT))
        a = plgpu.load(l_ref.at[h], layout=plgpu.Layout.MMA_LHS(DT), optimized=False)
        b = plgpu.load(b_ref.at[h].T, layout=plgpu.Layout.MMA_RHS(DT), optimized=False)
        o_ref.at[h][...] = plgpu.mma(acc, a, b).astype(DT)
    return body


def run(N):
    body = make(N)
    fn = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((H, M, N), DT),
                      scratch_types=[plgpu.SMEM((M, M), DT)],
                      compiler_params=plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Lane),
                      grid=(H,), grid_names=("head",))
    L = (jnp.tril(jnp.ones((H, M, K)), -1) * 0.01).astype(DT)
    B = (jnp.ones((H, N, K)) * 0.02).astype(DT)      # (n,k): k-контигуально
    out = jax.block_until_ready(fn(L, B))
    ref = jnp.einsum("hmk,hnk->hmn", L.astype(jnp.float32), B.astype(jnp.float32))
    return {"shape": list(out.shape), "max_abs": float(jnp.max(jnp.abs(out.astype(jnp.float32) - ref)))}


res = {}
for N in (8, 64, 256):
    try:
        res[f"N={N}"] = {"status": "ok", **run(N)}
    except Exception as e:
        res[f"N={N}"] = {"status": "blocked", "error_type": type(e).__name__,
                         "last": traceback.format_exc().strip().splitlines()[-1][:200]}
json.dump(res, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
for k, v in res.items():
    print(f"{k:<8} {v['status']:<9} {v.get('shape','')} {v.get('max_abs', v.get('last',''))}")
