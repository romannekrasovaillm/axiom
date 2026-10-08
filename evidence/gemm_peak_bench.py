#!/usr/bin/env python3
"""GEMM-микробенчмарк знаменателя MFU (07-08.10.2026, директива «55% MFU на GB10»).

Знаменатель MFU = измеренный bf16-dense пик этого бокса (не маркетинговый
1 PFLOP FP4-sparse из datasheet и не теоретический ~125 TFLOPS).

Запуск: /home/roman/venv-axiom/bin/python evidence/gemm_peak_bench.py
Вывод: строки "RESULT <dtype> <M>x<K>x<N> <tflops> TFLOPS" + START/END маркеры.
"""
import time

import jax
import jax.numpy as jnp


def bench(dtype, M, K, N, iters=30):
    a = jax.random.normal(jax.random.PRNGKey(0), (M, K)).astype(dtype)
    b = jax.random.normal(jax.random.PRNGKey(1), (K, N)).astype(dtype)
    f = jax.jit(lambda x, y: x @ y)
    r = f(a, b)
    r.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(iters):
        r = f(a, b)
    r.block_until_ready()
    dt = (time.perf_counter() - t0) / iters
    return 2 * M * K * N / dt / 1e12


def main():
    print("GEMM-BENCH-START", jax.devices()[0], flush=True)
    for name, dt in [("bf16", jnp.bfloat16), ("fp32", jnp.float32)]:
        for M, K, N in [(16384, 1536, 1536), (16384, 1536, 6144), (65536, 2048, 8192)]:
            t = bench(dt, M, K, N)
            print(f"RESULT {name} {M}x{K}x{N} {t:.1f} TFLOPS", flush=True)
    print("GEMM-BENCH-END", flush=True)


if __name__ == "__main__":
    main()
