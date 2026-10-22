#!/usr/bin/env python3
"""c25-shape-probe.py — bf16 vs fp8 на РЕАЛЬНЫХ формах l3-full (цикл-24/25).

Пик 8192³ (131.7 TFLOPS fp8) не переносится на мелкие/тонкие GEMM модели
автоматически. Меряем реальные формы: эксперт MoE (5461x768x384 / 5461x384x768),
shared-эксперт (32768x1536x384), lm_head-чанк (2048x1536x160000), проекция
(32768x1536x1536), attention-тайл (128x128x128). Для каждой — bf16 и fp8-e4m3,
fp32-аккум; вывод — TFLOPS и доля от 8192³-пика того же класса.
"""
import time

import jax
import jax.numpy as jnp

BF16_PEAK = 88.8
FP8_PEAK = 131.7
SHAPES = [
    ("moe_gate", 5461, 768, 384),
    ("moe_down", 5461, 384, 768),
    ("shared", 32768, 1536, 384),
    ("lmhead_chunk", 2048, 1536, 160000),
    ("proj", 32768, 1536, 1536),
    ("attn_tile", 128, 128, 128),
]


def bench(m, k, n, dtype, iters=10):
    key = jax.random.PRNGKey(0)
    a = jax.random.normal(key, (m, k), dtype=jnp.float32).astype(dtype)
    b = jax.random.normal(jax.random.PRNGKey(1), (k, n), dtype=jnp.float32).astype(dtype)
    f = jax.jit(lambda x, y: jnp.matmul(x, y, preferred_element_type=jnp.float32))
    jax.block_until_ready(f(a, b))
    t = time.perf_counter()
    for _ in range(iters):
        r = f(a, b)
    jax.block_until_ready(r)
    dt = (time.perf_counter() - t) / iters
    return dt, 2 * m * k * n / dt / 1e12


def main():
    print("device:", jax.devices()[0].device_kind)
    fp8 = getattr(jnp, "float8_e4m3fn", None)
    print(f"{'shape':14s} {'M':>7s} {'K':>7s} {'N':>7s} | {'bf16 TF':>8s} {'%peak':>6s} | {'fp8 TF':>8s} {'%peak':>6s} | gain")
    for name, m, k, n in SHAPES:
        try:
            _, tb = bench(m, k, n, jnp.bfloat16)
            sb = f"{tb:8.2f} {100*tb/BF16_PEAK:6.1f}"
        except Exception as exc:  # noqa: BLE001
            sb = f"FAIL {str(exc)[:16]:>16s}"
            tb = 0.0
        if fp8 is None:
            sf = "—"
            tf = 0.0
        else:
            try:
                _, tf = bench(m, k, n, fp8)
                sf = f"{tf:8.2f} {100*tf/FP8_PEAK:6.1f}"
            except Exception as exc:  # noqa: BLE001
                sf = f"FAIL {str(exc)[:16]:>16s}"
                tf = 0.0
        gain = (tf / tb) if tb and tf else 0.0
        print(f"{name:14s} {m:7d} {k:7d} {n:7d} | {sb} | {sf} | x{gain:.2f}")


if __name__ == "__main__":
    main()
