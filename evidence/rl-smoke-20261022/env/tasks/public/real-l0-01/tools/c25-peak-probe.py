#!/usr/bin/env python3
"""c25-peak-probe.py — измеренные потолки GB10 под цель G4 (55 % MFU).

Меряет реальные TFLOPS больших матмулов (bf16 / fp16 / fp8-e4m3, fp32-аккум)
и пропускную способность памяти (копия). Нужен ответ: какой класс точности и
какой потолок доступен на sm_121, прежде чем планировать 55 % (ADR-050 G4).
Запуск на стенде: /home/roman/venv-axiom-0112/bin/python tools/c25-peak-probe.py
"""
import time

import jax
import jax.numpy as jnp

N = 8192
ITERS = 8
BYTES = 2 * N * N * 2  # one bf16 matrix in+out for the copy


def bench_matmul(dtype, iters=ITERS, n=N):
    key = jax.random.PRNGKey(0)
    a = jax.random.normal(key, (n, n), dtype=jnp.float32).astype(dtype)
    b = jax.random.normal(jax.random.PRNGKey(1), (n, n), dtype=jnp.float32).astype(dtype)
    f = jax.jit(lambda x, y: jnp.matmul(x, y, preferred_element_type=jnp.float32))
    r = f(a, b)
    jax.block_until_ready(r)
    t = time.perf_counter()
    for _ in range(iters):
        r = f(a, b)
    jax.block_until_ready(r)
    dt = (time.perf_counter() - t) / iters
    return dt, 2 * n ** 3 / dt / 1e12


def bench_copy(n=N, iters=20):
    x = jnp.ones((n, n), dtype=jnp.bfloat16)
    f = jax.jit(lambda v: v + 1)
    jax.block_until_ready(f(x))
    t = time.perf_counter()
    for _ in range(iters):
        y = f(x)
    jax.block_until_ready(y)
    dt = (time.perf_counter() - t) / iters
    return dt, 2 * BYTES / dt / 1e9


def main():
    print("device:", jax.devices()[0].device_kind, "jax", jax.__version__)
    for name, dt in (("bf16", jnp.bfloat16), ("fp16", jnp.float16),
                     ("fp8_e4m3", getattr(jnp, "float8_e4m3fn", None))):
        if dt is None:
            print(f"{name:9s} SKIP (no dtype)")
            continue
        try:
            sec, tflops = bench_matmul(dt)
            print(f"{name:9s} {sec:.4f} s  {tflops:8.1f} TFLOPS")
        except Exception as exc:  # noqa: BLE001 — диагностический зонд
            print(f"{name:9s} FAIL {type(exc).__name__}: {str(exc)[:120]}")
    sec, gbps = bench_copy()
    print(f"copy      {sec:.4f} s  {gbps:8.1f} GB/s")


if __name__ == "__main__":
    main()
