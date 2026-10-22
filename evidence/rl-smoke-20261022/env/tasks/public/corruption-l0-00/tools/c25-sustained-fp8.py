#!/usr/bin/env python3
"""c25-sustained-fp8.py — fp8-пик GB10: всплеск vs УСТОЙЧИВЫЙ (60 с) + мощность/частота.

Проверка критики «реально ~60 TFLOPS bf16»: 8192^3 matmul (f32-аккум) в цикле,
посекундная производительность; параллельно — nvidia-smi power/clocks/temp.
"""
import subprocess
import threading
import time

import jax
import jax.numpy as jnp

N = 8192
SECONDS = 60


def sampler(stop, rows):
    while not stop.is_set():
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw,clocks.sm,temperature.gpu",
             "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        rows.append((time.time(), out))
        time.sleep(2)


def main():
    key = jax.random.PRNGKey(0)
    a = jax.random.normal(key, (N, N), dtype=jnp.float32).astype(jnp.float8_e4m3fn)
    b = jax.random.normal(jax.random.PRNGKey(1), (N, N), dtype=jnp.float32).astype(jnp.float8_e4m3fn)
    f = jax.jit(lambda x, y: jnp.matmul(x, y, preferred_element_type=jnp.float32))
    jax.block_until_ready(f(a, b))
    stop = threading.Event()
    rows = []
    th = threading.Thread(target=sampler, args=(stop, rows), daemon=True)
    th.start()
    t0 = time.time()
    last = t0
    it = 0
    while time.time() - t0 < SECONDS:
        jax.block_until_ready(f(a, b))
        it += 1
        now = time.time()
        if now - last >= 1.0:
            fl = 2 * N ** 3 * it / (now - t0)
            print(f"t={now-t0:5.1f}s  {fl/1e12:6.1f} TFLOPS  ({it} it)", flush=True)
            last = now
    stop.set()
    total = time.time() - t0
    fl = 2 * N ** 3 * it / total
    print(f"SUSTAINED {SECONDS}s: {fl/1e12:.1f} TFLOPS ({it} it)")
    print("power/clock samples:")
    for ts, row in rows[-5:]:
        print(" ", row)


if __name__ == "__main__":
    main()
