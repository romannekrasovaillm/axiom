"""TEMP diagnostic (deleted before commit): where does the variant-B bwd nan?

Sweeps the per-token decay depth |log alpha| = g0 at the campaign geometry and
reports, per arm (the kernel's declared numerics = ``_emulate_bf16_bwd`` and the
jnp adjoint reference = ``reference_bwd_scan``), which cotangent first turns
inf/nan and at which chunk.
"""

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402

from net.kernels import kda_cc_fused_persist as persist  # noqa: E402
from net.tests.test_kda_cc_fused_persist import _emulate_bf16_bwd  # noqa: E402

NAMES = ("dq", "dk", "dv", "dalpha", "dbeta32", "dS_in")


def _flags(x):
    x = jnp.asarray(x, jnp.float32)
    n = bool(jnp.any(jnp.isnan(x)))
    i = bool(jnp.any(jnp.isinf(x)))
    return ("NAN" if n else "   ") + ("INF" if i else "   ")


def _maxabs(x):
    x = jnp.asarray(x, jnp.float32)
    finite = jnp.where(jnp.isfinite(x), jnp.abs(x), 0.0)
    return float(jnp.max(finite))


def make(g0, T, Cc, Dk, Dv, H, seed, profile="uniform"):
    keys = jr.split(jr.PRNGKey(seed), 6)
    q = jr.normal(keys[0], (T, H * Dk)) * 0.5
    k = jr.normal(keys[1], (T, H * Dk)) * 0.5
    v = jr.normal(keys[2], (T, H * Dv)) * 0.5
    if profile == "uniform":
        alpha = jnp.exp(-g0 * jnp.ones((T, H * Dk)))
    elif profile == "ramp":
        # a chunk-deepening profile: per-token |g| grows 0.5 -> g0 within a chunk
        pos = jnp.arange(T) % Cc
        per = 0.5 + (g0 - 0.5) * pos / (Cc - 1)
        alpha = jnp.exp(-per[:, None] * jnp.ones((T, H * Dk)))
    else:
        alpha = jnp.exp(-g0 * jr.uniform(keys[3], (T, H * Dk)))
    beta = jr.uniform(keys[4], (T, H))
    beta32 = jnp.broadcast_to(beta[:, :, None], (T, H, 32)).reshape(T, H * 32)
    S = jr.normal(keys[5], (H, Dk, Dv)) * 0.1
    # do must be shaped like the kernel contract: (T, H*Dv)
    do = jr.normal(jr.PRNGKey(seed + 17), (T, H * Dv))
    dS = jr.normal(jr.PRNGKey(seed + 18), (H, Dk, Dv))
    return q, k, v, alpha, beta32, S, do, dS


def run_arm(fn, args):
    try:
        out = fn(*args)
    except Exception as exc:  # noqa: BLE001
        return None, f"EXC {type(exc).__name__}: {str(exc)[:80]}"
    return out, ""


def main():
    T, Cc, Dk, Dv, H = 64, 32, 128, 128, 1
    for profile in ("uniform", "ramp", "noisy"):
        print(f"=== profile={profile} T={T} C={Cc} dk={Dk} dv={Dv} H={H} ===")
        for g0 in (2.0, 2.5, 2.7, 2.8, 2.9, 3.0, 3.5, 4.0, 5.0):
            args = make(g0, T, Cc, Dk, Dv, H, 0, profile)
            emu, emsg = run_arm(
                lambda *a: _emulate_bf16_bwd(*a, tile=16, chunk=Cc), args)
            ref, rmsg = run_arm(
                lambda *a: persist.reference_bwd_scan(*a, tile=16, chunk=Cc), args)
            def fmt(out, msg):
                if out is None:
                    return msg
                parts = []
                for nm, val in zip(NAMES, out):
                    parts.append(f"{nm}:{_flags(val)} mx={_maxabs(val):.1e}")
                return " | ".join(parts)
            print(f"g0={g0:4.2f}")
            print(f"  emu: {fmt(emu, emsg)}")
            print(f"  ref: {fmt(ref, rmsg)}")


if __name__ == "__main__":
    main()
