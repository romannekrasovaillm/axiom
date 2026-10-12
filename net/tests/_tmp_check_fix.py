"""TEMP diagnostic (deleted before commit): fixed emulation vs reference."""

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402

from net.kernels import kda_cc_fused_persist as persist  # noqa: E402
from net.tests.test_kda_cc_fused_persist import (  # noqa: E402
    _bwd_inputs,
    _emulate_bf16_bwd,
)

NAMES = ("dq", "dk", "dv", "dalpha", "dbeta32", "dS_in")


def run(tag, q, k, v, alpha, beta32, S, do, dS, tol=2e-2):
    emu = _emulate_bf16_bwd(q, k, v, alpha, beta32, S, do, dS, tile=16, chunk=32)
    ref = persist.reference_bwd_scan(q, k, v, alpha, beta32, S, do, dS,
                                     tile=16, chunk=32)
    for name, e, r in zip(NAMES, emu, ref):
        e = jnp.asarray(e, jnp.float32)
        r = jnp.asarray(r, jnp.float32)
        scale = max(1.0, float(jnp.max(jnp.abs(r))))
        err = float(jnp.max(jnp.abs(e - r)))
        fin = bool(jnp.all(jnp.isfinite(e)))
        print(f"  [{tag}] {name}: err={err:.3e} scale={scale:.2e} "
              f"rel={err / scale:.2e} finite={fin} "
              f"{'OK' if err <= tol * scale and fin else '*** FAIL ***'}")


def main():
    T, Cc, Dk, Dv, H = 64, 32, 128, 128, 1
    # (1) the bwd-inputs distribution (mild decay)
    q, k, v, alpha, beta32, S = _bwd_inputs(T, Dk, Dv, H, 0)
    do = jr.normal(jr.PRNGKey(50), (T, H * Dv))
    dS = jr.normal(jr.PRNGKey(51), (H, Dk, Dv))
    run("mild", q, k, v, alpha, beta32, S, do, dS)
    # (2) uniform deep decay 2.8/token: pre-fix emulation inf/nan, ref finite
    alpha = jnp.exp(-2.8 * jnp.ones((T, H * Dk)))
    run("deep-2.8", q, k, v, alpha, beta32, S, do, dS)
    # (3) the working regime of the training loop: deep + big do
    alpha = jnp.exp(-2.75 * jnp.ones((T, H * Dk)))
    do_big = do * 100.0
    run("deep-2.75-bigdo", q, k, v, alpha, beta32, S, do_big, dS)


if __name__ == "__main__":
    main()
