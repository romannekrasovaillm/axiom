"""TEMP diagnostic (deleted before commit): per-chunk emu-vs-reference errors."""

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


def q3(a, b, C):
    a = jnp.asarray(a, jnp.float32).reshape(-1, C, a.shape[-1])
    b = jnp.asarray(b, jnp.float32).reshape(-1, C, b.shape[-1])
    out = []
    for i in range(a.shape[0]):
        sc = max(1.0, float(jnp.max(jnp.abs(b[i]))))
        out.append(float(jnp.max(jnp.abs(a[i] - b[i]))) / sc)
    return out


def main():
    T, Cc, Dk, Dv, H = 64, 32, 128, 128, 1
    q, k, v, alpha, beta32, S = _bwd_inputs(T, Dk, Dv, H, 0)
    do = jr.normal(jr.PRNGKey(50), (T, H * Dv))
    dS = jr.normal(jr.PRNGKey(51), (H, Dk, Dv))
    emu = _emulate_bf16_bwd(q, k, v, alpha, beta32, S, do, dS, tile=16, chunk=Cc)
    ref = persist.reference_bwd_scan(q, k, v, alpha, beta32, S, do, dS,
                                     tile=16, chunk=Cc)
    for nm, e, r in zip(("dq", "dk", "dv", "dalpha", "dbeta32"), emu, ref):
        print(nm, ["%.2e" % x for x in q3(e, r, Cc)])
    print("dS_in rel:", float(jnp.max(jnp.abs(emu[5] - ref[5]))) /
          max(1.0, float(jnp.max(jnp.abs(ref[5])))))


if __name__ == "__main__":
    main()
