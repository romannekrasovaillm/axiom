"""TEMP diagnostic (deleted before commit): per-chunk isolation with real carries."""

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402

from net.kernels import kda_cc_fused as fused  # noqa: E402
from net.kernels import kda_cc_fused_bwd  # noqa: E402
from net.tests.test_kda_cc_fused_persist import (  # noqa: E402
    _bwd_inputs,
    _emulate_bf16_bwd,
)

f32 = jnp.float32


def rel(a, b):
    a = jnp.asarray(a, f32); b = jnp.asarray(b, f32)
    return float(jnp.max(jnp.abs(a - b))) / max(1e-30, float(jnp.max(jnp.abs(b))))


def main():
    T, Cc, Dk, Dv, H = 64, 32, 128, 128, 1
    q, k, v, alpha, beta32, S = _bwd_inputs(T, Dk, Dv, H, 0)
    do = jr.normal(jr.PRNGKey(50), (T, H * Dv))
    dS = jr.normal(jr.PRNGKey(51), (H, Dk, Dv))
    n = T // Cc
    sl = [slice(i * Cc, (i + 1) * Cc) for i in range(n)]

    # fp32 chain + per-chunk reference adjoint with the reference's own carries
    states = [S]
    s = S
    for i in range(n):
        _o, s = fused.reference(q[sl[i]], k[sl[i]], v[sl[i]], alpha[sl[i]],
                                beta32[sl[i]], s, tile=16)
        states.append(s)
    ref_chunks = [None] * n
    carry = dS
    for i in reversed(range(n)):
        out = kda_cc_fused_bwd.reference_bwd(
            q[sl[i]], k[sl[i]], v[sl[i]], alpha[sl[i]], beta32[sl[i]],
            states[i], do[sl[i]], carry, tile=16)
        ref_chunks[i] = out
        carry = out[5]
    # emulation, chunk by chunk, with the SAME states and carries
    carry_e = dS
    emu1 = [None] * n
    for i in reversed(range(n)):
        out = _emulate_bf16_bwd(
            q[sl[i]], k[sl[i]], v[sl[i]], alpha[sl[i]], beta32[sl[i]],
            states[i], do[sl[i]], carry_e, tile=16, chunk=Cc)
        emu1[i] = out
        carry_e = out[5]
    print("per-chunk single-chunk emulation vs reference (same states/carries):")
    for i in reversed(range(n)):
        names = ("dq", "dk", "dv", "dalpha", "dbeta32", "dS_in")
        errs = [rel(a, b) for a, b in zip(emu1[i], ref_chunks[i])]
        print(f"  chunk {i}: " + " ".join(
            f"{nm}={e:.2e}" for nm, e in zip(names, errs)))
    # the full-function emulation, chunk-sliced, against the same references
    emuF = _emulate_bf16_bwd(q, k, v, alpha, beta32, S, do, dS, tile=16, chunk=Cc)
    print("full emulation vs per-chunk reference:")
    for i in reversed(range(n)):
        names = ("dq", "dk", "dv", "dalpha", "dbeta32")
        errs = [rel(jnp.asarray(emuF[k_])[sl[i]], ref_chunks[i][k_])
                for k_, nm in enumerate(names)]
        print(f"  chunk {i}: " + " ".join(
            f"{nm}={e:.2e}" for nm, e in zip(names, errs)))
        if i == 0:
            print(f"    dS_in={rel(emuF[5], ref_chunks[i][5]):.2e}")


if __name__ == "__main__":
    main()
