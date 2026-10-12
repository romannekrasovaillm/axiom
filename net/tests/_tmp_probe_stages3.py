"""TEMP diagnostic (deleted before commit): faithful first-nan stage.

Reproduces exactly what the persistent arm's ``custom_vjp`` hands to the
backward rule at z0 = 0.18: the whole-sequence interval of
``_apply_chunked_cc_persistent`` (q/k/v bf16, alpha/beta32 fp32, S0 = 0), the
emulated forward output o, and ``do`` = d(sum(output_gate(o) * w))/do of the
SAME scalar loss the training loop uses.  Then runs the instrumented emulation.
"""

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402

from net import kda  # noqa: E402
from net.config import ModelConfig  # noqa: E402
from net.tests._tmp_probe_stages import FIRST, emulate  # noqa: E402
from net.tests.test_kda_cc_fused_persist import _cpu_emulated_kernel  # noqa: E402


def main():
    z0 = float(sys.argv[1]) if len(sys.argv) > 1 else 0.18
    H, T, C = 2, 256, 32
    cfg = ModelConfig(
        hidden=H * 128, num_heads=H, head_dim=128,
        kda_dk=128, kda_dv=128, kda_decay_rank=128, kda_short_conv_kernel=4,
        kda_g_min=-5.0, kda_wyut_chunk=32,
    )
    params = kda.init_kda(jr.PRNGKey(0), cfg)
    params = params._replace(
        W_a_down=jnp.zeros_like(params.W_a_down),
        W_a_up=jnp.zeros_like(params.W_a_up),
        b_alpha=jnp.full_like(params.b_alpha, z0),
        A=jnp.zeros_like(params.A),
    )
    x = jr.normal(jr.PRNGKey(5), (T, cfg.hidden))
    w = jr.normal(jr.PRNGKey(6), (T, cfg.hidden))

    # the persistent wrapper's prologue, verbatim
    proj = kda._project(params, cfg, x)
    qc = kda.short_conv(proj["qp"], params.conv_q)
    kc = kda.short_conv(proj["kp"], params.conv_k)
    vc = kda.short_conv(proj["vp"], params.conv_v)
    q, k, v = kda._postprocess(qc, kc, vc, cfg)
    dt = jnp.bfloat16
    alpha = proj["alpha"].reshape(T, H * 128).astype(jnp.float32)
    beta32 = jnp.broadcast_to(proj["beta"][:, :, None], (T, H, 32)).reshape(
        T, H * 32).astype(jnp.float32)
    qb, kb, vb = (t.reshape(T, H * 128).astype(dt) for t in (q, k, v))
    S0 = jnp.zeros((H, 128, 128), jnp.float32)

    o, _s = _cpu_emulated_kernel(qb, kb, vb, alpha, beta32, S0, tile=16, chunk=C)

    def gate_part(o):
        return jnp.sum(kda._output_gate(o.reshape(T, H, 128), proj["gate"],
                                        params) * w)

    do = jax.grad(gate_part)(o)
    print(f"do: max|.|={float(jnp.max(jnp.abs(do))):.3e} "
          f"vs w: {float(jnp.max(jnp.abs(w))):.3e}")
    la = jnp.log(alpha.reshape(T // C, C, -1))
    G = jnp.cumsum(la, axis=1)
    print(f"z0={z0} max|G|={float(jnp.max(jnp.abs(G))):.3f}")

    out = emulate(qb, kb, vb, alpha, beta32, S0, do, jnp.zeros_like(S0),
                  tile=16, chunk=C)
    print("FIRST inf:", FIRST["inf"])
    print("FIRST nan:", FIRST["nan"])
    for name, t in zip(("dq", "dk", "dv", "dalpha", "dbeta32", "dS_in"), out):
        t = jnp.asarray(t, jnp.float32)
        print(f"{name}: nan={bool(jnp.any(jnp.isnan(t)))} "
              f"inf={bool(jnp.any(jnp.isinf(t)))}")


if __name__ == "__main__":
    main()
