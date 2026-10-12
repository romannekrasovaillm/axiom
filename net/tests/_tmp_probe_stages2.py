"""TEMP diagnostic (deleted before commit): first-nan stage on MODEL inputs.

Same instrumented emulation as ``_tmp_probe_stages`` but the inputs come from
the model path (projections/alpha of the z0-regime config), mirroring the
training-loop cotangent structure (do = loss weights, dS = 0).
"""

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402

from net import kda  # noqa: E402
from net.config import ModelConfig  # noqa: E402
from net.tests._tmp_probe_stages import FIRST, emulate  # noqa: E402
from net.tests.test_kda_cc_fused_persist import _interval_and_committed  # noqa: E402


def main():
    z0 = float(sys.argv[1]) if len(sys.argv) > 1 else 0.18
    H, T = 2, 256
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
    interval, states = _interval_and_committed(params, cfg, x, 32)
    q, k, v, alpha, beta32 = interval
    S0 = states[0]
    do = w
    dS = jnp.zeros_like(S0)
    la = jnp.log(alpha.reshape(T // 32, 32, -1))
    G = jnp.cumsum(la, axis=1)
    print(f"z0={z0} max|G|={float(jnp.max(jnp.abs(G))):.3f} "
          f"min G={float(jnp.min(G)):.3f} max span="
          f"{float(jnp.max(G[:, 0] - G[:, 31])):.3f}")
    out = emulate(q, k, v, alpha, beta32, S0, do, dS, tile=16, chunk=32)
    print("FIRST inf:", FIRST["inf"])
    print("FIRST nan:", FIRST["nan"])
    for name, t in zip(("dq", "dk", "dv", "dalpha", "dbeta32", "dS_in"), out):
        t = jnp.asarray(t, jnp.float32)
        print(f"{name}: nan={bool(jnp.any(jnp.isnan(t)))} "
              f"inf={bool(jnp.any(jnp.isinf(t)))}")


if __name__ == "__main__":
    main()
