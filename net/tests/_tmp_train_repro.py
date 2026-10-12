"""TEMP diagnostic (deleted before commit): 4-step training nan repro.

Persistent-B arm = the emulated declared numerics of the variant-B kernel
(forward ``_cpu_emulated_kernel`` + backward ``_emulate_bf16_bwd``); jax arm =
the committed ``cc_chunk_step`` body.  The decay depth is set through the
model's own parameterisation (``b_alpha``/``A``), so the regime is reachable
by weights, not by synthesised kernel inputs.

Usage: python net/tests/_tmp_train_repro.py [z0] [lr] [steps] [seed]
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
from net.kernels import kda_cc_fused_persist as persist  # noqa: E402
from net.tests.test_kda_cc_fused_persist import (  # noqa: E402
    _cpu_emulated_kernel,
    _emulate_bf16_bwd,
)


def cfg_of(H=2):
    return ModelConfig(
        hidden=H * 128, num_heads=H, head_dim=128,
        kda_dk=128, kda_dv=128, kda_decay_rank=128, kda_short_conv_kernel=4,
        kda_g_min=-5.0, kda_wyut_chunk=32,
    )


def nan_report(tree, path=""):
    out = []
    for k, v in jax.tree_util.tree_flatten_with_path(tree)[0]:
        key = jax.tree_util.keystr(k)
        arr = jnp.asarray(v, jnp.float32)
        if bool(jnp.any(jnp.isnan(arr))):
            out.append(f"nan@{key}")
        elif bool(jnp.any(jnp.isinf(arr))):
            out.append(f"inf@{key}")
    return out


def main():
    z0 = float(sys.argv[1]) if len(sys.argv) > 1 else 2.8
    lr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    cfg = cfg_of()
    params0 = kda.init_kda(jr.PRNGKey(seed), cfg)
    params0 = params0._replace(
        W_a_down=jnp.zeros_like(params0.W_a_down),
        W_a_up=jnp.zeros_like(params0.W_a_up),
        b_alpha=jnp.full_like(params0.b_alpha, z0),
        A=jnp.zeros_like(params0.A),
    )
    T = 256
    x = jr.normal(jr.PRNGKey(seed + 5), (T, cfg.hidden))
    w = jr.normal(jr.PRNGKey(seed + 6), (T, cfg.hidden))

    # the decay depth this z0 gives (through the jax body's own project path)
    g = cfg.kda_g_min * jax.nn.sigmoid(jnp.exp(params0.A)[..., None]
                                       * jnp.full((T, cfg.num_heads, cfg.kda_dk), z0))
    print(f"z0={z0} per-token |g|={-float(jnp.mean(g)):.3f} "
          f"chunk max |G|={32 * -float(jnp.mean(g)):.1f}")

    def _arm(p, impl):
        old = os.environ.get("AXIOM_KDA_CC_IMPL")
        os.environ["AXIOM_KDA_CC_IMPL"] = impl
        try:
            return jnp.sum(kda.apply_chunked_cc(p, cfg, x, chunk_size=32) * w)
        finally:
            if old is None:
                os.environ.pop("AXIOM_KDA_CC_IMPL", None)
            else:
                os.environ["AXIOM_KDA_CC_IMPL"] = old

    def loss_jax(p):
        return _arm(p, "jax")

    # --- persistent arm seams -------------------------------------------------
    saved = (persist.kernel, persist.kernel_bwd, persist.device_available,
             os.environ.get("AXIOM_KDA_CC_IMPL"))
    persist.kernel = _cpu_emulated_kernel
    persist.kernel_bwd = _emulate_bf16_bwd
    persist.device_available = lambda: True

    def loss_p(p):
        return _arm(p, "persistent")

    def sgd(p, gr):
        return jax.tree_util.tree_map(lambda a, b: a - lr * b, p, gr)

    sgd_jit = jax.jit(sgd) if lr else None

    pj, pp = params0, params0
    ok = True
    try:
        for step in range(1, steps + 1):
            lj = float(loss_jax(pj))
            gj = jax.grad(loss_jax)(pj)
            lp = float(loss_p(pp))
            gp = jax.grad(loss_p)(pp)
            bj = nan_report(gj)
            bp = nan_report(gp)
            print(f"step {step}: loss_jax={lj:.6f} loss_p={lp:.6f} "
                  f"d={lp - lj:+.3e} | nan jax={bj[:3]} pers={bp[:3]}")
            if bp:
                ok = False
                break
            if lr:
                pj = sgd_jit(pj, gj)
                pp = sgd_jit(pp, gp)
                # decay depth after the update (through the jax body's project)
                gp_proj = kda._project(pp, cfg, x)["alpha"]
                la = jnp.log(gp_proj.reshape(T // 32, 32, -1))
                G = jnp.cumsum(la, axis=1)
                print(f"   after update: loss_jax_next={float(loss_jax(pj)):.6f} "
                      f"loss_p_next={float(loss_p(pp)):.6f} "
                      f"max|G|={float(jnp.max(jnp.abs(G))):.2f} "
                      f"b_alpha={float(jnp.mean(pp.b_alpha)):.4f}")
        if ok:
            print("no nan within", steps, "steps")
    finally:
        persist.kernel, persist.kernel_bwd, persist.device_available, impl = saved
        if impl is None:
            os.environ.pop("AXIOM_KDA_CC_IMPL", None)
        else:
            os.environ["AXIOM_KDA_CC_IMPL"] = impl


if __name__ == "__main__":
    main()
