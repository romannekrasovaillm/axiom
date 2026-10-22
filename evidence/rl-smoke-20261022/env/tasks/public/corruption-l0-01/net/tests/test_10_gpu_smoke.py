"""Acceptance criterion 10 — pretrain smoke on a GPU.

Runs a short training smoke and records tokens/second.  The test stops being
"passing by circumstance" (ADR-010, decision 3):

* in the **gate profile** (``NET_GATE_PROFILE=1`` or ``pytest --net-gate``) a
  missing CUDA-enabled JAX device is a FAIL with an explicit message: the A4
  verdict cannot be produced on a backend the mechanism will not train on;
* **outside** the gate profile the same absence is an explicit SKIP marked as a
  preliminary run — the skeleton's first smoke is exactly this check on the
  GB10 (ADR-008), but a local CPU run must not be reported as a gate pass.
"""

from __future__ import annotations

import time

import pytest

import jax
import jax.numpy as jnp
import jax.random as jr

from conftest import describe_devices, gate_profile, gpu_devices

from net import model, optimizer


def _cuda_device() -> str | None:
    devices = gpu_devices()
    return devices[0] if devices else None


def test_pretrain_smoke_tokens_per_second(tiny_cfg):
    device = _cuda_device()
    if device is None:
        devices = describe_devices()
        if gate_profile():
            pytest.fail(
                "gate profile requires a CUDA-enabled JAX device, "
                f"jax.devices()={devices}. The A4 verdict is not produced on CPU "
                "(ADR-010): fix the CUDA plugin (see net/README.md) or run "
                "without the gate marker for a preliminary run.",
                pytrace=False,
            )
        pytest.skip(
            "preliminary run: no CUDA-enabled JAX device, and the local profile "
            f"does not require one (jax.devices()={devices}; ADR-010)"
        )

    key = jr.PRNGKey(0)
    params = model.init_params(key, tiny_cfg)
    ids = jr.randint(key, (2, 128), 0, tiny_cfg.vocab_size)

    def loss_fn(p, x):
        return model.compute_loss(p, tiny_cfg, x, chunk_size=16)

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    step = optimizer.make_step(tiny_cfg)
    state = optimizer.init_state(params)

    # warm up (compile) then time
    loss, grads = grad_fn(params, ids)
    t0 = time.time()
    for _ in range(10):
        loss, grads = grad_fn(params, ids)
        params, state = step(params, grads, state, 1e-3)
    dt = time.time() - t0
    tokens = 10 * 2 * 128
    tok_per_s = tokens / dt
    assert tok_per_s > 0
    # the threshold N is fixed against the actual hardware in the report, not
    # hard-coded here (MODEL-L3-SKELETON.md section 6.10)
    print(f"GPU smoke: {tok_per_s:.1f} tok/s on {device}")
