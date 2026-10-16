"""Acceptance criterion 8 — QAT fake-quant MXFP4 (straight-through).

Two checks:

1. The quantise -> dequantise round-trip error is bounded in the MXFP4 format
   (per-block shared power-of-two scale + 3-bit mantissa), and the dequantised
   values are exactly representable.
2. A fine-tune smoke with the fake-quant enabled (the SFT-stage flag) reduces
   the loss, while the pretrain default keeps QAT off (``config.qat_enabled``
   is False in ``net/config.json``).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr

from net import mla, model, optimizer, quant
from net.config import ModelConfig

CONFIG = Path(__file__).resolve().parent.parent / "config.json"


def test_mxfp4_roundtrip_error_bounded():
    key = jr.PRNGKey(0)
    w = jr.normal(key, (64, 64))
    err = float(quant.quant_error(w))
    assert err < 0.5, err  # 4-bit mantissa: relative L2 error well below 1.0
    # dequantised values are exactly representable (quantise is idempotent)
    dq = quant.quantize_dequantize(w)
    dq2 = quant.quantize_dequantize(dq)
    assert float(jnp.max(jnp.abs(dq - dq2))) < 1e-6


def test_pretrain_qat_disabled_by_default():
    data = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert data["qat_enabled"] is False


def test_finetune_smoke_with_fake_quant(tiny_cfg):
    key = jr.PRNGKey(0)
    params = model.init_params(key, tiny_cfg)
    ids = jr.randint(key, (4, 16), 0, tiny_cfg.vocab_size)

    def loss_fn(p, x):
        # fake-quantise the weights (SFT-stage QAT) before the forward pass
        qp = quant.apply_fake_quant_tree(p)
        return model.compute_loss(qp, tiny_cfg, x, chunk_size=8)

    step = optimizer.make_step(tiny_cfg)
    state = optimizer.init_state(params)
    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    loss0, _ = grad_fn(params, ids)
    for _ in range(40):
        loss, grads = grad_fn(params, ids)
        params, state = step(params, grads, state, 1e-2)
    assert float(loss) < float(loss0), "fine-tune smoke with fake-quant must reduce loss"


# ---------------------------------------------------------------------------
# ADR-009 D3 — MXFP4 latent KV (E2M1 + one E4M3 scale / 16 channels) and FP8 SWA
# ---------------------------------------------------------------------------
#
# The weight branch above is untouched; these extend QAT to the latent KV cache
# of the MLA layers, behind ``qat_kv_enabled`` (off at pretrain).


def test_latent_mxfp4_values_are_e2m1():
    """Every dequantised magnitude is level * scale with the level in E2M1."""
    c = jr.normal(jr.PRNGKey(0), (8, 64))
    dq = quant.mxfp4_latent_roundtrip(c)
    cb = c.reshape(8, 4, 16)
    dqb = dq.reshape(8, 4, 16)
    max_abs = jnp.max(jnp.abs(cb), axis=-1, keepdims=True)
    scale = jnp.clip(max_abs / 6.0, -448.0, 448.0).astype(jnp.float8_e4m3fn).astype(jnp.float32)
    ok = jnp.zeros_like(dqb, dtype=jnp.bool_)
    for level in quant.E2M1_LEVELS:
        ok = ok | jnp.isclose(jnp.abs(dqb), level * scale, atol=1e-6)
    assert bool(jnp.all(ok)), "dequantised latent is not on the E2M1 grid"


def test_latent_mxfp4_scales_are_per_block_with_no_global_scale():
    """Rescaling one 16-channel block must not move any other block.

    If a second-level (global) scale were present, a large excursion in one
    block would change the dequantisation of every block; the source format
    omits it (ADR-009 D3 / V4.1-Flash section 2.4.4).
    """
    c = jr.normal(jr.PRNGKey(1), (16, 32))
    scaled = c.at[:, :16].multiply(1000.0)
    dq = quant.mxfp4_latent_roundtrip(c)
    dq_scaled = quant.mxfp4_latent_roundtrip(scaled)
    assert bool(jnp.allclose(dq_scaled[:, 16:], dq[:, 16:], atol=1e-5)), "global scale leaked"
    assert float(quant.latent_quant_error(c)) < 0.5


def test_latent_mxfp4_roundtrip_is_idempotent():
    c = jr.normal(jr.PRNGKey(2), (4, 48))
    dq = quant.mxfp4_latent_roundtrip(c)
    dq2 = quant.mxfp4_latent_roundtrip(dq)
    assert float(jnp.max(jnp.abs(dq - dq2))) < 1e-6


def test_qat_kv_disabled_by_default():
    data = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert data["qat_kv_enabled"] is False
    assert ModelConfig().qat_kv_enabled is False


def test_latent_fake_quant_straight_through_gradient(tiny_cfg):
    """QAT uses the straight-through estimator: gradients pass unchanged."""
    c = jr.normal(jr.PRNGKey(3), (4, 16))

    def f(v):
        return jnp.sum(quant.fake_quant_mxfp4_latent(v))

    g = jax.grad(f)(c)
    assert bool(jnp.allclose(g, jnp.ones_like(c), atol=1e-6))


def test_swa_kv_fp8_stays_in_e4m3_range():
    """SWA KV is fake-quantised to FP8 (E4M3), not FP4 (ADR-009 D3)."""
    kv = jr.normal(jr.PRNGKey(4), (2, 64)) * 3.0
    q = quant.fake_quant_fp8_e4m3(kv)
    assert bool(jnp.all(jnp.isfinite(q)))
    assert float(jnp.max(jnp.abs(q - kv))) < 0.5  # E4M3 is a fine format
    assert float(jnp.linalg.norm(q - kv)) / float(jnp.linalg.norm(kv)) < 0.1


def test_qat_kv_quantises_only_the_sparse_path(cfg):
    """The dense oracle ignores qat_kv_enabled; the sparse latent path honours it."""
    params = mla.init_mla(jr.PRNGKey(5), cfg)
    x = jr.normal(jr.PRNGKey(6), (2, 16, cfg.hidden))
    sparse_on = dataclasses.replace(cfg, attn_dense_reference=False, qat_kv_enabled=True)
    sparse_off = dataclasses.replace(cfg, attn_dense_reference=False, qat_kv_enabled=False)
    dense_on = dataclasses.replace(cfg, attn_dense_reference=True, qat_kv_enabled=True)
    # the oracle is not quantised: the flag changes nothing there
    assert bool(jnp.array_equal(mla.apply(params, dense_on, x), mla.apply(params, cfg, x)))
    # on the sparse path, FP4 on the latent is a real (finite) perturbation
    a = mla.apply(params, sparse_off, x)
    b = mla.apply(params, sparse_on, x)
    assert bool(jnp.all(jnp.isfinite(b)))
    assert float(jnp.max(jnp.abs(a - b))) > 0.0
