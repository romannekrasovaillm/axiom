"""Per-layer grad-checkpointing of the backbone (D-8, ``grad_ckpt_policy``).

The coarse wrap of the whole ``compute_loss`` in ``net/train_loop.py`` does not
cut the activation request: XLA keeps the whole graph's intermediates on the
recompute boundary, and the full preset still asked for ~972 GiB at T=8192 even
with ``--grad-checkpointing`` (D-8).  The cut has to be *per layer*: each
backbone layer is wrapped in ``jax.checkpoint`` so the backward pass recomputes
it from ``h``, the candidate-pool carry and the previous layer deltas instead of
retaining its attention/MoE intermediates.

What these tests pin:

* the declared policy is read from ``net/config.json`` (``per_layer`` for the
  ``l3-full`` preset) and validated by ``net.config.validate_config``;
* ``per_layer`` really puts a ``remat`` boundary around every backbone layer in
  the traced graph (a policy that is merely parsed but not consumed would be a
  declaration without a mechanism — the AD-9 failure mode);
* the loss and its gradients are unchanged by the remat (``allclose``, not
  bitwise: recompute may reorder floating-point operations — an XLA property,
  not a bug);
* ``none`` reproduces the pre-D-8 graph bit-for-bit, so the switch is honest in
  both directions.

The full-config memory manifest (compile at T=8192, vocab 160k, assert the XLA
temp request stays under the 50 GiB goal) needs a compiled full graph and is
opt-in via ``AXIOM_GRAD_CKPT_MEMORY=1`` — it is meant for the GB10 confirmation
run, not for the CPU suite.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from conftest import small_config

from net import model
from net.config import load_config, validate_config

#: The case's declarative config (the file C-035-style declarations live in).
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"


def _retrace(cfg, *, policy: str):
    key = jr.PRNGKey(0)
    params = model.init_params(key, cfg)
    ids = jr.randint(key, (2, 64), 0, cfg.vocab_size)

    def loss(p):
        return model.compute_loss(
            p, cfg, ids, chunk_size=16, grad_ckpt_policy=policy
        )

    return params, loss


def _remat_count(cfg, policy: str) -> int:
    """Number of ``remat`` boundaries in the traced loss graph."""
    params, loss = _retrace(cfg, policy=policy)
    closed = jax.make_jaxpr(loss)(params)
    count = 0
    for eqn in closed.jaxpr.eqns:
        # JAX names the checkpoint primitive ``remat`` with a version suffix
        # (``remat2`` on jax 0.10.x); match the family, not one exact name.
        if str(eqn.primitive.name).startswith("remat"):
            count += 1
    return count


def _tree_allclose(a, b, *, rtol=1e-4, atol=1e-5) -> bool:
    leaves_a = jax.tree_util.tree_leaves(a)
    leaves_b = jax.tree_util.tree_leaves(b)
    assert len(leaves_a) == len(leaves_b)
    return all(
        bool(jnp.allclose(x, y, rtol=rtol, atol=atol, equal_nan=True))
        for x, y in zip(leaves_a, leaves_b)
    )


# ---------------------------------------------------------------------------
# declaration: the config file is the switch (spine AD-9 / C-035 form)
# ---------------------------------------------------------------------------


def test_declared_policy_is_per_layer_for_pretrain():
    cfg = load_config(CONFIG_PATH)
    assert cfg.grad_ckpt_policy == "per_layer"
    validate_config(cfg)  # the declared value is one the loop implements


def test_schema_default_is_none():
    """A config built in code (tests/smokes) keeps the pre-D-8 graph."""
    assert small_config().grad_ckpt_policy == "none"


def test_unknown_policy_is_rejected():
    cfg = dataclasses.replace(small_config(), grad_ckpt_policy="sometimes")
    with pytest.raises(AssertionError):
        validate_config(cfg)


# ---------------------------------------------------------------------------
# mechanism: per_layer puts a remat boundary around every backbone layer
# ---------------------------------------------------------------------------


def test_per_layer_traces_one_remat_per_layer(cfg):
    assert _remat_count(cfg, "per_layer") == cfg.num_layers


def test_none_traces_no_remat(cfg):
    assert _remat_count(cfg, "none") == 0


def test_unknown_policy_raises_in_forward(cfg):
    key = jr.PRNGKey(0)
    params = model.init_params(key, cfg)
    ids = jr.randint(key, (1, 16), 0, cfg.vocab_size)
    with pytest.raises(ValueError):
        model.compute_loss(params, cfg, ids, chunk_size=8, grad_ckpt_policy="nope")


# ---------------------------------------------------------------------------
# parity: remat changes where activations live, not the numbers
# ---------------------------------------------------------------------------


def test_forward_parity(cfg):
    key = jr.PRNGKey(1)
    params = model.init_params(key, cfg)
    ids = jr.randint(key, (2, 64), 0, cfg.vocab_size)
    none = model.forward(params, cfg, ids, chunk_size=16, grad_ckpt_policy="none")
    per_layer = model.forward(
        params, cfg, ids, chunk_size=16, grad_ckpt_policy="per_layer"
    )
    assert _tree_allclose(none, per_layer)


def test_loss_parity(cfg):
    _, loss_none = _retrace(cfg, policy="none")
    _, loss_per_layer = _retrace(cfg, policy="per_layer")
    key = jr.PRNGKey(0)
    params = model.init_params(key, cfg)
    a = loss_none(params)
    b = loss_per_layer(params)
    assert _tree_allclose(a, b)


def test_grad_parity(cfg):
    params_none, loss_none = _retrace(cfg, policy="none")
    params_per_layer, loss_per_layer = _retrace(cfg, policy="per_layer")
    g_none = jax.grad(loss_none)(params_none)
    g_per_layer = jax.grad(loss_per_layer)(params_per_layer)
    assert _tree_allclose(g_none, g_per_layer, rtol=2e-2, atol=2e-3)


def test_none_is_bitwise_identical_to_default(cfg):
    """``none`` is the old path: switching the field off changes nothing."""
    key = jr.PRNGKey(3)
    params = model.init_params(key, cfg)
    ids = jr.randint(key, (2, 48), 0, cfg.vocab_size)
    default = model.compute_loss(params, cfg, ids, chunk_size=16)
    explicit_none = model.compute_loss(
        params, cfg, ids, chunk_size=16, grad_ckpt_policy="none"
    )
    assert jnp.array_equal(default, explicit_none)


def test_grad_checkpointing_under_jit(cfg):
    """The mechanism survives jit + value_and_grad (the training path)."""
    _, loss_none = _retrace(cfg, policy="none")
    _, loss_per_layer = _retrace(cfg, policy="per_layer")
    key = jr.PRNGKey(4)
    params = model.init_params(key, cfg)
    v_none, g_none = jax.jit(jax.value_and_grad(loss_none))(params)
    v_per, g_per = jax.jit(jax.value_and_grad(loss_per_layer))(params)
    assert _tree_allclose(v_none, v_per)
    assert _tree_allclose(g_none, g_per, rtol=2e-2, atol=2e-3)


# ---------------------------------------------------------------------------
# full-config memory manifest (opt-in, for the GB10 confirmation run)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("AXIOM_GRAD_CKPT_MEMORY") != "1",
    reason="full-config compile is meant for the GB10 run "
    "(set AXIOM_GRAD_CKPT_MEMORY=1)",
)
def test_full_config_compile_memory_manifest():
    """Compiling value_and_grad at T=8192, vocab 160k must not ask for >50 GiB.

    ``memory_analysis().temp_size_in_bytes`` is the XLA scratch request — the
    quantity that OOMed at ~972 GiB under the coarse wrap (D-8).  The
    parameters are passed as ``ShapeDtypeStruct`` leaves (``eval_shape``), so
    the ~1B-parameter tree is never materialised just to measure the graph.
    """
    cfg = load_config(CONFIG_PATH)
    assert cfg.vocab_size == 160_000
    params_abstract = jax.eval_shape(
        lambda k: model.init_params(k, cfg), jr.PRNGKey(0)
    )
    ids = jax.ShapeDtypeStruct((1, 8192), jnp.int32)

    def loss(p, x):
        return model.compute_loss(p, cfg, x, chunk_size=64)

    lowered = jax.jit(jax.value_and_grad(loss)).lower(params_abstract, ids)
    analysis = lowered.compile().memory_analysis()
    temp = getattr(analysis, "temp_size_in_bytes", None)
    assert temp is not None, "backend did not report memory_analysis"
    temp_gib = temp / (1 << 30)
    print(f"[D-8] per_layer temp_size_in_bytes={temp} ({temp_gib:.2f} GiB)")
    assert temp_gib <= 50.0, (
        f"per-layer remat still asks for {temp_gib:.2f} GiB of scratch "
        f"(goal <= 50 GiB, coarse-wrap baseline ~972 GiB)"
    )
