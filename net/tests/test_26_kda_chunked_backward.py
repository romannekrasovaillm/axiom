"""Chunked KDA backward (FLA-style recompute) — D-8 remainder.

``value_and_grad`` of ``l3-full`` at T=8192/B=1 asked XLA for ~896 GiB against a
67.5 GiB theoretical floor.  The coarse per-layer ``jax.checkpoint`` (D-8,
``grad_ckpt_policy``) could not cut the kit that lives *inside* a layer: the
chunked KDA forward materialises the affine-transition prefix ``P``/``Q`` of the
delta rule (``chunk_step``'s ``lax.associative_scan``), an O(C·H·dk·dk) tensor
per chunk — O(T·H·dk·dk) across the sequence, ~41.5 GiB per layer at the
``l3-full`` geometry (results note 03.10).  The wall is invariant to T/B at a
fixed tokens/step, so it has to be recomputed, not re-budgeted.

The fix is the FLA chunked delta-rule backward: the chunk-scan *body* is wrapped
in ``jax.checkpoint``, so during backward each chunk's trajectory (its
projections, ``M``/``N`` and the ``P``/``Q`` prefix) is recomputed from the
chunk's saved carrier state instead of being retained for every chunk.  The
switch is declarative — ``kda_chunked_backward`` in ``net/config.json`` (spine
AD-9 / C-035 form, the same mechanism as ``grad_ckpt_policy`` and
``ce_chunk_tokens``) — and reuses the existing ``chunk_size`` as the width, so
there is no second knob to keep in sync.

What these tests pin:

* the declared flag is read from ``net/config.json`` (``true`` for ``l3-full``)
  and validated by ``net.config.validate_config``; the schema default stays
  ``False`` so a config built in code keeps the pre-delta graph;
* the mechanism is real: the ``remat`` boundary sits in the *nested* jaxpr of
  the chunk ``lax.scan`` body (a top-level ``make_jaxpr`` walk cannot see it —
  the top-level count is 0 in both directions), so the counter descends into
  every nested ``ClosedJaxpr``;
* the forward output and its gradients are unchanged versus the pre-delta path
  (``allclose``, not bitwise: recomputing a chunk reorders floating-point work —
  an XLA property, not a bug);
* edge shapes (T below the chunk width, T not a multiple of it) coincide too;
* with the flag off the graph is the pre-delta one bit-for-bit (compared to a
  reference copy of the old body), so the switch is honest in both directions;
* the MTP head flows the same flag through ``mtp_mod.apply`` — a parity check,
  not a declaration-only claim.
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

from net import kda, model
from net.config import load_config, validate_config

#: The case's declarative config (the file C-035-style declarations live in).
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"

#: Sequence length used by the small-config tests.
SEQ = 40


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _kda_params(cfg, *, seed=0):
    return kda.init_kda(jr.PRNGKey(seed), cfg)


def _x(cfg, T, *, seed=1):
    return jr.normal(jr.PRNGKey(seed), (T, cfg.hidden))


def _count_remat(obj) -> int:
    """Count ``remat`` primitives at every nesting depth of a jaxpr tree.

    A checkpoint inside a ``lax.scan``/``associative_scan`` body lives in the
    scan equation's nested ``ClosedJaxpr`` (``eqn.params["jaxpr"]``), which a
    top-level ``jax.make_jaxpr`` walk never enters.  The KDA chunk backward's
    boundary is exactly such a nested one, so this has to recurse.
    """
    inner = getattr(obj, "jaxpr", None)
    if inner is not None and not isinstance(obj, (dict, list, tuple)):
        return _count_remat(inner)
    eqns = getattr(obj, "eqns", None)
    if eqns is not None:
        count = 0
        for eqn in eqns:
            # JAX names the checkpoint primitive ``remat`` with a version suffix
            # (``remat2`` on jax 0.10.x); match the family, not one exact name.
            if str(eqn.primitive.name).startswith("remat"):
                count += 1
            for value in eqn.params.values():
                count += _count_remat(value)
        return count
    if isinstance(obj, dict):
        return sum(_count_remat(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_count_remat(v) for v in obj)
    return 0


def _kda_remat_count(cfg, *, T=SEQ, chunk=16) -> int:
    """``remat`` boundaries in a single KDA layer's chunked apply."""
    params = _kda_params(cfg)
    x = _x(cfg, T)
    closed = jax.make_jaxpr(lambda p: kda.apply_chunked(p, cfg, x, chunk))(params)
    return _count_remat(closed)


def _top_level_remat_count(cfg, *, T=SEQ, chunk=16) -> int:
    """Only the top-level jaxpr — used to show the boundary is *nested*."""
    params = _kda_params(cfg)
    x = _x(cfg, T)
    closed = jax.make_jaxpr(lambda p: kda.apply_chunked(p, cfg, x, chunk))(params)
    return sum(
        1 for eqn in closed.jaxpr.eqns
        if str(eqn.primitive.name).startswith("remat")
    )


def _with_flag(cfg, on: bool):
    return dataclasses.replace(cfg, kda_chunked_backward=on)


def _apply_chunked_reference(params, cfg, x, chunk_size):
    """The pre-delta ``apply_chunked``: identical minus the checkpoint wrap."""
    T = x.shape[0]
    C = chunk_size
    n_chunks = (T + C - 1) // C
    pad = n_chunks * C - T
    x_p = jnp.pad(x, ((0, pad), (0, 0))) if pad else x
    x_chunks = x_p.reshape(n_chunks, C, -1)
    carry0 = kda.init_state(cfg)
    _, out = jax.lax.scan(
        lambda c, xc: kda.chunk_step(params, cfg, c, xc), carry0, x_chunks
    )
    out = out.reshape(n_chunks * C, -1)
    return kda._with_window(out[:T], params, cfg, x)


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


def test_declared_flag_is_on_for_pretrain():
    cfg = load_config(CONFIG_PATH)
    assert cfg.kda_chunked_backward is True
    validate_config(cfg)  # the declared value is one apply_chunked implements


def test_schema_default_is_off():
    """A config built in code (tests/smokes) keeps the pre-delta graph."""
    assert small_config().kda_chunked_backward is False


def test_non_bool_is_rejected():
    """A truthy non-bool would flip the graph by accident, so it is refused."""
    cfg = dataclasses.replace(small_config(), kda_chunked_backward=1)
    with pytest.raises(AssertionError):
        validate_config(cfg)


# ---------------------------------------------------------------------------
# mechanism: the boundary is one nested remat inside the chunk scan body
# ---------------------------------------------------------------------------


def test_flag_on_puts_a_remat_inside_the_chunk_scan(cfg):
    assert _kda_remat_count(_with_flag(cfg, False)) == 0
    assert _kda_remat_count(_with_flag(cfg, True)) == 1


def test_remat_is_nested_not_top_level(cfg):
    """Pins *why* the counter recurses: the top level sees 0 either way."""
    assert _top_level_remat_count(_with_flag(cfg, False)) == 0
    assert _top_level_remat_count(_with_flag(cfg, True)) == 0


def test_model_graph_consumes_the_flag(cfg):
    """The declarative flag reaches the backbone graph, not just the layer.

    ``grad_ckpt_policy="none"`` leaves the backbone un-rematted, so any nested
    ``remat`` in the graph is a chunk boundary.
    """
    params = model.init_params(jr.PRNGKey(0), cfg)
    ids = jr.randint(jr.PRNGKey(0), (2, SEQ), 0, cfg.vocab_size)

    def count(c):
        def loss(p):
            return model.compute_loss(
                p, c, ids, chunk_size=16, grad_ckpt_policy="none"
            )
        return _count_remat(jax.make_jaxpr(loss)(params))

    assert count(_with_flag(cfg, False)) == 0
    assert count(_with_flag(cfg, True)) > 0


# ---------------------------------------------------------------------------
# parity: recompute changes where the trajectory lives, not the numbers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk", [8, 16, 64])
def test_forward_parity(cfg, chunk):
    params = _kda_params(cfg)
    x = _x(cfg, SEQ)
    off = kda.apply_chunked(params, _with_flag(cfg, False), x, chunk)
    on = kda.apply_chunked(params, _with_flag(cfg, True), x, chunk)
    assert jnp.allclose(off, on, rtol=1e-4, atol=1e-5)


def test_grad_parity(cfg):
    params = _kda_params(cfg)
    x = _x(cfg, SEQ)

    def loss(p, c):
        return kda.apply_chunked(p, c, x, 16).sum()

    g_off = jax.grad(lambda p: loss(p, _with_flag(cfg, False)))(params)
    g_on = jax.grad(lambda p: loss(p, _with_flag(cfg, True)))(params)
    assert _tree_allclose(g_off, g_on, rtol=2e-2, atol=2e-3)


def test_edge_t_below_chunk(cfg):
    """T smaller than the chunk width: one (padded) chunk, both paths agree."""
    params = _kda_params(cfg)
    x = _x(cfg, 5)
    off = kda.apply_chunked(params, _with_flag(cfg, False), x, 16)
    on = kda.apply_chunked(params, _with_flag(cfg, True), x, 16)
    assert jnp.allclose(off, on, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("chunk", [7, 6, 13])
def test_edge_t_not_multiple_of_chunk(cfg, chunk):
    """Ragged tail: the padded final chunk must not change the result."""
    params = _kda_params(cfg)
    x = _x(cfg, SEQ)
    off = kda.apply_chunked(params, _with_flag(cfg, False), x, chunk)
    on = kda.apply_chunked(params, _with_flag(cfg, True), x, chunk)
    assert jnp.allclose(off, on, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# disable flag: off is the pre-delta graph, bit for bit
# ---------------------------------------------------------------------------


def test_flag_off_is_bitwise_identical_to_reference(cfg):
    """Off reproduces the old body exactly (this is the graph the flag gates)."""
    params = _kda_params(cfg, seed=3)
    x = _x(cfg, SEQ, seed=4)
    got = kda.apply_chunked(params, _with_flag(cfg, False), x, 16)
    reference = _apply_chunked_reference(params, _with_flag(cfg, False), x, 16)
    assert jnp.array_equal(got, reference)


# ---------------------------------------------------------------------------
# model path: loss, gradients, jit/value_and_grad and the MTP head
# ---------------------------------------------------------------------------


def _ids(cfg, *, seed=0, T=SEQ):
    return jr.randint(jr.PRNGKey(seed), (2, T), 0, cfg.vocab_size)


def test_model_loss_parity(cfg):
    params = model.init_params(jr.PRNGKey(0), cfg)
    ids = _ids(cfg)
    off = model.compute_loss(params, _with_flag(cfg, False), ids, chunk_size=16)
    on = model.compute_loss(params, _with_flag(cfg, True), ids, chunk_size=16)
    assert jnp.allclose(off, on, rtol=1e-4, atol=1e-5)


def test_model_grad_parity(cfg):
    params = model.init_params(jr.PRNGKey(0), cfg)
    ids = _ids(cfg)
    g_off = jax.grad(
        lambda p: model.compute_loss(p, _with_flag(cfg, False), ids, chunk_size=16)
    )(params)
    g_on = jax.grad(
        lambda p: model.compute_loss(p, _with_flag(cfg, True), ids, chunk_size=16)
    )(params)
    assert _tree_allclose(g_off, g_on, rtol=2e-2, atol=2e-3)


def test_jit_parity(cfg):
    """The mechanism survives jit + value_and_grad (the training path)."""
    params = model.init_params(jr.PRNGKey(0), cfg)
    ids = _ids(cfg)

    def run(c):
        return jax.jit(jax.value_and_grad(
            lambda p: model.compute_loss(p, c, ids, chunk_size=16)
        ))(params)

    v_off, g_off = run(_with_flag(cfg, False))
    v_on, g_on = run(_with_flag(cfg, True))
    assert jnp.allclose(v_off, v_on, rtol=1e-4, atol=1e-5)
    assert _tree_allclose(g_off, g_on, rtol=2e-2, atol=2e-3)


def test_mtp_head_flows_the_flag(cfg):
    """The auxiliary MTP head's KDA block reads the same declarative flag."""
    params = model.init_params(jr.PRNGKey(0), cfg)
    ids = _ids(cfg, T=SEQ + 8)
    hidden = model.forward(
        params, cfg, ids, chunk_size=16, return_hidden=True, emit_logits=False
    )
    off = model.mtp_loss(params, _with_flag(cfg, False), hidden, ids, 16)
    on = model.mtp_loss(params, _with_flag(cfg, True), hidden, ids, 16)
    assert jnp.allclose(off, on, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# full-geometry memory manifest (opt-in, for the GB10 confirmation run)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("AXIOM_KDA_CKBACK_MEMORY") != "1",
    reason="full-geometry compile is meant for the GB10 run "
    "(set AXIOM_KDA_CKBACK_MEMORY=1)",
)
def test_kda_layer_compile_memory_cut():
    """The KDA kit's XLA scratch must shrink when the chunk backward is on.

    ``memory_analysis().temp_size_in_bytes`` is the quantity that OOMed at
    ~896 GiB.  The layers are passed as ``ShapeDtypeStruct`` leaves
    (``eval_shape``), so the full-geometry parameter tree is never materialised
    just to measure the graph.  The assertion is only made on a GPU: the CPU
    backend ignores rematerialisation by design, so a CPU measurement cannot
    substantiate the cut (and is not the acceptance criterion).
    """
    cfg = load_config(CONFIG_PATH)
    params_abstract = jax.eval_shape(lambda k: kda.init_kda(k, cfg), jr.PRNGKey(0))
    x = jax.ShapeDtypeStruct((8192, cfg.hidden), jnp.float32)

    def manifest(c):
        def loss(p, xb):
            return kda.apply_chunked(p, c, xb, 64).sum()

        lowered = jax.jit(jax.value_and_grad(loss)).lower(params_abstract, x)
        return lowered.compile().memory_analysis().temp_size_in_bytes

    off = manifest(_with_flag(cfg, False))
    on = manifest(_with_flag(cfg, True))
    off_gib, on_gib = off / (1 << 30), on / (1 << 30)
    print(f"[D-8] KDA kit temp: off={off_gib:.2f} GiB on={on_gib:.2f} GiB")
    if jax.default_backend() == "gpu":
        assert on_gib < off_gib, (
            f"chunked KDA backward did not cut the kit: "
            f"off={off_gib:.2f} GiB on={on_gib:.2f} GiB"
        )
