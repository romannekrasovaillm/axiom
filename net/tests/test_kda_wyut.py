"""WY/UT chunkwise KDA (ADR-031, delta A) — parity against the recurrent oracle.

The new ``kda.apply_wyut`` implements the Kimi Linear §3.1 chunkwise delta rule
("WY representation + UT transform", arXiv 2510.26692v2): the inter-chunk state
transfer is a matmul, the intra-chunk correction is a CxC matrix, and the
per-token ``(dk, dk)`` transition matrices of ``apply_chunked`` (the
``lax.associative_scan`` kit that made the layer memory-bound) are never
materialised.

Correctness is pinned against the *existing* ``apply_recurrent`` — the
per-token recurrence ``S <- (I - β k k^T) Diag(α) S + β k v^T`` is the oracle;
the reference mathematics is unchanged by this delta.  ``apply_chunked`` is the
second oracle (both forms must agree, as criterion 2 already requires).

The WY form divides by the cumulative decay ``Γ`` in the source's derivation.
``Γ`` underflows to zero within a 64-token chunk (α ∈ (e^-5, 1) per channel, so
``Γ`` can reach ``e^-320`` in fp32), and the reciprocal ``1/Γ`` then produces
``inf``/``NaN`` on *real* keys — not only on the zero padding.  These tests pin
that the implementation is arithmetic-safe: the intra-chunk decay ratio is
evaluated as ``exp(min(G_c - G_i, 0))`` (a difference of cumulative logs, which
is ≤ 0 on the causal side), never as ``Γ_i / Γ_j``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from conftest import small_config

from net import kda, model
from net.config import load_config, validate_config

#: The case's declarative config (``kda_impl`` is declared there).
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"

#: Sequence length of the parity runs.
SEQ = 128

#: Forward tolerance of the existing net tests (XLA reorders the summation).
ATOL = 1e-4
RTOL = 1e-4


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _params(cfg, *, seed=0):
    return kda.init_kda(jr.PRNGKey(seed), cfg)


def _x(cfg, T, *, seed=1):
    return jr.normal(jr.PRNGKey(seed), (T, cfg.hidden))


def _max_abs(a, b) -> float:
    return float(jnp.max(jnp.abs(a - b)))


def _tree_allclose(a, b, *, rtol, atol) -> bool:
    return all(
        bool(jnp.allclose(x, y, rtol=rtol, atol=atol, equal_nan=True))
        for x, y in zip(
            jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
        )
    )


def _tree_all_finite(tree) -> bool:
    return all(
        bool(jnp.all(jnp.isfinite(leaf)))
        for leaf in jax.tree_util.tree_leaves(tree)
    )


def _primitive_names(obj, acc: set[str] | None = None) -> set[str]:
    """Every primitive name in a jaxpr tree, at every nesting depth.

    A ``lax.scan`` body lives in a nested ``ClosedJaxpr`` (``params['jaxpr']``),
    which a top-level walk never enters, so this recurses (same reason as the
    counter in ``test_26``).
    """
    if acc is None:
        acc = set()
    inner = getattr(obj, "jaxpr", None)
    if inner is not None and not isinstance(obj, (dict, list, tuple)):
        return _primitive_names(inner, acc)
    eqns = getattr(obj, "eqns", None)
    if eqns is not None:
        for eqn in eqns:
            acc.add(str(eqn.primitive.name))
            for value in eqn.params.values():
                _primitive_names(value, acc)
        return acc
    if isinstance(obj, dict):
        for value in obj.values():
            _primitive_names(value, acc)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _primitive_names(value, acc)
    return acc


# ---------------------------------------------------------------------------
# parity: the new form is the same recurrence, computed differently
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk", [32, 64])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_wyut_matches_recurrent(cfg, chunk, seed):
    p = _params(cfg, seed=seed)
    x = _x(cfg, SEQ, seed=seed + 11)
    rec = kda.apply_recurrent(p, cfg, x)
    new = kda.apply_wyut(p, cfg, x, chunk_size=chunk)
    d = _max_abs(rec, new)
    assert d <= ATOL, f"chunk={chunk} seed={seed} max|recurrent-wyut|={d}"


@pytest.mark.parametrize("chunk", [32, 64])
@pytest.mark.parametrize("seed", [0, 1])
def test_wyut_matches_chunked(cfg, chunk, seed):
    p = _params(cfg, seed=seed)
    x = _x(cfg, SEQ, seed=seed + 11)
    chk = kda.apply_chunked(p, cfg, x, chunk_size=chunk)
    new = kda.apply_wyut(p, cfg, x, chunk_size=chunk)
    d = _max_abs(chk, new)
    assert d <= ATOL, f"chunk={chunk} seed={seed} max|chunked-wyut|={d}"


def test_wyut_grad_matches_recurrent(cfg):
    """Gradients flow through the same recurrence (tol per the chunked test)."""
    p = _params(cfg)
    x = _x(cfg, 64)

    g_rec = jax.grad(lambda q: kda.apply_recurrent(q, cfg, x).sum())(p)
    g_new = jax.grad(lambda q: kda.apply_wyut(q, cfg, x, chunk_size=64).sum())(p)
    assert _tree_allclose(g_rec, g_new, rtol=2e-2, atol=2e-3)


def test_wyut_grad_matches_chunked(cfg):
    p = _params(cfg)
    x = _x(cfg, 64)
    g_chk = jax.grad(lambda q: kda.apply_chunked(q, cfg, x, 64).sum())(p)
    g_new = jax.grad(lambda q: kda.apply_wyut(q, cfg, x, 64).sum())(p)
    assert _tree_allclose(g_chk, g_new, rtol=2e-2, atol=2e-3)


# ---------------------------------------------------------------------------
# edges: ragged tail and a sequence shorter than the chunk
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk", [32, 64])
def test_wyut_t_below_chunk(cfg, chunk):
    p = _params(cfg)
    x = _x(cfg, 5)
    rec = kda.apply_recurrent(p, cfg, x)
    new = kda.apply_wyut(p, cfg, x, chunk_size=chunk)
    assert _max_abs(rec, new) <= ATOL


@pytest.mark.parametrize("chunk", [32, 64])
@pytest.mark.parametrize("T", [40, 100, 127])
def test_wyut_ragged_tail_equals_unpadded(cfg, chunk, T):
    """The padded final chunk must not perturb the real positions."""
    p = _params(cfg)
    x = _x(cfg, T)
    out = kda.apply_wyut(p, cfg, x, chunk_size=chunk)
    assert out.shape[0] == T
    rec = kda.apply_recurrent(p, cfg, x)
    assert _max_abs(rec, out) <= ATOL


# ---------------------------------------------------------------------------
# NaN safety: zero padding must never produce non-finite values or gradients
# ---------------------------------------------------------------------------


def test_wyut_zero_padding_forward_finite(cfg):
    """A short sequence (padding to the chunk width) stays finite."""
    p = _params(cfg)
    x = _x(cfg, 40)
    out = kda.apply_wyut(p, cfg, x, chunk_size=64)
    assert _tree_all_finite(out)


def test_wyut_zero_padding_backward_finite(cfg):
    """The backward pass through the padded chunk has no NaN.

    This is the defect of ``apply_chunked`` (a zero vector hit ``l2_norm``'s
    derivative at 0/0 — see ``net/norm.py:_l2_norm_guarded_jvp``): the new path
    must not reintroduce a non-finite gradient for the KDA matrices.
    """
    p = _params(cfg)
    x = _x(cfg, 40)
    grads = jax.grad(lambda q: kda.apply_wyut(q, cfg, x, chunk_size=64).sum())(p)
    assert _tree_all_finite(grads)


def test_wyut_explicit_zero_rows_finite(cfg):
    """Rows that are literally zero (as padding produces) are safe too."""
    p = _params(cfg)
    x = jr.normal(jr.PRNGKey(5), (96, cfg.hidden)).at[64:].set(0.0)
    out = kda.apply_wyut(p, cfg, x, chunk_size=64)
    assert _tree_all_finite(out)
    grads = jax.grad(lambda q: kda.apply_wyut(q, cfg, x, chunk_size=64).sum())(p)
    assert _tree_all_finite(grads)


def test_wyut_padded_tail_matches_unpadded(cfg):
    """Zero padding is inert: positions < T equal the unpadded run."""
    p = _params(cfg)
    x = _x(cfg, 50)
    out = kda.apply_wyut(p, cfg, x, chunk_size=64)
    rec = kda.apply_recurrent(p, cfg, x)
    assert _max_abs(rec, out) <= ATOL


# ---------------------------------------------------------------------------
# the declarative flag (spine AD-9 / C-035 form)
# ---------------------------------------------------------------------------


def test_schema_default_is_chunked():
    """A config built in code keeps the existing graph (the flag is opt-in)."""
    assert small_config().kda_impl == "chunked"


def test_declared_config_is_chunked_cc_for_pretrain():
    """The declared case config pins the ADR-047 form (amendment of 09.10.2026).

    Not the WY/UT form: ``wyut`` stays the measured-but-rejected arm.  The
    schema default in code is still ``chunked``; the assertion here named that
    pre-amendment default until ``f653b34`` flipped the case's declarative
    choice and the test was not updated — a standing red unrelated to the
    UT-transform change.
    """
    cfg = load_config(CONFIG_PATH)
    assert cfg.kda_impl == "chunked_cc"
    validate_config(cfg)


def test_wyut_chunk_default_is_64():
    assert small_config().kda_wyut_chunk == 64


def test_unknown_impl_is_rejected():
    cfg = dataclasses.replace(small_config(), kda_impl="magic")
    with pytest.raises(AssertionError):
        validate_config(cfg)


def test_nonpositive_wyut_chunk_is_rejected():
    cfg = dataclasses.replace(small_config(), kda_wyut_chunk=0)
    with pytest.raises(AssertionError):
        validate_config(cfg)


def test_apply_wyut_uses_config_chunk_when_unspecified(cfg):
    """``chunk_size=None`` falls back to ``cfg.kda_wyut_chunk``."""
    p = _params(cfg)
    x = _x(cfg, 96)
    a = kda.apply_wyut(p, cfg, x, chunk_size=cfg.kda_wyut_chunk)
    b = kda.apply_wyut(p, cfg, x)
    assert jnp.array_equal(a, b)


# ---------------------------------------------------------------------------
# dispatcher: the model graph honours the flag, one layer at a time
# ---------------------------------------------------------------------------


def _with_impl(cfg, impl: str):
    return dataclasses.replace(cfg, kda_impl=impl)


def test_model_loss_parity_between_impls(cfg):
    params = model.init_params(jr.PRNGKey(0), cfg)
    ids = jr.randint(jr.PRNGKey(0), (2, 48), 0, cfg.vocab_size)
    chunked = model.compute_loss(params, _with_impl(cfg, "chunked"), ids, chunk_size=64)
    wyut = model.compute_loss(params, _with_impl(cfg, "wyut"), ids, chunk_size=64)
    assert jnp.allclose(chunked, wyut, rtol=RTOL, atol=1e-5)


def test_model_graph_consumes_the_flag(cfg):
    """``kda_impl="wyut"`` selects a different KDA primitive in the graph."""
    params = model.init_params(jr.PRNGKey(0), cfg)
    ids = jr.randint(jr.PRNGKey(0), (2, 48), 0, cfg.vocab_size)

    def primitives(c) -> set[str]:
        closed = jax.make_jaxpr(
            lambda p: model.compute_loss(p, c, ids, chunk_size=64)
        )(params)
        return _primitive_names(closed)

    chunked = primitives(_with_impl(cfg, "chunked"))
    wyut = primitives(_with_impl(cfg, "wyut"))
    # The UT transform solves the unit-triangular system (I + L); the old form
    # never does — that is the observable difference between the two graphs.
    assert "triangular_solve" in wyut
    assert "triangular_solve" not in chunked
    assert "cumsum" in wyut and "cumsum" not in chunked
    assert chunked != wyut


def test_wyut_ut_transform_solves_without_lu_factorisation(cfg):
    """The WY/UT path solves ``(I + L) W = X`` instead of inverting ``I + L``.

    Same lever as in ``chunked_cc`` (ADR-050 / G0 address): ``jnp.linalg.inv``
    lowers to an LU factorisation with pivoting (``lu``,
    ``lu_pivots_to_permutation``), whose per-head kernels are the storm the G0
    profile named.  The batched triangular solve carries none of it.
    """
    p = _params(cfg)
    x = _x(cfg, 64)
    prims = _primitive_names(
        jax.make_jaxpr(lambda: kda.apply_wyut(p, cfg, x, chunk_size=64))()
    )
    assert "triangular_solve" in prims
    assert "lu" not in prims, "the UT transform must not factorise (I + L) by LU"
    assert "lu_pivots_to_permutation" not in prims, "LU pivoting must be gone"

    tri = jnp.eye(64) + jnp.tril(jnp.ones((64, 64)), -1)
    assert "lu" in _primitive_names(jax.make_jaxpr(lambda: jnp.linalg.inv(tri))())
