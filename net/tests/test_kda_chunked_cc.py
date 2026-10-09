"""``chunked_cc`` — KDA chunkwise delta rule with a C x C intra-chunk matrix.

ADR-047 rewrites the *implementation* of the KDA layer without touching the
mathematics: inside a chunk the delta rule is a masked ``C x C`` attention-like
matrix; between chunks only the compact ``dk x dv`` state travels.  The two
existing forms are kept as fallbacks:

* ``chunked`` builds the per-token transition monoid ``(C, H, dk, dk)`` (12.6M
  elements per chunk at the case geometry) and runs ``lax.associative_scan``
  over the matrix compositions — memory and bandwidth bound (2.17% MFU);
* ``wyut`` has the target *structure* but materialises the decay-ratio tensor
  ``e = (H, C, C, dk)`` for the score matrices — measured slower (97.9 s vs
  85.5 s per step) and OOM (46.81 GiB) on the model.

``apply_chunked_cc`` is the same WY/UT recurrence with the score matrices built
**tile-wise**: the row factor is ``exp(G_c - G_{p})`` and the column factor is
``exp(G_{p} - G_i)`` against the first position ``p`` of the row's tile, so
both are ``<= 1`` except within one tile, where the exponent is bounded by
``|g_min| * (tile - 1)`` and stays inside fp32/bf16.  No ``(C, C, dk)`` tensor
is built; the intra-chunk object is the ``C x C`` matrix itself.

Correctness is pinned against ``apply_recurrent`` (the per-token oracle), with
``apply_chunked`` as the second oracle — exactly as the existing WY/UT tests do.
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
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
# helpers (same conventions as ``test_kda_wyut.py``)
# ---------------------------------------------------------------------------


def _params(cfg, *, seed=0):
    return kda.init_kda(jr.PRNGKey(seed), cfg)


def _x(cfg, T, *, seed=1, dtype=jnp.float32):
    return jr.normal(jr.PRNGKey(seed), (T, cfg.hidden), dtype=dtype)


def _max_abs(a, b) -> float:
    return float(jnp.max(jnp.abs(a - b)))


def _tree_allclose(a, b, *, rtol, atol) -> bool:
    return all(
        bool(jnp.allclose(x, y, rtol=rtol, atol=atol, equal_nan=True))
        for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b))
    )


def _tree_all_finite(tree) -> bool:
    return all(
        bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree_util.tree_leaves(tree)
    )


def _primitive_names(obj, acc: set[str] | None = None) -> set[str]:
    """Every primitive name in a jaxpr tree, at every nesting depth.

    A ``lax.scan`` body lives in a nested ``ClosedJaxpr`` (``params['jaxpr']``),
    which a top-level walk never enters, so this recurses (same reason as in
    ``test_kda_wyut.py``).
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


def _var_shapes(obj, acc: set[tuple] | None = None) -> set[tuple]:
    """Every intermediate shape in a jaxpr tree, at every nesting depth.

    This is how the rewrite's *memory* claim is checked without a GPU: the
    ``wyut`` form must show a ``(H, C, C, dk)`` variable (the decay-ratio tensor
    that made it heavier than ``chunked``), ``chunked`` must show the per-token
    ``(C, H, dk, dk)`` transition kit, and ``chunked_cc`` must show neither.
    """
    if acc is None:
        acc = set()
    inner = getattr(obj, "jaxpr", None)
    if inner is not None and not isinstance(obj, (dict, list, tuple)):
        return _var_shapes(inner, acc)
    eqns = getattr(obj, "eqns", None)
    if eqns is not None:
        for eqn in eqns:
            for var in list(eqn.invars) + list(eqn.outvars):
                try:
                    acc.add(tuple(var.aval.shape))
                except AttributeError:  # literals / tracers without an aval
                    pass
            for value in eqn.params.values():
                _var_shapes(value, acc)
        return acc
    if isinstance(obj, dict):
        for value in obj.values():
            _var_shapes(value, acc)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _var_shapes(value, acc)
    return acc


# ---------------------------------------------------------------------------
# parity: the new form is the same recurrence, computed differently
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk", [16, 32, 64, 128])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cc_matches_recurrent(cfg, chunk, seed):
    p = _params(cfg, seed=seed)
    x = _x(cfg, SEQ, seed=seed + 11)
    rec = kda.apply_recurrent(p, cfg, x)
    new = kda.apply_chunked_cc(p, cfg, x, chunk_size=chunk)
    d = _max_abs(rec, new)
    assert d <= ATOL, f"chunk={chunk} seed={seed} max|recurrent-cc|={d}"


@pytest.mark.parametrize("chunk", [16, 32, 64, 128])
@pytest.mark.parametrize("seed", [0, 1])
def test_cc_matches_chunked(cfg, chunk, seed):
    p = _params(cfg, seed=seed)
    x = _x(cfg, SEQ, seed=seed + 11)
    chk = kda.apply_chunked(p, cfg, x, chunk_size=chunk)
    new = kda.apply_chunked_cc(p, cfg, x, chunk_size=chunk)
    d = _max_abs(chk, new)
    assert d <= ATOL, f"chunk={chunk} seed={seed} max|chunked-cc|={d}"


@pytest.mark.parametrize("chunk", [32, 64])
def test_cc_matches_wyut(cfg, chunk):
    """The two C x C forms are the same arithmetic, tiled differently."""
    p = _params(cfg)
    x = _x(cfg, SEQ)
    a = kda.apply_wyut(p, cfg, x, chunk_size=chunk)
    b = kda.apply_chunked_cc(p, cfg, x, chunk_size=chunk)
    assert _max_abs(a, b) <= ATOL


def test_cc_scores_match_the_full_decay_ratio_tensor(cfg):
    """The tile-wise factors are the same number as the ``(C, C, dk)`` ratio.

    Direct unit check of :func:`kda._cc_scores` against the brute-force form
    ``sum_d exp(min(G_c - G_i, 0)) row_c[d] col_i[d]`` (``wyut``'s tensor, exact
    by construction because the clamp keeps the exponent ``<= 0``) under the
    case's own decay range ``alpha in (e^-5, 1)`` — where the undefined
    single-reference factorisation would already overflow.
    """
    C = 32
    alpha = jnp.exp(-5.0 * jax.nn.sigmoid(3.0 * jr.normal(jr.PRNGKey(4), (C, cfg.num_heads, cfg.kda_dk))))
    log_g = kda._log_cumulative_decay(alpha)
    log_g_t = log_g.transpose(1, 0, 2)
    row = jr.normal(jr.PRNGKey(5), (C, cfg.num_heads, cfg.kda_dk)).transpose(1, 0, 2)
    col = jr.normal(jr.PRNGKey(6), (C, cfg.num_heads, cfg.kda_dk)).transpose(1, 0, 2)

    ratio = kda._decay_ratio_exp(log_g)  # (H, C, C, dk), exact and finite
    assert bool(jnp.all(jnp.isfinite(ratio)))
    for strict in (False, True):
        brute = jnp.einsum("hcid,hcd,hid->hci", ratio, row, col)
        mask = jnp.tril(jnp.ones((C, C), dtype=bool), k=-1 if strict else 0)
        brute = jnp.where(mask, brute, 0.0)
        got = kda._cc_scores(row, col, log_g_t, kda._cc_tile(cfg), strict=strict)
        scale = max(1.0, float(jnp.max(jnp.abs(brute))))
        assert float(jnp.max(jnp.abs(got - brute))) / scale <= 1e-5, (
            f"strict={strict}: tile-wise scores diverge from the ratio tensor"
        )


def test_cc_long_decay_chunk_stays_finite(cfg):
    """A long chunk (many tiles) with the strongest decay must stay finite.

    This is the defect the tile-wise factors exist for: the single-reference
    form (``wyut``) divides by the cumulative decay, which underflows to zero
    within a 64-token chunk.
    """
    p = _params(cfg)
    x = _x(cfg, 256)
    out = kda.apply_chunked_cc(p, cfg, x, chunk_size=256)
    assert _tree_all_finite(out)
    g = jax.grad(lambda q: kda.apply_chunked_cc(q, cfg, x, chunk_size=256).sum())(p)
    assert _tree_all_finite(g)


# ---------------------------------------------------------------------------
# edges: ragged tail and a sequence shorter than the chunk
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk", [16, 64])
def test_cc_t_below_chunk(cfg, chunk):
    p = _params(cfg)
    x = _x(cfg, 5)
    rec = kda.apply_recurrent(p, cfg, x)
    new = kda.apply_chunked_cc(p, cfg, x, chunk_size=chunk)
    assert _max_abs(rec, new) <= ATOL


@pytest.mark.parametrize("chunk", [16, 64])
@pytest.mark.parametrize("T", [40, 100, 127])
def test_cc_ragged_tail_equals_unpadded(cfg, chunk, T):
    """The zero-padded final chunk must not perturb the real positions."""
    p = _params(cfg)
    x = _x(cfg, T)
    out = kda.apply_chunked_cc(p, cfg, x, chunk_size=chunk)
    assert out.shape[0] == T
    rec = kda.apply_recurrent(p, cfg, x)
    assert _max_abs(rec, out) <= ATOL


def test_cc_zero_padding_backward_finite(cfg):
    """Zero padding (the 0/0 tangent of ``l2_norm``) must not leak a NaN."""
    p = _params(cfg)
    x = _x(cfg, 40)
    out = kda.apply_chunked_cc(p, cfg, x, chunk_size=64)
    assert _tree_all_finite(out)
    grads = jax.grad(lambda q: kda.apply_chunked_cc(q, cfg, x, chunk_size=64).sum())(p)
    assert _tree_all_finite(grads)


def test_cc_explicit_zero_rows_finite(cfg):
    p = _params(cfg)
    x = jr.normal(jr.PRNGKey(5), (96, cfg.hidden)).at[64:].set(0.0)
    grads = jax.grad(lambda q: kda.apply_chunked_cc(q, cfg, x, chunk_size=64).sum())(p)
    assert _tree_all_finite(kda.apply_chunked_cc(p, cfg, x, chunk_size=64))
    assert _tree_all_finite(grads)


# ---------------------------------------------------------------------------
# gradients: the backward pass is built and agrees with the oracles
# ---------------------------------------------------------------------------


def test_cc_grad_matches_recurrent(cfg):
    p = _params(cfg)
    x = _x(cfg, 64)
    g_rec = jax.grad(lambda q: kda.apply_recurrent(q, cfg, x).sum())(p)
    g_new = jax.grad(lambda q: kda.apply_chunked_cc(q, cfg, x, chunk_size=64).sum())(p)
    assert _tree_allclose(g_rec, g_new, rtol=2e-2, atol=2e-3)


def test_cc_grad_matches_chunked(cfg):
    p = _params(cfg)
    x = _x(cfg, 64)
    g_chk = jax.grad(lambda q: kda.apply_chunked(q, cfg, x, 64).sum())(p)
    g_new = jax.grad(lambda q: kda.apply_chunked_cc(q, cfg, x, 64).sum())(p)
    assert _tree_allclose(g_chk, g_new, rtol=2e-2, atol=2e-3)


#: The finite-difference leg runs in its own process: it needs ``float64``, and
#: ``jax.config.update("jax_enable_x64", True)`` would otherwise leak into every
#: test that runs afterwards in the same session (ADR-010 pins the rest).
_NUMERIC_SCRIPT = r"""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr

from net import kda
from net.config import ModelConfig

cfg = ModelConfig(
    hidden=16, num_heads=2, head_dim=8, kda_dk=8, kda_dv=8,
    kda_decay_rank=8, swa_window=0, kda_wyut_chunk=8,
)
params = jax.tree_util.tree_map(
    lambda a: jnp.asarray(a, dtype=jnp.float64), kda.init_kda(jr.PRNGKey(0), cfg)
)
x = jr.normal(jr.PRNGKey(1), (24, cfg.hidden), dtype=jnp.float64) * 0.5


def loss(xv):
    return kda.apply_chunked_cc(params, cfg, xv, chunk_size=8).sum()


grads = jax.grad(loss)(x)
eps = 1e-6
worst = 0.0
for (t, d) in ((0, 0), (7, 3), (23, cfg.hidden - 1)):
    numeric = (loss(x.at[t, d].add(eps)) - loss(x.at[t, d].add(-eps))) / (2 * eps)
    worst = max(worst, float(abs(numeric - grads[t, d])))
assert worst < 1e-6, f"worst |numeric - analytic| = {worst}"
print("worst_abs", worst)
"""


def test_cc_grad_matches_numerical_check(tmp_path: Path):
    """Central differences (float64, own process) confirm the autodiff path."""
    script = tmp_path / "cc_numerical_grad.py"
    script.write_text(_NUMERIC_SCRIPT, encoding="utf-8")
    root = str(Path(__file__).resolve().parents[2])
    env = {
        **os.environ,
        "JAX_ENABLE_X64": "1",
        "JAX_PLATFORMS": "cpu",
        # ``sys.path[0]`` is the *script's* directory, so the repo root has to be
        # passed explicitly for ``from net import kda`` to resolve.
        "PYTHONPATH": os.pathsep.join(p for p in (root, os.environ.get("PYTHONPATH", "")) if p),
    }
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert "worst_abs" in proc.stdout


def test_cc_grad_through_chunked_backward_flag(cfg):
    """``kda_chunked_backward`` (remat of the scan body) keeps the gradient."""
    p = _params(cfg)
    x = _x(cfg, 64)
    off = dataclasses.replace(cfg, kda_chunked_backward=False)
    on = dataclasses.replace(cfg, kda_chunked_backward=True)
    g_off = jax.grad(lambda q: kda.apply_chunked_cc(q, off, x, 64).sum())(p)
    g_on = jax.grad(lambda q: kda.apply_chunked_cc(q, on, x, 64).sum())(p)
    # Rematerialisation recomputes the chunk trajectory, so the rounding of the
    # two legs differs; the tolerance is the one the existing grad tests use.
    assert _tree_allclose(g_off, g_on, rtol=1e-3, atol=1e-4)
    assert _tree_all_finite(g_on)


# ---------------------------------------------------------------------------
# the carried state and the window branch are the same objects
# ---------------------------------------------------------------------------


def test_cc_chunk_state_matches_chunk_step(cfg):
    """One chunk through either body yields the same state and ShortConv buffers."""
    p = _params(cfg)
    x = _x(cfg, 32)
    carry = kda.init_state(cfg)
    s_chk, o_chk = kda.chunk_step(p, cfg, carry, x)
    s_cc, o_cc = kda.cc_chunk_step(p, cfg, carry, x)
    assert s_cc.S.shape == (cfg.num_heads, cfg.kda_dk, cfg.kda_dv)
    assert _max_abs(s_chk.S, s_cc.S) <= ATOL
    # ShortConv buffers are head-independent bookkeeping: identical by construction.
    assert jnp.array_equal(s_chk.q_buf, s_cc.q_buf)
    assert jnp.array_equal(s_chk.k_buf, s_cc.k_buf)
    assert jnp.array_equal(s_chk.v_buf, s_cc.v_buf)
    assert o_cc.shape == o_chk.shape == (32, cfg.hidden)
    assert _max_abs(o_chk, o_cc) <= ATOL


def _windowed_cfg(cfg):
    return dataclasses.replace(cfg, attn_dense_reference=False, swa_window=8)


def test_cc_window_branch_parity(cfg):
    """With the SWA window on, both forms still agree (``_with_window`` shared)."""
    wcfg = _windowed_cfg(cfg)
    p = _params(wcfg)
    x = _x(wcfg, 64)
    chk = kda.apply_chunked(p, wcfg, x, chunk_size=32)
    new = kda.apply_chunked_cc(p, wcfg, x, chunk_size=32)
    assert _max_abs(chk, new) <= ATOL


def test_cc_dedicated_window_projections_parity(cfg):
    """The rejected (dedicated-window-projection) window variant routes the same."""
    wcfg = dataclasses.replace(
        cfg, attn_dense_reference=False, swa_window=8, swa_share_kda_projections=False
    )
    p = _params(wcfg)
    x = _x(wcfg, 64)
    chk = kda.apply_chunked(p, wcfg, x, chunk_size=32)
    new = kda.apply_chunked_cc(p, wcfg, x, chunk_size=32)
    assert _max_abs(chk, new) <= ATOL


# ---------------------------------------------------------------------------
# the declarative flag (spine AD-9 / C-035 form)
# ---------------------------------------------------------------------------


def test_schema_default_is_chunked():
    assert small_config().kda_impl == "chunked"


def test_declared_config_is_chunked_for_pretrain():
    cfg = load_config(CONFIG_PATH)
    assert cfg.kda_impl == "chunked_cc"
    validate_config(cfg)


def test_schema_accepts_chunked_cc():
    cfg = dataclasses.replace(small_config(), kda_impl="chunked_cc")
    validate_config(cfg)  # must not raise


def test_unknown_impl_is_rejected():
    cfg = dataclasses.replace(small_config(), kda_impl="magic")
    with pytest.raises(AssertionError):
        validate_config(cfg)


def test_apply_cc_uses_config_chunk_when_unspecified(cfg):
    p = _params(cfg)
    x = _x(cfg, 96)
    a = kda.apply_chunked_cc(p, cfg, x, chunk_size=cfg.kda_wyut_chunk)
    b = kda.apply_chunked_cc(p, cfg, x)
    assert jnp.array_equal(a, b)


def test_dispatcher_routes_chunked_cc(cfg):
    cc_cfg = dataclasses.replace(cfg, kda_impl="chunked_cc")
    p = _params(cfg)
    x = _x(cfg, 96)
    via_flag = kda.apply_kda(p, cc_cfg, x, chunk_size=32)
    direct = kda.apply_chunked_cc(p, cc_cfg, x, chunk_size=32)
    assert jnp.array_equal(via_flag, direct)


def test_default_dispatcher_path_is_unchanged(cfg):
    """``chunked`` stays the working default: the flag's absence means the old form."""
    p = _params(cfg)
    x = _x(cfg, 96)
    assert jnp.array_equal(
        kda.apply_kda(p, cfg, x, chunk_size=32),
        kda.apply_chunked(p, cfg, x, chunk_size=32),
    )


# ---------------------------------------------------------------------------
# the model graph consumes the flag, one layer at a time
# ---------------------------------------------------------------------------


def _with_impl(cfg, impl: str):
    return dataclasses.replace(cfg, kda_impl=impl)


def test_model_loss_parity_between_impls(cfg):
    params = model.init_params(jr.PRNGKey(0), cfg)
    ids = jr.randint(jr.PRNGKey(0), (2, 48), 0, cfg.vocab_size)
    chunked = model.compute_loss(params, _with_impl(cfg, "chunked"), ids, chunk_size=32)
    cc = model.compute_loss(params, _with_impl(cfg, "chunked_cc"), ids, chunk_size=32)
    assert jnp.allclose(chunked, cc, rtol=RTOL, atol=1e-5)


def test_model_graph_consumes_the_flag(cfg):
    """``chunked_cc`` selects a graph without the transition-monoid scan."""
    params = model.init_params(jr.PRNGKey(0), cfg)
    ids = jr.randint(jr.PRNGKey(0), (2, 48), 0, cfg.vocab_size)

    def primitives(c) -> set[str]:
        closed = jax.make_jaxpr(
            lambda p: model.compute_loss(p, c, ids, chunk_size=32)
        )(params)
        return _primitive_names(closed)

    chunked = primitives(_with_impl(cfg, "chunked"))
    cc = primitives(_with_impl(cfg, "chunked_cc"))
    # The UT transform solves the unit-triangular system (I + L) and accumulates
    # the cumulative decay; the transition-monoid form never does — that is the
    # observable difference between the two graphs (same convention as the
    # WY/UT test).
    assert "triangular_solve" in cc
    assert "triangular_solve" not in chunked
    assert "cumsum" in cc and "cumsum" not in chunked
    assert chunked != cc


def test_cc_structural_memory_is_the_point_of_the_rewrite(cfg):
    """No ``(H, C, C, dk)`` decay ratio and no ``(C, H, dk, dk)`` monoid.

    The two allocations ADR-047 names — the ``wyut`` decay-ratio tensor and the
    ``chunked`` per-token transition kit — are read straight off the traced
    shapes, so the memory claim is pinned on a CPU, without a GPU profile.
    """
    chunk = 64
    H, C, dk = cfg.num_heads, chunk, cfg.kda_dk
    p = _params(cfg)
    x = _x(cfg, chunk)

    def shapes(fn) -> set[tuple]:
        return _var_shapes(jax.make_jaxpr(fn)())

    cc = shapes(lambda: kda.apply_chunked_cc(p, cfg, x, chunk_size=C))
    wyut = shapes(lambda: kda.apply_wyut(p, cfg, x, chunk_size=C))
    chunked = shapes(lambda: kda.apply_chunked(p, cfg, x, C))

    assert (H, C, C, dk) in wyut, "the wyut decay-ratio tensor should be visible"
    assert (H, C, C, dk) not in cc, "chunked_cc must not materialise it"
    assert (C, H, dk, dk) in chunked, "the chunked transition kit should be visible"
    assert (C, H, dk, dk) not in cc, "chunked_cc must not build per-token dk x dk"
