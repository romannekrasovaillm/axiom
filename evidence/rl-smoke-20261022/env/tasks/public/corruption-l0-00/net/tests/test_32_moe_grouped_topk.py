"""MFU fix #2 — the routed experts are computed only for the selected top-k.

``net/moe.py:apply`` used to build ``(N, n_routed, expert_inter)`` with an einsum
over *every* routed expert and only then gather the top-k (``take_along_axis``):
the hardware paid for all ``n_routed`` experts while the algorithm needs
``top_k``.  ``tools/mfu_ladder.py:flops_moe_ffn`` reports the two counts as
"executed" and "active"; the bf16 L2 cell measured 36.8-42.1% MFU on the executed
count against 17.2-19.6% on the active one (``evidence/mfu-ladder/
ladder-report-bf16.json``) — a 6x waste at ``n_routed=12``, ``top_k=2``.

The refactor gathers the selected tokens per expert (a token appears once per
selected expert), runs the expert MLP on the ragged groups and scatters back.
This file pins what that change must not break:

* (а) numeric equivalence with the dense path on a fixed seed — a **tolerance**,
      not bit-equality: the grouped contraction is a different XLA lowering;
* (б) gate-off (``AXIOM_COMPUTE_DTYPE=fp32``) keeps the declared behaviour: the
      routing decision and the QB term are *exact* (they read router statistics
      only) and the forward moves only within the tolerance;
* (в) the unselected experts receive exactly zero gradient in both paths;
* and the sparsity itself: a sparse ``ragged_dot`` is **bit-identical** to a
  dense one over all experts, so gathering the top-k costs nothing numerically.

Tolerances (CPU, jax 0.10.2, ``jax_default_matmul_precision=highest`` — the
suite's pin, ``net/tests/conftest.py``):

* sparsity is exact — asserted with ``array_equal``, no tolerance;
* against the pre-refactor **einsum** lowering the gap is <= 1.2e-6 relative at
  the L2 width (hidden 1536 / latent 768 / ei 384) and <= 5e-8 on the suite's
  small config — at most ~10 ulp of fp32 (eps 1.19e-7).  ``RTOL`` is set to
  1e-5, an ~8x margin over the worst measurement; it is a *lowering* budget, not
  a slack fudge: the two spellings compute the same contraction.

bf16 is exercised on the routed step directly (``_routed_experts_grouped``),
because the *full* ``apply`` under the bf16 gate hits the documented CPU
limitation of the shared path's einsum spelling (``net/compute_dtype.py``,
"Unsupported element type for DotThunk::Execute: BF16 x BF16 = F32") — the bf16
campaign runs on the GPU stand.  ``ragged_dot`` itself runs in bf16 on CPU.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from net import compute_dtype, mlp, moe
from net.norm import rms_norm

#: Relative tolerance of the routed mix against the pre-refactor einsum lowering
#: (fp32).  See the module docstring for the measurement behind it.
RTOL = 1e-5


# --------------------------------------------------------------------------- #
# Test-local references — the *dense* (all-expert) contraction the refactor
# replaces.  Kept independent of the implementation on purpose.
# --------------------------------------------------------------------------- #


def _rdot(a: jnp.ndarray, b: jnp.ndarray, group_sizes: jnp.ndarray) -> jnp.ndarray:
    """``ragged_dot`` with the gate's boundary cast (mirrors the module helper)."""
    if compute_dtype.is_bf16():
        return jax.lax.ragged_dot(
            compute_dtype.cast_in(a), compute_dtype.cast_in(b), group_sizes,
            preferred_element_type=jnp.float32,
        )
    return jax.lax.ragged_dot(a, b, group_sizes)


def _dense_over_all_experts(params, cfg, z, topk, sel_p):
    """The pre-refactor routed mix, spelled with the same einsum/batched lowering."""
    hg = compute_dtype.gemm_einsum("nl,elj->nej", z, params.expert_g)
    hu = compute_dtype.gemm_einsum("nl,elj->nej", z, params.expert_u)
    a = mlp.siti_glu((hg, hu), cfg.siti_beta_gate, cfg.siti_beta_up)
    if compute_dtype.is_bf16():
        e_out = compute_dtype.gemm_batched(a.transpose(1, 0, 2), params.expert_d).transpose(1, 0, 2)
    else:
        e_out = jnp.einsum("nej,ejl->nel", a, params.expert_d)
    return compute_dtype.gemm_einsum(
        "nk,nkl->nl", sel_p, jnp.take_along_axis(e_out, topk[..., None], axis=1)
    )


def _dense_ragged_over_all_experts(params, cfg, z, topk, sel_p):
    """Every expert, on a *ragged* layout — the sparse path's exact-math twin.

    Runs in both gate modes on CPU, so it is the reference for "the sparsity
    change is numerically free": same primitive family, all groups dense.
    """
    n_routed, n_tokens = cfg.moe_num_routed, z.shape[0]
    sizes = jnp.full((n_routed,), n_tokens, dtype=jnp.int32)
    z_rep = jnp.broadcast_to(z, (n_routed, n_tokens, z.shape[-1])).reshape(-1, z.shape[-1])
    a = mlp.siti_glu(
        (_rdot(z_rep, params.expert_g, sizes), _rdot(z_rep, params.expert_u, sizes)),
        cfg.siti_beta_gate,
        cfg.siti_beta_up,
    )
    e_out = _rdot(a, params.expert_d, sizes).reshape(n_routed, n_tokens, -1).transpose(1, 0, 2)
    return compute_dtype.gemm_einsum(
        "nk,nkl->nl", sel_p, jnp.take_along_axis(e_out, topk[..., None], axis=1)
    )


def _routed_operands(params, cfg, x):
    """``(z, topk, sel_p)`` — the routed sub-problem, shared by the cases below.

    ``x`` is flattened exactly as ``moe.apply`` does before ``W_down``.
    """
    xf = x.reshape(-1, x.shape[-1])
    z = compute_dtype.gemm(xf, params.W_down)
    _, topk, sel_p, _ = moe._dispatch(params, cfg, x)
    return z, topk, sel_p


def _max_rel(a: jnp.ndarray, ref: jnp.ndarray) -> float:
    return float(jnp.max(jnp.abs(a - ref)) / (jnp.max(jnp.abs(ref)) + 1e-30))


# --------------------------------------------------------------------------- #
# The feature: only the selected experts are computed
# --------------------------------------------------------------------------- #


def test_routed_experts_are_computed_only_for_selected_tokens(cfg, monkeypatch):
    """The routed MLP sees ``N * top_k`` rows, not ``N * n_routed``.

    The routed activation is materialised once, for the gathered selected
    tokens; a dense all-expert path would show ``(N, n_routed, ei)`` here.  The
    shared path is the contrast case: it stays full-width ``(N, ns, si)``.
    """
    params = moe.init_moe(jr.PRNGKey(cfg.routing_seed), cfg)
    x = jr.normal(jr.PRNGKey(1), (4, 16, cfg.hidden))
    n = 4 * 16

    seen: list[tuple[int, ...]] = []
    real = mlp.siti_glu

    def spy(vals, beta_gate, beta_up):
        seen.append(tuple(vals[0].shape))
        return real(vals, beta_gate, beta_up)

    monkeypatch.setattr(mlp, "siti_glu", spy)
    moe.apply(params, cfg, x)

    assert (n * cfg.moe_top_k, cfg.moe_expert_intermediate) in seen, (
        f"routed experts did not run on the gathered top-k rows: {seen}"
    )
    assert (n, cfg.moe_num_routed, cfg.moe_expert_intermediate) not in seen, (
        f"routed experts were materialised for every expert: {seen}"
    )
    assert (n, cfg.moe_num_shared, cfg.moe_shared_intermediate) in seen, (
        f"shared path left the full-width layout: {seen}"
    )


@pytest.mark.parametrize("gate", ["fp32", "bf16"])
def test_sparsity_does_not_move_the_mix(cfg, monkeypatch, gate):
    """A sparse ``ragged_dot`` equals a dense one over all experts, bit for bit.

    This is the numerical price of the refactor *by itself*: zero.  It holds in
    both gate modes and is what lets the equivalence tolerance below be
    attributed entirely to the einsum -> ragged lowering, not to the gathering.
    """
    monkeypatch.setenv(compute_dtype.MODE_ENV, gate)
    params = moe.init_moe(jr.PRNGKey(cfg.routing_seed), cfg)
    x = jr.normal(jr.PRNGKey(2), (8, 16, cfg.hidden))
    z, topk, sel_p = _routed_operands(params, cfg, x)

    sparse = moe._routed_experts_grouped(params, cfg, z, topk, sel_p)
    dense = _dense_ragged_over_all_experts(params, cfg, z, topk, sel_p)

    assert jnp.array_equal(sparse, dense), _max_rel(sparse, dense)


# --------------------------------------------------------------------------- #
# (а) numeric equivalence with the pre-refactor path, on a fixed seed
# --------------------------------------------------------------------------- #


def test_grouped_routed_matches_the_dense_routed_path(cfg):
    """Fixed seed: the grouped mix tracks the einsum (all-expert) mix within ``RTOL``."""
    params = moe.init_moe(jr.PRNGKey(cfg.routing_seed), cfg)
    x = jr.normal(jr.PRNGKey(3), (8, 16, cfg.hidden))
    z, topk, sel_p = _routed_operands(params, cfg, x)

    got = moe._routed_experts_grouped(params, cfg, z, topk, sel_p)
    ref = _dense_over_all_experts(params, cfg, z, topk, sel_p)

    assert _max_rel(got, ref) <= RTOL


def test_layer_output_matches_the_dense_path(cfg):
    """The whole layer, not just the routed mix: same tolerance, same seed."""
    params = moe.init_moe(jr.PRNGKey(cfg.routing_seed), cfg)
    x = jr.normal(jr.PRNGKey(4), (4, 16, cfg.hidden))
    got = moe.apply(params, cfg, x).reshape(-1, cfg.hidden)

    # Reference: the pre-refactor layer, with the routed step kept dense.
    xf = x.reshape(-1, cfg.hidden)
    _, topk, sel_p, _ = moe._dispatch(params, cfg, x)
    z = compute_dtype.gemm(xf, params.W_down)
    u = _dense_over_all_experts(params, cfg, z, topk, sel_p)

    sg = compute_dtype.gemm_einsum("nh,shj->nsj", xf, params.shared_g)
    su = compute_dtype.gemm_einsum("nh,shj->nsj", xf, params.shared_u)
    sa = mlp.siti_glu((sg, su), cfg.siti_beta_gate, cfg.siti_beta_up)
    shared = jnp.sum(compute_dtype.gemm_einsum("nsj,sjh->nsh", sa, params.shared_d), axis=1)
    ref = shared + compute_dtype.gemm(rms_norm(u, params.norm), params.W_up)

    assert got.shape == ref.shape
    assert _max_rel(got, ref) <= RTOL


# --------------------------------------------------------------------------- #
# (б) gate-off: the declared behaviour is unchanged
# --------------------------------------------------------------------------- #


def test_gate_off_keeps_the_routing_decision_exact(cfg):
    """The router is untouched by the refactor — same seed, same experts."""
    params = moe.init_moe(jr.PRNGKey(cfg.routing_seed), cfg)
    x = jr.normal(jr.PRNGKey(5), (4, 16, cfg.hidden))

    _, topk, sel_p, p_full = moe._dispatch(params, cfg, x)
    # The dispatch is a pure function of the scores; the declared surface must
    # reproduce it exactly (the refactor touches only the expert application).
    assert jnp.array_equal(topk, moe.routing_indices(params, cfg, x).reshape(-1, cfg.moe_top_k))
    assert jnp.allclose(jnp.sum(p_full, axis=-1), 1.0, atol=1e-6)
    frac = moe.load_fraction(params, cfg, x)
    assert float(jnp.sum(frac)) == pytest.approx(cfg.moe_top_k, abs=1e-5)
    assert jnp.all(sel_p > 0)


def test_dense_response_is_finite_and_shaped_like_the_model(cfg):
    """Guard: the grouped path still returns ``(..., hidden)`` with finite values."""
    params = moe.init_moe(jr.PRNGKey(cfg.routing_seed), cfg)
    x = jr.normal(jr.PRNGKey(6), (2, 8, cfg.hidden))
    out = moe.apply(params, cfg, x)
    assert out.shape == x.shape
    assert bool(jnp.all(jnp.isfinite(out)))


# --------------------------------------------------------------------------- #
# (в) unselected experts get no gradient — in both paths
# --------------------------------------------------------------------------- #


def _unselected_experts(params, cfg, x) -> jnp.ndarray:
    return jnp.nonzero(moe.load_fraction(params, cfg, x) == 0.0, size=cfg.moe_num_routed)[0]


def test_unselected_experts_get_zero_gradient_in_the_grouped_path(cfg):
    params = moe.init_moe(jr.PRNGKey(cfg.routing_seed), cfg)
    x = jr.normal(jr.PRNGKey(7), (1, cfg.hidden))  # one token -> top_k of n_routed experts fire

    unselected = _unselected_experts(params, cfg, x)
    assert unselected.size > 0, "fixture has no unselected expert to check"

    def loss_d(expert_d):
        return jnp.sum(moe.apply(params._replace(expert_d=expert_d), cfg, x) ** 2)

    def loss_u(expert_u):
        return jnp.sum(moe.apply(params._replace(expert_u=expert_u), cfg, x) ** 2)

    grad_d = jax.grad(loss_d)(params.expert_d)
    grad_u = jax.grad(loss_u)(params.expert_u)

    assert jnp.all(grad_d[unselected] == 0.0), "unselected expert_d received gradient"
    assert jnp.all(grad_u[unselected] == 0.0), "unselected expert_u received gradient"
    # And the selected ones *do* move — otherwise the check above is vacuous.
    selected = jnp.nonzero(moe.load_fraction(params, cfg, x) > 0.0, size=cfg.moe_num_routed)[0]
    assert jnp.any(grad_d[selected] != 0.0)


def test_unselected_experts_get_zero_gradient_in_the_dense_path(cfg):
    """The invariant holds in the pre-refactor path too (it is a property of top-k)."""
    params = moe.init_moe(jr.PRNGKey(cfg.routing_seed), cfg)
    x = jr.normal(jr.PRNGKey(7), (1, cfg.hidden))
    unselected = _unselected_experts(params, cfg, x)
    assert unselected.size > 0

    def loss_dense(expert_d):
        p = params._replace(expert_d=expert_d)
        xf = x.reshape(-1, cfg.hidden)
        _, topk, sel_p, _ = moe._dispatch(p, cfg, x)
        z = compute_dtype.gemm(xf, p.W_down)
        u = _dense_over_all_experts(p, cfg, z, topk, sel_p)
        return jnp.sum(u**2)

    grad_d = jax.grad(loss_dense)(params.expert_d)
    assert jnp.all(grad_d[unselected] == 0.0)


# --------------------------------------------------------------------------- #
# The load-balancing term reads router statistics only — it must not move
# --------------------------------------------------------------------------- #


def test_qb_term_is_unchanged_by_the_grouping(cfg):
    """``apply(want_qb=True)`` returns the same QB value as a fresh ``qb_aux_loss``.

    The QB estimator is built from ``topk`` and ``p_full`` — scores and dispatch,
    never an expert output (``net/moe.py`` module docstring, Eq. 14).  Gathering
    the top-k experts therefore cannot perturb it; this asserts the reuse path
    (``routed=(topk, p_full)``) still agrees with the standalone computation.
    """
    params = moe.init_moe(jr.PRNGKey(cfg.routing_seed), cfg)
    x = jr.normal(jr.PRNGKey(8), (4, 16, cfg.hidden))

    _, qb_from_apply = moe.apply(params, cfg, x, want_qb=True)
    qb_standalone = moe.qb_aux_loss(params, cfg, x)

    assert bool(jnp.array_equal(qb_from_apply, qb_standalone))
    # And the value itself is a finite, positive scalar (k is the balanced floor).
    assert bool(jnp.isfinite(qb_standalone)) and float(qb_standalone) > 0.0
