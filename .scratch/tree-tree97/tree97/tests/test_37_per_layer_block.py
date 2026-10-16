"""``per_layer_block`` — the block-scoped grad-checkpoint that keeps the AttnRes
carry out of the per-sub-layer saves (ADR-054 Am.18, milestone-8).

Why the policy exists.  ``per_layer`` wraps every backbone layer *transition*
in ``jax.checkpoint``; among a boundary's inputs is the whole ``sources``
carry (``num_layers + 1`` slots of ``(B, T, hidden)``, built for AttnRes), and
the differentiated group-scan collects those per-boundary saves into a stacked
array *per sub-layer position*: 4 stacked copies of
``(scanned_units, L+1, B, T, hidden)`` at the pinned ``[K,K,K,M]`` layout —
measured in the campaign HLO dump as ~47 GiB of the ~109 GiB B=4 floor, the
memory wall the amendment exists to remove (``GRAD_CKPT_POLICIES`` in
``net/model.py`` carries the full story).

What the policy does.  ``per_layer_block`` narrows the boundary to the
KDA/MLA+MoE *block* (bounded (B, H, T, T) score transients) and, in the
scanned backbone, additionally covers the AttnRes read — but over a view
rebuilt inside the boundary from the **shared unit-entry carry** plus the
unit's deltas.  The big input of all four sub-layer boundaries is then one
loop-invariant buffer, so the scan collects a single stacked copy.  Reading
the carry *outside* the boundary instead does not work (measured: the masked
read's softmax residuals are saved per sub-layer position — 12 stacked
copies, worse than ``per_layer``'s 4).

What this module pins:

* declaration — the value is accepted by ``net.config.validate_config`` and is
  one of ``net.model.GRAD_CKPT_POLICIES`` (two lists that must agree; the
  model imports the config, so the import cannot go the other way);
* mechanism — one ``remat`` boundary per layer under both policies: the
  *scopes* differ, not the count; no double wrapping anywhere;
* parity — the loss is unchanged (measured: bitwise equal to ``none`` and
  ``per_layer`` on the CPU configs; the documented tolerance is 1e-6) and the
  gradients agree within the recompute reordering (measured on the 12-layer
  scan reproduction at T=16: max |Δ| 8.3e-07 against ``per_layer``,
  6.4e-06 against ``none``);
* graph pin — in the backward (``jax.grad``) jaxpr of the scanned backbone the
  number of arrays with the stacked-carry shape
  ``(scanned_units, L+1, B, T, hidden)`` is 4 under ``per_layer`` (one per
  sub-layer position) and **1** under ``per_layer_block``; the compiled-HLO
  loop-state census of the same reproduction agrees (12/4/1 for
  none/per_layer/per_layer_block);
* gate-off — ``none`` and ``per_layer`` reproduce the pre-delta composition
  bitwise: ``none`` equals the default-policy call, and both equal the
  hand-written layer loop with the whole transition (not the block) wrapped.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from conftest import small_config, tiny_config

from net import model
from net.config import load_config, validate_config

#: The case's declarative config (the pinned ``l3-full`` value lives there).
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"

#: Sequence length of the small oracles — the suite runs on CPU, so short.
SEQ = 8

#: Layer batch of the oracles (the tests scale the *structure*, not the math).
BATCH = 2

#: A multi-unit ``[K,K,K,M]`` layout: 3 units (12 layers), so the group-scan
#: actually runs and the stacked-carry effect exists (one unit never scans).
_SCAN_KW = dict(num_layers=12, num_kda_layers=9, num_mla_layers=3)

#: Caches — the eager CPU executions are the expensive part of this module.
_LOSSES: dict[tuple[str, str], float] = {}
_GRADS: dict[tuple[str, str], tuple] = {}


def _scan_cfg(**overrides):
    """The scanned oracle's config: tiny geometry, 3 ``[K,K,K,M]`` units."""
    return tiny_config(**{**_SCAN_KW, **overrides})


def _ids(cfg, *, seed=1, T=SEQ):
    return jr.randint(jr.PRNGKey(seed), (BATCH, T), 0, cfg.vocab_size)


def _params(cfg, *, seed=0):
    return model.init_params(jr.PRNGKey(seed), cfg)


def _loss(tag: str, cfg, policy: str) -> float:
    """Loss of ``cfg`` under ``policy`` (cached per (tag, policy))."""
    key = (tag, policy)
    if key not in _LOSSES:
        _LOSSES[key] = float(
            model.compute_loss(
                _params(cfg), cfg, _ids(cfg), chunk_size=SEQ,
                grad_ckpt_policy=policy,
            )
        )
    return _LOSSES[key]


def _grad(tag: str, cfg, policy: str):
    """Gradient of the cached loss under ``policy`` (cached per (tag, policy))."""
    key = (tag, policy)
    if key not in _GRADS:
        params = _params(cfg)
        ids = _ids(cfg)
        _GRADS[key] = jax.grad(
            lambda p: model.compute_loss(
                p, cfg, ids, chunk_size=SEQ, grad_ckpt_policy=policy
            )
        )(params)
    return _GRADS[key]


def _tree_allclose(a, b, *, rtol=1e-3, atol=1e-4) -> bool:
    leaves_a = jax.tree_util.tree_leaves(a)
    leaves_b = jax.tree_util.tree_leaves(b)
    assert len(leaves_a) == len(leaves_b)
    return all(
        bool(jnp.allclose(x, y, rtol=rtol, atol=atol, equal_nan=True))
        for x, y in zip(leaves_a, leaves_b)
    )


def _remat_count(cfg, policy: str) -> int:
    """Number of ``remat`` boundaries in the traced loss, scan bodies included."""
    closed = jax.make_jaxpr(
        lambda p: model.compute_loss(
            p, cfg, _ids(cfg), chunk_size=SEQ, grad_ckpt_policy=policy
        )
    )(_params(cfg))
    count = 0
    for eqn in _all_eqns(closed.jaxpr):
        # JAX names the checkpoint primitive ``remat`` with a version suffix
        # (``remat2`` on jax 0.11.x); match the family, not one exact name.
        if str(eqn.primitive.name).startswith("remat"):
            count += 1
    return count


def _all_eqns(jaxpr):
    """Recursive walk over a jaxpr and every nested jaxpr (scan bodies)."""
    for eqn in jaxpr.eqns:
        yield eqn
        for leaf in jax.tree_util.tree_leaves(eqn.params):
            if hasattr(leaf, "jaxpr"):
                yield from _all_eqns(leaf.jaxpr)


def _stacked_carry_copies(cfg, policy: str) -> int:
    """Arrays of the stacked-carry shape in the *backward* jaxpr.

    The carry is ``(num_layers + 1, B, T, hidden)``; the differentiated
    ``lax.scan`` stacks a save per scanned unit, so one copy has shape
    ``(scanned_units, num_layers + 1, B, T, hidden)``.  Counting these exact
    shapes in ``jax.grad``'s jaxpr is the jaxpr-level form of the campaign's
    HLO loop-state census (both read 12/4/1 for none/per_layer/per_layer_block
    on the 12-layer reproduction).
    """
    n_units = cfg.num_layers // 4
    first_unit, _mode = model._group_scan_units(cfg)
    assert first_unit is not None, "the oracle must scan (multi-unit layout)"
    scanned_units = n_units - first_unit
    shape = f"f32[{scanned_units},{cfg.num_layers + 1},{BATCH},{SEQ},{cfg.hidden}]"
    closed = jax.make_jaxpr(
        jax.grad(
            lambda p: model.compute_loss(
                p, cfg, _ids(cfg), chunk_size=SEQ, grad_ckpt_policy=policy
            )
        )
    )(_params(cfg))
    return str(closed).count(shape)


def _reference_logits(params, cfg, ids, *, boundary: str):
    """The layer loop written out here: the pre-delta composition.

    ``boundary="none"`` runs the plain loop; ``boundary="layer"`` wraps each
    whole transition in ``jax.checkpoint`` — exactly what ``per_layer`` does
    (the boundary is around the *transition*, not the block).  Written out so
    the gate-off pin does not depend on :func:`net.model.forward` calling
    itself.
    """
    emb = params.embedding[ids]
    h = emb
    deltas: list[jnp.ndarray] = []
    for i, block in enumerate(params.layers):
        kind = model.layer_kind(cfg, i)
        is_kda = kind == "kda"
        mode = (
            "full"
            if is_kda
            else model.mla_mod.layer_mode(cfg, model._mla_ordinal_at(cfg, i))
        )
        aw = params.attnres.w[i]

        def step(block=block, aw=aw, _i=i, _is_kda=is_kda, _mode=mode):
            return model._layer_transition(
                block, aw, h, None, tuple(deltas), emb,
                cfg=cfg, chunk_size=SEQ, collect_qb=False, use_attnres=True,
                index=_i, is_kda=_is_kda, mode=_mode,
            )

        if boundary == "layer":
            step = jax.checkpoint(step)
        h, delta, _qb, _pool = step()
        deltas.append(delta)
    h = model.rms_norm(h, params.norm_final)
    return model.compute_dtype.gemm(h, params.embedding.T)


# ---------------------------------------------------------------------------
# declaration: both lists know the value; the pinned config stays per_layer
# ---------------------------------------------------------------------------


def test_config_validator_accepts_every_declared_policy():
    """``validate_config`` and ``GRAD_CKPT_POLICIES`` agree value by value.

    The two lists live in different modules (``net/model.py`` imports
    ``net/config.py``, so the import cannot go the other way); this test is
    what keeps them equal — a policy registered in the model but refused by
    the validator (or vice versa) fails here.
    """
    for policy in model.GRAD_CKPT_POLICIES:
        validate_config(
            dataclasses.replace(small_config(), grad_ckpt_policy=policy)
        )
    assert set(model.GRAD_CKPT_POLICIES) == {
        "none", "per_layer", "per_layer_block"
    }


def test_unknown_policy_is_still_rejected():
    cfg = dataclasses.replace(small_config(), grad_ckpt_policy="sometimes")
    with pytest.raises(AssertionError):
        validate_config(cfg)


def test_declared_pretrain_config_is_still_per_layer():
    """The pinned ``l3-full`` policy is untouched by this delta (gate-off)."""
    cfg = load_config(CONFIG_PATH)
    assert cfg.grad_ckpt_policy == "per_layer"
    validate_config(cfg)


# ---------------------------------------------------------------------------
# mechanism: one boundary per layer under both scopes
# ---------------------------------------------------------------------------


def test_unrolled_remat_count_matches_the_layer_count(cfg):
    assert _remat_count(cfg, "none") == 0
    assert _remat_count(cfg, "per_layer") == cfg.num_layers
    assert _remat_count(cfg, "per_layer_block") == cfg.num_layers


def test_scanned_boundary_count_is_the_same_for_both_scopes():
    """The scopes differ, not the number of boundaries (prefix + body)."""
    cfg = _scan_cfg()
    assert model._group_scan_units(cfg) is not None
    per_layer = _remat_count(cfg, "per_layer")
    block = _remat_count(cfg, "per_layer_block")
    assert per_layer > 0
    assert block == per_layer


# ---------------------------------------------------------------------------
# parity: remat changes where the backward recomputes, not the numbers
# ---------------------------------------------------------------------------


def test_loss_parity_unrolled():
    cfg = small_config()
    none = _loss("small", cfg, "none")
    per_layer = _loss("small", cfg, "per_layer")
    block = _loss("small", cfg, "per_layer_block")
    # measured: all three are bitwise equal on the CPU oracle; the documented
    # tolerance is 1e-6 (recompute may reorder floating point on other XLA
    # backends).
    for value, reference in ((per_layer, none), (block, none), (block, per_layer)):
        assert abs(value - reference) <= 1e-6
    assert none == per_layer  # forward-transparent boundary: bitwise


def test_loss_parity_scanned():
    cfg = _scan_cfg()
    none = _loss("scan", cfg, "none")
    per_layer = _loss("scan", cfg, "per_layer")
    block = _loss("scan", cfg, "per_layer_block")
    for value, reference in ((per_layer, none), (block, none), (block, per_layer)):
        assert abs(value - reference) <= 1e-6
    assert none == per_layer  # same forward ops: bitwise


def test_grad_parity_scanned():
    cfg = _scan_cfg()
    block = _grad("scan", cfg, "per_layer_block")
    per_layer = _grad("scan", cfg, "per_layer")
    none = _grad("scan", cfg, "none")
    # measured on the 12-layer reproduction at T=16: max |Δ| 8.3e-07 against
    # per_layer, 6.4e-06 against none (recompute reordering only).
    assert _tree_allclose(block, per_layer)
    assert _tree_allclose(block, none)


def test_disabled_attnres_path_traces_and_computes():
    """``use_attnres=False`` on the rebuild policy: no carry, no rebuild."""
    cfg = _scan_cfg(grad_ckpt_policy="per_layer_block")
    params = _params(cfg)
    ids = _ids(cfg)
    value = model.compute_loss(
        params, cfg, ids, chunk_size=SEQ, use_attnres=False
    )
    assert bool(jnp.isfinite(value))


# ---------------------------------------------------------------------------
# graph pin: one shared stacked copy instead of one per sub-layer position
# ---------------------------------------------------------------------------


def test_no_per_sub_layer_stacked_carry_copies():
    """The measured property the policy exists for (see module docstring)."""
    cfg = _scan_cfg()
    # per_layer: one stacked save per sub-layer position of [K,K,K,M] = 4.
    assert _stacked_carry_copies(cfg, "per_layer") == 4
    # per_layer_block: the shared unit-entry save = exactly one copy.
    assert _stacked_carry_copies(cfg, "per_layer_block") == 1
    # none: every masked read keeps its input and softmax residuals as the
    # transpose's residuals — 3 versions per sub-layer position = 12 (this is
    # also what the rejected "read the carry outside the boundary" variant
    # measured; the rebuild is what collapses it to one).
    assert _stacked_carry_copies(cfg, "none") == 12


# ---------------------------------------------------------------------------
# gate-off: none/per_layer are the pre-delta composition, bitwise
# ---------------------------------------------------------------------------


def test_none_equals_default_policy_bitwise(cfg):
    params = _params(cfg)
    ids = _ids(cfg)
    default = model.compute_loss(params, cfg, ids, chunk_size=SEQ)
    explicit = model.compute_loss(
        params, cfg, ids, chunk_size=SEQ, grad_ckpt_policy="none"
    )
    assert jnp.array_equal(default, explicit)


@pytest.mark.parametrize("boundary", ["none", "layer"])
def test_gate_off_modes_are_bitwise_the_pre_delta_composition(boundary):
    cfg = small_config()
    params = _params(cfg)
    ids = _ids(cfg)
    policy = "none" if boundary == "none" else "per_layer"
    traced = model.forward(
        params, cfg, ids, chunk_size=SEQ, grad_ckpt_policy=policy
    )
    reference = _reference_logits(params, cfg, ids, boundary=boundary)
    assert jnp.array_equal(traced, reference)
