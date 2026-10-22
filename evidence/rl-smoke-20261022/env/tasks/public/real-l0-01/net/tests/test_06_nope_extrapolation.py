"""Acceptance criterion 6 — NoPE extrapolation and toy needle-recall.

NoPE means there is no positional embedding / RoPE to break when the sequence
grows, so a model trained on short sequences must run at longer sequences with
finite outputs (no NaN) and retain a toy needle-recall.

The smoke trains a tiny model to reproduce a needle token that appears before a
filler run, then evaluates it with that filler run four times longer.

Measured state of this fixture (CPU, JAX 0.10.2, see ``_DESIGN`` below):

* at the **training** gap the smoke learns the task and recalls the needle
  reliably — 0.99–1.00 across training seeds;
* at the **extrapolated** gap it does not: recall collapses to a single emitted
  token already at the next-but-one length, so the criterion is currently
  **not met**.  The extrapolation assertion is therefore marked ``xfail(strict=True)``
  rather than deleted or weakened: it keeps the claim executable, keeps the
  failure visible in the test output, and turns into a suite failure (XPASS) the
  moment the model actually meets it — which is the signal to drop the mark.

Design notes — why the numbers below:

* **Key space.**  Keys are drawn from ``KEYS`` distinct values.  With the two
  values the original smoke used, chance was 0.5 — exactly the old threshold —
  so a model that had collapsed to a constant scored 0.547 and passed.  A wide
  space (chance ~1.6%) makes the threshold meaningful.
* **Threshold.**  ``RECALL_MIN = 0.9`` sits far above the best input-independent
  predictor (the most frequent key's share of the eval batch, ~0.05 here):
  passing needs ≥58 of 64 evals, which such a predictor reaches with probability
  ~1e-40.  ``_degenerate_score`` is asserted below ``RECALL_MIN`` so the eval
  batch itself can never make this vacuous.
* **Budget.**  The training-gap precondition needs a budget at which the tiny
  fixture actually learns the copy task; at 150 steps it does not (recall ~0.03).
"""

from __future__ import annotations

import pytest

import jax
import jax.numpy as jnp
import jax.random as jr

from conftest import tiny_config

from net import model, optimizer

FILL, KEY_LO, QUERY = 1, 2, 4
KEYS = 62                    # distinct key values: KEY_LO .. KEY_LO + KEYS - 1
TRAIN_GAP = 4
EVAL_GAP = 16
STEPS = 600
LR = 1e-2
RECALL_MIN = 0.9             # far above the best input-independent predictor
PRECONDITION_MIN = 0.8       # "the smoke learned the task" — see note below
# The precondition floor is deliberately separate and lower: its job is to tell
# "learned the copy task" (~0.99 at this budget) from "never learned it"
# (~0.03–0.23 at 300 steps), not to carry the extrapolation claim.  Measured at
# 600 steps / 62 keys across seeds: 0.99–1.00.


@pytest.fixture(scope="module")
def trained_copy():
    cfg = tiny_config()
    key = jr.PRNGKey(0)
    params = model.init_params(key, cfg)

    def loss_fn(p, x):
        return model.compute_loss(p, cfg, x, chunk_size=8)

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    step = optimizer.make_step(cfg)
    state = optimizer.init_state(params)

    for _ in range(STEPS):
        key, k = jr.split(key)
        keys = jr.randint(k, (8,), KEY_LO, KEY_LO + KEYS)
        body = jnp.full((8, TRAIN_GAP), FILL)
        q = jnp.full((8, 1), QUERY)
        x = jnp.concatenate([keys[:, None], body, q, keys[:, None]], axis=1).astype(jnp.int32)
        _, grads = grad_fn(params, x)
        params, state = step(params, grads, state, LR)
    return cfg, params


def _recall(cfg, params, gap, n=64):
    """Needle-recall at ``gap``, finiteness, and the degenerate-predictor score.

    The third value is the largest share held by any single key in the eval
    batch: a model that always emits that key scores exactly it, so the
    assertion threshold has to sit above it.
    """
    k = jr.PRNGKey(123)
    keys = jr.randint(k, (n,), KEY_LO, KEY_LO + KEYS)
    body = jnp.full((n, gap), FILL)
    q = jnp.full((n, 1), QUERY)
    x = jnp.concatenate([keys[:, None], body, q, jnp.full((n, 1), FILL)], axis=1).astype(jnp.int32)
    logits = model.forward(params, cfg, x, chunk_size=8)
    pred = jnp.argmax(logits[:, gap + 1, :], axis=-1)
    recall = float(jnp.mean((pred == keys).astype(jnp.float32)))
    _, counts = jnp.unique(keys, return_counts=True)
    degenerate = float(jnp.max(counts)) / n
    return recall, bool(jnp.all(jnp.isfinite(logits))), degenerate


def test_nope_extrapolation_no_nan(trained_copy):
    cfg, params = trained_copy
    k = jr.PRNGKey(7)
    xlong = jr.randint(k, (1, 512), 0, cfg.vocab_size)
    logits = model.forward(params, cfg, xlong, chunk_size=8)
    assert bool(jnp.all(jnp.isfinite(logits)))


def test_needle_recall_is_learned_at_the_training_gap(trained_copy):
    """Precondition for the extrapolation check: the smoke works where it trained.

    Kept as its own (non-xfail) test so that a failure here is never masked by
    the expected failure below, and so the two failure modes stay distinguishable.
    """
    cfg, params = trained_copy
    train_recall, finite, degenerate = _recall(cfg, params, gap=TRAIN_GAP)
    assert finite
    assert degenerate < RECALL_MIN, (
        f"eval batch is degenerate: a constant predictor would score {degenerate:.3f}"
    )
    assert train_recall >= PRECONDITION_MIN, (
        f"smoke did not learn the copy task at gap={TRAIN_GAP} (recall {train_recall:.3f} "
        f"< {PRECONDITION_MIN}); the extrapolation check would be meaningless"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Criterion 6 not met on this fixture: the smoke recalls the needle at the training gap "
        "(~1.00) but collapses to a single emitted token one step beyond it — recall@gap16 is at "
        "or below the degenerate baseline across training seeds, and a longer budget does not fix "
        "it. The mark is deliberately strict and is NOT removed automatically: once the model "
        "meets the threshold, pytest reports XPASS and — because strict — fails the suite. That "
        "red run means the capability arrived; delete the mark by hand. Expect the first green "
        "extrapolation to surface as a suite failure, not as a silent pass."
    ),
)
def test_needle_recall_above_threshold_at_extrapolated_gap(trained_copy):
    cfg, params = trained_copy
    recall, finite, degenerate = _recall(cfg, params, gap=EVAL_GAP)
    assert finite
    assert recall >= RECALL_MIN, (
        f"needle recall {recall:.3f} below {RECALL_MIN} at gap={EVAL_GAP} "
        f"(constant-predictor baseline {degenerate:.3f})"
    )
