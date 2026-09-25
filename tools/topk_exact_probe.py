"""A/B cost profile of the exact tiled top-k vs ``jax.lax.top_k`` (criterion 16).

The acceptance criterion (A4 No. 16, ``net/tests/test_15_cost_8k_64k.py``)
measures the sparse MLA stack against the dense oracle.  This probe isolates
what the *selection* costs inside that stack, by running the very same stack
once per leg — ``attn_topk_exact = true`` (``net/topk_exact.py``), the legacy
``jax.lax.top_k`` leg and the pinned pre-ADR-012 baseline — and reporting each
leg's ratio to the dense oracle stack measured in the same round.

The two selection legs pick the *same records* (equivalence:
``net/tests/test_18_exact_topk.py``), so a difference between them is the
selection algorithm's own cost, not a change of what is computed.

Method: the **ADR-015 procedure** (``net/tests/cost_method.py``) — clocks put
under control (hard ``nvidia-smi -lgc`` pin or the user-level PowerMizer
fallback, actual clocks printed), both legs warmed up to a clock plateau,
alternating leg order per round with the first round dropped, the verdict on the
median with the spread always reported, and "не определён" instead of a verdict
when the spread exceeds 5% of the threshold.  This replaces the probe's previous
ad-hoc alternation: the numbers below are only comparable to earlier ones when
read as (median, min…max), never as single shots — this machine's GPU ramps its
clocks and drifts, which is exactly what ADR-015 was written about.

Usage::

    export LD_LIBRARY_PATH=$(ls -d ~/venv-axiom/lib/python3.11/site-packages/nvidia/*/lib | tr '\n' ':')
    python tools/topk_exact_probe.py --length 8192 --rounds 10

``--length 65536`` measures the long-context leg (the dense oracle is timed with
the memory-bounded probe, exactly as in ``test_15``).
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "net" / "tests"))

import jax  # noqa: E402
import jax.random as jr  # noqa: E402

from conftest import small_config  # noqa: E402

import cost_method  # noqa: E402
from cost_method import (  # noqa: E402
    THRESHOLD_8K,
    THRESHOLD_STRICTLY_BETTER,
    ClockControl,
    measure,
    verdict,
)

from net import mla  # noqa: E402

STACK = 6


def cost_config():
    """The criterion's reduced-width config (Q-width : indexer-width = 12:1)."""
    return dataclasses.replace(
        small_config(),
        hidden=192, num_heads=6, head_dim=32,
        kda_dk=32, kda_dv=32, kda_decay_rank=32,
        mla_head_dim=32, mla_latent_dim=64,
        mla_index_heads=1, mla_index_dim=16,
        swa_window=128, mla_top_k=512, attn_dense_reference=False,
    )


def _stack(params_list, cfg, x, modes):
    pool = None
    out = None
    for p, mode in zip(params_list, modes):
        out, pool = mla.apply_with_pool(p, cfg, x, pool=pool, mode=mode)
    return out


def _dense_probe_stack(params_list, cfg, x):
    """The 64K dense baseline of test_15 (the oracle cannot be allocated there)."""
    from test_15_cost_8k_64k import _dense_probe

    out = None
    for p in params_list:
        out = _dense_probe(p, cfg, x)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--length", type=int, default=8192)
    ap.add_argument("--rounds", type=int, default=cost_method.DEFAULT_ROUNDS)
    ap.add_argument("--inner", type=int, default=cost_method.DEFAULT_INNER_REPEATS)
    args = ap.parse_args()

    T = args.length
    base = cost_config()
    after = dataclasses.replace(
        base, mla_pool_block=64, mla_pool_size=16,
        mla_layer_modes=("full",) + ("reindex",) * (STACK - 1),
    )
    exact = dataclasses.replace(after, attn_topk_exact=True)
    legacy = dataclasses.replace(after, attn_topk_exact=False)
    dense_cfg = dataclasses.replace(base, attn_dense_reference=True)
    params = [mla.init_mla(k, base) for k in jr.split(jr.PRNGKey(0), STACK)]
    x = jr.normal(jr.PRNGKey(1), (1, T, base.hidden))

    if T <= 8192:
        dense = jax.jit(lambda x: _stack(params, dense_cfg, x, ("full",) * STACK))
        dense_label = "dense oracle stack"
    else:
        dense = jax.jit(lambda x: _dense_probe_stack(params, base, x))
        dense_label = "dense probe stack"
    exact_fn = jax.jit(lambda x: _stack(params, exact, x, exact.mla_layer_modes))
    legacy_fn = jax.jit(lambda x: _stack(params, legacy, x, legacy.mla_layer_modes))
    no_pool = dataclasses.replace(base, mla_pool_size=0, mla_layer_modes=())
    before_fn = jax.jit(lambda x: _stack(params, no_pool, x, ("full",) * STACK))
    legs = {
        "dense": lambda: dense(x),
        "exact": lambda: exact_fn(x),
        "legacy": lambda: legacy_fn(x),
        "before": lambda: before_fn(x),
    }

    label = f"topk A/B | T={T} | stack x{STACK} ({dense_label})"
    threshold = THRESHOLD_8K if T <= 8192 else THRESHOLD_STRICTLY_BETTER
    print(f"device: {jax.devices()}  T={T}  top_k={base.mla_top_k}  window={base.swa_window}  "
          f"pool={after.mla_pool_size}x{after.mla_pool_block}  stack=x{STACK}")
    with ClockControl() as clock:
        cost_method.print_clock(clock, label)
        m = measure(legs, label=label, clock=clock, rounds=args.rounds,
                    inner_repeats=args.inner)
        cost_method.print_measurement(m, ref="dense", threshold=threshold)
        for name in ("exact", "legacy", "before"):
            cost_method.print_verdict(
                verdict(m.stats(name, "dense"), threshold, clock=clock, measurement=m,
                        strictly_better=T > 8192),
                f"{label} | {name}/dense",
            )


if __name__ == "__main__":
    main()
