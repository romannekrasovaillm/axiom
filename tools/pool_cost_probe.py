"""Per-mode cost profile of the sparse MLA path (ADR-012 follow-up).

The acceptance criterion (A4 No. 16, ``net/tests/test_15_cost_8k_64k.py``) says
*whether* the sparse path is within 5% of the dense oracle at 8K.  This probe
says *where* the remaining time goes, so the next delta can aim at the right
place: one layer is timed per selection mode (``full`` = prefix scan + pool
build, ``reindex`` = in-pool scoring, ``reuse`` = no scoring), with and without
the SWA window, against the dense oracle layer.

The probe is diagnostic: it reports every leg's ratio with its spread and does
not turn them into a criterion verdict — the gate above is what decides.

Method: the **ADR-015 procedure** (``net/tests/cost_method.py``) — clocks put
under control before the measurement (hard ``nvidia-smi -lgc`` pin, or the
user-level PowerMizer fallback, with the actual clocks printed), all legs warmed
up to a clock plateau, alternating leg order per round with the first round
dropped, each round's per-leg value the minimum over ``--inner`` calls, and the
report carrying the median together with the spread (min…max).  Device is
whatever JAX sees — set ``LD_LIBRARY_PATH`` to the venv's ``nvidia/*/lib``
directories for a GPU run (ADR-010), otherwise this degrades to a CPU profile.

Usage::

    export LD_LIBRARY_PATH=$(ls -d ~/venv-axiom/lib/python3.11/site-packages/nvidia/*/lib | tr '\n' ':')
    python tools/pool_cost_probe.py [--length 8192] [--rounds 10] [--inner 4] [--assembly fused|unfused]

The ``--assembly`` switch selects between the fused and the verbatim unfused
union assembly (``attn_fused_assembly``): same selection, same arithmetic, so a
difference between the two profiles is the fuse's own contribution.
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
from cost_method import THRESHOLD_8K, ClockControl, measure  # noqa: E402

from net import mla  # noqa: E402


def cost_config():
    """The criterion's reduced-width config (Q-width : indexer-width = 12:1)."""
    base = small_config()
    return dataclasses.replace(
        base,
        hidden=192, num_heads=6, head_dim=32,
        kda_dk=32, kda_dv=32, kda_decay_rank=32,
        mla_head_dim=32, mla_latent_dim=64,
        mla_index_heads=1, mla_index_dim=16,
        swa_window=128, mla_top_k=512, attn_dense_reference=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--length", type=int, default=8192)
    ap.add_argument("--rounds", type=int, default=cost_method.DEFAULT_ROUNDS)
    ap.add_argument("--inner", type=int, default=cost_method.DEFAULT_INNER_REPEATS)
    ap.add_argument("--pool-block", type=int, default=64)
    ap.add_argument("--pool-size", type=int, default=16)
    ap.add_argument(
        "--assembly",
        choices=("fused", "unfused"),
        default="fused",
        help="union-attention assembly (attn_fused_assembly): 'fused' (default) "
        "vs the verbatim 'unfused' one — the same selection and arithmetic, so "
        "the two profiles differ only by the fuse's own contribution",
    )
    args = ap.parse_args()
    fused = args.assembly == "fused"

    T = args.length
    base = dataclasses.replace(cost_config(), attn_fused_assembly=fused)
    modes = ("full",) + ("reindex",) * 5
    cfg = dataclasses.replace(
        base, mla_pool_block=args.pool_block, mla_pool_size=args.pool_size,
        mla_layer_modes=modes,
    )
    no_pool = dataclasses.replace(base, mla_pool_size=0, mla_layer_modes=())
    no_window = dataclasses.replace(cfg, swa_window=0)

    builder = mla.init_mla(jr.PRNGKey(0), base)
    consumer = mla.init_mla(jr.PRNGKey(1), base)
    x = jr.normal(jr.PRNGKey(2), (1, T, base.hidden))

    print(f"device: {jax.devices()}  T={T}  top_k={base.mla_top_k}  "
          f"window={base.swa_window}  pool={args.pool_size}x{args.pool_block}  "
          f"assembly={args.assembly}")
    pool = mla.apply_with_pool(builder, cfg, x, mode="full")[1]
    # ``reuse`` consumes the builder's own top_k, which is only published when a
    # reuse layer is declared — hence its own layout and its own pool.
    reuse_cfg = dataclasses.replace(cfg, mla_layer_modes=("full",) + ("reuse",) * 5)
    reuse_pool = mla.apply_with_pool(builder, reuse_cfg, x, mode="full")[1]

    dense_cfg = dataclasses.replace(base, attn_dense_reference=True)
    case = [
        ("dense", "dense (oracle)", jax.jit(lambda: mla.apply(builder, dense_cfg, x))),
        ("full", "full, pool off", jax.jit(lambda: mla.apply(builder, no_pool, x))),
        ("builder", "full, pool built", jax.jit(lambda: mla.apply_with_pool(builder, cfg, x, mode="full")[0])),
        ("reindex", "reindex in pool", jax.jit(lambda: mla.apply_with_pool(consumer, cfg, x, pool=pool, mode="reindex")[0])),
        ("reuse", "reuse (no scoring)", jax.jit(lambda: mla.apply_with_pool(consumer, reuse_cfg, x, pool=reuse_pool, mode="reuse")[0])),
        ("reindex_nw", "reindex, window off", jax.jit(lambda: mla.apply_with_pool(consumer, no_window, x, pool=pool, mode="reindex")[0])),
    ]

    label = f"pool cost | T={T} | layer ({args.assembly} assembly)"
    with ClockControl() as clock:
        cost_method.print_clock(clock, label)
        m = measure({key: fn for key, _, fn in case}, label=label, clock=clock,
                    rounds=args.rounds, inner_repeats=args.inner)
        cost_method.print_measurement(m, ref="dense", threshold=THRESHOLD_8K, legs=("dense",))
        for key, name, _ in case:
            if key == "dense":
                continue
            r = m.stats(key, "dense")
            print(f"  {name:<22} median {r.leg_median * 1e3:8.3f} ms   "
                  f"ratio median {r.median:6.3f}x  min {r.lo:.3f}  max {r.hi:.3f}  "
                  f"разброс {r.spread:.4f}  n={r.n}")

    per_layer = {key: m.stats(key, "dense").leg_median for key, _, _ in case}
    dense = per_layer["dense"]

    # Stack ratios derived from the per-layer numbers (the skeleton MLA stack is
    # one builder + five consumers; the gate measures it end to end).
    n = 6
    print(f"  derived MLA stack (x{n}, ratio to the dense stack):")
    for label, consumer in (("pinned: 1 Full + 5 Reindex", per_layer["reindex"]),
                            ("alternative: 1 Full + 5 Reuse", per_layer["reuse"])):
        stack = per_layer["builder"] + (n - 1) * consumer
        print(f"    {label:<32} {stack / (n * dense):6.3f}x  "
              f"({stack * 1e3:.1f} ms vs {n * dense * 1e3:.1f} ms)")


if __name__ == "__main__":
    main()
