"""KDA form benchmark — fwd+bwd tok/s and XLA scratch, ``chunked`` vs ``wyut``.

ADR-031 delta A replaces the chunked KDA forward's per-token transition kit with
the Kimi Linear §3.1 WY/UT form.  The claim (the reason for the delta) is a
memory cut, not a correctness change — correctness is pinned separately by
``net/tests/test_kda_wyut.py`` against ``apply_recurrent``.  This instrument
measures the two numbers the claim is about, on one frozen geometry:

* **throughput** — tokens/second of a *forward + backward* pass (the training
  step's shape: ``jax.value_and_grad`` of the KDA layer's summed output), timed
  as a median over ``--rounds`` after ``--warmup`` warm-up rounds.  Both legs
  run the same input on the same device in the same process, so the comparison
  isolates the algorithm, not the machine.
* **XLA scratch** — ``memory_analysis().temp_size_in_bytes`` of the compiled
  fwd+bwd executable: the temporary buffer XLA reserves, which is the quantity
  that made ``apply_chunked`` memory-bound (the ``P``/``Q`` prefix of the
  associative scan).

Geometry is a CLI parameter; the defaults are the case's L3 geometry
(``T=8192``, ``H=12``, ``dk=dv=128``, ``C=64``) so the script produces the
pretrain-relevant number out of the box.  The KDA layer is benchmarked in
isolation (not the whole backbone): the delta changes exactly this object.

Run it on the free device only.  The GB10 is under the AD-7/C-040 lock while a
leg is live, so this script is delivered ready-to-run and is *not* executed as
part of the delta; on a CPU backend the numbers are indicative only (XLA drops
rematerialisation on CPU and the kernels differ).

Example::

    python tools/bench_kda_wyut.py --seq-len 8192 --chunk 64 \\
        --heads 12 --dk 128 --rounds 10 --out kda_bench.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import median

import jax
import jax.numpy as jnp
import jax.random as jr

# Run as ``python tools/bench_kda_wyut.py``: ``net`` lives in the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from net import kda  # noqa: E402
from net.config import ModelConfig  # noqa: E402


def build_config(args: argparse.Namespace) -> ModelConfig:
    """A config carrying the benchmark geometry (KDA-relevant fields only).

    The invariants ``validate_config`` checks are respected: head dims agree,
    ``num_heads * head_dim == hidden``, ``num_kda_layers + num_mla_layers ==
    num_layers`` with ``num_layers % 4 == 0``.  The window branch is disabled
    (``swa_window = 0``) so the measurement is the delta-rule form itself.
    """
    return ModelConfig(
        vocab_size=1024,
        hidden=args.heads * args.dk,
        num_layers=4,
        num_kda_layers=3,
        num_mla_layers=1,
        num_heads=args.heads,
        head_dim=args.dk,
        kda_dk=args.dk,
        kda_dv=args.dk,
        kda_decay_rank=args.dk,
        mla_head_dim=args.dk,
        mla_latent_dim=max(16, args.dk),
        swa_window=0,
        attn_dense_reference=True,
        kda_wyut_chunk=args.chunk,
    )


def _make_step(impl: str, cfg: ModelConfig, x: jnp.ndarray):
    """``params -> (loss, grads)`` for one KDA primitive, vmapped over batch."""

    def loss_fn(params):
        if impl == "wyut":
            out = jax.vmap(lambda xb: kda.apply_wyut(params, cfg, xb, cfg.kda_wyut_chunk))(x)
        else:
            out = jax.vmap(lambda xb: kda.apply_chunked(params, cfg, xb, cfg.kda_wyut_chunk))(x)
        return jnp.sum(out)

    return jax.jit(jax.value_and_grad(loss_fn))


def bench_impl(impl: str, args: argparse.Namespace, cfg: ModelConfig):
    params = kda.init_kda(jr.PRNGKey(args.seed), cfg)
    x = jr.normal(jr.PRNGKey(args.seed + 1), (args.batch, args.seq_len, cfg.hidden))
    step = _make_step(impl, cfg, x)

    lowered = step.lower(params)
    compiled = lowered.compile()
    scratch = compiled.memory_analysis().temp_size_in_bytes

    for _ in range(args.warmup):
        jax.block_until_ready(step(params))
    times: list[float] = []
    for _ in range(args.rounds):
        tick = time.perf_counter()
        jax.block_until_ready(step(params))
        times.append(time.perf_counter() - tick)

    seconds = median(times)
    tokens = args.batch * args.seq_len
    return {
        "impl": impl,
        "seconds_p50": seconds,
        "seconds_min": min(times),
        "seconds_max": max(times),
        "tokens_per_sec": tokens / seconds if seconds > 0 else None,
        "xla_scratch_bytes": scratch,
        "xla_scratch_gib": scratch / (1 << 30),
    }


def run(args: argparse.Namespace) -> dict:
    cfg = build_config(args)
    results = {impl: bench_impl(impl, args, cfg) for impl in ("chunked", "wyut")}
    off, new = results["chunked"], results["wyut"]
    # After the itemised warm-up both legs share the process/clock, so the ratio
    # is the deliverable; the spread is printed next to it (no verdict on noise).
    ratio = (
        new["tokens_per_sec"] / off["tokens_per_sec"]
        if off["tokens_per_sec"] and new["tokens_per_sec"]
        else None
    )
    report = {
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "geometry": {
            "seq_len": args.seq_len,
            "chunk": args.chunk,
            "batch": args.batch,
            "heads": args.heads,
            "dk": args.dk,
            "rounds": args.rounds,
            "warmup": args.warmup,
        },
        "results": results,
        "wyut_over_chunked_tok_per_sec": ratio,
        "scratch_ratio_chunked_over_wyut": (
            off["xla_scratch_bytes"] / new["xla_scratch_bytes"]
            if new["xla_scratch_bytes"]
            else None
        ),
        "note": (
            "CPU-backend numbers are indicative only (rematerialisation is a GPU "
            "mechanism); run on the free GPU with the AD-7 lock respected."
        ),
    }
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--seq-len", type=int, default=8192, help="sequence length T")
    ap.add_argument("--chunk", type=int, default=64, help="chunk width C (both legs)")
    ap.add_argument("--batch", type=int, default=1, help="batch (independent sequences)")
    ap.add_argument("--heads", type=int, default=12, help="KDA heads")
    ap.add_argument("--dk", type=int, default=128, help="head dim (dk = dv = head_dim)")
    ap.add_argument("--rounds", type=int, default=10, help="timed rounds per leg (median)")
    ap.add_argument("--warmup", type=int, default=3, help="untimed warm-up rounds per leg")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None, help="write the JSON report here")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="tiny geometry for a wiring check (T=128, heads=2, dk=16, 2 rounds)",
    )
    args = ap.parse_args(argv)
    if args.smoke:
        args.seq_len, args.heads, args.dk, args.batch = 128, 2, 16, 1
        args.rounds, args.warmup = 2, 1

    report = run(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
