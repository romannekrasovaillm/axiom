"""ADR-018 block-wise token merging — the 64K gap bench: needle recall, cost, verdict.

The ADR (p. 3) gates the mechanism on a measurement at the 64K gap: the merged
path is accepted only when the needle recall is not worse than the unmerged
baseline within the noise tolerance *and* the cost drops measurably, measured by
the ADR-015 procedure.  Until then ``mla_block_merge`` is pinned **off** and this
bench is what produces the evidence.

What is measured (both legs differ only by the declared flag)
------------------------------------------------------------

* **recall** — a synthetic sequence of ``T = 65536`` tokens (deterministic seed,
  ``net/config.json``'s ``context_curriculum`` maximum).  Needle facts are
  written at ``10% / 50% / 90%`` of the length, three facts per position
  (9 probes), one declared block apart so that the three facts of a position
  land in three different merged blocks; a question that repeats the needle
  tokens closes the sequence.  The measurement is the *MLA indexer's* selection
  at the final query position, read per MLA layer of the real model: the share
  of the needle's own records that the selection covers — record-level ``top_k``
  for the unmerged leg, block-level ``top_k`` (``block = 16``) for the merged
  one.  Reported per probe, per position and overall, with the needle's score
  rank (is it indexer-salient at all?) and the chance coverage of each mode; the
  merged leg is additionally measured at the **equal record budget**
  (``top_k / block`` blocks — the same number of records attended), because the
  mechanism's own width covers ``block`` times more history per selection, so
  the raw comparison is one of budgets rather than of granularity.
* **cost** — one forward of the six-layer MLA stack at 64K, fed with the real
  captured activations of each MLA layer, timed by the **ADR-015 procedure**
  (``net/tests/cost_method.py``: clocks pinned or controlled and printed, both
  legs warmed up to a clock plateau, leg order reversed every round with the
  first round dropped, the verdict on the median with the spread always in the
  report, and *no verdict on noise*).  The measured quantity is the per-forward
  time of the object ADR-018 changes; the full-model ratio is dominated by the
  18 KDA layers and the dense MoE path, which the mechanism does not touch.

The flag is switched **at runtime, not by editing** ``net/config.json``: the
declared value is read from a temporary copy of the pinned config with
``mla_block_merge`` flipped (``net.config.CONFIG_PATH``), which is exactly the
switch ``net/attn_sparse.py`` consumes (spine AD-9, guard C-035).  The pinned
config file is never touched, and the flag stays off after the run.

Verdict (mechanical, from the two measurements above)
-----------------------------------------------------

``flag_on`` iff **every** probe satisfies ``recall_on >= recall_off - 0.02`` AND
``cost_on.p50 < cost_off.p50 * 0.95``; otherwise ``flag_off``.  The ADR-015
noise/clock gate on the cost arm is reported next to it (its own verdict can be
*undetermined*, which the ADR-015 decision allows and which this bench does not
launder into either direction).

Honest limits (declared, because they bound what the numbers can mean)
---------------------------------------------------------------------

1. **The weights are at initialisation.**  There is no trained L3 checkpoint in
   the case, so no indexer has learned to retrieve anything: the recall arm
   measures the *selection's coverage* of the needle position under the real
   projections, not learned retrieval.  ``needle score rank`` and the chance
   coverage are printed so a reader can see that; the instrument is the same
   one criterion 6 uses (``net/tests/test_06_nope_extrapolation.py``), applied
   at the 64K gap instead of the train gap.
2. **The recall leg hands both modes identical activations** (one capture, flag
   off), which is the delta's own idiom (``net/tests/test_23_block_merge.py``:
   identical inputs isolate the mechanism from the projections feeding it).
   ``--recall-source e2e`` instead captures one forward per leg, so the merged
   leg sees the activations its own mechanism produced.
3. **Memory bounds of the 64K forward on a 16 GB card**, all of them outside the
   measured mechanism (each is printed into the report):

   * the dense oracle cannot run at all at 64K — its ``(B, H, T, T)`` scores are
     ~206 GB — so the sparse path (the ADR-009 v1.5 delta, where ADR-018 lives)
     is the one that runs: ``attn_dense_reference = false``;
   * the tied 160K output head is sliced to ``--head-vocab`` rows of the *pinned*
     embedding table.  The capture below computes no logits at all (this bench
     reads no logits), but the head is what makes ``model.forward`` itself
     unrunnable at 64K — its logits are ``(1, 65536, 160000)`` fp32 = 42 GB — so
     the slice is declared as the bound under which the model is exercised, and
     it is also what keeps the synthetic vocabulary small;
   * AttnRes is off by default (``--attnres`` re-enables it for a small ``T``):
     its source stack at 64K is ``25 x (1, 65536, 1536)`` fp32 = 10 GB plus a
     second copy for the normalised keys, which does not fit next to the params.

   The capture itself is the fourth bound, and it is a *choice of form*, not a
   scope cut: ``--capture-form`` / :func:`capture_form` (the report records
   which form ran, in ``bounds.capture_form``).  The reference form — the
   model's own layer loop — keeps ~3.4 activation copies per layer live, so at
   64K it needs ~15.3 GiB and XLA's remat cannot lower that (it reports "only
   reduced to 20.27GiB"); the scan form runs the repeated units in one
   ``lax.scan`` and fits.  Both compute the same forward;

   The allocator limit is the fifth, and it is easy to mistake for the card:
   jax caps itself at ``XLA_PYTHON_CLIENT_MEM_FRACTION`` (75% by default) of
   the GPU, so a 16 GB card silently limits the run to ~11.7 GiB regardless of
   what is free.  The 64K capture needs the fraction raised (the run below uses
   0.95; ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` is an equivalent escape when
   the pool fragments — see the usage).  Neither touches the measurement: the
   same forward, the same clock procedure, the same rule.

Usage::

    export LD_LIBRARY_PATH=$(ls -d ~/venv-axiom/lib/python3.11/site-packages/nvidia/*/lib | tr '\\n' ':')
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95    # else the default 75% cap OOMs the 64K capture
    python tools/bench_block_merge.py                        # the боeвой 64K run
    python tools/bench_block_merge.py --smoke                # ~minute dry run
    python tools/bench_block_merge.py --skip-cost --length 16384
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import statistics
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, NamedTuple, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "net" / "tests"))  # cost_method (ADR-015)
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

# ADR-041: дисциплина памяти JAX — префлайт ДО import jax (лимит XLA + гейт стенда).
import jax_preflight  # noqa: E402

jax_preflight.ensure_mem_fraction()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402

import cost_method  # noqa: E402
from cost_method import ClockControl, measure  # noqa: E402

from net import attn_sparse, mla, model  # noqa: E402
from net import config as net_config  # noqa: E402
from net.config import BlockMergeConfig, ModelConfig, load_config, validate_config  # noqa: E402
from net.norm import rms_norm  # noqa: E402

CONFIG_PATH = ROOT / "net" / "config.json"
DEFAULT_OUT = ROOT / "evidence" / "block-merge-bench-20260929.json"

#: The gap the ADR-018 gate is written on (``context_curriculum`` maximum).
DEFAULT_LENGTH = 65536

#: Needle placement: three facts at each of three fractions of the length.
POSITION_FRACTIONS = (0.10, 0.50, 0.90)
FACTS_PER_POSITION = 3
NEEDLE_LEN = 4
#: One declared block apart, so the three facts of a position fall into three
#: different merged blocks and the per-position aggregate is not one block
#: sampled three times.
NEEDLE_STRIDE = 16
#: The question that closes the sequence repeats each needle's key token.
QUESTION_LEN = 1

#: The task's mechanical rule (ADR-018 p. 3, measurement side).
RECALL_TOLERANCE = 0.02
COST_RATIO_THRESHOLD = 0.95

#: Output head slice for the 64K forward (see the module docstring, limit 3).
DEFAULT_HEAD_VOCAB = 4096


# ---------------------------------------------------------------------------
# The synthetic needle sequence (deterministic, seed-pinned)
# ---------------------------------------------------------------------------


class NeedleProbe(NamedTuple):
    """One needle fact: where it sits and which records it occupies."""

    position_fraction: float
    fact_index: int
    start: int  # first record of the needle
    length: int  # records in the needle
    key_token: int  # the token the closing question repeats


class NeedleSequence(NamedTuple):
    token_ids: jnp.ndarray  # (T,)
    probes: tuple[NeedleProbe, ...]
    length: int
    seed: int


def build_needle_sequence(
    length: int = DEFAULT_LENGTH,
    seed: int = 0,
    vocab: int = 256,
    *,
    fractions: Sequence[float] = POSITION_FRACTIONS,
    facts: int = FACTS_PER_POSITION,
    needle_len: int = NEEDLE_LEN,
    stride: int = NEEDLE_STRIDE,
) -> NeedleSequence:
    """Fillers + needle facts + a closing question, all from one seed.

    Keys are distinct per fact so a probe's records are identifiable in the
    selection; fillers are drawn from ``[1, vocab)``.  The needles are written
    first and the question last, so the *last* position is the measurement query
    of every probe (the standard needle-in-a-haystack layout).
    """
    if length <= 0:
        raise ValueError("length must be positive")
    if vocab < 1 + facts * len(fractions) + 1:
        raise ValueError("vocab too small for distinct needle keys")
    key = jr.PRNGKey(seed)
    k_fill, k_gap = jr.split(key)
    ids = jr.randint(k_fill, (length,), 1, vocab)

    probes: list[NeedleProbe] = []
    key_base = 1  # key tokens start at 1, fillers avoid it
    for frac in fractions:
        base = int(frac * length)
        for fact in range(facts):
            start = base + fact * stride
            if start + needle_len + QUESTION_LEN + 1 >= length:
                raise ValueError(
                    f"needle at {frac:.2f} (fact {fact}) does not fit in {length} records"
                )
            key_token = key_base + len(probes)
            ids = ids.at[start : start + needle_len].set(key_token)
            probes.append(
                NeedleProbe(
                    position_fraction=float(frac),
                    fact_index=fact,
                    start=int(start),
                    length=int(needle_len),
                    key_token=int(key_token),
                )
            )
    # The closing question repeats the needle keys: the indexer query at the
    # final position is built from a hidden state that has just seen them.
    q_start = length - QUESTION_LEN - len(probes)
    for offset, probe in enumerate(probes):
        ids = ids.at[q_start + offset].set(probe.key_token)
    del k_gap
    return NeedleSequence(token_ids=ids.astype(jnp.int32), probes=tuple(probes),
                          length=int(length), seed=int(seed))


def needle_block(probe: NeedleProbe, block: int) -> set[int]:
    """Merged-block ids the probe's records belong to (a needle may straddle)."""
    return {r // block for r in range(probe.start, probe.start + probe.length)}


def coverage(probe: NeedleProbe, selected: set[int], *, mode: str, block: int) -> float:
    """Share of the probe's records covered by a selection (ADR-018 recall@k).

    ``mode = "records"``: the selection holds record ids (the unmerged leg), so
    a record is covered when it is selected itself.  ``mode = "blocks"``: the
    selection holds merged-block ids (the merged leg), so a record is covered
    when *its block* is selected — the merged representation is what attention
    then reads, so that is the honest coverage at block granularity.
    """
    if probe.length <= 0:
        raise ValueError("empty probe")
    if mode == "records":
        hits = sum(1 for r in range(probe.start, probe.start + probe.length) if r in selected)
    elif mode == "blocks":
        hits = sum(
            1
            for r in range(probe.start, probe.start + probe.length)
            if (r // block) in selected
        )
    else:  # pragma: no cover - guarded by the callers
        raise ValueError(f"unknown coverage mode {mode!r}")
    return hits / probe.length


# ---------------------------------------------------------------------------
# The declared switch, flipped at runtime (never by editing net/config.json)
# ---------------------------------------------------------------------------


@contextmanager
def declared_switch(enabled: bool, block: int, base: Path = CONFIG_PATH) -> Iterator[ModelConfig]:
    """Point the declared-config reader at a temp config with the flag flipped.

    Yields the matching :class:`ModelConfig` (so the schema object and the file
    the reader consumes never disagree).  The pinned ``net/config.json`` is only
    ever *read*: the switch lives in a copy under a temporary directory, and
    ``net.config.CONFIG_PATH`` is restored on exit.
    """
    payload = json.loads(base.read_text(encoding="utf-8"))
    payload["mla_block_merge"] = {"enabled": bool(enabled), "block": int(block)}
    previous = net_config.CONFIG_PATH
    with tempfile.TemporaryDirectory(prefix="block-merge-declared-") as tmp:
        path = Path(tmp) / "config.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        net_config.CONFIG_PATH = path
        try:
            cfg = load_config(path)
            validate_config(cfg)
            declared = attn_sparse.declared_block_merge()
            expected = int(block) if enabled else 0
            if declared != expected:
                raise RuntimeError(
                    f"declared switch did not take effect: read {declared}, expected {expected}"
                )
            yield cfg
        finally:
            net_config.CONFIG_PATH = previous


# ---------------------------------------------------------------------------
# The real model forward, with the MLA layer inputs captured
# ---------------------------------------------------------------------------


#: ``net/model.py``'s backbone pattern: ``_layer_is_kda(i) = (i % 4) != 3``, so
#: the 24 layers are six repetitions of the unit ``[KDA, KDA, KDA, MLA]``.  The
#: scan form below (``_capture_forward_scan``) is built on exactly that period.
_LAYERS_PER_UNIT = 4

#: Activation copies the reference (loop) form keeps live per layer.  Measured
#: on the 64K form: XLA's ``memory_analysis()`` reports ~250 KB per record at
#: 24 layers x 1536 hidden in bf16 (one ``(B, T, hidden)`` copy is 3072 B), so
#: ``250e3 / 3072 / 24 = 3.4`` copies per layer.  Scaled to 64K that estimates
#: ``24 x 65536 x 1536 x 2 x 3.4 = 16.4e9 B ~= 15.3 GiB`` of live set — above a
#: 16 GB card (the same figure the scan form below is measured against),
#: which is why the reference form dies there with ``RESOURCE_EXHAUSTED`` and
#: the scan form exists.  Only :func:`capture_form` reads it; ``form=``
#: overrides the choice.
_REFERENCE_LIVE_COPIES_PER_LAYER = 3.4


def _device_budget_bytes() -> int | None:
    """Bytes the device lets this process allocate, or ``None`` when unreported.

    ``jax`` reports the allocator limit on a GPU (its share of the card, by
    default ``XLA_PYTHON_CLIENT_MEM_FRACTION`` = 75% of the total — a cap that
    is *not* the card and that a 64K capture must clear, see the module
    docstring) and reports nothing on the CPU backend.

    An unreported budget is read as "the reference form fits": it is the model's
    own form and the eager-exact one (this function's callers document why that
    matters), and the scan form stays reachable through ``form="scan"``.
    """
    try:
        stats = jax.local_devices()[0].memory_stats()
    except Exception:  # pragma: no cover - a backend without accounting
        return None
    if not stats:
        return None
    limit = stats.get("bytes_limit")
    return int(limit) if limit else None


def _reference_form_bytes(cfg: ModelConfig, length: int, itemsize: int) -> int:
    """Estimated live set of the reference (loop) form at ``length`` records."""
    return int(
        int(cfg.num_layers) * int(length) * int(cfg.hidden) * int(itemsize)
        * _REFERENCE_LIVE_COPIES_PER_LAYER
    )


def capture_form(
    params: model.ModelParams,
    cfg: ModelConfig,
    input_ids: jnp.ndarray,
    *,
    use_attnres: bool = False,
    form: str = "auto",
) -> str:
    """Which form :func:`capture_forward` will take — ``"loop"`` or ``"scan"``.

    The bench records this in its report, so the instrument the numbers came
    from is declared rather than inferred by a reader.

    ``"loop"`` (the reference form, ``_capture_forward_loop``) is the model's
    forward traced one layer at a time: it is bit-equal to ``model.forward``
    both eagerly and compiled, and it is the only form that carries AttnRes.
    ``"scan"`` (``_capture_forward_scan``) runs the repeated units in one
    ``lax.scan`` and is what makes 64K fit a 16 GB card — but a compiled body
    cannot reproduce an *eagerly* evaluated ``model.forward`` bit for bit (XLA
    lowers the RMSNorm reduction differently inside a compiled body than op by
    op; ``jax.jit(rms_norm) != rms_norm``, ~1 ULP per layer, ~3.5e-5 on the
    tiny fixture after 8 layers).  Under compilation the two agree exactly, and
    compilation is how the bench consumes the capture (``capture_leg`` jits it,
    as does any real 64K measurement).  So the scan form is chosen when the
    reference form's estimated live set does not fit the device, and the
    reference form is chosen otherwise — which keeps the eager-exact form as
    the default wherever it is affordable.
    """
    if form not in ("auto", "loop", "scan"):
        raise ValueError(f"unknown capture form {form!r}: expected auto, loop or scan")
    if form == "loop":
        return "loop"
    if form == "auto":
        budget = _device_budget_bytes()
        need = _reference_form_bytes(
            cfg, int(input_ids.shape[-1]), int(params.embedding.dtype.itemsize)
        )
        # An unreported budget reads as "the reference form fits": it is the
        # model's own layer loop and the eager-exact form, and a caller that
        # knows better can ask for the scan form by name.  That is the case a
        # 16 GiB-sized forward can still slip through on, so the bench warns
        # when it picks the reference form for a need this large.
        if budget is None or need <= budget:
            return "loop"
    if _scanned_units(cfg, use_attnres) is None:
        return "loop"
    return "scan"


def _capture_forward_loop(
    params: model.ModelParams,
    cfg: ModelConfig,
    input_ids: jnp.ndarray,
    *,
    use_attnres: bool,
    chunk_size: int,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, ...]]:
    """The layer loop as written, one traced layer at a time.

    This is the reference form: it is what ``model.forward`` does and it is the
    only form that can carry AttnRes (which needs every earlier layer's delta
    live, ``O(num_layers x T x hidden)`` by construction).  At 64K its memory
    is what :func:`_capture_forward_scan` exists to fix — see that function.
    """
    emb = params.embedding[input_ids]
    h = emb
    embed_src = emb
    layer_deltas: list[jnp.ndarray] = []
    mla_inputs: list[jnp.ndarray] = []
    pool = None
    for i, block in enumerate(params.layers):
        is_kda = model._layer_is_kda(i)
        mode = "full" if is_kda else mla.layer_mode(cfg, model._mla_ordinal(i))
        if not is_kda:
            mla_inputs.append(rms_norm(h, block.norm_attn))  # what the MLA layer reads
        delta, _qb, pool = model._block_delta(block, is_kda, cfg, h, chunk_size, False, mode, pool)
        if use_attnres:
            if i > 0:
                sources = jnp.stack([embed_src] + layer_deltas, axis=0)
                corr = model.attnres_mod.apply_layer(params.attnres.w[i], sources)
            else:
                corr = 0.0
            layer_deltas.append(delta)
        else:
            # AttnRes off: no consumer for per-layer deltas — do not append.
            # A live python-list reference defeats XLA DCE and holds
            # 24 x (T, hidden) in memory (the 15.4 GiB capture OOM at 64K).
            corr = 0.0
        h = h + delta + corr
    return rms_norm(h, params.norm_final), tuple(mla_inputs)


def _stack_units(tree_list: Sequence) -> object:
    """``jax.tree_util`` stack with a leading unit axis (``None`` leaves pass)."""
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *tree_list)


def _first_scanned_unit(cfg: ModelConfig) -> int:
    """First unit whose four layers are all past the dense-MLP prefix.

    Only whole units are scanned: the unit body indexes its four sub-layers
    statically, and the dense layer-0 MLP has a different parameter tree from
    the LatentMoE layers that follow it, so a unit containing a dense layer
    cannot share the stacked parameter tree.  ``moe_dense_layers`` layers are
    dense, i.e. ``ceil(moe_dense_layers / 4)`` leading units.
    """
    return -(-int(cfg.moe_dense_layers) // _LAYERS_PER_UNIT)


def _scanned_mode(cfg: ModelConfig, first_unit: int) -> str | None:
    """The one MLA mode shared by the scanned units, or ``None`` if not shared.

    ``mla_layer_modes`` is indexed by MLA ordinal, and a unit's MLA layer is the
    unit's fourth layer, so its ordinal *is* the unit index — the mode is a
    property of the unit.  ``mode`` is consumed by ``net/mla.py`` in Python
    control flow (it is not a traced value), so the scan can only be built when
    every scanned unit agrees on it.  A scanned ``full`` mode that publishes the
    ADR-012 pool is also rejected: the pool would change the scan carry's type
    part-way through, which a scan cannot carry.
    """
    n_units = int(cfg.num_layers) // _LAYERS_PER_UNIT
    modes = {mla.layer_mode(cfg, u) for u in range(first_unit, n_units)}
    if len(modes) != 1:
        return None
    mode = modes.pop()
    if mode == "full" and int(cfg.mla_pool_size) > 0:
        return None
    return mode


def _scanned_units(cfg: ModelConfig, use_attnres: bool) -> tuple[int, str] | None:
    """``(first_scanned_unit, mode)`` when the scan form is usable, else ``None``.

    The single place the scan form's preconditions live, so the form decision
    (:func:`capture_form`) and the dispatch (:func:`capture_forward`) cannot
    disagree about them.
    """
    n_units = int(cfg.num_layers) // _LAYERS_PER_UNIT
    first_unit = _first_scanned_unit(cfg)
    if (
        use_attnres
        or int(cfg.num_layers) % _LAYERS_PER_UNIT != 0
        or first_unit >= n_units
    ):
        return None
    mode = _scanned_mode(cfg, first_unit)
    if mode is None:
        return None
    return first_unit, mode


def _capture_forward_scan(
    params: model.ModelParams,
    cfg: ModelConfig,
    input_ids: jnp.ndarray,
    *,
    chunk_size: int,
    first_unit: int,
    mode: str,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, ...]]:
    """The same layer loop with its repeated units run by one ``lax.scan``.

    Why this exists (measurement, RTX 4080 16 GiB, T = 16384, temp = XLA's
    ``memory_analysis()``): the traced-one-layer-at-a-time form costs
    ``~250 KB per record`` at 24 layers and grows *linearly in the layer count*
    (KDA-only: 31 KB/rec at 1 layer, 99 KB at 6, 247 KB at 18), so at 64K its
    working set is ~15.3 GiB and the forward dies with ``RESOURCE_EXHAUSTED`` on
    a 15.4 GiB card.  The growth is real liveness, not a reporting artifact:
    ``jax.checkpoint`` on every layer leaves the number *bit-identical*
    (XLA's own remat pass already reports "only reduced to 20.27GiB ... down
    from 20.27GiB"), and XLA's while-loop double buffering is not the cause.
    Running the repeated units inside a single ``lax.scan`` makes XLA reuse one
    unit's buffers across iterations instead of keeping every layer's live:
    18 KDA layers cost 247 KB/rec in a python chain and 38 KB/rec inside one
    scan, flat in the layer count.

    The arithmetic is untouched — same :func:`net.model._block_delta`, same
    mode, same order, same parameters (only stacked along a leading axis and
    sliced back) — so the hidden state is the model's.  Equality is checked in
    the environment the bench uses, i.e. compiled: the scan body is XLA-compiled
    by construction, so it reproduces ``jax.jit(model.forward(..., return_hidden=True))``
    bit for bit but *cannot* reproduce an eagerly evaluated ``model.forward``
    (XLA lowers a reduction inside a compiled body in a different association
    order than op-by-op — ``jax.jit(rms_norm) != rms_norm`` by ~1 ULP, which
    RMSNorm compounds across layers).  :func:`capture_form` carries the
    measurement; T-b4 pins the eager equality for the reference form and the
    compiled one for this form.

    The units are ``[KDA, KDA, KDA, MLA]`` (``net/model.py``'s pattern), the
    ``first_unit`` leading units are run in python (they hold the dense MLP) and
    the rest by the scan.  The scan body returns each unit's MLA layer input, so
    the stacked scan output is exactly the per-MLA-layer capture, in order.
    """
    n_units = int(cfg.num_layers) // _LAYERS_PER_UNIT
    units = list(range(first_unit, n_units))

    def layer(u: int, s: int) -> model.BlockParams:
        return params.layers[u * _LAYERS_PER_UNIT + s]

    # One leading-unit stack per sub-layer and field, so ``lax.scan`` slices the
    # unit axis for us and the body needs no dynamic indexing (a NamedTuple
    # parameter tree cannot be indexed on a stacked axis).
    xs = (
        jnp.stack([jnp.stack([layer(u, s).norm_attn for s in range(4)]) for u in units]),
        jnp.stack([jnp.stack([layer(u, s).norm_mlp for s in range(4)]) for u in units]),
        _stack_units([layer(u, 0).attn for u in units]),  # KDA
        _stack_units([layer(u, 1).attn for u in units]),  # KDA
        _stack_units([layer(u, 2).attn for u in units]),  # KDA
        _stack_units([layer(u, 3).attn for u in units]),  # MLA
        tuple(_stack_units([layer(u, s).mlp for u in units]) for s in range(4)),
    )

    def body(carry, unit):
        h, pool = carry
        norm_attn, norm_mlp, attn0, attn1, attn2, attn_mla, mlp = unit
        mla_in = None
        for s, attn_p in enumerate((attn0, attn1, attn2, attn_mla)):
            is_kda = s != _LAYERS_PER_UNIT - 1
            block = model.BlockParams(
                norm_attn=norm_attn[s], attn=attn_p, norm_mlp=norm_mlp[s], mlp=mlp[s]
            )
            if not is_kda:
                mla_in = rms_norm(h, norm_attn[s])  # what the MLA layer reads
            delta, _qb, pool = model._block_delta(
                block, is_kda, cfg, h, chunk_size, False, "full" if is_kda else mode, pool
            )
            h = h + delta
        return (h, pool), mla_in

    h = params.embedding[input_ids]
    mla_inputs: list[jnp.ndarray] = []
    pool = None
    for i in range(first_unit * _LAYERS_PER_UNIT):
        block = params.layers[i]
        is_kda = model._layer_is_kda(i)
        layer_mode = "full" if is_kda else mla.layer_mode(cfg, model._mla_ordinal(i))
        if not is_kda:
            mla_inputs.append(rms_norm(h, block.norm_attn))
        delta, _qb, pool = model._block_delta(
            block, is_kda, cfg, h, chunk_size, False, layer_mode, pool
        )
        h = h + delta

    (h, _pool), ys = jax.lax.scan(body, (h, pool), xs)
    mla_inputs.extend(ys[u] for u in range(len(units)))
    return rms_norm(h, params.norm_final), tuple(mla_inputs)


def capture_forward(
    params: model.ModelParams,
    cfg: ModelConfig,
    input_ids: jnp.ndarray,
    *,
    use_attnres: bool = False,
    chunk_size: int = 64,
    form: str = "auto",
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, ...]]:
    """``net/model.py``'s forward, returning the MLA layer inputs as well.

    The layer loop is ``model.forward``'s, verbatim: the same
    :func:`net.model._block_delta`, the same layer modes and the same AttnRes
    call, in the same order — only the output head is dropped (this bench reads
    no logits) and the RMSNorm that feeds each MLA layer is kept.  The equality
    of the hidden state with ``model.forward(..., return_hidden=True)`` is
    asserted by ``tools/tests/test_bench_block_merge.py`` (T-b4: eagerly for the
    reference form, which is what the fixture takes, and compiled for the scan
    form — :func:`capture_form` documents why the scan form cannot be held to
    the eager comparison), so the capture cannot drift from the model it claims
    to be.

    The loop is traced one layer at a time (``_capture_forward_loop``) or, when
    the reference form's live set does not fit the card, with its repeated units
    run by a single ``lax.scan`` (``_capture_forward_scan``, which documents the
    64K measurement).  The scan form needs AttnRes off (AttnRes keeps every
    earlier layer's delta live by construction, so it is intractable at 64K
    anyway — module docstring, limit 3), a whole number of four-layer units, at
    least one scanned unit, one shared MLA mode across them and no pool
    published inside the scan; anything else falls back to the reference form.
    Either way the arithmetic is the same and, compiled, so is the result.
    """
    resolved = capture_form(params, cfg, input_ids, use_attnres=use_attnres, form=form)
    if resolved == "loop":
        return _capture_forward_loop(
            params, cfg, input_ids, use_attnres=use_attnres, chunk_size=chunk_size
        )
    scanned = _scanned_units(cfg, use_attnres)
    if scanned is None:  # ``capture_form`` cannot return "scan" here; belt and braces
        return _capture_forward_loop(
            params, cfg, input_ids, use_attnres=use_attnres, chunk_size=chunk_size
        )
    first_unit, mode = scanned
    return _capture_forward_scan(
        params, cfg, input_ids, chunk_size=chunk_size, first_unit=first_unit, mode=mode
    )


def _params_with_head_slice(
    params: model.ModelParams, head_vocab: int
) -> model.ModelParams:
    """The pinned params with the tied output head sliced to its first rows.

    The slice is a view of the *pinned* embedding table, so the rows the model
    reads are the pinned ones; only the head's width shrinks.  Token ids must
    stay below ``head_vocab`` (the callers assert it).
    """
    if head_vocab >= params.embedding.shape[0]:
        return params
    return params._replace(embedding=params.embedding[:head_vocab])


def cast_params(params, dtype: jnp.dtype):
    """Cast a parameter tree to ``dtype`` (the pinned init draw, cast, not re-drawn).

    ``net/config.json`` declares ``pretrain_dtype: "bf16"``; the skeleton's own
    tests run fp32 because they measure numeric parity (ADR-010), but at 64K the
    fp32 forward does not fit on a 16 GB card — the MoE layers materialise
    ``(T, n_routed, expert_inter)`` activations that fp32 doubles.  The cast is
    applied to *both* legs, so the comparison stays like-for-like.
    """
    if dtype == jnp.float32:
        return params
    return jax.tree_util.tree_map(lambda a: a.astype(dtype), params)


# ---------------------------------------------------------------------------
# The indexer selection, mirrored from net/mla.py's sparse path
# ---------------------------------------------------------------------------


def indexer_projections(
    mla_params: mla.MLAParams, cfg: ModelConfig, x_layer: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(idx_q, idx_k)`` for one MLA layer, as ``mla._sparse_apply`` builds them.

    The projections are the layer's own (``W_c`` / ``W_idx_q`` / ``W_idx_k``),
    including the FP4 fake-quant of the latent when ``qat_kv_enabled`` — the
    selection must see the representations the model selected from.
    """
    x32 = x_layer.astype(jnp.float32)
    c = x32 @ mla_params.W_c.astype(jnp.float32)
    if cfg.qat_kv_enabled:
        from net import quant

        c = quant.fake_quant_mxfp4_latent(c)
    B, T = x_layer.shape[:2]
    idx_q = (x32 @ mla_params.W_idx_q.astype(jnp.float32)).reshape(
        B, T, cfg.mla_index_heads, cfg.mla_index_dim
    )
    idx_k = (c @ mla_params.W_idx_k.astype(jnp.float32)).reshape(
        B, T, cfg.mla_index_heads, cfg.mla_index_dim
    )
    return idx_q, idx_k


def unmerged_selection(
    idx_q: jnp.ndarray,
    idx_k: jnp.ndarray,
    cfg: ModelConfig,
    query: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Record-level ``top_k`` of the full causal prefix at ``query``.

    Returns ``(selected_record_ids, scores)`` for the single query position (the
    bench runs batch 1); the scores of *all* causal records are returned too,
    because the bench reports where the needle ranks among them (is it
    indexer-salient at all?).
    """
    Hi = idx_q.shape[2]
    qi = idx_q[:, query : query + 1]  # (B, 1, Hi, Di)
    scores = jnp.einsum("bqhi,bshi->bqs", qi, idx_k) / Hi  # (B, 1, T)
    width = min(int(cfg.mla_top_k), int(idx_k.shape[1]))
    if cfg.attn_topk_exact:
        ids = attn_sparse.topk_exact.topk_indices(scores, width)
    else:
        ids = jax.lax.top_k(scores, width)[1]
    return ids[0, 0, :], scores[0, 0, :]


def merged_selection(
    idx_q: jnp.ndarray,
    idx_k: jnp.ndarray,
    cfg: ModelConfig,
    query: int,
    block: int,
    top_k: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Block-level ``top_k`` at ``query`` (ADR-018), plus the merged block scores.

    The pooling and the ranking are the mechanism's own functions
    (:func:`net.attn_sparse.mean_blocks`, :func:`merged_block_selection`), so the
    bench ranks exactly what the merged path ranks.  ``top_k`` overrides the
    width for the equal-budget diagnostic (blocks vs records covering the same
    number of records); the mechanism's own width is ``cfg.mla_top_k`` blocks.
    """
    T = int(idx_k.shape[1])
    n_blk = T // block
    if n_blk < 1:
        raise ValueError("sequence shorter than one merged block")
    idx_k_merged = attn_sparse.mean_blocks(idx_k, block, n_blk)
    pos = jnp.asarray([query], jnp.int32)
    qi = idx_q[:, query : query + 1]  # the queries whose selection is measured
    ids = attn_sparse.merged_block_selection(
        qi, idx_k_merged, pos, block,
        int(cfg.mla_top_k) if top_k is None else int(top_k),
        exact_topk=bool(cfg.attn_topk_exact),
    )
    # The scores themselves, for the rank diagnostic (same formula as the
    # primitive's: indexer affinity of the merged records, averaged per head).
    scores = jnp.einsum("bqhi,bjhi->bqj", qi.astype(jnp.float32),
                        idx_k_merged.astype(jnp.float32)) / idx_q.shape[2]
    return ids[0, 0, :], scores[0, 0, :], idx_k_merged


def _rank_of(values: jnp.ndarray, targets: Sequence[int]) -> int:
    """1-based rank of the best target inside ``values`` (1 = highest score)."""
    flat = jnp.asarray(values).reshape(-1)
    best = max(float(flat[t]) for t in targets)
    return int(jnp.sum(flat > best)) + 1


# ---------------------------------------------------------------------------
# Recall measurement
# ---------------------------------------------------------------------------


def capture_leg(
    cfg: ModelConfig,
    params: model.ModelParams,
    ids: jnp.ndarray,
    *,
    use_attnres: bool,
    chunk_size: int = 64,
    form: str = "auto",
    label: str = "",
    verbose: bool = True,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, ...]]:
    """One jitted 64K forward of the real model, returning the MLA layer inputs."""
    fn = jax.jit(lambda p, x: capture_forward(p, cfg, x, use_attnres=use_attnres,
                                              chunk_size=chunk_size, form=form))
    t0 = time.perf_counter()
    out = fn(params, ids)
    jax.block_until_ready(out)
    if verbose:
        print(f"[bench] forward ({label}) {time.perf_counter() - t0:.1f}s "
              f"(compile+run, hidden {tuple(out[0].shape)})", flush=True)
    return out


def mla_layer_params(params: model.ModelParams) -> list[mla.MLAParams]:
    """The MLA layers' parameter sets, in layer order."""
    return [b.attn for i, b in enumerate(params.layers) if not model._layer_is_kda(i)]


def measure_recall(
    cfg: ModelConfig,
    params: model.ModelParams,
    sequence: NeedleSequence,
    captures: dict[str, tuple[jnp.ndarray, tuple[jnp.ndarray, ...]]],
    *,
    block: int,
    recall_source: str = "isolated",
) -> dict:
    """Needle recall of the two selections, per probe / position / MLA layer.

    ``captures`` holds the forward already run for each leg: for the ``isolated``
    source both keys point at the same capture (the delta's own idiom — identical
    inputs isolate the mechanism from the projections feeding it), for ``e2e``
    the merged leg has its own.
    """
    T = sequence.length
    _, mla_inputs_off = captures["off"]
    hidden_off = captures["off"][0]
    hidden_on, mla_inputs_on = captures["on"]
    if recall_source == "e2e" and bool(jnp.array_equal(hidden_off, hidden_on)):
        raise RuntimeError(
            "the runtime switch did not reach the path: both legs produced identical "
            "hidden states with block merging on — the measurement would be vacuous"
        )

    mla_params = mla_layer_params(params)
    layer_indices = [i for i in range(len(params.layers)) if not model._layer_is_kda(i)]
    if len(mla_inputs_off) != len(mla_params):
        raise RuntimeError("capture/layer mismatch: %d inputs, %d MLA layers"
                           % (len(mla_inputs_off), len(mla_params)))

    query = T - 1
    # The equal-budget width: the number of blocks whose records add up to the
    # unmerged selection width, so the two selections read the same number of
    # records and the comparison is not the coverage of a 16x larger budget.
    equal_budget_blocks = max(1, min(int(cfg.mla_top_k), T) // block)
    records: list[dict] = []
    for probe in sequence.probes:
        per_layer: dict[str, list[float]] = {"off": [], "on": [], "eq": []}
        ranks: dict[str, list[int]] = {"off": [], "on": []}
        for layer, (p_mla, x_off, x_on) in enumerate(zip(mla_params, mla_inputs_off, mla_inputs_on)):
            needle_records = range(probe.start, probe.start + probe.length)
            # unmerged leg: record-level top_k over the full causal prefix
            idx_q, idx_k = indexer_projections(p_mla, cfg, x_off)
            sel_ids, scores = unmerged_selection(idx_q, idx_k, cfg, query)
            per_layer["off"].append(
                coverage(probe, set(int(v) for v in sel_ids), mode="records", block=block)
            )
            ranks["off"].append(_rank_of(scores, needle_records))
            # merged leg (ADR-018): block-level top_k over the pooled history
            idx_q_m, idx_k_m = indexer_projections(p_mla, cfg, x_on)
            sel_blocks, block_scores, _ = merged_selection(idx_q_m, idx_k_m, cfg, query, block)
            per_layer["on"].append(
                coverage(probe, set(int(v) for v in sel_blocks), mode="blocks", block=block)
            )
            ranks["on"].append(_rank_of(block_scores, sorted(needle_block(probe, block))))
            # ... and at the width that reads the same number of records
            sel_eq, _eq_scores, _ = merged_selection(
                idx_q_m, idx_k_m, cfg, query, block, top_k=equal_budget_blocks
            )
            per_layer["eq"].append(
                coverage(probe, set(int(v) for v in sel_eq), mode="blocks", block=block)
            )
        n_blocks = T // block
        width_off = min(int(cfg.mla_top_k), T)
        width_on = min(int(cfg.mla_top_k), n_blocks)
        records.append(
            {
                "position_fraction": probe.position_fraction,
                "fact_index": probe.fact_index,
                "start": probe.start,
                "length": probe.length,
                "block_ids": sorted(needle_block(probe, block)),
                "per_layer": {k: v for k, v in per_layer.items()},
                "recall_off": statistics.fmean(per_layer["off"]),
                "recall_on": statistics.fmean(per_layer["on"]),
                "recall_on_equal_budget": statistics.fmean(per_layer["eq"]),
                "equal_budget_blocks": equal_budget_blocks,
                "recall_delta": statistics.fmean(per_layer["on"]) - statistics.fmean(per_layer["off"]),
                "needle_rank_off": statistics.median(ranks["off"]),
                "needle_rank_on": statistics.median(ranks["on"]),
                "needle_percentile_off": 1.0 - statistics.median(ranks["off"]) / width_off,
                "needle_percentile_on": 1.0 - statistics.median(ranks["on"]) / n_blocks,
                "chance_off": width_off / T,
                "chance_on": width_on * block / T,
            }
        )

    by_position = {}
    for frac in POSITION_FRACTIONS:
        group = [r for r in records if r["position_fraction"] == frac]
        if not group:
            continue
        by_position[f"{frac:.2f}"] = {
            "probes": len(group),
            "recall_off": statistics.fmean(r["recall_off"] for r in group),
            "recall_on": statistics.fmean(r["recall_on"] for r in group),
            "recall_on_equal_budget": statistics.fmean(
                r["recall_on_equal_budget"] for r in group
            ),
        }
    overall_off = statistics.fmean(r["recall_off"] for r in records)
    overall_on = statistics.fmean(r["recall_on"] for r in records)
    overall_eq = statistics.fmean(r["recall_on_equal_budget"] for r in records)
    return {
        "source": recall_source,
        "length": T,
        "seed": sequence.seed,
        "block": block,
        "layers": layer_indices,
        "query_position": query,
        "tolerance": RECALL_TOLERANCE,
        "probes": records,
        "by_position": by_position,
        "overall": {
            "recall_off": overall_off,
            "recall_on": overall_on,
            "recall_on_equal_budget": overall_eq,
            "equal_budget_blocks": equal_budget_blocks,
            "delta": overall_on - overall_off,
            "delta_equal_budget": overall_eq - overall_off,
            "chance_off": min(int(cfg.mla_top_k), T) / T,
            "chance_on": min(int(cfg.mla_top_k), T // block) * block / T,
            "chance_on_equal_budget": equal_budget_blocks * block / T,
        },
        "all_probes_within_tolerance": all(
            r["recall_on"] >= r["recall_off"] - RECALL_TOLERANCE for r in records
        ),
    }


# ---------------------------------------------------------------------------
# Cost measurement (ADR-015, via net/tests/cost_method.py)
# ---------------------------------------------------------------------------


def _stack_runner(cfg: ModelConfig, params_list: Sequence[mla.MLAParams],
                  inputs: Sequence[jnp.ndarray], modes: Sequence[str]):
    """One forward of the MLA stack, each layer fed with its own captured input.

    This is the object ADR-018 changes: the six MLA layers of the pinned config,
    with their own real activations, threaded through the ADR-012 pool exactly as
    ``model.forward`` threads it (a merged layer publishes none, which is the
    declared behaviour of the mechanism).
    """
    layers = tuple(params_list)
    xs = tuple(inputs)
    mode_list = tuple(modes)

    def run():
        pool = None
        out = None
        for p_mla, x_layer, mode in zip(layers, xs, mode_list):
            out, pool = mla.apply_with_pool(p_mla, cfg, x_layer, pool=pool, mode=mode)
        return out

    return run


def _percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile of a small sample (p50 = median by construction)."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty sample")
    if q <= 0:
        return ordered[0]
    if q >= 1:
        return ordered[-1]
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def measure_cost(
    cfg: ModelConfig,
    params_list: Sequence[mla.MLAParams],
    inputs: Sequence[jnp.ndarray],
    *,
    block: int,
    rounds: int,
    inner: int,
    verbose: bool = True,
) -> dict:
    """ADR-015 measurement of the MLA stack at 64K, flag off vs flag on."""
    modes = tuple(mla.layer_mode(cfg, i) for i in range(len(params_list)))
    legs: dict[str, object] = {}
    for enabled in (False, True):
        with declared_switch(enabled, block):
            leg_cfg = dataclasses.replace(
                cfg, mla_block_merge=BlockMergeConfig(enabled=enabled, block=block)
            )
            fn = jax.jit(_stack_runner(leg_cfg, params_list, inputs, modes))
            t0 = time.perf_counter()
            out = fn()
            jax.block_until_ready(out)
            if verbose:
                print(f"[cost] leg {'on' if enabled else 'off'} compiled+ran in "
                      f"{time.perf_counter() - t0:.1f}s", flush=True)
            legs["on" if enabled else "off"] = fn

    # The switch must be visible in the numbers, or the cost comparison would be
    # two copies of the same graph.
    with declared_switch(False, block):
        out_off = jax.block_until_ready(legs["off"]())
    with declared_switch(True, block):
        out_on = jax.block_until_ready(legs["on"]())
    if bool(jnp.array_equal(out_off, out_on)):
        raise RuntimeError(
            "block merging changed nothing: the two cost legs are the same graph, "
            "so the measurement would be vacuous"
        )

    label = f"ADR-018 block merge | MLA stack x{len(params_list)} | {inputs[0].shape[1] // 1024}K"
    with ClockControl() as clock:
        cost_method.print_clock(clock, label)
        measurement = measure(legs, label=label, clock=clock, rounds=rounds, inner_repeats=inner)

    times_off = measurement.leg_times("off")
    times_on = measurement.leg_times("on")
    ratios = measurement.stats("on", "off")
    cost_method.print_measurement(measurement, ref="off", threshold=COST_RATIO_THRESHOLD,
                                  unit_scale=1e3, unit="ms")
    adr015 = cost_method.verdict(
        ratios, COST_RATIO_THRESHOLD, clock=clock, measurement=measurement, strictly_better=True
    )
    cost_method.print_verdict(adr015, label + " [ADR-015 gate: on/off < 0.95]")

    p50_off, p50_on = statistics.median(times_off), statistics.median(times_on)
    ratio_of_medians = p50_on / p50_off
    return {
        "label": label,
        "layers": len(params_list),
        "length": int(inputs[0].shape[1]),
        "mode_layout": list(modes),
        "legs": {
            "off": {
                "p50_s": p50_off,
                "p90_s": _percentile(times_off, 0.9),
                "min_s": min(times_off),
                "max_s": max(times_off),
                "spread_s": max(times_off) - min(times_off),
                "n": len(times_off),
                "per_round_s": list(times_off),
            },
            "on": {
                "p50_s": p50_on,
                "p90_s": _percentile(times_on, 0.9),
                "min_s": min(times_on),
                "max_s": max(times_on),
                "spread_s": max(times_on) - min(times_on),
                "n": len(times_on),
                "per_round_s": list(times_on),
            },
        },
        "ratio_of_medians": ratio_of_medians,
        "ratio_threshold": COST_RATIO_THRESHOLD,
        "cost_drop_measurable": bool(ratio_of_medians < COST_RATIO_THRESHOLD),
        "rounds": {
            "declared": len(measurement.rounds),
            "dropped": measurement.dropped,
            "counted": len(measurement.counted),
            "inner_repeats": measurement.inner_repeats,
        },
        "adr015": {
            "verdict": adr015.status,
            "threshold": adr015.threshold,
            "dispersion_limit": adr015.dispersion_limit,
            "ratio_median": ratios.median,
            "ratio_min": ratios.lo,
            "ratio_max": ratios.hi,
            "ratio_spread": ratios.spread,
            "reasons": list(adr015.reasons),
            "clock": clock.report.clock_line(),
            "clock_controlled": bool(clock.report.controlled),
            "hard_pin_ok": bool(clock.report.hard_pin_ok),
        },
    }


# ---------------------------------------------------------------------------
# The mechanical verdict
# ---------------------------------------------------------------------------


def decide(recall: dict | None, cost: dict | None) -> dict:
    """The ADR-018 gate verdict: ``flag_on`` iff both arms hold, else ``flag_off``.

    The rule is the task's, mechanically: every probe satisfies
    ``recall_on >= recall_off - RECALL_TOLERANCE`` and ``cost_on.p50`` is below
    ``0.95 x cost_off.p50``.  The ADR-015 noise/clock gate is reported as its own
    verdict and never rewritten into the flag decision.
    """
    reasons: list[str] = []
    recall_ok: bool | None = None
    cost_ok: bool | None = None
    if recall is not None:
        failing = [
            (r["position_fraction"], r["fact_index"], r["recall_off"], r["recall_on"])
            for r in recall["probes"]
            if r["recall_on"] < r["recall_off"] - RECALL_TOLERANCE
        ]
        recall_ok = not failing
        if failing:
            shown = ", ".join(
                f"p={f:.2f}/fact{j}: off {a:.3f} -> on {b:.3f}" for f, j, a, b in failing
            )
            reasons.append(f"recall dropped beyond the tolerance on {len(failing)} probe(s): {shown}")
    else:
        reasons.append("recall arm not measured")

    if cost is not None:
        cost_ok = bool(cost["cost_drop_measurable"])
        if not cost_ok:
            reasons.append(
                f"cost_on.p50 {cost['legs']['on']['p50_s'] * 1e3:.3f} ms is not below "
                f"{COST_RATIO_THRESHOLD:.2f} x cost_off.p50 "
                f"{cost['legs']['off']['p50_s'] * 1e3:.3f} ms "
                f"(ratio {cost['ratio_of_medians']:.4f})"
            )
    else:
        reasons.append("cost arm not measured")

    flag_on = bool(recall_ok) and bool(cost_ok)
    return {
        "rule": (
            f"flag_on iff recall_on >= recall_off - {RECALL_TOLERANCE} on all probes "
            f"AND cost_on.p50 < {COST_RATIO_THRESHOLD} x cost_off.p50; else flag_off"
        ),
        "verdict": "flag_on" if flag_on else "flag_off",
        "recall_condition": recall_ok,
        "cost_condition": cost_ok,
        "reasons": reasons if not flag_on else [],
        "applied_by_architect": False,
        "note": (
            "the verdict is reported, not applied: net/config.json keeps "
            "mla_block_merge.enabled = false until the architect rules on this report"
        ),
    }


# ---------------------------------------------------------------------------
# Environment / report helpers
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gpu_name() -> str:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,clocks.max.sm",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        return f"nvidia-smi unavailable: {type(exc).__name__}"
    return proc.stdout.strip().replace("\n", " | ") or "nvidia-smi returned nothing"


def _print_recall(recall: dict, block: int) -> None:
    print(f"[recall] {recall['length']} records, seed {recall['seed']}, block {block}, "
          f"source {recall['source']}, query position {recall['query_position']}, "
          f"MLA layers {recall['layers']}")
    print(f"[recall] {'position':>9} {'fact':>4} {'start':>7} {'off':>7} {'on':>7} {'on_eq':>7} "
          f"{'delta':>7} {'rank_off':>9} {'rank_on':>8} {'chance_off':>10} {'chance_on':>9}")
    for r in recall["probes"]:
        print(f"[recall] {r['position_fraction']:>9.2f} {r['fact_index']:>4} {r['start']:>7} "
              f"{r['recall_off']:>7.3f} {r['recall_on']:>7.3f} "
              f"{r['recall_on_equal_budget']:>7.3f} {r['recall_delta']:>+7.3f} "
              f"{r['needle_rank_off']:>9} {r['needle_rank_on']:>8} "
              f"{r['chance_off']:>10.5f} {r['chance_on']:>9.4f}")
    for frac, agg in recall["by_position"].items():
        print(f"[recall] by position {frac}: off {agg['recall_off']:.3f} "
              f"on {agg['recall_on']:.3f} on@equal_budget {agg['recall_on_equal_budget']:.3f} "
              f"(n={agg['probes']})")
    o = recall["overall"]
    print(f"[recall] overall: off {o['recall_off']:.3f} on {o['recall_on']:.3f} "
          f"(delta {o['delta']:+.3f}); on at the equal record budget "
          f"({o['equal_budget_blocks']} blocks) {o['recall_on_equal_budget']:.3f} "
          f"(delta {o['delta_equal_budget']:+.3f}); chance coverage off {o['chance_off']:.5f} "
          f"on {o['chance_on']:.4f} on@equal_budget {o['chance_on_equal_budget']:.5f}; "
          f"all probes within tolerance: {recall['all_probes_within_tolerance']}")


def _print_cost(cost: dict) -> None:
    off, on = cost["legs"]["off"], cost["legs"]["on"]
    print(f"[cost] {cost['label']}: modes {cost['mode_layout']}")
    print(f"[cost] off: p50 {off['p50_s'] * 1e3:.3f} ms  p90 {off['p90_s'] * 1e3:.3f} ms  "
          f"min {off['min_s'] * 1e3:.3f}  max {off['max_s'] * 1e3:.3f}  n={off['n']}")
    print(f"[cost] on : p50 {on['p50_s'] * 1e3:.3f} ms  p90 {on['p90_s'] * 1e3:.3f} ms  "
          f"min {on['min_s'] * 1e3:.3f}  max {on['max_s'] * 1e3:.3f}  n={on['n']}")
    print(f"[cost] ratio on/off (p50) {cost['ratio_of_medians']:.4f} "
          f"(threshold {cost['ratio_threshold']}); measurable drop: "
          f"{cost['cost_drop_measurable']}")
    print(f"[cost] ADR-015 gate: {cost['adr015']['verdict']} — "
          f"ratio median {cost['adr015']['ratio_median']:.4f} "
          f"(min {cost['adr015']['ratio_min']:.4f}…max {cost['adr015']['ratio_max']:.4f}, "
          f"spread {cost['adr015']['ratio_spread']:.4f}); clock: {cost['adr015']['clock']}")
    if cost["adr015"]["reasons"]:
        print(f"[cost] ADR-015 gate reasons: {'; '.join(cost['adr015']['reasons'])}")


def _print_verdict(verdict: dict) -> None:
    head = "FLAG_ON" if verdict["verdict"] == "flag_on" else "FLAG_OFF"
    print(f"[verdict] {head} — recall condition {verdict['recall_condition']}, "
          f"cost condition {verdict['cost_condition']}")
    print(f"[verdict] rule: {verdict['rule']}")
    for reason in verdict["reasons"]:
        print(f"[verdict]   why not flag_on: {reason}")
    print(f"[verdict] {verdict['note']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    cfg = load_config(CONFIG_PATH)
    validate_config(cfg)
    block = int(args.block if args.block is not None else cfg.mla_block_merge.block)
    if cfg.swa_window < block:
        raise SystemExit(
            f"swa_window {cfg.swa_window} < block {block}: the merged path requires the "
            f"window to cover the query's own partially filled block (validate_config)"
        )
    # The measured mechanism lives in the sparse path (ADR-009 v1.5); the dense
    # oracle cannot be allocated at 64K at all (module docstring, limit 3).
    active_cfg = dataclasses.replace(cfg, attn_dense_reference=False)
    print(f"[bench] config {CONFIG_PATH} sha256 {_sha256(CONFIG_PATH)[:16]}… "
          f"declared block merge {cfg.mla_block_merge} (pinned off)")
    print(f"[bench] active overrides: attn_dense_reference=False (64K sparse path), "
          f"use_attnres={bool(args.attnres)}, head_vocab={args.head_vocab}")
    print(f"[bench] gpu: {_gpu_name()}")
    print(f"[bench] jax {jax.__version__} devices {jax.devices()}")

    length = int(args.length)
    sequence = build_needle_sequence(length, seed=args.seed, vocab=args.head_vocab)
    print(f"[bench] sequence: {length} records, seed {args.seed}, "
          f"{len(sequence.probes)} probes at {list(POSITION_FRACTIONS)} of the length")

    key = jr.PRNGKey(0)
    params = jax.jit(lambda k: model.init_params(k, cfg))(key)
    jax.block_until_ready(params)
    print(f"[bench] params initialised (pinned l3-full init, PRNGKey(0))", flush=True)

    # One forward per leg; the cost arm is then timed on the MLA stack fed with
    # the real activations captured here (the same inputs for both legs).
    dtype = {"bf16": jnp.bfloat16, "fp32": jnp.float32}[args.dtype]
    model_params = cast_params(_params_with_head_slice(params, args.head_vocab), dtype)
    del params  # the fp32 draw is not needed again; the cast tree is the model
    ids = sequence.token_ids[None, :]
    if int(ids.max()) >= args.head_vocab:
        raise SystemExit(f"token id {int(ids.max())} exceeds the head slice {args.head_vocab}")
    form = capture_form(model_params, active_cfg, ids, use_attnres=bool(args.attnres),
                        form=args.capture_form)
    need = _reference_form_bytes(active_cfg, length, int(model_params.embedding.dtype.itemsize))
    budget = _device_budget_bytes()
    budget_text = f"{budget / 2 ** 30:.1f} GiB" if budget else "unreported"
    print(f"[bench] capture form: {form} (reference form would need ~{need / 2 ** 30:.1f} GiB, "
          f"device budget {budget_text})", flush=True)
    if form == "loop" and need > 8 * 2 ** 30:
        print(f"[bench] WARNING: the reference form is not the memory-safe one at this length "
              f"(~{need / 2 ** 30:.1f} GiB); --capture-form scan forces the other form",
              flush=True)
    with declared_switch(False, block):
        captures: dict[str, tuple] = {"off": capture_leg(
            active_cfg, model_params, ids, use_attnres=bool(args.attnres), form=form,
            label="flag off")}
    if args.recall_source == "e2e":
        with declared_switch(True, block):
            captures["on"] = capture_leg(active_cfg, model_params, ids,
                                         use_attnres=bool(args.attnres), form=form,
                                         label="flag on")
    else:
        captures["on"] = captures["off"]

    recall = None
    if not args.skip_recall:
        recall = measure_recall(active_cfg, model_params, sequence, captures, block=block,
                                recall_source=args.recall_source)
        _print_recall(recall, block)

    cost = None
    if not args.skip_cost:
        cost = measure_cost(active_cfg, mla_layer_params(model_params), captures["off"][1], block=block,
                            rounds=int(args.rounds), inner=int(args.inner))
        _print_cost(cost)

    verdict = decide(recall, cost)
    _print_verdict(verdict)

    report = {
        "report": "ADR-018 block-wise token merging — 64K gap bench",
        "date": time.strftime("%Y-%m-%d"),
        "status": "measured",
        "gpu": _gpu_name(),
        "platform": [str(d) for d in jax.devices()],
        "jax_version": jax.__version__,
        "config": {
            "path": str(CONFIG_PATH.relative_to(ROOT)),
            "sha256": _sha256(CONFIG_PATH),
            "declared_mla_block_merge": dataclasses.asdict(cfg.mla_block_merge),
            "switch": (
                "runtime: a temporary copy of the pinned config with mla_block_merge flipped "
                "is read through net.config.CONFIG_PATH (spine AD-9 / C-035); the pinned "
                "net/config.json is not edited and stays off"
            ),
            "block": block,
            "mla_top_k": int(cfg.mla_top_k),
            "swa_window": int(cfg.swa_window),
            "mla_layer_modes": list(cfg.mla_layer_modes),
        },
        "bounds": {
            "capture_form": form,
            "capture_form_requested": args.capture_form,
            "capture_form_reference_need_gib": round(need / 2 ** 30, 2),
            "capture_form_device_budget_gib": (round(budget / 2 ** 30, 2) if budget else None),
            "capture_form_why": (
                "the capture is the model's forward either way; the form is a memory choice. "
                "'loop' is the model's layer-by-layer form, bit-equal to model.forward both "
                "eagerly and compiled, and the only form AttnRes can use; its traced chain "
                "keeps ~3.4 activation copies per layer live, so at 64K it needs ~15.3 GiB and "
                "dies on a 16 GB card (measured: XLA remat 'only reduced to 20.27GiB'). 'scan' "
                "runs the repeated four-layer units in one lax.scan (18 KDA layers: 247 KB/rec "
                "in the python chain vs 38 KB/rec in one scan) and fits; it is compiled by "
                "construction, so it is bit-equal to model.forward compiled, not eager — a "
                "compiled body lowers the RMSNorm reduction in a different association order "
                "than op-by-op. The bench consumes the capture compiled (capture_leg), so the "
                "numbers are on the model's own activations."
            ),
            "attn_dense_reference": False,
            "attnres": bool(args.attnres),
            "head_vocab": int(args.head_vocab),
            "why": (
                "64K on a 16 GB card: the dense oracle needs (B,H,T,T) fp32 ~206 GB; the tied "
                "160K head needs (1,T,160000) fp32 42 GB; the AttnRes source stack needs "
                "25 x (1,T,1536) fp32 plus a copy for the normalised keys ~20 GB. None of the "
                "three is part of the measured mechanism (the MLA indexer selection)."
            ),
        },
        "recall": recall,
        "cost": cost,
        "verdict": verdict,
        "caveats": [
            "The two recall arms do not attend the same number of records: the unmerged leg "
            "reads mla_top_k records, the merged leg mla_top_k blocks (= block times as many "
            "records). recall_on_equal_budget is the merged arm at the same record budget, and "
            "the chance columns are the coverage of each budget — read the recall table "
            "together with them.",
            "Weights are at initialisation: no trained L3 checkpoint exists in the case, so the "
            "recall arm measures selection coverage of the needle position, not learned "
            "retrieval. The needle rank and the chance coverage in this report let a reader "
            "see whether the instrument had a signal to lose.",
            "The recall arm hands both modes identical activations (isolated source): that is "
            "the delta's own idiom and isolates the mechanism from the projections feeding it.",
            "The cost arm measures the MLA stack at 64K — the object ADR-018 changes — not the "
            "full forward: 18 of the 24 layers are KDA and the dense MoE path is untouched by "
            "the flag, so a whole-model ratio would dilute the effect it is meant to measure.",
        ],
        "runs": {
            "command": " ".join(sys.argv),
            "wall_seconds": time.time() - started,
            "skip_recall": bool(args.skip_recall),
            "skip_cost": bool(args.skip_cost),
            "shards_or_caches_written_to_repo": False,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[bench] report written to {out} ({out.stat().st_size} bytes, "
          f"wall {report['runs']['wall_seconds']:.1f}s)")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--length", type=int, default=DEFAULT_LENGTH,
                    help=f"sequence length in records (default {DEFAULT_LENGTH} = 64K gap)")
    ap.add_argument("--seed", type=int, default=0, help="PRNG seed of the sequence")
    ap.add_argument("--block", type=int, default=None,
                    help="merged-block width (default: the declaration in net/config.json)")
    ap.add_argument("--head-vocab", type=int, default=DEFAULT_HEAD_VOCAB,
                    help="rows of the pinned embedding kept as the output head (memory bound)")
    ap.add_argument("--rounds", type=int, default=cost_method.DEFAULT_ROUNDS,
                    help="ADR-015 rounds (the first is dropped)")
    ap.add_argument("--inner", type=int, default=cost_method.DEFAULT_INNER_REPEATS,
                    help="inner repeats per round (the round value is their minimum)")
    ap.add_argument("--recall-source", choices=("isolated", "e2e"), default="isolated",
                    help="'isolated': one capture, both selections on identical activations; "
                         "'e2e': one capture per leg")
    ap.add_argument("--attnres", action="store_true",
                    help="run the forward with AttnRes (only tractable at a small --length)")
    ap.add_argument("--skip-recall", action="store_true")
    ap.add_argument("--skip-cost", action="store_true")
    ap.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument("--capture-form", choices=("auto", "loop", "scan"), default="auto",
                    help="which capture form to run: 'auto' picks the reference (loop) form "
                         "when its estimated live set fits the device budget, else the scan "
                         "form; 'loop' forces the model's layer-by-layer form (needs ~15.3 GiB "
                         "at 64K), 'scan' forces the lax.scan form (memory-safe, compiled)")
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--smoke", action="store_true",
                    help="small length and few rounds — a dry run of the whole path")
    args = ap.parse_args(argv)
    jax_preflight.gate_or_exit()  # ADR-041: состояние стенда до реального прогона
    if args.smoke:
        args.length = min(args.length, 2048)
        args.rounds = min(args.rounds, 3)
        args.inner = min(args.inner, 1)
        args.out = str(Path(tempfile.gettempdir()) / "block-merge-bench-smoke.json")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
