#!/usr/bin/env python3
"""MFU ceiling ladder L0..L4 for the GB10 (DGX Spark) — where the MFU is lost.

The ladder walks from a pure GEMM (the measured cuBLAS bf16 peak, i.e. the MFU
denominator itself) to a real training step, one compute profile per rung.  Each
rung reports analytic FLOPs, wall time and MFU, so the drop of MFU from rung to
rung is attributed to a *class* of work instead of to a single total.

    L0  pure jnp.matmul (bf16 + fp32)          — the XLA-path ceiling
    L1  dense-FFN block, fwd+bwd               — matmul-dominated, large N
    L2  LatentMoE block, fwd+bwd               — small batched GEMMs + gather
    L3  single attention layers (KDA, MLA)     — sequential fine-grained kernels
    L4  a full forward+backward step of net/   — assembly (six-ND convention)

Two invariants are enforced in code, not in prose:

1. **The MFU denominator is read from ``evidence/kpi-pins.json``** and nowhere
   else.  If the pin file (or a measured peak in it) is missing the tool
   *fails closed* — the declared/theoretical peak is never substituted
   (``tools/sensors/derived_mfu.py``, ``peak_declared_not_measured``).

2. **No CUDA device → every rung is ``EMPTY-PENDING`` with a reason.**  A CPU
   run is only ever produced behind the explicit ``--allow-cpu`` opt-in and its
   rows say so in ``notes``; a CPU number is never dressed up as a GB10 number.

FLOPs convention (used identically everywhere, and cross-checked by
``flops_from_jaxpr`` against the *compiled* program):

* one multiply-accumulate (MAC) is counted as 2 FLOPs;
* ``fwd_bwd=True`` counts 2 forward + 1 backward = **3x** the forward FLOPs
  (2x for the forward pass, 2x again for the backward pass, minus the forward
  already counted — the standard training-FLOPs convention, e.g. Chinchilla /
  PaLM / Megatron: ``6 * N * tokens`` for a dense transformer);
* elementwise work (activations, norms, softmax, decay) is **not** counted —
  it is O(tokens) with a small constant and does not move an MFU figure; each
  rung's ``notes`` states what is in and out of its count.

ADR-041 (memory discipline): ``XLA_PYTHON_CLIENT_MEM_FRACTION`` is set
**before** ``import jax`` below, with an operator-set value always winning
(``setdefault``).  The OOM incident of 08.10 (34 oom-kill, stand reboot) came
from a JAX process on a shared box with no explicit memory limit.

Run (one command, on the GB10)::

    python tools/mfu_ladder.py --levels L0,L1,L2,L3 --dtype bf16

Off-stand (no CUDA) the same command writes the report with every rung
EMPTY-PENDING; add ``--allow-cpu`` to get scaled CPU numbers for smoke only.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

# --------------------------------------------------------------------------- #
# Path bootstrap: the tool is meant to run as ``python tools/mfu_ladder.py``
# from the repository root, which puts ``tools/`` on sys.path instead of the
# root — the ``net/`` imports below then need the root added explicitly.
# --------------------------------------------------------------------------- #
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# --------------------------------------------------------------------------- #
# ADR-041: memory discipline BEFORE the JAX import (order matters — the XLA
# client reads the variable when the backend initialises, i.e. at `import jax`).
# --------------------------------------------------------------------------- #
MEM_FRACTION_ENV = "XLA_PYTHON_CLIENT_MEM_FRACTION"
DEFAULT_MEM_FRACTION = "0.5"


def ensure_mem_fraction(env: dict | None = None) -> str:
    """Pin the XLA memory fraction unless the operator already chose one."""
    target = os.environ if env is None else env
    return target.setdefault(MEM_FRACTION_ENV, DEFAULT_MEM_FRACTION)


MEM_FRACTION = ensure_mem_fraction()

import jax  # noqa: E402  (must follow the memory pin above)
import jax.numpy as jnp  # noqa: E402

SCHEMA = "mfu-ladder/1"
DEFAULT_PINS = REPO / "evidence" / "kpi-pins.json"
DEFAULT_OUT = REPO / "evidence" / "mfu-ladder" / "ladder-report.json"
DEFAULT_L4_CONFIG = REPO / "net" / "config-dense124m.json"

LEVEL_NAMES = ("L0", "L1", "L2", "L3", "L4")

# Measurement protocol (TASK.md requirement 2): >=3 warmup iterations, median of
# >=10 timed iterations, every timed window closed by a device synchronisation.
PROTOCOL_WARMUP = 3
PROTOCOL_ITERS = 10

# Token counts pinned by the task (the l3-full step window).
TOKENS = (8192, 32768)
T_SEQ = 8192


class DenominatorUnavailable(RuntimeError):
    """The measured MFU denominator is not available — fail closed."""


# --------------------------------------------------------------------------- #
# Device probe
# --------------------------------------------------------------------------- #
def has_cuda() -> bool:
    """True when JAX sees at least one GPU device."""
    try:
        return any(getattr(d, "platform", "") == "gpu" for d in jax.devices())
    except Exception:  # pragma: no cover - backend init failure is "no device"
        return False


def _block(tree: Any) -> None:
    """Close a timed window on the device (requirement 2: synchronise)."""
    jax.block_until_ready(tree)


# --------------------------------------------------------------------------- #
# MFU denominator — pinned, never declared
# --------------------------------------------------------------------------- #
def load_denominator(path: str | Path = DEFAULT_PINS) -> dict:
    """Read the measured peak(s) from the pin file.  Fails closed.

    Returns ``{"bf16": 98.2, "fp32": 45.2, "source": ...}``.  The theoretical /
    declared peak is deliberately *not* a fallback: MFU computed from a
    declaration is the ``peak_declared_not_measured`` error the MFU sensor
    (S-027) exists to prevent.
    """
    path = Path(path)
    if not path.is_file():
        raise DenominatorUnavailable(f"pin-файл знаменателя MFU не найден: {path}")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DenominatorUnavailable(f"pin-файл нечитаем: {path}: {exc}") from exc

    ref = payload.get("mfu_reference") or {}
    used = ref.get("peak_used") or {}
    measured = ref.get("measured_denominator") or {}

    bf16 = used.get("bf16_dense_tflops_measured")
    if bf16 is None:
        bf16 = measured.get("denominator_bf16_best")
    fp32 = used.get("fp32_tflops_measured")
    if fp32 is None:
        fp32 = measured.get("denominator_fp32_best")

    out: dict[str, Any] = {}
    for name, value in (("bf16", bf16), ("fp32", fp32)):
        if value is None:
            continue
        value = float(value)
        if not value > 0:
            raise DenominatorUnavailable(f"знаменатель {name} неположителен: {value}")
        out[name] = value

    if "bf16" not in out:
        raise DenominatorUnavailable(
            f"в {path} нет ИЗМЕРЕННОГО bf16-пика (measured); объявленный пик не подставляется"
        )
    out["source"] = (
        f"{path.name}: mfu_reference.peak_used (измеренный) · "
        f"лог {ref.get('measured_denominator', {}).get('source_log', 'н/д')}"
    )
    return out


def mfu_pct(tflops: float, denominator_tflops: float) -> float:
    """MFU in percent of the measured peak."""
    if denominator_tflops <= 0:
        raise ValueError("знаменатель MFU должен быть положительным")
    return 100.0 * tflops / denominator_tflops


# --------------------------------------------------------------------------- #
# Analytic FLOPs model (MAC x 2).  One function per rung, documented.
# --------------------------------------------------------------------------- #
def flops_matmul(M: int, K: int, N: int, batch: int = 1) -> int:
    """L0: one GEMM.  MACs = M*K*N per matrix, so FLOPs = 2*M*K*N."""
    return 2 * M * K * N * batch


def _fwd_bwd(macs: int, fwd_bwd: bool) -> int:
    """MACs -> FLOPs, x3 when the backward pass is included (see module docstring)."""
    return 2 * macs * (3 if fwd_bwd else 1)


def dense_ffn_macs_per_token(cfg, inter: int | None = None) -> int:
    """L1: dense SiTU-GLU MLP — three matmuls per token, ``3 * hidden * inter``.

    ``x @ W_g`` (hidden->inter), ``x @ W_u`` (hidden->inter),
    ``a @ W_down`` (inter->hidden).  The SiTU-GLU activation itself is
    elementwise and is not counted.

    ``inter`` defaults to the task's 4x expansion (``4 * hidden`` = 6144, the
    width the L0 GEMM shapes also use).  Note the *model's* own dense layer-0
    MLP is narrower: ``cfg.mlp_intermediate = 4096`` (~2.67x).  Pass
    ``inter=cfg.mlp_intermediate`` to score the layer as ``net/`` builds it —
    which is also what makes the parameter-tree cross-check exact.
    """
    inter = (4 * cfg.hidden) if inter is None else int(inter)
    return 3 * cfg.hidden * inter


def flops_dense_ffn(tokens: int, cfg, *, inter: int | None = None, fwd_bwd: bool = False) -> int:
    """L1 FLOPs over ``tokens`` tokens, fwd only or fwd+bwd (x3)."""
    return _fwd_bwd(tokens * dense_ffn_macs_per_token(cfg, inter), fwd_bwd)


def moe_macs_per_token(cfg, *, active: bool = False) -> int:
    """L2: Stable LatentMoE of ``net/moe.py``.

    ``active=False`` (the default) counts what the **hardware executes**:
    ``net/moe.py:apply`` builds ``(N, n_routed, expert_inter)`` with an einsum
    over *every* routed expert and only then gathers top-k via
    ``take_along_axis`` — so the routed experts are paid for in full, not
    ``top_k`` of them.  ``active=True`` counts the algorithmically necessary
    work (top-k experts); reporting both separates "the algorithm is sparse"
    from "the kernel is dense", which is exactly what L2 is asked to decide.

    Terms: latent down/up projections, routed experts, shared experts
    (full width), router.
    """
    hid = cfg.hidden
    lat = cfg.moe_latent_dim
    ei = cfg.moe_expert_intermediate
    si = cfg.moe_shared_intermediate
    nr = cfg.moe_num_routed
    ns = cfg.moe_num_shared
    expert_terms = cfg.moe_top_k if active else nr
    return (
        2 * hid * lat  # W_down, W_up (latent projections)
        + expert_terms * 3 * lat * ei  # routed experts (3 matmuls each)
        + ns * 3 * hid * si  # shared experts on the full-width path
        + hid * nr  # router scores
    )


def flops_moe_ffn(tokens: int, cfg, *, active: bool = False, fwd_bwd: bool = False) -> int:
    """L2 FLOPs.  ``active`` selects algorithmic (top-k) vs executed (all-expert)."""
    return _fwd_bwd(tokens * moe_macs_per_token(cfg, active=active), fwd_bwd)


def scan_combines_per_chunk(chunk: int) -> int:
    """Combine operations ``jax.lax.associative_scan`` lowers to, for ``chunk``.

    ``associative_scan`` is not ``chunk - 1`` combines: JAX lowers it as a
    two-pass (reduce then sweep) tree, which measures as
    ``2*C - log2(C) - 2`` combines for a chunk of ``C``.  This is a property of
    the lowering, not of the delta rule — but it is what the hardware executes,
    and MFU must be scored on executed FLOPs, so it is modelled explicitly.
    """
    c = max(int(chunk), 2)
    return 2 * c - (c.bit_length() - 1) - 2


def kda_macs_per_token(cfg, tokens: int = T_SEQ, chunk: int | None = None) -> dict[str, int]:
    """L3a: KDA (chunked delta rule) of ``net/kda.py`` — MACs per token by term.

    Terms (``dk`` = ``dv`` = 128, ``H`` = 12, ``r`` = decay rank, ``K`` = the
    ShortConv kernel width, ``C`` = chunk size, ``T`` = sequence length):

    ``proj``    q/k/v/g/beta projections, the low-rank decay projection
                (``W_a_down`` then ``W_a_up``) and the output ``W_o``.
    ``conv``    depthwise ShortConv over q/k/v: ``K`` taps x ``H*dk`` channels.
    ``scan``    the chunked form's affine-transition monoid
                (``M_t = Diag(alpha) - beta k (alpha k)^T``, ``N_t = beta k v^T``)
                composed by ``jax.lax.associative_scan``.  Each combine is
                ``H*(dk^3 + dk^2*dv)`` MACs (``M2@M1`` and ``M2@N1``); there are
                ``scan_combines_per_chunk(C)`` of them per chunk.  **This term
                dominates the layer** — it is the ``O(T*H*dk*dk)`` kit the
                module docstring names, and the reason the pinned ``chunked``
                KDA is expensive; the WY/UT form (``apply_wyut``, ADR-031)
                replaces it with a ``C x C`` correction and does not pay dk^3.
    ``pair``    ``P^T q`` per token (``H*dk*dk``) and the state read
                ``S0^T (P^T q)`` (``H*dk*dv``).
    ``intra``   ``Q^T q`` per token (``H*dk*dv``).
    ``carry``   the once-per-chunk state transfer ``P[-1] @ S0`` (``H*dk^2*dv``
                per chunk, i.e. divided by ``C`` per token).
    ``window``  the sliding-window branch, only when the window is on.

    The state terms are the algorithmic core of the chunked form.  They are not
    trusted: ``tools/tests/test_mfu_ladder.py`` re-derives them from the
    compiled program (``flops_from_jaxpr``) and fails if they drift >10%.
    """
    hid = cfg.hidden
    H, dk, dv = cfg.num_heads, cfg.kda_dk, cfg.kda_dv
    r = cfg.kda_decay_rank
    K = cfg.kda_short_conv_kernel
    C = int(chunk or getattr(cfg, "kda_wyut_chunk", 64))

    proj = (
        hid * H * dk  # W_q
        + hid * H * dk  # W_k
        + hid * H * dv  # W_v
        + hid * hid  # W_g (output gate)
        + hid * H  # W_beta
        + hid * r  # W_a_down (decay, low rank)
        + r * H * dk  # W_a_up
        + H * dv * hid  # W_o (output projection)
    )
    conv = 3 * K * H * dk
    combine = H * (dk * dk * dk + dk * dk * dv)
    scan = combine * scan_combines_per_chunk(C) // C
    pair = H * dk * dk + H * dk * dv
    intra = H * dk * dv
    carry = H * dk * dk * dv // C
    window = 0
    if getattr(cfg, "swa_window", 0) > 0:
        W = cfg.swa_window
        window = W * H * (dk + dv)

    return {
        "proj": proj,
        "conv": conv,
        "scan": scan,
        "pair": pair,
        "intra": intra,
        "carry": carry,
        "window": window,
    }


def flops_kda(tokens: int, cfg, *, fwd_bwd: bool = False, chunk: int | None = None) -> int:
    """L3a FLOPs = MACs x 2 (x3 when fwd+bwd)."""
    return _fwd_bwd(tokens * sum(kda_macs_per_token(cfg, tokens, chunk).values()), fwd_bwd)


def mla_macs_per_token(cfg, tokens: int = T_SEQ) -> dict[str, int]:
    """L3b: Gated MLA of ``net/mla.py`` — MACs per token by term.

    Two paths, selected by ``cfg.attn_dense_reference`` (the pinned oracle):

    ``proj``   latent compression ``W_c``, the up-projections ``W_k_up`` /
               ``W_v_up`` from the latent, the query ``W_q``, the gate ``W_g``
               and the output ``W_o``.
    ``scores`` / ``values``  the attention itself.  Dense oracle: ``T`` records
               per query, so ``T*H*dq`` MACs each — the ``O(T^2)`` term that a
               long-context step cannot avoid.  Sparse path: the same with
               ``top_k`` in place of ``T``, plus the indexer.
    ``pool``   the ADR-012 candidate pool / reindex scoring, when enabled.

    **The MLA layer follows the compute-dtype gate** (``net/compute_dtype.py``,
    since ``net/mla.py`` was routed through ``gemm``): with
    ``AXIOM_COMPUTE_DTYPE`` off (the default, ``fp32``) every GEMM is the pinned
    byte-exact fp32 expression, with the gate on (``bf16``) the *operands* are
    cast to bf16 at the boundary and the product accumulates in fp32.  The
    FLOPs counted here are the same either way — dtype changes the rate, not the
    MAC shape — so the rung's honest denominator is the pin of the mode it
    actually ran (see ``_gate_mode`` and the ``env=`` of the L3b case).
    """
    hid = cfg.hidden
    lat = cfg.mla_latent_dim
    H, dq = cfg.num_heads, cfg.mla_head_dim
    T = tokens

    proj = (
        hid * lat  # W_c (compress)
        + lat * H * dq  # W_k_up
        + lat * H * dq  # W_v_up
        + hid * H * dq  # W_q
        + hid * hid  # W_g
        + H * dq * hid  # W_o
    )
    if bool(getattr(cfg, "attn_dense_reference", True)):
        width = T
        idx = 0
        pool = 0  # the oracle is dense: the ADR-012 pool is not executed at all
    else:
        width = min(int(getattr(cfg, "mla_top_k", T)), T)
        hi, di = cfg.mla_index_heads, cfg.mla_index_dim
        idx = T * hi * di + hid * hi * di  # indexer: keys + queries
        pool = 0
        if getattr(cfg, "mla_pool_size", 0):
            pool = int(cfg.mla_pool_size) * int(cfg.mla_pool_block) * hi * di
    scores = width * H * dq
    values = width * H * dq

    return {
        "proj": proj,
        "scores": scores,
        "values": values,
        "indexer": idx,
        "pool": pool,
    }


def flops_mla(tokens: int, cfg, *, fwd_bwd: bool = False) -> int:
    """L3b FLOPs = MACs x 2 (x3 when fwd+bwd)."""
    return _fwd_bwd(tokens * sum(mla_macs_per_token(cfg, tokens).values()), fwd_bwd)


def flops_dense_model(cfg, tokens: int, fwd_bwd: bool = True) -> int:
    """L4: the project's own six-ND convention — ``6 * N_active * tokens``.

    Identical to ``tools/sensors/derived_mfu.py`` (``S-027``), so the ladder's
    top rung and the project's MFU sensor read the same number.
    """
    import net.model as model

    active = model.active_param_count(cfg)
    return (6 if fwd_bwd else 2) * active * tokens


# --------------------------------------------------------------------------- #
# Independent FLOPs count: from the *compiled* program (jaxpr)
# --------------------------------------------------------------------------- #
def _shape_of(v) -> tuple[int, ...]:
    return tuple(v.aval.shape) if hasattr(v, "aval") else tuple(getattr(v, "shape", ()))


def _dot_general_macs(eqn) -> int:
    """MACs of one ``dot_general``: (batch x free-lhs x free-rhs) x contracted."""
    lhs, rhs = _shape_of(eqn.invars[0]), _shape_of(eqn.invars[1])
    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = eqn.params["dimension_numbers"]
    contract = 1
    for d in lhs_contract:
        contract *= lhs[d]
    if not contract:
        return 0
    out = 1
    for d, s in enumerate(lhs):
        if d not in lhs_contract and d not in lhs_batch:
            out *= s
    for d, s in enumerate(rhs):
        if d not in rhs_contract and d not in rhs_batch:
            out *= s
    batch = 1
    for d in lhs_batch:
        batch *= lhs[d]
    return out * contract * batch


def _flops_jaxpr(jaxpr) -> int:
    """Sum dot-products in a jaxpr, recursing into scan/remat bodies."""
    total = 0
    for eqn in jaxpr.eqns:
        name = eqn.primitive.name
        if name == "dot_general":
            total += _dot_general_macs(eqn)
        elif name == "conv_general_dilated":
            # depthwise/standard conv: MACs = out elems x kernel footprint x in ch
            lhs = _shape_of(eqn.invars[0])
            out_shape = _shape_of(eqn.outvars[0])
            ksize = _shape_of(eqn.invars[1])
            reps = eqn.params.get("rhs_dilation", (1,) * max(len(ksize) - 2, 1))
            kern = 1
            for i, k in enumerate(ksize[:-2] or ksize):
                kern *= k * (reps[i] if i < len(reps) else 1)
            total += kern * lhs[-1] * (out_shape[-2] if len(out_shape) >= 2 else 1) * out_shape[-1]
        elif name in ("scan", "associative_scan"):
            sub = eqn.params.get("jaxpr")
            if sub is None:
                continue
            if name == "scan":
                n = eqn.params.get("length")
            else:
                axis = eqn.params.get("axis", 0)
                shape = _shape_of(eqn.invars[0])
                n = (shape[axis] - 1) if shape else None
            if n is None:
                continue
            total += int(n) * _flops_jaxpr(sub.jaxpr)
        elif name.startswith(("remat", "checkpoint")) or name in ("pjit", "jit", "xla_pmap"):
            # ``jax.checkpoint`` lands here as ``remat2`` on current JAX; recurse
            # into its body rather than dropping the whole recomputed chunk.
            sub = eqn.params.get("jaxpr") or eqn.params.get("call_jaxpr")
            if sub is not None:
                total += _flops_jaxpr(getattr(sub, "jaxpr", sub))
        elif name in ("cond", "while"):
            # not used by the blocks under test; counted as zero and flagged by
            # the caller (the analytic model is the primary source of truth).
            continue
    return total


def flops_from_jaxpr(fn: Callable[..., Any], *args) -> int:
    """MACs x 2 of the *compiled* program, summed over its dot-products.

    This is the independent count the rollback criterion asks for: it reads the
     program JAX actually builds (projections, attention scores, the chunked
     KDA scan, autodiff's backward dots) instead of restating the analytic
     formula.  Divergence between the two is a bug in the ladder, not a
     property of the hardware.
    """
    closed = jax.make_jaxpr(fn)(*args)
    return 2 * _flops_jaxpr(closed.jaxpr)


# --------------------------------------------------------------------------- #
# Measurement harness
# --------------------------------------------------------------------------- #
def bench(fn: Callable[[], Any], *, iters: int = PROTOCOL_ITERS, warmup: int = PROTOCOL_WARMUP) -> dict:
    """Compile, warm up, then take the median of ``iters`` synchronised runs.

    Returns ``seconds`` (median), ``samples``, ``warmup_s`` and ``compile_s``
    (first call = tracing + compilation) as separate fields, per requirement 2.
    """
    t0 = time.perf_counter()
    out = fn()
    _block(out)
    compile_s = time.perf_counter() - t0

    tw = time.perf_counter()
    for _ in range(max(0, int(warmup))):
        out = fn()
    _block(out)
    warmup_s = time.perf_counter() - tw

    samples: list[float] = []
    for _ in range(int(iters)):
        t0 = time.perf_counter()
        out = fn()
        _block(out)
        samples.append(time.perf_counter() - t0)

    seconds = statistics.median(samples) if samples else float("nan")
    return {
        "seconds": seconds,
        "samples": samples,
        "warmup_s": warmup_s,
        "compile_s": compile_s,
        "iters": int(iters),
    }


def bench_matmul(dtype, M: int, K: int, N: int, *, iters: int = PROTOCOL_ITERS, warmup: int = PROTOCOL_WARMUP) -> dict:
    """The L0 primitive: a single jitted ``jnp.matmul`` on ``(M,K)x(K,N)``."""
    a = jax.random.normal(jax.random.PRNGKey(0), (M, K)).astype(dtype)
    b = jax.random.normal(jax.random.PRNGKey(1), (K, N)).astype(dtype)
    f = jax.jit(lambda x, y: x @ y)
    res = bench(lambda: f(a, b), iters=iters, warmup=warmup)
    res["tflops"] = flops_matmul(M, K, N) / res["seconds"] / 1e12
    return res


# --------------------------------------------------------------------------- #
# Rung builders
# --------------------------------------------------------------------------- #
@dataclass
class Case:
    """One measurable rung instance (a level may emit several)."""

    level: str
    shape: str
    fn: Callable[[], Any]
    flops: int
    dtype: str
    compute_dtype: str
    flops_active: int | None = None
    notes: str = ""
    requires_cuda: bool = False
    #: Environment a gate-aware rung must run under (e.g. ``AXIOM_COMPUTE_DTYPE``).
    #: Applied around the cell's own compile+bench window, restored afterwards.
    env: dict[str, str] | None = None


def _l3_cfg():
    import net.config as config

    return config.load_config(str(REPO / "net" / "config.json"))


def _fwd_bwd_fn(params, cfg, apply, x):
    """A jitted forward+backward over **both** params and input.

    ``argnums=(0, 1)`` matters for the FLOPs accounting: differentiating w.r.t.
    the parameters alone elides the input-gradient matmuls (``dx = dg @ W_g^T +
    du @ W_u^T``), leaving a program that is ~2.33x the forward instead of the
    3x the training convention (and the analytic model here) assumes.  A layer
    inside a stack pays for both, so both are differentiated.
    """
    def loss(p, xx):
        out = apply(p, cfg, xx)
        return jnp.sum(out.astype(jnp.float32) ** 2)

    grad = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))
    return lambda: grad(params, x)


def _cast_tree(tree, dtype):
    return jax.tree_util.tree_map(
        lambda a: a.astype(dtype) if hasattr(a, "dtype") and a.dtype == jnp.float32 else a,
        tree,
    )


def build_level0(dtype: str, *, scale: int = 1, cfg=None) -> list[Case]:
    """L0 — pure GEMM ceiling of the XLA path, bf16 and fp32 for comparison."""
    shapes = [
        (4096, 4096, 4096),
        (8192, 1536, 1536),
        (8192, 1536, 6144),
        (1024, 1536, 1536),
    ]
    cases: list[Case] = []
    for dt, jdt in (("bf16", jnp.bfloat16), ("fp32", jnp.float32)):
        for M, K, N in shapes:
            # the batched form is carried by its own row for clarity
            batch = 8 if (M, K, N) == (1024, 1536, 1536) else 1
            m, k, n = M // scale, K // scale, N // scale
            m, k, n = max(m, 8), max(k, 8), max(n, 8)
            shape = f"{m}x{k}x{n}" + (f"x{batch}" if batch > 1 else "")
            cases.append(
                Case(
                    level="L0",
                    shape=shape,
                    fn=_matmul_runner(jdt, m, k, n, batch),
                    flops=flops_matmul(m, k, n, batch),
                    dtype=dt,
                    compute_dtype=dt,
                    notes=(
                        f"чистый jnp.matmul ({dt}); потолок XLA-пути для этого класса. "
                        "Замечание к чтению: знаменатель 98.2 снят на бо́льших формах "
                        "(65536x2048x8192 и 16384x1536x*, evidence/gemm_peak_0810.log), "
                        "поэтому на M=8192 ожидаемо ~85-90 TFLOPS — это форма, а не "
                        "потеря XLA-пути"
                    ),
                )
            )
    return cases


def _matmul_runner(jdt, m, k, n, batch=1):
    a = jax.random.normal(jax.random.PRNGKey(0), (batch, m, k)).astype(jdt)
    b = jax.random.normal(jax.random.PRNGKey(1), (batch, k, n)).astype(jdt)
    f = jax.jit(lambda x, y: x @ y)
    return lambda: f(a, b)


def build_level1(dtype: str, *, scale: int = 1, cfg=None) -> list[Case]:
    """L1 — dense FFN (hidden->inter->hidden) forward+backward.

    The rung is pinned to the task's 4x expansion (inter = 6144), so the block
    is built at that width rather than at the model's own
    ``cfg.mlp_intermediate`` (4096): a rung must measure the shape it declares,
    otherwise its MFU is inflated by the width ratio.
    """
    import dataclasses

    import net.mlp as mlp

    cfg = cfg or _l3_cfg()
    model_inter = cfg.mlp_intermediate  # the model's own dense layer-0 width
    cfg = dataclasses.replace(cfg, mlp_intermediate=4 * cfg.hidden)
    jdt = _jax_dtype(dtype)
    params = _cast_tree(mlp.init_mlp(jax.random.PRNGKey(0), cfg), jdt)
    cases = []
    for tokens in TOKENS:
        t = max(int(tokens) // scale, 16)
        x = jax.random.normal(jax.random.PRNGKey(1), (t, cfg.hidden)).astype(jdt)
        cases.append(
            Case(
                level="L1",
                shape=f"tokens={t} hidden={cfg.hidden} inter={cfg.mlp_intermediate}",
                fn=_fwd_bwd_fn(params, cfg, mlp.apply, x),
                flops=flops_dense_ffn(t, cfg, fwd_bwd=True),
                flops_active=flops_dense_ffn(t, cfg, fwd_bwd=True),
                dtype=dtype,
                compute_dtype=dtype,
                notes=(
                    "dense SiTU-GLU MLP fwd+bwd (x3); учтены 3 матмула, активация "
                    f"SiTU-GLU/нормы — элементные, не в счёте; inter=4*hidden="
                    f"{cfg.mlp_intermediate} (профиль задачи); собственный dense-MLP "
                    f"слоя-0 модели — cfg.mlp_intermediate={model_inter}"
                ),
            )
        )
    return cases


def build_level2(dtype: str, *, scale: int = 1, cfg=None) -> list[Case]:
    """L2 — LatentMoE forward+backward; active vs executed FLOPs reported."""
    import net.moe as moe

    cfg = cfg or _l3_cfg()
    jdt = _jax_dtype(dtype)
    params = _cast_tree(moe.init_moe(jax.random.PRNGKey(0), cfg), jdt)
    cases = []
    for tokens in TOKENS:
        t = max(int(tokens) // scale, 16)
        x = jax.random.normal(jax.random.PRNGKey(1), (t, cfg.hidden)).astype(jdt)
        cases.append(
            Case(
                level="L2",
                shape=(
                    f"tokens={t} hidden={cfg.hidden} latent={cfg.moe_latent_dim} "
                    f"routed={cfg.moe_num_routed} shared={cfg.moe_num_shared} "
                    f"top_k={cfg.moe_top_k} ei={cfg.moe_expert_intermediate}"
                ),
                fn=_fwd_bwd_fn(params, cfg, moe.apply, x),
                flops=flops_moe_ffn(t, cfg, active=False, fwd_bwd=True),
                flops_active=flops_moe_ffn(t, cfg, active=True, fwd_bwd=True),
                dtype=dtype,
                compute_dtype=dtype,
                notes=(
                    "LatentMoE fwd+bwd (x3); ИСПОЛНЕННЫЕ FLOPs = все "
                    f"{cfg.moe_num_routed} экспертов (einsum по всем, потом top-k "
                    f"gather) — mfu_pct на исполненных; mfu_pct_active = на "
                    f"top_k={cfg.moe_top_k}"
                ),
            )
        )
    return cases


def build_level3(dtype: str, *, scale: int = 1, cfg=None) -> list[Case]:
    """L3 — single attention layers: (a) KDA, (b) MLA, forward+backward at T=8192."""
    import net.compute_dtype as compute_dtype
    import net.kda as kda
    import net.mla as mla

    cfg = cfg or _l3_cfg()
    jdt = _jax_dtype(dtype)
    t = max(T_SEQ // scale, 16)

    cases: list[Case] = []

    kparams = _cast_tree(kda.init_kda(jax.random.PRNGKey(0), cfg), jdt)
    kx = jax.random.normal(jax.random.PRNGKey(1), (t, cfg.hidden)).astype(jdt)

    def kda_apply(p, c, xx):
        return kda.apply_kda(p, c, xx)

    cases.append(
        Case(
            level="L3",
            shape=f"KDA T={t} H={cfg.num_heads} dk={cfg.kda_dk} dv={cfg.kda_dv} chunk={cfg.kda_wyut_chunk}",
            fn=_fwd_bwd_fn(kparams, cfg, kda_apply, kx),
            flops=flops_kda(t, cfg, fwd_bwd=True),
            flops_active=flops_kda(t, cfg, fwd_bwd=True),
            dtype=dtype,
            compute_dtype=dtype,
            notes=(
                f"KDA слой ({cfg.kda_impl}, chunked delta rule) fwd+bwd (x3); "
                "проекции + ShortConv + intra-chunk + inter-chunk monoid scan + "
                "state read; элементные (sigmoid/swish/l2/RMS) не в счёте"
            ),
        )
    )

    # MLA routes its GEMMs through the compute-dtype gate (net/compute_dtype.py,
    # since 556cf08): gate off is the pinned fp32 path verbatim, gate on casts
    # the *operands* to bf16 with an fp32 accumulator — the master weights stay
    # fp32 (the gate never touches a parameter leaf).  So the cell is
    # gate-aware like the other rungs: --dtype selects the gate for the cell's
    # own compile+bench window, and the row's denominator follows the mode it
    # actually measured (the pin is chosen from ``case.compute_dtype`` below).
    mode = _gate_mode(dtype)
    mparams = mla.init_mla(jax.random.PRNGKey(0), cfg)
    mx = jax.random.normal(jax.random.PRNGKey(1), (1, t, cfg.hidden)).astype(jnp.float32)

    def mla_apply(p, c, xx):
        return mla.apply(p, c, xx)

    cases.append(
        Case(
            level="L3",
            shape=(
                f"MLA T={t} latent={cfg.mla_latent_dim} H={cfg.num_heads} "
                f"dq={cfg.mla_head_dim} dense_reference={bool(cfg.attn_dense_reference)}"
            ),
            fn=_fwd_bwd_fn(mparams, cfg, mla_apply, mx),
            flops=flops_mla(t, cfg, fwd_bwd=True),
            flops_active=flops_mla(t, cfg, fwd_bwd=True),
            dtype=mode,
            compute_dtype=mode,
            env={compute_dtype.MODE_ENV: mode},
            notes=(
                f"MLA слой fwd+bwd (x3); режим {mode} через compute-dtype-гейт "
                f"({compute_dtype.MODE_ENV}): gate-off — побайтовый fp32-путь, "
                "gate-on — bf16-операнды + fp32-накопление; знаменатель строки — "
                "по режиму; attention O(T*T) по T-записям на запрос"
                if cfg.attn_dense_reference else
                f"MLA слой fwd+bwd (x3); sparse top-k путь; режим {mode} через "
                f"compute-dtype-гейт ({compute_dtype.MODE_ENV}); знаменатель — по режиму"
            ),
        )
    )
    return cases


def build_level4(dtype: str, *, scale: int = 1, cfg=None) -> list[Case]:
    """L4 — a full forward+backward step of the dense model from ``net/``.

    Uses the dense arm (``net/config-dense124m.json``): a KDA/MoE config cannot
    be scored by the six-ND convention the project's MFU sensor uses, because
    ``N_active`` legitimately ignores the sparse parameters.  GPU-only.
    """
    import net.config as config
    import net.model as model

    if cfg is None:
        cfg = config.load_config(str(DEFAULT_L4_CONFIG))
    jdt = _jax_dtype(dtype)
    tokens = max(TOKENS[0] // scale, 16)

    def run():
        # Parameters are built inside the runner: an EMPTY-PENDING rung (no
        # CUDA, or CPU without --allow-cpu) must not allocate the ~0.3B-parameter
        # dense model just to be marked pending.
        params = _cast_tree(model.init_params(jax.random.PRNGKey(0), cfg), jdt)
        ids = jnp.ones((1, tokens), dtype=jnp.int32)

        def step(p, xx):
            return model.compute_loss(p, cfg, xx, chunk_size=min(64, tokens))

        return jax.jit(jax.value_and_grad(step))(params, ids)

    return [
        Case(
            level="L4",
            shape=f"dense124m step tokens={tokens} layers={cfg.num_layers} hidden={cfg.hidden}",
            fn=run,
            flops=flops_dense_model(cfg, tokens, fwd_bwd=True),
            flops_active=flops_dense_model(cfg, tokens, fwd_bwd=True),
            dtype=dtype,
            compute_dtype=dtype,
            notes=(
                "полный fwd+bwd шаг плотной модели net/ (dense-арм); FLOPs по "
                "конвенции проекта 6*N_active*tokens (как S-027)"
            ),
            requires_cuda=True,
        )
    ]


_BUILDERS = {
    "L0": build_level0,
    "L1": build_level1,
    "L2": build_level2,
    "L3": build_level3,
    "L4": build_level4,
}


def _jax_dtype(name: str):
    table = {
        "bf16": jnp.bfloat16,
        "bfloat16": jnp.bfloat16,
        "fp32": jnp.float32,
        "float32": jnp.float32,
        "fp16": jnp.float16,
    }
    try:
        return table[str(name).lower()]
    except KeyError as exc:
        raise ValueError(f"неизвестный dtype: {name!r}; ожидается один из {sorted(table)}") from exc


def _gate_mode(dtype: str) -> str:
    """The ``AXIOM_COMPUTE_DTYPE`` mode a gate-aware rung runs under, from ``--dtype``.

    The gate (``net/compute_dtype.py``) knows exactly two modes, so a rung that
    cannot be expressed as one of them fails closed here rather than silently
    measuring one mode under the other mode's denominator.
    """
    value = str(dtype).strip().lower()
    if value in ("bf16", "bfloat16"):
        return "bf16"
    if value in ("fp32", "float32"):
        return "fp32"
    raise ValueError(
        f"--dtype={dtype!r} не выражается режимом compute-dtype-гейта (fp32 | bf16)"
    )


@contextlib.contextmanager
def _env_override(env: dict | None):
    """Apply ``env`` for the duration of the block, restoring the prior values.

    The compute-dtype gate is read when a graph is *traced*, so a cell that
    declares a mode must hold the variable over its own compile+bench window —
    and must not leak it to the next rung.
    """
    if not env:
        yield
        return
    saved = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def _empty_row(case: Case, reason: str, denom: float | None, denominator_name: str) -> dict:
    return {
        "level": case.level,
        "dtype": case.dtype,
        "compute_dtype": case.compute_dtype,
        "shape": case.shape,
        "flops": int(case.flops),
        "flops_active": int(case.flops_active if case.flops_active is not None else case.flops),
        "seconds": None,
        "tflops": None,
        "mfu_pct": None,
        "mfu_pct_active": None,
        "denominator_tflops": denom,
        "denominator_name": denominator_name,
        "warmup_s": None,
        "compile_s": None,
        "iters": 0,
        "status": "EMPTY-PENDING",
        "notes": f"{reason}; {case.notes}" if case.notes else reason,
    }


def run_levels(
    levels: Sequence[str],
    dtype: str,
    *,
    pins_path: str | Path = DEFAULT_PINS,
    out: str | Path = DEFAULT_OUT,
    iters: int = PROTOCOL_ITERS,
    warmup: int = PROTOCOL_WARMUP,
    allow_cpu: bool = False,
    cpu_scale: int = 8,
) -> dict:
    """Run the requested rungs and write the JSON report.

    On a machine without a CUDA device every rung is EMPTY-PENDING with the
    reason, unless ``allow_cpu`` is passed — and then the shapes are scaled by
    ``cpu_scale`` and each row says so.
    """
    requested = [str(l).upper() for l in levels]
    for name in requested:
        if name not in _BUILDERS:
            raise ValueError(f"неизвестный уровень {name!r}; известны {list(LEVEL_NAMES)}")

    pins = load_denominator(pins_path)
    cuda = has_cuda()
    cfg = None if not any(l in ("L1", "L2", "L3") for l in requested) else _l3_cfg()

    scale = cpu_scale if (allow_cpu and not cuda) else 1
    results: list[dict] = []
    for name in requested:
        builder = _BUILDERS[name]
        cases = builder(dtype, scale=scale, cfg=cfg)
        for case in cases:
            denom_name = case.compute_dtype
            denom = pins.get("bf16" if denom_name in ("bf16", "bfloat16") else "fp32")
            # Without CUDA a rung is EMPTY-PENDING unless --allow-cpu was given
            # explicitly AND the rung is CPU-feasible (L4 is not).
            if not cuda and not (allow_cpu and not case.requires_cuda):
                reason = (
                    "нет CUDA-устройства (L4 требует GB10)"
                    if case.requires_cuda
                    else "нет CUDA-устройства"
                )
                results.append(_empty_row(case, reason, denom, denom_name))
                continue

            # A gate-aware rung (L3-MLA) declares the compute-dtype mode it must
            # be traced under; hold it over the cell's compile+bench window only.
            with _env_override(case.env):
                res = bench(case.fn, iters=iters, warmup=warmup)
            tflops = case.flops / res["seconds"] / 1e12 if res["seconds"] else float("nan")
            active = case.flops_active if case.flops_active is not None else case.flops
            notes = case.notes
            if not cuda:
                notes = (
                    f"CPU-прогон (--allow-cpu, масштаб /{scale}; фактические формы — "
                    f"в поле shape) — числа НЕ сопоставимы с GB10; {notes}"
                )
            results.append(
                {
                    "level": case.level,
                    "dtype": case.dtype,
                    "compute_dtype": case.compute_dtype,
                    "shape": case.shape,
                    "flops": int(case.flops),
                    "flops_active": int(active),
                    "seconds": res["seconds"],
                    "tflops": tflops,
                    "mfu_pct": mfu_pct(tflops, denom) if denom else None,
                    "mfu_pct_active": (100.0 * (active / res["seconds"] / 1e12) / denom) if denom else None,
                    "denominator_tflops": denom,
                    "denominator_name": denom_name,
                    "warmup_s": res["warmup_s"],
                    "compile_s": res["compile_s"],
                    "iters": res["iters"],
                    "status": "OK",
                    "notes": notes,
                }
            )

    ok = [r for r in results if r["status"] == "OK"]
    report = {
        "schema": SCHEMA,
        "device": str(jax.devices()[0]) if jax.devices() else "none",
        "cuda": cuda,
        "allow_cpu": bool(allow_cpu and not cuda),
        "cpu_scale": scale,
        "dtype": dtype,
        "mem_fraction": {MEM_FRACTION_ENV: os.environ.get(MEM_FRACTION_ENV)},
        "denominator_bf16": pins.get("bf16"),
        "denominator_fp32": pins.get("fp32"),
        "denominator_source": pins.get("source"),
        "protocol": {"iters": iters, "warmup": warmup, "ok": iters >= PROTOCOL_ITERS and warmup >= PROTOCOL_WARMUP},
        "status": "ok" if ok else "empty-pending",
        "results": results,
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def format_table(report: dict) -> str:
    """Plain-text table for stdout (level, dtype, shape, FLOPs, s, TFLOPS, MFU%)."""
    head = (
        f"{'lvl':<4}{'dtype':<7}{'shape':<58}{'GFLOPs':>12}"
        f"{'sec':>9}{'TFLOPS':>9}{'MFU%':>7}{'MFU%act':>8}  status"
    )
    lines = [head, "-" * len(head)]
    for row in report["results"]:
        gflops = row["flops"] / 1e9
        sec = f"{row['seconds']:.4f}" if row["seconds"] is not None else "-"
        tf = f"{row['tflops']:.3f}" if row["tflops"] is not None else "-"
        mfu = f"{row['mfu_pct']:.2f}" if row["mfu_pct"] is not None else "-"
        mfua = f"{row['mfu_pct_active']:.2f}" if row.get("mfu_pct_active") is not None else "-"
        lines.append(
            f"{row['level']:<4}{row['dtype']:<7}{row['shape'][:57]:<58}{gflops:>12.1f}"
            f"{sec:>9}{tf:>9}{mfu:>7}{mfua:>8}  {row['status']}"
        )
    lines.append("")
    lines.append(
        f"знаменатель: bf16 {report['denominator_bf16']} TFLOPS, fp32 {report['denominator_fp32']} "
        f"TFLOPS (измеренный) — {report['denominator_source']}"
    )
    if report["status"] == "empty-pending":
        lines.append("нет CUDA-устройства: уровни EMPTY-PENDING (прогон на GB10 — архитектор)")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="MFU ceiling ladder L0..L4 (GB10)")
    ap.add_argument("--levels", default="L0,L1,L2,L3", help="через запятую: L0,L1,L2,L3,L4")
    ap.add_argument("--dtype", default="bf16", help="bf16 | fp32")
    ap.add_argument("--pins", default=str(DEFAULT_PINS), help="файл знаменателя MFU")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="JSON-отчёт")
    ap.add_argument("--iters", type=int, default=PROTOCOL_ITERS, help="таймируемых итераций (>=10)")
    ap.add_argument("--warmup", type=int, default=PROTOCOL_WARMUP, help="прогрев (>=3)")
    ap.add_argument(
        "--allow-cpu",
        action="store_true",
        help="явный опт-ин: гонять уменьшенные формы на CPU (смоук); по умолчанию без CUDA — EMPTY-PENDING",
    )
    ap.add_argument("--cpu-scale", type=int, default=8, help="делитель форм для --allow-cpu")
    args = ap.parse_args(argv)

    levels = [s.strip() for s in args.levels.split(",") if s.strip()]
    report = run_levels(
        levels,
        args.dtype,
        pins_path=args.pins,
        out=args.out,
        iters=args.iters,
        warmup=args.warmup,
        allow_cpu=args.allow_cpu,
        cpu_scale=args.cpu_scale,
    )
    print(format_table(report))
    print(f"\nотчёт: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
