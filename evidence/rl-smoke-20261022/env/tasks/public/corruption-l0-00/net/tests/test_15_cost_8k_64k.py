"""Acceptance criterion 16 — cost of the sparse+SWA path at 8K and 64K.

ADR-009 criterion 5: the forward on 8K must be no worse than 5% against the
dense path (time/memory); at 64K it must be strictly better.  The point of the
delta is that the MLA attention cost drops from ``O(T^2)`` to
``O(T * (top_k + n_win))``.

Measurement: this gate runs the **ADR-015 procedure** (``cost_method``): GPU
clocks put under control before the measurement (hard ``nvidia-smi -lgc`` pin,
or the user-level PowerMizer fallback, with the actual clocks printed), both legs
warmed up to a clock plateau, at least five counted rounds with the leg order
reversed every round and the first round dropped, the verdict on the **median**
with the **spread (min…max)** and the round count always in the output, and an
explicit **"не определён"** (neither PASS nor FAIL) when the spread exceeds 5% of
the measurement threshold or the clock state is not established.  Thresholds and
the metric are unchanged (ADR-015 p.6): that is the whole point of the ADR.

Honest limits of the 64K dense baseline (recorded in the report):

* The dense oracle materialises a ``(B, H, T, T)`` fp32 score tensor; at 64K
  with the skeleton's dims that is ~206 GB and cannot be allocated.  The 64K
  dense *time* is therefore measured on ``_dense_probe`` — a memory-bounded
  query-blocked implementation of the *same* causal math, verified equal to the
  oracle at small T — while the dense *memory* is computed from the oracle's
  activation shapes.  At 8K the real oracle is timed directly.
* A reduced-width cost config is used so that both paths are tractable in fp32;
  the structural ratio that matters (Q-head width ``H*dq`` vs indexer width
  ``Di``) is kept at 12:1, matching the skeleton (12 heads x 128 = 1536 vs
  indexer 128).
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from conftest import small_config

import cost_method
from cost_method import (
    THRESHOLD_8K,
    THRESHOLD_STRICTLY_BETTER,
    ClockControl,
    measure,
    verdict,
)

from net import mla

T8, T64 = 8192, 65536
DENSE_BLOCK = 256
SPARSE_BLOCK = 128  # attn_sparse._QUERY_BLOCK

# The skeleton runs six MLA layers (``num_mla_layers`` in ``net/config.json``).
STACK = 6


def cost_config():
    """Reduced-width config with the skeleton's Q-width : indexer-width = 12:1."""
    base = small_config()
    return dataclasses.replace(
        base,
        hidden=192, num_heads=6, head_dim=32,
        kda_dk=32, kda_dv=32, kda_decay_rank=32,
        mla_head_dim=32, mla_latent_dim=64,
        mla_index_heads=1, mla_index_dim=16,
        swa_window=128, mla_top_k=512, attn_dense_reference=False,
    )


def _dense_cfg(cfg):
    return dataclasses.replace(cfg, attn_dense_reference=True)


def _dense_probe(params, cfg, x, block: int = DENSE_BLOCK):
    """Memory-bounded causal attention: same math as the dense oracle."""
    x32 = x.astype(jnp.float32)
    H, dq = cfg.num_heads, cfg.mla_head_dim
    c = x32 @ params.W_c.astype(jnp.float32)
    q, k, v = mla._project(params, cfg, x32, c)
    B, T = q.shape[:2]
    scale = 1.0 / jnp.sqrt(jnp.asarray(dq, jnp.float32))
    neg = jnp.finfo(jnp.float32).min
    pos_all = jnp.arange(T)
    blk = min(block, T)
    n = -(-T // blk)

    def step(_, i):
        s = i * blk
        pos = s + jnp.arange(blk)
        pos_c = jnp.minimum(pos, T - 1)
        alive = pos < T
        qb = jnp.take(q, pos_c, axis=1)
        causal = pos_all[None, :] <= pos_c[:, None]
        sc = jnp.einsum("bqhd,bshd->bqhs", qb, k) * scale
        sc = jnp.where(causal[None, :, None, :], sc, neg)
        attn = jax.nn.softmax(sc, axis=-1)
        o = jnp.einsum("bqhs,bshd->bqhd", attn, v)
        return None, jnp.where(alive[None, :, None, None], o, 0.0)

    _, ys = jax.lax.scan(step, None, jnp.arange(n))
    o = jnp.transpose(ys, (1, 0, 2, 3, 4)).reshape(B, n * blk, H, dq)[:, :T]
    o = o.reshape(B, T, H * dq)
    gate = jax.nn.sigmoid(x32 @ params.W_g.astype(jnp.float32))
    return (gate * o) @ params.W_o.astype(jnp.float32)


def _stack_params(cfg, n: int, key=jr.PRNGKey(0)):
    """``n`` independent MLA layer parameter sets (one per skeleton MLA layer)."""
    return [mla.init_mla(k, cfg) for k in jr.split(key, n)]


def _stack(params_list, cfg, x, modes):
    """Run the MLA stack, threading the ADR-012 candidate pool between layers.

    Returns the last layer's output *array* (not the pool): the caller times it
    with ``block_until_ready``, which is a no-op on a ``None`` pool — timing the
    pool would measure nothing but dispatch.
    """
    pool = None
    out = None
    for p, mode in zip(params_list, modes):
        out, pool = mla.apply_with_pool(p, cfg, x, pool=pool, mode=mode)
    return out


def _dense_probe_stack(params_list, cfg, x):
    """Dense baseline for the 64K stack: the oracle cannot be allocated there."""
    out = None
    for p in params_list:
        out = _dense_probe(p, cfg, x)
    return out


def _dense_peak_bytes(B, H, T, dtype_bytes=4) -> int:
    """Oracle activation: the ``(B, H, T, T)`` scores and softmax tensors."""
    return 2 * B * H * T * T * dtype_bytes


def _sparse_peak_bytes(B, H, D, T, top_k, window, dtype_bytes=4) -> int:
    """Blocked-path activation: gathered KV + scores, bounded by the block."""
    q = min(SPARSE_BLOCK, T)
    u = min(top_k, T) + min(window, T)
    return 2 * (B * q * u * H * D * dtype_bytes) + (B * q * H * u * dtype_bytes) + (B * q * T * dtype_bytes)


def _judge_all(pairs: list[tuple[str, cost_method.Verdict]]) -> None:
    """Print every verdict, then decide the test's own state (ADR-015 p.5).

    A determined FAIL is a red criterion and wins (criterion 16 is a conjunction
    of its segments).  Only when nothing is red and something is "не определён"
    is the test neither PASS nor FAIL: ``pytest.skip`` is the one pytest state
    that is exactly that, and it carries the reason into ``-rA`` output.
    """
    for label, v in pairs:
        cost_method.print_verdict(v, label)
    failed = [(label, v) for label, v in pairs if v.status == "fail"]
    undetermined = [(label, v) for label, v in pairs if not v.determined]
    if failed:
        message = "; ".join(v.line(label) for label, v in failed)
        if undetermined:
            message += " | также не определены: " + ", ".join(label for label, _ in undetermined)
        raise AssertionError(message)
    if undetermined:
        message = "; ".join(v.line(label) for label, v in undetermined)
        determined = [label for label, v in pairs if v.determined]
        if determined:
            message += f" | определены (не красные): {', '.join(determined)}"
        pytest.skip(message)


def test_dense_probe_matches_oracle():
    """The 64K dense baseline is the same math as the oracle, not a new path."""
    cfg = cost_config()
    params = mla.init_mla(jr.PRNGKey(0), cfg)
    x = jr.normal(jr.PRNGKey(1), (1, 512, cfg.hidden))
    a = mla.apply(params, _dense_cfg(cfg), x)
    b = _dense_probe(params, cfg, x)
    assert float(jnp.max(jnp.abs(a - b))) <= 1e-4


def test_cost_8k_not_worse_and_64k_strictly_better():
    """One MLA layer, ADR-009 D2 (no pool): the pinned pre-ADR-012 baseline.

    Dense is the real oracle at 8K and the memory-bounded probe at 64K.
    """
    cfg = cost_config()
    params = mla.init_mla(jr.PRNGKey(0), cfg)
    dense_fn = jax.jit(lambda x: mla.apply(params, _dense_cfg(cfg), x))
    sparse_fn = jax.jit(lambda x: mla.apply(params, cfg, x))

    x8 = jr.normal(jr.PRNGKey(1), (1, T8, cfg.hidden))
    x64 = jr.normal(jr.PRNGKey(2), (1, T64, cfg.hidden))
    dense64_fn = jax.jit(lambda x: _dense_probe(params, cfg, x))

    label8 = "criterion 16 | one MLA layer | 8K"
    label64 = "criterion 16 | one MLA layer | 64K"
    with ClockControl() as clock:
        cost_method.print_clock(clock, label8)
        m8 = measure({"dense": lambda: dense_fn(x8), "sparse": lambda: sparse_fn(x8)},
                     label=label8, clock=clock)
        m64 = measure({"dense": lambda: dense64_fn(x64), "sparse": lambda: sparse_fn(x64)},
                      label=label64, clock=clock)

    H, D, B = cfg.num_heads, cfg.mla_head_dim, 1
    m_dense8 = _dense_peak_bytes(B, H, T8)
    m_sparse8 = _sparse_peak_bytes(B, H, D, T8, cfg.mla_top_k, cfg.swa_window)
    m_dense64 = _dense_peak_bytes(B, H, T64)
    m_sparse64 = _sparse_peak_bytes(B, H, D, T64, cfg.mla_top_k, cfg.swa_window)

    cost_method.print_measurement(m8, ref="dense", threshold=THRESHOLD_8K, unit_scale=1e3, unit="ms")
    cost_method.print_measurement(m64, ref="dense", threshold=THRESHOLD_STRICTLY_BETTER, unit_scale=1e3, unit="ms")
    print(f"[{label8}] memory: dense={m_dense8 / 1e6:.1f}MB sparse={m_sparse8 / 1e6:.1f}MB "
          f"(ratio {m_sparse8 / m_dense8:.4f}, threshold {THRESHOLD_8K})")
    print(f"[{label64}] memory: dense={m_dense64 / 1e9:.1f}GB sparse={m_sparse64 / 1e6:.1f}MB "
          f"(ratio {m_sparse64 / m_dense64:.4f}, threshold {THRESHOLD_STRICTLY_BETTER} strictly better)")

    # Memory is analytic and deterministic — asserted as before, no noise gate.
    assert m_sparse8 <= THRESHOLD_8K * m_dense8, f"8K memory sparse {m_sparse8} vs dense {m_dense8}"
    assert m_sparse64 < m_dense64, f"64K memory sparse {m_sparse64} vs dense {m_dense64}"

    _judge_all([
        (label8, verdict(m8.stats("sparse", "dense"), THRESHOLD_8K,
                         clock=clock, measurement=m8)),
        (label64, verdict(m64.stats("sparse", "dense"), THRESHOLD_STRICTLY_BETTER,
                          clock=clock, measurement=m64, strictly_better=True)),
    ])


def test_cost_8k_64k_mla_stack_shared_pool():
    """Criterion 16 on the MLA stack, with and without the ADR-012 pool.

    The same length, threshold (8K <= 1.05x) and timing method as the criterion,
    applied to the object the ADR-012 mechanism actually acts on: the stack of
    MLA layers.  ``before`` is six independent full-prefix selections (ADR-009
    D2, the pinned 2.05x baseline of the per-layer test), ``after`` is the
    declared layout — the first layer scans the prefix and publishes the pool,
    the other five re-score only the pool.  Both are compared against the dense
    oracle stack of the same depth, so the ratio is the delta's cost ratio.

    The pool's own memory is ``B * T * m`` indices (int32): 0.5 MB at 8K and
    4 MB at 64K for m = 16, i.e. it does not move the memory criterion, which
    stays on the same analytic per-layer activation model as the test above.
    """
    base = cost_config()
    before = dataclasses.replace(base, mla_pool_size=0, mla_layer_modes=())
    after = dataclasses.replace(
        base,
        mla_pool_block=64,
        mla_pool_size=16,
        mla_layer_modes=("full",) + ("reindex",) * (STACK - 1),
    )
    dense_cfg = _dense_cfg(base)
    params_list = _stack_params(base, STACK)

    x8 = jr.normal(jr.PRNGKey(1), (1, T8, base.hidden))
    d8 = jax.jit(lambda x: _stack(params_list, dense_cfg, x, ("full",) * STACK))
    b8 = jax.jit(lambda x: _stack(params_list, before, x, ("full",) * STACK))
    a8 = jax.jit(lambda x: _stack(params_list, after, x, after.mla_layer_modes))

    x64 = jr.normal(jr.PRNGKey(2), (1, T64, base.hidden))
    d64 = jax.jit(lambda x: _dense_probe_stack(params_list, base, x))
    b64 = jax.jit(lambda x: _stack(params_list, before, x, ("full",) * STACK))
    a64 = jax.jit(lambda x: _stack(params_list, after, x, after.mla_layer_modes))

    label8 = f"criterion 16 | MLA stack x{STACK} | 8K"
    label64 = f"criterion 16 | MLA stack x{STACK} | 64K"
    with ClockControl() as clock:
        cost_method.print_clock(clock, label8)
        m8 = measure(
            {"dense": lambda: d8(x8), "before": lambda: b8(x8), "after": lambda: a8(x8)},
            label=label8, clock=clock,
        )
        m64 = measure(
            {"dense": lambda: d64(x64), "before": lambda: b64(x64), "after": lambda: a64(x64)},
            label=label64, clock=clock,
        )

    H, D, B = base.num_heads, base.mla_head_dim, 1
    m_dense8 = _dense_peak_bytes(B, H, T8)
    m_sparse8 = _sparse_peak_bytes(B, H, D, T8, base.mla_top_k, base.swa_window)
    m_dense64 = _dense_peak_bytes(B, H, T64)
    m_sparse64 = _sparse_peak_bytes(B, H, D, T64, base.mla_top_k, base.swa_window)

    cost_method.print_measurement(m8, ref="dense", threshold=THRESHOLD_8K)
    cost_method.print_measurement(m64, ref="dense", threshold=THRESHOLD_STRICTLY_BETTER)
    print(f"[{label8}] memory: dense={m_dense8 / 1e6:.1f}MB sparse={m_sparse8 / 1e6:.1f}MB "
          f"(ratio {m_sparse8 / m_dense8:.4f}); pool={T8 * after.mla_pool_size * 4 / 1e6:.1f}MB")
    print(f"[{label64}] memory: dense={m_dense64 / 1e9:.1f}GB sparse={m_sparse64 / 1e6:.1f}MB "
          f"(ratio {m_sparse64 / m_dense64:.4f}); pool={T64 * after.mla_pool_size * 4 / 1e6:.1f}MB")

    assert m_sparse8 <= THRESHOLD_8K * m_dense8, f"8K memory sparse {m_sparse8} vs dense {m_dense8}"
    assert m_sparse64 < m_dense64, f"64K memory sparse {m_sparse64} vs dense {m_dense64}"

    # The task's strict reading of ADR-015 p.1: without a hard pin, no verdict.
    # Reported next to the implemented one so the architect sees both.
    if clock.report.on_gpu and not clock.report.hard_pin_ok:
        cost_method.print_verdict(
            verdict(m8.stats("after", "dense"), THRESHOLD_8K,
                    clock=clock, measurement=m8, strict=True),
            f"{label8} [strict p.3 reading, reported only]",
        )
    _judge_all([
        (label8, verdict(m8.stats("after", "dense"), THRESHOLD_8K,
                         clock=clock, measurement=m8)),
        (label64, verdict(m64.stats("after", "dense"), THRESHOLD_STRICTLY_BETTER,
                          clock=clock, measurement=m64, strictly_better=True)),
    ])
