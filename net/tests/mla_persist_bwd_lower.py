"""``lower_check`` adapter: the persistent *backward* MLA kernel.

The pallas-gb10-kernel skill's ``lower_check.py`` lowers ``mod.kernel`` for
every ``CASES`` entry that ``mod.valid_config`` accepts.  The shipped module
(:mod:`net.kernels.mla_persist`) exposes its forward kernel as ``kernel``;
this thin adapter exposes the backward kernel
(:func:`net.kernels.mla_persist.kernel_bwd`) with the same contract, so the
static check covers both kernels of the module:

    python3 <skill>/scripts/lower_check.py net/kernels/mla_persist.py
    python3 <skill>/scripts/lower_check.py net/tests/mla_persist_bwd_lower.py

The adapter is not a test module (no ``test_`` prefix: pytest does not collect
it); the lowering is also pinned from the CPU test suite
(``net/tests/test_44_mla_persist.py``), which measures the same SMEM numbers
with the skill's artificial limit.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from net.kernels import mla_persist as _impl  # noqa: E402

#: Same case grid as the forward module (the campaign forms B=2/B=4 at
#: T=8192/H=12 and the CPU-cheap mini case; the bk=128 entry stays honestly
#: skipped by the backward sieve).
CASES = _impl.CASES

#: The tile-width sweep of the forward module, with the backward floor as the
#: binding sieve.
TUNE_SPACE = _impl.TUNE_SPACE


def make_inputs(key, **case):
    """One backward pass's inputs: ``(q, k, v, do, D, m, l)``."""
    return _impl.make_inputs_bwd(key, **case)


def valid_config(case, params):
    """The backward sieve: the tile rules plus the backward SMEM floor."""
    return _impl.valid_config_bwd(case, params)


def kernel(q, k, v, do, D, m, l, *, bq=_impl.BQ, bk=_impl.BK):
    """The persistent backward kernel (one launch per layer-pass)."""
    return _impl.kernel_bwd(q, k, v, do, D, m, l, bq=int(bq), bk=int(bk))


def reference(q, k, v, do, D, m, l, *, bq=_impl.BQ, bk=_impl.BK):
    """The declared-numerics adjoint replica (the kernel's numerical pin)."""
    return _impl.emulate_attention_bwd(q, k, v, do, D, m, l, bq=bq, bk=bk)
