"""``lower_check`` adapter: the persistent *backward* kernel (variant B).

The pallas-gb10-kernel skill's ``lower_check.py`` lowers ``mod.kernel`` for
every ``CASES`` entry that ``mod.valid_config`` accepts.  The shipped forward
module (:mod:`net.kernels.kda_cc_fused_persist`) exposes its forward kernel as
``kernel``; this thin adapter exposes the variant-B backward kernel
(:func:`net.kernels.kda_cc_fused_persist.kernel_bwd`) with the same contract,
so the static check covers both kernels of the module:

    python3 <skill>/scripts/lower_check.py net/tests/kda_persist_bwd_lower.py --configs all

The adapter is not a test module (no ``test_`` prefix: pytest does not collect
it); the lowering is also pinned from the CPU test suite
(``net/tests/test_kda_cc_fused_persist.py``), which measures the same SMEM
numbers with the skill's artificial limit.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from net.kernels import kda_cc_fused_persist as _persist  # noqa: E402

#: Same case grid as the forward module (T=2048/H=12 is the campaign geometry;
#: T=256/H=2 keeps the CPU-side runs cheap); the C=16/C=64 entries stay
#: honestly skipped by the sieve.
CASES = _persist.CASES

#: The static knobs passed through to the kernel: the decay-factor tile and the
#: chunk width (the live grid is C=32; the sieve keeps the rest out).
TUNE_SPACE = dict(tile=[16], chunk=[_persist.CHUNK])


def make_inputs(key, **case):
    """One backward pass's inputs: the forward interval plus ``(do, dS)``."""
    return _persist.make_bwd_inputs(key, **case)


def valid_config(case, params):
    """The backward sieve: the forward shape rules plus the backward floor."""
    return _persist.valid_config_bwd(case, params)


def kernel(q, k, v, alpha, beta32, S, do, dS, *, tile=16, chunk=32):
    """The persistent backward kernel (one launch per layer-pass, variant B)."""
    return _persist.kernel_bwd(q, k, v, alpha, beta32, S, do, dS,
                               tile=tile, chunk=int(chunk))


def reference(q, k, v, alpha, beta32, S, do, dS, *, tile=16, chunk=32):
    """The fp32 adjoint reference (the kernel's numerical pin)."""
    return _persist.reference_bwd_scan(q, k, v, alpha, beta32, S, do, dS,
                                       tile=tile, chunk=chunk)
