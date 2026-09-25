"""Context-length curriculum: 8K -> 64K, 90/10 token split.

MODEL-L3-SKELETON.md section 3 / kda-formulas.md section 3.4: the pretrain
budget is split 90% at 8K context and 10% at 64K.  The 256K -> 1M cooldown
stages are deferred (FIDELITY section 4).  The 90/10 split is our constant (the
source does not disclose the long-sequence fraction).
"""

from __future__ import annotations


def context_length_at(
    fraction: float,
    lengths: tuple[int, ...] = (8192, 65536),
    split: tuple[float, ...] = (0.90, 0.10),
) -> int:
    """Context length for a given position in the token budget.

    ``fraction`` is the fraction of the total token budget consumed so far, in
    ``[0, 1]``.  Returns the first stage whose cumulative budget covers it.
    """
    cumulative = 0.0
    for frac, length in zip(split, lengths):
        cumulative += frac
        if fraction < cumulative:
            return length
    return lengths[-1]


def stages() -> list[tuple[int, float]]:
    """The curriculum as ``(context_length, budget_fraction)`` pairs."""
    return [(8192, 0.90), (65536, 0.10)]
