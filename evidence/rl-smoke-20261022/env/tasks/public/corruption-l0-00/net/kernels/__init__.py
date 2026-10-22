"""Custom GPU kernels for the L3 skeleton (net/).

First entry: :mod:`net.kernels.kda_ut_solve` — the batched unit-triangular
solve ``(I + L) X = B`` of the KDA chunk, which XLA executes as tens of
thousands of per-batch ``batch_trsm_left_kernel`` launches (see the nsys
profile of the stationary phase).  Nothing in ``net/`` imports this package
yet: integration into ``net/kda.py`` is a separate delta after the on-device
check.
"""
