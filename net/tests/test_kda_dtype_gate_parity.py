"""Gate-off bit-exactness of the compute-dtype port across the merge.

``556cf08`` (BF16 campaign, phase 1) wraps the KDA projections of ``net/kda.py``
in ``compute_dtype.gemm``.  Under ``AXIOM_COMPUTE_DTYPE=fp32`` — the default,
what an unset environment gives — that wrapper is defined to return the caller's
own ``x @ W``, so the graph is meant to be *byte-for-byte* the pre-port one, not
merely close.

The cherry-pick of ``556cf08`` into ``arch/kda-rewrite`` conflicted in
``net/kda.py``, because that file now carries ADR-047's ``chunked_cc`` form.
That conflict is exactly where the promise is easiest to break silently: a
projection left unwrapped, an operand re-typed under fp32, or a helper renamed
out from under the new form would all move the pinned oracle without any
existing test noticing.  ``net/tests/test_29_compute_dtype.py`` compares the
``chunked`` path against ``773e389``, which predates ``wyut``/``chunked_cc``
(and the ADR-049 remat boundaries), so it cannot see the merged forms.

This module pins the promise for the merged tree directly, for **every** declared
``kda_impl`` and for the chunk-loop remat switch on and off: with the gate unset,
``apply_kda`` must return the bit patterns of the commit the port was applied on
top of.  The reference is extracted with ``git archive`` (the same idiom as
``test_29_compute_dtype.py``) into ``net/tests/.preport-net``, so the fleet that
churns the working tree cannot perturb it and no checkout state is touched.

The bf16 arm is deliberately absent: the acceptance suite runs gate-off, the
local XLA:CPU backend cannot execute the fused bf16 GEMMs (see
``test_29_compute_dtype.py``), and the bf16-vs-fp32 measurement on l3-full is the
GB10 stand's job (``tools/mfu_bf16_protocol.py``, ADR-040/ADR-045).
"""

from __future__ import annotations

import dataclasses
import io
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from conftest import small_config

from net import compute_dtype, kda

NET_DIR = Path(__file__).resolve().parents[1]
CASE_DIR = NET_DIR.parent

#: The merge anchor of the port (rollback plan: ``evidence``/``TASK`` name it).
#: The whole point of this test is that the pre-port tree of *this* branch, not
#: an older upstream commit, is what gate-off must reproduce.
PREPORT_COMMIT = "edb83a0"

GATE = compute_dtype.MODE_ENV

#: Declared implementations of ``kda`` (``net/config.py::validate_config``).
IMPLS = ("chunked", "wyut", "chunked_cc")

#: Smoke geometry: long enough for two chunks (the inter-chunk state transfer is
#: where a mis-typed operand would show), short enough to stay cheap on CPU.
SEQ = 64
CHUNK = 16


@pytest.fixture(scope="module")
def preport():
    """``net/`` as of :data:`PREPORT_COMMIT`, importable as ``preport_net``.

    ``git archive`` (not a worktree) so the fleet that churns the working tree
    cannot perturb the reference and no checkout state is touched.  The package
    is renamed because the modules import each other relatively
    (``from .config import ...``), which keeps working under the new name.
    """
    root = Path(__file__).resolve().parent / ".preport-net"
    if root.exists():
        import shutil

        shutil.rmtree(root)
    root.mkdir()
    try:
        archive = subprocess.run(
            ["git", "-C", str(CASE_DIR), "archive", PREPORT_COMMIT, "net/"],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:  # pragma: no cover
        pytest.skip(f"pre-port {PREPORT_COMMIT} недоступен: {exc}")
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as bundle:
        bundle.extractall(root)
    (root / "net").rename(root / "preport_net")

    sys.path.insert(0, str(root))
    try:
        import preport_net.config as bconfig
        import preport_net.kda as bkda

        yield SimpleNamespace(config=bconfig, kda=bkda)
    finally:
        sys.path.remove(str(root))
        for name in [m for m in list(sys.modules) if m.startswith("preport_net")]:
            del sys.modules[name]
        import shutil

        shutil.rmtree(root, ignore_errors=True)


def _bits(x: jnp.ndarray) -> np.ndarray:
    """Raw fp32 bit patterns (the comparison the port promises, not a tolerance)."""
    arr = np.asarray(x)
    assert arr.dtype == np.float32, f"bitwise comparison expects fp32, got {arr.dtype}"
    return arr.view(np.uint32)


def _cfg(impl: str, **overrides):
    """The smoke config on one declared implementation (defaults untouched)."""
    return dataclasses.replace(
        small_config(**overrides),
        kda_impl=impl,
        # The declarative default is read from ``net/config.json``; the test
        # varies the chunk-loop boundary explicitly instead of relying on it.
    )


@pytest.mark.parametrize("backward", [False, True], ids=["noremat", "remat"])
@pytest.mark.parametrize("impl", IMPLS)
def test_gate_off_is_bitwise_the_preport_tree(preport, monkeypatch, impl, backward):
    """Unset gate: ``apply_kda`` is byte-for-byte the pre-port implementation.

    Covers all three declared implementations and the ADR-049 chunk-loop remat
    boundary in both states.  The floor is exact equality of the fp32 bit
    patterns — a tolerance would accept the very drift this port must not
    introduce.
    """
    monkeypatch.delenv(GATE, raising=False)
    assert compute_dtype.mode() == compute_dtype.FP32

    cfg = _cfg(impl, kda_chunked_backward=backward)
    params = preport.kda.init_kda(jr.PRNGKey(0), cfg)
    x = jr.normal(jr.PRNGKey(1), (SEQ, cfg.hidden))

    got = kda.apply_kda(params, cfg, x, chunk_size=CHUNK)
    ref = preport.kda.apply_kda(params, cfg, x, chunk_size=CHUNK)

    assert got.shape == ref.shape
    assert np.array_equal(_bits(got), _bits(ref)), (
        f"kda_impl={impl!r}, kda_chunked_backward={backward}: гейт выключен, "
        f"но результат не побитово равен дорпортовому дереву "
        f"(max|Δ| = {float(np.abs(np.asarray(got) - np.asarray(ref)).max()):.3e})"
    )


def test_the_window_branch_is_gate_off_transparent(preport, monkeypatch):
    """``_with_window``/``_window_projection`` — the second gated call site.

    The window branch either reuses the delta-rule projections (shared) or uses
    its own (``W_swa_*``/``W_g``) — the port gates both spellings, so it gets its
    own case rather than riding on the no-window runs above.
    """
    monkeypatch.delenv(GATE, raising=False)
    overrides = dict(swa_window=8, swa_share_kda_projections=False)
    cfg = _cfg("chunked_cc", kda_chunked_backward=False, **overrides)
    params = preport.kda.init_kda(jr.PRNGKey(0), cfg)
    x = jr.normal(jr.PRNGKey(1), (SEQ, cfg.hidden))

    assert np.array_equal(
        _bits(kda.apply_kda(params, cfg, x, chunk_size=CHUNK)),
        _bits(preport.kda.apply_kda(params, cfg, x, chunk_size=CHUNK)),
    )


def test_the_gate_still_refuses_an_unknown_value(monkeypatch):
    """The merge must not have softened the fail-closed contract (rollback (3)).

    The typo has to be rejected by the *layer*, not only by the gate module: the
    gated call sites resolve the mode when the graph is traced, so a ``bogus``
    value must raise there rather than quietly producing an fp32 graph.
    """
    monkeypatch.setenv(GATE, "bogus")
    cfg = _cfg("chunked_cc")
    params = kda.init_kda(jr.PRNGKey(0), cfg)
    x = jr.normal(jr.PRNGKey(1), (2, cfg.hidden))
    with pytest.raises(compute_dtype.ComputeDtypeError):
        kda.apply_kda(params, cfg, x, chunk_size=2)
    with pytest.raises(compute_dtype.ComputeDtypeError):
        compute_dtype.gemm(jnp.ones((2, 2)), jnp.ones((2, 2)))
