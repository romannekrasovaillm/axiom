"""Gate-off parity of the compute-dtype port against the pre-port tree.

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
``kda_impl`` and for the chunk-loop remat switch on and off, against the commit
the port was applied on top of.  The reference is extracted with ``git archive``
(the same idiom as ``test_29_compute_dtype.py``) into
``net/tests/.preport-net``, so the fleet that churns the working tree cannot
perturb it and no checkout state is touched.

Two criteria, one per form (ADR-040 Amendment, 09.10.2026)
----------------------------------------------------------

The bitwise promise is a statement about the *port*: with the gate off, the code
that the port touched must compute what it computed before.  ADR-050/G1 later
re-wrote the UT transform of two forms — ``(I + L)^{-1}`` via ``jnp.linalg.inv``
became the batched triangular solve ``(I + L) W = X`` (``jax.lax.linalg.
triangular_solve``) — keeping the algebra and changing the rounding *by
construction*.  Asserting bit equality against the pre-port tree for the
re-written algebra therefore asserts something unreachable, and chasing it would
mean reverting the very lever (G0: ~493k LU kernels per 120 s) that the rewrite
removed.  The Amendment splits the criterion instead:

* ``kda_impl="chunked"`` — **not touched** by ADR-047/ADR-050, so it stays the
  regression baseline and is compared **bitwise** (raw fp32 bit patterns, no
  tolerance), exactly as before;
* ``kda_impl="wyut"`` and ``"chunked_cc"`` — re-written forms, compared to the
  pre-port tree **within a documented ``atol``** (numerical parity, not bits).

Which criterion applies to which form, and why, is data in :data:`CRITERIA` and
is *printed* on every run (banner + one verdict line per comparison), so the
verdict is readable: a green ``chunked`` line means bits, a green ``wyut`` line
means a bounded numerical difference — never "close enough for all three".

``PREPORT_COMMIT`` is deliberately unchanged: the port tree stays the reference
of "the gate is transparent relative to the port".  Enlarging the *bitwise* set
and moving the anchor are architect decisions (Amendment, points 1 and 4).

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
from dataclasses import dataclass
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
#: an older upstream commit, is what gate-off must reproduce.  Changing it is a
#: separate architect decision (ADR-040 Amendment, point 4) — not a lever of the
#: re-anchor, which is exactly why it is still spelled out here unchanged.
PREPORT_COMMIT = "edb83a0"

GATE = compute_dtype.MODE_ENV

#: Declared implementations of ``kda`` (``net/config.py::validate_config``).
IMPLS = ("chunked", "wyut", "chunked_cc")

#: Forms ADR-047/ADR-050 re-wrote: their gate-off graph is *not* the pre-port one
#: (the UT transform is solved, not inverted), so they get numerical parity.
REWRITTEN_FORMS = ("wyut", "chunked_cc")

#: Forms the Amendment keeps bitwise.  Today exactly the untouched ``chunked``.
BITWISE_FORMS = ("chunked",)

#: Tolerance for the re-written forms against the pre-port tree.
#:
#: Observed divergence (G1 tree ``a0aece9`` vs the port anchor ``edb83a0``; CPU,
#: this module's smoke geometry, both remat states):
#:
#:   * ``wyut``       max|delta| = 3.725e-08  (~2 ulp of fp32 at max|ref| 0.26)
#:   * ``chunked_cc`` max|delta| = 4.470e-08  (~3 ulp)
#:   * ``chunked``    max|delta| = 0.000e+00  (bit-identical — see the bitwise arm)
#:
#: ``1e-5`` is ~220x the largest observed value: enough headroom that the guard
#: does not redden on re-rounding alone, tight enough that any real defect (an
#: unwrapped projection, an operand re-typed, a wrong solve side) — which moves
#: results by O(1) relative error, orders of magnitude above this bound — still
#: reddens.  The looser ``1e-4`` of the cross-form parity suites
#: (``test_kda_wyut.py``/``test_kda_chunked_cc.py``) is deliberately *not* reused
#: here: those compare two different algorithms, this compares one algorithm
#: across a rounding change, so the bound must stay near the rounding floor.
_REWRITTEN_FORM_ATOL = 1e-5


@dataclass(frozen=True)
class _Criterion:
    """Which comparison a ``kda_impl`` gets against the pre-port tree, and why."""

    kind: str  # "bitwise" | "atol"
    why: str
    atol: float | None = None

    def label(self) -> str:
        if self.kind == "bitwise":
            return "bitwise (raw fp32 patterns, atol=0)"
        return f"atol ({self.atol:.1e} on max|got-ref|)"


#: The criterion table (ADR-040 Amendment).  Data, not branching in the test
#: body, so the run banner can print it verbatim and a future architect decision
#: (enlarging the bitwise set, or re-anchoring ``PREPORT_COMMIT``) is one edit
#: with a visible diff, not a scatter of conditionals.
CRITERIA = {
    "chunked": _Criterion(
        kind="bitwise",
        why=(
            "форма НЕ тронута ADR-047/ADR-050 — регрессионный эталон ADR-040 п.1; "
            "побитовое равенство порту обязательно"
        ),
    ),
    "wyut": _Criterion(
        kind="atol",
        atol=_REWRITTEN_FORM_ATOL,
        why=(
            "переписана ADR-047 + ADR-050/G1 (батчевый TRSM вместо inv) — "
            "округление иное по построению, побитовость с портом недостижима "
            "(ADR-040 Amendment п.2), критерий — численный паритет"
        ),
    ),
    "chunked_cc": _Criterion(
        kind="atol",
        atol=_REWRITTEN_FORM_ATOL,
        why=(
            "переписана ADR-047 + ADR-050/G1 (батчевый TRSM вместо inv) — "
            "округление иное по построению, побитовость с портом недостижима "
            "(ADR-040 Amendment п.2), критерий — численный паритет"
        ),
    ),
}

#: Print tag — every verdict line carries it, so a log reader can grep the
#: guard's criterion out of a whole-suite run.
_TAG = "[dtype-gate-parity]"

#: Mutation used by the teeth below: 100x :data:`_REWRITTEN_FORM_ATOL` (so 1e-3
#: at the current value), ~2e4x the observed 4.5e-08 and the size a real defect
#: would produce.  Scaled off the tolerance, not pinned absolutely, so the tooth
#: keeps its meaning if the architect revises the bound.  If a divergence of this
#: size ever passed, the tolerance would be hiding real breakage.
_MUTATION = 100 * _REWRITTEN_FORM_ATOL

#: Smoke geometry: long enough for two chunks (the inter-chunk state transfer is
#: where a mis-typed operand would show), short enough to stay cheap on CPU.
SEQ = 64
CHUNK = 16


@pytest.fixture(scope="module", autouse=True)
def _print_criteria_table():
    """Print which criterion applies to which form, and why (Amendment p.3).

    Without this the run says only "8 passed" and the reader cannot tell that a
    green ``chunked`` line is bit-exactness while a green ``wyut`` line is a
    bounded difference — that ambiguity is what the Amendment forbids.
    """
    print(
        f"{_TAG} критерии сравнения с портовым деревом {PREPORT_COMMIT} "
        f"(ADR-040 + Amendment 09.10.2026):",
        flush=True,
    )
    for impl in IMPLS:
        crit = CRITERIA[impl]
        print(f"{_TAG}   kda_impl={impl!r:<13} -> {crit.label()}", flush=True)
        print(f"{_TAG}       почему: {crit.why}", flush=True)


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


def _observe(impl: str, got, ref, *, expect: bool | None = None) -> tuple[bool, str]:
    """Apply ``impl``'s criterion, print the verdict, return ``(ok, detail)``.

    The single place the criterion is *executed* — the tests below and the
    mutation teeth go through it, so a tooth proves the criterion the guard
    actually applies, not a re-implementation of it.  ``expect`` is set only by
    the teeth: their deliberate divergences print ``FAIL`` too, and the tag keeps
    a reader grepping the log for failures from mistaking a probe for a break.
    """
    crit = CRITERIA[impl]
    got_arr = np.asarray(got)
    ref_arr = np.asarray(ref)
    assert got_arr.shape == ref_arr.shape, (
        f"kda_impl={impl!r}: форма результата изменилась — "
        f"got {got_arr.shape}, ref {ref_arr.shape}"
    )
    delta = float(np.abs(got_arr - ref_arr).max())
    if crit.kind == "bitwise":
        same = np.array_equal(_bits(got), _bits(ref))
        ok = bool(same)
        detail = f"bitwise_identical={same} max|delta|={delta:.3e}"
    else:
        ok = delta <= crit.atol
        detail = f"max|delta|={delta:.3e} atol={crit.atol:.1e}"
    verdict = "PASS" if ok else "FAIL"
    if expect is not None:
        verdict += f" [tooth probe, ожидалось {'PASS' if expect else 'FAIL'}]"
    print(
        f"{_TAG} kda_impl={impl!r} criterion={crit.kind} {detail} -> {verdict}",
        flush=True,
    )
    return ok, detail


def _assert_preport_parity(impl: str, got, ref, *, context: str = "") -> None:
    """Assert ``got`` matches the pre-port tree under ``impl``'s own criterion."""
    ok, detail = _observe(impl, got, ref)
    crit = CRITERIA[impl]
    where = f" [{context}]" if context else ""
    if crit.kind == "bitwise":
        assert ok, (
            f"kda_impl={impl!r}{where}: критерий побитовый — гейт выключен, но "
            f"результат не побитово равен дорпортовому дереву "
            f"({PREPORT_COMMIT}); {detail}"
        )
    else:
        assert ok, (
            f"kda_impl={impl!r}{where}: критерий — допуск atol={crit.atol:.1e} "
            f"(переписанная ADR-047/ADR-050 форма), но расхождение превышает "
            f"допуск: {detail}"
        )


@pytest.mark.parametrize("backward", [False, True], ids=["noremat", "remat"])
@pytest.mark.parametrize("impl", BITWISE_FORMS)
def test_gate_off_is_bitwise_the_preport_tree(preport, monkeypatch, impl, backward):
    """Unset gate: the *unchanged* form is byte-for-byte the pre-port one.

    ``chunked`` is not touched by ADR-047/ADR-050, so the ADR-040 promise holds
    for it in full.  The floor is exact equality of the fp32 bit patterns — a
    tolerance here would accept the very drift the port must not introduce.
    """
    monkeypatch.delenv(GATE, raising=False)
    assert compute_dtype.mode() == compute_dtype.FP32

    cfg = _cfg(impl, kda_chunked_backward=backward)
    params = preport.kda.init_kda(jr.PRNGKey(0), cfg)
    x = jr.normal(jr.PRNGKey(1), (SEQ, cfg.hidden))

    got = kda.apply_kda(params, cfg, x, chunk_size=CHUNK)
    ref = preport.kda.apply_kda(params, cfg, x, chunk_size=CHUNK)

    _assert_preport_parity(
        impl, got, ref, context=f"kda_chunked_backward={backward}"
    )


@pytest.mark.parametrize("backward", [False, True], ids=["noremat", "remat"])
@pytest.mark.parametrize("impl", REWRITTEN_FORMS)
def test_rewritten_forms_match_the_preport_tree_within_atol(
    preport, monkeypatch, impl, backward
):
    """Re-written forms: numerical parity with the pre-port tree, not bits.

    ADR-050/G1 solves the UT transform (``(I + L) W = X``) instead of inverting
    ``I + L``, so the gate-off graph of ``wyut``/``chunked_cc`` is deliberately
    *not* the pre-port one.  ADR-040 Amendment fixes their criterion as a
    documented tolerance (:data:`_REWRITTEN_FORM_ATOL`); the bitwise promise
    remains where it is still reachable — see
    :func:`test_gate_off_is_bitwise_the_preport_tree`.
    """
    monkeypatch.delenv(GATE, raising=False)
    assert compute_dtype.mode() == compute_dtype.FP32

    cfg = _cfg(impl, kda_chunked_backward=backward)
    params = preport.kda.init_kda(jr.PRNGKey(0), cfg)
    x = jr.normal(jr.PRNGKey(1), (SEQ, cfg.hidden))

    got = kda.apply_kda(params, cfg, x, chunk_size=CHUNK)
    ref = preport.kda.apply_kda(params, cfg, x, chunk_size=CHUNK)

    _assert_preport_parity(
        impl, got, ref, context=f"kda_chunked_backward={backward}"
    )


def test_the_window_branch_is_gate_off_transparent(preport, monkeypatch):
    """``_with_window``/``_window_projection`` — the second gated call site.

    The window branch either reuses the delta-rule projections (shared) or uses
    its own (``W_swa_*``/``W_g``) — the port gates both spellings, so it gets its
    own case rather than riding on the no-window runs above.  It rides on
    ``chunked_cc``, so it inherits that form's atol criterion.
    """
    monkeypatch.delenv(GATE, raising=False)
    overrides = dict(swa_window=8, swa_share_kda_projections=False)
    cfg = _cfg("chunked_cc", kda_chunked_backward=False, **overrides)
    params = preport.kda.init_kda(jr.PRNGKey(0), cfg)
    x = jr.normal(jr.PRNGKey(1), (SEQ, cfg.hidden))

    _assert_preport_parity(
        "chunked_cc",
        kda.apply_kda(params, cfg, x, chunk_size=CHUNK),
        preport.kda.apply_kda(params, cfg, x, chunk_size=CHUNK),
        context="window branch",
    )


@pytest.mark.parametrize("impl", REWRITTEN_FORMS)
def test_the_atol_arm_reddens_beyond_the_documented_atol(impl):
    """Tooth: the tolerance is a bound, not a blanket pass.

    A divergence 100x the documented ``atol`` — the size a real defect (an
    unwrapped projection, a re-typed operand, a wrong solve side) would produce —
    must redden; one tenth of it must stay green, so the bound is not slack.
    Synthetic inputs on purpose: this checks the *criterion the guard applies*
    (:func:`_observe`), isolated from KDA numerics.
    """
    crit = CRITERIA[impl]
    assert crit.kind == "atol", f"{impl!r} must be an atol form, got {crit.kind!r}"
    ref = jnp.asarray(np.linspace(-0.5, 0.5, 32, dtype=np.float32))

    beyond = np.asarray(ref) + _MUTATION
    assert not _observe(impl, beyond, ref, expect=False)[0], (
        f"kda_impl={impl!r}: расхождение {_MUTATION:.1e} > atol "
        f"{crit.atol:.1e} принято — допуск превратился в пропуск"
    )

    within = np.asarray(ref) + crit.atol / 10
    assert _observe(impl, within, ref, expect=True)[0], (
        f"kda_impl={impl!r}: расхождение {crit.atol / 10:.1e} << atol "
        f"{crit.atol:.1e} отвергнуто — допуск необоснованно жёсткий"
    )


def test_the_bitwise_arm_reddens_on_one_ulp(preport, monkeypatch):
    """Tooth: ``chunked`` is still compared by bits, not silently by atol.

    One ulp is far below everything the loose criterion would tolerate, so this
    reddens only if the bitwise floor is really still in force for the form the
    Amendment did not re-write.  Uses the real pre-port reference, so the plant
    is a single-bit flip of a genuine result.
    """
    monkeypatch.delenv(GATE, raising=False)
    cfg = _cfg("chunked")
    params = preport.kda.init_kda(jr.PRNGKey(0), cfg)
    x = jr.normal(jr.PRNGKey(1), (SEQ, cfg.hidden))
    ref = np.asarray(preport.kda.apply_kda(params, cfg, x, chunk_size=CHUNK))

    mutant = ref.copy()
    flat = np.argmax(np.abs(ref))
    mutant.flat[flat] = np.nextafter(mutant.flat[flat], np.float32(np.inf))
    assert float(np.abs(mutant - ref).max()) > 0.0

    ok, detail = _observe("chunked", mutant, ref, expect=False)
    assert not ok, (
        f"одиночный ulp не покраснил побитовую ветвь 'chunked' — "
        f"регрессионный эталон ADR-040 ослаблен ({detail})"
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
