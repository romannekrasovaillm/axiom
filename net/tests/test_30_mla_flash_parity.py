"""Re-test of the ``AXIOM_MLA_DENSE_FLASH`` gate on the current checkout (§3).

The claim this test replaces is the 07.10 reading "flash gives **0 delta**" —
measured on a stale stand checkout (``caf5a855``) and therefore not a statement
about the code in this tree.  Re-measured on the current checkout the claim does
**not** hold, and the reason is a layout bug rather than rounding:

``jax.nn.dot_product_attention`` takes ``(B, T, N, H)`` — sequence *before*
heads.  The pre-re-test flash branch transposed the projections (already
``(B, T, H, dq)``) to ``(B, H, T, dq)`` and the kernel then read the *head* axis
as the sequence axis.  The output shapes still round-tripped
(``(B,H,T,dq) -> (B,T,H,dq) -> (B,T,H*dq)``), so nothing crashed and a shape-only
check would have passed; the arithmetic was wrong, by up to ~0.83 relative on
the smoke form (measured 08.10, before the fix: ``max_abs`` 3.9e-3 against an
output whose own maximum is 4.7e-3).

With the operands passed in their native ``(B, T, H, dq)`` order the flash path
is **bit-for-bit** the legacy oracle on CPU for every shape exercised here —
not "close", identical bit patterns.  That is the strongest form the task
allows ("бит-точность или задокументированное расхождение"), so it is what the
test asserts; ``FLASH_PARITY_TOL`` is the fallback envelope for a backend whose
fused kernel legitimately re-associates the reductions (a GPU/cuDNN flash
kernel may), and the measured deviation is reported either way.

``test_the_old_transposed_call_is_what_this_test_catches`` keeps the teeth: it
reproduces the pre-fix call inline and asserts it *fails* the parity check, so a
future regression of the same kind cannot pass unnoticed.

The oracle itself (flash off, the default) is untouched by all of this: ADR-009
D4 byte-exactness is pinned in ``net/tests/test_29_compute_dtype.py`` against
the baseline commit.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from net import mla

#: Fallback envelope for a backend whose fused attention kernel re-associates
#: the reductions (CPU measures exactly 0.0 — see the docstring).
FLASH_PARITY_TOL = 1e-5

FLASH_ENV = "AXIOM_MLA_DENSE_FLASH"

#: Small forms: the smoke shapes of the acceptance suite plus a couple that make
#: the head count differ from the sequence length (the axis mix-up's signature).
SHAPES = ((1, 8), (2, 16), (2, 64), (3, 5), (1, 33))


def _cfg():
    import conftest  # net/tests is on sys.path (pytest rootdir insertion)

    return conftest.small_config()


@pytest.fixture
def setup():
    cfg = _cfg()
    params = mla.init_mla(jr.PRNGKey(0), cfg)
    return cfg, params


def _bits(x) -> np.ndarray:
    arr = np.asarray(x)
    assert arr.dtype == np.float32
    return arr.view(np.uint32)


def _deviation(a, b) -> float:
    return float(np.abs(np.asarray(a) - np.asarray(b)).max())


def test_flash_is_bitwise_the_legacy_oracle(setup, monkeypatch) -> None:
    """The measurement the stale stand reading claimed, done on this checkout."""
    cfg, params = setup
    monkeypatch.delenv(FLASH_ENV, raising=False)
    measured = []
    for b, t in SHAPES:
        x = jr.normal(jr.PRNGKey(1), (b, t, cfg.hidden))
        legacy = mla._dense_apply(params, cfg, x)
        monkeypatch.setenv(FLASH_ENV, "1")
        flash = mla._dense_apply(params, cfg, x)
        monkeypatch.delenv(FLASH_ENV, raising=False)

        assert flash.shape == legacy.shape
        assert bool(jnp.all(jnp.isfinite(flash)))
        dev = _deviation(flash, legacy)
        measured.append((b, t, dev))
        if np.array_equal(_bits(legacy), _bits(flash)):
            continue
        # Not bit-identical on this backend: only a documented, bounded
        # re-association of the same reductions is acceptable.
        assert dev <= FLASH_PARITY_TOL, (
            f"flash/legacy divergence at (b={b}, t={t}) is {dev:.3e}, above the "
            f"documented envelope {FLASH_PARITY_TOL:.0e}: {measured}"
        )


def test_flash_parity_holds_with_an_explicit_mask(setup, monkeypatch) -> None:
    """The caller's ``(T, S)`` mask must be broadcast in the kernel's layout."""
    cfg, params = setup
    x = jr.normal(jr.PRNGKey(1), (2, 16, cfg.hidden))
    mask = jnp.tril(jnp.ones((16, 16), jnp.bool_))
    monkeypatch.delenv(FLASH_ENV, raising=False)
    legacy = mla._dense_apply(params, cfg, x, mask)
    monkeypatch.setenv(FLASH_ENV, "1")
    flash = mla._dense_apply(params, cfg, x, mask)
    monkeypatch.delenv(FLASH_ENV, raising=False)
    assert np.array_equal(_bits(legacy), _bits(flash)) or (
        _deviation(flash, legacy) <= FLASH_PARITY_TOL
    )


def test_flash_path_is_off_by_default(setup, monkeypatch) -> None:
    """The oracle must stay byte-exact unless the diagnostic gate is set."""
    monkeypatch.delenv(FLASH_ENV, raising=False)
    assert mla._flash_dense() is False
    monkeypatch.setenv(FLASH_ENV, "0")
    assert mla._flash_dense() is False
    monkeypatch.setenv(FLASH_ENV, "1")
    assert mla._flash_dense() is True


def test_the_old_transposed_call_is_what_this_test_catches(setup) -> None:
    """Mutation test: the pre-re-test layout must *fail* the parity check.

    Reproduces the removed branch (transpose to ``(B, H, T, dq)``, hand it to
    the kernel, transpose back) and shows it is neither bit-identical nor within
    the envelope — i.e. the assertion above would have caught the stale stand
    reading had it been run here.
    """
    cfg, params = setup
    b, t = 2, 16
    x = jr.normal(jr.PRNGKey(1), (b, t, cfg.hidden))

    x32 = x.astype(jnp.float32)
    c = x32 @ params.W_c.astype(jnp.float32)
    q, k, v = mla._project(params, cfg, x32, c)

    def old_flash(q, k, v):
        # The layout as it was before the 08.10 re-test.
        of = jax.nn.dot_product_attention(
            jnp.transpose(q, (0, 2, 1, 3)),
            jnp.transpose(k, (0, 2, 1, 3)),
            jnp.transpose(v, (0, 2, 1, 3)),
            mask=None,
            is_causal=True,
        )
        return jnp.transpose(of, (0, 2, 1, 3))

    mutant_o = old_flash(q, k, v).reshape(*x.shape[:-1], cfg.num_heads * cfg.mla_head_dim)
    gate = jax.nn.sigmoid(x32 @ params.W_g.astype(jnp.float32))
    mutant = (gate * mutant_o) @ params.W_o.astype(jnp.float32)

    legacy = mla._dense_apply(params, cfg, x)
    assert not np.array_equal(_bits(legacy), _bits(mutant)), (
        "мутант совпал с оракулом — проверка паритета перестала ловить подмену оси"
    )
    assert _deviation(legacy, mutant) > FLASH_PARITY_TOL, (
        "расхождение мутанта ниже задокументированного допуска — "
        "проверка паритета слишком слабая"
    )
