"""Регрессия находки `kda-chunk-padding-nan-backward` (дефект сети, не дельты).

`net/kda.py:apply_chunked` дополняет последовательность нулями до кратности
`chunk_size`.  Производная `net/norm.py:l2_norm` на строго нулевом векторе —
`0/0`; в reverse-mode это NaN даже при нулевом upstream-градиенте строки
(`0 · NaN = NaN`), поэтому NaN доходит до KDA-параметров
(`W_q/W_k/conv_q/conv_k`) при паддинге, хотя значение loss то же самое.

Фикс (`net/norm.py`): primal считается дословно `x / (‖x‖ + eps)`, а тангенс в
нуле *определён* нулём — в нуле норма недифференцируема, берётся стандартная
субградиентная конвенция; при любом `x ≠ 0` производная точная.  Проверяются
три инварианта приёмки:

(а) градиент по нулевому входу конечен (и равен нулю);
(б) градиент по дополненной последовательности не зависит от паддинга:
    совпадает с градиентом при `pad = 0`, а нулевые строки дают ровно нуль;
(в) forward на ненулевых входах побитово совпадает с до-фиксовым выражением,
    градиент на них — с его аналитической производной (защита от регрессии).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from net import kda, model
from net.norm import l2_norm


def _reference_l2_norm(x: jnp.ndarray, eps: float = 1e-12, axis: int = -1) -> jnp.ndarray:
    """До-фиксовое выражение — оракул неизменности forward."""
    return x / (jnp.linalg.norm(x, axis=axis, keepdims=True) + eps)


def _grad_sum(f, x: jnp.ndarray) -> jnp.ndarray:
    return jax.grad(lambda y: jnp.sum(f(y)))(x)


# ---------------------------------------------------------------------------
# (а) нулевой вход → конечный (нулевой) градиент
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(4,), (2, 4), (3, 5, 8)])
def test_gradient_at_zero_is_finite_and_zero(shape):
    g = _grad_sum(l2_norm, jnp.zeros(shape))
    assert bool(jnp.all(jnp.isfinite(g))), "градиент на нулевом векторе обязан быть числовым"
    assert bool(jnp.all(g == 0.0)), (
        "в нуле норма недифференцируема; выбранная конвенция — нулевой тангенс"
    )


def test_zero_rows_contribute_exactly_zero_gradient():
    """Нулевая строка не влияет ни на свой, ни на чужие градиенты."""
    x = jnp.asarray([
        [3.0, 4.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ])
    g = _grad_sum(l2_norm, x)
    assert bool(jnp.all(g[1] == 0.0)), "строка паддинга обязана давать нулевой градиент"

    # Ненулевые строки считаются так, как если бы нулевой строки не было.
    rows = jnp.stack([x[0], x[2]])
    g_rows = _grad_sum(_reference_l2_norm, rows)
    assert jnp.allclose(g[jnp.asarray([0, 2])], g_rows, rtol=1e-6, atol=1e-7)


# ---------------------------------------------------------------------------
# (в) forward и backward на ненулевых входах не изменились
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(2, 4), (5, 7, 8), (64, 128)])
def test_forward_is_bit_identical_on_nonzero_inputs(shape):
    x = jr.normal(jr.PRNGKey(0), shape)
    assert bool(jnp.all(jnp.linalg.norm(x, axis=-1) > 0.0))
    assert bool(jnp.array_equal(l2_norm(x), _reference_l2_norm(x))), (
        "фикс не имеет права менять forward на ненулевых входах (ADR-010)"
    )


def test_forward_and_gradient_unchanged_on_tiny_nonzero_norms():
    """Сингулярна ровно точка нуля — сколь угодно малый ненулевой вход не задет."""
    x = jnp.full((3, 4), 1e-6)
    assert bool(jnp.array_equal(l2_norm(x), _reference_l2_norm(x)))
    g = _grad_sum(l2_norm, x)
    g_ref = _grad_sum(_reference_l2_norm, x)
    assert bool(jnp.all(jnp.isfinite(g)))
    assert jnp.allclose(g, g_ref, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("scale", [1.0, 1e-3, 1e3])
def test_gradient_matches_reference_on_nonzero_inputs(scale):
    x = jr.normal(jr.PRNGKey(1), (6, 8)) * scale
    g = _grad_sum(l2_norm, x)
    g_ref = _grad_sum(_reference_l2_norm, x)
    assert bool(jnp.all(jnp.isfinite(g)))
    assert jnp.allclose(g, g_ref, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# (б) паддинг чанка KDA не влияет на градиент
# ---------------------------------------------------------------------------


def _kda_loss(params, cfg, x, chunk: int):
    return jnp.sum(kda.apply_chunked(params, cfg, x, chunk_size=chunk)[: x.shape[0]])


def test_padded_chunk_gradient_is_finite_and_padding_independent(tiny_cfg):
    cfg = tiny_cfg
    params = kda.init_kda(jr.PRNGKey(0), cfg)
    x = jr.normal(jr.PRNGKey(1), (6, cfg.hidden))

    loss = lambda p, c: _kda_loss(p, cfg, x, c)
    base_grad = jax.grad(loss)(params, x.shape[0])  # pad = 0
    base_leaves = jax.tree_util.tree_leaves(base_grad)
    base_loss = float(loss(params, x.shape[0]))
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in base_leaves)

    # seq_len = 6 < chunk → pad = chunk - 6 > 0.  Набор размеров ограничен
    # значениями, на которых XLA-компилятор GPU не падает сам по себе
    # (преэкзистентный crash GemmFusion на tiny-конфиге, воспроизводится и без
    # фикса при некоторых chunk: 8/9/10/11/16/32/64).
    for chunk in (7, 13, 14):
        grad = jax.grad(loss)(params, chunk)
        leaves = jax.tree_util.tree_leaves(grad)
        assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves), (
            f"chunk={chunk}: паддинг не имеет права отравлять градиент"
        )
        assert float(loss(params, chunk)) == pytest.approx(base_loss, abs=1e-6)
        for a, b in zip(base_leaves, leaves):
            assert jnp.allclose(a, b, rtol=1e-5, atol=1e-6), (
                f"chunk={chunk}: градиент обязан совпасть с pad = 0"
            )


def test_padded_model_gradient_is_finite_and_padding_independent(tiny_cfg):
    """Сценарий находки: loss-путь RL-роллаута по токенам ответа."""
    cfg = tiny_cfg
    params = model.init_params(jr.PRNGKey(0), cfg)
    prompt_ids, gen_ids = list(range(1, 4)), [9, 10, 11]
    ids = jnp.asarray([prompt_ids + gen_ids], dtype=jnp.int32)
    seq_len = len(prompt_ids) + len(gen_ids)

    def loss_at(chunk: int, params):
        logits = model.forward(params, cfg, ids, chunk_size=chunk)
        logp = jax.nn.log_softmax(logits[:, :-1], axis=-1)
        picked = jnp.take_along_axis(logp, ids[:, 1:][..., None], axis=-1)[0, :, 0]
        return -jnp.sum(picked[-len(gen_ids):])

    value_base, grad_base = jax.value_and_grad(loss_at, argnums=1)(seq_len, params)
    leaves_base = jax.tree_util.tree_leaves(grad_base)
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves_base)

    padded_chunk = seq_len + 8  # pad > 0: заведомо нулевые строки внутри чанка
    value_padded, grad_padded = jax.value_and_grad(loss_at, argnums=1)(
        padded_chunk, params
    )
    leaves_padded = jax.tree_util.tree_leaves(grad_padded)
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves_padded), (
        "паддинг чанка не имеет права давать нечисловой градиент"
    )
    assert float(value_padded) == pytest.approx(float(value_base), abs=1e-6)
    for a, b in zip(leaves_base, leaves_padded):
        assert jnp.allclose(a, b, rtol=1e-5, atol=1e-6)
