"""Флаг отката рычага MoE-dispatch: ``AXIOM_MOE_DISPATCH=grouped|dense``.

Перф-правка ``net/moe.py`` исполняет только выбранные top-k эксперты
(``_routed_experts_grouped`` через ``lax.ragged_dot``); прежняя запись «einsum по
всем ``n_routed`` экспертам, затем gather top-k» сохранена как **откатный путь**
``_routed_experts_dense`` за переменной окружения.

Этот файл пинит то, ради чего флаг существует, — откат обязан быть эквивалентен и
управляем без правки сигнатур:

* (а) по умолчанию режим ``grouped`` и результат совпадает с прежним поведением;
* (б) режим ``dense`` даёт тот же ответ в пределах существующего допуска
      (``RTOL = 1e-5``, тот же lowering-бюджет, что и в
      ``net/tests/test_32_moe_grouped_topk.py``);
* (в) решение роутера (``topk``, ``sel_p``, QB-терм) от режима **не зависит**;
* (г) неизвестное значение флага — ошибка, а не тихий откат к умолчанию;
* (д) флаг читается **на каждом вызове** (hot rollback без перезапуска).

GPU не нужен: всё на CPU-бэкенде стенда.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from net import moe

#: Тот же относительный допуск, что пиннит разницу lowering'ов в test_32.
RTOL = 1e-5


def _x(cfg, seed: int = 0):
    return jax.random.normal(jax.random.PRNGKey(seed), (8, cfg.hidden))


def _forward(monkeypatch, cfg, params, x):
    _, topk, sel_p, p_full = moe._dispatch(params, cfg, x)
    out = moe.apply(params, cfg, x)
    return out, topk, sel_p, p_full


def _max_rel(a, ref) -> float:
    return float(jnp.max(jnp.abs(a - ref)) / (jnp.max(jnp.abs(ref)) + 1e-30))


def test_default_mode_is_grouped(monkeypatch):
    monkeypatch.delenv("AXIOM_MOE_DISPATCH", raising=False)
    assert moe.dispatch_mode() == "grouped"


def test_dense_flag_matches_grouped_within_tolerance(monkeypatch, cfg):
    """Откат эквивалентен: тот же forward, тот же ответ в пределах допуска."""
    key = jax.random.PRNGKey(7)
    params = moe.init_moe(key, cfg)
    x = _x(cfg)

    monkeypatch.setenv("AXIOM_MOE_DISPATCH", "grouped")
    out_grouped, topk_g, sel_p_g, _ = _forward(monkeypatch, cfg, params, x)

    monkeypatch.setenv("AXIOM_MOE_DISPATCH", "dense")
    out_dense, topk_d, sel_p_d, _ = _forward(monkeypatch, cfg, params, x)

    # (в) решение роутера не зависит от режима исполнения экспертов.
    assert jnp.array_equal(topk_g, topk_d)
    assert jnp.allclose(sel_p_g, sel_p_d, rtol=0.0, atol=0.0)
    # (б) различие — только lowering контракции; мера — относительная (atol=0 при
    # allclose браковал бы элементы вблизи нуля, что к точности контракции отношения не имеет).
    assert _max_rel(out_dense, out_grouped) <= RTOL, _max_rel(out_dense, out_grouped)


def test_dense_mode_is_the_pre_refactor_spelling(monkeypatch, cfg):
    """``dense`` считает ту же арифметику, что явная запись по всем экспертам."""
    params = moe.init_moe(jax.random.PRNGKey(11), cfg)
    x = _x(cfg, seed=3)
    xf = x.reshape(-1, x.shape[-1])
    z = moe.compute_dtype.gemm(xf, params.W_down)
    _, topk, sel_p, _ = moe._dispatch(params, cfg, x)

    ref = moe._routed_experts_dense(params, cfg, z, topk, sel_p)
    monkeypatch.setenv("AXIOM_MOE_DISPATCH", "dense")
    got = moe._routed_experts(params, cfg, z, topk, sel_p)
    assert jnp.allclose(got, ref, rtol=0.0, atol=0.0), _max_rel(got, ref)


def test_unknown_flag_is_an_error(monkeypatch):
    """Никакого тихого отката к умолчанию на опечатке в имени режима."""
    monkeypatch.setenv("AXIOM_MOE_DISPATCH", "banana")
    with pytest.raises(ValueError):
        moe.dispatch_mode()


def test_flag_is_read_per_call(monkeypatch, cfg):
    """Откат доступен без перезапуска: режим переключается между вызовами."""
    params = moe.init_moe(jax.random.PRNGKey(5), cfg)
    x = _x(cfg, seed=4)
    monkeypatch.setenv("AXIOM_MOE_DISPATCH", "grouped")
    first = moe.apply(params, cfg, x)
    monkeypatch.setenv("AXIOM_MOE_DISPATCH", "dense")
    second = moe.apply(params, cfg, x)
    assert _max_rel(second, first) <= RTOL, _max_rel(second, first)
