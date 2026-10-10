"""Флаг решения UT-системы: ``AXIOM_KDA_SOLVE=jax|pallas`` (интеграция кернела).

Рычаг MFU-55 «первый Pallas-кернел в KDA»: XLA-путь порождает в графе 115 200 мелких
ядер ``batch_trsm_left_kernel<...,64,4,...>`` (+230 400 ``MakeBatchPointers``), поэтому
в ``net/kda.py`` появилась единая точка решения ``(I + L)^{-1}`` с ручкой отката.
Этот файл пинит то, ради чего ручка существует:

* (а) **дефолт не изменился**: без переменной окружения идёт XLA-путь
      (``jnp.linalg.inv`` + свёртки) — семантика по умолчанию прежняя;
* (б) ``AXIOM_KDA_SOLVE=pallas`` действительно уводит решение в **дифференцируемый**
      путь ``net.kernels.kda_ut_solve.solve`` (``custom_vjp``: вперёд — кернел, назад —
      аналитический adjoint через ``kernel_t``) — одним вызовом, с
      **конкатенированной** правой частью ``[xw | vw]`` (не два вызова);
      голый ``kernel`` без JVP на этом пути не вызывается (он и валил обучение с
      «Linearization failed to produce known values for all output primals»);
* (в) **градиент протекает**: ``jax.grad`` через ``_ut_solve_pair`` конечен и
      совпадает с дефолтным (``jax``) путём в пределах допуска;
* (в) неизвестное значение флага — явная ошибка, без тихого отката к умолчанию;
* (г) математический паритет: результат кернела и XLA-пути совпадает в пределах
      допусков; на CPU хост-путь кернела — ``solve_jax`` с тем же пином точности,
      поэтому вердикт паритета относится к программе кернела, а не к другой арифметике.

GPU не нужен: всё исполняется на CPU-бэкенде стенда.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from net import kda
from net.kernels import kda_ut_solve


def _small():
    """Малые формы для юнит-уровня: ``a=(H,C,C)``, ``xw=(H,C,dk)``, ``vw=(H,C,dv)``."""
    H, C, dk, dv = 2, 8, 4, 4
    key = jr.PRNGKey(0)
    k1, k2, k3 = jr.split(key, 3)
    l_mat = jnp.tril(jr.normal(k1, (H, C, C)) * 0.05, -1)  # строго нижняя
    xw = jr.normal(k2, (H, C, dk))
    vw = jr.normal(k3, (H, C, dv))
    return l_mat, xw, vw


def test_default_mode_is_jax(monkeypatch):
    """Дефолт обязан остаться прежним: ``jax`` (условие приёмки)."""
    monkeypatch.delenv("AXIOM_KDA_SOLVE", raising=False)
    assert kda.kda_solve_mode() == "jax"


def test_unknown_flag_is_an_error(monkeypatch):
    """Опечатка в значении флага — ошибка, а не молчаливый откат."""
    monkeypatch.setenv("AXIOM_KDA_SOLVE", "banana")
    with pytest.raises(ValueError):
        kda.kda_solve_mode()


def test_default_path_uses_the_xla_inverse(monkeypatch):
    """Без флага решение идёт XLA-путём, кернел не вызывается вовсе."""
    monkeypatch.delenv("AXIOM_KDA_SOLVE", raising=False)
    l_mat, xw, vw = _small()

    calls = {"jax": 0, "kernel": 0}
    real_jax = kda._solve_pair_jax

    def counting_jax(a, x, v):
        calls["jax"] += 1
        return real_jax(a, x, v)

    monkeypatch.setattr(kda, "_solve_pair_jax", counting_jax)
    monkeypatch.setattr(kda_ut_solve, "solve",
                        lambda *a, **k: calls.__setitem__("kernel", calls["kernel"] + 1))

    w, u = kda._ut_solve_pair(l_mat, xw, vw)
    assert calls == {"jax": 1, "kernel": 0}
    assert w.shape == xw.shape and u.shape == vw.shape


def test_pallas_flag_routes_one_solve_call_with_concatenated_rhs(monkeypatch):
    """``pallas``: ровно один вызов дифференцируемого ``solve``, справа — ``[xw | vw]``."""
    monkeypatch.setenv("AXIOM_KDA_SOLVE", "pallas")
    l_mat, xw, vw = _small()

    seen = {"calls": 0, "rhs_width": None, "a_shape": None}
    real_solve = kda_ut_solve.solve

    def counting_solve(a, b):
        seen["calls"] += 1
        seen["a_shape"] = a.shape
        seen["rhs_width"] = b.shape[-1]
        return real_solve(a, b)

    monkeypatch.setattr(kda_ut_solve, "solve", counting_solve)
    w, u = kda._ut_solve_pair(l_mat, xw, vw)

    assert seen["calls"] == 1, "правая часть обязана идти одним вызовом (конкатенация)"
    assert seen["rhs_width"] == xw.shape[-1] + vw.shape[-1]
    assert seen["a_shape"] == (2, 8, 8)
    assert w.shape == xw.shape and u.shape == vw.shape


def test_pair_parity_between_modes(monkeypatch):
    """Кернел и XLA-путь решают одну и ту же систему — расхождение только в арифметике."""
    l_mat, xw, vw = _small()

    monkeypatch.setenv("AXIOM_KDA_SOLVE", "jax")
    w_jax, u_jax = kda._ut_solve_pair(l_mat, xw, vw)

    monkeypatch.setenv("AXIOM_KDA_SOLVE", "pallas")
    w_ker, u_ker = kda._ut_solve_pair(l_mat, xw, vw)

    assert jnp.allclose(w_ker, w_jax, rtol=1e-5, atol=1e-6)
    assert jnp.allclose(u_ker, u_jax, rtol=1e-5, atol=1e-6)


def test_layer_level_parity_between_modes(monkeypatch, cfg):
    """Слой целиком: обе формы (``wyut`` и ``chunked_cc``) совпадают между режимами."""
    from net.tests.test_kda_wyut import _params, _x  # те же фикстуры, что у формы wyut

    p = _params(cfg)
    x = _x(cfg, cfg.kda_wyut_chunk * 2 if hasattr(cfg, "kda_wyut_chunk") else 64)

    for apply_form in (kda.apply_wyut, kda.apply_chunked_cc):
        monkeypatch.setenv("AXIOM_KDA_SOLVE", "jax")
        ref = apply_form(p, cfg, x)
        monkeypatch.setenv("AXIOM_KDA_SOLVE", "pallas")
        got = apply_form(p, cfg, x)
        assert jnp.allclose(got, ref, rtol=2e-2, atol=2e-3), apply_form.__name__


def _loss(l_mat, xw, vw):
    w, u = kda._ut_solve_pair(l_mat, xw, vw)
    return jnp.sum(w**2) + jnp.sum(u**2)


def test_gradient_flows_through_both_modes(monkeypatch):
    """Градиент протекает: конечен и совпадает с дефолтным путём в пределах допуска.

    Это проверка причины правки: голый ``pallas_call`` не даёт JVP, и ``jax.grad``
    по нему падал («Linearization failed to produce known values for all output
    primals»). Дифференцируемый ``solve`` (custom_vjp, backward — аналитический
    adjoint через ``kernel_t``) обязан дать конечный градиент, согласованный с XLA-путём.
    """
    l_mat, xw, vw = _small()

    monkeypatch.setenv("AXIOM_KDA_SOLVE", "jax")
    g_jax = jax.grad(_loss, argnums=(0, 1, 2))(l_mat, xw, vw)

    monkeypatch.setenv("AXIOM_KDA_SOLVE", "pallas")
    g_pallas = jax.grad(_loss, argnums=(0, 1, 2))(l_mat, xw, vw)

    for got, ref in zip(g_pallas, g_jax):
        assert bool(jnp.all(jnp.isfinite(got))), "градиент pallas-пути не конечен"
        assert bool(jnp.all(jnp.isfinite(ref))), "градиент jax-пути не конечен"

    # Правые части (xw, vw) сравниваются целиком.
    for got, ref in zip(g_pallas[1:], g_jax[1:]):
        assert jnp.allclose(got, ref, rtol=2e-2, atol=1e-4), float(
            jnp.max(jnp.abs(got - ref)))

    # Носитель L — строго нижний треугольник (в модели L = where(strict_tril, ...)).
    # Верхний треугольник: jnp.linalg.inv инвертирует ПОЛНУЮ матрицу, поэтому у XLA-пути
    # там ненулевой градиент — артефакт dense-инверсии, который модель не использует;
    # кернел решает треугольную систему и возвращает там строго ноль. Сравнение
    # градиента по L ведётся на носителе, и разница семантики зафиксирована явно.
    mask = jnp.tril(jnp.ones_like(l_mat, dtype=bool), -1)
    assert jnp.allclose(g_pallas[0][mask], g_jax[0][mask], rtol=2e-2, atol=1e-4), float(
        jnp.max(jnp.abs(g_pallas[0][mask] - g_jax[0][mask])))
    upper_pallas = jnp.where(mask, jnp.zeros_like(g_pallas[0]), g_pallas[0])
    assert jnp.allclose(upper_pallas, jnp.zeros_like(upper_pallas), atol=0.0), (
        "кернел обязан давать строго нулевой градиент вне носителя L")
