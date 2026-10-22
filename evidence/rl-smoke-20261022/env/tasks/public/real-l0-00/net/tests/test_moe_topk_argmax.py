"""Селектор top-k роутера MoE без CUB-сорта: итеративный argmax (ADR-054, стадия-2a).

``net/moe.py:_dispatch`` выбирает экспертов через ``jax.lax.top_k``. На GB10 тот
понижается в CUB ``DeviceRadixSort`` custom call, чей шаг ``CopyOffsets`` гоняет
offsets-массив ``(N+1)``-слов pageable-копией host→device; gdb-пробы стадии-2a
показывают ``CubSortPairsExecute`` → ``CopyOffsets`` → ``cudaMemcpyAsync``,
а профиль шага относит к этим копиям ~сотни мс-столлов хост-потока (ADR-054
Am.2 — submission-контеншн; ~14 с/шаг MoE-столлов). Замена — итеративный
``argmax`` с маскированием (:func:`net.moe._topk_argmax`), окно малых ``k``/``n``
и фолбэк ``jax.lax.top_k`` (:func:`net.moe._topk_indices`); A/B-переключатель —
``AXIOM_MOE_TOPK=argmax|jax``, читается **при импорте** (``jax`` — референс).

Этот файл пинит паритет нового селектора с референсом (CPU-бэкенд стенда):

* (а) индексы: ``_topk_argmax`` == ``jax.lax.top_k`` на случайных матрицах
      N=1000, n=12, k=2 — включая дубликаты скоров, ``±inf`` и вырожденные
      строки (все значения равны / все ``-inf`` / ``+inf``+``-inf``);
* (б) веса: выходы ``_dispatch`` (индексы, ``sel_p``, ``p_full``) **побитово**
      совпадают в режимах ``argmax`` и ``jax`` — и ``sel_p`` равны нормировке по
      выбранным скорам (публичный путь);
* (в) окно применимости: при ``k > ARGMAX_MAX_K`` / ``n > ARGMAX_MAX_N`` и в
      режиме ``jax`` селектор уходит на фолбэк ``lax.top_k`` (spy на
      ``_topk_argmax``);
* (г) селектор не порождает примитива ``top_k``: трассировка ``_topk_argmax``
      не содержит ``top_k``, фолбэк — содержит;
* (д) флаг окружения читается при импорте (сабпроцесс), неизвестное значение —
      ошибка (fail-closed, без тихого отката к умолчанию).

``net/tests/test_32_moe_grouped_topk.py`` обязан быть зелёным в обоих режимах —
это прогон-матрица кейса (``AXIOM_MOE_TOPK=jax`` и ``=argmax``), сам модуль
дополнительно не дублируется.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from conftest import small_config

from net import moe

#: Числа из формулировки дельты и кампании: роутер MoE на 12 экспертов, top-2.
N_ROWS, N_EXPERTS, TOP_K = 1000, 12, 2

#: Корень репозитория — для сабпроцессных проверок импорта (пути от теста).
REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# (а) паритет индексов с референсом jax.lax.top_k
# --------------------------------------------------------------------------- #


def _random_scores(seed: int) -> jnp.ndarray:
    """``(1000, 12)`` матрица скоров: случайность, дубликаты, ``±inf``."""
    rng = jr.PRNGKey(seed)
    base = jr.normal(rng, (N_ROWS, N_EXPERTS))
    # Дубликаты: колонки 4 и 9 равны на каждой второй строке.
    dup = base.at[::2, 9].set(base[::2, 4])
    # ±inf: строка из одних -inf, строка с +inf/-inf, «полу-бесконечные» строки.
    inf = dup.at[::131, :].set(-jnp.inf)
    inf = inf.at[::211, 0].set(jnp.inf)
    inf = inf.at[::211, 1].set(-jnp.inf)
    return inf


def test_argmax_selector_matches_lax_top_k_on_random_scores():
    """N=1000×n=12×k=2: индексы совпадают с CUB-референсом как множества и порядок."""
    scores = _random_scores(0)
    got = moe._topk_argmax(scores, TOP_K)
    ref = jax.lax.top_k(scores, TOP_K)[1]
    assert got.shape == ref.shape == (N_ROWS, TOP_K)
    assert jnp.array_equal(got, ref), (
        "итеративный argmax разошёлся с jax.lax.top_k: "
        f"{(jnp.nonzero(jnp.any(got != ref, axis=-1))[0][:5]).tolist()}"
    )


def test_argmax_selector_matches_lax_top_k_on_ties_and_infinities():
    """Вырожденные строки: равные скоры, все ``-inf``, ``+inf``, ``±inf`` вперемешку."""
    rows = jnp.array(
        [
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            [-jnp.inf] * N_EXPERTS,
            [jnp.inf] * N_EXPERTS,
            [jnp.inf, -jnp.inf, jnp.inf, 0.0, 1.0, -jnp.inf, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-jnp.inf, 3.0, -jnp.inf, 3.0, -jnp.inf, 3.0, -jnp.inf, 3.0, -jnp.inf, 3.0, -jnp.inf, 3.0],
            [2.0, 2.0, 1.0, 1.0, 0.0, 0.0, -1.0, -1.0, -2.0, -2.0, -3.0, -3.0],
        ]
    )
    got = moe._topk_argmax(rows, TOP_K)
    ref = jax.lax.top_k(rows, TOP_K)[1]
    assert jnp.array_equal(got, ref), f"ties/inf: got={got.tolist()} ref={ref.tolist()}"
    # Тай-брейкинг явно: при равных скорах первым идёт меньший индекс.
    assert got[0].tolist() == [0, 1]
    assert got[1].tolist() == [0, 1]


@pytest.mark.parametrize("k", [1, 2, 3])
def test_argmax_selector_matches_reference_for_other_k(k):
    """Окно допускает k<=8: паритет не только на боевом k=2."""
    scores = _random_scores(seed=k)
    got = moe._topk_argmax(scores, k)
    ref = jax.lax.top_k(scores, k)[1]
    assert jnp.array_equal(got, ref)


# --------------------------------------------------------------------------- #
# (б) паритет выходов dispatch и веса выбранных экспертов
# --------------------------------------------------------------------------- #


def _dispatch_both_modes(monkeypatch, params, cfg, x):
    """``_dispatch`` под обоими режимами селектора (без перезагрузки модуля)."""
    monkeypatch.setattr(moe, "TOPK_MODE", moe.ARGMAX_SELECTOR)
    got_argmax = moe._dispatch(params, cfg, x)
    monkeypatch.setattr(moe, "TOPK_MODE", moe.JAX_SELECTOR)
    got_jax = moe._dispatch(params, cfg, x)
    return got_argmax, got_jax


def test_dispatch_outputs_are_bit_identical_in_both_modes(monkeypatch):
    """``AXIOM_MOE_TOPK=jax`` = референс A/B: dispatch в обоих режимах совпадает."""
    cfg = small_config(moe_num_routed=N_EXPERTS)
    params = moe.init_moe(jr.PRNGKey(cfg.routing_seed), cfg)
    x = jr.normal(jr.PRNGKey(1), (N_ROWS, cfg.hidden))

    (s_a, topk_a, sel_p_a, p_full_a), (s_j, topk_j, sel_p_j, p_full_j) = (
        _dispatch_both_modes(monkeypatch, params, cfg, x)
    )

    assert jnp.array_equal(s_a, s_j)  # вход селектора — общий
    assert jnp.array_equal(topk_a, topk_j), "индексы разошлись между режимами"
    assert jnp.array_equal(sel_p_a, sel_p_j), "веса разошлись между режимами"
    assert jnp.array_equal(p_full_a, p_full_j)
    # Индексы — ровно из итеративного селектора по тем же biased-скорам.
    assert jnp.array_equal(
        topk_a, moe._topk_argmax(s_a + params.router_b, cfg.moe_top_k)
    )


def test_dispatch_weights_are_the_selected_scores(monkeypatch):
    """``sel_p`` — нормировка по выбранным (несмещённым) скорам, не перестановка."""
    cfg = small_config(moe_num_routed=N_EXPERTS)
    params = moe.init_moe(jr.PRNGKey(cfg.routing_seed), cfg)
    x = jr.normal(jr.PRNGKey(2), (256, cfg.hidden))

    monkeypatch.setattr(moe, "TOPK_MODE", moe.ARGMAX_SELECTOR)
    s, topk, sel_p, p_full = moe._dispatch(params, cfg, x)

    sel_s = jnp.take_along_axis(s, topk, axis=-1)
    expected = sel_s / (jnp.sum(sel_s, axis=-1, keepdims=True) + 1e-8)
    assert jnp.array_equal(sel_p, expected)
    # p_full — one-hot выбор, взвешенный sel_p (тот же контракт, что до правки).
    assert jnp.allclose(jnp.sum(p_full, axis=-1), 1.0, atol=1e-6)


def test_dispatch_ties_and_infinite_bias_are_stable(monkeypatch):
    """Равные скоры роутера и ``±inf`` в biased-скорах: те же индексы, тот же порядок."""
    cfg = small_config(moe_num_routed=N_EXPERTS)
    params = moe.init_moe(jr.PRNGKey(cfg.routing_seed), cfg)
    x = jr.normal(jr.PRNGKey(3), (64, cfg.hidden))

    # Точные дубликаты скоров: колонка 3 копирует колонку 1 (роутер видит равные
    # скоры у двух экспертов на каждом токене) — тай-брейкинг обязан совпасть.
    tied = params._replace(router_w=params.router_w.at[:, 3].set(params.router_w[:, 1]))
    (_, topk_a, sel_p_a, _), (_, topk_j, sel_p_j, _) = _dispatch_both_modes(
        monkeypatch, tied, cfg, x
    )
    assert jnp.array_equal(topk_a, topk_j)
    assert jnp.array_equal(sel_p_a, sel_p_j)

    # -inf у всех экспертов: выбор вырождается в «первые k индексов» в обоих режимах.
    all_neg = tied._replace(router_b=jnp.full((N_EXPERTS,), -jnp.inf))
    (_, topk_a, _, _), (_, topk_j, _, _) = _dispatch_both_modes(monkeypatch, all_neg, cfg, x)
    assert jnp.array_equal(topk_a, topk_j)
    assert jnp.all(topk_a == jnp.arange(cfg.moe_top_k)[None, :])

    # +inf у двух экспертов: +inf-выбор также тай-брейкится меньшим индексом.
    pos = tied._replace(
        router_b=jnp.zeros((N_EXPERTS,)).at[0].set(jnp.inf).at[5].set(jnp.inf)
    )
    (_, topk_a, _, _), (_, topk_j, _, _) = _dispatch_both_modes(monkeypatch, pos, cfg, x)
    assert jnp.array_equal(topk_a, topk_j)
    assert jnp.all(topk_a[:, 0] == 0)


# --------------------------------------------------------------------------- #
# (в) окно применимости и фолбэк
# --------------------------------------------------------------------------- #


def test_fallback_window_and_forced_jax_mode(monkeypatch):
    """Вне окна (k>8, n>64) и в режиме ``jax`` селектор уходит на ``lax.top_k``."""
    calls: list[tuple[tuple[int, ...], int]] = []
    real = moe._topk_argmax

    def spy(scores, k):
        calls.append((tuple(scores.shape), k))
        return real(scores, k)

    monkeypatch.setattr(moe, "_topk_argmax", spy)
    monkeypatch.setattr(moe, "TOPK_MODE", moe.ARGMAX_SELECTOR)
    small = jnp.asarray(_random_scores(7)[:4])

    moe._topk_indices(small, TOP_K)  # окно: k=2 <= 8, n=12 <= 64
    assert len(calls) == 1

    wide = jnp.zeros((4, moe.ARGMAX_MAX_N + 1))  # n > 64
    ref_wide = jax.lax.top_k(wide, TOP_K)[1]
    assert jnp.array_equal(moe._topk_indices(wide, TOP_K), ref_wide)
    assert len(calls) == 1, "n > ARGMAX_MAX_N обязан идти на фолбэк"

    assert jnp.array_equal(
        moe._topk_indices(small, moe.ARGMAX_MAX_K + 1),  # k > 8
        jax.lax.top_k(small, moe.ARGMAX_MAX_K + 1)[1],
    )
    assert len(calls) == 1, "k > ARGMAX_MAX_K обязан идти на фолбэк"

    monkeypatch.setattr(moe, "TOPK_MODE", moe.JAX_SELECTOR)  # вынужденный референс
    assert jnp.array_equal(
        moe._topk_indices(small, TOP_K), jax.lax.top_k(small, TOP_K)[1]
    )
    assert len(calls) == 1, "режим jax обязан идти на фолбэк в окне"


def test_argmax_selector_trace_has_no_top_k_primitive():
    """Ядро замены: в трассировке селектора нет примитива ``top_k``."""
    scores = jnp.asarray(_random_scores(8)[:4])

    traced = str(jax.make_jaxpr(lambda s: moe._topk_argmax(s, TOP_K))(scores))
    assert "top_k" not in traced, f"итеративный селектор всё ещё содержит top_k: {traced}"
    assert "argmax" in traced

    fallback = str(jax.make_jaxpr(lambda s: jax.lax.top_k(s, TOP_K)[1])(scores))
    assert "top_k" in fallback  # позитивный контроль самого детектора


# --------------------------------------------------------------------------- #
# (д) флаг окружения: чтение при импорте, fail-closed
# --------------------------------------------------------------------------- #


def test_env_flag_values_are_parsed_fail_closed():
    """Разбор ``AXIOM_MOE_TOPK``: дефолт argmax, регистр/пробелы, ошибка на опечатке."""
    assert moe._read_topk_mode(None) == moe.ARGMAX_SELECTOR
    assert moe._read_topk_mode("") == moe.ARGMAX_SELECTOR
    assert moe._read_topk_mode("jax") == moe.JAX_SELECTOR
    assert moe._read_topk_mode(" JAX ") == moe.JAX_SELECTOR
    assert moe._read_topk_mode("argmax") == moe.ARGMAX_SELECTOR
    with pytest.raises(ValueError):
        moe._read_topk_mode("banana")


def _import_topk_mode(env_value: str | None) -> subprocess.CompletedProcess:
    """Свежий интерпретатор: печатает ``net.moe.TOPK_MODE`` (или падает на импорте)."""
    env = os.environ.copy()
    env.pop(moe.TOPK_MODE_ENV, None)
    if env_value is not None:
        env[moe.TOPK_MODE_ENV] = env_value
    return subprocess.run(
        [sys.executable, "-c", "import net.moe as m; print(m.TOPK_MODE)"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_env_flag_is_read_at_import():
    """Переключатель действует на импорте процесса: argmax (дефолт) | jax | ошибка."""
    out_default = _import_topk_mode(None)
    assert out_default.returncode == 0, out_default.stderr[-500:]
    assert out_default.stdout.strip() == moe.ARGMAX_SELECTOR

    out_jax = _import_topk_mode(moe.JAX_SELECTOR)
    assert out_jax.returncode == 0, out_jax.stderr[-500:]
    assert out_jax.stdout.strip() == moe.JAX_SELECTOR

    out_bad = _import_topk_mode("banana")
    assert out_bad.returncode != 0, "неизвестный режим не должен импортироваться молча"
    assert moe.TOPK_MODE_ENV in (out_bad.stderr + out_bad.stdout)
