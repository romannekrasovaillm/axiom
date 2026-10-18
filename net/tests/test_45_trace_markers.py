"""Провенанс-метки шага — gated name scopes (ADR-054 Am.31, цикл-19).

Что этот файл пиннует:

* **гейт** ``AXIOM_TRACE_MARKERS`` читается fail-closed (``0``/``1``, пусто =
  ``0``, неизвестное — :class:`net.provenance.TraceMarkersError` *до*
  трейса/исполнения), регистр и пробелы нормализуются, явный аргумент сильнее
  окружения;
* **off — прямой вызов**: ``mark`` зовёт ``fn`` без ``jax.named_call`` (spy на
  ``jax.named_call`` падает, если его тронули), результат равен прямому вызову;
  имя scope ``probe_kda`` в HLO отсутствует;
* **on — имя в HLO**: ``jax.jit(...).lower(...).as_text(debug_info=True)``
  содержит ``probe_kda`` (имя scope живёт в location-метаданных StableHLO и
  уезжает в ``op_name`` скомпилированного HLO — то, по чему трейс атрибутирует
  ядра; в тексте *скомпилированного* HLO это видно и без debug_info);
* **тонкая карта (Am.33)**: ``compute_loss`` несёт ``probe_forward``/
  ``probe_ntp_ce``/``probe_mtp`` в обеих ветках CE; внешнего loss-scope
  ``probe_ce`` больше нет — метка вокруг всего лосса захватывала backward
  через ``transpose(jvp(probe_ce))`` и не давала отделить CE-голову;
* **побитовый паритет**: loss/градиенты/forward маленькой модели при ``off`` и
  ``on`` равны ``jnp.array_equal`` — метка диагностическая и не меняет
  численные операции (блокирующий критерий задачи);
* **неизвестное значение** падает до компиляции/исполнения, а не тихо
  оставляет немаркированный граф.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from conftest import tiny_config

from net import model, optimizer
from net import provenance
from net.provenance import TRACE_MARKERS_ENV, TraceMarkersError


@pytest.fixture(autouse=True)
def _clean_marker_env(monkeypatch):
    """Каждый тест начинает с незаданного гейта (дефолт — off)."""
    monkeypatch.delenv(TRACE_MARKERS_ENV, raising=False)


# ---------------------------------------------------------------------------
# a. markers_mode: значения, регистр/пробелы, явный аргумент сильнее env
# ---------------------------------------------------------------------------


def test_unset_and_empty_mean_off():
    assert provenance.markers_mode() is False
    assert provenance.markers_mode("") is False
    assert provenance.markers_mode("   ") is False


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0", False),
        ("1", True),
        (" 1 ", True),
        ("\t1\n", True),
        (" 0 ", False),
    ],
)
def test_values_and_whitespace(raw, expected):
    assert provenance.markers_mode(raw) is expected


@pytest.mark.parametrize("raw", ["true", "on", "yes", "2", "mark", "0,1"])
def test_unknown_value_is_fail_closed(raw):
    with pytest.raises(TraceMarkersError):
        provenance.markers_mode(raw)


def test_explicit_argument_beats_environment(monkeypatch):
    monkeypatch.setenv(TRACE_MARKERS_ENV, "0")
    assert provenance.markers_mode() is False
    assert provenance.markers_mode("1") is True
    monkeypatch.setenv(TRACE_MARKERS_ENV, "1")
    assert provenance.markers_mode() is True
    assert provenance.markers_mode("0") is False


def test_env_is_read_at_call_time(monkeypatch):
    assert provenance.markers_mode() is False
    monkeypatch.setenv(TRACE_MARKERS_ENV, "1")
    assert provenance.markers_mode() is True


# ---------------------------------------------------------------------------
# b. off: прямой вызов fn, без named_call; результат равен прямому вызову
# ---------------------------------------------------------------------------


def test_off_calls_fn_directly_without_named_call(monkeypatch):
    calls = []

    def fn(x):
        calls.append(x)
        return x * 3

    def _named_call_must_not_be_used(*args, **kwargs):  # pragma: no cover
        raise AssertionError("off: mark не должен звать jax.named_call")

    monkeypatch.setattr(provenance.jax, "named_call", _named_call_must_not_be_used)
    assert provenance.mark("kda", fn, 7) == 21
    assert calls == [7]


def test_off_result_equals_direct_call():
    x = jnp.arange(8.0)
    fn = lambda x: jnp.sum(x * 2)  # noqa: E731 — прямой вызов, не определение API

    assert jnp.array_equal(provenance.mark("kda", fn, x), fn(x))


def test_off_keeps_tuple_results():
    def fn(a, b):
        return a + b, a - b

    got = provenance.mark("mla", fn, 5, 2)
    assert got == (7, 3)


# ---------------------------------------------------------------------------
# c. on: имя scope в HLO; off: имени нет
# ---------------------------------------------------------------------------


def _lowered(x):
    return jax.jit(
        lambda x: provenance.mark("kda", lambda: jnp.sum(x * 2))
    ).lower(x)


def test_on_puts_probe_name_into_hlo(monkeypatch):
    monkeypatch.setenv(TRACE_MARKERS_ENV, "1")
    lowered = _lowered(jnp.ones((4,), jnp.float32))
    # Имя scope — в location-метаданных StableHLO (debug_info=True); XLA несёт
    # его как op_name операции, поэтому в скомпилированном HLO оно есть и так.
    assert "probe_kda" in lowered.as_text(debug_info=True)
    assert "probe_kda" in lowered.compile().as_text()


def test_off_has_no_probe_name_in_hlo(monkeypatch):
    monkeypatch.setenv(TRACE_MARKERS_ENV, "0")
    lowered = _lowered(jnp.ones((4,), jnp.float32))
    assert "probe_kda" not in lowered.as_text(debug_info=True)
    assert "probe_kda" not in lowered.compile().as_text()


# ---------------------------------------------------------------------------
# d. побитовый паритет off/on на маленькой модели
# ---------------------------------------------------------------------------


def _loss_and_grad(cfg, params, ids):
    def loss(p):
        return model.compute_loss(p, cfg, ids, chunk_size=8)

    return jax.jit(jax.value_and_grad(loss))(params)


def _assert_trees_bitwise_equal(a, b):
    leaves_a = jax.tree_util.tree_leaves(a)
    leaves_b = jax.tree_util.tree_leaves(b)
    assert len(leaves_a) == len(leaves_b)
    for x, y in zip(leaves_a, leaves_b):
        assert x.shape == y.shape and x.dtype == y.dtype
        assert jnp.array_equal(x, y), "off/on разошлись побитово"


@pytest.mark.parametrize(
    "cfg_overrides",
    [
        {},  # production [K,K,K,M]: KDA + MLA + MoE + dense MLP + AttnRes
        {"dense_standard_layers": 4, "num_kda_layers": 0, "num_mla_layers": 0},
        # ^ pure dense-standard build: ветка ``mla_dense`` (оракул MLA)
        {"ce_chunk_tokens": 8},
        # ^ chunked-CE ветка ``compute_loss``: новый call-site ``ntp_ce``
        #   вокруг ``_chunked_cross_entropy`` (и ``forward`` с emit_logits=False)
    ],
    ids=["kda-mla-moe", "dense-standard", "chunked-ce"],
)
def test_off_on_bitwise_parity(monkeypatch, cfg_overrides):
    cfg = tiny_config(**cfg_overrides)
    params = model.init_params(jr.PRNGKey(0), cfg)
    ids = jr.randint(jr.PRNGKey(1), (2, 16), 0, cfg.vocab_size)

    monkeypatch.setenv(TRACE_MARKERS_ENV, "0")
    loss_off, grads_off = _loss_and_grad(cfg, params, ids)
    logits_off = jax.jit(lambda p: model.forward(p, cfg, ids, chunk_size=8))(params)

    monkeypatch.setenv(TRACE_MARKERS_ENV, "1")
    loss_on, grads_on = _loss_and_grad(cfg, params, ids)
    logits_on = jax.jit(lambda p: model.forward(p, cfg, ids, chunk_size=8))(params)

    assert jnp.array_equal(loss_off, loss_on)
    _assert_trees_bitwise_equal(grads_off, grads_on)
    assert jnp.array_equal(logits_off, logits_on)


def test_loss_composition_parity_matches_train_loop(monkeypatch):
    """Та же композиция, что в ``train_loop.loss_fn``: прямой ``compute_loss``.

    Внешней метки ``ce`` больше нет (Am.33): тонкие метки живут внутри
    ``compute_loss``.  Здесь пиннуется побитовый паритет off/on ровно этой
    композиции — вместе с новыми call-site ``forward``/``ntp_ce``/``mtp``.
    """
    cfg = tiny_config()
    params = model.init_params(jr.PRNGKey(0), cfg)
    ids = jr.randint(jr.PRNGKey(1), (2, 16), 0, cfg.vocab_size)

    def loss(p):
        return model.compute_loss(p, cfg, ids, chunk_size=8)

    monkeypatch.setenv(TRACE_MARKERS_ENV, "0")
    loss_off, grads_off = jax.jit(jax.value_and_grad(loss))(params)
    monkeypatch.setenv(TRACE_MARKERS_ENV, "1")
    loss_on, grads_on = jax.jit(jax.value_and_grad(loss))(params)

    assert jnp.array_equal(loss_off, loss_on)
    _assert_trees_bitwise_equal(grads_off, grads_on)


def test_all_component_scopes_visible_in_loss_hlo(monkeypatch):
    """Все обёрнутые границы действительно дают scope ``probe_<name>``.

    Тонкая карта Am.33: ``probe_forward``/``probe_ntp_ce``/``probe_mtp`` в
    обеих ветках CE; внешнего loss-scope ``probe_ce`` больше нет — метка
    вокруг всего ``compute_loss`` захватывала backward через
    ``transpose(jvp(probe_ce))`` и не позволяла отделить CE-голову.
    """
    monkeypatch.setenv(TRACE_MARKERS_ENV, "1")
    cfg = tiny_config()
    params = model.init_params(jr.PRNGKey(0), cfg)
    ids = jr.randint(jr.PRNGKey(1), (2, 16), 0, cfg.vocab_size)

    text = jax.jit(
        lambda p: model.compute_loss(p, cfg, ids, chunk_size=8)
    ).lower(params).as_text(debug_info=True)
    for name in (
        "forward", "ntp_ce", "mtp", "kda", "mla", "moe", "mlp", "attnres",
    ):
        assert f"probe_{name}" in text, f"scope probe_{name} отсутствует в HLO"
    assert "probe_ce" not in text, "внешний loss-scope probe_ce вернулся"

    # Chunked-ветка: ``forward`` без логитов + ``_chunked_cross_entropy`` —
    # те же три тонких имени, внешнего ``probe_ce`` нет.
    chunked_cfg = dataclasses.replace(cfg, ce_chunk_tokens=8)
    chunked = jax.jit(
        lambda p: model.compute_loss(p, chunked_cfg, ids, chunk_size=8)
    ).lower(params).as_text(debug_info=True)
    for name in ("forward", "ntp_ce", "mtp"):
        assert f"probe_{name}" in chunked, (
            f"chunked-ветка: scope probe_{name} отсутствует в HLO"
        )
    assert "probe_ce" not in chunked, "внешний loss-scope probe_ce вернулся"

    dense_cfg = tiny_config(
        dense_standard_layers=4, num_kda_layers=0, num_mla_layers=0
    )
    dense_params = model.init_params(jr.PRNGKey(0), dense_cfg)
    dense = jax.jit(
        lambda p: model.compute_loss(p, dense_cfg, ids, chunk_size=8)
    ).lower(dense_params)
    assert "probe_mla_dense" in dense.as_text(debug_info=True)


def test_off_has_no_thin_probe_names_in_loss_hlo(monkeypatch):
    """При ``off`` тонких имён в HLO нет — граф и числа побитово прежние."""
    monkeypatch.setenv(TRACE_MARKERS_ENV, "0")
    cfg = tiny_config()
    params = model.init_params(jr.PRNGKey(0), cfg)
    ids = jr.randint(jr.PRNGKey(1), (2, 16), 0, cfg.vocab_size)

    text = jax.jit(
        lambda p: model.compute_loss(p, cfg, ids, chunk_size=8)
    ).lower(params).as_text(debug_info=True)
    for name in ("forward", "ntp_ce", "mtp", "ce"):
        assert f"probe_{name}" not in text


# ---------------------------------------------------------------------------
# d2. оптимизатор: метка ``opt`` на границе шага (Am.33)
# ---------------------------------------------------------------------------


def _random_grads(params):
    """Детерминированные ненулевые градиенты той же структуры, что параметры."""
    leaves, treedef = jax.tree_util.tree_flatten(params)
    keys = jr.split(jr.PRNGKey(2), len(leaves))
    grads = [jr.normal(k, leaf.shape, leaf.dtype) for k, leaf in zip(keys, leaves)]
    return jax.tree_util.tree_unflatten(treedef, grads)


def test_opt_marker_names_the_optimizer_step(monkeypatch):
    """``mark("opt", step_fn)`` даёт ``probe_opt`` в lowered HLO шага.

    В тренировочном цикле ``step_fn`` уже jit-функция
    (:func:`net.optimizer.make_step`): ``jax.named_call`` расширяет name stack
    только на трассировке, а тело готового jit не перестраивается — поэтому
    имя внутрь модуля шага не проникает (шаг в дампе остаётся модулем своего
    jit — ``jit(step)``/``jit_step``; по нему оптимизатор и атрибутируется).
    Здесь пиннуется сам механизм метки: шаг, трассируемый внутри jit, несёт
    ``probe_opt`` на границе.
    """
    monkeypatch.setenv(TRACE_MARKERS_ENV, "1")
    cfg = tiny_config()
    params = model.init_params(jr.PRNGKey(0), cfg)
    grads = _random_grads(params)
    state = optimizer.init_state(params, key_seed=0)
    step = optimizer.make_step(cfg, jit=False)

    def outer(master, grads_, state_):
        return provenance.mark("opt", lambda: step(master, grads_, state_, 1e-3))

    text = jax.jit(outer).lower(params, grads, state).as_text(debug_info=True)
    assert "probe_opt" in text


def test_train_loop_wraps_every_optimizer_call_site():
    """Оба call-site шага (2 и 3 выхода) помечены ``provenance.mark("opt")``."""
    source = (Path(__file__).resolve().parents[1] / "train_loop.py").read_text()
    assert len(re.findall(r'provenance\.mark\(\s*"opt"', source)) >= 3
    assert "master, state = provenance.mark(" in source
    assert "master, state, params = provenance.mark(" in source


def test_off_on_bitwise_parity_optimizer_step(monkeypatch):
    """Новый call-site ``mark("opt", step_fn)``: off/on — побитовый паритет."""
    cfg = tiny_config()
    params = model.init_params(jr.PRNGKey(0), cfg)
    grads = _random_grads(params)
    state = optimizer.init_state(params, key_seed=0)
    step = optimizer.make_step(cfg, jit=False)

    def run():
        return jax.jit(
            lambda m, g, s: provenance.mark("opt", lambda: step(m, g, s, 1e-3))
        )(params, grads, state)

    monkeypatch.setenv(TRACE_MARKERS_ENV, "0")
    off = run()
    monkeypatch.setenv(TRACE_MARKERS_ENV, "1")
    on = run()
    _assert_trees_bitwise_equal(off, on)


# ---------------------------------------------------------------------------
# e. неизвестное значение — до компиляции/исполнения
# ---------------------------------------------------------------------------


def test_unknown_env_raises_before_trace(monkeypatch):
    monkeypatch.setenv(TRACE_MARKERS_ENV, "mark")
    called = []

    def fn(x):  # pragma: no cover — до вызова дело дойти не должно
        called.append(x)
        return x

    with pytest.raises(TraceMarkersError):
        jax.jit(lambda x: provenance.mark("kda", fn, x)).lower(jnp.ones((3,)))
    assert called == []


def test_unknown_env_raises_before_any_execution(monkeypatch):
    monkeypatch.setenv(TRACE_MARKERS_ENV, "on")
    called = []

    def fn():  # pragma: no cover — до вызова дело дойти не должно
        called.append(1)
        return jnp.zeros(())

    with pytest.raises(TraceMarkersError):
        provenance.mark("ce", fn)
    assert called == []
