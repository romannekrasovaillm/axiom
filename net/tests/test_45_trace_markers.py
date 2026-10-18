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
* **побитовый паритет**: loss/градиенты/forward маленькой модели при ``off`` и
  ``on`` равны ``jnp.array_equal`` — метка диагностическая и не меняет
  численные операции (блокирующий критерий задачи);
* **неизвестное значение** падает до компиляции/исполнения, а не тихо
  оставляет немаркированный граф.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from conftest import tiny_config

from net import model
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
    ],
    ids=["kda-mla-moe", "dense-standard"],
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


def test_ce_mark_parity_matches_train_loop_composition(monkeypatch):
    """Та же композиция, что в ``train_loop.loss_fn``: ``mark("ce", compute_loss)``."""
    cfg = tiny_config()
    params = model.init_params(jr.PRNGKey(0), cfg)
    ids = jr.randint(jr.PRNGKey(1), (2, 16), 0, cfg.vocab_size)

    def loss(p):
        return provenance.mark(
            "ce", lambda: model.compute_loss(p, cfg, ids, chunk_size=8)
        )

    monkeypatch.setenv(TRACE_MARKERS_ENV, "0")
    loss_off, grads_off = jax.jit(jax.value_and_grad(loss))(params)
    monkeypatch.setenv(TRACE_MARKERS_ENV, "1")
    loss_on, grads_on = jax.jit(jax.value_and_grad(loss))(params)

    assert jnp.array_equal(loss_off, loss_on)
    _assert_trees_bitwise_equal(grads_off, grads_on)


def test_all_component_scopes_visible_in_loss_hlo(monkeypatch):
    """Все обёрнутые границы действительно дают scope ``probe_<name>``."""
    monkeypatch.setenv(TRACE_MARKERS_ENV, "1")
    cfg = tiny_config()
    params = model.init_params(jr.PRNGKey(0), cfg)
    ids = jr.randint(jr.PRNGKey(1), (2, 16), 0, cfg.vocab_size)

    def loss(p):
        return provenance.mark(
            "ce", lambda: model.compute_loss(p, cfg, ids, chunk_size=8)
        )

    text = jax.jit(loss).lower(params).as_text(debug_info=True)
    for name in ("ce", "kda", "mla", "moe", "mlp", "attnres"):
        assert f"probe_{name}" in text, f"scope probe_{name} отсутствует в HLO"

    dense_cfg = tiny_config(
        dense_standard_layers=4, num_kda_layers=0, num_mla_layers=0
    )
    dense_params = model.init_params(jr.PRNGKey(0), dense_cfg)
    dense = jax.jit(
        lambda p: provenance.mark(
            "ce",
            lambda: model.compute_loss(p, dense_cfg, ids, chunk_size=8),
        )
    ).lower(dense_params)
    assert "probe_mla_dense" in dense.as_text(debug_info=True)


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
