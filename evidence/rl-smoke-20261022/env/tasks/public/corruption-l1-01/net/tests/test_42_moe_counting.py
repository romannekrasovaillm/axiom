"""ADR-054 Am.23 (цикл-14, веха-11) — MoE counting-sort вместо argsort-группировки.

Разбор остатка ядер (Am.23): CUB radix-sort — ~270 ядер/шаг (``argsort``
группировки ``_routed_experts_grouped``), при копеечном GPU-времени цена — хост-
эпизоды.  Дельта вехи-11: ``AXIOM_MOE_COUNTING=1`` (дефолт 0) заменяет
``jnp.argsort(expert_of, stable=True)`` на counting-sort — ``bincount``
(размеры групп) + ``cumsum`` (ранги «по появлению» + смещения блоков) +
``scatter``/``gather`` — без сорта как такового.

Семантика не меняется: counting-sort **стабилен по построению**, и его
раскладка совпадает с раскладкой stable-argsort **побитово** (порядок внутри
эксперта — порядок появления; ``pos`` есть обратная перестановка к stable
argsort).  Детерминизм обязателен и выполнен: все операции — elementwise/scan/
scatter с однозначным порядком, CUB-сорт (и его ``CopyOffsets`` host-эпизод) не
вызывается.

Что пинуется здесь (CPU, без GPU):

* **(а) гейт-off — побитово прежний путь**: ``AXIOM_MOE_COUNTING=0`` / не
  задан — прежний ``argsort`` (сквозной пин гейт-off против базовой ревизии —
  ``test_41_optimizer_fuse.py::test_a_...``, там нога идёт через MoE);
* **(б) паритет counting ↔ argsort**: полный выход ``moe.apply`` — побитово
  (допуск 0; индексы экспертов и раскладка — тем более), на двух конфигурациях
  и под jit; перестановка counting-sort равна обратной к stable argsort на
  вырожденных входах (пустой батч, один эксперт, без повторов/с дублями);
* **(в) гейт-контракт** ``AXIOM_MOE_COUNTING`` — дефолт 0, опечатка падает
  (fail-closed), флаг читается на каждом вызове (откат без перезапуска,
  форма ``AXIOM_MOE_DISPATCH``), явный аргумент сильнее окружения;
* **(г) в трассированном графе counting-пути нет ``sort``** (и есть ``cumsum``):
  «CUB-сорта нет» — свойство графа, а не обещание; в argsort-пути ``sort``
  есть;
* **(д) compute-dtype bf16** (кампанийный режим) паритет не ломает.
"""

from __future__ import annotations

import collections
import os

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from conftest import small_config, tiny_config

from net import compute_dtype
from net import moe

#: Индексы экспертов обязаны совпадать в точности — допуск 0 (топ-2 маршрутизация).
ATOL = 0.0


def _x(cfg, n: int = 9, seed: int = 3):
    return jr.normal(jr.PRNGKey(seed), (n, cfg.hidden))


def _with_gate(mode: str | None):
    """Временный режим гейта (``None`` — снять переменную)."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        saved = os.environ.get(moe.COUNTING_MODE_ENV)
        try:
            if mode is None:
                os.environ.pop(moe.COUNTING_MODE_ENV, None)
            else:
                os.environ[moe.COUNTING_MODE_ENV] = mode
            yield
        finally:
            if saved is None:
                os.environ.pop(moe.COUNTING_MODE_ENV, None)
            else:
                os.environ[moe.COUNTING_MODE_ENV] = saved

    return _ctx()


def _reference_positions(expert_of: np.ndarray, n: int) -> np.ndarray:
    """Эталон counting-sort: позиция каждого назначения (порядок появления)."""
    pos = np.empty(expert_of.shape[0], dtype=np.int64)
    counts = np.zeros(n, dtype=np.int64)
    for i, expert in enumerate(np.asarray(expert_of)):
        pos[i] = counts[expert]
        counts[expert] += 1
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]]) if n else np.zeros(0)
    return pos + offsets[np.asarray(expert_of)]


# ---------------------------------------------------------------------------
# (в) Гейт: дефолт 0, fail-closed, чтение на каждом вызове, явный аргумент
# ---------------------------------------------------------------------------


def test_gate_default_off_fail_closed_and_explicit(monkeypatch):
    """``AXIOM_MOE_COUNTING``: дефолт ``0``; опечатка — ошибка (fail-closed)."""
    monkeypatch.delenv(moe.COUNTING_MODE_ENV, raising=False)
    assert moe.counting_sort_mode() is False
    monkeypatch.setenv(moe.COUNTING_MODE_ENV, "")
    assert moe.counting_sort_mode() is False
    assert moe.counting_sort_mode(" 1 ") is True
    assert moe.counting_sort_mode("0") is False
    monkeypatch.setenv(moe.COUNTING_MODE_ENV, "true")  # опечатка
    with pytest.raises(ValueError, match=moe.COUNTING_MODE_ENV):
        moe.counting_sort_mode()
    # явный аргумент сильнее окружения (ручка отката без правки env).
    assert moe.counting_sort_mode("0") is False
    monkeypatch.delenv(moe.COUNTING_MODE_ENV, raising=False)
    assert moe.counting_sort_mode("1") is True


def test_flag_is_read_per_call_without_restart(cfg):
    """Откат без перезапуска: режим переключается между вызовами (Am.23)."""
    params = moe.init_moe(jr.PRNGKey(1), cfg)
    x = _x(cfg, n=8, seed=4)
    with _with_gate("0"):
        argsort_out = moe.apply(params, cfg, x)
    with _with_gate("1"):
        counting_out = moe.apply(params, cfg, x)
    with _with_gate(None):
        default_out = moe.apply(params, cfg, x)
    assert np.array_equal(np.asarray(argsort_out), np.asarray(default_out)), (
        "снятие переменной не вернуло дефолтный argsort-путь"
    )
    assert np.array_equal(np.asarray(argsort_out), np.asarray(counting_out)), (
        "выход MoE зависит от режима группировки — семантика изменилась"
    )


# ---------------------------------------------------------------------------
# (б) Паритет counting ↔ argsort: перестановка и полный выход
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expert_of",
    [
        np.random.default_rng(0).integers(0, 6, size=32),   # случай
        np.array([3] * 10),                                  # всё в одного
        np.array(list(range(6)) * 3),                        # без дублей в строке
        np.array([0, 0, 5, 5, 5, 1, 1, 5, 0]),               # повторы и смена блоков
        np.zeros(1, dtype=np.int64),                         # одно назначение
        np.array([], dtype=np.int64),                        # пустой батч
    ],
    ids=["random", "one-expert", "round-robin", "duplicates", "single", "empty"],
)
def test_counting_positions_equal_inverse_stable_argsort(expert_of):
    """Перестановка counting-sort == обратная к stable argsort (допуск 0).

    ``pos`` — позиция каждого назначения в упорядоченной раскладке; равенство
    проверяется против эталонного counting-sort и против
    ``jnp.argsort(stable=True)``: стабильность — по построению, а не по
    обещанию документации JAX.
    """
    n_routed = 6
    expert_of_j = jnp.asarray(expert_of)
    group_sizes = jnp.bincount(expert_of_j, length=n_routed).astype(jnp.int32)
    pos = np.asarray(moe._counting_sort_positions(expert_of_j, group_sizes))
    ref = _reference_positions(expert_of, n_routed)
    assert np.array_equal(pos, ref), f"перестановка разошлась: {pos} vs {ref}"
    if expert_of.size:
        order = np.asarray(jnp.argsort(expert_of_j, stable=True))
        # pos[i] — место i-го назначения в порядке: order[pos[i]] == i.
        assert np.array_equal(order[pos], np.arange(expert_of.size))
    # позиции — перестановка 0..N-1 (или пустая).
    assert np.array_equal(np.sort(pos), np.arange(expert_of.size))


@pytest.mark.parametrize("cfg_factory", [tiny_config, small_config], ids=["tiny", "small"])
def test_full_apply_parity_bitwise(cfg_factory):
    """Полный ``moe.apply`` (и роутинг) — побитово равны в обоих режимах.

    Семантика топ-2 маршрутизации не меняется: тот же набор назначений и тот же
    порядок внутри эксперта (стабильность counting-sort = порядок появления).
    Допуск 0 на индексах — прямое требование вехи-11.
    """
    cfg = cfg_factory()
    params = moe.init_moe(jr.PRNGKey(5), cfg)
    x = _x(cfg, n=37, seed=6)
    with _with_gate("0"):
        out_argsort = moe.apply(params, cfg, x)
        indices_argsort = moe.routing_indices(params, cfg, x)
        load_argsort = moe.load_fraction(params, cfg, x)
    with _with_gate("1"):
        out_counting = moe.apply(params, cfg, x)
        indices_counting = moe.routing_indices(params, cfg, x)
        load_counting = moe.load_fraction(params, cfg, x)
    assert np.array_equal(np.asarray(indices_argsort), np.asarray(indices_counting)), (
        "индексы top-k разъехались — маршрутизация изменилась"
    )
    assert np.array_equal(np.asarray(load_argsort), np.asarray(load_counting))
    assert np.array_equal(np.asarray(out_argsort), np.asarray(out_counting)), (
        f"выход MoE не побитово равен: max|d|="
        f"{float(jnp.max(jnp.abs(out_argsort - out_counting))):.3e}"
    )


def test_full_apply_parity_under_jit(cfg):
    """То же под ``jax.jit``: fused-путь не «оптимизируется» в другой результат."""
    params = moe.init_moe(jr.PRNGKey(7), cfg)
    x = _x(cfg, n=16, seed=8)
    with _with_gate("0"):
        argsort_out = jax.jit(lambda p, xx: moe.apply(p, cfg, xx))(params, x)
    with _with_gate("1"):
        counting_out = jax.jit(lambda p, xx: moe.apply(p, cfg, xx))(params, x)
    assert np.array_equal(np.asarray(argsort_out), np.asarray(counting_out)), (
        "под jit выходы режимов разошлись"
    )


# ---------------------------------------------------------------------------
# (г) «CUB-сорта нет» — свойство трассированного графа
# ---------------------------------------------------------------------------


def _walk_primitives(jaxpr) -> collections.Counter:
    """Примитивы трассировки, включая вложенные closed-jaxpr (``jit``-янцы)."""
    counter: collections.Counter = collections.Counter()

    def walk(jp):
        for eqn in jp.jaxpr.eqns:
            counter[str(eqn.primitive)] += 1
            params = eqn.params if isinstance(eqn.params, dict) else {}
            for value in params.values():
                for v in value if isinstance(value, (tuple, list)) else (value,):
                    if hasattr(v, "jaxpr"):
                        walk(v)

    walk(jaxpr)
    return counter


def test_counting_path_has_no_sort_op_and_argsort_has_it(cfg):
    """В counting-графе нет ``sort`` (и есть ``cumsum``); в argsort-графе — есть.

    Это механическая сторона «CUB-сорта нет»: ``jnp.argsort`` понижается в
    ``sort`` (на GB10 — ``DeviceRadixSort`` с pageable-копией offsets,
    ADR-054 Am.2/Am.23), counting-путь эмитит только nincount/cumsum/scatter.
    """
    params = moe.init_moe(jr.PRNGKey(9), cfg)
    x = _x(cfg, n=12, seed=10)
    _, topk, sel_p, _ = moe._dispatch(params, cfg, x)
    z = jnp.zeros((12, cfg.moe_latent_dim))

    def trace_for(gate_mode: str):
        with _with_gate(gate_mode):
            # Свежая функция на режим: make_jaxpr кешируется по объекту функции.
            def fn(p, zz, t, s, _cfg=cfg):
                return moe._routed_experts_grouped(p, _cfg, zz, t, s)

            fn.__name__ = f"routed_{gate_mode}"
            return _walk_primitives(jax.make_jaxpr(fn)(params, z, topk, sel_p))

    argsort_counts = trace_for("0")
    counting_counts = trace_for("1")
    print(f"[graph] argsort: {dict(argsort_counts)}\n[graph] counting: {dict(counting_counts)}")
    assert argsort_counts["sort"] >= 1, "argsort-путь перестал эмитить sort — путь не тот"
    assert counting_counts["sort"] == 0, "counting-путь всё ещё содержит sort"
    assert counting_counts["cumsum"] >= 2, "counting-путь не содержит cumsum — раскладка пропала"
    assert counting_counts["ragged_dot_general"] == argsort_counts["ragged_dot_general"] >= 3, (
        "исполнение экспертов изменилось вместе с группировкой (ragged_dot)"
    )


# ---------------------------------------------------------------------------
# (д) compute-dtype bf16: паритет не ломается
# ---------------------------------------------------------------------------


def test_bf16_compute_dtype_parity(cfg):
    """Кампанийный bf16 compute-dtype: паритет остаётся побитовым.

    Локальный XLA:CPU не исполняет fused bf16-GEMM (``DotThunk: Unsupported
    element type`` — измерено 08.10 на jax 0.10.2, см. ``test_29_compute_dtype``,
    секция (b)): там числовая нога скипается с той же формулировкой, а форма
    выхода сверяется абстрактно — «путь не другой» остаётся проверенным без
    исполнения.  Группировка counting-sort от compute-dtype не зависит:
    раскладка — те же gather/scatter над индексами.
    """
    params = moe.init_moe(jr.PRNGKey(11), cfg)
    x = _x(cfg, n=19, seed=12)
    saved = os.environ.get(compute_dtype.MODE_ENV)
    os.environ[compute_dtype.MODE_ENV] = compute_dtype.BF16
    outputs = None
    shapes = None
    try:
        try:
            with _with_gate("0"):
                argsort_out = np.asarray(moe.apply(params, cfg, x))
            with _with_gate("1"):
                counting_out = np.asarray(moe.apply(params, cfg, x))
            outputs = (argsort_out, counting_out)
        except jax.errors.JaxRuntimeError as exc:
            if "Unsupported element type for DotThunk" not in str(exc):
                raise
            shapes = jax.eval_shape(lambda p, xx: moe.apply(p, cfg, xx), params, x)
    finally:
        if saved is None:
            os.environ.pop(compute_dtype.MODE_ENV, None)
        else:
            os.environ[compute_dtype.MODE_ENV] = saved
    if outputs is None:
        assert np.dtype(shapes.dtype) == np.dtype(jnp.float32)
        pytest.skip(
            "локальный XLA:CPU не исполняет fused bf16-GEMM (DotThunk); "
            "числовая проверка паритета — на стенде GB10"
        )
    assert np.array_equal(outputs[0], outputs[1]), (
        "bf16 compute-dtype: выходы режимов разошлись"
    )
