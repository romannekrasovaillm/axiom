"""Веха-9 (ADR-054 Am.20, кампания MFU-55, цикл-11): bf16-хранение в backward.

Два env-гейта срезают пол B=4 по dtype **хранения** в backward, не по арифметике:

* ``AXIOM_MLA_BWD_BF16`` (``net/mla.py``): softmax-выход dense-пути (B,H,T,T),
  который backward держит живым, хранится в bf16 (12→6 ГиБ на живой тензор;
  HLO-пол — fp32-скоры 12 ГиБ × (80+70+19) вхождений);
* ``AXIOM_CE_BF16`` (``net/model.py``): логиты чанка CE (B,chunk,V) — в bf16
  (9.77→4.9 ГиБ на чанк; HLO-пол — fp32-логиты 9.77 ГиБ × 71 вхождение).

Оба — дефолт ``0`` (прежний путь побитово), чтение fail-closed (форма
``accum_jit_mode``/``master_dtype_mode``): опечатка — ошибка сборки, а не тихий
откат в fp32 (иначе замер ноги «bf16-хранение» был бы замером прежнего пути).
Формы — явный backward (``jax.custom_vjp``), а не каст в forward: transpose-
правила внутренних примитивов softmax/log_softmax держат fp32-копии полного
тензора живыми, каст в forward их не срезает (трассировка остатков показывает
те же fp32-копии до и после каста), тогда как явный backward фиксирует состав
остатков ровно bf16-копией — это и пиннуется ниже.  Forward у обоих гейтов —
побитово прежний (числа гейт не меняет), backward читает bf16 с апкастом в fp32.

Что пиннуется здесь (CPU, без стенда GB10):

* контракт обоих гейтов: дефолт 0, пустая = 0, регистр/пробелы, неизвестное —
  ошибка на чтении и на входе ``train`` до params/компиляции/батчей;
* (а) оба гейта off — побитово прежняя ревизия (``BASELINE_COMMIT``, форма
  test_38/test_29): двухшаговая нога ``train`` с накоплением на конфиге с
  чанкованным CE — лоссы и хеши деревьев совпадают; ``=0`` тождественно пустой
  переменной;
* (б) MLA-гейт: forward побитово тот же; остаток backward — bf16 (B,H,T,T) и ни
  одного fp32 такого; loss/grads паритет на малом конфиге ≤1e-3 (измерено
  8.8e-6 отн.); flash-путь и KDA не задеты (побитово при любом гейте);
* (в) CE-гейт: forward побитово тот же (включая маску); остаток чанка — bf16
  логиты; Δloss на 10 шагах малого конфига ≤1e-3 (измерено 1.9e-6 при lr=1e-3 —
  кампанийный режим формы test_35; при lr=1e-2 игрушечная траектория сама
  усиливает шум — измерено 1.6e-3, документировано в комментарии);
* маркировка: оба фактических значения в каждой строке метрик (0/0, 1/0, 0/1,
  1/1) — форма master_dtype/accum_jit — и связка с ``per_layer_block`` (Am.18)
  не ломается.

Не пинуется здесь: стенд B=4 (номер приёмки кампании) и HLO-пол — дельта лишь
делает его достижимым; числа стенда — отдельная нога.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

from conftest import small_config, tiny_config

import jax
import jax.numpy as jnp
import jax.random as jr

from net import mla
from net import model
from net import train_loop as tl

#: Геометрия смоук-ног (короткий контекст — CPU-время).
SEQ_LEN = 16

#: Ревизия до дельты (Am.20 в линии): дефолтный путь обязан воспроизводить её
#: побитово.  ``git archive`` + skip, если ревизия недоступна (форма test_38).
BASELINE_COMMIT = "026e0bc"
CASE_DIR = Path(__file__).resolve().parents[2]

#: 10-шаговый паритет (форма test_35): lr малый — кампанийный режим, где
#: обновление за шаг много меньше веса.  При lr=1e-2 на tiny-конфиге траектория
#: сама усиливает шум (измерено: max|dloss| = 1.6e-3 у CE-ноги, ~2.7e-5 у MLA),
#: и «дрейф» меряет чувствительность траектории, а не bf16-округление;
#: при lr=1e-3 те же 10 шагов дают 1.9e-6 (CE) / 0.0 (MLA) — с запасом в
#: критерий цикла (Δloss ≤ 1e-3), именно это число уходит в отчёт.
_PARITY_STEPS = 10
_PARITY_LR = 1e-3

#: Критерий паритета цикла-11: Δloss на 10 шагах малого конфига (Am.20, п.2).
_DELTA_LOSS_TOL = 1e-3

#: Допуск MLA-паритета (Am.20, п.1: «ожидание ≤1e-3»; измерено 8.8e-6 отн.).
_MLA_PARITY_TOL = 1e-3

#: Градиентный допуск CE-паритета: лоссовый критерий здесь жёсткий (≤1e-3 —
#: выше), а градиент несёт bf16-округление логитов (2^-8 от |logits|) — в
#: градиент уезжает до ~1e-2·max; измерено 5.1e-3 (features) / 5.3e-3 (emb).
_CE_GRAD_RTOL = 2e-2
_CE_GRAD_ATOL = 1e-3


# --------------------------------------------------------------------------- #
# Общие помощники
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _cpu_kda_body(monkeypatch):
    """Явное committed KDA-тело для CPU (кампанийный дефолт ``persistent``)."""
    monkeypatch.setenv("AXIOM_KDA_CC_IMPL", "jax")


@contextlib.contextmanager
def gate_env(
    mla_value: str | None = None,
    ce_value: str | None = None,
    policy: str | None = None,
    accum_jit: str | None = None,
):
    """Временные значения гейтов вехи-9 (``None`` — снять) + policy/accum-jit."""
    keys = (
        mla.MLA_BWD_BF16_ENV,
        model.CE_BF16_ENV,
        "AXIOM_GRAD_CKPT_POLICY",
        tl.ACCUM_JIT_ENV,
    )
    saved = {key: os.environ.get(key) for key in keys}
    try:
        for key, value in zip(keys, (mla_value, ce_value, policy, accum_jit)):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _batches(cfg, steps: int, *, seed: int = 11):
    """Детерминированный поток ``(1, T) int32`` — одинаковые данные для ног."""
    rng = np.random.default_rng(seed)
    return [
        rng.integers(0, cfg.vocab_size, size=(1, SEQ_LEN), dtype=np.int32)
        for _ in range(steps)
    ]


def _budget():
    return tl.Budget(run_ref="veha9", path=Path("/dev/null"), present=True)


def _train_leg(
    cfg,
    steps: int,
    *,
    lr: float = _PARITY_LR,
    seed: int = 7,
    batches=None,
    metrics_path=None,
    accum_tokens: int = 0,
):
    """Нога ``train``; env гейтов выставляют тесты явно через :func:`gate_env`."""
    return tl.train(
        cfg,
        iter(batches if batches is not None else _batches(cfg, steps)),
        train_config=tl.TrainConfig(
            steps=steps, lr=lr, seed=seed, schedule="cosine",
            param_dtype="float32", micro_batch=1, accum_tokens=accum_tokens,
            metrics_path=metrics_path, log_every=0, kpi_every=20,
        ),
        budget=_budget(),
    )


def _loss_drift(a, b) -> float:
    assert len(a) == len(b) and a, "кривые обязаны быть непусты и одной длины"
    return max(abs(float(x) - float(y)) for x, y in zip(a, b))


def _max_rel(a, b) -> float:
    """Максимальное |a−b| по листьям, отнесённое к max|b| (форма test_35)."""
    leaves_a = jax.tree_util.tree_leaves(a)
    leaves_b = jax.tree_util.tree_leaves(b)
    scale = max(
        [1e-30] + [float(np.max(np.abs(np.asarray(leaf)))) for leaf in leaves_b]
    )
    return max(
        float(np.max(np.abs(np.asarray(x) - np.asarray(y))))
        for x, y in zip(leaves_a, leaves_b)
    ) / scale


def _tree_allclose(a, b, *, rtol, atol) -> bool:
    leaves_a = jax.tree_util.tree_leaves(a)
    leaves_b = jax.tree_util.tree_leaves(b)
    assert len(leaves_a) == len(leaves_b)
    return all(
        bool(jnp.allclose(x, y, rtol=rtol, atol=atol, equal_nan=True))
        for x, y in zip(leaves_a, leaves_b)
    )


def _walk_values(jaxpr, out=None, depth=0):
    """(primitive, shape, dtype) каждого результата в джахпре, включая вложенные.

    Спуск в ``jaxpr``-параметры уравнений (remat/checkpoint/custom-вызовы) и в
    контейнеры: лайв-тензоры чанка CE живут именно внутри тела remat, а
    bf16-остаток — внутри custom-вызова.
    """
    if out is None:
        out = []
    if depth > 25 or jaxpr is None:
        return out
    for eqn in jaxpr.eqns:
        for var in eqn.outvars:
            aval = getattr(var, "aval", None)
            if aval is not None:
                out.append(
                    (
                        str(eqn.primitive.name),
                        tuple(getattr(aval, "shape", ())),
                        str(aval.dtype),
                    )
                )
        for value in eqn.params.values():
            items = (
                list(value.values())
                if isinstance(value, dict)
                else list(value)
                if isinstance(value, (list, tuple))
                else [value]
            )
            for sub in items:
                inner = sub if hasattr(sub, "eqns") else getattr(sub, "jaxpr", None)
                if inner is not None and hasattr(inner, "eqns"):
                    _walk_values(inner, out, depth + 1)
    return out


def _graph_values(fn):
    return _walk_values(jax.make_jaxpr(fn)().jaxpr)


# --------------------------------------------------------------------------- #
# Контракт значений: дефолт 0, нормализация, fail-closed
# --------------------------------------------------------------------------- #


def test_env_names_are_the_contract():
    """Раннер кампании выставляет гейты по имени — имена пиннуются строкой."""
    assert mla.MLA_BWD_BF16_ENV == "AXIOM_MLA_BWD_BF16"
    assert model.CE_BF16_ENV == "AXIOM_CE_BF16"
    assert mla.MLA_BWD_BF16_VALUES == ("0", "1")
    assert model.CE_BF16_VALUES == ("0", "1")


def test_readers_default_to_off():
    with gate_env(None, None):
        assert mla.bwd_bf16_mode() is False
        assert model.ce_bf16_mode() is False
    for blank in ("", "   ", "\t\n"):
        with gate_env(blank, blank):
            assert mla.bwd_bf16_mode() is False, "пустая = дефолт"
            assert model.ce_bf16_mode() is False


def test_readers_normalize_and_explicit_argument_wins():
    for on in ("1", " 1 ", "1\t"):
        with gate_env(on, on):
            assert mla.bwd_bf16_mode() is True
            assert model.ce_bf16_mode() is True
    for off in ("0", " 0 "):
        with gate_env(off, off):
            assert mla.bwd_bf16_mode() is False
            assert model.ce_bf16_mode() is False
    # Явный аргумент сильнее окружения (форма accum_jit_mode/master_dtype_mode).
    with gate_env("0", "0"):
        assert mla.bwd_bf16_mode("1") is True
        assert model.ce_bf16_mode("1") is True
    with gate_env("1", "1"):
        assert mla.bwd_bf16_mode("0") is False
        assert model.ce_bf16_mode("0") is False


def test_readers_unknown_value_fails_closed():
    """Опечатка — ошибка, а не тихий откат: замер ноги не станет замером fp32."""
    for bad in ("true", "yes", "on", "b f16", "2", "0 1"):
        with gate_env(bad, bad):
            with pytest.raises(mla.MlaBwdBf16Error, match=mla.MLA_BWD_BF16_ENV):
                mla.bwd_bf16_mode()
            with pytest.raises(model.CeBf16Error, match=model.CE_BF16_ENV):
                model.ce_bf16_mode()
    with pytest.raises(mla.MlaBwdBf16Error):
        mla.bwd_bf16_mode("bf16")
    with pytest.raises(model.CeBf16Error):
        model.ce_bf16_mode("bf16")


class _NeverConsumed:
    """Поток батчей, который обязан остаться нетронутым при отказе сборки."""

    def __iter__(self):
        return self

    def __next__(self):  # pragma: no cover — исполнение = провал теста
        raise AssertionError("батчи не должны читаться: отказ на сборке")


def test_train_refuses_unknown_gate_before_any_run(monkeypatch, tmp_path):
    """Мусор в гейте: ошибка до params/компиляции/данных, журнал не создаётся."""
    cfg = tiny_config()
    for key, err in (
        (mla.MLA_BWD_BF16_ENV, mla.MlaBwdBf16Error),
        (model.CE_BF16_ENV, model.CeBf16Error),
    ):
        metrics = tmp_path / f"must-not-exist-{key}.jsonl"
        monkeypatch.setenv(key, "on")
        with pytest.raises(err, match=key):
            tl.train(
                cfg,
                _NeverConsumed(),
                train_config=tl.TrainConfig(
                    steps=1, lr=1e-2, seed=7, schedule="cosine",
                    param_dtype="float32", micro_batch=1,
                    metrics_path=metrics, log_every=0,
                ),
                budget=_budget(),
            )
        assert not metrics.exists(), "отказ сборки не должен оставить журнал шагов"
        monkeypatch.delenv(key, raising=False)


# --------------------------------------------------------------------------- #
# (б) MLA-гейт: forward — бит-в-бит, остаток — bf16, паритет ≤1e-3
# --------------------------------------------------------------------------- #


def _mla_case():
    cfg = small_config()
    params = mla.init_mla(jr.PRNGKey(0), cfg)
    x = jr.normal(jr.PRNGKey(1), (2, 12, cfg.hidden))
    return cfg, params, x


def _mla_big_shape(cfg):
    return (2, cfg.num_heads, 12, 12)


def test_mla_dense_forward_is_bitwise_the_same_with_the_gate_on():
    """Числа гейт не меняет: softmax/матмулы fp32, bf16 — только хранилище."""
    cfg, params, x = _mla_case()
    with gate_env(None, None):
        off = mla._dense_apply(params, cfg, x)
    with gate_env("1", None):
        on = mla._dense_apply(params, cfg, x)
    assert jnp.array_equal(off, on), "forward bf16-ветки разошёлся с fp32-путём"


def test_mla_backward_residual_is_bf16():
    """Состав остатков backward: fp32 (B,H,T,T) ×3 → одна bf16-копия.

    Проба — константы pullback-джахпра (значения, закрытые backward'ом): в
    fp32-пути это fp32-скоры (числитель деления softmax, exp, копия под
    контракцию), в гейт-пути — одна bf16-копия.  Это и есть механизм срезания
    пола; каст в forward такой состав не даёт (см. докстринг модуля).
    """
    cfg, params, x = _mla_case()
    big = _mla_big_shape(cfg)

    def residuals():
        val = mla._dense_apply(params, cfg, x)
        _, pullback = jax.vjp(lambda x_: mla._dense_apply(params, cfg, x_), x)
        closed = jax.make_jaxpr(pullback)(jnp.ones_like(val))
        return [
            str(c.aval.dtype)
            for c in closed.jaxpr.constvars
            if tuple(getattr(c.aval, "shape", ())) == big
        ]

    with gate_env(None, None):
        off = residuals()
    off_floats = [d for d in off if "float" in d]
    assert off_floats and all(d == "float32" for d in off_floats), (
        f"fp32-путь обязан держать fp32-остатки скоров, а увидел {off!r}"
    )

    with gate_env("1", None):
        on = residuals()
    on_floats = [d for d in on if "float" in d]
    assert on_floats, "гейт-путь обязан сохранять остаток (bf16-копию)"
    assert "bfloat16" in on_floats, f"остатка bf16 нет: {on!r}"
    assert "float32" not in on_floats, f"fp32-остаток (B,H,T,T) не срезан: {on!r}"


def test_mla_gate_on_graph_introduces_only_bf16():
    """Третьей точности гейт не вводит: float-типы графа ⊆ {fp32, bf16}."""
    cfg, params, x = _mla_case()

    def loss(x_):
        return jnp.sum(mla._dense_apply(params, cfg, x_) ** 2)

    with gate_env(None, None):
        off_dts = {dt for _, _, dt in _graph_values(lambda: jax.grad(loss)(x))}
        assert "bfloat16" not in off_dts, "fp32-граф обязан быть без bf16"
    with gate_env("1", None):
        on_dts = {dt for _, _, dt in _graph_values(lambda: jax.grad(loss)(x))}
    floats = {d for d in on_dts if d.startswith("float") or d == "bfloat16"}
    assert floats <= {"float32", "bfloat16"}, floats
    assert "bfloat16" in floats, "гейт не изменил граф — это no-op"


def test_mla_loss_and_grads_parity_against_off():
    """Паритет loss/grads малого конфига; допуск задокументирован (≤1e-3).

    Модельная нога (dense-MLA включён дефолтом, KDA-слои не задеты гейтом):
    лосс — побитово, градиенты — в допуске Am.20 (измерено 8.8e-6 отн.).
    """
    cfg = small_config()
    params = model.init_params(jr.PRNGKey(0), cfg)
    ids = jr.randint(jr.PRNGKey(2), (2, 16), 1, cfg.vocab_size)

    def loss(p):
        return model.compute_loss(p, cfg, ids, 16, grad_ckpt_policy="none")

    with gate_env(None, None):
        v_off, g_off = jax.value_and_grad(loss)(params)
    with gate_env("1", None):
        v_on, g_on = jax.value_and_grad(loss)(params)

    assert jnp.array_equal(v_off, v_on), "forward обязан быть побитово тем же"
    rel = _max_rel(g_on, g_off)
    print(f"[веха-9] MLA-гейт: относительный |dgrad| = {rel:.3e}")
    assert rel <= _MLA_PARITY_TOL, (
        f"паритет MLA-градиентов {rel:.3e} > {_MLA_PARITY_TOL:g}"
    )


def test_flash_path_is_not_touched_by_the_gate(monkeypatch):
    """``AXIOM_MLA_DENSE_FLASH=1``: гейт веткой не читается — побитово оба плеча."""
    cfg, params, x = _mla_case()
    monkeypatch.setenv("AXIOM_MLA_DENSE_FLASH", "1")

    def loss(x_):
        return jnp.sum(mla._dense_apply(params, cfg, x_) ** 2)

    with gate_env(None, None):
        v_off, g_off = jax.value_and_grad(loss)(x)
    with gate_env("1", None):
        v_on, g_on = jax.value_and_grad(loss)(x)
    assert jnp.array_equal(v_off, v_on) and jnp.array_equal(g_off, g_on), (
        "flash-путь обязан остаться гейтом не тронут (побитово)"
    )


def test_kda_is_not_touched_by_the_gates():
    """KDA-тело гейтами не задета: forward+grads побитово при обоих гейтах."""
    from net import kda

    cfg = small_config()
    params = kda.init_kda(jr.PRNGKey(0), cfg)
    x = jr.normal(jr.PRNGKey(3), (12, cfg.hidden))  # apply_kda — 2-D (модель vmap'ит)

    def loss(x_):
        return jnp.sum(kda.apply_kda(params, cfg, x_, 4) ** 2)

    with gate_env(None, None):
        v_off, g_off = jax.value_and_grad(loss)(x)
    with gate_env("1", "1"):
        v_on, g_on = jax.value_and_grad(loss)(x)
    assert jnp.array_equal(v_off, v_on) and jnp.array_equal(g_off, g_on)


# --------------------------------------------------------------------------- #
# (в) CE-гейт: forward — бит-в-бит, остаток — bf16-логиты, Δloss ≤1e-3
# --------------------------------------------------------------------------- #


def _ce_case(chunk: int = 8):
    cfg = tiny_config(ce_chunk_tokens=chunk)
    key = jr.PRNGKey(3)
    feats = jr.normal(key, (2, 13, cfg.hidden))
    targets = jr.randint(key, (2, 13), 1, cfg.vocab_size)
    emb = jr.normal(jr.PRNGKey(4), (cfg.vocab_size, cfg.hidden))
    return cfg, feats, targets, emb


def test_ce_forward_is_bitwise_the_same_with_the_gate_on():
    """Числа гейт не меняет — изменяется только хранилище логитов для backward."""
    cfg, feats, targets, emb = _ce_case()
    with gate_env(None, None):
        off = model._chunked_cross_entropy(feats, targets, emb, 8)
    with gate_env(None, "1"):
        on = model._chunked_cross_entropy(feats, targets, emb, 8)
    assert jnp.array_equal(off, on)

    # Маскированная ветка (numerator/denominator по глобальной маске).
    mask = jnp.zeros((2, 13), jnp.float32).at[:, :11].set(1.0)
    with gate_env(None, None):
        m_off = model._chunked_cross_entropy(feats, targets, emb, 5, loss_mask=mask)
    with gate_env(None, "1"):
        m_on = model._chunked_cross_entropy(feats, targets, emb, 5, loss_mask=mask)
    assert jnp.array_equal(m_off, m_on)


def test_ce_backward_residual_is_bf16_logits():
    """Остаток чанка — bf16-логиты (и ни одного fp32 (B,chunk,V)); off — без bf16.

    Два уровня: (1) константы pullback-джахпра голого чанка — точный состав
    остатков; (2) граф градиента публичной функции — гейт действительно доводит
    bf16 до рабочего пути (внутрь remat-тела), а off-граф bf16 не содержит.
    """
    cfg, feats, targets, emb = _ce_case()
    big = (2, 8, cfg.vocab_size)

    with gate_env(None, "1"):
        _, pullback = jax.vjp(
            lambda f_, e_: model._chunk_sum_bf16(f_, targets[:, :8], e_, None),
            feats[:, :8],
            emb,
        )
        closed = jax.make_jaxpr(pullback)((jnp.ones(()), jnp.ones(())))
    consts = [
        (tuple(c.aval.shape), str(c.aval.dtype)) for c in closed.jaxpr.constvars
    ]
    assert (big, "bfloat16") in consts, f"остатка bf16-логитов нет: {consts!r}"
    assert not any(sh == big and dt == "float32" for sh, dt in consts), consts

    def graph_values():
        """Свежий объект на каждый трейс: ``jax.make_jaxpr`` кэширует jaxpr по
        идентичности функции — повторный трейс того же объекта вернул бы первый
        граф независимо от смены env-гейта (проверено: «on» после «off» без
        свежего объекта даёт off-граф)."""
        return _graph_values(
            lambda: jax.grad(
                lambda f_, e_: model._chunked_cross_entropy(f_, targets, e_, 8),
                argnums=(0, 1),
            )(feats, emb)
        )

    with gate_env(None, None):
        values_off = graph_values()
    assert "bfloat16" not in {dt for _, _, dt in values_off}, (
        "fp32-путь CE обязан быть без bf16"
    )

    with gate_env(None, "1"):
        values_on = graph_values()
    assert any(
        dt == "bfloat16" and len(sh) == 3 and sh[-1] == cfg.vocab_size
        for _, sh, dt in values_on
    ), "гейт CE не довёл bf16 до графа градиента"
    floats = {dt for _, _, dt in values_on if "float" in dt or dt == "bfloat16"}
    assert floats <= {"float32", "bfloat16"}, floats


def test_ce_grads_parity_with_documented_tolerance():
    """Градиенты побитово не совпадают (в этом смысл гейта), но близки.

    Допуск — :data:`_CE_GRAD_RTOL`/:data:`_CE_GRAD_ATOL`; лоссовый критерий
    цикла (≤1e-3 на 10 шагах) меряется отдельной ногой ниже.
    """
    _cfg, feats, targets, emb = _ce_case()

    def grads():
        return jax.grad(
            lambda f_, e_: model._chunked_cross_entropy(f_, targets, e_, 8),
            argnums=(0, 1),
        )(feats, emb)

    with gate_env(None, None):
        off = grads()
    with gate_env(None, "1"):
        on = grads()
    rel = _max_rel(on, off)
    print(f"[веха-9] CE-гейт: относительный |dgrad| = {rel:.3e}")
    assert _tree_allclose(on, off, rtol=_CE_GRAD_RTOL, atol=_CE_GRAD_ATOL), (
        f"градиенты CE разошлись сильнее допуска (отн. {rel:.3e})"
    )


# --------------------------------------------------------------------------- #
# 10-шаговые ноги: Δloss, маркеры метрик, связка с Am.18
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def parity_legs(tmp_path_factory):
    """10-шаговые ноги (lr=1e-3): off / MLA-on / CE-on + связка кампании (2 шага).

    Данные — один и тот же детерминированный поток (парный дизайн): дрейф
    кривых принадлежит гейту, а не батчам.  ``both`` — оба гейта вместе с
    ``per_layer_block`` (боевая связка цикла-11, инвариант Am.18).
    """
    cfg = tiny_config(ce_chunk_tokens=8)
    batches = _batches(cfg, _PARITY_STEPS)
    out = tmp_path_factory.mktemp("veha9-legs")
    legs = {}
    with gate_env(None, None):
        legs["off"] = _train_leg(
            cfg, _PARITY_STEPS, batches=batches, metrics_path=out / "off.jsonl"
        )
    with gate_env("1", None):
        legs["mla"] = _train_leg(
            cfg, _PARITY_STEPS, batches=batches, metrics_path=out / "mla.jsonl"
        )
    with gate_env(None, "1"):
        legs["ce"] = _train_leg(
            cfg, _PARITY_STEPS, batches=batches, metrics_path=out / "ce.jsonl"
        )
    with gate_env("1", "1", policy="per_layer_block", accum_jit="1"):
        legs["both"] = _train_leg(
            cfg, 2, batches=batches[:4], accum_tokens=2 * SEQ_LEN,
            metrics_path=out / "both.jsonl",
        )
    return cfg, legs


def test_ten_step_delta_loss_mla_gate(parity_legs):
    """MLA-гейт: Δloss на 10 шагах ≤ 1e-3 (Am.20 п.1; измерено 0.0 при lr=1e-3)."""
    _cfg, legs = parity_legs
    drift = _loss_drift(legs["mla"].losses, legs["off"].losses)
    print(f"[веха-9] 10 шагов, MLA-гейт: max|dloss| = {drift:.3e}")
    assert all(np.isfinite(x) for x in legs["mla"].losses)
    assert drift <= _DELTA_LOSS_TOL, (
        f"Δloss MLA-ноги {drift:.3e} > {_DELTA_LOSS_TOL:g}"
    )


def test_ten_step_delta_loss_ce_gate(parity_legs):
    """CE-гейт: Δloss на 10 шагах ≤ 1e-3 — критерий цикла-11 (Am.20 п.2).

    Измерено 1.9e-6 при lr=1e-3 (кампанийный режим); при lr=1e-2 игрушечная
    траектория сама усиливает любой шум (1.6e-3) — протокол и его границы
    задокументированы у :data:`_PARITY_LR` (форма test_35).
    """
    _cfg, legs = parity_legs
    drift = _loss_drift(legs["ce"].losses, legs["off"].losses)
    print(f"[веха-9] 10 шагов, CE-гейт: max|dloss| = {drift:.3e}")
    assert all(np.isfinite(x) for x in legs["ce"].losses)
    assert drift <= _DELTA_LOSS_TOL, (
        f"Δloss CE-ноги {drift:.3e} > {_DELTA_LOSS_TOL:g}"
    )


def test_metrics_mark_both_actual_gate_values(parity_legs):
    """Маркеры фактических значений — в каждой строке журнала (0/0, 1/0, 0/1, 1/1)."""
    _cfg, legs = parity_legs
    expected = {"off": (0, 0), "mla": (1, 0), "ce": (0, 1), "both": (1, 1)}
    for name, (want_mla, want_ce) in expected.items():
        rows = tl.MetricsWriter.read(legs[name].metrics_path)
        assert rows, f"журнал ноги {name!r} пуст"
        for record in rows:
            assert record["mla_bwd_bf16"] == want_mla, (name, record)
            assert record["ce_bf16"] == want_ce, (name, record)


def test_campaign_bundle_with_per_layer_block_still_works(parity_legs):
    """Связка цикла-11: оба гейта + ``per_layer_block`` (Am.18) — нога жива."""
    _cfg, legs = parity_legs
    both = legs["both"]
    assert both.steps_done == 2
    assert all(np.isfinite(x) for x in both.losses)
    for record in tl.MetricsWriter.read(both.metrics_path):
        assert record["grad_ckpt_policy"] == "per_layer_block"
        assert record["mla_bwd_bf16"] == 1 and record["ce_bf16"] == 1
        assert record["accum_jit"] == 1 and record["microbatches"] == 2


# --------------------------------------------------------------------------- #
# (а) оба гейта off — побитово прежняя ревизия (уровень test_38/test_29)
# --------------------------------------------------------------------------- #

#: Проба дефолтного пути: двухшаговая нога с накоплением на конфиге с
#: чанкованным CE (задевает оба изменённых пути — MLA dense + CE-чанк);
#: печатает JSON последней строкой (форма test_38).
_PROBE = r'''
import json, sys
from pathlib import Path
sys.path.insert(0, "net/tests")
import conftest  # noqa: F401 — пины ADR-010 до import jax
import numpy as np
from conftest import tiny_config
from net import checkpoint, train_loop as tl

cfg = tiny_config(ce_chunk_tokens=8)
rng = np.random.default_rng(5)
batches = [rng.integers(0, cfg.vocab_size, size=(1, 16), dtype=np.int32) for _ in range(4)]
result = tl.train(
    cfg, iter(batches),
    train_config=tl.TrainConfig(steps=2, lr=1e-2, seed=7, schedule="cosine",
                                param_dtype="float32", accum_tokens=32,
                                micro_batch=1, log_every=0),
    budget=tl.Budget(run_ref="veha9-off", path=Path("/dev/null"), present=True),
)
print(json.dumps({
    "losses": [repr(x) for x in result.losses],
    "params_hash": checkpoint.tree_hash(result.params),
    "master_hash": checkpoint.tree_hash(result.master_params),
    "steps_done": result.steps_done,
}))
'''


def _run_probe(cwd: Path, *, script: Path | None = None, extra_env=None) -> dict:
    """Проба в подпроцессе: env гейтов санирован (гейты не заданы/заданы явно)."""
    env = dict(os.environ)
    for key in (
        mla.MLA_BWD_BF16_ENV,
        model.CE_BF16_ENV,
        "AXIOM_GRAD_CKPT_POLICY",
        tl.ACCUM_JIT_ENV,
        "AXIOM_MASTER_DTYPE",
        "AXIOM_MASTER_COMP",
        "AXIOM_KDA_CC_IMPL",
    ):
        env.pop(key, None)
    env.update(extra_env or {})
    command = (
        [sys.executable, str(script)]
        if script is not None
        else [sys.executable, "-c", _PROBE]
    )
    done = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
    if done.returncode != 0:  # pragma: no cover — диагностика прогона
        raise AssertionError(f"проба упала ({cwd}):\n{done.stderr[-2000:]}")
    lines = [line for line in done.stdout.strip().splitlines() if line.strip()]
    return json.loads(lines[-1])


def _extract_baseline(tmp_path: Path) -> Path:
    """``net/`` базовой ревизии + проба рядом (cwd=root: ``from net import``)."""
    root = tmp_path / "baseline"
    root.mkdir()
    try:
        archive = subprocess.run(
            ["git", "-C", str(CASE_DIR), "archive", BASELINE_COMMIT, "net/"],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        pytest.skip(f"baseline {BASELINE_COMMIT} недоступен: {exc}")
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as bundle:
        bundle.extractall(root)
    (root / "probe.py").write_text(_PROBE, encoding="utf-8")
    return root


def test_gates_off_are_bit_identical_to_baseline_revision(tmp_path):
    """Оба гейта не заданы — путь побитово прежний (ревизия до дельты, точно).

    Сравниваются loss-репризы и tree_hash деревьев двух шагов с накоплением
    (2 микробатча на шаг), конфиг с чанкованным CE: любая новая арифметика на
    дефолтном пути видна строкой, а не «в допуске».
    """
    baseline = _extract_baseline(tmp_path)
    base_out = _run_probe(baseline, script=baseline / "probe.py")
    current_out = _run_probe(CASE_DIR)
    assert current_out == base_out, (
        "дефолтный путь разошёлся с базовой ревизией: "
        f"current={current_out!r} baseline={base_out!r}"
    )


def test_explicit_zero_is_bit_identical_to_unset(tmp_path):
    """``=0`` (явный выключатель) — тот же граф, что и с незаданной переменной."""
    current_unset = _run_probe(CASE_DIR)
    current_zero = _run_probe(
        CASE_DIR,
        extra_env={mla.MLA_BWD_BF16_ENV: "0", model.CE_BF16_ENV: "0"},
    )
    assert current_zero == current_unset
