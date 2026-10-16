"""Веха-12 (ADR-054 Am.25, цикл-15 кампании MFU-55): flash-backward MLA — jax-уровень.

Пол B=4 (Am.20, HLO-разбор): fp32-скоры (B,12,8192,8192) по 12 ГиБ живут в
backward десятками вхождений; bf16-хранение (Am.20) срезало dtype, не пол
(plain-B=4 OOM 78,95 ГиБ).  Гейт ``AXIOM_MLA_FLASH_BWD=1`` подменяет backward
dense-внимания MLA на тайловое переисчисление (custom_vjp по границе
``(q,k,v,mask) -> o``): forward — плотные числа дословно, для backward
остаются ``q``, ``k``, ``v``, маска, выход ``o`` и статистики строк
(rowmax/rowsum), а QK^T переисчитывается во внешнем цикле по KV-тайлам и
внутреннем по Q-тайлам; полная (B,H,T,T) не существует ни в одном промежутке.
Дефолт ``0`` — побитово прежний граф; forward не тронут (flash-fwd
``AXIOM_MLA_DENSE_FLASH`` — отдельная ветка, новый гейт туда не читается).

Что пиннуется здесь (CPU, без стенда GB10):

* контракт гейта: имя ``AXIOM_MLA_FLASH_BWD``, дефолт 0, пустая = 0,
  регистр/пробелы, неизвестное — ошибка на чтении и на входе ``train`` до
  params/компиляции/батчей (форма ``bwd_bf16_mode`` Am.20);
* (а) гейт off (не задан и явный ``=0``) — побитово прежняя ревизия
  (``BASELINE_COMMIT``, форма test_39): двухшаговая нога ``train`` с
  накоплением и чанкованным CE — лоссы и хеши деревьев совпадают; fp32- и
  bf16-ветки рабочей копии параметров;
* (б) паритет loss/grads против dense-ссылки на малом конфиге: forward
  побитово тот же, градиенты в допуске ≤1e-3 (измерено ~2e-7 отн. в
  fp32-режиме; фактическое число печатается); прямая сверка градиентов
  тайлового оператора с autodiff плотного softmax-блока — тот же допуск;
  границы тайловой нарезки (tile не делит T) — точно те же числа forward и
  допуск градиентов (форма секции ниже);
* (в) пин «живых скоров в backward ≤ 1 тайла»: в jaxpr backward нет ни одной
  float-переменной формы (B,H,T,T) — и на pullback графа слоя, и на pullback
  реального графа модели (``grad_ckpt_policy="none"``; под remat-forward
  пересчитывается forward, там полные скоры — это forward, не backward: пин
  см. в докстринге теста); все скор-формы ≤ (B,H,tile,tile); тайловые формы
  присутствуют (пин не вакуумный); у гейта off те же пробы видят полные скоры
  (зубы); тот же пин на StableHLO-тексте пониженного pullback (реальный граф
  после jaxpr);
* (г) nan-контроль 10 шагов на CPU-миниконфиге (прецедент 4b/4c — дефекты
  проявлялись на 4+ шагах): Δloss off/on ≤1e-3 (измерено при lr=1e-3),
  все числа конечны; связка кампании (flash-bwd × bf16-хранение ×
  per_layer_block × accum-jit) — 10 шагов, все лоссы конечны;
* совместимость: маркер ``mla_flash_bwd`` в каждой строке журнала (0/1,
  фактические значения гейтов), связка с ``AXIOM_MLA_BWD_BF16`` (Am.20),
  ``per_layer_block`` (Am.18), ``AXIOM_COMPUTE_DTYPE=bf16`` (bf16-допуск
  градиентов документирован отдельно — измерено ~3e-3) и нетронутость
  flash-fwd пути (побитово), котangенты несут dtype прималов (форма
  фикс-раунда вехи-9);
* ширина тайла — конфиг ``ModelConfig.mla_flash_bwd_tile`` (дефолт 1024;
  ``validate_config`` отвергает неположительное).

Не пинуется здесь: стендовая нога B=4/B=8 (номер приёмки кампании — отдельный
прогон на GB10) и NVTX-зонд фаз; HLO-пол проверяется пинном jaxpr/StableHLO
(«XLA материализует полные скоры вопреки custom_vjp» — вопрос стенда; на
jax-уровне скор-тензоров полной формы в backward нет структурно).
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
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

#: Ревизия до дельты: doc-коммит Am.25 ``net/`` не трогал, поэтому дефолтный
#: путь обязан воспроизводить её побитово.  ``git archive`` + skip, если
#: ревизия недоступна (форма test_38/test_39).
BASELINE_COMMIT = "97c6eb8"
CASE_DIR = Path(__file__).resolve().parents[2]

#: 10-шаговый паритет (форма test_35/test_39): lr кампанийного режима — малое
#: обновление за шаг, чтобы кривые сравнивали гейт, а не чувствительность
#: траектории.
_PARITY_STEPS = 10
_PARITY_LR = 1e-3

#: Критерий вехи: Δloss на 10 шагах ≤ 1e-3 (Am.25, приёмка «паритет ≤1e-3»).
_DELTA_LOSS_TOL = 1e-3

#: Допуск паритета градиентов (Am.25: «ожидание ≤1e-3»); фактическое число
#: печатается каждой ногой (fp32-режим измерен ~2e-7 отн.).
_PARITY_TOL = 1e-3

#: Градиентный допуск связки ``AXIOM_COMPUTE_DTYPE=bf16``: арифметика bf16-
#: границы в backward тайлов несёт округление (2^-8 на операнде), и сверка идёт
#: против autodiff того же bf16-пути — измерено ~3e-3 отн.; сам лосс побитово.
_COMPUTE_BF16_GRAD_RTOL = 1e-2

#: Ширина тайла пина: меньше T — иначе тайловая граница совпала бы с полной
#: матрицей и пин «≤1 тайла» стал бы вакуумным.
_PIN_TILE = 4


# --------------------------------------------------------------------------- #
# Общие помощники
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _cpu_kda_body(monkeypatch):
    """Явное committed KDA-тело для CPU (кампанийный дефолт ``persistent``)."""
    monkeypatch.setenv("AXIOM_KDA_CC_IMPL", "jax")


@contextlib.contextmanager
def gate_env(
    flash_bwd: str | None = None,
    mla_bf16: str | None = None,
    ce_bf16: str | None = None,
    policy: str | None = None,
    accum_jit: str | None = None,
    compute: str | None = None,
    flash_fwd: str | None = None,
):
    """Временные значения env-гейтов вехи-12 + соседних условий прогона.

    ``None`` — снять переменную (дефолт); явное ``"0"`` — выключатель.
    """
    keys = (
        mla.MLA_FLASH_BWD_ENV,
        mla.MLA_BWD_BF16_ENV,
        "AXIOM_CE_BF16",
        "AXIOM_GRAD_CKPT_POLICY",
        tl.ACCUM_JIT_ENV,
        "AXIOM_COMPUTE_DTYPE",
        "AXIOM_MLA_DENSE_FLASH",
    )
    values = (flash_bwd, mla_bf16, ce_bf16, policy, accum_jit, compute, flash_fwd)
    saved = {key: os.environ.get(key) for key in keys}
    try:
        for key, value in zip(keys, values):
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
    return tl.Budget(run_ref="veha12", path=Path("/dev/null"), present=True)


def _train_leg(
    cfg,
    steps: int,
    *,
    lr: float = _PARITY_LR,
    seed: int = 7,
    batches=None,
    metrics_path=None,
    accum_tokens: int = 0,
    param_dtype: str = "float32",
):
    """Нога ``train``; env гейтов выставляют тесты явно через :func:`gate_env`."""
    return tl.train(
        cfg,
        iter(batches if batches is not None else _batches(cfg, steps)),
        train_config=tl.TrainConfig(
            steps=steps, lr=lr, seed=seed, schedule="cosine",
            param_dtype=param_dtype, micro_batch=1, accum_tokens=accum_tokens,
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


def _walk_values(jaxpr, out=None, depth=0):
    """(primitive, shape, dtype) каждого результата в джахпре, включая вложенные.

    Спуск в ``jaxpr``-параметры уравнений (remat/checkpoint/custom-вызовы) и в
    контейнеры — пин смотрит весь граф, а не только верхний уровень.
    """
    if out is None:
        out = []
    if depth > 40 or jaxpr is None:
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


def _full_score_shapes(values, B, H, T):
    """Float-переменные, чья форма — ровно полная скор-матрица (B,H,T,T).

    Точная форма (а не «лейаут»): в backward реального графа рядом живут
    KDA-матрицы (B,H,dk,dv) и транзиенты активаций, поэтому проверка формы —
    единственная устойчивая; пик «любой лейаут полной матрицы» ловится
    numel-связкой в пробе самого оператора (там граф замкнут).
    """
    return [
        (p, s, d) for p, s, d in values
        if len(s) == 4 and s == (B, H, T, T) and "float" in d
    ]


def _hlo_f32_shapes(text: str) -> list[str]:
    return sorted(set(re.findall(r"tensor<([0-9x]+)xf32>", text)))


# --------------------------------------------------------------------------- #
# Контракт значений: дефолт 0, нормализация, fail-closed
# --------------------------------------------------------------------------- #


def test_env_name_is_the_contract():
    """Раннер кампании выставляет гейт по имени — имя пиннуется строкой."""
    assert mla.MLA_FLASH_BWD_ENV == "AXIOM_MLA_FLASH_BWD"
    assert mla.MLA_FLASH_BWD_VALUES == ("0", "1")


def test_reader_defaults_to_off():
    with gate_env(None):
        assert mla.flash_bwd_mode() is False
    for blank in ("", "   ", "\t\n"):
        with gate_env(blank):
            assert mla.flash_bwd_mode() is False, "пустая = дефолт"


def test_reader_normalizes_and_explicit_argument_wins():
    for on in ("1", " 1 ", "1\t"):
        with gate_env(on):
            assert mla.flash_bwd_mode() is True
    for off in ("0", " 0 "):
        with gate_env(off):
            assert mla.flash_bwd_mode() is False
    # Явный аргумент сильнее окружения (форма bwd_bf16_mode/accum_jit_mode).
    with gate_env("0"):
        assert mla.flash_bwd_mode("1") is True
    with gate_env("1"):
        assert mla.flash_bwd_mode("0") is False


def test_reader_unknown_value_fails_closed():
    """Опечатка — ошибка, а не тихий откат: замер ноги не станет замером dense."""
    for bad in ("true", "yes", "on", "flash", "2", "0 1"):
        with gate_env(bad):
            with pytest.raises(mla.MlaFlashBwdError, match=mla.MLA_FLASH_BWD_ENV):
                mla.flash_bwd_mode()
    with pytest.raises(mla.MlaFlashBwdError):
        mla.flash_bwd_mode("flash")


class _NeverConsumed:
    """Поток батчей, который обязан остаться нетронутым при отказе сборки."""

    def __iter__(self):
        return self

    def __next__(self):  # pragma: no cover — исполнение = провал теста
        raise AssertionError("батчи не должны читаться: отказ на сборке")


def test_train_refuses_unknown_gate_before_any_run(monkeypatch, tmp_path):
    """Мусор в гейте: ошибка до params/компиляции/данных, журнал не создаётся."""
    cfg = tiny_config()
    metrics = tmp_path / "must-not-exist.jsonl"
    monkeypatch.setenv(mla.MLA_FLASH_BWD_ENV, "on")
    with pytest.raises(mla.MlaFlashBwdError, match=mla.MLA_FLASH_BWD_ENV):
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


# --------------------------------------------------------------------------- #
# Ширина тайла — константа конфига
# --------------------------------------------------------------------------- #


def test_tile_is_a_config_constant_with_positive_validation():
    from net import config as config_mod

    assert config_mod.ModelConfig().mla_flash_bwd_tile == 1024, (
        "дефолт тайла — 1024 (Am.25); смена дефолта — решение, не мелочь"
    )
    config_mod.validate_config(small_config())  # дефолт валиден
    for bad in (0, -1, 1.5, True):
        with pytest.raises(AssertionError, match="mla_flash_bwd_tile"):
            config_mod.validate_config(small_config(mla_flash_bwd_tile=bad))


# --------------------------------------------------------------------------- #
# (б) паритет: forward — бит-в-бит, градиенты — в допуске
# --------------------------------------------------------------------------- #


def _mla_case():
    cfg = small_config(mla_flash_bwd_tile=_PIN_TILE)
    params = mla.init_mla(jr.PRNGKey(0), cfg)
    x = jr.normal(jr.PRNGKey(1), (2, SEQ_LEN, cfg.hidden))
    return cfg, params, x


def test_flash_bwd_forward_is_bitwise_the_same_with_the_gate_on():
    """Числа гейт не меняет: forward — дословно плотный путь (те же операции)."""
    cfg, params, x = _mla_case()
    with gate_env(None):
        off = mla._dense_apply(params, cfg, x)
    with gate_env("1"):
        on = mla._dense_apply(params, cfg, x)
    assert jnp.array_equal(off, on), "forward поменялся — гейт обязан трогать лишь backward"


def test_flash_bwd_loss_and_grads_parity_with_documented_tolerance():
    """Паритет loss/grads малого конфига; допуск ≤1e-3 (Am.25).

    Модельная нога (dense-оракул дефолтом, KDA не задета гейтом): лосс —
    побитово (forward тот же), градиенты — в допуске; фактическое число
    печатается (fp32-режим измерен ~2e-7 отн.).
    """
    cfg = small_config(mla_flash_bwd_tile=_PIN_TILE)
    params = model.init_params(jr.PRNGKey(0), cfg)
    ids = jr.randint(jr.PRNGKey(2), (2, SEQ_LEN), 1, cfg.vocab_size)

    def loss(p):
        return model.compute_loss(p, cfg, ids, SEQ_LEN, grad_ckpt_policy="none")

    with gate_env(None):
        v_off, g_off = jax.value_and_grad(loss)(params)
    with gate_env("1"):
        v_on, g_on = jax.value_and_grad(loss)(params)

    assert jnp.array_equal(v_off, v_on), "forward обязан быть побитово тем же"
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree_util.tree_leaves(g_on)), (
        "в градиентах гейта появились нефинитные значения"
    )
    rel = _max_rel(g_on, g_off)
    print(f"[веха-12] flash-bwd: относительный |dgrad| = {rel:.3e}")
    assert rel <= _PARITY_TOL, f"паритет градиентов {rel:.3e} > {_PARITY_TOL:g}"


def test_flash_bwd_grads_match_autodiff_of_the_dense_softmax():
    """Прямая сверка: градиенты тайлового оператора = autodiff плотного блока.

    Плотная ссылка — дословная формула прежнего пути (QK^T → где-маска →
    softmax → контракция) с autodiff; тайловый оператор — с несколькими
    тайлами (tile=4 < T).  Допуск вехи, фактическое число печатается.
    """
    cfg, params, x = _mla_case()
    B, T, H, dq = 2, SEQ_LEN, cfg.num_heads, cfg.mla_head_dim
    q = jr.normal(jr.PRNGKey(2), (B, T, H, dq))
    k = jr.normal(jr.PRNGKey(3), (B, T, H, dq))
    v = jr.normal(jr.PRNGKey(4), (B, T, H, dq))
    mask = mla._causal_mask(T)

    def loss_dense(q_, k_, v_):
        scale = 1.0 / jnp.sqrt(dq)
        s = jnp.einsum("bthd,bshd->bhts", q_, k_) * scale
        a = jax.nn.softmax(jnp.where(mask, s, jnp.finfo(jnp.float32).min), axis=-1)
        return jnp.sum(jnp.einsum("bhts,bshd->bthd", a, v_) ** 2)

    def loss_tiled(q_, k_, v_):
        op = mla._attn_out_flashbwd_op(int(cfg.mla_flash_bwd_tile))
        return jnp.sum(op(q_, k_, v_, mask) ** 2)

    g_ref = jax.grad(loss_dense, argnums=(0, 1, 2))(q, k, v)
    g_tiled = jax.grad(loss_tiled, argnums=(0, 1, 2))(q, k, v)
    rel = _max_rel(g_tiled, g_ref)
    print(f"[веха-12] оператор vs autodiff: относительный |dgrad| = {rel:.3e}")
    assert rel <= _PARITY_TOL, f"градиенты оператора разошлись: {rel:.3e}"


def test_tile_width_boundaries_are_exact():
    """T не делится тайлом (5×3+1), делится (16), тайл больше T (32): числа те же.

    Границы нарезки обрабатываются статически (остаток — отдельная итерация),
    forward обязан совпасть побитово с гейтом off (ширина тайла вперёд не
    входит), градиенты — в допуске.
    """
    cfg0 = small_config()
    params = mla.init_mla(jr.PRNGKey(0), cfg0)
    x = jr.normal(jr.PRNGKey(1), (2, SEQ_LEN, cfg0.hidden))

    def loss_for(cfg):
        def loss(p):
            return jnp.sum(mla._dense_apply(p, cfg, x) ** 2)
        return loss

    with gate_env(None):
        v_off, g_off = jax.value_and_grad(loss_for(cfg0))(params)
    for tile in (5, 16, 32):
        cfg = small_config(mla_flash_bwd_tile=tile)
        with gate_env("1"):
            v_on, g_on = jax.value_and_grad(loss_for(cfg))(params)
        assert jnp.array_equal(v_off, v_on), f"forward tile={tile} разошёлся"
        rel = _max_rel(g_on, g_off)
        assert rel <= _PARITY_TOL, f"tile={tile}: градиенты {rel:.3e}"


# --------------------------------------------------------------------------- #
# (в) пин: живые скоры backward ≤ 1 тайла
# --------------------------------------------------------------------------- #


def test_backward_jaxpr_holds_at_most_one_tile_of_scores():
    """Pullback оператора: полной скор-матрицы нет ни в одном лейауте.

    Проба замкнута на графе самого оператора (B=2, T=8, H=3, dq=5, tile=2 —
    все размерности различны, чтобы формы не путались), поэтому кроме точной
    (B,H,T,T) пиннуется и numel-связка: ни одна float-переменная не крупнее
    max(B*T*H*dq, B*H*tile²) — полная матрица в ЛЮБОЙ перестановке осей
    превышает связку (T > dq в этой пробе), а транзиенты тайлов и
    активаций — нет.  Тайловые скоры (B,H,tile,tile) присутствуют — пин не
    вакуумный; у плотной ссылки полные скоры есть — проба различает пути.
    """
    B, T, H, dq, tile = 2, 8, 3, 5, 2
    q = jr.normal(jr.PRNGKey(2), (B, T, H, dq))
    k = jr.normal(jr.PRNGKey(3), (B, T, H, dq))
    v = jr.normal(jr.PRNGKey(4), (B, T, H, dq))
    mask = mla._causal_mask(T)

    def tiled_values():
        op = mla._attn_out_flashbwd_op(tile)
        _, pb = jax.vjp(lambda q_, k_, v_: jnp.sum(op(q_, k_, v_, mask)), q, k, v)
        return _walk_values(jax.make_jaxpr(lambda c: pb(c))(jnp.ones(())).jaxpr)

    def dense_values():
        def dense(q_, k_, v_):
            s = jnp.einsum("bthd,bshd->bhts", q_, k_)
            a = jax.nn.softmax(jnp.where(mask, s, jnp.finfo(jnp.float32).min), axis=-1)
            return jnp.sum(jnp.einsum("bhts,bshd->bthd", a, v_))

        _, pb = jax.vjp(dense, q, k, v)
        return _walk_values(jax.make_jaxpr(lambda c: pb(c))(jnp.ones(())).jaxpr)

    tiled = tiled_values()
    assert tiled, "backward пуст — пин не тот граф"
    full = _full_score_shapes(tiled, B, H, T)
    assert not full, f"полная (B,H,T,T) в backward: {full[:2]}"
    bound = max(B * T * H * dq, B * H * tile * tile)
    over = [
        (p, s, d) for p, s, d in tiled
        if len(s) == 4 and "float" in d and int(np.prod(s)) > bound
    ]
    assert not over, f"переменная крупнее связки {bound}: {over[:2]}"
    assert any(
        s == (B, H, tile, tile) and "float" in d for _, s, d in tiled
    ), "нет тайловых (B,H,tile,tile) — тайловый backward не исполнялся"

    dense = dense_values()
    assert _full_score_shapes(dense, B, H, T), (
        "у dense-ссылки нет полных скоров — проба не различает пути"
    )


def test_backward_jaxpr_of_the_whole_model_graph_holds_one_tile():
    """Реальный граф модели (policy none): pullback без полной скор-матрицы.

    T=12 при kda_dk=16 — чтобы (B,H,T,T) attention не путалась с матрицами
    чанков KDA (у них свои формы); tile=3 (T делится, H=4 != tile — форма
    dk/dv-аккумуляций с тайловым индексом на месте H не совпадает со
    скор-лейаутом); numel-связка здесь неприменима (в графе живут
    KDA-состояния (B,H,dk,dv)) — пин точной формы плюс наличие тайлов.
    Под remat-forward (``per_layer_block``) backward пересчитывает forward
    слоя — там полные скоры живут как forward-транзиент, это не хранение
    backward'а: пин — на графе без remat-пересчёта.
    """
    cfg = small_config(mla_flash_bwd_tile=3)
    params = model.init_params(jr.PRNGKey(0), cfg)
    B, T, H = 2, 12, cfg.num_heads
    ids = jr.randint(jr.PRNGKey(2), (B, T), 1, cfg.vocab_size)

    def loss(p):
        return model.compute_loss(p, cfg, ids, 5, grad_ckpt_policy="none")

    def values():
        _, pb = jax.vjp(loss, params)
        return _walk_values(jax.make_jaxpr(lambda c: pb(c))(jnp.ones(())).jaxpr)

    with gate_env("1"):
        on = values()
    assert on, "pullback модели пуст — пин не тот граф"
    full = _full_score_shapes(on, B, H, T)
    assert not full, f"полная (B,H,T,T) в pullback модели: {full[:2]}"
    assert any(
        s == (B, H, 3, 3) and "float" in d for _, s, d in on
    ), "нет тайловых скор-форм — тайловый backward не исполнялся"

    with gate_env(None):
        off = values()
    assert _full_score_shapes(off, B, H, T), (
        "у гейта off pullback модели обязан держать полные скоры (зубы пина)"
    )


def test_backward_hlo_of_the_real_graph_has_no_full_scores():
    """StableHLO пониженного pullback слоя: ни одного тензора (B,H,T,T).

    jaxpr-пин проверяет граф до выбора бэкендом раскладок; этот — текст модуля
    после понижения (реальный граф исполнения, форма «jaxpr/HLO» из Am.25).
    """
    cfg, params, x = _mla_case()
    B, T, H, dq = 2, SEQ_LEN, cfg.num_heads, cfg.mla_head_dim
    tile = int(cfg.mla_flash_bwd_tile)

    def loss(p):
        return jnp.sum(mla._dense_apply(p, cfg, x) ** 2)

    with gate_env("1"):
        _, pb = jax.vjp(loss, params)
        low = jax.jit(lambda c: pb(c)).lower(jnp.ones(()))
        text = low.as_text()
    shapes = _hlo_f32_shapes(text)
    assert f"{B}x{H}x{T}x{T}" not in shapes, f"полный скор-буфер в HLO: {shapes}"
    assert f"{B}x{H}x{tile}x{tile}" in shapes, (
        f"тайлового буфера нет в HLO — тайловый backward не доведён до модуля: {shapes}"
    )


# --------------------------------------------------------------------------- #
# (г) 10-шаговые ноги: Δloss, nan-контроль, маркеры, связки
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def parity_legs(tmp_path_factory):
    """10-шаговые ноги (lr=1e-3): off / on / связка кампании.

    Данные — один детерминированный поток (парный дизайн).  ``combo`` —
    боевая связка вехи: flash-bwd × bf16-хранение (Am.20) × per_layer_block
    (Am.18) × accum-jit — nan-контроль прецедента 4b/4c (дефекты проявлялись
    на 4+ шагах) и совместимость условий прогона.
    """
    cfg = tiny_config(ce_chunk_tokens=8)
    batches = _batches(cfg, _PARITY_STEPS)
    out = tmp_path_factory.mktemp("veha12-legs")
    legs = {}
    with gate_env(None):
        legs["off"] = _train_leg(
            cfg, _PARITY_STEPS, batches=batches, metrics_path=out / "off.jsonl"
        )
    with gate_env("1"):
        legs["on"] = _train_leg(
            cfg, _PARITY_STEPS, batches=batches, metrics_path=out / "on.jsonl"
        )
    with gate_env("1", mla_bf16="1", ce_bf16="1", policy="per_layer_block", accum_jit="1"):
        legs["combo"] = _train_leg(
            cfg, _PARITY_STEPS,
            batches=_batches(cfg, 2 * _PARITY_STEPS),  # 2 микробатча на шаг
            accum_tokens=2 * SEQ_LEN,
            metrics_path=out / "combo.jsonl",
        )
    return cfg, legs


def test_ten_step_delta_loss_flash_bwd(parity_legs):
    """Δloss на 10 шагах ≤ 1e-3 (Am.25); фактическое число печатается."""
    _cfg, legs = parity_legs
    drift = _loss_drift(legs["on"].losses, legs["off"].losses)
    print(f"[веха-12] 10 шагов, flash-bwd: max|dloss| = {drift:.3e}")
    assert all(np.isfinite(x) for x in legs["on"].losses)
    assert drift <= _DELTA_LOSS_TOL, f"Δloss ноги {drift:.3e} > {_DELTA_LOSS_TOL:g}"


def test_ten_step_nan_control_campaign_bundle(parity_legs):
    """Связка кампании, 10 шагов: ни одного нефинитного числа (прецедент 4b/4c)."""
    _cfg, legs = parity_legs
    combo = legs["combo"]
    assert combo.steps_done == _PARITY_STEPS
    assert all(np.isfinite(x) for x in combo.losses), combo.losses
    leaves = jax.tree_util.tree_leaves(combo.params)
    assert all(bool(jnp.all(jnp.isfinite(l))) for l in leaves), (
        "в параметрах связки появился nan/inf"
    )


def test_metrics_mark_the_flash_bwd_gate(parity_legs):
    """Маркер ``mla_flash_bwd`` — фактические значения в каждой строке журнала."""
    _cfg, legs = parity_legs
    expected = {"off": 0, "on": 1, "combo": 1}
    for name, want in expected.items():
        rows = tl.MetricsWriter.read(legs[name].metrics_path)
        assert rows, f"журнал ноги {name!r} пуст"
        for record in rows:
            assert record["mla_flash_bwd"] == want, (name, record)
    # Связка: тайловый путь исполняется при bf16-гейте — оба маркера несут
    # фактические значения (Am.20), политика и аккумуляция видны рядом.
    for record in tl.MetricsWriter.read(legs["combo"].metrics_path):
        assert record["mla_flash_bwd"] == 1 and record["mla_bwd_bf16"] == 1
        assert record["ce_bf16"] == 1
        assert record["grad_ckpt_policy"] == "per_layer_block"
        assert record["accum_jit"] == 1 and record["microbatches"] == 2


# --------------------------------------------------------------------------- #
# Совместимость: flash-fwd, bf16-compute, dtype-контракт котангентов
# --------------------------------------------------------------------------- #


def test_flash_fwd_path_is_not_touched_by_the_gate(monkeypatch):
    """``AXIOM_MLA_DENSE_FLASH=1``: гейт веткой не читается — побитово оба плеча."""
    cfg, params, x = _mla_case()

    def loss(p):
        return jnp.sum(mla._dense_apply(p, cfg, x) ** 2)

    with gate_env(None, flash_fwd="1"):
        v_off, g_off = jax.value_and_grad(loss)(params)
    with gate_env("1", flash_fwd="1"):
        v_on, g_on = jax.value_and_grad(loss)(params)
    assert jnp.array_equal(v_off, v_on)
    assert all(
        bool(jnp.array_equal(a, b))
        for a, b in zip(
            jax.tree_util.tree_leaves(g_on), jax.tree_util.tree_leaves(g_off)
        )
    ), "flash-fwd путь обязан остаться гейтом не тронут (побитово)"


def test_compute_dtype_bf16_combo_is_documented():
    """``AXIOM_COMPUTE_DTYPE=bf16`` (режим кампании) × гейт: лосс бит-в-бит.

    Градиенты: тайловый backward идёт через те же bf16-границы, что и
    контракции forward (``compute_dtype.gemm_einsum``); сверка — против
    autodiff того же bf16-пути, расхождение несёт округление границы и
    документируется отдельным допуском (измерено ~3e-3, fp32-режим ~2e-7).
    """
    cfg, params, x = _mla_case()

    def loss(p):
        return jnp.sum(mla._dense_apply(p, cfg, x) ** 2)

    with gate_env(None, compute="bf16"):
        v_off, g_off = jax.value_and_grad(loss)(params)
    with gate_env("1", compute="bf16"):
        v_on, g_on = jax.value_and_grad(loss)(params)
    assert jnp.array_equal(v_off, v_on), "forward bf16-режима обязан быть бит-в-бит"
    assert all(bool(jnp.all(jnp.isfinite(l))) for l in jax.tree_util.tree_leaves(g_on))
    rel = _max_rel(g_on, g_off)
    print(f"[веха-12] bf16-compute связка: относительный |dgrad| = {rel:.3e}")
    assert rel <= _COMPUTE_BF16_GRAD_RTOL, f"bf16-связка разошлась: {rel:.3e}"


def test_flash_bwd_with_bf16_params_grads_carry_primal_dtypes():
    """Гейт × bf16-параметры: градиенты несут dtype прималов (форма вехи-9)."""
    cfg = small_config(mla_flash_bwd_tile=_PIN_TILE)
    params = jax.tree_util.tree_map(
        lambda leaf: leaf.astype(jnp.bfloat16),
        model.init_params(jr.PRNGKey(0), cfg),
    )
    mla_index = next(
        i for i, kind in enumerate(model.layer_kinds(cfg)) if kind == "mla"
    )
    x = jr.normal(jr.PRNGKey(1), (2, 12, cfg.hidden))

    def loss(p_):
        return jnp.sum(mla._dense_apply(p_.layers[mla_index].attn, cfg, x) ** 2)

    with gate_env("1"):
        grads = jax.grad(loss)(params)
    leaves = jax.tree_util.tree_leaves(grads)
    assert leaves and all(leaf.dtype == jnp.bfloat16 for leaf in leaves), (
        f"градиенты MLA обязаны нести dtype прималов (bf16): "
        f"{sorted({str(leaf.dtype) for leaf in leaves})}"
    )


# --------------------------------------------------------------------------- #
# (а) гейт off — побитово прежняя ревизия (уровень test_38/test_39)
# --------------------------------------------------------------------------- #

#: Проба дефолтного пути: двухшаговая нога с накоплением на конфиге с
#: чанкованным CE (задевает CE-чанк; MLA-путь — дефолтный dense); печатает
#: JSON последней строкой (форма test_39).  ``PROBE_PARAM_DTYPE`` выбирает
#: точность рабочей копии параметров.
_PROBE = r'''
import json, os, sys
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
                                param_dtype=os.environ.get("PROBE_PARAM_DTYPE", "float32"),
                                accum_tokens=32,
                                micro_batch=1, log_every=0),
    budget=tl.Budget(run_ref="veha12-off", path=Path("/dev/null"), present=True),
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
        mla.MLA_FLASH_BWD_ENV,
        mla.MLA_BWD_BF16_ENV,
        "AXIOM_CE_BF16",
        "AXIOM_GRAD_CKPT_POLICY",
        tl.ACCUM_JIT_ENV,
        "AXIOM_MASTER_DTYPE",
        "AXIOM_MASTER_COMP",
        "AXIOM_KDA_CC_IMPL",
        "AXIOM_COMPUTE_DTYPE",
        "AXIOM_MLA_DENSE_FLASH",
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


def test_gate_off_is_bit_identical_to_baseline_revision(tmp_path):
    """Гейт не задан — путь побитово прежний (ревизия до дельты, точно).

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


def test_gate_off_bf16_params_is_bit_identical_to_baseline_revision(tmp_path):
    """То же с ``param_dtype=bfloat16`` — режим кампании рабочей копии."""
    env = {"PROBE_PARAM_DTYPE": "bfloat16"}
    baseline = _extract_baseline(tmp_path)
    base_out = _run_probe(baseline, script=baseline / "probe.py", extra_env=env)
    current_out = _run_probe(CASE_DIR, extra_env=env)
    assert current_out == base_out, (
        "bf16-дефолтный путь разошёлся с базовой ревизией: "
        f"current={current_out!r} baseline={base_out!r}"
    )


def test_explicit_zero_is_bit_identical_to_unset(tmp_path):
    """``=0`` (явный выключатель) — тот же граф, что и с незаданной переменной."""
    current_unset = _run_probe(CASE_DIR)
    current_zero = _run_probe(
        CASE_DIR, extra_env={mla.MLA_FLASH_BWD_ENV: "0"}
    )
    assert current_zero == current_unset
