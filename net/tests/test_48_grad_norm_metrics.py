"""Диагностика нормы градиента в per-step метриках претрейна (CPU).

Схема ``pretrain-metrics/v1`` расширяется **аддитивно** тремя полями:
``grad_norm_preclip`` / ``grad_norm_postclip`` / ``grad_clip_coeff``.  Что
пинуется здесь:

* **(а) норма pytree глобальная** — ``sqrt(Σ x²)`` по всем листьям и всем
  элементам, а не среднее по листьям/устройствам; нулевой градиент даёт ноль,
  нечисловая норма остаётся нечисловой (это диагностический сигнал);
* **(б) клиппинга градиента в цикле нет** (проверено по коду: ``weight_clip`` в
  ``net/optimizer.py`` клиппит ВЕСА после шага, Muon нормирует матрицу внутри
  NS-итераций; глобального clip-by-norm градиента нет) → ``postclip == preclip``,
  ``coeff == 1.0``;
* **(в) строка метрик** несёт три поля рядом с ``step``/``loss``/
  ``step_seconds`` — и в синхронном пути, и через окно async-метрик; значения
  совпадают (те же градиенты — та же норма);
* **(г) нечисловая норма** пишется в журнал как есть (``nan``), а не
  подменяется нулём;
* **(д) старый парсер** метрик не падает на новых ключах (см. также
  ``tools/tests/test_sensors_wrap_pretrain_metrics.py``).

Тесты — CPU/моки: настоящие шаги гоняются на ``tiny_config`` (как в
``test_40_async_metrics.py``), длинных GPU-прогонов нет.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import tiny_config

from net import train_loop as tl

#: Геометрия смоука: одна последовательность, короткий контекст (CPU-время).
SEQ_LEN = 16
#: Цель накопления — два микробатча на шаг (tokens_per_micro = 1 * SEQ_LEN).
ACCUM2 = 2 * SEQ_LEN

#: Поля диагностики градиента (аддитивные ключи ``pretrain-metrics/v1``).
GRAD_FIELDS = ("grad_norm_preclip", "grad_norm_postclip", "grad_clip_coeff")


def _as_float(value) -> float:
    """Скаляр с устройства в Python-float (для unit-проверок нормы)."""
    return float(np.asarray(jax.device_get(value)))


# ---------------------------------------------------------------------------
# (а) Unit: глобальная L2-норма pytree
# ---------------------------------------------------------------------------


def test_tree_l2_norm_single_leaf():
    """Один лист: sqrt(3² + 4²) = 5."""
    tree = {"w": jnp.asarray([3.0, 4.0])}
    assert _as_float(tl.tree_l2_norm(tree)) == pytest.approx(5.0)


def test_tree_l2_norm_multiple_leaves_is_global_sum_not_mean():
    """Несколько листьев: норма — длина общего вектора, не среднее норм.

    Нормы листьев 3 и 4: глобальная норма 5, среднее было бы 3.5.  Тест ловит
    подмену «сумма квадратов» на «среднее по листьям».
    """
    tree = {
        "a": [jnp.asarray([3.0]), jnp.asarray([0.0])],
        "b": jnp.asarray([[4.0]]),
    }
    assert _as_float(tl.tree_l2_norm(tree)) == pytest.approx(5.0)


def test_tree_l2_norm_zero_gradient_is_zero():
    tree = {"w": jnp.zeros((2, 3)), "b": jnp.zeros((4,))}
    assert _as_float(tl.tree_l2_norm(tree)) == 0.0


def test_tree_l2_norm_keeps_non_finite_diagnostic():
    """nan/inf не подменяются нулём: это и есть диагностический сигнал."""
    assert math.isnan(_as_float(tl.tree_l2_norm({"w": jnp.asarray([jnp.nan])})))
    assert math.isinf(_as_float(tl.tree_l2_norm({"w": jnp.asarray([jnp.inf])})))


# ---------------------------------------------------------------------------
# (б) Unit: поля нормы без клиппинга
# ---------------------------------------------------------------------------


def test_grad_norm_fields_without_clipping():
    """Клиппинга в цикле нет → postclip == preclip, coeff == 1.0."""
    fields = tl.grad_norm_fields(1.25)
    assert set(fields) == set(GRAD_FIELDS)
    assert fields["grad_norm_preclip"] == 1.25
    assert fields["grad_norm_postclip"] == fields["grad_norm_preclip"]
    assert fields["grad_clip_coeff"] == 1.0


def test_grad_norm_fields_preserve_non_finite():
    """Нечисловая норма остаётся нечисловой и в полях строки метрик."""
    nan_fields = tl.grad_norm_fields(float("nan"))
    assert math.isnan(nan_fields["grad_norm_preclip"])
    assert math.isnan(nan_fields["grad_norm_postclip"])
    assert nan_fields["grad_clip_coeff"] == 1.0

    inf_fields = tl.grad_norm_fields(float("inf"))
    assert math.isinf(inf_fields["grad_norm_preclip"])
    assert math.isinf(inf_fields["grad_norm_postclip"])
    assert inf_fields["grad_clip_coeff"] == 1.0


# ---------------------------------------------------------------------------
# Общие помощники интеграционных проб
# ---------------------------------------------------------------------------


def _batches(cfg, count: int, *, seed: int = 5):
    """Детерминированный поток ``(1, T) int32`` — одинаковые данные для ног."""
    rng = np.random.default_rng(seed)
    return [
        rng.integers(0, cfg.vocab_size, size=(1, SEQ_LEN), dtype=np.int32)
        for _ in range(count)
    ]


def _budget():
    return tl.Budget(run_ref="grad-norm", path=Path("/dev/null"), present=True)


def _train_leg(
    cfg,
    *,
    steps: int,
    accum: int = 0,
    metrics_path=None,
    batches=None,
    log_every: int = 0,
):
    """Нога ``train`` без управления env (гейт выставляет контекст)."""
    return tl.train(
        cfg,
        iter(batches if batches is not None else _batches(cfg, steps * 2)),
        train_config=tl.TrainConfig(
            steps=steps,
            lr=1e-2,
            seed=7,
            schedule="cosine",
            param_dtype="float32",
            accum_tokens=accum,
            micro_batch=1,
            metrics_path=metrics_path,
            log_every=log_every,
            kpi_every=20,
        ),
        budget=_budget(),
    )


@contextlib.contextmanager
def gate_env(value: str | None):
    """Временное значение ``AXIOM_ASYNC_METRICS`` (``None`` — снять)."""
    saved = os.environ.get(tl.ASYNC_METRICS_ENV)
    try:
        if value is None:
            os.environ.pop(tl.ASYNC_METRICS_ENV, None)
        else:
            os.environ[tl.ASYNC_METRICS_ENV] = value
        yield
    finally:
        if saved is None:
            os.environ.pop(tl.ASYNC_METRICS_ENV, None)
        else:
            os.environ[tl.ASYNC_METRICS_ENV] = saved


def _assert_grad_fields(row: dict) -> None:
    assert set(GRAD_FIELDS) <= set(row), f"поля нормы отсутствуют: {sorted(row)}"
    preclip = row["grad_norm_preclip"]
    assert isinstance(preclip, float), f"grad_norm_preclip не float: {type(preclip)}"
    assert math.isfinite(preclip) and preclip >= 0.0
    assert row["grad_norm_postclip"] == preclip
    assert row["grad_clip_coeff"] == 1.0


# ---------------------------------------------------------------------------
# (в) Интеграция: поля в записанной строке метрик
# ---------------------------------------------------------------------------


def test_metrics_row_carries_grad_norm_fields(tmp_path):
    """Синхронный путь: каждая строка журнала несёт три поля нормы."""
    cfg = tiny_config()
    metrics = tmp_path / "metrics.jsonl"
    with gate_env(None):
        _train_leg(cfg, steps=2, metrics_path=metrics)

    rows = tl.MetricsWriter.read(metrics)
    assert [row["step"] for row in rows] == [1, 2]
    for row in rows:
        _assert_grad_fields(row)


def test_accumulated_step_writes_grad_norm_of_stepped_gradient(tmp_path):
    """Python-ветка накопления: норма — у градиента, который получил шаг.

    Два микробатча складываются в один шаг оптимизатора; поле
    ``microbatches == 2`` подтверждает, что исполнена именно ветка накопления,
    а норма записана конечной и положительной.
    """
    cfg = tiny_config()
    metrics = tmp_path / "accum.jsonl"
    with gate_env(None):
        _train_leg(cfg, steps=1, accum=ACCUM2, metrics_path=metrics)

    row = tl.MetricsWriter.read(metrics)[0]
    assert row["microbatches"] == 2
    _assert_grad_fields(row)
    assert row["grad_norm_preclip"] > 0.0


def test_async_window_writes_same_grad_norm(tmp_path):
    """Окно async-метрик несёт те же значения нормы, что синхронный путь.

    Значения не пересчитываются при сбросе окна: это те же device-скаляры,
    только скопированные на хост реже.  Допуск — нулевой (``repr``).
    """
    cfg = tiny_config()
    batches = _batches(cfg, 4)
    off_path = tmp_path / "off.jsonl"
    on_path = tmp_path / "on.jsonl"
    with gate_env(None):
        _train_leg(cfg, steps=2, metrics_path=off_path, batches=list(batches))
    with gate_env("1"):
        _train_leg(cfg, steps=2, metrics_path=on_path, batches=list(batches))

    off_rows = tl.MetricsWriter.read(off_path)
    on_rows = tl.MetricsWriter.read(on_path)
    assert [row["step"] for row in on_rows] == [1, 2]
    assert [row["async_metrics"] for row in on_rows] == [1, 1]
    assert [
        repr(row["grad_norm_preclip"]) for row in off_rows
    ] == [
        repr(row["grad_norm_preclip"]) for row in on_rows
    ]


# ---------------------------------------------------------------------------
# (г) Нечисловая норма — в журнал как есть, не нулём
# ---------------------------------------------------------------------------


def test_non_finite_norm_is_written_as_is(monkeypatch, tmp_path):
    """Норма nan пишется в строку как nan (диагностика), а не 0.0.

    Патчится только вычислитель нормы: сам шаг и loss остаются штатными,
    поэтому строка обязана быть записана.
    """
    cfg = tiny_config()
    metrics = tmp_path / "nan.jsonl"

    def nan_norm(tree):
        return jnp.asarray(jnp.nan, dtype=jnp.float32)

    monkeypatch.setattr(tl, "tree_l2_norm", nan_norm)
    with gate_env(None):
        _train_leg(cfg, steps=1, metrics_path=metrics)

    row = tl.MetricsWriter.read(metrics)[0]
    assert math.isnan(row["grad_norm_preclip"])
    assert math.isnan(row["grad_norm_postclip"])
    assert row["grad_clip_coeff"] == 1.0


# ---------------------------------------------------------------------------
# (д) Обратная совместимость: старый парсер не падает на новых ключах
# ---------------------------------------------------------------------------


def test_metrics_reader_accepts_rows_with_new_keys(tmp_path):
    """``MetricsWriter.read`` читает строки со всеми тремя новыми ключами."""
    metrics = tmp_path / "old-parser.jsonl"
    row = {
        "schema": tl.METRICS_SCHEMA,
        "step": 1,
        "loss": 5.0,
        "step_seconds": 1.0,
        "grad_norm_preclip": 1.5,
        "grad_norm_postclip": 1.5,
        "grad_clip_coeff": 1.0,
    }
    metrics.write_text(json.dumps(row) + "\n", encoding="utf-8")

    rows = tl.MetricsWriter.read(metrics)
    assert len(rows) == 1
    assert rows[0]["loss"] == 5.0
    assert rows[0]["step_seconds"] == 1.0
    assert rows[0]["grad_norm_preclip"] == 1.5
