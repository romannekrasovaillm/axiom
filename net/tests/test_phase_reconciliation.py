"""Сверка суммы фаз шага с его временем (следствие: диагностика покрывала 22%).

Расширение ``phase_profile`` (``net/train_loop.py``): кроме декомпозиции прямого
прохода (kda/mla/moe/ce) профиль несёт ноги **сверки** — ``sec_forward``,
``sec_backward`` (оценка обратного прохода разностью «fwd+bwd минус forward» на
реальном ``grad_fn``), ``sec_backopt``, — остаток ``sec_other`` (шаг − сумма ног,
явно НЕ «прочее»), справку ``sec_loader`` (вне шага) и долю покрытия
``reconciliation_pct``.  Покрытие ниже 95% — находка ``unexplained_share_high``,
а не тишина (ADR-011).

Что проверяется (контракт дельты):

* **T-PR-1** — ``reconcile_phases`` — чистая арифметика: ``sec_other`` = шаг −
  сумма измеренных ног, порог 95% включителен; ноги декомпозиции и лоадер в
  сумму покрытия НЕ входят (иначе forward считался бы дважды);
* **T-PR-2** — неполный набор ног (одна ``null``) роняет покрытие и поднимает
  находку ``unexplained_share_high``; вырожденный шаг (``None``/``0``) даёт
  неопределённую сверку, а не выдуманную;
* **T-PR-3** — без флага профильные проходы не заводятся (``_PhaseProfiler`` не
  строится); с флагом — строится;
* **T-PR-4** — реальная профильная нога несёт новые ноги сверки, и арифметика
  записи воспроизводится из неё же.

Тесты идут на CPU (приёмочный пиннинг ``net/tests/conftest.py``, ADR-010) в
смоук-масштабе: проверяется провод и контракт записи, а не железо.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from conftest import tiny_config

from net import train_loop as tl

#: Ноги покрытия (контракт сверки) и ноги, которые в него не входят.
COVERING_LEGS = ("sec_forward", "sec_backward", "sec_backopt")
NON_COVERING_LEGS = ("sec_kda", "sec_mla", "sec_moe", "sec_ce", "sec_loader")


def profile_batches(cfg, steps: int, *, seq_len: int = 16, seed: int = 5):
    """Поток батчей ``(1, T) int32`` — детерминирован по сиду (как в T-PP)."""
    rng = np.random.default_rng(seed)
    return iter(
        [
            rng.integers(0, cfg.vocab_size, size=(1, seq_len), dtype=np.int32)
            for _ in range(steps)
        ]
    )


def free_budget() -> "tl.Budget":
    """Смета без границ: тест гоняет провод, а не стоп-правило AD-8."""
    return tl.Budget(run_ref="phase-reconciliation-test", path=Path("/dev/null"), present=True)


def run_leg(
    tmp_path: Path,
    *,
    phase_profile: bool,
    steps: int = 2,
    kpi_every: int = 1,
    tag: str = "",
):
    """Одна нога ``tl.train`` с метриками в ``tmp_path`` (флаг — единственная разница)."""
    cfg = tiny_config()
    metrics_path = tmp_path / f"metrics{tag}.jsonl"
    result = tl.train(
        cfg,
        profile_batches(cfg, steps),
        train_config=tl.TrainConfig(
            steps=steps,
            lr=1e-2,
            seed=7,
            schedule="cosine",
            param_dtype="float32",
            grad_checkpointing=False,
            kpi_every=kpi_every,
            phase_profile=phase_profile,
            metrics_path=metrics_path,
        ),
        budget=free_budget(),
    )
    return result, metrics_path


def test_pr1_reconcile_is_arithmetic_and_excludes_sublegs():
    """Сверка — арифметика: остаток = шаг − сумма; декомпозиция и лоадер не в сумме."""
    assert set(COVERING_LEGS) == set(tl.RECONCILIATION_LEGS)
    assert set(tl.RECONCILIATION_LEGS).isdisjoint(NON_COVERING_LEGS)

    record = tl.reconcile_phases(
        10.0, {"sec_forward": 4.0, "sec_backward": 4.5, "sec_backopt": 1.0}
    )
    assert record["reconciliation_pct"] == pytest.approx(95.0)  # порог включителен
    assert record["sec_other"] == pytest.approx(0.5)  # 10.0 − 9.5, остаток
    assert record["finding"] is None

    # Ноги вне сверки не подкрадываются в сумму, даже если переданы рядом.
    with_sublegs = tl.reconcile_phases(
        10.0,
        {
            "sec_forward": 4.0,
            "sec_backward": 4.5,
            "sec_backopt": 1.0,
            "sec_kda": 3.0,
            "sec_loader": 2.0,
        },
    )
    assert with_sublegs["reconciliation_pct"] == pytest.approx(95.0)
    assert with_sublegs["sec_other"] == pytest.approx(0.5)


def test_pr1_just_below_threshold_raises_finding():
    """Ровно под порогом — находка; остаток остаётся честным вычетом."""
    record = tl.reconcile_phases(10.0, {"sec_forward": 9.4})
    assert record["reconciliation_pct"] == pytest.approx(94.0)
    assert record["sec_other"] == pytest.approx(0.6)
    assert record["finding"] == tl.FINDING_UNEXPLAINED_SHARE_HIGH


def test_pr2_incomplete_set_raises_finding():
    """``null``-нога не подставляется нулём: покрытие падает, находка срабатывает."""
    record = tl.reconcile_phases(
        20.2, {"sec_forward": 3.0, "sec_backward": None, "sec_backopt": 1.1}
    )
    assert record["reconciliation_pct"] == pytest.approx(100.0 * 4.1 / 20.2)
    assert record["reconciliation_pct"] < 95.0
    assert record["finding"] == tl.FINDING_UNEXPLAINED_SHARE_HIGH


def test_pr2_degenerate_step_is_undefined_not_invented():
    """Шаг неизвестен/неположителен — сверка неопределена, без деления на ноль."""
    for step in (None, 0.0, -1.0):
        record = tl.reconcile_phases(step, {"sec_forward": 3.0})
        assert record == {"reconciliation_pct": None, "sec_other": None, "finding": None}


class _SpyProfiler:
    """Считает факты постройки: без флага профиль не должен строиться вообще."""

    constructions = 0

    def __init__(self, cfg, train_config, *, grad_fn, forward_fn):
        type(self).constructions += 1

    def measure(self, **kwargs):
        return {name: None for name in tl.PHASE_PROFILE_FIELDS}


def test_pr3_no_flag_builds_no_profiler(tmp_path: Path, monkeypatch):
    """Дефолт (без флага) не заводит профильных проходов; с флагом — заводит."""
    monkeypatch.setattr(tl, "_PhaseProfiler", _SpyProfiler)

    _SpyProfiler.constructions = 0
    run_leg(tmp_path, phase_profile=False, steps=1, kpi_every=1, tag="-no-flag")
    assert _SpyProfiler.constructions == 0, "профиль построен без флага"

    _SpyProfiler.constructions = 0
    run_leg(tmp_path, phase_profile=True, steps=1, kpi_every=1, tag="-with-flag")
    assert _SpyProfiler.constructions == 1, "профиль не построен с флагом"


def test_pr4_profiled_leg_records_reconciliation(tmp_path: Path):
    """Профильная нога несёт ноги сверки; арифметика записи самосогласована."""
    _, metrics_path = run_leg(tmp_path, phase_profile=True, steps=2, kpi_every=1)
    rows = tl.MetricsWriter.read(metrics_path)

    assert len(rows) == 2, "профильный прогон не записал метрики шагов"
    for index, row in enumerate(rows):
        assert row.get("phase_profile") is True, f"шаг {index}: нет phase_profile"
        for name in tl.PHASE_PROFILE_FIELDS:
            assert name in row, f"шаг {index}: нет поля {name}"

        loader = row["sec_loader"]
        assert loader is None or loader >= 0.0, f"шаг {index}: sec_loader={loader!r}"

        step = row["step_seconds"]
        measured = [
            row[name] for name in tl.RECONCILIATION_LEGS if row.get(name) is not None
        ]
        assert measured, f"шаг {index}: ни одна нога сверки не измерена"
        total = math.fsum(measured)

        # Записанная сверка воспроизводится из самих чисел строки.
        assert row["reconciliation_pct"] == pytest.approx(100.0 * total / step)
        assert row["sec_other"] == pytest.approx(step - total)
        expected_finding = (
            tl.FINDING_UNEXPLAINED_SHARE_HIGH
            if row["reconciliation_pct"] < tl.UNEXPLAINED_SHARE_THRESHOLD * 100.0
            else None
        )
        assert row["finding"] == expected_finding, f"шаг {index}: находка не согласована"


def test_pr5_flag_does_not_change_losses(tmp_path: Path):
    """Флаг не меняет численный результат: лоссы 2 шагов (один seed) совпадают.

    Ноги сверки — дополнительные ЧТЕНИЯ тех же параметров (forward и grad
    выбрасываются): тренировка идёт штатным fused-путём, поэтому лоссы побитово
    те же.  Паритет весов (``tree_hash``) с шумовым полом среды — в T-PP-3.
    """
    plain, _ = run_leg(tmp_path, phase_profile=False, steps=2, kpi_every=1, tag="-plain")
    profiled, _ = run_leg(
        tmp_path, phase_profile=True, steps=2, kpi_every=1, tag="-profiled"
    )
    assert profiled.steps_done == plain.steps_done == 2
    assert profiled.losses == plain.losses, "профиль изменил лоссы шагов"
