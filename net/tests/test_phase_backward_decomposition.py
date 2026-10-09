"""Декомпозиция backward шага претрейна (opt-in ``phase_profile``) — диагностика.

Мотив дельты: ``sec_backward`` занимает ~73 % шага l3-full, а удвоение пика
матмулов (bf16 против fp32) ускоряет его лишь ×1.05 — то есть обратный проход
упирается в память/трафик активаций, а не в FLOPs.  Профиль дополняется ногами
по компонентам — отдельными ``jax.grad``-проходами по скалярному отклику своего
подмножества (``sec_bwd_kda`` / ``sec_bwd_mla`` / ``sec_bwd_moe`` /
``sec_bwd_ce`` / ``sec_bwd_other``) — и пиковой XLA-памятью их исполнителей.

Что проверяется (контракт дельты):

* **T-BD-1** — ``reconcile_backward_legs`` — чистая арифметика ориентира: сумма
  ног vs ``sec_backward``, порог ``BWD_RECONCILIATION_TOLERANCE`` включителен,
  пометка «ориентир, не партиция» присутствует всегда, ``null``-нога не
  подставляется нулём, вырожденный полный backward даёт неопределённую сверку;
* **T-BD-2** — профильная нога несёт все новые поля: ноги backward
  (``BWD_LEG_FIELDS``), карту памяти (``phase_memory_bytes``) и сверку
  (``bwd_reconciliation``); арифметика сверки воспроизводится из самой строки;
* **T-BD-3** — дефолт (без флага) новых полей не пишет — продакшн-ветка не тронута;
* **T-BD-4** — флаг не меняет численный результат: лоссы 2 шагов совпадают;
* **T-BD-5** — ``render_phase_memory`` — чистая и ``null``-безопасная.

Тесты идут на CPU (приёмочный пиннинг ``net/tests/conftest.py``, ADR-010) в
смоук-масштабе: проверяется провод и контракт записи, а не железо.  Лимит
XLA-памяти выставляется до создания клиента XLA (ADR-041): без него JAX
резервирует дефолтные ~75 % устройства.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

# ADR-041: лимит памяти XLA — до инициализации рантайма (значение вызывающего
# неприкосновенно).  Держим рядом с остальными пинами контура (ADR-010).
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from conftest import tiny_config  # noqa: E402

from net import train_loop as tl  # noqa: E402


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
    return tl.Budget(run_ref="phase-backward-test", path=Path("/dev/null"), present=True)


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


# ---------------------------------------------------------------------------
# T-BD-1: сверка backward-ног — арифметика ориентира
# ---------------------------------------------------------------------------


def test_bd1_backward_legs_sum_reconciles_exactly_within_tolerance():
    """Сумма ног = полному backward -> 100 %, находки нет, пометка на месте."""
    legs = {
        "sec_bwd_kda": 2.0,
        "sec_bwd_mla": 3.0,
        "sec_bwd_moe": 4.0,
        "sec_bwd_ce": 1.0,
        "sec_bwd_other": 0.0,
    }
    record = tl.reconcile_backward_legs(10.0, legs)
    assert record["legs_sum"] == pytest.approx(10.0)
    assert record["full_backward"] == pytest.approx(10.0)
    assert record["pct"] == pytest.approx(100.0)
    assert record["tolerance"] == tl.BWD_RECONCILIATION_TOLERANCE
    assert record["finding"] is None
    assert record["note"] == tl.BWD_RECONCILIATION_NOTE


def test_bd1_threshold_is_inclusive_and_direction_agnostic():
    """Ровно на допуске — проход; дальше — находка (в обе стороны)."""
    inside = tl.reconcile_backward_legs(
        10.0, {"sec_bwd_kda": 10.0 * (1.0 + tl.BWD_RECONCILIATION_TOLERANCE)}
    )
    assert inside["pct"] == pytest.approx(120.0)
    assert inside["finding"] is None

    outside = tl.reconcile_backward_legs(
        10.0, {"sec_bwd_kda": 10.0 * (1.0 + tl.BWD_RECONCILIATION_TOLERANCE) + 1e-3}
    )
    assert outside["finding"] == tl.FINDING_BACKWARD_LEGS_DIVERGE

    below = tl.reconcile_backward_legs(
        10.0, {"sec_bwd_kda": 10.0 * (1.0 - tl.BWD_RECONCILIATION_TOLERANCE) - 1e-3}
    )
    assert below["finding"] == tl.FINDING_BACKWARD_LEGS_DIVERGE


def test_bd1_null_leg_is_not_substituted_by_zero():
    """``null``-нога честно уменьшает сумму, а не подставляется нулём (ADR-011)."""
    record = tl.reconcile_backward_legs(
        10.0, {"sec_bwd_kda": 3.0, "sec_bwd_mla": None, "sec_bwd_moe": 3.0}
    )
    assert record["legs_sum"] == pytest.approx(6.0)  # не 10.0
    assert record["pct"] == pytest.approx(60.0)
    assert record["finding"] == tl.FINDING_BACKWARD_LEGS_DIVERGE


def test_bd1_degenerate_full_backward_is_undefined_not_invented():
    """Полный backward неизвестен/неположителен — сверка не определена, пометка есть."""
    for full in (None, 0.0, -1.0):
        record = tl.reconcile_backward_legs(full, {"sec_bwd_kda": 1.0})
        assert record["legs_sum"] == pytest.approx(1.0)
        assert record["full_backward"] is None
        assert record["pct"] is None
        assert record["finding"] is None
        assert record["note"] == tl.BWD_RECONCILIATION_NOTE


def test_bd1_no_measured_leg_gives_no_sum():
    """Ни одной измеренной ноги — суммы нет (не выдуманный ноль)."""
    record = tl.reconcile_backward_legs(10.0, {"sec_bwd_kda": None})
    assert record["legs_sum"] is None
    assert record["pct"] is None
    assert record["note"] == tl.BWD_RECONCILIATION_NOTE


def test_bd1_leg_fields_are_a_diagnostic_partition_of_the_profile():
    """Ноги backward входят в поля профиля и НЕ входят в сверку шага."""
    assert set(tl.BWD_LEG_FIELDS) <= set(tl.PHASE_PROFILE_FIELDS)
    assert set(tl.BWD_LEG_FIELDS).isdisjoint(tl.RECONCILIATION_LEGS)
    assert tl.BWD_RECONCILIATION_FIELD not in tl.RECONCILIATION_LEGS
    assert tl.PHASE_MEMORY_FIELD not in tl.RECONCILIATION_LEGS


# ---------------------------------------------------------------------------
# T-BD-2: запись профильной ноги
# ---------------------------------------------------------------------------


def test_bd2_profiled_leg_records_backward_legs_memory_and_reconciliation(
    tmp_path: Path,
):
    """Профильная нога несёт ноги backward, память ног и сверку-ориентир."""
    _, metrics_path = run_leg(tmp_path, phase_profile=True, steps=2, kpi_every=1)
    rows = tl.MetricsWriter.read(metrics_path)

    assert len(rows) == 2, "профильный прогон не записал метрики шагов"
    for index, row in enumerate(rows):
        assert row.get("phase_profile") is True, f"шаг {index}: нет phase_profile"

        # Ноги backward: поле есть, значение null или неотрицательное число.
        measured = []
        for name in tl.BWD_LEG_FIELDS:
            assert name in row, f"шаг {index}: нет поля {name}"
            value = row[name]
            assert value is None or (
                isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0
            ), f"шаг {index}: {name}={value!r} (ожидалось null или >= 0)"
            if value is not None:
                measured.append(value)
        assert measured, f"шаг {index}: ни одна backward-нога не измерена"

        # Карта памяти: ключ на каждую ногу профиля; значение null или разбор XLA.
        memory = row[tl.PHASE_MEMORY_FIELD]
        assert isinstance(memory, dict), f"шаг {index}: phase_memory_bytes не словарь"
        assert set(memory) == set(tl.PHASE_PROFILE_FIELDS), (
            f"шаг {index}: память не покрывает поля профиля"
        )
        for name, entry in memory.items():
            if entry is None:
                continue
            assert isinstance(entry, dict), f"шаг {index}: память {name} не разбор"
            peak = entry.get("peak_bytes")
            assert peak is None or (isinstance(peak, int) and peak >= 0), (
                f"шаг {index}: память {name} peak_bytes={peak!r}"
            )

        # Сверка backward — ориентир, а не партиция: пометка обязательна.
        bwd = row[tl.BWD_RECONCILIATION_FIELD]
        assert isinstance(bwd, dict), f"шаг {index}: нет сверки backward"
        assert bwd["note"] == tl.BWD_RECONCILIATION_NOTE
        assert bwd["legs_sum"] == pytest.approx(math.fsum(measured))
        if bwd["full_backward"] is not None:
            # Арифметика записи воспроизводится из самой строки.
            assert bwd["pct"] == pytest.approx(100.0 * bwd["legs_sum"] / bwd["full_backward"])
            expected = (
                tl.FINDING_BACKWARD_LEGS_DIVERGE
                if abs(bwd["pct"] - 100.0) > tl.BWD_RECONCILIATION_TOLERANCE * 100.0
                else None
            )
            assert bwd["finding"] == expected, f"шаг {index}: находка не согласована"


def test_bd2_backward_legs_do_not_enter_step_reconciliation(tmp_path: Path):
    """Ноги backward не партиционируют шаг: сверка шага считается как раньше."""
    _, metrics_path = run_leg(tmp_path, phase_profile=True, steps=2, kpi_every=1)
    rows = tl.MetricsWriter.read(metrics_path)

    for index, row in enumerate(rows):
        step = row["step_seconds"]
        covering = [
            row[name] for name in tl.RECONCILIATION_LEGS if row.get(name) is not None
        ]
        assert covering, f"шаг {index}: ни одна нога сверки не измерена"
        total = math.fsum(covering)
        assert row["reconciliation_pct"] == pytest.approx(100.0 * total / step)
        assert row["sec_other"] == pytest.approx(step - total)


# ---------------------------------------------------------------------------
# T-BD-3/T-BD-4: флаг не трогает продакшн-путь и результат
# ---------------------------------------------------------------------------


def test_bd3_default_path_writes_no_backward_fields(tmp_path: Path):
    """Дефолт (без флага) новых полей не пишет — продакшн-ветка не тронута."""
    _, metrics_path = run_leg(tmp_path, phase_profile=False, steps=2, kpi_every=1)
    rows = tl.MetricsWriter.read(metrics_path)

    assert len(rows) == 2
    for index, row in enumerate(rows):
        assert "phase_profile" not in row, f"шаг {index}: phase_profile в дефолте"
        for name in tl.BWD_LEG_FIELDS:
            assert name not in row, f"шаг {index}: {name} в дефолте"
        assert tl.PHASE_MEMORY_FIELD not in row, f"шаг {index}: память в дефолте"
        assert tl.BWD_RECONCILIATION_FIELD not in row, f"шаг {index}: сверка в дефолте"
        assert "sec_backward" not in row, f"шаг {index}: sec_backward в дефолте"


def test_bd4_flag_does_not_change_losses(tmp_path: Path):
    """Диагностические grad-проходы выбрасываются: лоссы 2 шагов совпадают."""
    plain, _ = run_leg(tmp_path, phase_profile=False, steps=2, kpi_every=1, tag="-plain")
    profiled, _ = run_leg(
        tmp_path, phase_profile=True, steps=2, kpi_every=1, tag="-profiled"
    )
    assert profiled.steps_done == plain.steps_done == 2
    assert profiled.losses == plain.losses, "профиль изменил лоссы шагов"


# ---------------------------------------------------------------------------
# T-BD-5: рендер памяти ног — чистая функция
# ---------------------------------------------------------------------------


def test_bd5_render_phase_memory_is_null_safe_and_pure():
    """``render_phase_memory`` не падает на ``None`` и показывает МиБ/``null``."""
    empty = tl.render_phase_memory(None)
    assert empty.count("null") == len(tl.PHASE_PROFILE_FIELDS)
    assert "sec_kda=null" in empty

    memory = {"sec_kda": {"peak_bytes": 1024 * 1024}, "sec_bwd_mla": None}
    rendered = tl.render_phase_memory(memory)
    assert "sec_kda=1.0МиБ" in rendered
    assert "sec_bwd_mla=null" in rendered
    assert rendered == tl.render_phase_memory(memory)  # без побочных эффектов
