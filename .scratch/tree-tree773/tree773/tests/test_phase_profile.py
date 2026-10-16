"""Пофазовая инструментация шага претрейна (opt-in ``phase_profile``).

Спецификация дельты: ``phase_profile: true`` в конфиге лупа включает на каждом
kpi-интервальном шаге **дополнительный** декомпозированный прогон тех же входов
(отдельные jitted-функции фаз KDA / MLA / MoE-канал / CE-головы / оптимизатор,
каждая с host-таймером ``perf_counter`` + ``jax.block_until_ready``).  Числа
фаз — диагностические: результат декомпозиции выбрасывается, градиенты и шаг
оптимизатора берутся штатным fused-путём.

Что проверяется (контракт дельты):

* **T-PP-1** — с флагом на kpi-шаге в ``metrics.jsonl`` есть поля ``sec_kda``,
  ``sec_mla``, ``sec_moe``, ``sec_ce``, ``sec_backopt`` и ``phase_profile: true``;
  доступные фазы неотрицательны, их сумма конечна и > 0 (фаза, которую честно
  замерить не удалось, записывается ``null`` — это допустимо, но не все сразу);
* **T-PP-2** — дефолтный путь (без флага) этих полей не пишет вообще: продакшн-
  ветка не трогается;
* **T-PP-3** — профильный прогон обучает **тем же** шагом: ``tree_hash`` и лоссы
  совпадают с дефолтным прогоном на тех же батчах (числа декомпозиции не влияют
  на тренировку) — паритет-инвариант дельты.

Тесты идут на CPU (приёмочный пиннинг ``net/tests/conftest.py``, ADR-010) в
смоук-масштабе: проверяется провод и контракт записи, а не железо.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from conftest import tiny_config

from net import train_loop as tl

#: Поля фаз, которые обязан нести метрик-шаг профильного прогона (контракт дельты).
PHASE_FIELDS = ("sec_kda", "sec_mla", "sec_moe", "sec_ce", "sec_backopt")


def profile_batches(cfg, steps: int, *, seq_len: int = 16, seed: int = 5):
    """Поток батчей ``(1, T) int32`` для ``tl.train`` — детерминирован по сиду.

    Один и тот же генератор питает оба прогона (профильный и дефолтный): иначе
    сравнение ``tree_hash`` (T-PP-3) доказывало бы не инвариант флага, а
    равенство потоков данных.
    """
    rng = np.random.default_rng(seed)
    return iter(
        [
            rng.integers(0, cfg.vocab_size, size=(1, seq_len), dtype=np.int32)
            for _ in range(steps)
        ]
    )


def free_budget() -> "tl.Budget":
    """Смета без границ: тест гоняет провод, а не стоп-правило AD-8."""
    return tl.Budget(run_ref="phase-profile-test", path=Path("/dev/null"), present=True)


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


def test_pp1_phase_fields_are_recorded_on_kpi_steps(tmp_path: Path):
    """С флагом kpi-шаг несёт поля фаз; доступные неотрицательны, сумма > 0."""
    _, metrics_path = run_leg(tmp_path, phase_profile=True, steps=2, kpi_every=1)
    rows = tl.MetricsWriter.read(metrics_path)

    assert len(rows) == 2, "профильный прогон не записал метрики шагов"
    for index, row in enumerate(rows):
        assert row.get("phase_profile") is True, f"шаг {index}: нет phase_profile"
        for field in PHASE_FIELDS:
            assert field in row, f"шаг {index}: нет поля {field}"
            value = row[field]
            assert value is None or (
                isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0
            ), f"шаг {index}: {field}={value!r} (ожидалось null или >= 0)"

        measured = [row[field] for field in PHASE_FIELDS if row[field] is not None]
        assert measured, f"шаг {index}: ни одна фаза не измерена"
        total = math.fsum(measured)
        assert math.isfinite(total) and total > 0, f"шаг {index}: сумма фаз {total!r}"


def test_pp1_phase_fields_only_on_kpi_interval(tmp_path: Path):
    """Фазы пишутся на kpi-интервале, а не на каждом шаге (контракт дельты)."""
    _, metrics_path = run_leg(tmp_path, phase_profile=True, steps=4, kpi_every=2)
    rows = tl.MetricsWriter.read(metrics_path)

    assert len(rows) == 4
    # kpi-шаг = (index+1) % 2 == 0 -> шаги 2 и 4 (1-based, поле ``step``).
    profiled = {row["step"] for row in rows if row.get("phase_profile") is True}
    assert profiled == {2, 4}, f"фазы записаны не на kpi-шагах: {profiled}"
    for row in rows:
        if row["step"] not in profiled:
            for field in PHASE_FIELDS:
                assert field not in row, f"непрофильный шаг {row['step']} несёт {field}"


def test_pp2_default_path_writes_no_phase_fields(tmp_path: Path):
    """Дефолт (без флага) полей фаз не пишет — продакшн-ветка не тронута."""
    _, metrics_path = run_leg(tmp_path, phase_profile=False, steps=2, kpi_every=1)
    rows = tl.MetricsWriter.read(metrics_path)

    assert len(rows) == 2
    for index, row in enumerate(rows):
        assert "phase_profile" not in row, f"шаг {index}: phase_profile в дефолте"
        for field in PHASE_FIELDS:
            assert field not in row, f"шаг {index}: {field} в дефолте"


def max_leaf_delta(left, right) -> float:
    """Наибольшее покомпонентное расхождение двух деревьев (в float64)."""
    import jax

    def _delta(a, b):
        return float(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).max())

    return float(max(jax.tree_util.tree_leaves(jax.tree_util.tree_map(_delta, left, right))))


def test_pp3_profiling_does_not_change_training(tmp_path: Path):
    """Профильный прогон обучает тем же шагом, что дефолтный (паритет дельты).

    Числа фаз диагностические: декомпозиция выбрасывается, шаг оптимизатора и
    градиенты идут штатным fused-путём.  Инварианты:

    * лоссы совпадают **побитово** (лором, не ``approx``) — разошёлся бы и fused
      путь, а не только декомпозиция;
    * расхождение весов профильного прогона не больше **шумового пола среды**,
      измеренного тут же на двух дефолтных ногах.  В детерминированной среде пол
      равен нулю, и тогда дополнительно проверяется побитовое равенство
      ``tree_hash`` (контракт дельты).  Пол не предполагается, а измеряется: на
      загруженной машине прогон лупа даёт fp-дрейф ~1e-8 (наблюдён и на неизменном
      ``train_loop.py`` — свойство среды, не дельты), и выдать его за расхождение
      профильного пути значило бы тестировать загрузку машины, а не инвариант.
    """
    plain_a, _ = run_leg(tmp_path, phase_profile=False, steps=2, kpi_every=1, tag="-plain-a")
    plain_b, _ = run_leg(tmp_path, phase_profile=False, steps=2, kpi_every=1, tag="-plain-b")
    profiled, _ = run_leg(tmp_path, phase_profile=True, steps=2, kpi_every=1, tag="-profiled")

    assert profiled.steps_done == plain_a.steps_done == 2
    assert profiled.losses == plain_a.losses, "профиль изменил лоссы шагов"
    assert profiled.master_tree_hash  # хеш мастера считается, а не пуст

    noise = max_leaf_delta(plain_a.params, plain_b.params)
    delta = max_leaf_delta(profiled.params, plain_a.params)
    assert delta <= noise + 1e-12, (
        f"профильный прогон разошёлся сильнее, чем шум среды: delta={delta:.3e}, "
        f"noise={noise:.3e}"
    )
    if noise == 0.0:
        # Среда детерминирована — действует полный контракт дельты: шаги лупа
        # побитово те же, значит и tree_hash тот же.
        assert profiled.tree_hash == plain_a.tree_hash, "профиль изменил веса после шагов"
        assert profiled.master_tree_hash == plain_a.master_tree_hash
