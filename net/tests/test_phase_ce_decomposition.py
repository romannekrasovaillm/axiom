"""Декомпозиция CE-компонента шага (opt-in ``phase_profile``) — диагностика.

Мотив дельты: CE держит ~31 % шага l3-full и НЕ реагирует ни на dtype, ни на
``ce_chunk_tokens`` (fp32 ``sec_bwd_ce`` 3.850 с, bf16 — 3.887 с; 1024→4096 дал
лишь −3 %), то есть его узкое место не там, где его искали.  Профиль дополняется
ногами CE-головы: проекция в словарь (``sec_ce_logits``), приведение к loss
(``sec_ce_softmax``), MTP-голова (``sec_ce_mtp``) — плюс обратные двойники
(``sec_bwd_ce_*``), dtype стадий каждой ноги (``phase_dtype``) и пиковая
XLA-память исполнителей.

Что проверяется (контракт дельты):

* **T-CD-1** — ``reconcile_ce_legs`` — чистая арифметика: сумма ног-партиции
  (``sec_ce_ntp`` + ``sec_ce_mtp``) против ``sec_ce``, порог включителен,
  ``null``-нога не подставляется нулём, вырожденный CE даёт неопределённую
  сверку, а внутреннее расщепление NTP (logits/softmax) в сумму НЕ входит —
  иначе NTP считался бы дважды (ADR-011: несходимость видна, а не растворяется);
* **T-CD-2** — разложение не меняет математику: ``_phase_ce_head`` побитово
  равен ``_ce_ntp_head + _ce_mtp_head`` (наивный и chunked путь),
  ``_ce_logits_only`` — та же проекция, сведённая к скаляру;
* **T-CD-3** — dtype-проба: в fp32 logits/softmax/loss — float32; под
  ``AXIOM_COMPUTE_DTYPE=bf16`` операнд GEMM становится bfloat16, а logits —
  по-прежнему float32 (fp32-аккумулятор гейта) — это и есть ответ на вопрос
  «почему bf16 не помог», полученный кодом, а не прозой;
* **T-CD-4** — профильная нога несёт ноги CE, dtype-карту, память ног и
  сверку; разностная нога воспроизводится из самой строки метрик;
* **T-CD-5** — дефолт (без флага) полей CE не пишет, профиль не строится и
  лоссы шагов не меняет.

Тесты идут на CPU (приёмочный пиннинг ``net/tests/conftest.py``, ADR-010) в
смоук-масштабе: проверяется провод и контракт записи, а не железо.  Лимит
XLA-памяти выставляется до создания клиента XLA (ADR-041).
"""

from __future__ import annotations

import dataclasses
import math
import os
from pathlib import Path

# ADR-041: лимит памяти XLA — до инициализации рантайма (значение вызывающего
# неприкосновенно).  Держим рядом с остальными пинами контура (ADR-010).
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from conftest import tiny_config  # noqa: E402

from net import model  # noqa: E402
from net import train_loop as tl  # noqa: E402

#: Последовательность смоука (NTP видит T−1 строк, MTP — T−2).
SEQ = 16

#: Ширина чанка CE в chunked-тестах (< числа строк, чтобы чанков было несколько).
CHUNK = 4


def ce_params_ids(cfg, *, seed: int = 0, seq: int = SEQ):
    """Параметры и ``(1, T)`` токены конфига — входы CE-ног."""
    key = jr.PRNGKey(seed)
    params = model.init_params(key, cfg)
    ids = jr.randint(key, (1, seq), 1, cfg.vocab_size)  # 1..V−1: цели в диапазоне
    return params, ids


def with_ce_chunk(cfg, width: int):
    """Тот же конфиг с включённым chunked-путём CE."""
    return dataclasses.replace(cfg, ce_chunk_tokens=width)


# ---------------------------------------------------------------------------
# Провод ``tl.train`` (как в T-PP/T-BD: флаг — единственная разница)
# ---------------------------------------------------------------------------


def profile_batches(cfg, steps: int, *, seq_len: int = SEQ, seed: int = 5):
    rng = np.random.default_rng(seed)
    return iter(
        [
            rng.integers(0, cfg.vocab_size, size=(1, seq_len), dtype=np.int32)
            for _ in range(steps)
        ]
    )


def free_budget() -> "tl.Budget":
    return tl.Budget(run_ref="phase-ce-test", path=Path("/dev/null"), present=True)


def run_leg(
    tmp_path: Path,
    *,
    phase_profile: bool,
    steps: int = 2,
    kpi_every: int = 1,
    tag: str = "",
    ce_chunk_tokens: int | None = None,
):
    """Одна нога ``tl.train`` с метриками в ``tmp_path`` (флаг — разница)."""
    cfg = tiny_config()
    if ce_chunk_tokens is not None:
        cfg = with_ce_chunk(cfg, ce_chunk_tokens)
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
# T-CD-1: сверка ног CE — арифметика партиции
# ---------------------------------------------------------------------------


def test_cd1_ce_legs_are_diagnostic_legs_not_step_legs():
    """Ноги CE входят в профиль, но НЕ в сверку шага; партиция и расщепление названы."""
    assert set(tl.CE_LEG_FIELDS) <= set(tl.PHASE_PROFILE_FIELDS)
    assert set(tl.CE_LEG_FIELDS).isdisjoint(tl.RECONCILIATION_LEGS)
    assert set(tl.CE_PARTITION_LEGS) == {"sec_ce_ntp", "sec_ce_mtp"}
    assert set(tl.CE_INTERNAL_SPLIT_LEGS) == {"sec_ce_logits", "sec_ce_softmax"}
    assert set(tl.CE_PARTITION_LEGS) | set(tl.CE_INTERNAL_SPLIT_LEGS) == set(
        tl.CE_FWD_LEG_FIELDS
    )
    assert set(tl.CE_BWD_LEG_FIELDS) == {f"sec_bwd_{name[4:]}" for name in tl.CE_FWD_LEG_FIELDS}
    assert tl.CE_RECONCILIATION_FIELD not in tl.RECONCILIATION_LEGS
    assert tl.PHASE_DTYPE_FIELD not in tl.RECONCILIATION_LEGS


def test_cd1_partition_sum_reconciles_exactly():
    """Сумма ног-партиции = полному CE -> 100 %, находки нет, пометка на месте."""
    record = tl.reconcile_ce_legs(4.0, {"sec_ce_ntp": 3.0, "sec_ce_mtp": 1.0})
    assert record["legs_sum"] == pytest.approx(4.0)
    assert record["full_ce"] == pytest.approx(4.0)
    assert record["pct"] == pytest.approx(100.0)
    assert record["tolerance"] == tl.CE_RECONCILIATION_TOLERANCE
    assert record["finding"] is None
    assert record["note"] == tl.CE_RECONCILIATION_NOTE
    assert set(record["partition_legs"]) == set(tl.CE_PARTITION_LEGS)


def test_cd1_internal_split_legs_are_not_summed():
    """logits/softmax — расщепление ``sec_ce_ntp``: в сумму они не подкрадываются."""
    record = tl.reconcile_ce_legs(
        4.0,
        {
            "sec_ce_ntp": 3.0,
            "sec_ce_mtp": 1.0,
            "sec_ce_logits": 2.0,
            "sec_ce_softmax": 1.0,
        },
    )
    assert record["legs_sum"] == pytest.approx(4.0)  # не 8.0 — NTP не считается дважды
    assert record["pct"] == pytest.approx(100.0)
    assert record["finding"] is None


def test_cd1_threshold_is_inclusive_and_direction_agnostic():
    """Ровно на допуске — проход; дальше — находка (в обе стороны)."""
    tol = tl.CE_RECONCILIATION_TOLERANCE
    inside = tl.reconcile_ce_legs(10.0, {"sec_ce_ntp": 10.0 * (1.0 + tol)})
    assert inside["pct"] == pytest.approx(120.0)
    assert inside["finding"] is None

    outside = tl.reconcile_ce_legs(10.0, {"sec_ce_ntp": 10.0 * (1.0 + tol) + 1e-3})
    assert outside["finding"] == tl.FINDING_CE_LEGS_DIVERGE

    below = tl.reconcile_ce_legs(10.0, {"sec_ce_ntp": 10.0 * (1.0 - tol) - 1e-3})
    assert below["finding"] == tl.FINDING_CE_LEGS_DIVERGE


def test_cd1_null_leg_is_not_substituted_by_zero():
    """``null``-нога честно уменьшает сумму, а не подставляется нулём (ADR-011)."""
    record = tl.reconcile_ce_legs(10.0, {"sec_ce_ntp": 3.0, "sec_ce_mtp": None})
    assert record["legs_sum"] == pytest.approx(3.0)  # не 10.0
    assert record["pct"] == pytest.approx(30.0)
    assert record["finding"] == tl.FINDING_CE_LEGS_DIVERGE


def test_cd1_degenerate_full_ce_is_undefined_not_invented():
    """Полный CE неизвестен/неположителен — сверка не определена, пометка есть."""
    for full in (None, 0.0, -1.0):
        record = tl.reconcile_ce_legs(full, {"sec_ce_ntp": 1.0})
        assert record["legs_sum"] == pytest.approx(1.0)
        assert record["full_ce"] is None
        assert record["pct"] is None
        assert record["finding"] is None
        assert record["note"] == tl.CE_RECONCILIATION_NOTE


def test_cd1_no_measured_leg_gives_no_sum():
    """Ни одной измеренной ноги — суммы нет (не выдуманный ноль)."""
    record = tl.reconcile_ce_legs(10.0, {"sec_ce_ntp": None})
    assert record["legs_sum"] is None
    assert record["pct"] is None
    assert record["note"] == tl.CE_RECONCILIATION_NOTE


# ---------------------------------------------------------------------------
# T-CD-2: разложение не меняет математику CE
# ---------------------------------------------------------------------------


def test_cd2_phase_head_is_bitwise_sum_of_ntp_and_mtp_legs():
    """``_phase_ce_head`` = ``_ce_ntp_head + _ce_mtp_head`` — та же формула, побитово."""
    for cfg in (tiny_config(), with_ce_chunk(tiny_config(), CHUNK)):
        params, ids = ce_params_ids(cfg)
        hidden = params.embedding[ids]
        ce_tokens = int(cfg.ce_chunk_tokens)
        whole = tl._phase_ce_head(params, cfg, hidden, ids, SEQ, ce_tokens)
        split = tl._ce_ntp_head(params, cfg, hidden, ids, ce_tokens) + tl._ce_mtp_head(
            params, cfg, hidden, ids, SEQ, ce_tokens
        )
        assert jnp.array_equal(whole, split), "разложение изменило формулу CE"


def test_cd2_logits_only_leg_is_the_projection_reduced_to_scalar():
    """``_ce_logits_only`` — проекция NTP в словарь, сведённая ``jnp.sum`` (без softmax)."""
    cfg = tiny_config()  # ce_chunk_tokens=0 — наивный путь
    params, ids = ce_params_ids(cfg)
    hidden = params.embedding[ids]
    expected = jnp.sum(hidden[:, :-1] @ params.embedding.T)

    naive = tl._ce_logits_only(params, cfg, hidden, ids, 0)
    assert jnp.allclose(naive, expected, rtol=1e-6, atol=1e-6)

    # Chunked-путь: та же проекция по чанкам (иная порядок суммирования — допуск).
    chunked = tl._ce_logits_only(params, cfg, hidden, ids, CHUNK)
    assert jnp.allclose(chunked, expected, rtol=1e-4, atol=1e-4)


def test_cd2_derive_leg_difference_is_nonnegative_and_null_safe():
    """Разностная нога: ``full − part``, не меньше нуля; неизвестное — ``None``."""
    assert tl.derive_leg_difference(3.0, 1.0) == pytest.approx(2.0)
    assert tl.derive_leg_difference(1.0, 1.5) == 0.0  # шум таймера не даёт минуса
    assert tl.derive_leg_difference(None, 1.0) is None
    assert tl.derive_leg_difference(1.0, None) is None
    assert tl.derive_leg_difference(True, 1.0) is None  # bool — не измерение


# ---------------------------------------------------------------------------
# T-CD-3: dtype стадий CE-ног — «почему bf16 не помог» кодом
# ---------------------------------------------------------------------------


def cd_dtypes(cfg, *, ce_tokens: int):
    params, ids = ce_params_ids(cfg)
    return tl.ce_leg_dtypes(
        cfg,
        params=params,
        hidden=params.embedding[ids],
        input_ids=ids,
        ce_tokens=ce_tokens,
        chunk_size=SEQ,
    )


def test_cd3_dtype_map_covers_every_ce_leg_with_named_stages():
    """Карта dtype покрывает ноги CE; стадии названы, значения — непустые строки."""
    dtypes = cd_dtypes(tiny_config(), ce_tokens=0)
    assert set(tl.CE_LEG_FIELDS) <= set(dtypes)
    assert "sec_ce" in dtypes
    for leg, stages in dtypes.items():
        assert isinstance(stages, dict) and stages, f"{leg}: пустой отчёт стадий"
        for stage, value in stages.items():
            assert isinstance(stage, str) and stage
            assert isinstance(value, str) and value, f"{leg}.{stage}={value!r}"


def test_cd3_fp32_mode_logits_and_softmax_are_float32():
    """Без гейта операнд, logits, softmax и loss — float32 (обе головы)."""
    dtypes = cd_dtypes(tiny_config(), ce_tokens=0)
    for leg in ("sec_ce_logits", "sec_ce_ntp", "sec_ce_mtp"):
        assert dtypes[leg]["logits"] == "float32", f"{leg}: logits не fp32"
    assert dtypes["sec_ce_ntp"]["softmax"] == "float32"
    assert dtypes["sec_ce_ntp"]["loss"] == "float32"
    assert dtypes["sec_ce_logits"]["operand"] == "float32"


def test_cd3_bf16_gate_casts_the_operand_but_keeps_logits_fp32(monkeypatch):
    """bf16 кастует ОПЕРАНД GEMM, а logits остаются fp32 (аккумулятор) — ответ дельты."""
    cfg = tiny_config()
    monkeypatch.setenv("AXIOM_COMPUTE_DTYPE", "bf16")
    bf16 = cd_dtypes(cfg, ce_tokens=0)
    assert bf16["sec_ce_logits"]["operand"] == "bfloat16", "гейт не читается пробой"
    assert bf16["sec_ce_logits"]["logits"] == "float32", "аккумулятор должен быть fp32"
    assert bf16["sec_ce_ntp"]["logits"] == "float32"
    assert bf16["sec_ce_ntp"]["softmax"] == "float32"
    assert bf16["sec_ce_mtp"]["logits"] == "float32"

    monkeypatch.delenv("AXIOM_COMPUTE_DTYPE", raising=False)
    fp32 = cd_dtypes(cfg, ce_tokens=0)
    assert fp32["sec_ce_logits"]["operand"] == "float32"


def test_cd3_chunked_path_reports_the_same_stage_dtypes():
    """Chunked-путь (гейт `gemm`) даёт ту же картину dtype, что наивный."""
    dtypes = cd_dtypes(with_ce_chunk(tiny_config(), CHUNK), ce_tokens=CHUNK)
    assert dtypes["sec_ce_logits"]["logits"] == "float32"
    assert dtypes["sec_ce_ntp"]["softmax"] == "float32"


# ---------------------------------------------------------------------------
# T-CD-4: запись профильной ноги
# ---------------------------------------------------------------------------


def test_cd4_profiled_leg_records_ce_legs_dtypes_and_reconciliation(tmp_path: Path):
    """Профильная нога несёт ноги CE, dtype-карту, память ног и сверку."""
    _, metrics_path = run_leg(tmp_path, phase_profile=True, steps=2, kpi_every=1)
    rows = tl.MetricsWriter.read(metrics_path)

    assert len(rows) == 2, "профильный прогон не записал метрики шагов"
    for index, row in enumerate(rows):
        assert row.get("phase_profile") is True, f"шаг {index}: нет phase_profile"
        assert "sec_ce" in row, f"шаг {index}: пропала исходная нога sec_ce"

        measured = []
        for name in tl.CE_LEG_FIELDS:
            assert name in row, f"шаг {index}: нет поля {name}"
            value = row[name]
            assert value is None or (
                isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0
            ), f"шаг {index}: {name}={value!r} (ожидалось null или >= 0)"
            if value is not None:
                measured.append(name)
        for name in ("sec_ce_ntp", "sec_ce_logits", "sec_ce_mtp"):
            assert name in measured, f"шаг {index}: нога {name} не измерена"

        # Разностная нога воспроизводится из самой строки.
        if row["sec_ce_ntp"] is not None and row["sec_ce_logits"] is not None:
            expected = max(0.0, row["sec_ce_ntp"] - row["sec_ce_logits"])
            assert row["sec_ce_softmax"] == pytest.approx(expected), (
                f"шаг {index}: sec_ce_softmax не разность"
            )
        if row["sec_bwd_ce_ntp"] is not None and row["sec_bwd_ce_logits"] is not None:
            expected = max(0.0, row["sec_bwd_ce_ntp"] - row["sec_bwd_ce_logits"])
            assert row["sec_bwd_ce_softmax"] == pytest.approx(expected)

        # Сверка CE — арифметика партиции, пометка обязательна.
        ce = row[tl.CE_RECONCILIATION_FIELD]
        assert isinstance(ce, dict), f"шаг {index}: нет сверки CE"
        assert ce["note"] == tl.CE_RECONCILIATION_NOTE
        partition = [row[name] for name in tl.CE_PARTITION_LEGS if row[name] is not None]
        assert ce["legs_sum"] == pytest.approx(math.fsum(partition))
        if ce["full_ce"] is not None:
            assert ce["pct"] == pytest.approx(100.0 * ce["legs_sum"] / ce["full_ce"])
            expected_finding = (
                tl.FINDING_CE_LEGS_DIVERGE
                if abs(ce["pct"] - 100.0) > tl.CE_RECONCILIATION_TOLERANCE * 100.0
                else None
            )
            assert ce["finding"] == expected_finding, f"шаг {index}: находка не согласована"

        # dtype-карта CE-ног: покрытие и непустые значения.
        dtypes = row[tl.PHASE_DTYPE_FIELD]
        assert isinstance(dtypes, dict), f"шаг {index}: нет phase_dtype"
        assert set(tl.CE_FWD_LEG_FIELDS) <= set(dtypes)
        for leg, stages in dtypes.items():
            assert isinstance(stages, dict) and stages, f"шаг {index}: {leg} без стадий"
            for stage, value in stages.items():
                assert isinstance(value, str) and value, f"шаг {index}: {leg}.{stage}={value!r}"

        # Память ног: ключ на каждую ногу CE (профиль уже покрывает поля целиком).
        memory = row[tl.PHASE_MEMORY_FIELD]
        assert isinstance(memory, dict)
        for name in tl.CE_LEG_FIELDS:
            assert name in memory, f"шаг {index}: память без ноги {name}"


def test_cd4_chunked_ce_path_records_the_legs(tmp_path: Path):
    """С включённым ``ce_chunk_tokens`` ноги CE меряются тем же проводом."""
    _, metrics_path = run_leg(
        tmp_path, phase_profile=True, steps=2, kpi_every=1, ce_chunk_tokens=CHUNK
    )
    rows = tl.MetricsWriter.read(metrics_path)
    for index, row in enumerate(rows):
        for name in ("sec_ce_ntp", "sec_ce_logits", "sec_ce_mtp"):
            assert row.get(name) is not None, f"шаг {index}: chunked-нога {name} не измерена"


# ---------------------------------------------------------------------------
# T-CD-5: дефолт не тронут, результат не изменён
# ---------------------------------------------------------------------------


def test_cd5_default_path_writes_no_ce_fields(tmp_path: Path):
    """Дефолт (без флага) полей CE не пишет — продакшн-ветка не тронута."""
    _, metrics_path = run_leg(tmp_path, phase_profile=False, steps=2, kpi_every=1)
    rows = tl.MetricsWriter.read(metrics_path)

    assert len(rows) == 2
    for index, row in enumerate(rows):
        for name in tl.CE_LEG_FIELDS:
            assert name not in row, f"шаг {index}: {name} в дефолте"
        assert tl.CE_RECONCILIATION_FIELD not in row, f"шаг {index}: сверка CE в дефолте"
        assert tl.PHASE_DTYPE_FIELD not in row, f"шаг {index}: dtype-карта в дефолте"


def test_cd5_no_flag_builds_no_profiler(tmp_path: Path, monkeypatch):
    """Без флага профильные проходы CE не заводятся вовсе (шаг не платит)."""

    class _SpyProfiler:
        constructions = 0

        def __init__(self, cfg, train_config, *, grad_fn, forward_fn):
            type(self).constructions += 1

        def measure(self, **kwargs):
            return {name: None for name in tl.PHASE_PROFILE_FIELDS}

    monkeypatch.setattr(tl, "_PhaseProfiler", _SpyProfiler)
    run_leg(tmp_path, phase_profile=False, steps=1, kpi_every=1, tag="-no-flag")
    assert _SpyProfiler.constructions == 0, "профиль построен без флага"


def test_cd5_flag_does_not_change_losses(tmp_path: Path):
    """Диагностические CE-проходы выбрасываются: лоссы 2 шагов совпадают."""
    plain, _ = run_leg(tmp_path, phase_profile=False, steps=2, kpi_every=1, tag="-plain")
    profiled, _ = run_leg(
        tmp_path, phase_profile=True, steps=2, kpi_every=1, tag="-profiled"
    )
    assert profiled.steps_done == plain.steps_done == 2
    assert profiled.losses == plain.losses, "профиль CE изменил лоссы шагов"
