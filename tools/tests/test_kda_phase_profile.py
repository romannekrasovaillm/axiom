"""Тесты прибора ``tools/kda_phase_profile.py`` (ADR-047 п. 3) — на CPU.

Проверяется контракт прибора, а не железо (числа фаз на CPU индикативны и
приёмкой не являются):

* **fail-closed без CUDA-устройства** — числа фаз не имитируются: отчёт уходит
  ``EMPTY-PENDING`` с причиной и ненулевым кодом возврата, геометрий в отчёте нет;
* **CPU-смоук** — прибор действительно меряет (малая геометрия), каждая фаза
  несёт время, долю и пиковую память, а цепочка стадий воспроизводит fused-ногу;
* **геометрия L3a** — те самые числа ADR-047 (T=8192, H=12, dk=dv=128, C=64);
* канонический путь отчёта — ``evidence/kda-rewrite/phase-profile.json``.

Модуль под тестом вызывает ``jax_preflight.ensure_mem_fraction()`` при импорте
(ADR-041: лимит памяти ДО ``import jax``) — тест возвращает окружение в исходное
состояние, чтобы прогон не наследовал переменную соседним модулям сьюта.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1]
CASE_DIR = TOOLS_DIR.parent
for _path in (str(TOOLS_DIR), str(CASE_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

#: Снимок окружения до импорта: модуль выставляет XLA_PYTHON_CLIENT_MEM_FRACTION.
_ENV_BEFORE = dict(os.environ)

import kda_phase_profile as kpp  # noqa: E402

if not _ENV_BEFORE.get("XLA_PYTHON_CLIENT_MEM_FRACTION"):
    os.environ.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)

PHASES = set(kpp.PHASE_ORDER)


@pytest.fixture(autouse=True)
def _no_stand_gate(monkeypatch: pytest.MonkeyPatch):
    """Гейт стенда (ADR-041) — не предмет этих тестов; на занятом GB10 он упал бы."""
    monkeypatch.setattr(kpp.jax_preflight, "gate_or_exit", lambda *a, **k: None)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# fail-closed без GPU
# ---------------------------------------------------------------------------


def test_without_gpu_is_fail_closed_and_fabricates_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Нет CUDA-устройства -> EMPTY-PENDING с причиной, ненулевой код, без чисел."""
    monkeypatch.setattr(kpp, "backend_is_gpu", lambda: False)
    out = tmp_path / "phase-profile.json"

    code = kpp.main(["--out", str(out)])

    assert code != 0, "отсутствие GPU обязано давать ненулевой код (fail-closed)"
    report = _read(out)
    assert report["status"] == kpp.STATUS_EMPTY
    assert report["geometries"] == {}, "числа не имитируются: геометрий быть не должно"
    assert "CUDA" in report["reason"], report["reason"]
    assert "phases" not in json.dumps(report), "в EMPTY-PENDING не должно быть фаз"


def test_smoke_without_gpu_is_allowed_and_measures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Явный --smoke на CPU честно меряет малую геометрию (SMOKE-CPU)."""
    monkeypatch.setattr(kpp, "backend_is_gpu", lambda: False)
    out = tmp_path / "smoke.json"

    code = kpp.main(["--smoke", "--impls", "chunked,chunked_cc", "--out", str(out)])

    assert code == 0
    report = _read(out)
    assert report["status"] == kpp.STATUS_SMOKE
    assert set(report["geometries"]) == {"smoke"}

    for impl, entry in report["geometries"]["smoke"]["impls"].items():
        assert set(entry["phases"]) == PHASES, impl
        shares = 0.0
        for name, phase in entry["phases"].items():
            assert phase["seconds"] >= 0.0, (impl, name)
            assert math.isfinite(phase["seconds"]), (impl, name)
            assert phase["calls"] > 0, (impl, name)
            assert phase["share"] is not None and phase["share"] >= 0.0
            shares += phase["share"]
        assert abs(shares - 1.0) <= 1e-6, (impl, shares)
        assert entry["phase_chain_matches_fused"] is True, impl


def test_smoke_report_carries_the_fused_legs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Fused-ноги есть, и backward заявлен производной, а не измеренной фазой."""
    monkeypatch.setattr(kpp, "backend_is_gpu", lambda: False)
    out = tmp_path / "smoke.json"
    assert kpp.main(["--smoke", "--impls", "chunked_cc", "--out", str(out)]) == 0

    entry = _read(out)["geometries"]["smoke"]["impls"]["chunked_cc"]
    fused = entry["fused"]
    for leg in ("fused_fwd", "fused_fwd_bwd"):
        assert fused[leg]["seconds_p50"] >= 0.0
        assert "xla_scratch_bytes" in fused[leg] and "xla_peak_bytes" in fused[leg]
    backward = fused["backward"]
    assert backward["derived"] is True
    assert backward["seconds_p50"] == pytest.approx(
        fused["fused_fwd_bwd"]["seconds_p50"] - fused["fused_fwd"]["seconds_p50"]
    )


# ---------------------------------------------------------------------------
# чистые части: статус, геометрия, путь отчёта
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("smoke", "gpu", "expected"),
    [
        (False, False, kpp.STATUS_EMPTY),
        (False, True, kpp.STATUS_OK),
        (True, False, kpp.STATUS_SMOKE),
        (True, True, kpp.STATUS_SMOKE),
    ],
)
def test_resolve_status(smoke: bool, gpu: bool, expected: str):
    assert kpp.resolve_status(smoke=smoke, gpu=gpu) == expected


def test_l3_geometry_is_the_adr_047_rank():
    """Геометрия L3a — объявленные ADR-047 числа (T=8192, H=12, dk=dv=128, C=64)."""
    geom = kpp.GEOMETRIES["l3"]
    assert (geom["seq_len"], geom["heads"], geom["dk"], geom["chunk"]) == (8192, 12, 128, 64)


@pytest.mark.parametrize("name", ["l3", "small"])
@pytest.mark.parametrize("impl", ["chunked", "wyut", "chunked_cc"])
def test_build_config_satisfies_the_schema(name: str, impl: str):
    """Конфиг профиля проходит validate_config для каждой формы."""
    cfg = kpp.build_config(kpp.GEOMETRIES[name], impl)
    assert cfg.kda_impl == impl
    assert cfg.hidden == kpp.GEOMETRIES[name]["heads"] * kpp.GEOMETRIES[name]["dk"]
    assert cfg.kda_wyut_chunk == kpp.GEOMETRIES[name]["chunk"]


def test_default_report_path_is_the_delta_path():
    expected = CASE_DIR / "evidence" / "kda-rewrite" / "phase-profile.json"
    assert kpp.DEFAULT_OUT == expected


def test_unknown_impl_is_rejected_by_the_cli():
    with pytest.raises(SystemExit):
        kpp.main(["--impls", "magic", "--smoke"])


def test_known_impls_are_exactly_the_three_forms():
    assert set(kpp.KNOWN_IMPLS) == {"chunked", "wyut", "chunked_cc"}


def test_table_renders_every_phase(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(kpp, "backend_is_gpu", lambda: False)
    out = tmp_path / "smoke.json"
    assert kpp.main(["--smoke", "--impls", "chunked_cc", "--out", str(out)]) == 0
    entry = _read(out)["geometries"]["smoke"]["impls"]["chunked_cc"]
    table = kpp.render_table("chunked_cc", entry)
    for name in kpp.PHASE_ORDER:
        assert name in table
    assert "fused_fwd" in table and "СУММА ФАЗ" in table
