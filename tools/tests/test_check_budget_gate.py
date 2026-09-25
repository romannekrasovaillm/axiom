"""Тесты поведенческого стража лимита стоимости C-041 (AD-8, ADR-011).

Страж решает, стартует прогон или нет, поэтому его проверки закреплены
фикстурами: смета отсутствует / превышает лимит / без стоп-правила /
датирована после прогона — красный гейт; состоятельная смета — зелёный.
Генератор смет проверяется на отказ без входных данных: он не имеет права
создать артефакт «из воздуха» (фабрикация доказательства, AD-8).

Обязательные сценарии (из постановки дельты):
  (i)    нет сметы -> FAIL и «смета отсутствует: запуск блокирован»;
  (ii)   usd_estimate > limit_usd -> FAIL;
  (iii)  смета без stop_rule -> FAIL;
  (iv)   смета, созданная ПОСЛЕ даты прогона -> FAIL;
  (v)    корректная смета (фикстура в tmp) -> PASS;
  (vi)   генерация без входных данных -> ненулевой exit, файл не создан.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import check_budget_gate as gate  # noqa: E402  (путь добавляется выше)


# --- построение фикстур -----------------------------------------------------


def _estimate(**overrides: Any) -> dict:
    """Состоятельная смета; переопределения задают проверяемый дефект."""
    data = {
        "schema": gate.ESTIMATE_SCHEMA,
        "run_ref": "a4-skeleton",
        "gpu_type": "H800",
        "gpu_hours_estimate": 97.0,
        "usd_estimate": 180.0,
        "limit_usd": 200.0,
        "budget_method": gate.CALIBRATION_DEFAULT,
        "stop_rule": gate.STOP_RULE_DEFAULT,
        "created_at": "2026-09-12T10:00:00+00:00",
        "approved_by": "владелец",
    }
    data.update(overrides)
    return data


def _write_estimate(case: Path, run_ref: str, data: dict) -> Path:
    path = case / gate.BUDGET_DIR / f"{run_ref}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_manifest(case: Path, run_ref: str, run_date: str) -> Path:
    path = case / "evidence" / f"{run_ref}-run-manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"run_ref": run_ref, "run_date": run_date, "schema": "fixture"}),
        encoding="utf-8",
    )
    return path


def _run(case: Path, *extra: str) -> int:
    return gate.main(["--case-dir", str(case), *extra])


def _verify(case: Path, run_ref: str = "a4-skeleton") -> tuple[int, str, str]:
    """Прогон стража по одному прогону; возвращает (код, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = gate.main(["--case-dir", str(case), "--verify", "--run-ref", run_ref])
    return code, out.getvalue(), err.getvalue()


def _verify_all(case: Path) -> tuple[int, str, str]:
    """Прогон стража по реестру прогонов кейса (как вызывает C-041)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = gate.main(["--case-dir", str(case), "--verify"])
    return code, out.getvalue(), err.getvalue()


def _remove(data: dict, field: str) -> dict:
    data.pop(field, None)
    return data


# --- (i) сметы нет ----------------------------------------------------------


def test_missing_estimate_blocks_run(tmp_path: Path) -> None:
    """(i) Нет файла сметы — запуск блокирован, ненулевой код."""
    code, out, _ = _verify(tmp_path)

    assert code != 0
    assert "смета отсутствует: запуск блокирован" in out
    assert out.strip().endswith("запуск блокирован (AD-8)")


def test_declared_runs_without_estimates_all_fail(tmp_path: Path) -> None:
    """Без --run-ref страж требует сметы всех объявленных прогонов кейса."""
    code, out, _ = _verify_all(tmp_path)

    assert code != 0
    for run_ref in gate.DECLARED_RUN_REFS:
        assert f"[{run_ref}] FAIL" in out
    assert "смета отсутствует: запуск блокирован" in out


# --- (ii) превышение лимита -------------------------------------------------


def test_usd_over_limit_fails(tmp_path: Path) -> None:
    """(ii) Смета дороже лимита — красный гейт, запуск не разрешён."""
    _write_estimate(tmp_path, "a4-skeleton", _estimate(usd_estimate=260.0, limit_usd=200.0))

    code, out, _ = _verify(tmp_path)

    assert code != 0
    assert "usd_estimate 260.0 > limit_usd 200.0" in out
    assert "превышает лимит" in out


def test_usd_equal_to_limit_passes(tmp_path: Path) -> None:
    """Ровно лимит — ещё допустимо: блокирует превышение, а не равенство."""
    _write_estimate(tmp_path, "a4-skeleton", _estimate(usd_estimate=200.0, limit_usd=200.0))

    code, out, _ = _verify(tmp_path)

    assert code == 0
    assert "OK" in out


# --- (iii) неполная смета ---------------------------------------------------


def test_missing_stop_rule_fails(tmp_path: Path) -> None:
    """(iii) Смета без стоп-правила не выполняет AD-8."""
    _write_estimate(tmp_path, "a4-skeleton", _remove(_estimate(), "stop_rule"))

    code, out, _ = _verify(tmp_path)

    assert code != 0
    assert "stop_rule: поле отсутствует или пусто" in out


@pytest.mark.parametrize(
    "field",
    ["run_ref", "gpu_type", "budget_method", "approved_by", "created_at"],
)
def test_each_required_text_field_is_enforced(tmp_path: Path, field: str) -> None:
    """Обязательные поля сметы: отсутствие любого — красный гейт."""
    _write_estimate(tmp_path, "a4-skeleton", _remove(_estimate(), field))

    code, out, _ = _verify(tmp_path)

    assert code != 0
    assert f"{field}: поле отсутствует или пусто" in out


@pytest.mark.parametrize("field", ["gpu_hours_estimate", "usd_estimate", "limit_usd"])
def test_missing_numeric_field_is_enforced(tmp_path: Path, field: str) -> None:
    """Числовые поля сметы обязательны и должны быть числами."""
    _write_estimate(tmp_path, "a4-skeleton", _remove(_estimate(), field))

    code, out, _ = _verify(tmp_path)

    assert code != 0
    assert f"{field}: не число" in out


def test_non_finite_numbers_fail(tmp_path: Path) -> None:
    """NaN и inf не проходят ни одно сравнение — гейт обязан их отвергнуть.

    NaN > limit_usd ложно, поэтому без явной проверки NaN-смета прошла бы
    как «валидная» — ложный PASS стража стоимости.
    """
    _write_estimate(tmp_path, "a4-skeleton", _estimate(usd_estimate=float("nan")))

    code, out, _ = _verify(tmp_path)

    assert code != 0
    assert "usd_estimate: не конечное число" in out


def test_infinite_limit_fails(tmp_path: Path) -> None:
    """inf в лимите делает проверку превышения бессмысленной."""
    _write_estimate(tmp_path, "a4-skeleton", _estimate(limit_usd=float("inf")))

    code, out, _ = _verify(tmp_path)

    assert code != 0
    assert "limit_usd: не конечное число" in out


def test_generate_with_non_finite_usd_writes_nothing(tmp_path: Path) -> None:
    """NaN на входе генератора — не данные прогона, записи нет."""
    code = _run(
        tmp_path,
        "--estimate",
        "a4-skeleton",
        "--gpu-hours",
        "97",
        "--usd",
        "nan",
        "--limit-usd",
        "200",
        "--approved-by",
        "владелец",
    )

    assert code != 0
    assert not (tmp_path / gate.BUDGET_DIR).exists()


def test_non_positive_gpu_hours_fails(tmp_path: Path) -> None:
    """Ноль GPU-часов — не оценка прогона."""
    _write_estimate(tmp_path, "a4-skeleton", _estimate(gpu_hours_estimate=0))

    code, out, _ = _verify(tmp_path)

    assert code != 0
    assert "gpu_hours_estimate: оценка часов должна быть > 0" in out


def test_budget_method_without_calibration_fails(tmp_path: Path) -> None:
    """Метод без калибровки AD-8 (arXiv 2412.19437, 343 TFLOP/s) не принимается."""
    _write_estimate(
        tmp_path,
        "a4-skeleton",
        _estimate(budget_method="прикидка на глаз"),
    )

    code, out, _ = _verify(tmp_path)

    assert code != 0
    assert "budget_method: не ссылается на калибровку AD-8" in out


def test_estimate_of_another_run_fails(tmp_path: Path) -> None:
    """Смета другого прогона не закрывает этот (иначе — чужой лимит)."""
    _write_estimate(tmp_path, "a4-skeleton", _estimate(run_ref="a5-rerun"))

    code, out, _ = _verify(tmp_path)

    assert code != 0
    assert "смета для прогона 'a5-rerun', а требуется 'a4-skeleton'" in out


def test_unsafe_run_ref_is_rejected(tmp_path: Path) -> None:
    """--run-ref — имя файла внутри каталога смет, а не путь наружу."""
    code, out, _ = _verify(tmp_path, run_ref="../../etc/passwd")

    assert code != 0
    assert "недопустимая ссылка на прогон" in out


def test_broken_json_fails(tmp_path: Path) -> None:
    """Нечитаемая смета — не смета."""
    path = tmp_path / gate.BUDGET_DIR / "a4-skeleton.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ это не json", encoding="utf-8")

    code, out, _ = _verify(tmp_path)

    assert code != 0
    assert "смета не читается" in out


# --- (iv) датировка ---------------------------------------------------------


def test_created_after_run_date_fails(tmp_path: Path) -> None:
    """(iv) Смета, написанная после прогона, — перерасход post factum."""
    _write_estimate(
        tmp_path,
        "a4-skeleton",
        _estimate(created_at="2026-09-14T09:00:00+00:00"),
    )
    _write_manifest(tmp_path, "a4-skeleton", "2026-09-13")

    code, out, _ = _verify(tmp_path)

    assert code != 0
    assert "не раньше даты прогона 2026-09-13" in out
    assert "смета не предшествует запуску (AD-8)" in out


def test_created_same_day_as_run_fails(tmp_path: Path) -> None:
    """«Раньше» — строго раньше: та же дата прогона доказательством не является."""
    _write_estimate(
        tmp_path,
        "a4-skeleton",
        _estimate(created_at="2026-09-13T08:00:00+00:00"),
    )
    _write_manifest(tmp_path, "a4-skeleton", "2026-09-13")

    code, out, _ = _verify(tmp_path)

    assert code != 0
    assert "не раньше даты прогона" in out


def test_date_only_created_at_is_parsed(tmp_path: Path) -> None:
    """Дата без времени тоже принимается (ISO-8601 date)."""
    _write_estimate(tmp_path, "a4-skeleton", _estimate(created_at="2026-09-12"))
    _write_manifest(tmp_path, "a4-skeleton", "2026-09-13")

    code, out, _ = _verify(tmp_path)

    assert code == 0
    assert "смета 2026-09-12 раньше прогона 2026-09-13" in out


# --- (v) корректная смета ---------------------------------------------------


def test_valid_estimate_passes(tmp_path: Path) -> None:
    """(v) Состоятельная смета до прогона — зелёный гейт."""
    _write_estimate(tmp_path, "a4-skeleton", _estimate())
    _write_manifest(tmp_path, "a4-skeleton", "2026-09-13")

    code, out, _ = _verify(tmp_path)

    assert code == 0
    assert "[a4-skeleton] OK: usd 180 ≤ лимит 200" in out
    assert "Итог: PASS" in out


def test_estimate_without_manifest_passes(tmp_path: Path) -> None:
    """Манифеста A4 ещё нет — смета всё равно проверяется по полям (AD-8)."""
    _write_estimate(tmp_path, "a4-skeleton", _estimate())

    code, out, _ = _verify(tmp_path)

    assert code == 0


def test_manifest_of_undeclared_run_is_required(tmp_path: Path) -> None:
    """Найденный манифест добавляет прогон в реестр: без сметы — блокировка."""
    _write_estimate(tmp_path, "a4-skeleton", _estimate())
    _write_estimate(tmp_path, "a5-rerun", _estimate(run_ref="a5-rerun"))
    _write_manifest(tmp_path, "pretrain-l1", "2026-10-01")

    code, out, _ = _verify_all(tmp_path)

    assert code != 0
    assert "[pretrain-l1] FAIL" in out
    assert "смета отсутствует: запуск блокирован" in out


# --- (vi) генератор ---------------------------------------------------------


def test_generate_without_inputs_writes_nothing(tmp_path: Path) -> None:
    """(vi) Без входных данных генератор не создаёт ни файла, ни каталога."""
    code = _run(tmp_path, "--estimate", "a4-skeleton")

    assert code != 0
    assert not (tmp_path / gate.BUDGET_DIR).exists()
    assert not list(tmp_path.rglob("*.json"))


def test_generate_without_usd_source_writes_nothing(tmp_path: Path) -> None:
    """Нет оценки стоимости (--usd / --usd-per-gpu-hour) — записи нет."""
    code = _run(
        tmp_path,
        "--estimate",
        "a4-skeleton",
        "--gpu-hours",
        "97",
        "--limit-usd",
        "200",
        "--approved-by",
        "владелец",
    )

    assert code != 0
    assert not (tmp_path / gate.BUDGET_DIR / "a4-skeleton.json").exists()


def test_generate_with_non_positive_hours_writes_nothing(tmp_path: Path) -> None:
    """Отрицательные часы — не данные прогона."""
    code = _run(
        tmp_path,
        "--estimate",
        "a4-skeleton",
        "--gpu-hours",
        "-5",
        "--usd",
        "10",
        "--limit-usd",
        "200",
        "--approved-by",
        "владелец",
    )

    assert code != 0
    assert not (tmp_path / gate.BUDGET_DIR).exists()


def test_generate_rejects_unsafe_run_ref(tmp_path: Path) -> None:
    """run-ref — имя файла: выход из каталога смет запрещён."""
    code = _run(
        tmp_path,
        "--estimate",
        "../escape",
        "--gpu-hours",
        "97",
        "--usd",
        "200",
        "--limit-usd",
        "200",
        "--approved-by",
        "владелец",
    )

    assert code != 0
    assert not (tmp_path / "escape.json").exists()
    assert not (tmp_path / "evidence").exists()


def test_generate_happy_path_round_trip(tmp_path: Path) -> None:
    """Сгенерированная смета проходит стража (создана до прогона)."""
    code = _run(
        tmp_path,
        "--estimate",
        "a4-skeleton",
        "--gpu-hours",
        "97",
        "--usd-per-gpu-hour",
        "2.0619",
        "--limit-usd",
        "200",
        "--approved-by",
        "владелец",
        "--created-at",
        "2026-09-12T10:00:00+00:00",
    )

    assert code == 0
    artifact = tmp_path / gate.BUDGET_DIR / "a4-skeleton.json"
    data = json.loads(artifact.read_text(encoding="utf-8"))
    assert data["usd_estimate"] == pytest.approx(200.0, abs=0.01)
    assert data["gpu_type"] == "H800"
    assert data["stop_rule"] == gate.STOP_RULE_DEFAULT

    _write_manifest(tmp_path, "a4-skeleton", "2026-09-13")
    verify_code, out, _ = _verify(tmp_path)
    assert verify_code == 0, out
    # частичных файлов не осталось
    assert not list((tmp_path / gate.BUDGET_DIR).glob("*.tmp"))


def test_generate_warns_when_estimate_is_late(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Генератор предупреждает, если дата сметы не предшествует известному прогону."""
    _write_manifest(tmp_path, "a4-skeleton", "2026-09-13")

    code = _run(
        tmp_path,
        "--estimate",
        "a4-skeleton",
        "--gpu-hours",
        "97",
        "--usd",
        "200",
        "--limit-usd",
        "200",
        "--approved-by",
        "владелец",
        "--created-at",
        "2026-09-14T10:00:00+00:00",
    )

    assert code == 0
    assert (tmp_path / gate.BUDGET_DIR / "a4-skeleton.json").exists()
    assert "не раньше" in capsys.readouterr().err
    # страж ту же смету не пропустит — предупреждение не декоративно
    verify_code, out, _ = _verify(tmp_path)
    assert verify_code != 0
    assert "не раньше даты прогона" in out


def test_generate_refuses_overwrite_without_force(tmp_path: Path) -> None:
    """Утверждённая смета не перезаписывается молча."""
    original = _estimate(usd_estimate=180.0)
    path = _write_estimate(tmp_path, "a4-skeleton", original)

    code = _run(
        tmp_path,
        "--estimate",
        "a4-skeleton",
        "--gpu-hours",
        "97",
        "--usd",
        "199",
        "--limit-usd",
        "200",
        "--approved-by",
        "владелец",
    )

    assert code != 0
    assert json.loads(path.read_text(encoding="utf-8")) == original


def test_generate_force_replaces_estimate(tmp_path: Path) -> None:
    """--force — осознанная замена (например, пересмотр сметы)."""
    _write_estimate(tmp_path, "a4-skeleton", _estimate(usd_estimate=180.0))

    code = _run(
        tmp_path,
        "--estimate",
        "a4-skeleton",
        "--gpu-hours",
        "97",
        "--usd",
        "199",
        "--limit-usd",
        "200",
        "--approved-by",
        "владелец",
        "--force",
    )

    assert code == 0
    data = json.loads(
        (tmp_path / gate.BUDGET_DIR / "a4-skeleton.json").read_text(encoding="utf-8")
    )
    assert data["usd_estimate"] == 199.0


# --- рабочий набор кейса ----------------------------------------------------


def test_case_has_no_fabricated_estimates() -> None:
    """В кейсе нет фиктивных смет: гейт красный по факту, а не по недосмотру.

    Это регрессия против соблазна «позеленить» C-041 подсунутым артефактом:
    пока реальные прогоны не объявлены и не оценены, смет в evidence/budget/
    быть не должно.
    """
    case_dir = Path(__file__).resolve().parents[2]
    budget_dir = case_dir / gate.BUDGET_DIR

    fabricated = sorted(path.name for path in budget_dir.glob("*.json")) if budget_dir.is_dir() else []
    assert fabricated == [], f"в кейсе появились сметы без прогонов: {fabricated}"


def test_case_gate_is_red_until_estimates_exist() -> None:
    """Ожидаемое состояние гейта кейса: C-041 красный (смет прогонов нет)."""
    case_dir = Path(__file__).resolve().parents[2]

    code, out, _ = _verify_all(case_dir)

    assert code != 0
    assert "смета отсутствует: запуск блокирован" in out
