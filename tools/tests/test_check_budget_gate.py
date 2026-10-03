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
  (iv)   смета, созданная ПОЗЖЕ ДНЯ прогона -> FAIL (дневная гранулярность:
         смета дня прогона законна — см. ниже), смета без `created_at` ->
         FAIL, как и любое отсутствующее обязательное поле;
  (v)    корректная смета (фикстура в tmp) -> PASS;
  (vi)   генерация без входных данных -> ненулевой exit, файл не создан.

Дневная гранулярность (C-041/AD-8). Порядок «смета до запуска» проверяется
по датам: сравнение идёт между `created_at` сметы и `run_date` манифеста,
обе стороны приводятся `parse_run_date` к дню. Смета, составленная в ДЕНЬ
прогона, нарушением не является (порядок внутри дня обеспечивается
процессом — смета коммитится до старта прогона), смета датой позже дня
прогона — является. Дальше этого гейт не ослаблен: `created_at` остаётся
обязательным полем (`REQUIRED_TEXT_FIELDS`).
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
    """Метод без ссылки на AD-8 (калибровка или формула 6·N·D) не принимается."""
    _write_estimate(
        tmp_path,
        "a4-skeleton",
        _estimate(budget_method="прикидка на глаз"),
    )

    code, out, _ = _verify(tmp_path)

    assert code != 0
    assert "budget_method: не ссылается на метод AD-8" in out


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


# --- (iv) датировка (дневная гранулярность) ---------------------------------


def test_created_after_run_date_fails(tmp_path: Path) -> None:
    """(iv) Смета, написанная после дня прогона, — перерасход post factum."""
    _write_estimate(
        tmp_path,
        "a4-skeleton",
        _estimate(created_at="2026-09-14T09:00:00+00:00"),
    )
    _write_manifest(tmp_path, "a4-skeleton", "2026-09-13")

    code, out, _ = _verify(tmp_path)

    assert code != 0
    assert "смета датирована ПОЗЖЕ дня прогона 2026-09-13" in out
    assert "смета не предшествует запуску (AD-8)" in out
    assert "порядок внутри дня обеспечивается процессом" in out


def test_t_d1_created_same_day_as_run_passes(tmp_path: Path) -> None:
    """T-d1: смета, составленная В ДЕНЬ прогона, — законна.

    Гранулярность проверки — день (C-041): порядок «смета до запуска» внутри
    дня обеспечивается процессом (смета коммитится до старта прогона), а не
    часовой меткой. Требовать строгого «раньше» значило бы блокировать
    прогон сметой того же дня — так и было до дельты.
    """
    _write_estimate(
        tmp_path,
        "a4-skeleton",
        _estimate(created_at="2026-09-13T08:00:00+00:00"),
    )
    _write_manifest(tmp_path, "a4-skeleton", "2026-09-13")

    code, out, _ = _verify(tmp_path)

    assert code == 0, out
    assert "[a4-skeleton] OK" in out
    assert "смета 2026-09-13 не позже дня прогона 2026-09-13" in out


def test_t_d2_created_next_day_fails(tmp_path: Path) -> None:
    """T-d2: смета датой на день ПОЗЖЕ прогона — FAIL с сообщением о том, что
    смета не предшествует запуску; ослабление ограничено днём прогона."""
    _write_estimate(
        tmp_path,
        "a4-skeleton",
        _estimate(created_at="2026-09-14T08:00:00+00:00"),
    )
    _write_manifest(tmp_path, "a4-skeleton", "2026-09-13")

    code, out, _ = _verify(tmp_path)

    assert code != 0
    assert "смета датирована ПОЗЖЕ дня прогона 2026-09-13" in out
    assert "порядок внутри дня обеспечивается процессом — смета коммитится " \
           "до старта прогона" in out
    assert "[a4-skeleton] FAIL" in out


def test_t_d3_missing_created_at_fails(tmp_path: Path) -> None:
    """T-d3: `created_at` остаётся ОБЯЗАТЕЛЬНЫМ полем (REQUIRED_TEXT_FIELDS):
    смета без даты не проверяема на «до запуска» и потому — FAIL, а не PASS
    «по умолчанию»."""
    assert "created_at" in gate.REQUIRED_TEXT_FIELDS, (
        "created_at обязан оставаться обязательным полем сметы (C-041)"
    )
    _write_estimate(tmp_path, "a4-skeleton", _remove(_estimate(), "created_at"))
    _write_manifest(tmp_path, "a4-skeleton", "2026-09-13")

    code, out, _ = _verify(tmp_path)

    assert code != 0
    assert "created_at: поле отсутствует или пусто" in out


def test_date_only_created_at_is_parsed(tmp_path: Path) -> None:
    """Дата без времени тоже принимается (ISO-8601 date)."""
    _write_estimate(tmp_path, "a4-skeleton", _estimate(created_at="2026-09-12"))
    _write_manifest(tmp_path, "a4-skeleton", "2026-09-13")

    code, out, _ = _verify(tmp_path)

    assert code == 0
    assert "смета 2026-09-12 не позже дня прогона 2026-09-13" in out


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


# --- T-registry: реестр объявленных прогонов --------------------------------


def test_t_registry_covers_stage_runs(tmp_path: Path) -> None:
    """T-registry: стадии, исполненные отдельными инструментами
    (sft-smoke/rl-smoke), объявлены в реестре прогонов и потому под стражем
    C-041 — смета требуется даже без манифеста A4 (AD-8: каждому прогону
    смета; реестр без стадий оставлял бы их лимиты непроверенными).
    """
    declared = set(gate.DECLARED_RUN_REFS)

    assert {"sft-smoke", "rl-smoke"} <= declared, (
        f"реестр объявленных прогонов не покрывает стадии: {sorted(declared)}"
    )

    # Манифестов стадий нет — прогон обязан попасть в реестр из объявления.
    refs = gate.required_runs(tmp_path)

    assert gate.required_runs(tmp_path, ["sft-smoke"]) == ["sft-smoke"]
    assert {"sft-smoke", "rl-smoke"} <= set(refs)
    assert refs == list(gate.DECLARED_RUN_REFS), (
        "пустой кейс: реестр — ровно объявленные прогоны, без дублей"
    )


def test_t_registry_stage_without_estimate_blocks_run(tmp_path: Path) -> None:
    """Стадия реестра без сметы блокирует запуск поимённо (не «прочие»)."""
    code, out, _ = _verify(tmp_path, run_ref="rl-smoke")

    assert code != 0
    assert "[rl-smoke] FAIL" in out
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
    """Генератор предупреждает, если смета датирована позже дня известного
    прогона (дневная гранулярность — та же, что у стража)."""
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
    assert "позже дня прогона" in capsys.readouterr().err
    # страж ту же смету не пропустит — предупреждение не декоративно
    verify_code, out, _ = _verify(tmp_path)
    assert verify_code != 0
    assert "смета датирована ПОЗЖЕ дня прогона" in out


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
#
# Сметы прогонов кейса — артефакты, а не заглушки: каждая проходит валидацию
# полей AD-8 (калибровка, стоп-правило, утвердивший, лимит) и названа своим
# прогоном (`run_ref` = имя файла). Прежняя форма регрессии — «смет в
# evidence/budget/ быть не должно, пока прогоны не объявлены и не оценены» —
# устарела вместе с объявлением прогонов и составлением смет (26.09.2026):
# гейт кейса теперь зелёный по факту, и охрана смещается с пустоты каталога на
# состоятельность артефактов. Красный гейт A4 по покрытию стадий
# (tools/a4_manifest.py --verify, pipeline_complete=false) это не затрагивает:
# C-041 — страж стоимости, другой гейт.


def test_case_estimates_are_not_stubs() -> None:
    """Ни одну смету кейса нельзя подсунуть вместо доказательства: каждая
    валидна по полям AD-8 и названа тем прогоном, чьим именем лежит."""
    case_dir = Path(__file__).resolve().parents[2]
    budget_dir = case_dir / gate.BUDGET_DIR

    artifacts = sorted(budget_dir.glob("*.json")) if budget_dir.is_dir() else []
    for path in artifacts:
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data.get("run_ref") == path.stem, (
            f"{path.name}: смета названа чужим прогоном "
            f"'{data.get('run_ref')}' (имя файла — имя сметы, C-041)"
        )
        errs = gate.validate_estimate(data, path.stem)
        assert errs == [], f"{path.name}: {errs}"


def test_case_estimates_cover_declared_runs() -> None:
    """Ни один прогон кейса не остался без сметы: объявленные прогоны
    (DECLARED_RUN_REFS) и прогоны найденных манифестов закрыты, иначе запуск
    блокирован по факту (AD-8)."""
    case_dir = Path(__file__).resolve().parents[2]
    budget_dir = case_dir / gate.BUDGET_DIR

    refs = gate.required_runs(case_dir)
    assert refs, "реестр прогонов кейса пуст — страху нечего проверять"
    missing = sorted(
        ref for ref in refs if not (budget_dir / f"{ref}.json").is_file()
    )
    assert missing == [], f"прогоны без сметы: {missing}"


def test_case_gate_passes_on_committed_estimates() -> None:
    """Состояние гейта кейса: C-041 зелёный по факту.

    Сметы объявленных прогонов существуют и не датированы позже дня прогона
    (дневная гранулярность: смета дня прогона законна — a4-skeleton
    created_at=2026-09-26 при run_date=2026-09-26); страж называет каждый
    прогон поимённо.
    """
    case_dir = Path(__file__).resolve().parents[2]

    code, out, _ = _verify_all(case_dir)

    assert code == 0, out
    assert "Итог: PASS" in out
    for run_ref in gate.DECLARED_RUN_REFS:
        assert f"[{run_ref}] OK" in out
