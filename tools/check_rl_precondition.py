#!/usr/bin/env python3
"""C-056 — предусловие RL-прогона: калибровка K3 с полным пиннингом (ADR-058).

RL-прогон «против K3» имеет смысл только после калибровки среды на baseline
K3.  Калибровочный отчёт ``calibration-*.json`` годится, если:

* модель — не заглушка (``model``/``model_name`` задано и не ``stub``);
* ``pass_rate`` (из ``aggregate.pass_rate`` или верхнего уровня) лежит в
  ``[0.10, 0.90]`` — среда не слишком лёгкая и не слишком тяжёлая для
  сравнения;
* ``ready`` не ``false`` (readiness отчёта);
* пиннинг непуст — отчёт несёт пины версий (AD-4).

Годный отчёт обязан быть **связан** из precondition-файла
(``calibration_report`` — путь отчёта): сканирование каталога без явной
ссылки предусловием не считается.  Пока K3-калибровки нет, страж обязан
быть красным — это known-red, а не повод ослабить проверку::

    K3-калибровка не найдена: known-red до прогона на GB10 (ADR-058)

Код 0 — предусловие выполнено, 1 — нет (known-red).  ``--selftest`` —
фикстуры во временном каталоге; сеть и jax не нужны.

Запуск (из каталога кейса)::

    python3 tools/check_rl_precondition.py
    python3 tools/check_rl_precondition.py --selftest
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_EVIDENCE_DIR = "evidence"
DEFAULT_PRECONDITION = "evidence/rl-precondition.json"

#: Классифицированный known-red (ADR-058): отсутствие K3-калибровки.
KNOWN_RED = "K3-калибровка не найдена: known-red до прогона на GB10 (ADR-058)"

#: Допустимый pass_rate среды для сравнения с K3 (ADR-058).
PASS_RATE_RANGE = (0.10, 0.90)

EXIT_PASS = 0
EXIT_FAIL = 1


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def pass_rate(report: dict) -> Optional[float]:
    aggregate = report.get("aggregate")
    if isinstance(aggregate, dict) and _is_number(aggregate.get("pass_rate")):
        return float(aggregate["pass_rate"])
    if _is_number(report.get("pass_rate")):
        return float(report["pass_rate"])
    return None


def ready(report: dict) -> Optional[bool]:
    """``ready`` отчёта (верхний уровень или ``readiness.ready``)."""
    readiness = report.get("readiness")
    if isinstance(readiness, dict) and isinstance(readiness.get("ready"), bool):
        return readiness["ready"]
    top = report.get("ready")
    return top if isinstance(top, bool) else None


def pinning_complete(report: dict) -> bool:
    """Пиннинг непуст: словарь с хотя бы одним непустым значением (AD-4)."""
    pinning = report.get("pinning")
    if not isinstance(pinning, dict) or not pinning:
        return False
    return all(v not in (None, "", [], {}) for v in pinning.values())


def report_problem(report: Any) -> Optional[str]:
    """Причина непригодности отчёта (None — годен)."""
    if not isinstance(report, dict):
        return "не JSON-объект"
    model = report.get("model", report.get("model_name"))
    if not isinstance(model, str) or not model.strip():
        return "не указана модель"
    if model.strip().lower() == "stub":
        return "модель-заглушка (stub)"
    rate = pass_rate(report)
    if rate is None:
        return "нет pass_rate (aggregate.pass_rate)"
    low, high = PASS_RATE_RANGE
    if not (low <= rate <= high):
        return f"pass_rate={rate:.3f} вне [{low:.2f}, {high:.2f}]"
    if ready(report) is False:
        return "ready=false"
    if not pinning_complete(report):
        return "пиннинг пуст (AD-4)"
    return None


def _resolve_ref(ref: str, precondition: Path, evidence_dir: Path) -> Optional[Path]:
    """Путь отчёта из ссылки: абсолютный, от evidence/, от precondition, от корня."""
    path = Path(ref)
    if path.is_absolute():
        return path if path.is_file() else None
    candidates = (evidence_dir / path, precondition.parent / path, _REPO_ROOT / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def evaluate(evidence_dir: Path, precondition: Path) -> tuple[bool, list[str]]:
    """Проверка предусловия: (выполнено?, диагностические строки)."""
    notes: list[str] = []
    reports = sorted(evidence_dir.rglob("calibration-*.json")) if evidence_dir.is_dir() else []
    notes.append(f"calibration-*.json в {evidence_dir}: найдено {len(reports)}")

    if not precondition.is_file():
        notes.append(f"precondition-файл не найден: {precondition}")
        return False, notes
    try:
        precondition_doc = json.loads(precondition.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        notes.append(f"precondition не читается как JSON: {exc}")
        return False, notes
    ref = precondition_doc.get("calibration_report") if isinstance(precondition_doc, dict) else None
    if not isinstance(ref, str) or not ref.strip():
        notes.append(
            "precondition не ссылается на отчёт (поле calibration_report): "
            "сканирование без ссылки предусловием не является"
        )
        return False, notes

    target = _resolve_ref(ref, precondition, evidence_dir)
    if target is None:
        notes.append(f"отчёт по ссылке не найден: {ref}")
        return False, notes
    try:
        report = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        notes.append(f"отчёт не читается как JSON: {target}: {exc}")
        return False, notes

    problem = report_problem(report)
    if problem:
        notes.append(f"отчёт {target}: {problem}")
        return False, notes
    notes.append(
        f"K3-калибровка: {target} (pass_rate={pass_rate(report):.3f}) — "
        "предусловие выполнено"
    )
    return True, notes


# ---------------------------------------------------------------------------
# selftest: фикстуры во временном каталоге, репозиторий не затрагивается
# ---------------------------------------------------------------------------


def _write_case(
    root: Path,
    *,
    model: str = "kimi-k3",
    rate: float = 0.5,
    is_ready: bool = True,
    pinning: bool = True,
    link: bool = True,
) -> tuple[Path, Path]:
    evidence = root / "evidence"
    evidence.mkdir(parents=True)
    report = {
        "calibration_id": "calib-selftest",
        "model": model,
        "model_seed": 7,
        "env_version": "environment-v1",
        "pinning": (
            {
                "arch_ml_build": "selftest-build",
                "constraints_sha256": "a" * 64,
                "hidden_constraints_sha256": "b" * 64,
            }
            if pinning else {}
        ),
        "aggregate": {"pass_rate": rate},
        "readiness": {"ready": is_ready},
    }
    (evidence / "calibration-k3.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    precondition = evidence / "rl-precondition.json"
    precondition.write_text(
        json.dumps(
            {"calibration_report": "calibration-k3.json"} if link else {},
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    return evidence, precondition


def run_selftest() -> int:
    checks: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory(prefix="rl-precondition-selftest-") as tmp:
        base = Path(tmp)

        good_ev, good_pre = _write_case(base / "good")
        checks.append(("годная K3-калибровка со ссылкой — PASS", evaluate(good_ev, good_pre)[0]))

        stub_ev, stub_pre = _write_case(base / "stub", model="stub")
        checks.append(("модель-заглушка — FAIL", not evaluate(stub_ev, stub_pre)[0]))

        easy_ev, easy_pre = _write_case(base / "easy", rate=0.95)
        checks.append(("pass_rate 0.95 вне окна — FAIL", not evaluate(easy_ev, easy_pre)[0]))

        unready_ev, unready_pre = _write_case(base / "unready", is_ready=False)
        checks.append(("ready=false — FAIL", not evaluate(unready_ev, unready_pre)[0]))

        unpinned_ev, unpinned_pre = _write_case(base / "unpinned", pinning=False)
        checks.append(("пиннинг пуст — FAIL", not evaluate(unpinned_ev, unpinned_pre)[0]))

        unlinked_ev, unlinked_pre = _write_case(base / "unlinked", link=False)
        checks.append(("нет ссылки из precondition — FAIL", not evaluate(unlinked_ev, unlinked_pre)[0]))

    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"[rl-precondition:selftest] {'OK' if ok else 'FAIL'}: {name}")
    if failed:
        print(f"rl-precondition:selftest: {len(failed)} из {len(checks)} сценариев не совпали")
        return EXIT_FAIL
    print(f"rl-precondition:selftest: OK ({len(checks)} сценариев)")
    return EXIT_PASS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="C-056: предусловие RL-прогона — K3-калибровка с пиннингом (ADR-058)",
    )
    parser.add_argument(
        "--evidence-dir", type=Path, default=None,
        help=f"каталог evidence (по умолчанию {DEFAULT_EVIDENCE_DIR} от корня кейса)",
    )
    parser.add_argument(
        "--precondition", type=Path, default=None,
        help=f"precondition-файл с ссылкой calibration_report (по умолчанию {DEFAULT_PRECONDITION})",
    )
    parser.add_argument(
        "--selftest", action="store_true",
        help="прогнать фикстуры во временном каталоге",
    )
    args = parser.parse_args(argv)

    if args.selftest:
        return run_selftest()

    def anchor(path: Path) -> Path:
        return path if path.is_absolute() else (_REPO_ROOT / path)

    evidence_dir = anchor(args.evidence_dir or Path(DEFAULT_EVIDENCE_DIR))
    precondition = anchor(args.precondition or Path(DEFAULT_PRECONDITION))

    ok, notes = evaluate(evidence_dir, precondition)
    for line in notes:
        print(f"[rl-precondition] {line}")
    if ok:
        print("rl-precondition: PASS — предусловие RL-прогона выполнено")
        return EXIT_PASS
    print(KNOWN_RED)
    return EXIT_FAIL


if __name__ == "__main__":
    raise SystemExit(main())
