#!/usr/bin/env python3
"""C-056/C-057 — два предусловия RL-прогона (ADR-058, амендмент B).

Precondition §10 ``ENVIRONMENT-V1`` разделён на два обязательства с разной
ценой ошибки:

* ``--part env`` (C-056, «среда откалибрована»).  Годен отчёт
  ``calibration-*.json`` **своей** модели — не заглушки (``stub``) и не K3
  (``k3``/``kimi*``), — у которого ``pass_rate`` (``aggregate.pass_rate`` или
  верхний уровень) ∈ ``[0.10, 0.90]``, ``ready`` не ``false``, полный пиннинг
  (AD-4) и есть ссылка ``calibration_report`` из
  ``evidence/rl-precondition.json``.  Достижимо на GB10; блокирует RL-расход.

* ``--part k3`` (C-057, «K3-baseline»).  Годен отчёт ``calibration-*.json``
  модели K3 (``k3``/``kimi*``) с тем же коридором, ``ready``, пиннингом и
  ссылкой из ``evidence/k3-baseline.json``.  На GB10 невозможен (2.8T ≈
  1,4 ТБ FP4 против 128 ГБ unified); блокирует заявление о превосходстве,
  не обучение.

Сканирование каталога без явной ссылки предусловием не считается: отчёт обязан
быть связан из precondition-файла.  Отсутствие годного отчёта —
классифицированный known-red (:data:`KNOWN_RED_ENV` / :data:`KNOWN_RED_K3`),
а не повод ослабить проверку.

Код 0 — предусловие выполнено, 1 — нет (known-red).  ``--selftest`` прогоняет
фикстуры обеих частей во временном каталоге; сеть и jax не нужны.

Запуск (из каталога кейса)::

    python3 tools/check_rl_precondition.py --part env
    python3 tools/check_rl_precondition.py --part k3
    python3 tools/check_rl_precondition.py --selftest
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_EVIDENCE_DIR = "evidence"

PART_ENV = "env"
PART_K3 = "k3"

#: Классифицированные known-red (ADR-058, амендмент B): два разных долга.
KNOWN_RED_ENV = (
    "Среда не откалибрована: нужен отчёт не-заглушки с pass_rate ∈ [10%, 90%] "
    "на GB10 (ADR-058, C-056)"
)
KNOWN_RED_K3 = (
    "K3-baseline отсутствует: known-red P-7b. На GB10 невозможен "
    "(2.8T ≈ 1,4 ТБ FP4 против 128 ГБ); путь: C-048 (KPI 800 ток/с) → "
    "смета AD-8 → аренда INT-001 или API для публичного набора (ADR-058, C-057)"
)

#: Допустимый pass_rate среды для сравнения с K3 (ADR-058).
PASS_RATE_RANGE = (0.10, 0.90)

#: K3-модель по имени: ``k3`` или ``kimi*`` (без учёта регистра).
_K3_MODEL_RE = re.compile(r"k3|kimi", re.IGNORECASE)

EXIT_PASS = 0
EXIT_FAIL = 1


@dataclass(frozen=True)
class Part:
    """Описание обязательства: правило, precondition-файл, вердикты."""

    name: str
    constraint: str
    precondition: str
    label: str
    pass_message: str
    known_red: str


PARTS: dict[str, Part] = {
    PART_ENV: Part(
        name=PART_ENV,
        constraint="C-056",
        precondition="evidence/rl-precondition.json",
        label="калибровка среды",
        pass_message="среда откалибрована — RL-расход разрешён (C-056)",
        known_red=KNOWN_RED_ENV,
    ),
    PART_K3: Part(
        name=PART_K3,
        constraint="C-057",
        precondition="evidence/k3-baseline.json",
        label="K3-baseline",
        pass_message="K3-baseline сверен — заявление о превосходстве допустимо (C-057)",
        known_red=KNOWN_RED_K3,
    ),
}


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


def declared_model(report: dict) -> Optional[str]:
    """Непустое имя модели из ``model`` или ``model_name``."""
    for key in ("model", "model_name"):
        value = report.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def is_stub_model(model: str) -> bool:
    """Заглушка — ровно ``stub`` (без учёта регистра)."""
    return model.strip().lower() == "stub"


def is_k3_model(model: str) -> bool:
    """K3 по имени модели: ``k3`` или ``kimi*`` (без учёта регистра)."""
    return _K3_MODEL_RE.search(model) is not None


def report_problem(report: Any, part: str = PART_ENV) -> Optional[str]:
    """Причина непригодности отчёта для части (None — годен)."""
    if part not in PARTS:
        raise ValueError(f"неизвестная часть: {part!r}")
    if not isinstance(report, dict):
        return "не JSON-объект"
    model = declared_model(report)
    if model is None:
        return "не указана модель"
    if part == PART_ENV:
        if is_stub_model(model):
            return "модель-заглушка (stub)"
        if is_k3_model(model):
            return f"K3-отчёт не калибрует среду: модель {model!r} (нужна своя, не k3/kimi)"
    elif not is_k3_model(model):
        return f"модель не K3 (k3|kimi): {model!r}"
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


def evaluate(
    evidence_dir: Path, precondition: Path, part: str = PART_ENV
) -> tuple[bool, list[str]]:
    """Проверка предусловия части: (выполнено?, диагностические строки)."""
    if part not in PARTS:
        raise ValueError(f"неизвестная часть: {part!r}")
    spec = PARTS[part]
    notes: list[str] = []
    reports = sorted(evidence_dir.rglob("calibration-*.json")) if evidence_dir.is_dir() else []
    notes.append(f"калибровочные отчёты calibration-*.json в {evidence_dir}: {len(reports)}")

    if not precondition.is_file():
        notes.append(f"precondition-файл не найден: {precondition}")
        return False, notes
    try:
        precondition_doc = json.loads(precondition.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        notes.append(f"precondition не читается как JSON: {exc}")
        return False, notes
    ref = (
        precondition_doc.get("calibration_report")
        if isinstance(precondition_doc, dict)
        else None
    )
    if not isinstance(ref, str) or not ref.strip():
        notes.append(
            f"precondition не ссылается на отчёт (поле calibration_report): "
            f"сканирование без ссылки предусловием не является ({spec.constraint})"
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

    problem = report_problem(report, part)
    if problem:
        notes.append(f"отчёт {target}: {problem}")
        return False, notes
    notes.append(
        f"{spec.label}: {target} (model={declared_model(report)}, "
        f"pass_rate={pass_rate(report):.3f}) — предусловие {spec.constraint} выполнено"
    )
    return True, notes


# ---------------------------------------------------------------------------
# selftest: фикстуры во временном каталоге, репозиторий не затрагивается
# ---------------------------------------------------------------------------


def _write_case(
    root: Path,
    *,
    part: str = PART_ENV,
    model: Optional[str] = None,
    rate: float = 0.5,
    is_ready: bool = True,
    pinning: bool = True,
    link: bool = True,
) -> tuple[Path, Path]:
    if part not in PARTS:
        raise ValueError(f"неизвестная часть: {part!r}")
    if model is None:
        model = "kimi-k3" if part == PART_K3 else "own-checkpoint-7b"
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
    report_name = f"calibration-{part}.json"
    (evidence / report_name).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    precondition = evidence / Path(PARTS[part].precondition).name
    precondition.write_text(
        json.dumps(
            {"calibration_report": report_name} if link else {},
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    return evidence, precondition


def _selftest_env(base: Path) -> list[tuple[str, bool]]:
    checks: list[tuple[str, bool]] = []

    ev, pre = _write_case(base / "good")
    checks.append((
        "env: валидный отчёт своей модели со ссылкой — PASS",
        evaluate(ev, pre, PART_ENV)[0],
    ))

    ev, pre = _write_case(base / "stub", model="stub")
    checks.append(("env: модель-заглушка — FAIL", not evaluate(ev, pre, PART_ENV)[0]))

    ev, pre = _write_case(base / "k3-report", model="kimi-k3")
    checks.append(("env: K3-отчёт не калибрует среду — FAIL", not evaluate(ev, pre, PART_ENV)[0]))

    ev, pre = _write_case(base / "unlinked", link=False)
    checks.append(("env: нет ссылки из rl-precondition.json — FAIL", not evaluate(ev, pre, PART_ENV)[0]))

    ev, pre = _write_case(base / "easy", rate=0.95)
    checks.append(("env: pass_rate 0.95 вне коридора — FAIL", not evaluate(ev, pre, PART_ENV)[0]))

    ev, pre = _write_case(base / "unready", is_ready=False)
    checks.append(("env: ready=false — FAIL", not evaluate(ev, pre, PART_ENV)[0]))

    ev, pre = _write_case(base / "unpinned", pinning=False)
    checks.append(("env: пиннинг пуст — FAIL", not evaluate(ev, pre, PART_ENV)[0]))

    return checks


def _selftest_k3(base: Path) -> list[tuple[str, bool]]:
    checks: list[tuple[str, bool]] = []

    ev, pre = _write_case(base / "good", part=PART_K3)
    checks.append((
        "k3: валидный K3-baseline со ссылкой — PASS",
        evaluate(ev, pre, PART_K3)[0],
    ))

    ev, pre = _write_case(base / "not-k3", part=PART_K3, model="own-checkpoint-7b")
    checks.append(("k3: отчёт без K3 — FAIL", not evaluate(ev, pre, PART_K3)[0]))

    ev, pre = _write_case(base / "outside", part=PART_K3, rate=0.05)
    checks.append(("k3: pass_rate 0.05 вне коридора — FAIL", not evaluate(ev, pre, PART_K3)[0]))

    ev, pre = _write_case(base / "unlinked", part=PART_K3, link=False)
    checks.append(("k3: нет ссылки из k3-baseline.json — FAIL", not evaluate(ev, pre, PART_K3)[0]))

    return checks


def run_selftest() -> int:
    """Фикстуры обеих частей (env + k3); код 0 только если все ожидания совпали."""
    with tempfile.TemporaryDirectory(prefix="rl-precondition-selftest-") as tmp:
        base = Path(tmp)
        checks = _selftest_env(base / "env") + _selftest_k3(base / "k3")
    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"[rl-precondition:selftest] {'OK' if ok else 'FAIL'}: {name}")
    if failed:
        print(f"rl-precondition:selftest: {len(failed)} из {len(checks)} сценариев не совпали")
        return EXIT_FAIL
    print(f"rl-precondition:selftest: OK ({len(checks)} сценариев, части env + k3)")
    return EXIT_PASS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "C-056/C-057: предусловия RL-прогона — калибровка среды (env) "
            "и K3-baseline (k3), ADR-058"
        ),
    )
    parser.add_argument(
        "--part", choices=tuple(PARTS), default=PART_ENV,
        help="env — C-056 «среда откалибрована» (по умолчанию); k3 — C-057 «K3-baseline»",
    )
    parser.add_argument(
        "--evidence-dir", type=Path, default=None,
        help=f"каталог evidence (по умолчанию {DEFAULT_EVIDENCE_DIR} от корня кейса)",
    )
    parser.add_argument(
        "--precondition", type=Path, default=None,
        help=(
            "precondition-файл с ссылкой calibration_report "
            f"(по умолчанию {PARTS[PART_ENV].precondition} для env, "
            f"{PARTS[PART_K3].precondition} для k3)"
        ),
    )
    parser.add_argument(
        "--selftest", action="store_true",
        help="прогнать фикстуры обеих частей во временном каталоге",
    )
    args = parser.parse_args(argv)

    if args.selftest:
        return run_selftest()

    spec = PARTS[args.part]

    def anchor(path: Path) -> Path:
        return path if path.is_absolute() else (_REPO_ROOT / path)

    evidence_dir = anchor(args.evidence_dir or Path(DEFAULT_EVIDENCE_DIR))
    precondition = anchor(args.precondition or Path(spec.precondition))

    ok, notes = evaluate(evidence_dir, precondition, args.part)
    for line in notes:
        print(f"[rl-precondition:{args.part}] {line}")
    if ok:
        print(f"rl-precondition: PASS — {spec.pass_message}")
        return EXIT_PASS
    print(spec.known_red)
    return EXIT_FAIL


if __name__ == "__main__":
    raise SystemExit(main())
