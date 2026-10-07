#!/usr/bin/env python3
"""C-048 — страж эффективности с блокировкой сметы аренды.

Инвариант (ADR-032 §4 + ADR-034 §5 + пин ``blocks_rental`` в
``evidence/kpi-pins.json``): **смета аренды GPU не может быть одобрена, пока
KPI эффективности (ток/с) не добит до пинового порога.** Неэффективный
алгоритм на аренде = дни аренды на годы обучения (ADR-034, двухсценарная
математика); числовой порог — единственный носитель evidence/kpi-pins.json.

Логика:
  1. Читаются пины; для каждого с ``blocks_rental: true`` определяется
     состояние KPI:
       - ``metrics_file`` не задан в пине → ``unmet`` (fail-closed: нет
         доказательства эффективности — блокировка действует);
       - файл метрик отсутствует/пуст/нечитаем → ``unmet``;
       - медиана ``tok_s`` по последнему окну ≥ MIN_WINDOW шагов ≥ threshold
         → ``met``, иначе ``unmet`` (вычисление — носитель
         tools/check_performance_roofline.py, ``median``/``_iter_tok_s``).
  2. Сканируются ``rental_budget_paths`` пина (glob-ы от корня кейса):
       - файл со ``status: "draft"`` → допустимо (смета-оценка для планирования);
       - файл со ``status: "approved"`` / без статуса / нечитаемый → нарушение,
         если KPI ``unmet`` (блокировка); при KPI ``met`` — допустимо.
  3. Вердикт: нарушения есть → FAIL (exit 1); нет → PASS (exit 0); пины
     нечитаемы → fail-closed FAIL (exit 2).

Selftest (``--selftest``, канон C-043/44/45): синтетический кейс, шесть
мутантов — зелёный базовый, одобрение при unmet, draft при unmet, файл без
статуса при unmet, одобрение при met, нечитаемые пины.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

CASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PINS = CASE_DIR / "evidence" / "kpi-pins.json"
REPO_ROOT = CASE_DIR

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_performance_roofline as _roofline  # noqa: E402  (носитель вычисления KPI)

EXIT_OK = 0
EXIT_VIOLATION = 1
EXIT_FAILCLOSED = 2

_MEDIAN_WINDOW = max(_roofline.MIN_WINDOW, 20)


def _read_pins(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    pins = raw.get("pins") if isinstance(raw, dict) else raw
    if not isinstance(pins, list) or not pins:
        raise ValueError("пины пусты или не список")
    return pins


def _kpi_state(pin: dict[str, Any], case: Path) -> tuple[str, str]:
    """('met'|'unmet', пояснение) — fail-closed: любое сомнение = unmet."""
    threshold = float(pin["threshold"])
    mfile = pin.get("metrics_file")
    if not mfile:
        return "unmet", "metrics_file не задан в пине — доказательства эффективности нет"
    mp = (case / mfile) if not Path(mfile).is_absolute() else Path(mfile)
    if not mp.is_file():
        return "unmet", f"файл метрик отсутствует: {mp.as_posix()}"
    try:
        values = _roofline._iter_tok_s(mp)
    except Exception as exc:  # noqa: BLE001 — нечитаемый вход обязан красить
        return "unmet", f"метрики нечитаемы: {exc}"
    if len(values) < _MEDIAN_WINDOW:
        return "unmet", f"метрик мало: {len(values)} < {_MEDIAN_WINDOW} шагов окна"
    fact = _roofline.median(values[-_MEDIAN_WINDOW:])
    if fact >= threshold:
        return "met", f"медиана {fact:.1f} ≥ порога {threshold:.0f} (окно {_MEDIAN_WINDOW})"
    ratio = threshold / fact if fact > 0 else float("inf")
    return "unmet", f"медиана {fact:.1f} < порога {threshold:.0f} (недобор ×{ratio:.1f})"


def _file_status(path: Path) -> str:
    """'draft' | 'approved' | 'unknown' — unknown трактуется как одобрено (fail-closed)."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return "unknown"
    try:
        obj = json.loads(text)
        status = obj.get("status") if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        m = re.search(r'status:\s*"?(\w+)"?', text)
        status = m.group(1) if m else None
    if status in ("draft", "approved"):
        return str(status)
    return "unknown"


def evaluate(case: Path | str = CASE_DIR, pins_path: Path | str | None = None) -> tuple[int, dict[str, Any]]:
    case = Path(case)
    pp = Path(pins_path) if pins_path else case / "evidence" / "kpi-pins.json"
    report: dict[str, Any] = {"guard": "C-048 rental-block", "pins_file": pp.as_posix(),
                              "checked": 0, "blocked_pins": 0, "violations": [], "passed": False}
    try:
        pins = _read_pins(pp)
    except Exception as exc:  # noqa: BLE001
        report["violations"].append({"class": "fail-closed", "pin": None,
                                     "message": f"пины нечитаемы ({pp.as_posix()}): {exc}"})
        report["summary"] = "FAIL: пины нечитаемы — блокировка аренды не верифицируема"
        return EXIT_FAILCLOSED, report

    blocking = [p for p in pins if p.get("blocks_rental") is True]
    report["checked"] = len(blocking)
    if not blocking:
        report["passed"] = True
        report["summary"] = "PASS: пинов с blocks_rental нет — инвариант тривиален"
        return EXIT_OK, report

    violations: list[dict[str, Any]] = []
    for pin in blocking:
        run = str(pin.get("run", "?"))
        state, detail = _kpi_state(pin, case)
        if state == "met":
            report["violations"].append if False else None
            continue
        report["blocked_pins"] += 1
        for pattern in pin.get("rental_budget_paths", []) or []:
            matches = sorted(case.glob(pattern)) if not Path(pattern).is_absolute() else sorted(Path("").glob(pattern))
            if not matches:
                continue
            for f in matches:
                status = _file_status(f)
                if status == "draft":
                    continue
                violations.append({
                    "class": "rental-blocked",
                    "pin": run,
                    "message": (f"смета аренды одобрена при невыполненном KPI: {f.as_posix()} "
                                f"(status={status}); {detail}; блокировка ADR-034 §5: "
                                f"сначала KPI ≥ {float(pin['threshold']):.0f} ток/с (пин), затем аренда"),
                })
    report["violations"] = violations
    if violations:
        report["passed"] = False
        report["summary"] = (f"FAIL: аренда заблокирована — нарушений: {len(violations)}; "
                             f"заблокированных пинов: {report['blocked_pins']}")
        return EXIT_VIOLATION, report
    report["passed"] = True
    report["summary"] = (f"PASS: блокировка эффективна — пинов с blocks_rental: {len(blocking)}, "
                         f"из них KPI unmet: {report['blocked_pins']} (одобренных смет при них нет)")
    return EXIT_OK, report


def _selftest() -> int:
    """Шесть мутантов на синтетическом кейсе (канон C-043/44/45)."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        case = Path(td)
        (case / "evidence").mkdir()
        pins = case / "evidence" / "kpi-pins.json"
        pins.write_text(json.dumps({
            "version": 1,
            "pins": [{"run": "selftest", "kpi_tok_s_baseline": 86, "threshold": 800,
                      "blocks_rental": True, "rental_budget_paths": ["evidence/budget/rental-*.json"]}],
        }), encoding="utf-8")
        budget = case / "evidence" / "budget"
        budget.mkdir()

        def _run() -> tuple[int, str]:
            code, rep = evaluate(case)
            return code, rep["summary"]

        # (1) unmet (metrics_file отсутствует) + смет нет → зелёный (блокировка эффективна)
        code, out = _run()
        if code != 0:
            print(f"SELFTEST FAIL (1): базовое состояние красное:\n{out}")
            return 1
        # (2) unmet + approved → красный
        (budget / "rental-h100.json").write_text('{"status": "approved", "h800_hours": 400}', encoding="utf-8")
        code, out = _run()
        if code != 1:
            print(f"SELFTEST FAIL (2): одобрение при unmet не поймано:\n{out}")
            return 1
        # (3) unmet + draft → зелёный (оценка для планирования допустима)
        (budget / "rental-h100.json").write_text('{"status": "draft", "h800_hours": 400}', encoding="utf-8")
        code, _ = _run()
        if code != 0:
            print("SELFTEST FAIL (3): draft ошибочно заблокирован")
            return 1
        # (4) unmet + файл без статуса → красный (fail-closed)
        (budget / "rental-h100.json").write_text('{"h800_hours": 400}', encoding="utf-8")
        code, _ = _run()
        if code != 1:
            print("SELFTEST FAIL (4): файл без статуса не пойман")
            return 1
        # (5) met KPI + approved → зелёный
        (budget / "rental-h100.json").write_text('{"status": "approved"}', encoding="utf-8")
        mf = case / "evidence" / "kda-wyut" / "metrics.jsonl"
        mf.parent.mkdir(parents=True)
        rows = "\n".join(json.dumps({"step": i, "tok_s": 900.0}) for i in range(1, 31))
        mf.write_text(rows + "\n", encoding="utf-8")
        # пину добавляем metrics_file
        pins.write_text(json.dumps({
            "version": 1,
            "pins": [{"run": "selftest", "kpi_tok_s_baseline": 86, "threshold": 800,
                      "blocks_rental": True, "metrics_file": "evidence/kda-wyut/metrics.jsonl",
                      "rental_budget_paths": ["evidence/budget/rental-*.json"]}],
        }), encoding="utf-8")
        code, out = _run()
        if code != 0:
            print(f"SELFTEST FAIL (5): met-KPI не снял блокировку:\n{out}")
            return 1
        # (6) нечитаемые пины → fail-closed красный
        pins.write_text("{битый json", encoding="utf-8")
        code, _ = _run()
        if code != 2:
            print("SELFTEST FAIL (6): битые пины не дали fail-closed")
            return 1
    print("SELFTEST PASS: 6/6 (базовый, approved-при-unmet, draft, без статуса, met снимает, битые пины)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C-048: блокировка сметы аренды до выполнения KPI эффективности")
    parser.add_argument("--case", default=str(CASE_DIR), help="корень кейса")
    parser.add_argument("--pins", default=None, help="файл пинов (по умолчанию evidence/kpi-pins.json)")
    parser.add_argument("--json", action="store_true", help="JSON-отчёт")
    parser.add_argument("--quiet", action="store_true", help="только вердикт")
    parser.add_argument("--selftest", action="store_true", help="мутантный самотест (канон C-043/44/45)")
    args = parser.parse_args(argv)

    if args.selftest:
        return _selftest()
    code, report = evaluate(args.case, args.pins)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.quiet:
        print(f"{'PASS' if report['passed'] else 'FAIL'} {report['summary']}")
    else:
        for v in report["violations"]:
            print(f"FAIL [{v['class']}] {v.get('pin')}: {v['message']}")
        print(report["summary"])
    return code


if __name__ == "__main__":
    sys.exit(main())
