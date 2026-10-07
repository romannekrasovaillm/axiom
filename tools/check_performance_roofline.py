#!/usr/bin/env python3
"""Страж performance-roofline: фактическая скорость KPI-прогона против порога.

Слепое пятно контура (rubric axiom_fitness_blind_spot_assessment, 4.88/5): ни
одно из 45 правил не сравнивало измеренную скорость с объявленным KPI.  KDA на
явных матрицах dk×dk с MFU < 1 % (80–86 ток/с против цели 10×) прошёл все гейты:
они проверяли прозу, трассируемость, пиннинг и детерминизм, но не то, что прогон
ещё и *быстрый*.  Этот страж закрывает дыру: он берёт **объявленный** KPI-пин
(ADR-032) и сравнивает с **фактической** медианой ток/с прогона.

Логика
------
1. Читается файл KPI-пинов (по умолчанию ``evidence/kpi-pins.json``) — список
   записей ``{"run", "kpi_tok_s_baseline", "threshold"}``.
2. Прогон ищется в пинах по имени (``--run``).
3. Фактическая скорость — медиана ``tok_s`` по последним записям файла метрик
   (``--metrics``, jsonl с ключами ``step``/``tok_s``); окно — не меньше
   последних 10 записей (детерминизм).

Вердикты
--------
* ``ok``        — пин есть, медиана ``tok_s ≥ threshold`` → exit 0;
* ``neutral``   — прогон не объявлен KPI-пином (нет записи) либо файл метрик
  отсутствует (``--metrics`` не существует: прогон не выполнялся, предмета
  проверки нет) → exit 0;
* ``regression``— пин есть, медиана ``tok_s < threshold`` → exit 1
  (печать: пин, факт, кратность недобора ``threshold / median``);
* ``no-data``   — пин есть, файл метрик существует, но пуст/бит/нечитаем → exit 1
  (fail-closed: «не оценён» ≠ «чисто»; прогон начат — молчаливый зелёный на
  отсутствии данных запрещён).

Граница отсутствия/пустоты
--------------------------
Отсутствующий файл метрик — не провал прогона, а его отсутствие: KPI-пин
(ADR-032) активен до прогона WY/UT, и fail-closed на несуществующий предмет
превратил бы гейт в вечный красный («запланированный красный» — отвергнутая
практика). Поэтому нет файла → ``neutral``; существует, но без данных → ``no-data``.

Коды возврата
-------------
* ``0`` — ``ok`` | ``neutral``;
* ``1`` — ``regression`` | ``no-data`` (fail-closed).

Запуск::

    python3 tools/check_performance_roofline.py --selftest
    python3 tools/check_performance_roofline.py --run kda-wyut-delta \\
        --metrics evidence/kda-wyut/metrics.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

#: Схема отчёта.
REPORT_SCHEMA = "axiom-performance-roofline/1"

#: Причина нейтрального вердикта, когда предмета проверки ещё нет (прогон не
#: выполнялся).  Точная формулировка — часть контракта отчёта.
REASON_NOT_RUN = "прогон не выполнялся — файл метрик отсутствует"

EXIT_OK = 0
EXIT_FAIL = 1
#: ``--require-verified``: недоказанное не открывает расход (ADR-036, дельта E1).
EXIT_UNVERIFIED = 3

#: Минимальный размер окна медианы (детерминизм: ≥10 последних записей).
MIN_WINDOW = 10
#: Размер окна по умолчанию.
WINDOW = 10

#: Якорь репозитория — дефолтный пин-файл ищется от каталога с ``tools/``,
#: чтобы страж находил свои данные независимо от рабочего каталога вызова.
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PINS = REPO_ROOT / "evidence" / "kpi-pins.json"

#: Обязательные поля записи KPI-пина.
PIN_FIELDS = ("run", "kpi_tok_s_baseline", "threshold")


class InputError(Exception):
    """Вход существует, но непригоден: fail-closed (вердикт ``no-data``)."""


# --------------------------------------------------------------------------- #
# Чтение входов
# --------------------------------------------------------------------------- #


def _is_number(value: Any) -> bool:
    """Число, но не ``bool`` (в Python ``bool`` — подкласс ``int``)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def load_pins(pins_path: str | Path) -> list[dict[str, Any]]:
    """Прочитать и провалидировать KPI-пины; сбой — fail-closed.

    Принимается либо список записей, либо объект ``{"pins": [...]}``.  Любая
    некорректная запись — ошибка: повреждённый пин-файл не имеет права тихо
    разоружить стража.
    """
    path = Path(pins_path)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise InputError(f"{path}: файл KPI-пинов не найден") from exc
    except OSError as exc:
        raise InputError(
            f"{path}: файл KPI-пинов не читается ({type(exc).__name__}): {exc}"
        ) from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InputError(f"{path}: не разбирается как JSON: {exc}") from exc

    entries = data.get("pins") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        raise InputError(
            f"{path}: ожидался список пинов (или объект с ключом «pins»)"
        )

    pins: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        pins.append(_coerce_pin(path, index, entry))
    return pins


def _coerce_pin(path: Path, index: int, entry: Any) -> dict[str, Any]:
    """Привести одну запись пина к контракту; нарушение — fail-closed."""
    if not isinstance(entry, dict):
        raise InputError(f"{path}: пин #{index}: запись не является объектом")
    missing = [field for field in PIN_FIELDS if field not in entry]
    if missing:
        raise InputError(
            f"{path}: пин #{index}: нет поля «{missing[0]}» "
            f"(обязательны: {', '.join(PIN_FIELDS)})"
        )
    run = entry["run"]
    if not isinstance(run, str) or not run.strip():
        raise InputError(f"{path}: пин #{index}: поле «run» должно быть непустой строкой")
    for field in ("kpi_tok_s_baseline", "threshold"):
        if not _is_number(entry[field]):
            raise InputError(
                f"{path}: пин #{index}: поле «{field}» не число: {entry[field]!r}"
            )
        if float(entry[field]) <= 0.0:
            raise InputError(
                f"{path}: пин #{index}: поле «{field}» должно быть > 0: {entry[field]!r}"
            )
    return {
        "run": run,
        "kpi_tok_s_baseline": float(entry["kpi_tok_s_baseline"]),
        "threshold": float(entry["threshold"]),
    }


def _iter_tok_s(path: Path) -> list[float]:
    """Значения ``tok_s`` из jsonl-метрик в порядке строк; сбой — fail-closed."""
    values: list[float] = []
    try:
        with path.open("rb") as handle:
            for lineno, raw in enumerate(handle, start=1):
                if not raw.strip():
                    continue
                try:
                    record = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise InputError(
                        f"{path}:{lineno}: строка не разбирается как JSON: {exc}"
                    ) from exc
                if not isinstance(record, dict):
                    raise InputError(
                        f"{path}:{lineno}: строка не является объектом JSON"
                    )
                if "tok_s" not in record or not _is_number(record["tok_s"]):
                    raise InputError(
                        f"{path}:{lineno}: нет числового поля «tok_s»"
                    )
                value = float(record["tok_s"])
                if value < 0.0:
                    raise InputError(
                        f"{path}:{lineno}: поле «tok_s» отрицательно: {value!r}"
                    )
                values.append(value)
    except OSError as exc:
        raise InputError(
            f"{path}: файл метрик не читается ({type(exc).__name__}): {exc}"
        ) from exc
    return values


def median(values: Iterable[float]) -> float:
    """Детерминированная медиана (для чётного числа — среднее двух центральных)."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("медиана пустого ряда не определена")
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


# --------------------------------------------------------------------------- #
# Проверка
# --------------------------------------------------------------------------- #


def _find_pin(pins: list[dict[str, Any]], run: str) -> dict[str, Any] | None:
    for pin in pins:
        if pin["run"] == run:
            return pin
    return None


def _run_check_impl(
    run: str,
    metrics_path: str | Path | None = None,
    *,
    pins_path: str | Path | None = None,
    window: int = WINDOW,
) -> tuple[int, dict[str, Any]]:
    """Проверить факт ток/с прогона против KPI-пина.  Возвращает ``(код, отчёт)``."""
    resolved_pins = Path(pins_path) if pins_path is not None else DEFAULT_PINS

    try:
        pins = load_pins(resolved_pins)
    except InputError as exc:
        return EXIT_FAIL, _fail_closed(
            run, "no-data", f"fail-closed: {exc}", resolved_pins, metrics_path, window
        )

    pin = _find_pin(pins, run)
    if pin is None:
        # Прогон не объявлен KPI-пином — нейтральный PASS (не наш контур).
        return EXIT_OK, {
            "schema": REPORT_SCHEMA,
            "verdict": "neutral",
            "run": run,
            "pin": None,
            "pins_file": str(resolved_pins),
            "metrics_file": str(metrics_path) if metrics_path is not None else None,
            "window": max(window, MIN_WINDOW),
            "records": 0,
            "samples": 0,
            "tok_s_median": None,
            "threshold": None,
            "kpi_tok_s_baseline": None,
            "speedup_vs_baseline": None,
            "shortfall_factor": None,
            "message": f"прогон «{run}» не объявлен KPI-пином — проверка нейтральна",
        }

    if metrics_path is None:
        return EXIT_FAIL, _fail_closed(
            run, "no-data", "не передан файл метрик (--metrics)", resolved_pins,
            metrics_path, window, pin=pin,
        )

    metrics = Path(metrics_path)
    if not metrics.exists():
        # Прогон не выполнялся: предмета проверки ещё нет.  KPI-пин активен до
        # прогона, поэтому fail-closed здесь был бы вечным красным — нейтрально.
        return EXIT_OK, _neutral_not_run(run, resolved_pins, metrics, window, pin=pin)

    try:
        values = _iter_tok_s(metrics)
    except InputError as exc:
        return EXIT_FAIL, _fail_closed(
            run, "no-data", f"fail-closed: {exc}", resolved_pins, metrics_path,
            window, pin=pin,
        )

    if not values:
        return EXIT_FAIL, _fail_closed(
            run, "no-data", f"нет данных: файл метрик пуст: {metrics}", resolved_pins,
            metrics_path, window, pin=pin,
        )

    step = max(window, MIN_WINDOW)
    samples = values[-step:]
    fact = median(samples)
    threshold = pin["threshold"]
    baseline = pin["kpi_tok_s_baseline"]

    report = {
        "schema": REPORT_SCHEMA,
        "verdict": "ok" if fact >= threshold else "regression",
        "run": run,
        "pin": pin,
        "pins_file": str(resolved_pins),
        "metrics_file": str(metrics),
        "window": step,
        "records": len(values),
        "samples": len(samples),
        "tok_s_median": round(fact, 6),
        "tok_s_min": round(min(samples), 6),
        "tok_s_max": round(max(samples), 6),
        "threshold": threshold,
        "kpi_tok_s_baseline": baseline,
        "speedup_vs_baseline": round(fact / baseline, 6) if baseline else None,
        # Кратность недобора определена только когда порог не взят.
        "shortfall_factor": (
            round(threshold / fact, 6) if fact < threshold and fact > 0 else None
        ),
    }

    if fact >= threshold:
        report["message"] = (
            f"прогон «{run}»: медиана {fact:.3f} ток/с ≥ порога {threshold:.3f} "
            f"(×{fact / baseline:.2f} к базису {baseline:.1f})"
        )
        return EXIT_OK, report

    shortfall = (
        f"недобор ×{threshold / fact:.2f}" if fact > 0 else "недобор бесконечен (факт 0)"
    )
    report["message"] = (
        f"регрессия «{run}»: медиана {fact:.3f} ток/с < порога {threshold:.3f} "
        f"({shortfall}; базис {baseline:.1f} ток/с, факт ×{fact / baseline:.2f} к базису)"
    )
    return EXIT_FAIL, report


#: Класс вердикта по строке отчёта (ADR-036, дельта E1).
_VERDICT_CLASS = {
    "ok": "ok",
    "regression": "regression",
    "neutral": "unverified",
    "no-data": "no-data",
}


def _facts_tok_s(run: str, facts_dir: str | Path | None) -> float | None:
    """Ток/с из факта S-012 для того же ``run_ref`` (если есть — иначе None)."""
    try:
        import sys as _sys

        root = Path(__file__).resolve().parents[1]
        if str(root) not in _sys.path:
            _sys.path.insert(0, str(root))
        from tools.sensors.fact import read_latest

        record = read_latest(
            "S-012", "tok_s_median_window", ["run_ref"],
            out_dir=facts_dir, subject={"run_ref": run},
        )
        if record and record.get("status") == "ok" and isinstance(record.get("value"), (int, float)):
            return float(record["value"])
    except Exception:  # noqa: BLE001 — нет фактов = прежний путь
        return None
    return None


def run_check(
    run: str,
    metrics_path: str | Path | None = None,
    *,
    pins_path: str | Path | None = None,
    window: int = WINDOW,
    require_verified: bool = False,
    facts_dir: str | Path | None = None,
) -> tuple[int, dict[str, Any]]:
    """Обёртка над :func:`_run_check_impl` с классом вердикта и политикой E1.

    Без ``require_verified`` поведение прежнее (нейтраль → 0). С флагом
    ``neutral`` → ``exit 3`` (unverified), ``no-data`` → ``exit 1``. Ток/с из
    факта S-012 (тот же ``run_ref``) добавляется в отчёт; расхождение с метриками
    помечается ``source_divergence`` — без смены вердикта.
    """
    code, report = _run_check_impl(run, metrics_path, pins_path=pins_path, window=window)
    report["verdict_class"] = _VERDICT_CLASS.get(str(report.get("verdict")), report.get("verdict"))
    fact_tok = _facts_tok_s(run, facts_dir)
    if fact_tok is not None:
        report["tok_s_facts"] = fact_tok
        median = report.get("tok_s_median")
        if isinstance(median, (int, float)) and abs(median - fact_tok) > 0.01 * max(1.0, abs(fact_tok)):
            report["source_divergence"] = True
            report["message"] = (
                str(report.get("message", ""))
                + f" [расхождение источников: метрики {median} vs факт S-012 {fact_tok}]"
            ).strip()
    if require_verified:
        if report.get("verdict") == "neutral":
            code = EXIT_UNVERIFIED
        elif report.get("verdict") == "no-data":
            code = EXIT_FAIL
    return code, report


def _fail_closed(
    run: str,
    verdict: str,
    reason: str,
    pins_path: Path,
    metrics_path: str | Path | None,
    window: int,
    *,
    pin: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": REPORT_SCHEMA,
        "verdict": verdict,
        "run": run,
        "pin": pin,
        "pins_file": str(pins_path),
        "metrics_file": str(metrics_path) if metrics_path is not None else None,
        "window": max(window, MIN_WINDOW),
        "records": 0,
        "samples": 0,
        "tok_s_median": None,
        "threshold": pin["threshold"] if pin else None,
        "kpi_tok_s_baseline": pin["kpi_tok_s_baseline"] if pin else None,
        "speedup_vs_baseline": None,
        "shortfall_factor": None,
        "reason": reason,
        "message": reason,
    }


def _neutral_not_run(
    run: str,
    pins_path: Path,
    metrics_path: Path,
    window: int,
    *,
    pin: dict[str, Any],
) -> dict[str, Any]:
    """Нейтральный отчёт: файл метрик отсутствует — прогон не выполнялся."""
    return {
        "schema": REPORT_SCHEMA,
        "verdict": "neutral",
        "run": run,
        "pin": pin,
        "pins_file": str(pins_path),
        "metrics_file": str(metrics_path),
        "window": max(window, MIN_WINDOW),
        "records": 0,
        "samples": 0,
        "tok_s_median": None,
        "threshold": pin["threshold"],
        "kpi_tok_s_baseline": pin["kpi_tok_s_baseline"],
        "speedup_vs_baseline": None,
        "shortfall_factor": None,
        "reason": REASON_NOT_RUN,
        "message": REASON_NOT_RUN,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _emit(report: dict[str, Any], json_path: str | None, quiet: bool) -> None:
    if json_path:
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(json_path).write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if not quiet:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    if report.get("verdict") in {"regression", "no-data"}:
        # Диагностика недобора — в stderr, чтобы быть видимой и под --quiet.
        print(f"[performance-roofline] FAIL: {report.get('message', '')}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Страж performance-roofline: факт ток/с прогона против KPI-пина"
    )
    parser.add_argument("--run", default=None, help="имя прогона (ключ KPI-пина)")
    parser.add_argument("--metrics", default=None,
                        help="jsonl метрик прогона (ключи step/tok_s)")
    parser.add_argument("--pins", dest="pins_path", default=None,
                        help=f"файл KPI-пинов (по умолчанию {DEFAULT_PINS})")
    parser.add_argument("--window", type=int, default=WINDOW,
                        help=f"размер окна медианы, не меньше {MIN_WINDOW} "
                             f"(по умолчанию {WINDOW})")
    parser.add_argument("--json", dest="json_path", default=None,
                        help="куда записать отчёт")
    parser.add_argument("--require-verified", action="store_true",
                        help="недоказанное не открывает расход: neutral → exit 3, no-data → exit 1 (ADR-036)")
    parser.add_argument("--facts-dir", default=None,
                        help="каталог фактов (evidence/facts) для сверки ток/с с S-012")
    parser.add_argument("--quiet", action="store_true", help="не печатать отчёт в stdout")
    parser.add_argument("--selftest", action="store_true",
                        help="синтетический selftest с мутантами (tmp; реальные данные не трогаются)")
    args = parser.parse_args(argv)

    if args.selftest:
        return run_selftest()

    if not args.run:
        parser.error("--run обязателен (кроме --selftest)")
    code, report = run_check(
        args.run, args.metrics, pins_path=args.pins_path, window=args.window,
        require_verified=args.require_verified, facts_dir=args.facts_dir,
    )
    report["exit_code"] = code
    _emit(report, args.json_path, args.quiet)
    return code


# --------------------------------------------------------------------------- #
# Selftest: мутанты краснеют, здоровый прогон зелёный, нет пина — нейтрально
# --------------------------------------------------------------------------- #


def _write_metrics(path: Path, values: list[float]) -> None:
    path.write_text(
        "".join(
            json.dumps({"step": step, "tok_s": value}, ensure_ascii=False) + "\n"
            for step, value in enumerate(values)
        ),
        encoding="utf-8",
    )


def run_selftest() -> int:
    """Синтетика в ``tmp``: порог/медиана/нейтральность/пустота + детерминизм."""
    import tempfile

    checks: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory(prefix="perf-roofline-selftest-") as tmp:
        root = Path(tmp)
        pins = root / "kpi-pins.json"
        pins.write_text(
            json.dumps(
                [{"run": "kda-wyut-delta", "kpi_tok_s_baseline": 86, "threshold": 800}]
            ),
            encoding="utf-8",
        )

        # (в) факт выше порога → PASS.
        fast = root / "fast.jsonl"
        _write_metrics(fast, [900.0 + i for i in range(20)])
        code_f, report_f = run_check("kda-wyut-delta", fast, pins_path=pins)
        checks.append(("факт выше порога → exit 0", code_f == EXIT_OK))
        checks.append(("факт выше порога → verdict ok", report_f["verdict"] == "ok"))

        # (а) факт ниже порога → FAIL с кратностью недобора.
        slow = root / "slow.jsonl"
        _write_metrics(slow, [86.0] * 20)
        code_s, report_s = run_check("kda-wyut-delta", slow, pins_path=pins)
        checks.append(("факт ниже порога → exit 1", code_s == EXIT_FAIL))
        checks.append(("факт ниже порога → verdict regression",
                       report_s["verdict"] == "regression"))
        checks.append(("факт ниже порога → кратность недобора 800/86",
                       abs(report_s["shortfall_factor"] - 800.0 / 86.0) < 1e-6))

        # (б) прогон не объявлен пином → neutral PASS.
        code_n, report_n = run_check("не-объявлен", slow, pins_path=pins)
        checks.append(("нет пина → exit 0", code_n == EXIT_OK))
        checks.append(("нет пина → verdict neutral", report_n["verdict"] == "neutral"))

        # (г) пустой файл метрик → fail-closed «нет данных».
        empty = root / "empty.jsonl"
        empty.write_text("\n", encoding="utf-8")
        code_e, report_e = run_check("kda-wyut-delta", empty, pins_path=pins)
        checks.append(("пустые метрики → exit 1", code_e == EXIT_FAIL))
        checks.append(("пустые метрики → verdict no-data",
                       report_e["verdict"] == "no-data"))
        checks.append(("пустые метрики → диагностика «нет данных»",
                       "нет данных" in report_e["message"]))

        # Отсутствующий файл метрик — прогон не выполнялся → neutral (exit 0);
        # существующий, но пустой — fail-closed (проверено выше).
        code_absent, report_absent = run_check(
            "kda-wyut-delta", root / "nope.jsonl", pins_path=pins
        )
        checks.append(("нет файла метрик → exit 0", code_absent == EXIT_OK))
        checks.append(("нет файла метрик → verdict neutral",
                       report_absent["verdict"] == "neutral"))
        checks.append(("нет файла метрик → reason «прогон не выполнялся»",
                       report_absent["reason"] == REASON_NOT_RUN))
        # Отсутствующий пин-файл — по-прежнему fail-closed.
        checks.append(
            ("нет пин-файла → exit 1",
             run_check("kda-wyut-delta", slow, pins_path=root / "no-pins.json")[0] == EXIT_FAIL)
        )

        # Граница: медиана ровно на пороге → PASS (≥).
        edge = root / "edge.jsonl"
        _write_metrics(edge, [800.0] * 12)
        code_b, _ = run_check("kda-wyut-delta", edge, pins_path=pins)
        checks.append(("медиана == порог → exit 0 (≥)", code_b == EXIT_OK))

        # Детерминизм медианы: порядок записей не влияет, чётное — среднее.
        checks.append(("медиана нечётного ряда", median([300.0, 100.0, 200.0]) == 200.0))
        checks.append(("медиана чётного ряда — среднее центральных",
                       median([400.0, 100.0, 300.0, 200.0]) == 250.0))
        shuffled = root / "shuffled.jsonl"
        _write_metrics(shuffled, [200.0, 400.0, 100.0, 300.0])
        plain = root / "plain.jsonl"
        _write_metrics(plain, [100.0, 200.0, 300.0, 400.0])
        checks.append(
            ("медиана инвариантна к порядку",
             run_check("kda-wyut-delta", shuffled, pins_path=pins)[1]["tok_s_median"]
             == run_check("kda-wyut-delta", plain, pins_path=pins)[1]["tok_s_median"])
        )

        # Окно: ранняя быстрая фаза, поздняя медленная → FAIL (берём последние).
        late_slow = root / "late-slow.jsonl"
        _write_metrics(late_slow, [900.0] * 20 + [80.0] * 20)
        checks.append(
            ("поздняя деградация ловится окном",
             run_check("kda-wyut-delta", late_slow, pins_path=pins)[0] == EXIT_FAIL)
        )
        # Ранняя медленная, поздняя быстрая → PASS.
        early_slow = root / "early-slow.jsonl"
        _write_metrics(early_slow, [80.0] * 20 + [900.0] * 20)
        checks.append(
            ("позднее ускорение → PASS",
             run_check("kda-wyut-delta", early_slow, pins_path=pins)[0] == EXIT_OK)
        )

        # Повреждённая строка метрик → fail-closed.
        broken = root / "broken.jsonl"
        broken.write_text("не json\n", encoding="utf-8")
        checks.append(
            ("мусорная строка метрик → exit 1",
             run_check("kda-wyut-delta", broken, pins_path=pins)[0] == EXIT_FAIL)
        )

        # Повреждённый пин-файл → fail-closed, а не тихое разоружение.
        bad_pins = root / "bad-pins.json"
        bad_pins.write_text(json.dumps([{"run": "kda-wyut-delta"}]), encoding="utf-8")
        checks.append(
            ("пин без threshold → exit 1",
             run_check("kda-wyut-delta", slow, pins_path=bad_pins)[0] == EXIT_FAIL)
        )

        # Поставленный по умолчанию пин-файл репозитория несёт запись ADR-032.
        if DEFAULT_PINS.exists():
            default_pins = load_pins(DEFAULT_PINS)
            armed = _find_pin(default_pins, "kda-wyut-delta")
            checks.append(
                ("дефолтный evidence/kpi-pins.json объявляет kda-wyut-delta",
                 armed is not None
                 and armed["threshold"] == 800.0
                 and armed["kpi_tok_s_baseline"] == 86.0)
            )

        # (E1) --require-verified: недоказанное не открывает расход.
        code_neutral_flag, rep_nf = run_check(
            "kda-wyut-delta", root / "nope.jsonl", pins_path=pins, require_verified=True
        )
        checks.append(
            ("require-verified: прогон не выполнялся → exit 3 (unverified)",
             code_neutral_flag == EXIT_UNVERIFIED
             and rep_nf.get("verdict_class") == "unverified")
        )
        code_nodata_flag, rep_nd = run_check(
            "kda-wyut-delta", empty, pins_path=pins, require_verified=True
        )
        checks.append(
            ("require-verified: пустые метрики → exit 1 (no-data)",
             code_nodata_flag == EXIT_FAIL and rep_nd.get("verdict_class") == "no-data")
        )
        # Без флага поведение прежнее (нейтраль не валит).
        code_neutral_noflag, _ = run_check("kda-wyut-delta", root / "nope.jsonl", pins_path=pins)
        checks.append(
            ("без флага нейтраль → exit 0 (C-046 не отменяется)",
             code_neutral_noflag == EXIT_OK)
        )

    ok = all(passed for _, passed in checks)
    for label, passed in checks:
        print(f"[selftest] {'PASS' if passed else 'FAIL'}: {label}")
    print(
        f"[selftest] {'PASS' if ok else 'FAIL'}: ниже порога — регрессия, нет пина "
        "или нет файла метрик — нейтрально, пусто/нечитаемо — fail-closed, "
        "медиана детерминирована"
    )
    return EXIT_OK if ok else EXIT_FAIL


if __name__ == "__main__":
    raise SystemExit(main())
