#!/usr/bin/env python3
"""C-048 — страж блокировки арендной сметы эффективностью прогона.

Запрос владельца: «страж эффективности с блокировкой сметы аренды».  Контекст:
ADR-034 (аренда 8×H100 под претрейн после посадки data-parallel и KPI WY/UT),
C-046 (roofline-гейт уже пиннит ток/с), пин ``kda-wyut-delta`` помечен
``blocks_rental: true``.

Смысл стража — институциональный, а не арифметический: **неэффективный алгоритм
на аренде — это дни аренды, растянутые на годы обучения**.  Пока performance-пин,
объявленный блокирующим аренду, не взят *измерением*, арендную смету подписывать
нельзя.  C-046 отвечает на вопрос «быстр ли прогон?» (сам по себе, без последствий
для сметы); C-048 отвечает на вопрос «можно ли на этом основании платить за
аренду?» — и отвечает отказом, пока блокер не снят фактом.

Логика
------
1. Читаются KPI-пины (``--pins``, по умолчанию ``evidence/kpi-pins.json``) и
   отбираются записи с ``blocks_rental: true``.
2. Для каждого блокирующего пина берётся фактический замер: путь метрик — из поля
   пина (``metrics_path``/``metrics``) либо выводится по конвенции
   ``evidence/<семья>/metrics.jsonl`` (нога ``-delta`` живёт в каталоге базовой
   семьи).  Механика замера — та же, что в C-046
   (``tools/check_performance_roofline.py``): медиана ``tok_s`` последнего окна
   (≥10 записей) против ``threshold`` пина.
3. **Блокер** — пин, не доказавший взятие порога: медиана ниже порога, файл
   метрик пуст/бит/нечитаем, либо файла нет вовсе.  В отличие от C-046, где
   отсутствующий файл метрик нейтрален (прогон ещё не начинался), здесь
   отсутствие данных — **недостиж**: смету нельзя разблокировать непроверенной
   эффективностью.  Это осознанное различие, а не рассинхрон: C-046 оценивает
   прогон, C-048 — право платить за аренду, и второе строже первого.

Вердикты и коды
---------------
* ``status``        — ``--rental-budget`` не передан: печатается статус блокировки
  (блокеры перечислены поимённо) → exit 0.  Нет сметы на входе — блокировать
  нечего, но статус обязателен.
* ``budget-absent`` — ``--rental-budget`` передан, файла нет: смета ещё не
  заведена → exit 0 с предупреждением (блокировать нечего).
* ``pass``          — смета существует и блокеров нет → exit 0.
* ``blocked``       — смета существует и есть неснятые блокеры → exit 1
  (сообщение: «арендная смета заблокирована неснятыми performance-блокерами:
  [список]»).

Отказ стража (fail-closed)
--------------------------
Повреждённый/отсутствующий файл пинов не имеет права тихо разоружить стража:
это ошибка входа → exit 1.  Аналогично блокирующий пин без числового
``threshold`` — не «нечего проверять», а сломанный страж (exit 1).

Запуск::

    python3 tools/check_rental_budget_gate.py --pins evidence/kpi-pins.json
    python3 tools/check_rental_budget_gate.py --pins evidence/kpi-pins.json \\
        --rental-budget evidence/budget/rental-l3.json
    python3 tools/check_rental_budget_gate.py --selftest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

#: Переиспользуем механику замера C-046 (разрешено задачей: «та же механика,
#: импорт оттуда допустим»).  Числа порога — данные пина, не константа здесь.
import check_performance_roofline as roofline

#: Схема отчёта.
REPORT_SCHEMA = "axiom-rental-budget-gate/1"

EXIT_OK = 0
EXIT_FAIL = 1

#: Якорь репозитория — дефолтные пути ищутся от каталога с ``tools/``.
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PINS = REPO_ROOT / "evidence" / "kpi-pins.json"

#: Поле пина, помечающее блокировку арендной сметы.
BLOCKS_FIELD = "blocks_rental"

#: Поля пина с явным путём метрик (первое непустое побеждает).
METRICS_FIELDS = ("metrics_path", "metrics")

#: Суффиксы имени прогона, у которых каталог метрик — каталог базовой семьи:
#: ``kda-wyut-delta`` → ``evidence/kda-wyut/metrics.jsonl``.
_DELTA_SUFFIXES = ("-delta",)

#: Причины недостижения (часть контракта отчёта).
REASON_NO_METRICS_FILE = "данных нет: файл метрик отсутствует — недостиж"
REASON_NO_DATA = "fail-closed: нет данных: файл метрик пуст"
REASON_BELOW_THRESHOLD = "медиана tok_s ниже порога пина"


class InputError(Exception):
    """Вход существует, но непригоден: fail-closed (страж разоружать нельзя)."""


# --------------------------------------------------------------------------- #
# Чтение входов
# --------------------------------------------------------------------------- #


def _is_number(value: Any) -> bool:
    """Число, но не ``bool`` (в Python ``bool`` — подкласс ``int``)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def load_rental_pins(pins_path: str | Path) -> list[dict[str, Any]]:
    """Прочитать пины целиком (сохраняя ``blocks_rental`` и путь метрик).

    Полный валидатор роуфлайна (``load_pins``) отбрасывает незнакомые поля —
    здесь же они несущие, поэтому читаем сами, но проверяем строго: сбой входа —
    fail-closed.
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
        raise InputError(f"{path}: ожидался список пинов (или объект с ключом «pins»)")

    pins: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        pins.append(_coerce_rental_pin(path, index, entry))
    return pins


def _coerce_rental_pin(path: Path, index: int, entry: Any) -> dict[str, Any]:
    """Проверить одну запись; блокирующий пин обязан несть числовой порог."""
    if not isinstance(entry, dict):
        raise InputError(f"{path}: пин #{index}: запись не является объектом")

    run = entry.get("run")
    if not isinstance(run, str) or not run.strip():
        raise InputError(
            f"{path}: пин #{index}: поле «run» должно быть непустой строкой"
        )

    blocking = entry.get(BLOCKS_FIELD, False)
    if not isinstance(blocking, bool):
        raise InputError(
            f"{path}: пин #{index}: поле «{BLOCKS_FIELD}» должно быть bool: "
            f"{blocking!r}"
        )

    pin: dict[str, Any] = {"run": run, BLOCKS_FIELD: blocking, "_raw": entry}

    if blocking:
        threshold = entry.get("threshold")
        if not _is_number(threshold):
            raise InputError(
                f"{path}: пин #{index}: блокирующий пин «{run}» без числового "
                f"«threshold»: {threshold!r}"
            )
        if float(threshold) <= 0.0:
            raise InputError(
                f"{path}: пин #{index}: поле «threshold» должно быть > 0: "
                f"{threshold!r}"
            )
        pin["threshold"] = float(threshold)

    for field in METRICS_FIELDS:
        value = entry.get(field)
        if isinstance(value, str) and value.strip():
            pin["metrics_path"] = value.strip()
            break

    return pin


def infer_metrics_family(run: str) -> str:
    """Каталог семьи метрик для прогона (нога ``-delta`` → базовая семья)."""
    for suffix in _DELTA_SUFFIXES:
        if run.endswith(suffix) and len(run) > len(suffix):
            return run[: -len(suffix)]
    return run


def resolve_metrics_path(pin: dict[str, Any], repo_root: Path) -> Path:
    """Путь метрик пина: явное поле либо конвенция ``evidence/<семья>/metrics.jsonl``."""
    explicit = pin.get("metrics_path")
    if explicit:
        path = Path(explicit)
        return path if path.is_absolute() else (repo_root / path)
    family = infer_metrics_family(pin["run"])
    return repo_root / "evidence" / family / "metrics.jsonl"


# --------------------------------------------------------------------------- #
# Замер и блокеры
# --------------------------------------------------------------------------- #


def measure_blocker(
    pin: dict[str, Any], repo_root: Path, window: int
) -> dict[str, Any]:
    """Замерить пин; вернуть запись-блокер при недостижении (иначе ``None``).

    Недостиж — всё, кроме доказанного взятия порога: нет файла, файл пуст/бит,
    либо медиана ниже порога (fail-closed: «не оценён» ≠ «взят»).
    """
    threshold = pin["threshold"]
    metrics = resolve_metrics_path(pin, repo_root)
    base = {
        "run": pin["run"],
        "metrics_file": str(metrics),
        "threshold": threshold,
        "tok_s_median": None,
        "shortfall_factor": None,
        "reason": None,
    }

    if not metrics.exists():
        base["reason"] = REASON_NO_METRICS_FILE
        return base

    try:
        values = roofline._iter_tok_s(metrics)
    except roofline.InputError as exc:
        base["reason"] = f"fail-closed: {exc}"
        return base

    if not values:
        base["reason"] = f"{REASON_NO_DATA}: {metrics}"
        return base

    step = max(window, roofline.MIN_WINDOW)
    samples = values[-step:]
    fact = roofline.median(samples)
    base["tok_s_median"] = round(fact, 6)
    base["samples"] = len(samples)
    base["records"] = len(values)

    if fact >= threshold:
        return None

    base["reason"] = REASON_BELOW_THRESHOLD
    base["shortfall_factor"] = round(threshold / fact, 6) if fact > 0 else None
    return base


def collect_blockers(
    pins: list[dict[str, Any]], repo_root: Path, window: int
) -> tuple[list[dict[str, Any]], int]:
    """Список блокеров и число блокирующих пинов (порядок — как в файле пинов)."""
    blockers: list[dict[str, Any]] = []
    blocking_count = 0
    seen: set[str] = set()
    for pin in pins:
        if not pin.get(BLOCKS_FIELD, False):
            continue
        blocking_count += 1
        # Одноимённые пины не должны двоить блокер (пин-файл — реестр, не набор).
        if pin["run"] in seen:
            continue
        seen.add(pin["run"])
        blocker = measure_blocker(pin, repo_root, window)
        if blocker is not None:
            blockers.append(blocker)
    return blockers, blocking_count


# --------------------------------------------------------------------------- #
# Проверка
# --------------------------------------------------------------------------- #


def run_check(
    *,
    pins_path: str | Path,
    rental_budget: str | Path | None = None,
    repo_root: Path | None = None,
    window: int = roofline.WINDOW,
) -> tuple[int, dict[str, Any]]:
    """Вердикт гейта.  Возвращает ``(код, отчёт)``."""
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    resolved_pins = Path(pins_path)

    try:
        pins = load_rental_pins(resolved_pins)
    except InputError as exc:
        return EXIT_FAIL, {
            "schema": REPORT_SCHEMA,
            "verdict": "no-data",
            "pins_file": str(resolved_pins),
            "rental_budget": str(rental_budget) if rental_budget is not None else None,
            "budget_exists": False,
            "blocking_pins": 0,
            "blockers": [],
            "reason": f"fail-closed: {exc}",
            "message": f"fail-closed: {exc}",
        }

    blockers, blocking_count = collect_blockers(pins, root, window)
    names = [blocker["run"] for blocker in blockers]

    budget = Path(rental_budget) if rental_budget is not None else None
    budget_exists = budget is not None and budget.is_file()

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "pins_file": str(resolved_pins),
        "rental_budget": str(budget) if budget is not None else None,
        "budget_exists": budget_exists,
        "blocking_pins": blocking_count,
        "blockers": blockers,
    }

    blocked_message = (
        "арендная смета заблокирована неснятыми performance-блокерами: "
        + ", ".join(names)
    )

    if budget is None:
        report["verdict"] = "status"
        report["reason"] = "смета не передана — блокировать нечего"
        report["message"] = (
            f"статус блокировки арендной сметы: блокеров {len(blockers)}"
            + (f" — {', '.join(names)}" if names else " (неснятых нет)")
            + " [смета не передана — блокировать нечего]"
        )
        return EXIT_OK, report

    if not budget_exists:
        report["verdict"] = "budget-absent"
        report["reason"] = "смета ещё не заведена (файл отсутствует)"
        report["message"] = (
            f"сметы ещё нет ({budget}) — блокировать нечего; "
            f"блокеров в очереди {len(blockers)}"
            + (f": {', '.join(names)}" if names else "")
        )
        return EXIT_OK, report

    if blockers:
        report["verdict"] = "blocked"
        report["reason"] = "есть неснятые performance-блокеры"
        report["message"] = blocked_message
        return EXIT_FAIL, report

    report["verdict"] = "pass"
    report["reason"] = "неснятых performance-блокеров нет"
    report["message"] = (
        "арендная смета разблокирована: неснятых performance-блокеров нет"
    )
    return EXIT_OK, report


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
    if report.get("verdict") == "blocked":
        print(f"[rental-budget-gate] FAIL: {report.get('message', '')}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Страж блокировки арендной сметы эффективностью (C-048)"
    )
    parser.add_argument("--pins", dest="pins_path", default=str(DEFAULT_PINS),
                        help=f"файл KPI-пинов (по умолчанию {DEFAULT_PINS})")
    parser.add_argument("--rental-budget", dest="rental_budget", default=None,
                        help="путь к артефакту арендной сметы; не передан — режим статуса")
    parser.add_argument("--window", type=int, default=roofline.WINDOW,
                        help=f"размер окна медианы, не меньше {roofline.MIN_WINDOW} "
                             f"(по умолчанию {roofline.WINDOW})")
    parser.add_argument("--json", dest="json_path", default=None,
                        help="куда записать отчёт")
    parser.add_argument("--quiet", action="store_true", help="не печатать отчёт в stdout")
    parser.add_argument("--selftest", action="store_true",
                        help="синтетический selftest с мутантами (tmp; реальные данные не трогаются)")
    args = parser.parse_args(argv)

    if args.selftest:
        return run_selftest()

    code, report = run_check(
        pins_path=args.pins_path,
        rental_budget=args.rental_budget,
        window=args.window,
    )
    report["exit_code"] = code
    _emit(report, args.json_path, args.quiet)
    return code


# --------------------------------------------------------------------------- #
# Selftest: мутанты краснеют, снятый блокер — зелёный, статус без сметы — exit 0
# --------------------------------------------------------------------------- #


def _write_metrics(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps({"step": step, "tok_s": value}, ensure_ascii=False) + "\n"
            for step, value in enumerate(values)
        ),
        encoding="utf-8",
    )


def _write_pins(path: Path, entries: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps({"pins": entries}, ensure_ascii=False), encoding="utf-8")


def run_selftest() -> int:
    """Синтетика в ``tmp``: блокер/снятие/нет сметы/нет пинов/детерминизм."""
    import tempfile

    checks: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory(prefix="rental-budget-selftest-") as tmp:
        root = Path(tmp)
        pins = root / "kpi-pins.json"
        _write_pins(pins, [
            {"run": "kda-wyut-delta", "kpi_tok_s_baseline": 86, "threshold": 800,
             BLOCKS_FIELD: True},
        ])
        budget = root / "rental-l3.json"
        budget.write_text('{"run_ref": "l3"}', encoding="utf-8")

        # (е) метрик нет → недостиж → блокер → FAIL со списком.
        code, report = run_check(pins_path=pins, rental_budget=budget, repo_root=root)
        checks.append(("нет метрик + смета → exit 1", code == EXIT_FAIL))
        checks.append(("нет метрик + смета → verdict blocked",
                       report["verdict"] == "blocked"))
        checks.append(("список блокеров называет пин",
                       [b["run"] for b in report["blockers"]] == ["kda-wyut-delta"]))
        checks.append(("сообщение — «арендная смета заблокирована…»",
                       "арендная смета заблокирована" in report["message"]))

        # (в) без сметы → exit 0 + статус с блокером.
        code_s, report_s = run_check(pins_path=pins, repo_root=root)
        checks.append(("без сметы → exit 0", code_s == EXIT_OK))
        checks.append(("без сметы → verdict status", report_s["verdict"] == "status"))
        checks.append(("без сметы → блокер перечислен",
                       [b["run"] for b in report_s["blockers"]] == ["kda-wyut-delta"]))

        # (д) смета передана, файла нет → exit 0 с предупреждением.
        code_ab, report_ab = run_check(
            pins_path=pins, rental_budget=root / "nope.json", repo_root=root
        )
        checks.append(("смета без файла → exit 0", code_ab == EXIT_OK))
        checks.append(("смета без файла → verdict budget-absent",
                       report_ab["verdict"] == "budget-absent"))

        # (б) блокер снят (медиана выше порога) → PASS.
        _write_metrics(root / "evidence" / "kda-wyut" / "metrics.jsonl",
                       [900.0 + i for i in range(20)])
        code_p, report_p = run_check(pins_path=pins, rental_budget=budget, repo_root=root)
        checks.append(("блокер снят → exit 0", code_p == EXIT_OK))
        checks.append(("блокер снят → verdict pass", report_p["verdict"] == "pass"))
        checks.append(("блокер снят → blockers пуст", report_p["blockers"] == []))

        # (е) файл метрик существует, но пуст → недостиж (fail-closed).
        _write_metrics(root / "evidence" / "kda-wyut" / "metrics.jsonl", [])
        code_e, report_e = run_check(pins_path=pins, rental_budget=budget, repo_root=root)
        checks.append(("пустые метрики → exit 1", code_e == EXIT_FAIL))
        checks.append(("пустые метрики → недостиж",
                       "fail-closed" in report_e["blockers"][0]["reason"]))

        # (г) пинов blocks_rental нет → PASS (нечем блокировать).
        pins_plain = root / "plain-pins.json"
        _write_pins(pins_plain, [
            {"run": "kda-wyut-delta", "kpi_tok_s_baseline": 86, "threshold": 800},
        ])
        code_g, report_g = run_check(
            pins_path=pins_plain, rental_budget=budget, repo_root=root
        )
        checks.append(("нет блокирующих пинов → exit 0", code_g == EXIT_OK))
        checks.append(("нет блокирующих пинов → verdict pass",
                       report_g["verdict"] == "pass"))
        checks.append(("нет блокирующих пинов → blocking_pins 0",
                       report_g["blocking_pins"] == 0))

        # (д) детерминизм медианы.
        checks.append(("медиана нечётного ряда",
                       roofline.median([300.0, 100.0, 200.0]) == 200.0))
        checks.append(("медиана чётного ряда — среднее центральных",
                       roofline.median([400.0, 100.0, 300.0, 200.0]) == 250.0))

        # Повреждённый пин-файл → fail-closed, а не тихое разоружение.
        bad = root / "bad-pins.json"
        bad.write_text("{не json", encoding="utf-8")
        checks.append(("битый пин-файл → exit 1",
                       run_check(pins_path=bad, rental_budget=budget)[0] == EXIT_FAIL))
        # Блокирующий пин без порога → fail-closed.
        pins_nothr = root / "no-threshold-pins.json"
        _write_pins(pins_nothr, [{"run": "kda-wyut-delta", BLOCKS_FIELD: True}])
        checks.append(("блокирующий пин без порога → exit 1",
                       run_check(pins_path=pins_nothr, rental_budget=budget)[0] == EXIT_FAIL))

        # Дефолтный пин-файл репозитория объявляет блокировку аренды (ADR-032).
        if DEFAULT_PINS.exists():
            default_pins = load_rental_pins(DEFAULT_PINS)
            armed = [p for p in default_pins if p.get(BLOCKS_FIELD)]
            checks.append((
                "дефолтный evidence/kpi-pins.json блокирует аренду",
                any(p["run"] == "kda-wyut-delta" and p["threshold"] == 800.0
                    for p in armed),
            ))

    ok = all(passed for _, passed in checks)
    for label, passed in checks:
        print(f"[selftest] {'PASS' if passed else 'FAIL'}: {label}")
    print(
        f"[selftest] {'PASS' if ok else 'FAIL'}: блокер недостиж краснит смету, "
        "снятый блокер зелёный, без сметы — статус exit 0, без блокирующих пинов — "
        "pass, пусто/бито — fail-closed, медиана детерминирована"
    )
    return EXIT_OK if ok else EXIT_FAIL


if __name__ == "__main__":
    raise SystemExit(main())
