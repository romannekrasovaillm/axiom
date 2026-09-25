#!/usr/bin/env python3
"""C-041 — поведенческий страж лимита стоимости прогона (AD-8, ADR-011).

Инвариант AD-8 («стоимость прогона лимитирована до запуска») защищён не прозой
ADR-004, а артефактом: смета лежит в ``evidence/budget/<run-ref>.json`` и
проверяется до старта прогона. Страж проверяет факт, а не надпись:

  (a) смета найдена для каждого объявленного прогона;
  (b) обязательные поля заполнены и ``usd_estimate <= limit_usd``;
  (c) смета датирована раньше старта прогона (``created_at`` строго раньше
      ``run_date`` манифеста прогона, если манифест есть);
  (d) сметы нет — ненулевой код и «смета отсутствует: запуск блокирован».

Режимы::

    python3 tools/check_budget_gate.py --verify [--run-ref RUN]...
    python3 tools/check_budget_gate.py --estimate RUN --gpu-hours 97 \\
        --limit-usd 200 --usd 200 --approved-by owner

Генерация атомарна и при отсутствии входных данных не пишет НИЧЕГО: смета без
прогона и прогон без сметы равнозначно недействительны (фабрикация
доказательства запрещена, AD-8/AD-4).

Коды возврата: ``0`` — сметы валидны (или генерация удалась); ``1`` — нарушение
(блокировка запуска); ``2`` — ошибка аргументов.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

#: Схема артефакта сметы (контракт — evidence/budget/README.md).
ESTIMATE_SCHEMA = "budget-estimate/v1"

#: Каталог смет относительно каталога кейса.
BUDGET_DIR = "evidence/budget"

#: Манифесты прогонов: `<run-ref>` берётся из поля run_ref (или из имени файла).
MANIFEST_GLOB = "evidence/**/*run-manifest*.json"

#: Прогоны, объявленные в evidence/README.md как доказательства гейтов A4/A5.
#: Смета требуется до старта каждого (AD-8); реестр пополняется вместе с
#: объявлением нового прогона — иначе у прогона нет ни сметы, ни требования.
DECLARED_RUN_REFS = ("a4-skeleton", "a5-rerun")

#: Обязательные поля сметы и их человекочитаемый смысл.
REQUIRED_TEXT_FIELDS = {
    "run_ref": "ссылка на прогон",
    "gpu_type": "тип GPU (напр. H800)",
    "budget_method": "метод оценки",
    "stop_rule": "стоп-правило",
    "approved_by": "утвердивший",
    "created_at": "дата создания (ISO-8601)",
}
REQUIRED_NUMERIC_FIELDS = {
    "gpu_hours_estimate": "оценка в GPU-часах",
    "usd_estimate": "оценка стоимости, USD",
    "limit_usd": "лимит, USD",
}

#: Метод обязан ссылаться на калибровку из AD-8 (arXiv 2412.19437, 343 TFLOP/s).
CALIBRATION_MARKERS = ("2412.19437", "343")
CALIBRATION_DEFAULT = (
    "калибровка по arXiv 2412.19437: 343 TFLOP/s эффективных на H800"
)
STOP_RULE_DEFAULT = "остановка после ближайшего чекпойнта при превышении лимита"
GPU_TYPE_DEFAULT = "H800"

#: run-ref — имя файла сметы: без разделителей пути и «..».
RUN_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class BudgetError(ValueError):
    """Смета отсутствует, не читается или не проходит проверку."""


@dataclass(frozen=True)
class RunManifest:
    """Манифест прогона в срезе, нужном стражу стоимости."""

    run_ref: str
    run_date: Optional[date]
    path: Path


def parse_run_date(value: Any) -> Optional[date]:
    """Дата из ISO-8601 (`2026-09-12` или полный datetime). None — не разобрать."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        return None


def _is_number(value: Any) -> bool:
    """Число, но не bool (True — не «1 GPU-час»)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    """Конечное число: NaN и inf не проходят ни одно сравнение — это дыра в гейте."""
    return _is_number(value) and math.isfinite(value)


def load_manifests(case_dir: Path) -> list[RunManifest]:
    """Читает манифесты прогонов: run_ref → run_date (AD-4, evidence/README.md)."""
    manifests: list[RunManifest] = []
    for path in sorted(case_dir.glob(MANIFEST_GLOB)):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        run_ref = data.get("run_ref")
        if not isinstance(run_ref, str) or not run_ref.strip():
            run_ref = path.name.removesuffix("-run-manifest.json")
        manifests.append(
            RunManifest(
                run_ref=run_ref.strip(),
                run_date=parse_run_date(data.get("run_date")),
                path=path,
            )
        )
    return manifests


def required_runs(case_dir: Path, explicit: Iterable[str] | None = None) -> list[str]:
    """Прогоны, для которых смета обязана существовать.

    Явный ``--run-ref`` задаёт адресную проверку одного прогона; без него
    множество — объявленные прогоны кейса плюс все найденные манифесты.
    """
    explicit_refs = [ref.strip() for ref in (explicit or []) if ref and ref.strip()]
    if explicit_refs:
        return list(dict.fromkeys(explicit_refs))

    declared = list(DECLARED_RUN_REFS)
    discovered = {manifest.run_ref for manifest in load_manifests(case_dir)}
    extra = sorted(discovered - set(declared))
    return declared + extra


def run_dates(case_dir: Path) -> dict[str, date]:
    """Карта run_ref → дата прогона по всем найденным манифестам."""
    dates: dict[str, date] = {}
    for manifest in load_manifests(case_dir):
        if manifest.run_date is not None:
            dates.setdefault(manifest.run_ref, manifest.run_date)
    return dates


def validate_estimate(estimate: Any, run_ref: str) -> list[str]:
    """Проверяет поля сметы. Возвращает список нарушений (пусто = валидна)."""
    if not isinstance(estimate, dict):
        return ["смета не является JSON-объектом"]

    errs: list[str] = []
    for field, meaning in REQUIRED_TEXT_FIELDS.items():
        value = estimate.get(field)
        if not isinstance(value, str) or not value.strip():
            errs.append(f"{field}: поле отсутствует или пусто ({meaning})")

    for field, meaning in REQUIRED_NUMERIC_FIELDS.items():
        value = estimate.get(field)
        if not _is_number(value):
            errs.append(f"{field}: не число ({meaning})")
        elif not math.isfinite(value):
            errs.append(f"{field}: не конечное число — NaN/inf не оценка ({meaning})")
        elif field == "gpu_hours_estimate" and value <= 0:
            errs.append(f"{field}: оценка часов должна быть > 0 (сейчас {value})")
        elif field != "gpu_hours_estimate" and value < 0:
            errs.append(f"{field}: отрицательная величина ({value})")

    actual_ref = estimate.get("run_ref")
    if isinstance(actual_ref, str) and actual_ref.strip() and actual_ref.strip() != run_ref:
        errs.append(
            f"run_ref: смета для прогона '{actual_ref.strip()}', "
            f"а требуется '{run_ref}'"
        )

    method = estimate.get("budget_method")
    if isinstance(method, str) and method.strip():
        if not any(marker in method for marker in CALIBRATION_MARKERS):
            errs.append(
                "budget_method: не ссылается на калибровку AD-8 "
                f"(arXiv {CALIBRATION_MARKERS[0]}, {CALIBRATION_MARKERS[1]} TFLOP/s)"
            )

    if isinstance(estimate.get("created_at"), str) and estimate["created_at"].strip():
        if parse_run_date(estimate["created_at"]) is None:
            errs.append(
                "created_at: не разбирается как ISO-8601 "
                f"('{estimate['created_at']}')"
            )

    usd = estimate.get("usd_estimate")
    limit = estimate.get("limit_usd")
    if _is_number(usd) and _is_number(limit) and usd > limit:
        errs.append(
            f"usd_estimate {usd} > limit_usd {limit}: "
            "смета превышает лимит — запуск блокирован (AD-8)"
        )
    return errs


def verify_run(case_dir: Path, run_ref: str, dates: dict[str, date]) -> tuple[list[str], str]:
    """Проверяет смету одного прогона. Возвращает (нарушения, строка статуса)."""
    if not RUN_REF_RE.match(run_ref):
        return [
            f"недопустимая ссылка на прогон '{run_ref}': "
            "буквы, цифры, '.', '_', '-' (имя файла сметы)"
        ], "недопустимая ссылка"

    path = case_dir / BUDGET_DIR / f"{run_ref}.json"
    rel = f"{BUDGET_DIR}/{run_ref}.json"
    if not path.is_file():
        return [f"смета отсутствует: запуск блокирован — нет {rel}"], "смета отсутствует"

    try:
        estimate = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"смета не читается: {rel}: {exc}"], "смета не читается"

    errs = validate_estimate(estimate, run_ref)

    run_date = dates.get(run_ref)
    created = parse_run_date(estimate.get("created_at")) if isinstance(estimate, dict) else None
    if run_date is not None and created is not None:
        if not created < run_date:
            errs.append(
                f"created_at {created.isoformat()} не раньше даты прогона "
                f"{run_date.isoformat()}: смета не предшествует запуску (AD-8)"
            )

    if errs:
        return errs, "FAIL"

    usd = estimate["usd_estimate"]
    limit = estimate["limit_usd"]
    detail = f"usd {usd:g} ≤ лимит {limit:g}"
    if run_date is not None and created is not None:
        detail += f"; смета {created.isoformat()} раньше прогона {run_date.isoformat()}"
    return [], f"OK: {detail}"


def cmd_verify(case_dir: Path, explicit: Iterable[str] | None) -> int:
    """Режим стража: сметы объявленных прогонов существуют и состоятельны."""
    refs = required_runs(case_dir, explicit)
    if not refs:
        print(
            "страж лимита стоимости: не объявлено ни одного прогона — "
            "проверять нечего (реестр пуст)",
            file=sys.stderr,
        )
        return 1

    dates = run_dates(case_dir)
    print("Страж лимита стоимости (AD-8, C-041): сметы прогонов до запуска")
    failed = 0
    for run_ref in refs:
        errs, status = verify_run(case_dir, run_ref, dates)
        if errs:
            failed += 1
            print(f"  [{run_ref}] FAIL")
            for err in errs:
                print(f"      - {err}")
        else:
            print(f"  [{run_ref}] {status}")

    if failed:
        print(
            f"\nИтог: FAIL — прогонов без валидной сметы: {failed} из {len(refs)}; "
            "запуск блокирован (AD-8)"
        )
        return 1
    print(f"\nИтог: PASS — сметы {len(refs)} прогонов валидны и предшествуют запуску")
    return 0


def write_estimate_atomic(path: Path, estimate: dict) -> None:
    """Атомарно пишет JSON (temp + os.replace); частичных файлов не остаётся."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(
            json.dumps(estimate, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _build_estimate(args: argparse.Namespace) -> tuple[Optional[dict], list[str]]:
    """Собирает смету из входных данных. Без данных — (None, причины)."""
    errs: list[str] = []

    run_ref = (args.estimate or "").strip()
    if not run_ref:
        errs.append("не указан прогон: --estimate <run-ref>")
    elif not RUN_REF_RE.match(run_ref):
        errs.append(
            f"недопустимая ссылка на прогон '{run_ref}': "
            "буквы, цифры, '.', '_', '-' (имя файла сметы)"
        )

    gpu_hours = args.gpu_hours
    if gpu_hours is None:
        errs.append("нет оценки GPU-часов: --gpu-hours")
    elif not _is_finite_number(gpu_hours) or gpu_hours <= 0:
        errs.append(f"--gpu-hours должна быть > 0 (получено {gpu_hours!r})")

    limit = args.limit_usd
    if limit is None:
        errs.append("нет лимита стоимости: --limit-usd")
    elif not _is_finite_number(limit) or limit < 0:
        errs.append(f"--limit-usd должен быть ≥ 0 (получено {limit!r})")

    usd: Optional[float] = None
    if args.usd is not None and args.usd_per_gpu_hour is not None:
        errs.append("задайте что-то одно: --usd или --usd-per-gpu-hour")
    elif args.usd is not None:
        if not _is_finite_number(args.usd) or args.usd < 0:
            errs.append(f"--usd должен быть ≥ 0 (получено {args.usd!r})")
        else:
            usd = round(float(args.usd), 2)
    elif args.usd_per_gpu_hour is not None:
        rate = args.usd_per_gpu_hour
        if not _is_finite_number(rate) or rate < 0:
            errs.append(f"--usd-per-gpu-hour должен быть ≥ 0 (получено {rate!r})")
        elif _is_finite_number(gpu_hours) and gpu_hours > 0:
            usd = round(float(gpu_hours) * float(rate), 2)
    else:
        errs.append("нет оценки стоимости: --usd или --usd-per-gpu-hour")

    approved_by = (args.approved_by or "").strip()
    if not approved_by:
        errs.append("смета не утверждена: --approved-by <кто утвердил>")

    gpu_type = (args.gpu_type or "").strip() or GPU_TYPE_DEFAULT
    budget_method = (args.budget_method or "").strip() or CALIBRATION_DEFAULT
    stop_rule = (args.stop_rule or "").strip() or STOP_RULE_DEFAULT
    created_at = (args.created_at or "").strip() or datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )

    if errs:
        return None, errs

    estimate = {
        "schema": ESTIMATE_SCHEMA,
        "run_ref": run_ref,
        "gpu_type": gpu_type,
        "gpu_hours_estimate": gpu_hours,
        "usd_estimate": usd,
        "limit_usd": limit,
        "budget_method": budget_method,
        "stop_rule": stop_rule,
        "created_at": created_at,
        "approved_by": approved_by,
        "generated_by": "tools/check_budget_gate.py",
    }
    return estimate, []


def cmd_estimate(case_dir: Path, args: argparse.Namespace) -> int:
    """Режим генератора. Без входных данных — ненулевой код и ни одной записи."""
    estimate, errs = _build_estimate(args)
    if errs or estimate is None:
        print(
            "смета не создана: входные данные прогона отсутствуют или неполны "
            "(фабрикация сметы запрещена, AD-8):",
            file=sys.stderr,
        )
        for err in errs:
            print(f"  - {err}", file=sys.stderr)
        return 1

    path = case_dir / BUDGET_DIR / f"{estimate['run_ref']}.json"
    if path.exists() and not args.force:
        print(
            f"смета не перезаписана: {path} уже существует "
            "(--force для осознанной замены)",
            file=sys.stderr,
        )
        return 1

    errs = validate_estimate(estimate, estimate["run_ref"])
    if errs:
        print("смета не записана: собранный артефакт не валиден:", file=sys.stderr)
        for err in errs:
            print(f"  - {err}", file=sys.stderr)
        return 1

    if estimate["usd_estimate"] > estimate["limit_usd"]:
        print(
            f"предупреждение: usd_estimate {estimate['usd_estimate']} > "
            f"limit_usd {estimate['limit_usd']} — страж C-041 заблокирует запуск",
            file=sys.stderr,
        )

    known_run_date = run_dates(case_dir).get(estimate["run_ref"])
    created = parse_run_date(estimate["created_at"])
    if known_run_date is not None and created is not None and not created < known_run_date:
        print(
            f"предупреждение: смета датирована {created.isoformat()} не раньше "
            f"прогона {known_run_date.isoformat()} — страж C-041 заблокирует "
            "запуск; смета обязана предшествовать прогону (AD-8)",
            file=sys.stderr,
        )

    write_estimate_atomic(path, estimate)
    print(f"смета записана: {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="C-041: страж лимита стоимости прогона (AD-8) и генератор смет.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="проверить сметы объявленных прогонов (режим гейта)",
    )
    parser.add_argument(
        "--run-ref",
        action="append",
        default=None,
        help="адресная проверка прогона (можно несколько раз; --verify)",
    )
    parser.add_argument(
        "--case-dir", default=".", help="каталог кейса (по умолчанию '.')"
    )
    parser.add_argument(
        "--estimate",
        metavar="RUN_REF",
        default=None,
        help="сгенерировать смету для прогона <run-ref>",
    )
    parser.add_argument("--gpu-type", default=None, help=f"тип GPU (по умолчанию {GPU_TYPE_DEFAULT})")
    parser.add_argument("--gpu-hours", type=float, default=None, help="оценка в GPU-часах (> 0)")
    parser.add_argument("--usd", type=float, default=None, help="оценка стоимости, USD")
    parser.add_argument(
        "--usd-per-gpu-hour",
        type=float,
        default=None,
        help="ставка USD за GPU-час (если не задан --usd)",
    )
    parser.add_argument("--limit-usd", type=float, default=None, help="лимит стоимости, USD")
    parser.add_argument("--budget-method", default=None, help="метод оценки (калибровка AD-8)")
    parser.add_argument("--stop-rule", default=None, help="стоп-правило прогона")
    parser.add_argument("--approved-by", default=None, help="кто утвердил смету")
    parser.add_argument("--created-at", default=None, help="дата создания (ISO-8601, по умолчанию сейчас)")
    parser.add_argument(
        "--force", action="store_true", help="перезаписать существующую смету (осознанно)"
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    case_dir = Path(args.case_dir)
    if args.verify:
        return cmd_verify(case_dir, args.run_ref)
    if args.estimate:
        return cmd_estimate(case_dir, args)
    build_parser().print_usage(sys.stderr)
    print(
        "укажите режим: --verify (страж) или --estimate <run-ref> (генератор)",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
