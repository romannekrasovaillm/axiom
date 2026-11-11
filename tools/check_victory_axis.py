#!/usr/bin/env python3
"""C-013 — страж оси победы: единственный путь в пакете решения (ADR-058, AD-1).

Ось победы объявлена один раз — в ``docs/DECISION-PACKAGE.json`` на пути
``choice.victory_axis``.  Страж проверяет три вещи, каждая из которых ломается
независимо:

* пакет существует, читается как JSON-объект и несёт непустую строку
  ``choice.victory_axis`` (ошибки — из :func:`env.axis.load_axis`);
* на верхнем уровне пакета НЕТ ``victory_axis`` — два места объявления дают
  двусмысленность (какая ось настоящая?);
* архитектурный спайн и карточка AD-1 ссылаются ровно на
  ``choice.victory_axis`` — ось не переизобретается в текстах.

Код 0 — PASS, 1 — FAIL; диагностика печатается по каждому пункту.

Запуск (из каталога кейса)::

    python3 tools/check_victory_axis.py
    python3 tools/check_victory_axis.py --selftest
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

PACKAGE_REL = "docs/DECISION-PACKAGE.json"
SPINE_REL = "ARCHITECTURE-SPINE.md"
AD1_REL = "model/AD-1-os-pobedy-odna.md"
AXIS_REF = "choice.victory_axis"

#: Путь оси в пакете решения — та же константа, что ``env/axis.py`` (ADR-058).
AXIS_PATH = ("choice", "victory_axis")
_AXIS_POINTER = "#/" + "/".join(AXIS_PATH)


class AxisError(ValueError):
    """Ось не читается: нет файла, нет ключа, пусто или не строка."""


def load_axis(package_path: Path) -> str:
    """Непустая ось из пакета решения; :class:`AxisError` — диагностика.

    Копия :func:`env.axis.load_axis` (нормативная реализация для манифеста v2)
    в объёме, нужном стражу. ``env/`` не копируется в снапшот воркспейса, а
    C-013 — не инфраструктурное правило (``infra: false``) и обязан исполняться
    В ВОРКСПЕЙСЕ: импорт ``env`` сделал бы страж неисполнимым в песочнице.
    Сообщения совпадают с ``env/axis.py``.
    """
    path = Path(package_path)
    try:
        doc: Any = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AxisError(f"{path}: пакет решения не найден") from exc
    except OSError as exc:
        raise AxisError(f"{path}: пакет решения не читается: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise AxisError(f"{path}: пакет решения не JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise AxisError(f"{path}: пакет решения — не JSON-объект")

    current: Any = doc
    for key in AXIS_PATH:
        if not isinstance(current, dict) or key not in current:
            raise AxisError(f"{path}{_AXIS_POINTER}: ключ оси отсутствует")
        current = current[key]
    if not isinstance(current, str) or not current.strip():
        raise AxisError(f"{path}{_AXIS_POINTER}: ось пуста или не строка")
    return current


def has_top_level_axis(doc: dict) -> bool:
    """Есть ли ``victory_axis`` на верхнем уровне (двусмысленность — ошибка)."""
    return isinstance(doc, dict) and "victory_axis" in doc

EXIT_PASS = 0
EXIT_FAIL = 1


def evaluate(root: Path) -> list[str]:
    """Нарушения C-013 в дереве ``root`` (пусто = PASS)."""
    root = Path(root)
    errors: list[str] = []
    package = root / PACKAGE_REL

    # 1. Пакет решения: файл, JSON-объект, ось по единственному пути.
    doc = None
    if not package.is_file():
        errors.append(f"{PACKAGE_REL}: файл не найден")
    else:
        try:
            doc = json.loads(package.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{PACKAGE_REL}: не читается как JSON: {exc}")
        else:
            if not isinstance(doc, dict):
                errors.append(f"{PACKAGE_REL}: пакет решения — не JSON-объект")
                doc = None
    if doc is not None:
        try:
            load_axis(package)
        except AxisError as exc:
            errors.append(str(exc))
        # 2. Ось только в choice: верхний уровень — двусмысленность.
        if has_top_level_axis(doc):
            errors.append(
                f"{PACKAGE_REL}: victory_axis на верхнем уровне — двусмысленность, "
                f"ось объявляется только в choice.victory_axis"
            )

    # 3. Ссылки на единственный путь оси в архитектурных текстах.
    for rel in (SPINE_REL, AD1_REL):
        path = root / rel
        if not path.is_file():
            errors.append(f"{rel}: файл не найден")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if AXIS_REF not in text:
            errors.append(f"{rel}: нет ссылки '{AXIS_REF}' на ось победы (AD-1)")
    return errors


# ---------------------------------------------------------------------------
# selftest: фикстуры во временном дереве, репозиторий не затрагивается
# ---------------------------------------------------------------------------


def _write_fixture(
    root: Path,
    *,
    axis_value: object = "победа по оси: фитнес-гейт на K3-задачах",
    top_level_axis: bool = False,
    refs: bool = True,
) -> None:
    package: dict = {"choice": {}}
    if axis_value is not None:
        package["choice"]["victory_axis"] = axis_value
    if top_level_axis:
        package["victory_axis"] = "дубль на верхнем уровне"
    (root / "docs").mkdir(parents=True)
    (root / "model").mkdir(parents=True)
    (root / PACKAGE_REL).write_text(
        json.dumps(package, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    ref = f"## AD-1\nось: `{AXIS_REF}`\n" if refs else "## AD-1\nось не названа\n"
    (root / SPINE_REL).write_text(ref, encoding="utf-8")
    (root / AD1_REL).write_text(ref, encoding="utf-8")


def run_selftest() -> int:
    """Мутанты C-013: зелёная фикстура и три независимых нарушения."""
    checks: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory(prefix="victory-axis-selftest-") as tmp:
        base = Path(tmp)

        good = base / "good"
        _write_fixture(good)
        checks.append(("валидная фикстура — PASS", evaluate(good) == []))

        no_field = base / "no-field"
        _write_fixture(no_field, axis_value=None)
        checks.append(("нет choice.victory_axis — FAIL", evaluate(no_field) != []))

        duplicate = base / "duplicate"
        _write_fixture(duplicate, top_level_axis=True)
        checks.append(("victory_axis на верхнем уровне — FAIL", evaluate(duplicate) != []))

        empty = base / "empty"
        _write_fixture(empty, axis_value="   ")
        checks.append(("пустая ось — FAIL", evaluate(empty) != []))

        no_ref = base / "no-ref"
        _write_fixture(no_ref, refs=False)
        checks.append(("нет ссылки choice.victory_axis в AD-1/спайне — FAIL", evaluate(no_ref) != []))

    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"[victory-axis:selftest] {'OK' if ok else 'FAIL'}: {name}")
    if failed:
        print(f"victory-axis:selftest: {len(failed)} из {len(checks)} сценариев не совпали")
        return EXIT_FAIL
    print(f"victory-axis:selftest: OK ({len(checks)} сценариев)")
    return EXIT_PASS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="C-013: ось победы — единственный путь choice.victory_axis (ADR-058)",
    )
    parser.add_argument(
        "--root", type=Path, default=_REPO_ROOT,
        help="каталог кейса (по умолчанию — корень репозитория)",
    )
    parser.add_argument(
        "--selftest", action="store_true",
        help="прогнать фикстуры-мутанты во временном каталоге",
    )
    args = parser.parse_args(argv)

    if args.selftest:
        return run_selftest()

    errors = evaluate(args.root.resolve())
    for err in errors:
        print(f"[victory-axis] FAIL: {err}")
    print(f"victory-axis: {'PASS' if not errors else f'{len(errors)} нарушений'}")
    return EXIT_PASS if not errors else EXIT_FAIL


if __name__ == "__main__":
    raise SystemExit(main())
