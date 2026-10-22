#!/usr/bin/env python3
"""C-037 — пакет обязан нести рабочий ruleset кейса, а не заготовку.

Проверяет конфигурацию канала гейта (ADR-011, п. 5): набор правил в рабочем
``кейс/CONSTRAINTS.yaml`` и в пакетном ``кейс/.arch-handoff/CONSTRAINTS.yaml``
совпадает по числу правил и sha256. Расхождение означает, что внешний
исполнитель получил пакет с чужими/устаревшими правилами и не узнал об этом
(либо работает дефолтный гейт по заготовке — «зелёный по пустоте»).

Почему сверка по байтам, а не «по смыслу»: генератор пакета копирует рабочий
файл как есть (``harness::generate_handoff``), поэтому источник истины —
корневой файл; любое расхождение пакета с ним — сигнал, а не повод для
снисходительности. Число правил печатается как диагноз (совпадает/нет), но
вердикт держится на sha256.

Входы (относительно каталога кейса, переопределяется ``--case``):
  * ``CONSTRAINTS.yaml``                     — рабочий ruleset кейса;
  * ``.arch-handoff/CONSTRAINTS.yaml``       — пакетный ruleset.

Запуск::

    python3 tools/check_handoff_ruleset.py [--case DIR] [--quiet]

Коды возврата:
  * ``0``  — PASS: пакетный ruleset совпадает с рабочим;
  * ``1``  — FAIL: пакетный ruleset расходится с рабочим;
  * ``2``  — NOT-VERIFIED: пакета (или рабочего файла) нет — сверка
             невозможна; ложный PASS запрещён.

Скрипт — stdlib-only (никакого PyYAML): в среде гейта внешних зависимостей нет,
а падать на импорте страж не имеет права.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

#: Имя рабочего ruleset'а в корне кейса.
WORKING_NAME = "CONSTRAINTS.yaml"
#: Путь пакетного ruleset'а относительно корня кейса.
PACKET_REL = Path(".arch-handoff") / "CONSTRAINTS.yaml"

#: Начало элемента списка правил (``- id:`` / ``- name:``). Эвристика без
#: YAML-парсера: годится для диагностики числа правил, вердикт — по sha256.
_RULE_ENTRY_RE = re.compile(r"^[ \t]*- (?:id|name):", re.MULTILINE)

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_NOT_VERIFIED = 2


def sha256_of(path: Path) -> str:
    """SHA-256 содержимого файла (hex)."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_rules(path: Path) -> int:
    """Число элементов-правил (``- id:``/``- name:``) в файле."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return len(_RULE_ENTRY_RE.findall(text))


def _describe(label: str, path: Path, case: Path) -> str:
    """Строка описания файла: путь, число правил, sha256 (укороченный)."""
    try:
        rel = path.relative_to(case)
    except ValueError:
        rel = path
    return (
        f"{label}: {rel} — правил: {count_rules(path)}, "
        f"sha256: {sha256_of(path)[:16]}…"
    )


def evaluate(case: Path) -> tuple[int, list[str]]:
    """Сверяет рабочий и пакетный ruleset'ы. Возвращает (код, строки вывода)."""
    working = case / WORKING_NAME
    packet = case / PACKET_REL
    lines: list[str] = []

    if not working.is_file():
        lines.append(
            f"NOT-VERIFIED: рабочий ruleset {working} не найден — "
            "сверять не с чем (ложный PASS запрещён)"
        )
        return EXIT_NOT_VERIFIED, lines
    if not packet.is_file():
        lines.append(
            f"NOT-VERIFIED: пакетный ruleset {packet} отсутствует — "
            "сверка невозможна, ложный PASS запрещён. "
            "Сгенерируйте пакет: `arch-ml handoff <harness> --repo . --task …`"
        )
        return EXIT_NOT_VERIFIED, lines

    lines.append(_describe("рабочий ruleset", working, case))
    lines.append(_describe("пакетный ruleset", packet, case))

    working_hash = sha256_of(working)
    packet_hash = sha256_of(packet)
    working_count = count_rules(working)
    packet_count = count_rules(packet)
    lines.append(f"правил: рабочий {working_count}, пакетный {packet_count}")
    lines.append(f"sha256: рабочий {working_hash}, пакетный {packet_hash}")

    if working_hash == packet_hash:
        lines.append(
            "PASS: пакет несёт рабочий ruleset кейса (sha256 и число правил совпали)"
        )
        return EXIT_PASS, lines

    reasons: list[str] = []
    if working_count != packet_count:
        reasons.append(f"число правил: {working_count} ≠ {packet_count}")
    reasons.append("sha256 не совпал")
    lines.append(
        "FAIL: пакетный ruleset расходится с рабочим ("
        + "; ".join(reasons)
        + ") — пакет несёт заготовку или устаревшую копию"
    )
    return EXIT_FAIL, lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="C-037: сверка рабочего и пакетного CONSTRAINTS.yaml",
    )
    parser.add_argument(
        "--case",
        type=Path,
        default=Path.cwd(),
        help="каталог кейса (по умолчанию — текущий)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="печатать только вердикт (stderr)",
    )
    args = parser.parse_args(argv)

    case = args.case.resolve()
    code, lines = evaluate(case)
    if not args.quiet:
        for line in lines[:-1]:
            print(line)
    # Вердикт — в stderr: его читает гейт, stdout остаётся диагностике.
    print(lines[-1], file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
