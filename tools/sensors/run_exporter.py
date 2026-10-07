"""CLI: прогнать один экспортёр пакета и записать факты (ADR-038, дельта G/H).

    python3 -m tools.sensors.run_exporter --id S-031

Находит экспортёр по ``describe().id`` среди пакетов, собирает пин предмета
(``config_subject``) и пишет факты в ``evidence/facts/``. Недоступный источник —
``unverified`` с причиной (C-007), не исключение.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from ._common import config_subject
from .packs import discover_exporters
from .protocol import check_conformance, collect_all
from .subject import REPO_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Прогон экспортёра пакета (ADR-038)")
    parser.add_argument("--id", default=None, help="id экспортёра (например S-031)")
    parser.add_argument("--out-dir", default=None, help="каталог фактов (evidence/facts)")
    parser.add_argument("--root", default=None, help="корень репозитория (по умолчанию)")
    parser.add_argument("--selftest", action="store_true", help="проверка протокола по всем экспортёрам")
    return parser


def _find(exporter_id: str):
    for exporter in discover_exporters():
        if exporter.describe().id == exporter_id:
            return exporter
    return None


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.selftest:
        errors = check_conformance(discover_exporters())
        for err in errors:
            print(f"[run_exporter] FAIL: {err}")
        print(f"conformance: {'OK' if not errors else str(len(errors)) + ' нарушений'}")
        return 1 if errors else 0

    if not args.id:
        print("run_exporter: укажите --id <S-NNN> или --selftest", file=sys.stderr)
        return 2
    exporter = _find(args.id)
    if exporter is None:
        print(f"run_exporter: экспортёр {args.id} не найден среди пакетов", file=sys.stderr)
        return 1
    root = args.root or str(REPO_ROOT)
    subject = config_subject(device="cpu")
    facts = collect_all(exporter, subject, root=root)
    for fact in facts:
        fact.write(args.out_dir)
        print(f"{fact.sensor} {fact.fact} = {fact.value} [{fact.status}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
