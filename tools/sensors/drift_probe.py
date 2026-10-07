"""S-026 — побитовая идентичность коротких прогонов (drift-проба, дельта C5).

Два коротких прогона по N шагов с одним сидом; сравнение sha256 параметров.
Исполняется только в окне GB10/на аренде (тяжёлый JAX-прогон); на исполнителе
факт ``unverified``. Сравнение — чистая функция :func:`compare_hashes`,
тестируемая на CPU.

Запуск::

    python3 -m tools.sensors.drift_probe --allow-device   # только в окне
    python3 -m tools.sensors.drift_probe --selftest
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

from ._common import config_subject, emit

METHOD = "два коротких прогона по N шагов с одним сидом → sha256 параметров"


def compare_hashes(first: str, second: str) -> bool:
    """Побитовая идентичность: совпадение sha256 параметров."""
    return bool(first) and first == second


def measure(
    *,
    allow_device: bool = False,
    out_dir: Optional[str | Path] = None,
    device: Optional[str] = None,
) -> dict[str, Any]:
    value = None
    note = "drift-проба исполняется только в окне GB10/на аренде — на исполнителе не запускается"
    if allow_device:
        # Полный прогон — в окне; здесь честная ветка отказа, чтобы не подставить число.
        value = None
        note = "окно запрошено, но прогон цикла не встроен (интеграция линии-1 — open_question)"
    written = emit(
        "S-026", "short_run_bitwise_identical", value, unit="bool", quality="measured",
        method=METHOD, subject=config_subject(device=device), out_dir=out_dir,
        status="ok" if value is not None else "unverified", note=note,
    )
    return {"short_run_bitwise_identical": written}


def run_selftest() -> int:
    import tempfile

    checks: list[tuple[str, bool]] = [
        ("совпадающие хеши → True", compare_hashes("ab" * 32, "ab" * 32) is True),
        ("разные хеши → False", compare_hashes("ab" * 32, "cd" * 32) is False),
        ("пустой хеш → False", compare_hashes("", "") is False),
    ]
    with tempfile.TemporaryDirectory(prefix="s026-selftest-") as tmp:
        written = measure(allow_device=False, out_dir=tmp)
    checks.append(("без окна → unverified",
                   written["short_run_bitwise_identical"]["status"] == "unverified"))
    ok = all(p for _, p in checks)
    for label, passed in checks:
        print(f"[selftest] {'PASS' if passed else 'FAIL'}: {label}")
    print(f"[selftest] {'PASS' if ok else 'FAIL'}: drift_probe")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="S-026: drift-проба (ADR-037)")
    parser.add_argument("--allow-device", action="store_true")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--selftest", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.selftest:
        return run_selftest()
    written = measure(allow_device=args.allow_device, out_dir=args.out_dir, device=args.device)
    for name, rec in written.items():
        print(f"S-026 {name} = {rec['value']} [{rec['status']}] {rec['note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
