"""S-019 — обёртка A5-drift-отчёта: побитовая идентичность (дельта C3).

Читает ``evidence/a5-drift-report-*.json`` и сводит все проверки к одному факту
``bitwise_identical``: истинно, если каждый пункт с полем ``identical`` истинен.
Отчёта нет → ``unverified``.

Запуск::

    python3 -m tools.sensors.wrap_drift [--report PATH] [--out-dir DIR]
    python3 -m tools.sensors.wrap_drift --selftest
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

from ._common import REPO_ROOT, config_subject, emit
from ._wrap import latest_glob, read_json

METHOD = "обёртка: evidence/a5-drift-report-*.json (все checks.*.identical)"


def _collect_identical(obj: Any, acc: list[bool]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "identical" and isinstance(value, bool):
                acc.append(value)
            else:
                _collect_identical(value, acc)
    elif isinstance(obj, list):
        for item in obj:
            _collect_identical(item, acc)


def measure(
    report_path: Optional[str | Path] = None,
    *,
    out_dir: Optional[str | Path] = None,
    device: Optional[str] = None,
) -> dict[str, Any]:
    path = Path(report_path) if report_path else latest_glob("evidence/a5-drift-report-*.json")
    data = read_json(path) if path else None
    flags: list[bool] = []
    _collect_identical(data, flags) if data is not None else None
    value = all(flags) if flags else None
    run_ref = None
    if isinstance(data, dict):
        run_ref = data.get("report")
    subject = config_subject(run_ref=run_ref, device=device)
    written = emit(
        "S-019", "bitwise_identical", value, unit="bool", quality="wrapped", method=METHOD,
        subject=subject, out_dir=out_dir,
        status="ok" if value is not None else "unverified",
        note=(f"проверок identical: {len(flags)}" if value is not None else "нет отчёта A5-drift"),
    )
    return {"bitwise_identical": written}


def run_selftest() -> int:
    import json
    import tempfile

    checks: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory(prefix="s019-selftest-") as tmp:
        good = Path(tmp) / "a5-drift-report-x.json"
        good.write_text(json.dumps({"checks": {"a": {"identical": True}, "b": {"identical": True}}}), encoding="utf-8")
        written = measure(good, out_dir=tmp)
        checks.append(("все identical → True", written["bitwise_identical"]["value"] is True))
        bad = Path(tmp) / "a5-drift-report-y.json"
        bad.write_text(json.dumps({"checks": {"a": {"identical": True}, "b": {"identical": False}}}), encoding="utf-8")
        broken = measure(bad, out_dir=tmp)
        checks.append(("одна не-identical → False", broken["bitwise_identical"]["value"] is False))
        missing = measure(Path(tmp) / "nope.json", out_dir=tmp)
        checks.append(("нет отчёта → unverified", missing["bitwise_identical"]["status"] == "unverified"))
    ok = all(p for _, p in checks)
    for label, passed in checks:
        print(f"[selftest] {'PASS' if passed else 'FAIL'}: {label}")
    print(f"[selftest] {'PASS' if ok else 'FAIL'}: wrap_drift")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="S-019: обёртка A5-drift (ADR-036)")
    parser.add_argument("--report", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--selftest", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.selftest:
        return run_selftest()
    written = measure(args.report, out_dir=args.out_dir, device=args.device)
    for name, rec in written.items():
        print(f"S-019 {name} = {rec['value']} [{rec['status']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
