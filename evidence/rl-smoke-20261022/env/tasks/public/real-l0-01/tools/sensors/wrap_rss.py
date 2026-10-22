"""S-013 — обёртка ``peak_rss_mb`` журнала: память ХОСТ-процесса (дельта C3).

Имя факта обязано говорить «host»: ``peak_rss_mb`` в журнале — это RSS
хост-процесса (``resource.getrusage``), а не память устройства. Факт
``host_rss_peak_mb`` фиксирует именно это, чтобы число не читалось как память
модели. Журнала нет → ``unverified``.

Запуск::

    python3 -m tools.sensors.wrap_rss [--journal PATH] [--out-dir DIR]
    python3 -m tools.sensors.wrap_rss --selftest
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

from ._common import REPO_ROOT, config_subject, emit
from ._wrap import latest_glob, read_json

METHOD = "обёртка: peak_rss_mb журнала tools/pretrain_run.py (память ХОСТ-процесса)"


def _find_peak_rss(obj: Any) -> Optional[float]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "peak_rss_mb" and isinstance(value, (int, float)):
                return float(value)
            found = _find_peak_rss(value)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_peak_rss(item)
            if found is not None:
                return found
    return None


def measure(
    journal_path: Optional[str | Path] = None,
    *,
    out_dir: Optional[str | Path] = None,
    device: Optional[str] = None,
) -> dict[str, Any]:
    path = Path(journal_path) if journal_path else None
    if path is None:
        candidate = latest_glob("evidence/**/*journal*.json")
        path = candidate
    value = None
    run_ref = None
    if path is not None:
        data = read_json(path)
        value = _find_peak_rss(data)
        if isinstance(data, dict):
            run_ref = data.get("run_ref")
    subject = config_subject(run_ref=run_ref, device=device)
    written = emit(
        "S-013", "host_rss_peak_mb", value, unit="MiB", quality="wrapped", method=METHOD,
        subject=subject, out_dir=out_dir,
        status="ok" if value is not None else "unverified",
        note="" if value is not None else "журнал с peak_rss_mb не найден (память устройства сюда не подставляется)",
    )
    return {"host_rss_peak_mb": written}


def run_selftest() -> int:
    import json
    import tempfile

    checks: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory(prefix="s013-selftest-") as tmp:
        journal = Path(tmp) / "journal.json"
        journal.write_text(json.dumps({"run": {"peak_rss_mb": 1234.5}}), encoding="utf-8")
        written = measure(journal, out_dir=tmp)
        checks.append(("peak_rss найден вложенно", written["host_rss_peak_mb"]["value"] == 1234.5))
        checks.append(("имя факта — host_rss_peak_mb",
                       written["host_rss_peak_mb"]["fact"] == "host_rss_peak_mb"))
        empty = measure(Path(tmp) / "nope.json", out_dir=tmp)
        checks.append(("нет журнала → unverified", empty["host_rss_peak_mb"]["status"] == "unverified"))
    ok = all(p for _, p in checks)
    for label, passed in checks:
        print(f"[selftest] {'PASS' if passed else 'FAIL'}: {label}")
    print(f"[selftest] {'PASS' if ok else 'FAIL'}: wrap_rss")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="S-013: обёртка peak_rss_mb (ADR-037)")
    parser.add_argument("--journal", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--selftest", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.selftest:
        return run_selftest()
    written = measure(args.journal, out_dir=args.out_dir, device=args.device)
    for name, rec in written.items():
        print(f"S-013 {name} = {rec['value']} [{rec['status']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
