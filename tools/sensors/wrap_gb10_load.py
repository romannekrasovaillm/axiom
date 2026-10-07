"""S-018 — обёртка ``tools/check_gb10_single_load.py`` (дельта C3).

Читает пиннутое чтение стража AD-7 (``evidence/gb10-single-load-*.txt``,
сенсор снят НА СТЕНДЕ) и/или запускает сам скрипт. Факты: ``model_loads_count``
и ``verdict_raw``. На чужом железе страж отказывается давать зелёный —
это ``unverified``, а не ноль и не подстановка.

Запуск::

    python3 -m tools.sensors.wrap_gb10_load [--evidence PATH] [--run] [--out-dir DIR]
    python3 -m tools.sensors.wrap_gb10_load --selftest
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Optional

from ._common import REPO_ROOT, config_subject, emit
from ._wrap import latest_glob, run_tool

METHOD = "обёртка: evidence/gb10-single-load-*.txt (пиннутое чтение стенда) + tools/check_gb10_single_load.py"
_LOADS_RE = re.compile(r"модельн\w* нагрузок:\s*(\d+)")


def _parse_evidence(path: Path) -> tuple[Optional[int], Optional[str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None, None
    loads = None
    match = _LOADS_RE.search(text)
    if match:
        loads = int(match.group(1))
    verdict = None
    for line in text.splitlines():
        stripped = line.strip()
        for prefix in ("Вывод:", "OK:", "НЕ ПРОВЕРЕНО:", "FAIL:"):
            if stripped.startswith(prefix):
                verdict = stripped
                break
        if verdict:
            break
    return loads, verdict


def measure(
    evidence_path: Optional[str | Path] = None,
    *,
    run: bool = False,
    out_dir: Optional[str | Path] = None,
    device: Optional[str] = None,
) -> dict[str, Any]:
    path = Path(evidence_path) if evidence_path else latest_glob("evidence/gb10-single-load-*.txt")
    loads = verdict = None
    note = ""
    run_ref = None
    if path is not None:
        run_ref = path.stem
        loads, verdict = _parse_evidence(path)
        note = f"источник: {path.name}"
    if loads is None and run:
        try:
            proc = run_tool("tools/check_gb10_single_load.py", [], timeout=20)
            verdict = (proc.stdout.strip() or proc.stderr.strip() or "").splitlines()[0] if (proc.stdout or proc.stderr) else None
            match = _LOADS_RE.search(proc.stdout or "")
            loads = int(match.group(1)) if match else None
            note = f"локальный прогон exit={proc.returncode}"
        except Exception:  # noqa: BLE001 — nvidia-smi может не отвечать
            note = "локальный страж не ответил (nvidia-smi) — стенд не проверялся"
    if loads is None and not note:
        note = "нет пиннутого чтения стенда (evidence/gb10-single-load-*.txt)"

    subject = config_subject(run_ref=run_ref, device=device)
    written: dict[str, Any] = {}
    written["model_loads_count"] = emit(
        "S-018", "model_loads_count", loads, unit="count", quality="wrapped", method=METHOD,
        subject=subject, out_dir=out_dir,
        status="ok" if loads is not None else "unverified", note=note,
    )
    written["verdict_raw"] = emit(
        "S-018", "verdict_raw", verdict, unit="text", quality="wrapped", method=METHOD,
        subject=subject, out_dir=out_dir,
        status="ok" if verdict else "unverified",
        note="" if verdict else note,
    )
    return written


def run_selftest() -> int:
    import tempfile

    checks: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory(prefix="s018-selftest-") as tmp:
        ev = Path(tmp) / "gb10-single-load-x.txt"
        ev.write_text(
            "Вывод:     OK: одна модельная нагрузка за раз — подтверждено "
            "(модельных нагрузок: 0)\n",
            encoding="utf-8",
        )
        written = measure(ev, out_dir=tmp)
        checks.append(("loads распарсены", written["model_loads_count"]["value"] == 0))
        checks.append(("verdict_raw распарсен", "OK" in (written["verdict_raw"]["value"] or "")))
        missing = measure(Path(tmp) / "nope.txt", out_dir=tmp)
        checks.append(("нет источника → unverified", missing["model_loads_count"]["status"] == "unverified"))
    ok = all(p for _, p in checks)
    for label, passed in checks:
        print(f"[selftest] {'PASS' if passed else 'FAIL'}: {label}")
    print(f"[selftest] {'PASS' if ok else 'FAIL'}: wrap_gb10_load")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="S-018: обёртка check_gb10_single_load (ADR-036)")
    parser.add_argument("--evidence", default=None)
    parser.add_argument("--run", action="store_true", help="дополнительно запустить локальный страж")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--selftest", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.selftest:
        return run_selftest()
    written = measure(args.evidence, run=args.run, out_dir=args.out_dir, device=args.device)
    for name, rec in written.items():
        print(f"S-018 {name} = {rec['value']} [{rec['status']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
