"""S-016 — обёртка ``tools/check_eval_leak.py`` на пиннутом holdout (дельта C3).

Запускает существующий аудитор утечки на пиннутых eval-наборах против
обучающих источников и переводит его ``--json``-отчёт в факты ``overlap_count``
и ``holdout_sha256``. Наборов нет → ``unverified`` (вердикт по числу не
выносится без чистого набора).

Запуск::

    python3 -m tools.sensors.wrap_eval_leak [--eval P] [--source P]... [--out-dir DIR]
    python3 -m tools.sensors.wrap_eval_leak --selftest
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Optional

from ._common import REPO_ROOT, config_subject, emit, sha256_file
from ._wrap import run_tool

METHOD = "обёртка: tools/check_eval_leak.py --json на пиннутом holdout и обучающих источниках"


def _default_eval() -> Optional[Path]:
    for pattern in ("evidence/**/eval_holdout*.jsonl", "data/**/eval_holdout*.jsonl"):
        matches = sorted(REPO_ROOT.glob(pattern))
        if matches:
            return matches[0]
    return None


def measure(
    eval_path: Optional[str | Path] = None,
    sources: Optional[list[str | Path]] = None,
    *,
    out_dir: Optional[str | Path] = None,
    device: Optional[str] = None,
) -> dict[str, Any]:
    eval_file = Path(eval_path) if eval_path else _default_eval()
    subject = config_subject(dataset_ref=str(eval_file) if eval_file else None, device=device)
    written: dict[str, Any] = {}
    holdout_sha = sha256_file(eval_file) if eval_file else None
    overlap = None
    note = "eval-набор не найден — вердикт по числу не выносится"
    if eval_file and sources:
        with tempfile.TemporaryDirectory(prefix="s016-leak-") as tmp:
            out = Path(tmp) / "leak.json"
            args = ["--eval", str(eval_file)]
            for src in sources:
                args += ["--source", str(src)]
            args += ["--json", str(out), "--quiet"]
            try:
                proc = run_tool("tools/check_eval_leak.py", args, timeout=120)
            except Exception:  # noqa: BLE001
                proc = None
            report = None
            if out.is_file():
                try:
                    report = json.loads(out.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    report = None
            if isinstance(report, dict):
                sources_report = report.get("sources") or []
                overlap = 0
                for entry in sources_report:
                    if isinstance(entry, dict) and entry.get("overlap"):
                        overlap += 1
                note = f"check_eval_leak verdict={report.get('verdict')}"
            elif proc is not None:
                note = f"check_eval_leak exit={proc.returncode}"
    elif eval_file and not sources:
        note = "обучающие источники не заданы — источники обязаны быть пиннуты"

    written["overlap_count"] = emit(
        "S-016", "overlap_count", overlap, unit="count", quality="wrapped", method=METHOD,
        subject=subject, out_dir=out_dir,
        status="ok" if overlap is not None else "unverified", note=note,
    )
    written["holdout_sha256"] = emit(
        "S-016", "holdout_sha256", holdout_sha, unit="sha256", quality="wrapped", method=METHOD,
        subject=subject, out_dir=out_dir,
        status="ok" if holdout_sha else "unverified",
        note="" if holdout_sha else "holdout-файл отсутствует",
    )
    return written


def run_selftest() -> int:
    with tempfile.TemporaryDirectory(prefix="s016-selftest-") as tmp:
        checks: list[tuple[str, bool]] = []
        written = measure(Path(tmp) / "nope.jsonl", out_dir=tmp)
        checks.append(("нет holdout → unverified", written["holdout_sha256"]["status"] == "unverified"))
        checks.append(("нет holdout → overlap unverified", written["overlap_count"]["status"] == "unverified"))
        ok = all(p for _, p in checks)
        for label, passed in checks:
            print(f"[selftest] {'PASS' if passed else 'FAIL'}: {label}")
        print(f"[selftest] {'PASS' if ok else 'FAIL'}: wrap_eval_leak")
        return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="S-016: обёртка check_eval_leak (ADR-036)")
    parser.add_argument("--eval", dest="eval_path", default=None)
    parser.add_argument("--source", dest="sources", action="append", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--selftest", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.selftest:
        return run_selftest()
    written = measure(args.eval_path, args.sources, out_dir=args.out_dir, device=args.device)
    for name, rec in written.items():
        print(f"S-016 {name} = {rec['value']} [{rec['status']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
