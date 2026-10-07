"""S-017 — обёртка ``tools/check_sft_structure.py`` на каноническом наборе (дельта C3).

Запускает существующий структурный аудитор SFT-набора и переводит его отчёт в
факты: доли дефектов по классам (``sft_defect_shares``) и ``dataset_sha256``
набора. Набор недоступен → ``unverified``.

Запуск::

    python3 -m tools.sensors.wrap_sft_structure [--input PATH] [--out-dir DIR]
    python3 -m tools.sensors.wrap_sft_structure --selftest
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Optional

from ._common import REPO_ROOT, config_subject, emit, sha256_file
from ._wrap import run_tool

METHOD = "обёртка: tools/check_sft_structure.py --json на каноническом наборе"
DEFAULT_INPUT = REPO_ROOT / "data" / "datasets" / "sft_env_block_v1-mini.jsonl"


def measure(
    input_path: Optional[str | Path] = None,
    *,
    out_dir: Optional[str | Path] = None,
    device: Optional[str] = None,
) -> dict[str, Any]:
    path = Path(input_path) if input_path else DEFAULT_INPUT
    subject = config_subject(dataset_ref=str(path), device=device)
    shares = None
    dataset_sha = sha256_file(path) if path.is_file() else None
    note = ""
    if path.is_file():
        with tempfile.TemporaryDirectory(prefix="s017-sft-") as tmp:
            out = Path(tmp) / "sft.json"
            try:
                proc = run_tool(
                    "tools/check_sft_structure.py",
                    ["--input", str(path), "--json", str(out), "--quiet"],
                    timeout=300,
                )
            except Exception:  # noqa: BLE001
                proc = None
            report = None
            if out.is_file():
                try:
                    report = json.loads(out.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    report = None
            if isinstance(report, dict) and isinstance(report.get("classes"), dict):
                shares = {
                    name: (info.get("share") if isinstance(info, dict) else None)
                    for name, info in report["classes"].items()
                }
                note = f"verdict={report.get('verdict')}"
            elif proc is not None:
                note = f"check_sft_structure exit={proc.returncode}"
    else:
        note = f"набор недоступен: {path}"

    written: dict[str, Any] = {}
    written["sft_defect_shares"] = emit(
        "S-017", "sft_defect_shares", shares, unit="fraction", quality="wrapped",
        method=METHOD, subject=subject, out_dir=out_dir,
        status="ok" if shares is not None else "unverified",
        note=note if shares is not None else (note or "отчёт аудитора не получен"),
    )
    written["dataset_sha256"] = emit(
        "S-017", "dataset_sha256", dataset_sha, unit="sha256", quality="wrapped",
        method=METHOD, subject=subject, out_dir=out_dir,
        status="ok" if dataset_sha else "unverified",
        note="" if dataset_sha else f"набор недоступен: {path}",
    )
    return written


def run_selftest() -> int:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="s017-selftest-") as tmp:
        checks: list[tuple[str, bool]] = []
        missing = measure(Path("/nonexistent/sft.jsonl"), out_dir=tmp)
        checks.append(("нет набора → shares unverified", missing["sft_defect_shares"]["status"] == "unverified"))
        checks.append(("нет набора → sha unverified", missing["dataset_sha256"]["status"] == "unverified"))
        if DEFAULT_INPUT.is_file():
            written = measure(DEFAULT_INPUT, out_dir=tmp)
            checks.append(("канонический мини-набор: sha посчитан",
                           written["dataset_sha256"]["value"] == sha256_file(DEFAULT_INPUT)))
            checks.append(("классы дефектов получены", isinstance(written["sft_defect_shares"]["value"], dict)))
    ok = all(p for _, p in checks)
    for label, passed in checks:
        print(f"[selftest] {'PASS' if passed else 'FAIL'}: {label}")
    print(f"[selftest] {'PASS' if ok else 'FAIL'}: wrap_sft_structure")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="S-017: обёртка check_sft_structure (ADR-036)")
    parser.add_argument("--input", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--selftest", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.selftest:
        return run_selftest()
    written = measure(args.input, out_dir=args.out_dir, device=args.device)
    for name, rec in written.items():
        print(f"S-017 {name} = {rec['value']} [{rec['status']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
