"""S-021 — память устройства по процессу (внешний сэмплер, дельта C4).

Читает ``nvidia-smi --query-compute-apps=pid,used_memory --format=csv``. На GB10
с общей памятью per-process ``used_memory`` может не отдаваться — тогда запись
``unverified`` с причиной «unified memory: per-process used_memory не
поддерживается», без подстановок. Фикстуры — ``--csv``.

Запуск::

    python3 -m tools.sensors.device_memory_external --csv fixtures/compute-apps.txt
    python3 -m tools.sensors.device_memory_external --selftest
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any, Optional

from ._common import config_subject, emit

METHOD = "nvidia-smi --query-compute-apps=pid,used_memory --format=csv"
QUERY = ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]


def parse_compute_apps(text: str) -> Optional[float]:
    """Сумма used_memory (МиБ) по процессам. ``None`` — данных нет/непригодны."""
    total = 0.0
    seen = False
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        raw = parts[1]
        if raw.lower() in ("[not supported]", "n/a", "[n/a]"):
            return None
        try:
            total += float(raw)
            seen = True
        except ValueError:
            continue
    return total if seen else None


def measure(
    csv_text: Optional[str] = None,
    *,
    run: bool = False,
    out_dir: Optional[str | Path] = None,
    device: Optional[str] = None,
) -> dict[str, Any]:
    note = ""
    if csv_text is None and run:
        try:
            proc = subprocess.run(QUERY, capture_output=True, text=True, timeout=15)
            if proc.returncode != 0:
                csv_text = None
                note = f"nvidia-smi код {proc.returncode}: {proc.stderr.strip()[:120]}"
            else:
                csv_text = proc.stdout
        except (OSError, subprocess.SubprocessError) as exc:
            note = f"nvidia-smi недоступен: {type(exc).__name__}"
    value = parse_compute_apps(csv_text or "")
    if value is None and not note:
        note = (
            "unified memory: per-process used_memory не поддерживается "
            "(значение не подставляется)"
        )
    subject = config_subject(device=device)
    written = emit(
        "S-021", "device_mem_used_mb", value, unit="MiB", quality="measured", method=METHOD,
        subject=subject, out_dir=out_dir,
        status="ok" if value is not None else "unverified", note=note,
    )
    return {"device_mem_used_mb": written}


def run_selftest() -> int:
    import tempfile

    checks: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory(prefix="s021-selftest-") as tmp:
        written = measure("1234, 2048\n1235, 1024\n", out_dir=tmp)
        checks.append(("сумма по процессам", written["device_mem_used_mb"]["value"] == 3072.0))
        unsupported = measure("1234, [Not supported]\n", out_dir=tmp)
        checks.append(("unified memory → unverified",
                       unsupported["device_mem_used_mb"]["status"] == "unverified"
                       and "unified" in unsupported["device_mem_used_mb"]["note"]))
        empty = measure("", out_dir=tmp)
        checks.append(("пустой вход → unverified", empty["device_mem_used_mb"]["status"] == "unverified"))
    ok = all(p for _, p in checks)
    for label, passed in checks:
        print(f"[selftest] {'PASS' if passed else 'FAIL'}: {label}")
    print(f"[selftest] {'PASS' if ok else 'FAIL'}: device_memory_external")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="S-021: память устройства по процессу (ADR-036)")
    parser.add_argument("--csv", dest="csv_path", default=None)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--selftest", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.selftest:
        return run_selftest()
    csv_text = Path(args.csv_path).read_text(encoding="utf-8") if args.csv_path else None
    written = measure(csv_text, run=args.run, out_dir=args.out_dir, device=args.device)
    for name, rec in written.items():
        print(f"S-021 {name} = {rec['value']} [{rec['status']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
