"""S-020 — внешний сэмплер стенда: мощность, частота, температура, утилизация (дельта C4).

Читает ``nvidia-smi --query-gpu=power.draw,clocks.sm,temperature.gpu,utilization.gpu
--format=csv,noheader,nounits`` в цикле (только чтение) и пишет медиану ряда с
числом проб. На исполнителе GB10 не запускается (AD-7): проверка — на фикстурах
(``--csv``) и CPU-бэкенде; недоступность ``nvidia-smi`` → ``unverified``.

Запуск::

    python3 -m tools.sensors.power_thermal --csv fixtures/nvidia-smi.txt
    python3 -m tools.sensors.power_thermal --samples 3 --interval 5
    python3 -m tools.sensors.power_thermal --selftest
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from ._common import config_subject, emit
from ._wrap import median

METHOD = "nvidia-smi --query-gpu=power.draw,clocks.sm,temperature.gpu,utilization.gpu (только чтение)"
QUERY = ["nvidia-smi", "--query-gpu=power.draw,clocks.sm,temperature.gpu,utilization.gpu",
         "--format=csv,noheader,nounits"]
FIELDS = ("power_draw_w", "sm_clock_mhz", "temperature_c", "sm_util_pct")


def parse_csv(text: str) -> dict[str, list[float]]:
    """Разбор строк nvidia-smi (по 4 числа в строке) в ряды по полям."""
    series: dict[str, list[float]] = {f: [] for f in FIELDS}
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < len(FIELDS):
            continue
        try:
            values = [float(p) for p in parts[: len(FIELDS)]]
        except ValueError:
            continue
        for field, value in zip(FIELDS, values):
            series[field].append(value)
    return series


def _collect(samples: int, interval: float) -> tuple[Optional[str], str]:
    if samples <= 0:
        return None, "число проб не задано"
    lines: list[str] = []
    try:
        for i in range(samples):
            proc = subprocess.run(QUERY, capture_output=True, text=True, timeout=15)
            if proc.returncode != 0:
                return None, f"nvidia-smi код {proc.returncode}"
            lines.append(proc.stdout)
            if i + 1 < samples:
                time.sleep(interval)
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"nvidia-smi недоступен: {type(exc).__name__}"
    return "\n".join(lines), ""


def measure(
    csv_text: Optional[str] = None,
    *,
    samples: int = 0,
    interval: float = 5.0,
    out_dir: Optional[str | Path] = None,
    device: Optional[str] = None,
) -> dict[str, Any]:
    note = ""
    if csv_text is None and samples > 0:
        collected, note = _collect(samples, interval)
        csv_text = collected
    series = parse_csv(csv_text or "")
    n = len(series[FIELDS[0]])
    subject = config_subject(device=device)
    written: dict[str, Any] = {}
    units = {"power_draw_w": "W", "sm_clock_mhz": "MHz", "temperature_c": "degC", "sm_util_pct": "pct"}
    for field in FIELDS:
        value = median(series[field])
        written[field] = emit(
            "S-020", field, value, unit=units[field], quality="measured", method=METHOD,
            subject=subject, out_dir=out_dir,
            status="ok" if value is not None else "unverified",
            note=(f"ряд, проб: {n}" if value is not None else (note or "нет данных nvidia-smi")),
        )
    return written


def run_selftest() -> int:
    import tempfile

    checks: list[tuple[str, bool]] = []
    sample = "25.6, 2412, 41, 96\n25.4, 2410, 42, 95\n"
    with tempfile.TemporaryDirectory(prefix="s020-selftest-") as tmp:
        written = measure(sample, out_dir=tmp)
        checks.append(("power median", written["power_draw_w"]["value"] == (25.6 + 25.4) / 2))
        checks.append(("sm_util median", written["sm_util_pct"]["value"] == 95.5))
        checks.append(("ряд помечен проб", "проб" in written["power_draw_w"]["note"]))
        empty = measure("", out_dir=tmp)
        checks.append(("нет данных → unverified", empty["power_draw_w"]["status"] == "unverified"))
    ok = all(p for _, p in checks)
    for label, passed in checks:
        print(f"[selftest] {'PASS' if passed else 'FAIL'}: {label}")
    print(f"[selftest] {'PASS' if ok else 'FAIL'}: power_thermal")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="S-020: сэмплер мощности/тепла (ADR-036)")
    parser.add_argument("--csv", dest="csv_path", default=None, help="сохранённый вывод nvidia-smi (фикстура)")
    parser.add_argument("--samples", type=int, default=0)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--selftest", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.selftest:
        return run_selftest()
    csv_text = Path(args.csv_path).read_text(encoding="utf-8") if args.csv_path else None
    written = measure(csv_text, samples=args.samples, interval=args.interval,
                      out_dir=args.out_dir, device=args.device)
    for name, rec in written.items():
        print(f"S-020 {name} = {rec['value']} [{rec['status']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
