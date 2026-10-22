"""S-008 — время вердикта arch-ml на CPU (дельта C2).

Что меряет: ``verdict_seconds_p50``/``p90`` времени вердикта ``env.verifier``
(бинарь ``arch-ml``) на CPU по выборке восстановленных задач — вход стоимости
роллаутов RL (R3 ~13 с). Числа — фактические на этой машине; на загруженном
хосте они выше, и это тоже факт (в ``note`` — число проб).

Запуск::

    python3 -m tools.sensors.verifier_latency [--n 3] [--level L1] [--out-dir DIR]
    python3 -m tools.sensors.verifier_latency --selftest
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from ._common import REPO_ROOT, config_subject, emit
from ._verdict import arch_ml_ready, prepare_task, run_verdict

METHOD = "wall-clock time env.verifier.verify(arch-ml) по восстановленным задачам L1, CPU"


def percentile(values: list[float], q: float) -> Optional[float]:
    """Линейная интерполяция перцентиля (детерминированная, без numpy)."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def measure(
    *,
    clean: str | Path = REPO_ROOT,
    n: int = 3,
    level: str = "L1",
    out_dir: Optional[str | Path] = None,
    device: Optional[str] = None,
) -> dict[str, Any]:
    clean_path = Path(clean)
    subject = config_subject(device=device)
    samples: list[float] = []
    if arch_ml_ready() and n > 0:
        with tempfile.TemporaryDirectory(prefix="s008-latency-") as tmp:
            root = Path(tmp)
            for index in range(n):
                ws = root / f"{level}-{index:02d}"
                prepare_task(clean_path, ws, level, index)
                started = time.perf_counter()
                run_verdict(ws, clean_path)
                samples.append(time.perf_counter() - started)

    p50 = percentile(samples, 0.5)
    p90 = percentile(samples, 0.9)
    note = f"проб: {len(samples)}"
    if not samples:
        note = "arch-ml недоступен или выборка пуста — вердикты не снимались"
    written: dict[str, Any] = {}
    for name, value in (("verdict_seconds_p50", p50), ("verdict_seconds_p90", p90)):
        written[name] = emit(
            "S-008", name, round(value, 4) if value is not None else None, unit="seconds",
            quality="measured", method=METHOD, subject=subject, out_dir=out_dir,
            status="ok" if value is not None else "unverified",
            note=note if value is not None else note,
        )
    return written


def run_selftest() -> int:
    import tempfile

    checks: list[tuple[str, bool]] = [
        ("p50 нечётного ряда", percentile([1.0, 3.0, 2.0], 0.5) == 2.0),
        ("p50 чётного ряда — интерполяция", abs(percentile([1.0, 2.0, 3.0, 4.0], 0.5) - 2.5) < 1e-9),
        ("p90 на одном значении", percentile([5.0], 0.9) == 5.0),
        ("пустой ряд → None", percentile([], 0.5) is None),
    ]
    with tempfile.TemporaryDirectory(prefix="s008-selftest-") as tmp:
        empty = measure(n=0, out_dir=tmp)
    checks.append(("n=0 → unverified", empty["verdict_seconds_p50"]["status"] == "unverified"))
    ok = all(p for _, p in checks)
    for label, passed in checks:
        print(f"[selftest] {'PASS' if passed else 'FAIL'}: {label}")
    print(f"[selftest] {'PASS' if ok else 'FAIL'}: verifier_latency")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="S-008: время вердикта arch-ml (ADR-037)")
    parser.add_argument("--clean", default=str(REPO_ROOT))
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--level", default="L1")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--selftest", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.selftest:
        return run_selftest()
    written = measure(clean=args.clean, n=args.n, level=args.level, out_dir=args.out_dir, device=args.device)
    for name, rec in written.items():
        print(f"S-008 {name} = {rec['value']} [{rec['status']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
