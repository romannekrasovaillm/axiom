"""S-025 — измеренный пик bf16-матмула (микробенч, дельта C5).

Микробенч bf16-матмула разных размеров с прогревом и медианой. Исполняется
**только в окне GB10 или на арендованной машине**; на исполнителе не
запускается (AD-7) — тогда пишется ``unverified`` с причиной, а объявленный пик
в факт не подставляется (иначе MFU считался бы от декларации).

Запуск::

    python3 -m tools.sensors.peak_tflops_bench --allow-device   # только в окне
    python3 -m tools.sensors.peak_tflops_bench --selftest
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

from ._common import config_subject, emit
from .inproc import PhaseTimer
from ._wrap import median

METHOD = "микробенч bf16-матмула (прогрев + медиана), издержки отсечены jax.block_until_ready"
SIZES = (1024, 2048, 4096, 8192)


def tflops_from_seconds(n: int, seconds: float) -> Optional[float]:
    """2·n³ FLOP на матмул n×n×n → TFLOP/s (None при неположительном времени)."""
    if seconds <= 0:
        return None
    return 2.0 * (n ** 3) / seconds / 1e12


def _bench(sizes=SIZES, *, warmup: int = 2, repeats: int = 5) -> Optional[float]:
    try:
        import jax
        import jax.numpy as jnp
    except Exception:  # noqa: BLE001
        return None
    best = None
    for n in sizes:
        try:
            x = jnp.ones((n, n), dtype=jnp.bfloat16)
            y = jnp.ones((n, n), dtype=jnp.bfloat16)

            def run():
                return jnp.dot(x, y)

            for _ in range(warmup):
                jax.block_until_ready(run())
            samples = []
            for _ in range(repeats):
                import time

                started = time.perf_counter()
                jax.block_until_ready(run())
                samples.append(time.perf_counter() - started)
            value = tflops_from_seconds(n, median(samples) or 0.0)
            if value is not None:
                best = value if best is None else max(best, value)
        except Exception:  # noqa: BLE001 — размер может не влезть: пропускаем
            continue
    return best


def measure(
    *,
    allow_device: bool = False,
    out_dir: Optional[str | Path] = None,
    device: Optional[str] = None,
) -> dict[str, Any]:
    subject = config_subject(device=device)
    value = None
    note = "микробенч исполняется только в окне GB10/на аренде (AD-7) — на исполнителе не запускается"
    if allow_device:
        value = _bench()
        note = "" if value is not None else "микробенч не дал числа (нет JAX/устройства)"
    written = emit(
        "S-025", "measured_peak_tflops_bf16", value, unit="TFLOP/s", quality="measured",
        method=METHOD, subject=subject, out_dir=out_dir,
        status="ok" if value is not None else "unverified", note=note,
    )
    return {"measured_peak_tflops_bf16": written}


def run_selftest() -> int:
    import tempfile

    checks: list[tuple[str, bool]] = [
        ("2·n³/t корректно", abs(tflops_from_seconds(1024, 0.01) - 2 * 1024**3 / 0.01 / 1e12) < 1e-9),
        ("нулевое время → None", tflops_from_seconds(1024, 0.0) is None),
        ("overhead_pct формулы", abs(PhaseTimer.overhead_pct(1.0, 1.1) - 10.0) < 1e-9),
    ]
    with tempfile.TemporaryDirectory(prefix="s025-selftest-") as tmp:
        written = measure(allow_device=False, out_dir=tmp)
    checks.append(("без окна → unverified",
                   written["measured_peak_tflops_bf16"]["status"] == "unverified"))
    checks.append(("объявленный пик не подставлен",
                   written["measured_peak_tflops_bf16"]["value"] is None))
    ok = all(p for _, p in checks)
    for label, passed in checks:
        print(f"[selftest] {'PASS' if passed else 'FAIL'}: {label}")
    print(f"[selftest] {'PASS' if ok else 'FAIL'}: peak_tflops_bench")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="S-025: микробенч пика bf16 (ADR-036)")
    parser.add_argument("--allow-device", action="store_true",
                        help="разрешить микробенч (только в окне GB10/на аренде)")
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
        print(f"S-025 {name} = {rec['value']} [{rec['status']}] {rec['note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
