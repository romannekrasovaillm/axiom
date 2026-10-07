"""S-029 — реплика помещается в устройство (производный факт, дельта C6).

``replica_fits_device`` = S-003 (``replica_bytes_total``) против объёма памяти
устройства, взятого из **факта**, а не из константы. Пока факта ёмкости нет,
результат ``unverified`` (константу подставлять запрещено); ёмкость передаётся
ссылкой из факта через ``--device-bytes`` (или ``--device-fact S:fact``).

Запуск::

    python3 -m tools.sensors.derived_fit [--device-bytes N] [--out-dir DIR]
    python3 -m tools.sensors.derived_fit --selftest
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

from ._common import config_subject, emit
from ._derive import fact_ref, missing

METHOD = "replica_fits_device = replica_bytes_total (S-003) < device capacity (из факта)"


def measure(
    *,
    device_bytes: Optional[int] = None,
    device_fact: Optional[str] = None,
    out_dir: Optional[str | Path] = None,
    device: Optional[str] = None,
) -> dict[str, Any]:
    ref_replica = fact_ref("S-003", "replica_bytes_total", out_dir=out_dir)
    ref_device = None
    if device_bytes is None and device_fact:
        sensor, _, fact = device_fact.partition(":")
        ref_device = fact_ref(sensor, fact, out_dir=out_dir)
        if ref_device is not None and ref_device.get("value") is not None:
            device_bytes = int(ref_device["value"])

    value = None
    inputs = [r for r in (ref_replica, ref_device) if r is not None]
    if ref_replica is None or ref_replica.get("value") is None:
        note = "нет S-003:replica_bytes_total"
    elif device_bytes is None:
        note = "ёмкость устройства не пришла фактом (--device-bytes/--device-fact) — константа не подставляется"
    else:
        value = int(ref_replica["value"]) < int(device_bytes)
        note = ""
        inputs.append({"device_bytes": int(device_bytes)})
    subject = config_subject(device=device)
    written = emit(
        "S-029", "replica_fits_device", value, unit="bool", quality="derived", method=METHOD,
        subject=subject, out_dir=out_dir, inputs=inputs,
        status="ok" if value is not None else "unverified", note=note,
    )
    return {"replica_fits_device": written}


def run_selftest() -> int:
    import tempfile

    from .fact import write_fact
    from .subject import build_subject

    checks: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory(prefix="s029-selftest-") as tmp:
        subject = build_subject(repo_root=Path(tmp), git_sha="a" * 40, dirty=False, device="cpu")
        write_fact("S-003", "replica_bytes_total", 8_000_000_000, unit="bytes",
                   quality="measured", method="fixture", subject=subject, out_dir=tmp)
        no_cap = measure(out_dir=tmp)
        checks.append(("нет ёмкости → unverified", no_cap["replica_fits_device"]["status"] == "unverified"))
        fits = measure(device_bytes=80 * 1024**3, out_dir=tmp)
        checks.append(("8 ГБ < 80 ГиБ → True", fits["replica_fits_device"]["value"] is True))
        doesnt = measure(device_bytes=4 * 1024**3, out_dir=tmp)
        checks.append(("8 ГБ < 4 ГиБ → False", doesnt["replica_fits_device"]["value"] is False))
    ok = all(p for _, p in checks)
    for label, passed in checks:
        print(f"[selftest] {'PASS' if passed else 'FAIL'}: {label}")
    print(f"[selftest] {'PASS' if ok else 'FAIL'}: derived_fit")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="S-029: производная вместимость реплики (ADR-037)")
    parser.add_argument("--device-bytes", type=int, default=None)
    parser.add_argument("--device-fact", default=None, help="ссылка S-021:device_mem_used_mb и т.п.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--selftest", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.selftest:
        return run_selftest()
    written = measure(device_bytes=args.device_bytes, device_fact=args.device_fact,
                      out_dir=args.out_dir, device=args.device)
    for name, rec in written.items():
        print(f"S-029 {name} = {rec['value']} [{rec['status']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
