"""S-027 — MFU по измеренному пику (производный факт, дельта C6).

``mfu_measured = 6 · N_active · tok/s / measured_peak``. Из S-002
(``params_active``) + S-012 (``tok_s_median_window``) + S-025
(``measured_peak_tflops_bf16``). Нет S-025 → ``unverified``: объявленный пик
подставлять запрещено (иначе MFU считался бы от декларации — та самая ошибка
``peak_declared_not_measured``).

Запуск::

    python3 -m tools.sensors.derived_mfu [--out-dir DIR]
    python3 -m tools.sensors.derived_mfu --selftest
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

from ._common import config_subject, emit
from ._derive import fact_ref, missing

METHOD = "mfu_measured = 6·N_active·tok/s / measured_peak_tflops_bf16 (S-002 × S-012 × S-025)"


def measure(*, out_dir: Optional[str | Path] = None, device: Optional[str] = None) -> dict[str, Any]:
    ref_active = fact_ref("S-002", "params_active", out_dir=out_dir)
    ref_tok = fact_ref("S-012", "tok_s_median_window", out_dir=out_dir)
    ref_peak = fact_ref("S-025", "measured_peak_tflops_bf16", out_dir=out_dir)

    value = None
    gaps = []
    if ref_active is None or ref_active.get("value") is None:
        gaps.append("S-002:params_active")
    if ref_tok is None or ref_tok.get("value") is None:
        gaps.append("S-012:tok_s_median_window")
    if ref_peak is None or ref_peak.get("value") is None:
        gaps.append("S-025:measured_peak_tflops_bf16")
    if not gaps:
        flops = 6.0 * float(ref_active["value"]) * float(ref_tok["value"])
        peak_flops = float(ref_peak["value"]) * 1e12
        value = flops / peak_flops if peak_flops > 0 else None
    inputs = [r for r in (ref_active, ref_tok, ref_peak) if r is not None]
    note = "" if value is not None else f"нет входов: {', '.join(gaps)} (объявленный пик не подставляется)"
    written = emit(
        "S-027", "mfu_measured", value, unit="fraction", quality="derived", method=METHOD,
        subject=config_subject(device=device), out_dir=out_dir, inputs=inputs,
        status="ok" if value is not None else "unverified", note=note,
    )
    return {"mfu_measured": written}


def run_selftest() -> int:
    import tempfile

    from .fact import write_fact
    from .subject import build_subject

    checks: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory(prefix="s027-selftest-") as tmp:
        subject = build_subject(repo_root=Path(tmp), git_sha="a" * 40, dirty=False, device="cpu")
        write_fact("S-002", "params_active", 500_000_000, unit="count", quality="measured",
                   method="fixture", subject=subject, out_dir=tmp)
        write_fact("S-012", "tok_s_median_window", 800.0, unit="tok_s", quality="wrapped",
                   method="fixture", subject=subject, out_dir=tmp)
        no_peak = measure(out_dir=tmp)
        checks.append(("нет S-025 → unverified",
                       no_peak["mfu_measured"]["status"] == "unverified"))
        write_fact("S-025", "measured_peak_tflops_bf16", 200.0, unit="TFLOP/s", quality="measured",
                   method="fixture", subject=subject, out_dir=tmp)
        with_peak = measure(out_dir=tmp)
        expected = 6.0 * 500_000_000 * 800.0 / (200.0 * 1e12)
        checks.append(("MFU посчитан по измеренному пику",
                       abs(with_peak["mfu_measured"]["value"] - expected) < 1e-12))
        checks.append(("inputs ссылаются на исходные записи",
                       all(r.get("sha256") for r in with_peak["mfu_measured"]["inputs"])))
    ok = all(p for _, p in checks)
    for label, passed in checks:
        print(f"[selftest] {'PASS' if passed else 'FAIL'}: {label}")
    print(f"[selftest] {'PASS' if ok else 'FAIL'}: derived_mfu")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="S-027: производный MFU (ADR-036)")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--selftest", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.selftest:
        return run_selftest()
    written = measure(out_dir=args.out_dir, device=args.device)
    for name, rec in written.items():
        print(f"S-027 {name} = {rec['value']} [{rec['status']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
