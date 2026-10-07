"""S-028 — ETA прогона для заданной D и машины (производный факт, дельта C6).

``eta_hours`` = D_токенов / (tok/s · 3600), где tok/s — из S-012 (факт на
совпадающем конфиге), D — из утверждения (аргумент). Основание для «7–11 ч с
ядром WY/UT» и сметы pretrain-l3. Нет S-012 → ``unverified``.

Запуск::

    python3 -m tools.sensors.derived_eta [--d-tokens 17840000000] [--run-ref REF] [--out-dir DIR]
    python3 -m tools.sensors.derived_eta --selftest
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

from ._common import config_subject, emit
from ._derive import fact_ref, missing

METHOD = "eta_hours = D / (tok_s_median_window · 3600); tok/s из S-012, D из аргумента"
DEFAULT_D = 17_840_000_000


def measure(
    *,
    d_tokens: int = DEFAULT_D,
    run_ref: Optional[str] = None,
    out_dir: Optional[str | Path] = None,
    device: Optional[str] = None,
) -> dict[str, Any]:
    ref_tok = fact_ref("S-012", "tok_s_median_window", out_dir=out_dir,
                       subject_filter=["run_ref"] if run_ref else None,
                       subject={"run_ref": run_ref} if run_ref else None)
    value = None
    gaps = missing(ref_tok)
    if not gaps and d_tokens > 0:
        rate = float(ref_tok["value"])
        value = (d_tokens / rate / 3600.0) if rate > 0 else None
    inputs = [r for r in (ref_tok,) if r is not None]
    inputs.append({"d_tokens": d_tokens})
    note = "" if value is not None else f"нет S-012:tok_s_median_window для run_ref={run_ref!r}"
    written = emit(
        "S-028", "eta_hours", value, unit="hours", quality="derived", method=METHOD,
        subject=config_subject(run_ref=run_ref, device=device), out_dir=out_dir,
        inputs=inputs, status="ok" if value is not None else "unverified", note=note,
    )
    return {"eta_hours": written}


def run_selftest() -> int:
    import tempfile

    from .fact import write_fact
    from .subject import build_subject

    checks: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory(prefix="s028-selftest-") as tmp:
        subject = build_subject(repo_root=Path(tmp), git_sha="a" * 40, dirty=False, device="cpu")
        missing_eta = measure(d_tokens=1000, out_dir=tmp)
        checks.append(("нет S-012 → unverified", missing_eta["eta_hours"]["status"] == "unverified"))
        write_fact("S-012", "tok_s_median_window", 1000.0, unit="tok_s", quality="wrapped",
                   method="fixture", subject=subject, out_dir=tmp)
        written = measure(d_tokens=3_600_000, out_dir=tmp)
        checks.append(("eta = D/(tok_s·3600)", abs(written["eta_hours"]["value"] - 1.0) < 1e-9))
    ok = all(p for _, p in checks)
    for label, passed in checks:
        print(f"[selftest] {'PASS' if passed else 'FAIL'}: {label}")
    print(f"[selftest] {'PASS' if ok else 'FAIL'}: derived_eta")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="S-028: производная ETA (ADR-036)")
    parser.add_argument("--d-tokens", type=int, default=DEFAULT_D)
    parser.add_argument("--run-ref", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--selftest", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.selftest:
        return run_selftest()
    written = measure(d_tokens=args.d_tokens, run_ref=args.run_ref, out_dir=args.out_dir, device=args.device)
    for name, rec in written.items():
        print(f"S-028 {name} = {rec['value']} [{rec['status']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
