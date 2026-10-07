"""S-009 — детерминизм вердикта (дельта C2).

Что меряет: два вердикта **одного** рабочего каталога → совпадение sha256
канонизированного JSON (относительные пути, сортировка). Это факт под
утверждение «вердикт машинонезависим / детерминирован» (ENVIRONMENT R-1, AD-4).

Запуск::

    python3 -m tools.sensors.verifier_determinism [--level L1] [--index 0] [--out-dir DIR]
    python3 -m tools.sensors.verifier_determinism --selftest
"""

from __future__ import annotations

import argparse
import hashlib
import tempfile
from pathlib import Path
from typing import Any, Optional

from ._common import REPO_ROOT, config_subject, emit
from ._verdict import arch_ml_ready, canonical_verdict, prepare_task, run_verdict

METHOD = "два env.verifier.verify(arch-ml) одного каталога → sha256 канонизированного вердикта"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def measure(
    *,
    clean: str | Path = REPO_ROOT,
    level: str = "L1",
    index: int = 0,
    out_dir: Optional[str | Path] = None,
    device: Optional[str] = None,
) -> dict[str, Any]:
    clean_path = Path(clean)
    subject = config_subject(device=device)
    value: Optional[bool] = None
    note = "arch-ml недоступен — вердикты не снимались"
    hashes: tuple[str, str] | None = None
    if arch_ml_ready():
        with tempfile.TemporaryDirectory(prefix="s009-determinism-") as tmp:
            ws = Path(tmp) / f"{level}-{index:02d}"
            prepare_task(clean_path, ws, level, index)
            _v1, payload1 = run_verdict(ws, clean_path)
            _v2, payload2 = run_verdict(ws, clean_path)
            h1, h2 = _sha(payload1), _sha(payload2)
            hashes = (h1, h2)
            value = h1 == h2
            note = f"sha256 #1={h1[:16]}…, #2={h2[:16]}…"
    written = emit(
        "S-009", "verdict_identical", value, unit="bool", quality="measured",
        method=METHOD, subject=subject, out_dir=out_dir,
        status="ok" if value is not None else "unverified", note=note,
    )
    result = {"verdict_identical": written}
    if hashes is not None:
        written["_hashes"] = list(hashes)
    return result


def run_selftest() -> int:
    import os
    import tempfile

    from env.verifier import GateResult, Verdict

    def _fake_verdict() -> Verdict:
        return Verdict(
            passed=True, objective_kind="restore-gates",
            fitness=GateResult(True), spine=GateResult(True), trace=GateResult(True),
            hidden=None, tests_passed=True,
            violations=frozenset({("C-007", "docs/x.md")}),
        )

    ws = Path("/tmp/ws")
    p1 = canonical_verdict(_fake_verdict(), ws)
    p2 = canonical_verdict(_fake_verdict(), ws)
    checks: list[tuple[str, bool]] = [
        ("канонизация вердикта детерминирована", p1 == p2),
        ("violations отсортированы и относительны", "docs/x.md" in p1),
        ("sha256 payload стабилен", _sha(p1) == _sha(p2)),
    ]
    saved = os.environ.get("ENV_ARCH_ML_BIN")
    try:
        os.environ["ENV_ARCH_ML_BIN"] = "/nonexistent/arch-ml"
        with tempfile.TemporaryDirectory(prefix="s009-selftest-") as tmp:
            empty = measure(out_dir=tmp)
        checks.append(("нет arch-ml → unverified", empty["verdict_identical"]["status"] == "unverified"))
    finally:
        if saved is None:
            os.environ.pop("ENV_ARCH_ML_BIN", None)
        else:
            os.environ["ENV_ARCH_ML_BIN"] = saved
    ok = all(p for _, p in checks)
    for label, passed in checks:
        print(f"[selftest] {'PASS' if passed else 'FAIL'}: {label}")
    print(f"[selftest] {'PASS' if ok else 'FAIL'}: verifier_determinism")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="S-009: детерминизм вердикта (ADR-036)")
    parser.add_argument("--clean", default=str(REPO_ROOT))
    parser.add_argument("--level", default="L1")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--selftest", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.selftest:
        return run_selftest()
    written = measure(clean=args.clean, level=args.level, index=args.index, out_dir=args.out_dir, device=args.device)
    for name, rec in written.items():
        print(f"S-009 {name} = {rec['value']} [{rec['status']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
