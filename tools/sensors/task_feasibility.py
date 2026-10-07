"""S-007 — выполнимость задач лесенки (дельта C2).

Что меряет: для выборки N задач на уровень берётся порча, восстановимая четырьмя
инструментами харнесса §13 (``list_files``/``read_file``/``edit_file``/``finish``),
применяется **эталонная починка** (реальная вставка текста через ``edit_file``) и
снимается вердикт ``env.verifier``. Доля задач, дошедших до ``pass``, — это
``feasibility_rate``; не дошедшие — список ``infeasible_tasks`` (класс DEF-1
«задача невыполнима by construction»).

Запуск::

    python3 -m tools.sensors.task_feasibility [--n 20] [--levels L0,L1,L2,L3] [--out-dir DIR]
    python3 -m tools.sensors.task_feasibility --selftest
"""

from __future__ import annotations

import argparse
import hashlib
import tempfile
from pathlib import Path
from typing import Any, Optional

from ._common import REPO_ROOT, config_subject, emit, safe_exists
from ._verdict import arch_ml_ready, run_verdict

METHOD = (
    "copy_case_snapshot + plan_repairable(порча) + эталонная вставка через "
    "WorkspaceTools.edit_file (4 инструмента §13) + env.verifier.verify(arch-ml)"
)
LEVELS = ("L0", "L1", "L2", "L3")


def _task_seed(level: str, index: int) -> int:
    digest = hashlib.sha256(f"feasibility:{level}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _repair(ws: Path, clean: Path, damages) -> None:
    import sys

    if str(REPO_ROOT / "tools") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "tools"))
    from rollout_harness import WorkspaceTools

    from env.sft_block import _insertion_edit

    from ._verdict import _NoopVerifier

    tools = WorkspaceTools(ws, _NoopVerifier())
    for damage in damages:
        damaged_text = (ws / damage.file).read_text(encoding="utf-8")
        clean_text = (clean / damage.file).read_text(encoding="utf-8")
        old, new = _insertion_edit(damaged_text, clean_text)
        tools.edit_file(damage.file, old, new)


def measure(
    *,
    clean: str | Path = REPO_ROOT,
    n: int = 20,
    levels: tuple[str, ...] = LEVELS,
    out_dir: Optional[str | Path] = None,
    device: Optional[str] = None,
    run_verdicts: bool = True,
) -> dict[str, Any]:
    from env import corruption
    from env.sft_block import plan_repairable

    clean_path = Path(clean)
    subject = config_subject(device=device)
    total = 0
    feasible = 0
    infeasible: list[str] = []
    not_run: list[str] = []
    with tempfile.TemporaryDirectory(prefix="s007-feasibility-") as tmp:
        root = Path(tmp)
        for level in levels:
            for index in range(n):
                task_id = f"{level}-{index:02d}"
                ws = root / task_id
                seed = _task_seed(level, index)
                try:
                    damages = plan_repairable(clean_path, seed, level)
                except ValueError:
                    total += 1
                    infeasible.append(task_id)
                    continue
                # Порча на свежем снапшоте.
                from env.util import copy_case_snapshot

                copy_case_snapshot(clean_path, ws)
                for d in damages:
                    corruption.apply_damage(ws, d)
                _repair(ws, clean_path, damages)
                total += 1
                if not run_verdicts:
                    not_run.append(task_id)
                    continue
                try:
                    verdict, _payload = run_verdict(ws, clean_path)
                except Exception:  # noqa: BLE001 — вердикт не снят = задача не подтверждена
                    infeasible.append(task_id)
                    continue
                if verdict.passed:
                    feasible += 1
                else:
                    infeasible.append(task_id)

    rate = (feasible / total) if total else None
    written: dict[str, Any] = {}
    written["feasibility_rate"] = emit(
        "S-007", "feasibility_rate", round(rate, 6) if rate is not None else None,
        unit="fraction", quality="measured", method=METHOD, subject=subject,
        out_dir=out_dir, status="ok" if (rate is not None and run_verdicts) else "unverified",
        note=(
            ""
            if (rate is not None and run_verdicts)
            else ("вердикты не снимались (--no-verdicts)" if not run_verdicts else "нет задач")
        ),
    )
    written["n_tasks"] = emit(
        "S-007", "n_tasks", total, unit="count", quality="measured", method=METHOD,
        subject=subject, out_dir=out_dir,
    )
    written["infeasible_tasks"] = emit(
        "S-007", "infeasible_tasks", sorted(infeasible), unit="list", quality="measured",
        method=METHOD, subject=subject, out_dir=out_dir,
    )
    return written


def run_selftest() -> int:
    """Механика эталонной починки — без arch-ml: после вставки файл == чистый."""
    from env import corruption
    from env.sft_block import plan_repairable
    from env.util import copy_case_snapshot

    checks: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory(prefix="s007-selftest-") as tmp:
        root = Path(tmp)
        seed = _task_seed("L1", 0)
        damages = plan_repairable(REPO_ROOT, seed, "L1")
        checks.append(("порча восстановима четырьмя инструментами", len(damages) >= 1))
        ws = root / "ws"
        copy_case_snapshot(REPO_ROOT, ws)
        for d in damages:
            corruption.apply_damage(ws, d)
        _repair(ws, REPO_ROOT, damages)
        for d in damages:
            same = (ws / d.file).read_text(encoding="utf-8") == (REPO_ROOT / d.file).read_text(encoding="utf-8")
            checks.append((f"файл {d.file} восстановлен эталонной вставкой", same))
    ok = all(p for _, p in checks)
    for label, passed in checks:
        print(f"[selftest] {'PASS' if passed else 'FAIL'}: {label}")
    print(f"[selftest] {'PASS' if ok else 'FAIL'}: task_feasibility")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="S-007: выполнимость задач (ADR-037)")
    parser.add_argument("--clean", default=str(REPO_ROOT))
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--levels", default=",".join(LEVELS))
    parser.add_argument("--no-verdicts", action="store_true",
                        help="только механика починки, без вердикта arch-ml")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--selftest", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.selftest:
        return run_selftest()
    levels = tuple(x.strip() for x in args.levels.split(",") if x.strip())
    if not args.no_verdicts and not arch_ml_ready():
        print("S-007: arch-ml недоступен — снимаю только механику починки")
    written = measure(
        clean=args.clean, n=args.n, levels=levels, out_dir=args.out_dir,
        device=args.device, run_verdicts=not args.no_verdicts and arch_ml_ready(),
    )
    for name, rec in written.items():
        print(f"S-007 {name} = {rec['value']} [{rec['status']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
