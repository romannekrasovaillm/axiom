"""CLI environment v1: ``generate | verify | reward | calibrate``.

Запуск: ``python3 -m env.main <command> ...`` (из каталога кейса) либо
``python3 env/main.py <command> ...``.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import calibrate as calibrate_mod
from . import generate as generate_mod
from . import manifest as manifest_mod
from . import run as run_mod
from .util import read_json, sha256_file, sha256_text, tree_sha256
from .verifier import arch_ml_bin, arch_ml_build_hash

CASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = CASE_DIR / "env" / "data"


def _load_task(path: Path) -> dict:
    spec = read_json(path)
    if not isinstance(spec, dict):
        raise SystemExit(f"task spec не объект: {path}")
    return spec


def _decode_default(spec: dict) -> dict:
    return {"temperature": 0.7, "top_p": 0.95, "seed": int(spec.get("seed", 42))}


def _print_json(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def cmd_generate(args) -> int:
    summary = generate_mod.generate(args.case, args.out, seed=args.seed, holdout_count=args.holdout_count)
    _print_json(summary)
    return 0


def _evaluate(args, spec: dict):
    return run_mod.evaluate_run(
        spec, args.base, args.state, args.spent_tokens,
        bin=arch_ml_bin(), hidden_constraints=args.hidden_constraints,
    )


def cmd_verify(args) -> int:
    spec = _load_task(args.task)
    rr = _evaluate(args, spec)
    verdict = rr.verdict
    out = {
        "task_id": spec["id"],
        "passed": verdict.passed,
        "gates": verdict.gates(),
        "violations": sorted(verdict.violations),
        "reward": rr.reward.to_manifest_dict(),
        "soft_fraction": rr.reward.soft_fraction,
        "issues_warn": verdict.warn_issues,
    }
    if args.manifest_out is not None:
        bin = arch_ml_bin()
        constraints_path = args.state / spec["verifier"]["constraints"]
        model_snapshot = args.model_snapshot or sha256_text("stub-model")
        run_id = args.run_id or str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        manifest = manifest_mod.build_manifest(
            run_id=run_id,
            task_spec=spec,
            arch_ml_build=arch_ml_build_hash(bin),
            constraints_sha256=sha256_file(constraints_path),
            hidden_constraints_sha256=spec["verifier"]["hidden_constraints_sha256"],
            workspace_sha256=tree_sha256(args.state),
            model_base=args.model_base,
            model_snapshot_sha256=model_snapshot,
            decoding=_decode_default(spec),
            attempts_used=1,
            usage={"tokens_in": 0, "tokens_out": args.spent_tokens, "cost_usd": 0.0, "host": args.model_base},
            timing={"started": now, "finished": now, "resume_count": 0},
            verdict={
                "pass": verdict.passed,
                "reward": rr.reward.to_manifest_dict(),
                "issues_warn": verdict.warn_issues,
            },
        )
        manifest_mod.dump_manifest(manifest, args.manifest_out)
        out["manifest"] = str(args.manifest_out)
    _print_json(out)
    return 0


def cmd_reward(args) -> int:
    spec = _load_task(args.task)
    rr = _evaluate(args, spec)
    _print_json(rr.reward.to_manifest_dict())
    return 0


def cmd_calibrate(args) -> int:
    report = calibrate_mod.calibrate(
        args.tasks,
        args.case,
        args.out,
        model_name=args.model_name,
        model_seed=args.model_seed,
        bin=arch_ml_bin(),
    )
    _print_json({
        "out": str(args.out / f"calibration-{args.model_name}.json"),
        "pass_rate": report["aggregate"]["pass_rate"],
        "ready": report["readiness"]["ready"],
        "tasks": len(report["matrix"][args.model_name]),
    })
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="env.main", description="environment v1 RL-среды axiom")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="генерировать публичный набор + holdout-пул")
    g.add_argument("--case", type=Path, default=CASE_DIR, help="каталог чистого кейса")
    g.add_argument("--out", type=Path, default=DEFAULT_DATA_DIR, help="каталог вывода")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--holdout-count", type=int, default=generate_mod.HOLDOUT_COUNT)
    g.set_defaults(fn=cmd_generate)

    v = sub.add_parser("verify", help="вердикт + награда по финальному состоянию")
    v.add_argument("--task", type=Path, required=True, help="Task Spec JSON")
    v.add_argument("--base", type=Path, required=True, help="базовый workspace (исходное состояние)")
    v.add_argument("--state", type=Path, required=True, help="финальный workspace")
    v.add_argument("--spent-tokens", type=int, default=0)
    v.add_argument("--hidden-constraints", type=Path, default=None)
    v.add_argument("--manifest-out", type=Path, default=None, help="записать Run Manifest")
    v.add_argument("--run-id", default=None)
    v.add_argument("--model-base", default="stub")
    v.add_argument("--model-snapshot", default=None)
    v.set_defaults(fn=cmd_verify)

    r = sub.add_parser("reward", help="только награда по финальному состоянию")
    r.add_argument("--task", type=Path, required=True)
    r.add_argument("--base", type=Path, required=True)
    r.add_argument("--state", type=Path, required=True)
    r.add_argument("--spent-tokens", type=int, default=0)
    r.add_argument("--hidden-constraints", type=Path, default=None)
    r.set_defaults(fn=cmd_reward)

    c = sub.add_parser("calibrate", help="кальбровка на заглушке → отчёт в evidence")
    c.add_argument("--tasks", type=Path, required=True, help="каталог с public/ и holdout/")
    c.add_argument("--case", type=Path, default=CASE_DIR)
    c.add_argument("--out", type=Path, required=True, help="каталог evidence/")
    c.add_argument("--model-name", default="stub")
    c.add_argument("--model-seed", type=int, default=7)
    c.set_defaults(fn=cmd_calibrate)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except Exception as exc:  # noqa: BLE001 — CLI печатает причину и код
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
