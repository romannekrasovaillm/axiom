"""Кальбровка — precondition запуска RL (§10).

Прогон модели-кандидата (здесь — детерминированная заглушка, НЕ LLM) по
публичному набору задач → отчёт-матрица «модель × набор» в evidence/ с пиннингом
(AD-4). Критерий готовности RL: pass-rate ∈ [10%, 90%].
"""

from __future__ import annotations

import shutil
import statistics
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from . import run as run_mod
from . import stub_model
from .util import (
    EMPTY_HIDDEN_SHA256,
    read_json,
    sha256_file,
    write_json,
)
from .verifier import arch_ml_build_hash, arch_ml_bin

PASS_RATE_RANGE = (0.10, 0.90)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def calibrate(
    tasks_dir: Path,
    clean_dir: Path,
    out_dir: Path,
    model_name: str = "stub",
    model_seed: int = 7,
    restore_probability: float = 0.5,
    solve_probability: float = 0.5,
    bin: Optional[str] = None,
) -> dict[str, Any]:
    """Прогоняет заглушку по набору и пишет отчёт в ``out_dir`` (evidence)."""
    public_dir = tasks_dir / "public"
    holdout_dir = tasks_dir / "holdout"
    b = bin or arch_ml_bin()

    hidden_path = holdout_dir / "hidden_constraints.yaml"
    hidden_path = hidden_path if hidden_path.exists() else None

    spec_paths = sorted(public_dir.glob("*.json"))
    if not spec_paths:
        raise RuntimeError(f"нет задач в {public_dir}")

    cells: list[dict[str, Any]] = []
    work_root = Path(tempfile.mkdtemp(prefix="calibrate-work-"))
    try:
        for sp in spec_paths:
            spec = read_json(sp)
            base_ws = public_dir / spec["id"]
            hc = hidden_path if spec["verifier"]["hidden_constraints_sha256"] != EMPTY_HIDDEN_SHA256 else None
            out_ws = work_root / spec["id"]
            stub = stub_model.run_stub_model(
                spec, base_ws, clean_dir, out_ws, model_seed,
                restore_probability=restore_probability,
                solve_probability=solve_probability,
            )
            rr = run_mod.evaluate_run(spec, base_ws, out_ws, stub.spent_tokens, bin=b, hidden_constraints=hc)
            cells.append({
                "task_id": spec["id"],
                "source": spec["source"],
                "level": spec["difficulty"]["level"],
                "pass": rr.verdict.passed,
                "reward": rr.reward.total,
                "pass_component": rr.reward.pass_component,
                "soft": rr.reward.soft,
                "new_violations": rr.reward.new_violations_count,
                "effort_penalty": rr.reward.effort_penalty,
                "spent_tokens": stub.spent_tokens,
                "attempts_used": 1,
            })
    finally:
        shutil.rmtree(work_root, ignore_errors=True)

    pass_rate = sum(1 for c in cells if c["pass"]) / len(cells)
    rewards = [c["reward"] for c in cells]
    report = {
        "calibration_id": f"calib-{model_name}-{model_seed}",
        "model": model_name,
        "model_seed": model_seed,
        "env_version": "environment-v1",
        "generated_at": _now_iso(),
        "pinning": {
            "arch_ml_build": arch_ml_build_hash(b),
            "constraints_sha256": sha256_file(clean_dir / "CONSTRAINTS.yaml"),
            "hidden_constraints_sha256": sha256_file(hidden_path) if hidden_path else EMPTY_HIDDEN_SHA256,
        },
        "matrix": {model_name: cells},
        "aggregate": {
            "pass_rate": pass_rate,
            "mean_reward": statistics.fmean(rewards),
            "reward_stdev": statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
            "mean_attempts_used": 1.0,
            "group_size": 1,
        },
        "readiness": {
            "ready": PASS_RATE_RANGE[0] <= pass_rate <= PASS_RATE_RANGE[1],
            "pass_rate_range": list(PASS_RATE_RANGE),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / f"calibration-{model_name}.json", report)
    return report
