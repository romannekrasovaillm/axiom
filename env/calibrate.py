"""Кальбровка — precondition запуска RL (§10).

Прогон модели-кандидата по публичному набору задач → отчёт-матрица «модель ×
набор» в evidence/ с пиннингом (AD-4). Критерий готовности RL: pass-rate ∈
[10%, 90%].

Источник эпизодов — ``model_factory`` (E-3.1): ``None`` — детерминированная
заглушка (НЕ LLM, прежнее поведение); заданная фабрика — реальная модель через
пиннутый harness-луп §13 (``tools/rollout_harness``), reward по arm-конфигу
``laguna``. Реальный адаптер (``JaxLMAdapter``) подключается CLI-командой,
объект фабрики соответствует интерфейсу ``generate(messages, seed, max_tokens)``.
"""

from __future__ import annotations

import shutil
import statistics
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from . import run as run_mod
from . import stub_model
from .util import (
    EMPTY_HIDDEN_SHA256,
    dir_total_bytes,
    read_json,
    sha256_file,
    write_json,
)
from .verifier import (
    EXCLUDED_INFRA_RULES,
    arch_ml_build_hash,
    arch_ml_bin,
    detect_excluded_infra_rules,
)

# ── Единый источник интерфейса агента §13: пиннутый harness-луп RL ─────────
# env/ импортирует tools/rollout_harness.py намеренно: калибровка на реальной
# модели обязана ходить тем же эпизодом, что и RL-роллауты (ENVIRONMENT-V1 §13,
# ADR-027). Парсер, инструменты, вердикт и shaping НЕ дублируются.
ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rollout_harness as rh  # noqa: E402  (пиннутый интерфейс §13)

PASS_RATE_RANGE = (0.10, 0.90)

#: Shaping-набор кальбровки на реальной модели: arm ``laguna`` (§13, ADR-027).
#: n_min пиннится раннером (``DEFAULT_N_MIN_TOOL_CALLS``).
DEFAULT_ARM = "laguna"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stub_cell(
    spec: dict[str, Any],
    base_ws: Path,
    clean_dir: Path,
    work_root: Path,
    model_seed: int,
    restore_probability: float,
    solve_probability: float,
    b: str,
    hidden_constraints: Optional[Path],
) -> dict[str, Any]:
    """Ячейка матрицы для детерминированной заглушки (поведение без изменений)."""
    out_ws = work_root / spec["id"]
    stub = stub_model.run_stub_model(
        spec, base_ws, clean_dir, out_ws, model_seed,
        restore_probability=restore_probability,
        solve_probability=solve_probability,
    )
    rr = run_mod.evaluate_run(
        spec, base_ws, out_ws, stub.spent_tokens, bin=b, hidden_constraints=hidden_constraints
    )
    return {
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
    }


def _model_cell(
    spec: dict[str, Any],
    base_ws: Path,
    model_factory: Callable[[], Any],
    model_seed: int,
    arm: str,
    work_root: Path,
    b: str,
    hidden_constraints: Optional[Path],
) -> tuple[dict[str, Any], str]:
    """Ячейка матрицы для реальной модели: эпизод harness-лупа §13.

    Модель ходит четырьмя инструментами по workspace-копии; вердикт — финальное
    состояние через ``EnvVerifier``/``evaluate_run`` (§5); reward — arm-конфиг
    (``laguna``: shaping §13 поверх формулы среды v1). Фабрика зовётся на КАЖДУЮ
    задачу → свежий адаптер на эпизод (журнал/скрипт не перетекает между задачами).
    """
    lm = model_factory()
    verifier = rh.EnvVerifier(spec, base_ws, bin=b, hidden_constraints=hidden_constraints)
    config = rh.EpisodeConfig(arm=arm)
    journal = rh.run_episode(
        spec, base_ws, lm,
        verifier=verifier, config=config,
        seed=model_seed, workdir=work_root / spec["id"],
    )
    chosen = None
    if journal.attempts:
        idx = journal.final_attempt_index
        chosen = journal.attempts[idx] if 0 <= idx < len(journal.attempts) else journal.attempts[-1]
    parts = chosen.reward_parts if chosen is not None else {}
    cell = {
        "task_id": spec["id"],
        "source": spec["source"],
        "level": spec["difficulty"]["level"],
        "pass": bool(journal.verdict_passed),
        "reward": journal.reward,
        "base_reward": chosen.base_reward if chosen is not None else 0.0,
        "pass_component": parts.get("pass_component", 0.0),
        "soft": parts.get("soft", 0.0),
        "new_violations": parts.get("new_violations", 0),
        "effort_penalty": parts.get("effort_penalty", 0.0),
        "spent_tokens": chosen.tokens_used if chosen is not None else 0,
        "attempts_used": journal.attempts_used,
        "termination": journal.termination,
        "policy_version": journal.policy_version,
    }
    checkpoint_sha = getattr(lm, "checkpoint_sha256", "") or ""
    return cell, checkpoint_sha


def calibrate(
    tasks_dir: Path,
    clean_dir: Path,
    out_dir: Path,
    model_name: str = "stub",
    model_seed: int = 7,
    restore_probability: float = 0.5,
    solve_probability: float = 0.5,
    bin: Optional[str] = None,
    model_factory: Optional[Callable[[], Any]] = None,
    arm: str = DEFAULT_ARM,
) -> dict[str, Any]:
    """Кальбровка набора → отчёт-матрица «модель × набор» в ``out_dir`` (evidence).

    ``model_factory`` — фабрика адаптера интерфейса §13
    (``generate(messages, seed, max_tokens)``); ``None`` (умолчание) — прежнее
    поведение на детерминированной заглушке, цифры отчёта не меняются. Заданная
    фабрика ведёт эпизоды через пиннутый harness-луп ``tools/rollout_harness``:
    реальная модель ходит инструментами, вердикт — ``evaluate_run``, reward —
    arm-конфиг (``laguna``: shaping §13 поверх формулы среды v1).
    """
    public_dir = tasks_dir / "public"
    holdout_dir = tasks_dir / "holdout"
    b = bin or arch_ml_bin()

    hidden_path = holdout_dir / "hidden_constraints.yaml"
    hidden_path = hidden_path if hidden_path.exists() else None

    spec_paths = sorted(public_dir.glob("*.json"))
    if not spec_paths:
        raise RuntimeError(f"нет задач в {public_dir}")

    cells: list[dict[str, Any]] = []
    workspace_total_bytes = 0
    model_checkpoint_sha256 = ""
    work_root = Path(tempfile.mkdtemp(prefix="calibrate-work-"))
    try:
        for sp in spec_paths:
            spec = read_json(sp)
            base_ws = public_dir / spec["id"]
            workspace_total_bytes += dir_total_bytes(base_ws)
            hc = hidden_path if spec["verifier"]["hidden_constraints_sha256"] != EMPTY_HIDDEN_SHA256 else None
            if model_factory is None:
                cells.append(_stub_cell(
                    spec, base_ws, clean_dir, work_root, model_seed,
                    restore_probability, solve_probability, b, hc,
                ))
            else:
                cell, sha = _model_cell(
                    spec, base_ws, model_factory, model_seed, arm, work_root, b, hc,
                )
                cells.append(cell)
                if not model_checkpoint_sha256 and sha:
                    model_checkpoint_sha256 = sha
    finally:
        shutil.rmtree(work_root, ignore_errors=True)

    pass_rate = sum(1 for c in cells if c["pass"]) / len(cells)
    rewards = [c["reward"] for c in cells]
    pinning = {
        "arch_ml_build": arch_ml_build_hash(b),
        # R-1' (§7): снапшот несёт ПОЛНЫЙ кейсовый ruleset, поэтому пиннинг —
        # его хеш (а не редуцированного workspace-ruleset, отклонённого E-2.5).
        "constraints_sha256": sha256_file(clean_dir / "CONSTRAINTS.yaml"),
        "hidden_constraints_sha256": sha256_file(hidden_path) if hidden_path else EMPTY_HIDDEN_SHA256,
        # Пин списка инфраструктурных правил, исключаемых фильтром вердикта
        # (R-1'): детектор от полного ruleset; сверяется с реестровой
        # константой (расхождение = дрейф ruleset, виден в отчёте).
        "excluded_infra_rules": list(detect_excluded_infra_rules(clean_dir / "CONSTRAINTS.yaml")),
        "excluded_infra_rules_registry": list(EXCLUDED_INFRA_RULES),
    }
    if model_checkpoint_sha256:  # AD-4: пин предмета замера реальной модели
        pinning["model_checkpoint_sha256"] = model_checkpoint_sha256
    report = {
        "calibration_id": f"calib-{model_name}-{model_seed}",
        "model": model_name,
        "model_seed": model_seed,
        "env_version": "environment-v1",
        "generated_at": _now_iso(),
        "pinning": pinning,
        "matrix": {model_name: cells},
        "workspace_total_bytes": workspace_total_bytes,
        "aggregate": {
            "pass_rate": pass_rate,
            "mean_reward": statistics.fmean(rewards),
            "reward_stdev": statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
            "mean_attempts_used": statistics.fmean([c["attempts_used"] for c in cells]),
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
