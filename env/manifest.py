"""Run Manifest (§8): запись, загрузка и валидация с пиннингом (AD-4).

Загрузчик отклоняет прогон без полного пиннинга — прогон без хеша весов,
CONSTRAINTS, holdout и финального состояния доказательством не считается (AD-4).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import schemas
from .util import read_json, write_json


class ManifestError(ValueError):
    """Манифест невалиден (структурно) или неполон по пиннингу (AD-4)."""


def build_manifest(
    run_id: str,
    task_spec: dict,
    arch_ml_build: str,
    constraints_sha256: str,
    hidden_constraints_sha256: str,
    workspace_sha256: str,
    model_base: str,
    model_snapshot_sha256: str,
    decoding: dict,
    attempts_used: int,
    usage: dict,
    timing: dict,
    verdict: dict,
) -> dict:
    """Собирает манифест §8 из компонентов прогона."""
    return {
        "run_id": run_id,
        "task_id": task_spec["id"],
        "env_version": schemas.ENV_VERSION,
        "gates_version": {
            "arch_ml_build": arch_ml_build,
            "constraints_sha256": constraints_sha256,
            "hidden_constraints_sha256": hidden_constraints_sha256,
        },
        "workspace_sha256": workspace_sha256,
        "model": {"snapshot_sha256": model_snapshot_sha256, "base": model_base},
        "decoding": decoding,
        "budget": {
            "max_tokens": task_spec.get("max_tokens", 0),
            "thinking_budget": task_spec.get("thinking_budget", ""),
            "attempts_allowed": task_spec.get("attempts", 1),
            "attempts_used": attempts_used,
        },
        "usage": usage,
        "timing": timing,
        "verdict": verdict,
    }


def full_validate(manifest: dict) -> list[str]:
    """Структурная валидация + полнота пиннинга. Пусто = валидно."""
    errs = schemas.validate_manifest(manifest)
    for p in schemas.missing_pinning(manifest):
        errs.append(f"{p}: поле пиннинга отсутствует или пусто (AD-4)")
    return errs


def dump_manifest(manifest: dict, path: Path) -> None:
    """Пишет манифест, предварительно валидируя его (fail-fast)."""
    errs = full_validate(manifest)
    if errs:
        raise ManifestError("манифест не валиден: " + "; ".join(errs))
    write_json(path, manifest)


def load_manifest(path: Path, require_pinning: bool = True) -> dict:
    """Загружает манифест; отклоняет невалидный/неполный (AD-4)."""
    if not path.exists():
        raise ManifestError(f"манифест не найден: {path}")
    try:
        obj = read_json(path)
    except Exception as exc:  # noqa: BLE001 — читаемость выше точности типа
        raise ManifestError(f"манифест не JSON: {path}: {exc}") from exc
    if require_pinning:
        missing = schemas.missing_pinning(obj)
        if missing:
            raise ManifestError("неполный пиннинг (AD-4): " + ", ".join(missing))
    errs = schemas.validate_manifest(obj)
    if errs:
        raise ManifestError("структурные ошибки манифеста: " + "; ".join(errs))
    return obj
