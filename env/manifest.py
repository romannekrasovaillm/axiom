"""Run Manifest (§8): запись, загрузка и валидация с пиннингом (AD-4, ADR-058).

Загрузчик отклоняет прогон без полного пиннинга — прогон без хеша весов,
CONSTRAINTS, holdout и финального состояния доказательством не считается (AD-4).

Схема v2 (``environment-v1/manifest-v2``) добавляет к прежнему контракту ось
победы из пакета решения (:mod:`env.axis`), пин baseline и признак заглушки
(``model.is_stub``): манифест заглушки и манифест с осью, не совпавшей с
пакетом, не записываются вовсе — fail-closed (ADR-058).  Исторические
манифесты v1 читаются только явным ``allow_legacy=True`` и доказательством
не являются.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from . import axis, schemas
from .util import read_json, write_json
from .verifier import EXCLUDED_INFRA_RULES

#: Пакет решения — источник оси победы (ADR-058, AD-1).
DEFAULT_PACKAGE_PATH = Path(__file__).resolve().parent.parent / "docs" / "DECISION-PACKAGE.json"


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
    excluded_infra_rules: Optional[list[str]] = None,
    *,
    baseline_model: str,
    baseline_snapshot_sha256: str,
    victory_axis: Optional[str] = None,
    model_is_stub: bool = False,
) -> dict:
    """Собирает манифест §8 (v2) из компонентов прогона.

    ``victory_axis`` по умолчанию читается из ``docs/DECISION-PACKAGE.json``
    (``choice.victory_axis``).  ``baseline_*`` обязательны: манифест без
    запиннутого baseline — не доказательство (ADR-058).  ``model_is_stub``
    ставится только явно (отладочная заглушка не пройдёт ``dump_manifest``).
    """
    if victory_axis is None:
        victory_axis = axis.load_axis(DEFAULT_PACKAGE_PATH)
    return {
        "schema_version": schemas.MANIFEST_SCHEMA,
        "run_id": run_id,
        "task_id": task_spec["id"],
        "env_version": schemas.ENV_VERSION,
        "victory_axis": victory_axis,
        "gates_version": {
            "arch_ml_build": arch_ml_build,
            "constraints_sha256": constraints_sha256,
            "hidden_constraints_sha256": hidden_constraints_sha256,
            # R-1' (§7): пин списка инфраструктурных правил, исключённых фильтром
            # вердикта; пусто/None → реестровый пин v1.
            "excluded_infra_rules": list(
                EXCLUDED_INFRA_RULES if excluded_infra_rules is None else excluded_infra_rules
            ),
        },
        "workspace_sha256": workspace_sha256,
        "model": {
            "snapshot_sha256": model_snapshot_sha256,
            "base": model_base,
            "is_stub": model_is_stub,
        },
        "baseline": {
            "model": baseline_model,
            "snapshot_sha256": baseline_snapshot_sha256,
        },
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


def _axis_errors(manifest: Any, package_path: Optional[Path]) -> list[str]:
    """Сверка оси v2-манифеста с пакетом решения (ADR-058)."""
    if package_path is None or not isinstance(manifest, dict):
        return []
    if manifest.get("schema_version") != schemas.MANIFEST_SCHEMA:
        return []
    actual = manifest.get("victory_axis")
    if not isinstance(actual, str) or not actual.strip():
        return []  # структурная ошибка/недостающий пин уже названы отдельно
    try:
        expected = axis.load_axis(Path(package_path))
    except axis.AxisError as exc:
        return [f"victory_axis: пакет решения не читается: {exc}"]
    if not axis.axis_matches(actual, expected):
        return ["victory_axis: не совпадает с пакетом решения (ADR-058)"]
    return []


def full_validate(
    manifest: dict,
    *,
    package_path: Optional[Path] = None,
    allow_legacy: bool = False,
) -> list[str]:
    """Структурная валидация + полнота пиннинга. Пусто = валидно.

    ``package_path`` включает сверку ``victory_axis`` с пакетом решения (v2).
    ``allow_legacy=True`` — исторический аудит v1 без пиннинга оси/baseline;
    такой манифест не является доказательством (ADR-058).
    """
    errs = schemas.validate_manifest(manifest, allow_legacy=allow_legacy)
    for p in schemas.missing_pinning(manifest, allow_legacy=allow_legacy):
        errs.append(f"{p}: поле пиннинга отсутствует или пусто (AD-4)")
    errs.extend(_axis_errors(manifest, package_path))
    return errs


def dump_manifest(
    manifest: dict,
    path: Path,
    *,
    package_path: Optional[Path] = None,
    allow_legacy: bool = False,
) -> None:
    """Пишет манифест, предварительно валидируя его (fail-fast)."""
    errs = full_validate(manifest, package_path=package_path, allow_legacy=allow_legacy)
    if errs:
        raise ManifestError("манифест не валиден: " + "; ".join(errs))
    write_json(path, manifest)


def load_manifest(path: Path, require_pinning: bool = True) -> dict:
    """Загружает манифест; отклоняет невалидный/неполный (AD-4).

    По умолчанию действует доказательная схема v2: исторический манифест v1
    отклоняется (``allow_legacy`` есть только у :func:`full_validate`).
    """
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
