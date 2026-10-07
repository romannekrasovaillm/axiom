"""Генерация набора задач environment v1 (§9): публичный набор + holdout-пул.

Публичный набор: ``{real, corruption} × {L0, L1} × ≥5`` = 20 задач. Holdout-пул:
≥50 задач (source=holdout) с набором скрытых правил, хранящимся ВНЕ workspace
задач (NFR-001: holdout не участвует в генерации публичных задач).

Детерминировано от ``seed``: одинаковый seed → одинаковый набор.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

import yaml

from . import corruption
from .util import (
    EMPTY_HIDDEN_SHA256,
    copy_case_snapshot,
    sha256_file,
    sha256_text,
    tree_sha256,
    workspace_size_cap,
    write_json,
)
from .verifier import CONSTRAINTS_FILE, SPINE_FILE

REAL_IMPL_FILE = "IMPLEMENTATION.md"

# Публичная сетка: (source, level, count). Минимум 20 = {real, corruption} × {L0, L1} × 5.
PUBLIC_GRID: tuple[tuple[str, str, int], ...] = (
    ("real", "L0", 5),
    ("real", "L1", 5),
    ("corruption", "L0", 5),
    ("corruption", "L1", 5),
)
HOLDOUT_COUNT = 50

# Скрытые правила (holdout), всегда истинны на чистом кейсе — механический
# верификатор пары «публичный + скрытый» (ADR-002 п.7, §6).
HIDDEN_RULES: list[dict[str, Any]] = [
    {
        "id": "H-001",
        "name": "Holdout: каждый NFR имеет измеримую цель (measure)",
        "type": "each_file_must_contain",
        "glob": "model/NFR-*.md",
        "pattern": "measure:",
        "severity": "critical",
    },
    {
        "id": "H-002",
        "name": "Holdout: spine фиксирует механический вердикт",
        "type": "must_contain",
        "glob": "ARCHITECTURE-SPINE.md",
        "pattern": "механический вердикт|детерминирован",
        "severity": "critical",
    },
]

_ATTEMPTS = {"L0": 1, "L1": 2, "L2": 3, "L3": 3}
_SOURCE_RANK = {"real": 0, "corruption": 1, "holdout": 2}
_LEVEL_RANK = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}


def real_impl_content(task_id: str) -> str:
    """Содержимое IMPLEMENTATION.md, удовлетворяющее задачным тестам."""
    return (
        f"# Реализация задачи {task_id}\n\n"
        f"{task_id}\n\n"
        f"## Решение\n\n"
        f"Реализация сохраняет архитектурные гейты зелёными (AD-2, AD-3, AD-4).\n"
    )


def real_tests_cmd(task_id: str) -> str:
    """Задачный тест keep-gates-implement: файл и его маркеры."""
    return (
        f"grep -qF {shlex.quote(task_id)} {REAL_IMPL_FILE} "
        f"&& grep -qF '## Решение' {REAL_IMPL_FILE}"
    )


def render_hidden_constraints() -> str:
    """YAML-текст скрытого набора правил (вне workspace)."""
    return yaml.safe_dump(
        {"constraints": HIDDEN_RULES}, sort_keys=False, allow_unicode=True
    )


def count_public_rules(constraints_path: Path) -> int:
    data = yaml.safe_load(constraints_path.read_text(encoding="utf-8"))
    return len(data.get("constraints", []))


def count_case_volume(ws_dir: Path) -> int:
    """Объём кейса S: суммарное число строк .md-файлов (детерминировано)."""
    total = 0
    for p in sorted(ws_dir.rglob("*.md")):
        if p.is_file():
            total += sum(1 for _ in p.open(encoding="utf-8"))
    return total


def hidden_fraction(public_rules: int, hidden_rules: int) -> float:
    return hidden_rules / (public_rules + hidden_rules)


def _task_seed(base_seed: int, source: str, level: str, index: int) -> int:
    return base_seed * 10000 + _SOURCE_RANK[source] * 1000 + _LEVEL_RANK[level] * 100 + index


def _prompt(source: str, kind: str) -> str:
    if kind == "restore-gates":
        return (
            "Восстанови архитектурные гейты кейса до зелёного статуса: "
            "fitness (CONSTRAINTS.yaml), spine и trace должны пройти без error-находок, "
            "не внося новых нарушений."
        )
    return (
        "Реализуй поставленную задачу, сохранив все архитектурные гейты зелёными "
        "(fitness, spine, trace) и добившись прохождения задачных тестов."
    )


def _make_spec(
    task_id: str,
    source: str,
    level: str,
    seed: int,
    workspace: dict[str, Any],
    volume: int,
    hidden_sha: str,
    public_rules: int,
    gates_sha: dict[str, str],
    atoms_version: str = "v1",
) -> dict[str, Any]:
    kind = "restore-gates" if source in ("corruption", "holdout") else "keep-gates-implement"
    tests_cmd = "true" if kind == "restore-gates" else real_tests_cmd(task_id)
    attempts = _ATTEMPTS.get(level, 1)
    h = 0.0 if level == "L0" else hidden_fraction(public_rules, len(HIDDEN_RULES))
    return {
        "id": task_id,
        "source": source,
        "prompt": _prompt(source, kind),
        "workspace": workspace,
        "objective": {"kind": kind, "tests_cmd": tests_cmd},
        "verifier": {
            "constraints": "CONSTRAINTS.yaml",
            "spine": True,
            "trace": True,
            "hidden_constraints_sha256": hidden_sha,
        },
        # H-слой (§10, E-3.3): пин гейтовых файлов из clean-состояния задачи.
        # Verifier сверяет финал воркспейса против этого пина (основной путь);
        # рантайм base_ws — только fallback для спек без пина.
        "gates_sha256": gates_sha,
        # ADR-036 (дельта D5, обратно совместимо): v1 — по умолчанию; v2 добавляет
        # атом drift_config_value и пин claims-пары (§D6). Задачи v1 — побайтово.
        "atoms_version": atoms_version,
        "budget_seconds": 1800,
        "max_tokens": 131072,
        "thinking_budget": 8192,
        "attempts": attempts,
        "difficulty": {
            "R": public_rules,
            "H": h,
            "S": volume,
            "level": level,
        },
        "seed": seed,
    }


def gates_sha256(clean_dir: Path, atoms_version: str = "v1") -> dict[str, str]:
    """Пин гейтовых файлов §10 из clean-состояния задачи (E-3.3).

    ``{constraints, spine}`` — sha256 CONSTRAINTS.yaml и ARCHITECTURE-SPINE.md
    чистого кейса. Порча (corruption) гейтовые файлы не трогает, поэтому пин
    равен и снапшоту задачи (real/corruption), и clean-состоянию holdout.

    Для ``atoms_version: v2`` (ADR-036 дельта D6) добавляются ``claims``
    (model/claims.yaml) и ``claims_checker`` (tools/check_claims.py).
    """
    pins = {
        "constraints": sha256_file(clean_dir / CONSTRAINTS_FILE),
        "spine": sha256_file(clean_dir / SPINE_FILE),
    }
    if atoms_version == "v2":
        pins["claims"] = sha256_file(clean_dir / "model" / "claims.yaml")
        pins["claims_checker"] = sha256_file(clean_dir / "tools" / "check_claims.py")
    return pins


def _workspace_digest(ws_dir: Path) -> dict[str, Any]:
    h = tree_sha256(ws_dir)
    return {"format": "git-bundle", "bundle_sha256": h, "base_commit": h[:40]}


def _build_public_task(
    clean_dir: Path,
    public_dir: Path,
    task_id: str,
    source: str,
    level: str,
    seed: int,
    hidden_sha: str,
    public_rules: int,
    gates_sha: dict[str, str],
    atoms_version: str = "v1",
) -> dict[str, Any]:
    ws_dir = public_dir / task_id
    if source == "corruption":
        corruption.corrupt(clean_dir, ws_dir, seed, level, atoms_version=atoms_version)
    else:
        copy_case_snapshot(clean_dir, ws_dir)
    # Гейт объёма: снапшот после исключений обязан укладываться в кап (§7, §11(8)).
    workspace_size_cap(ws_dir)
    spec = _make_spec(
        task_id, source, level, seed, _workspace_digest(ws_dir),
        count_case_volume(ws_dir), hidden_sha, public_rules, gates_sha, atoms_version,
    )
    write_json(public_dir / f"{task_id}.json", spec)
    return spec


def generate(
    clean_dir: Path,
    out_dir: Path,
    seed: int = 0,
    grid: tuple[tuple[str, str, int], ...] = PUBLIC_GRID,
    holdout_count: int = HOLDOUT_COUNT,
    atoms_version: str = "v1",
) -> dict[str, Any]:
    """Генерирует публичный набор и holdout-пул. Возвращает сводку.

    ``atoms_version`` (ADR-036 дельта D5): ``v1`` по умолчанию (побайтовое
    воспроизведение прежних задач); ``v2`` включает атом ``drift_config_value``
    и пин claims-пары.
    """
    public_dir = out_dir / "public"
    holdout_dir = out_dir / "holdout"
    public_dir.mkdir(parents=True, exist_ok=True)
    holdout_dir.mkdir(parents=True, exist_ok=True)

    public_rules = count_public_rules(clean_dir / "CONSTRAINTS.yaml")
    # H-слой (§10, E-3.3): пин гейтовых файлов из clean-состояния — один для
    # всего набора (порча гейтов не трогает; holdout-воркспейс не строится).
    gates_sha = gates_sha256(clean_dir, atoms_version)

    # Скрытый набор правил — вне workspace (в holdout/).
    hidden_yaml = render_hidden_constraints()
    hidden_path = holdout_dir / "hidden_constraints.yaml"
    hidden_path.write_text(hidden_yaml, encoding="utf-8")
    hidden_sha = sha256_text(hidden_yaml)

    summary: dict[str, Any] = {
        "public": [],
        "holdout": [],
        "hidden_constraints_sha256": hidden_sha,
        "public_rules": public_rules,
    }
    for source, level, count in grid:
        for i in range(count):
            task_id = f"{source}-{level.lower()}-{i:02d}"
            tseed = _task_seed(seed, source, level, i)
            hsha = EMPTY_HIDDEN_SHA256 if level == "L0" else hidden_sha
            spec = _build_public_task(
                clean_dir, public_dir, task_id, source, level, tseed, hsha, public_rules,
                gates_sha, atoms_version,
            )
            summary["public"].append({"id": task_id, "seed": spec["seed"], "level": level})

    for i in range(holdout_count):
        task_id = f"holdout-{i:03d}"
        level = "L0" if i % 2 == 0 else "L1"
        tseed = seed + 100000 + i
        hsha = EMPTY_HIDDEN_SHA256 if level == "L0" else hidden_sha
        # Workspace holdout-задачи не генерируется (невыпущенный кейс); хеш — плейсхолдер.
        held_hash = sha256_text(f"holdout-workspace:{task_id}")
        workspace = {"format": "git-bundle", "bundle_sha256": held_hash, "base_commit": held_hash[:40]}
        spec = _make_spec(
            task_id, "holdout", level, tseed, workspace,
            count_case_volume(clean_dir), hsha, public_rules, gates_sha, atoms_version,
        )
        write_json(holdout_dir / f"{task_id}.json", spec)
        summary["holdout"].append({"id": task_id, "seed": spec["seed"], "level": level})

    summary["public_count"] = len(summary["public"])
    summary["holdout_count"] = len(summary["holdout"])
    return summary
