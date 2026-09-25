"""Ручная валидация контрактов environment v1: Task Spec (§2) и Run Manifest (§8).

Без pydantic/jsonschema — проверки явные, ошибки собираются списком строк.
Пустой список ошибок = валидный документ.
"""

from __future__ import annotations

import re
from typing import Any

from .util import is_sha256_hex

ENV_VERSION = "environment-v1"
_SLUG = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_SOURCES = ("real", "corruption", "holdout")
_OBJECTIVE_KINDS = ("restore-gates", "keep-gates-implement")
_LEVELS = ("L0", "L1", "L2", "L3")


def _errs(errs: list[str], path: str, cond: bool, msg: str) -> None:
    if not cond:
        errs.append(f"{path}: {msg}")


def _is_str(v: Any) -> bool:
    return isinstance(v, str) and bool(v)


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# ── Task Spec §2 ────────────────────────────────────────────────────────────

def validate_task_spec(spec: Any) -> list[str]:
    """Валидация Task Spec (§2). Возвращает список ошибок (пусто = валидно)."""
    e: list[str] = []
    if not isinstance(spec, dict):
        return ["task spec: должен быть объектом"]

    _errs(e, "id", _is_str(spec.get("id")) and _SLUG.match(spec.get("id")), "id — непустой slug [a-z0-9-]+")
    _errs(e, "source", spec.get("source") in _SOURCES, f"source — один из {_SOURCES}")
    _errs(e, "prompt", _is_str(spec.get("prompt")), "prompt — непустая строка")

    ws = spec.get("workspace")
    if isinstance(ws, dict):
        _errs(e, "workspace.format", ws.get("format") == "git-bundle", "workspace.format = git-bundle")
        _errs(e, "workspace.bundle_sha256", is_sha256_hex(ws.get("bundle_sha256")), "workspace.bundle_sha256 — sha256")
        _errs(e, "workspace.base_commit", _is_str(ws.get("base_commit")), "workspace.base_commit — непустая строка")
    else:
        e.append("workspace: должен быть объектом")

    obj = spec.get("objective")
    if isinstance(obj, dict):
        _errs(e, "objective.kind", obj.get("kind") in _OBJECTIVE_KINDS, f"objective.kind — один из {_OBJECTIVE_KINDS}")
        _errs(e, "objective.tests_cmd", _is_str(obj.get("tests_cmd")), "objective.tests_cmd — непустая строка")
    else:
        e.append("objective: должен быть объектом")

    ver = spec.get("verifier")
    if isinstance(ver, dict):
        _errs(e, "verifier.constraints", _is_str(ver.get("constraints")), "verifier.constraints — путь к CONSTRAINTS")
        _errs(e, "verifier.spine", isinstance(ver.get("spine"), bool), "verifier.spine — bool")
        _errs(e, "verifier.trace", isinstance(ver.get("trace"), bool), "verifier.trace — bool")
        _errs(e, "verifier.hidden_constraints_sha256", is_sha256_hex(ver.get("hidden_constraints_sha256")), "verifier.hidden_constraints_sha256 — sha256")
    else:
        e.append("verifier: должен быть объектом")

    _errs(e, "budget_seconds", _is_int(spec.get("budget_seconds")) and spec["budget_seconds"] > 0, "budget_seconds — целое > 0")
    _errs(e, "max_tokens", _is_int(spec.get("max_tokens")) and spec["max_tokens"] > 0, "max_tokens — целое > 0")

    tb = spec.get("thinking_budget")
    _errs(e, "thinking_budget", _is_str(tb) or (_is_int(tb) and tb > 0), "thinking_budget — целое > 0 или строка")

    _errs(e, "attempts", _is_int(spec.get("attempts")) and spec["attempts"] > 0, "attempts — целое > 0")

    diff = spec.get("difficulty")
    if isinstance(diff, dict):
        _errs(e, "difficulty.R", _is_int(diff.get("R")) and diff["R"] >= 0, "difficulty.R — целое ≥ 0")
        _errs(e, "difficulty.H", _is_number(diff.get("H")) and 0.0 <= diff["H"] <= 1.0, "difficulty.H — число в [0,1]")
        _errs(e, "difficulty.S", _is_int(diff.get("S")) and diff["S"] >= 0, "difficulty.S — целое ≥ 0")
        _errs(e, "difficulty.level", diff.get("level") in _LEVELS, f"difficulty.level — один из {_LEVELS}")
    else:
        e.append("difficulty: должен быть объектом")

    _errs(e, "seed", _is_int(spec.get("seed")), "seed — целое")
    return e


# ── Run Manifest §8 ─────────────────────────────────────────────────────────

def _validate_verdict(verdict: Any, path: str, e: list[str]) -> None:
    if not isinstance(verdict, dict):
        e.append(f"{path}: должен быть объектом")
        return
    _errs(e, f"{path}.pass", isinstance(verdict.get("pass"), bool), "pass — bool")
    reward = verdict.get("reward")
    if isinstance(reward, dict):
        for key in ("pass_component", "soft", "new_violations", "effort_penalty"):
            _errs(e, f"{path}.reward.{key}", _is_number(reward.get(key)), f"reward.{key} — число")
    else:
        e.append(f"{path}.reward: должен быть объектом")
    _errs(e, f"{path}.issues_warn", isinstance(verdict.get("issues_warn"), list), "issues_warn — список")


def validate_manifest(manifest: Any) -> list[str]:
    """Структурная валидация Run Manifest (§8) без проверки пиннинга."""
    e: list[str] = []
    if not isinstance(manifest, dict):
        return ["manifest: должен быть объектом"]

    _errs(e, "run_id", _is_str(manifest.get("run_id")), "run_id — непустая строка")
    _errs(e, "task_id", _is_str(manifest.get("task_id")), "task_id — непустая строка")
    _errs(e, "env_version", manifest.get("env_version") == ENV_VERSION, f"env_version = {ENV_VERSION}")

    gv = manifest.get("gates_version")
    if isinstance(gv, dict):
        _errs(e, "gates_version.arch_ml_build", _is_str(gv.get("arch_ml_build")), "arch_ml_build — непустая строка")
        _errs(e, "gates_version.constraints_sha256", is_sha256_hex(gv.get("constraints_sha256")), "constraints_sha256 — sha256")
        _errs(e, "gates_version.hidden_constraints_sha256", is_sha256_hex(gv.get("hidden_constraints_sha256")), "hidden_constraints_sha256 — sha256")
    else:
        e.append("gates_version: должен быть объектом")

    _errs(e, "workspace_sha256", is_sha256_hex(manifest.get("workspace_sha256")), "workspace_sha256 — sha256")

    model = manifest.get("model")
    if isinstance(model, dict):
        _errs(e, "model.snapshot_sha256", is_sha256_hex(model.get("snapshot_sha256")), "model.snapshot_sha256 — sha256")
        _errs(e, "model.base", _is_str(model.get("base")), "model.base — непустая строка")
    else:
        e.append("model: должен быть объектом")

    dec = manifest.get("decoding")
    if isinstance(dec, dict):
        _errs(e, "decoding.temperature", _is_number(dec.get("temperature")), "temperature — число")
        _errs(e, "decoding.top_p", _is_number(dec.get("top_p")), "top_p — число")
        _errs(e, "decoding.seed", _is_int(dec.get("seed")), "seed — целое")
    else:
        e.append("decoding: должен быть объектом")

    budget = manifest.get("budget")
    if isinstance(budget, dict):
        _errs(e, "budget.max_tokens", _is_int(budget.get("max_tokens")), "max_tokens — целое")
        _errs(e, "budget.attempts_allowed", _is_int(budget.get("attempts_allowed")), "attempts_allowed — целое")
        _errs(e, "budget.attempts_used", _is_int(budget.get("attempts_used")), "attempts_used — целое")
    else:
        e.append("budget: должен быть объектом")

    usage = manifest.get("usage")
    if isinstance(usage, dict):
        _errs(e, "usage.tokens_in", _is_int(usage.get("tokens_in")), "tokens_in — целое")
        _errs(e, "usage.tokens_out", _is_int(usage.get("tokens_out")), "tokens_out — целое")
        _errs(e, "usage.cost_usd", _is_number(usage.get("cost_usd")), "cost_usd — число")
        _errs(e, "usage.host", _is_str(usage.get("host")), "host — непустая строка")
    else:
        e.append("usage: должен быть объектом")

    timing = manifest.get("timing")
    if isinstance(timing, dict):
        _errs(e, "timing.started", _is_str(timing.get("started")), "started — строка")
        _errs(e, "timing.finished", _is_str(timing.get("finished")), "finished — строка")
        _errs(e, "timing.resume_count", _is_int(timing.get("resume_count")), "resume_count — целое")
    else:
        e.append("timing: должен быть объектом")

    _validate_verdict(manifest.get("verdict"), "verdict", e)
    return e


# ── Пиннинг (AD-4) ──────────────────────────────────────────────────────────

# Поля, отсутствие/пустота которых лишает прогон статуса доказательства (AD-4).
PINNING_PATHS = (
    ("gates_version", "arch_ml_build"),
    ("gates_version", "constraints_sha256"),
    ("gates_version", "hidden_constraints_sha256"),
    ("workspace_sha256",),
    ("model", "snapshot_sha256"),
)


def missing_pinning(manifest: Any) -> list[str]:
    """Список недостающих/пустых полей пиннинга (AD-4). Пусто = пиннинг полон."""
    missing: list[str] = []
    for path in PINNING_PATHS:
        cur: Any = manifest
        ok = True
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                ok = False
                break
            cur = cur[key]
        if not ok or cur is None or cur == "":
            missing.append(".".join(path))
    return missing
