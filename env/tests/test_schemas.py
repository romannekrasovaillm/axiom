"""§11(1): схемы §2 и §8 валидируют эталонные валидный/невалидный примеры."""

from __future__ import annotations

from env import schemas

HASH = "0" * 64


def valid_task_spec() -> dict:
    return {
        "id": "corruption-l0-00",
        "source": "corruption",
        "prompt": "Восстанови гейты.",
        "workspace": {"format": "git-bundle", "bundle_sha256": HASH, "base_commit": "a" * 40},
        "objective": {"kind": "restore-gates", "tests_cmd": "true"},
        "verifier": {
            "constraints": "CONSTRAINTS.yaml",
            "spine": True,
            "trace": True,
            "hidden_constraints_sha256": HASH,
        },
        "budget_seconds": 1800,
        "max_tokens": 131072,
        "thinking_budget": 8192,
        "attempts": 1,
        "difficulty": {"R": 8, "H": 0.0, "S": 100, "level": "L0"},
        "seed": 42,
    }


def valid_manifest() -> dict:
    return {
        "run_id": "run-1",
        "task_id": "corruption-l0-00",
        "env_version": "environment-v1",
        "gates_version": {
            "arch_ml_build": "buildhash",
            "constraints_sha256": HASH,
            "hidden_constraints_sha256": HASH,
        },
        "workspace_sha256": HASH,
        "model": {"snapshot_sha256": HASH, "base": "stub"},
        "decoding": {"temperature": 0.7, "top_p": 0.95, "seed": 42},
        "budget": {"max_tokens": 131072, "thinking_budget": 8192, "attempts_allowed": 1, "attempts_used": 1},
        "usage": {"tokens_in": 0, "tokens_out": 10, "cost_usd": 0.0, "host": "stub"},
        "timing": {"started": "2026-09-12T00:00:00Z", "finished": "2026-09-12T00:01:00Z", "resume_count": 0},
        "verdict": {
            "pass": True,
            "reward": {"pass_component": 1.0, "soft": 0.5, "new_violations": 0, "effort_penalty": 0.0},
            "issues_warn": [],
        },
    }


def test_task_spec_valid():
    assert schemas.validate_task_spec(valid_task_spec()) == []


def test_task_spec_invalid_source():
    s = valid_task_spec()
    s["source"] = "bogus"
    errs = schemas.validate_task_spec(s)
    assert any("source" in e for e in errs)


def test_task_spec_invalid_sha256():
    s = valid_task_spec()
    s["workspace"]["bundle_sha256"] = "not-a-hash"
    errs = schemas.validate_task_spec(s)
    assert any("bundle_sha256" in e for e in errs)


def test_task_spec_missing_field():
    s = valid_task_spec()
    del s["objective"]["tests_cmd"]
    errs = schemas.validate_task_spec(s)
    assert any("tests_cmd" in e for e in errs)


def test_task_spec_bad_level():
    s = valid_task_spec()
    s["difficulty"]["level"] = "L9"
    errs = schemas.validate_task_spec(s)
    assert any("level" in e for e in errs)


def test_manifest_valid():
    assert schemas.validate_manifest(valid_manifest()) == []
    assert schemas.missing_pinning(valid_manifest()) == []


def test_manifest_invalid_env_version():
    m = valid_manifest()
    m["env_version"] = "environment-v0"
    errs = schemas.validate_manifest(m)
    assert any("env_version" in e for e in errs)


def test_manifest_invalid_workspace_sha():
    m = valid_manifest()
    m["workspace_sha256"] = "xyz"
    errs = schemas.validate_manifest(m)
    assert any("workspace_sha256" in e for e in errs)


def test_manifest_missing_pinning_field():
    m = valid_manifest()
    del m["model"]["snapshot_sha256"]
    assert "model.snapshot_sha256" in schemas.missing_pinning(m)


def test_manifest_empty_pinning_field():
    m = valid_manifest()
    m["gates_version"]["arch_ml_build"] = ""
    assert "gates_version.arch_ml_build" in schemas.missing_pinning(m)


def test_full_validate_flags_pinning():
    m = valid_manifest()
    del m["gates_version"]["constraints_sha256"]
    errs = schemas.validate_manifest(m) + [
        f"{p}: поле пиннинга отсутствует или пусто (AD-4)" for p in schemas.missing_pinning(m)
    ]
    # полная валидация в manifest.full_validate покрывается в test_manifest.py
    assert any("constraints_sha256" in e for e in errs)
