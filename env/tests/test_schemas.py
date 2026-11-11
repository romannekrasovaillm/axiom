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
        # H-слой (§10, E-3.3): пин гейтовых файлов — обязательное поле.
        "gates_sha256": {"constraints": HASH, "spine": HASH},
        "budget_seconds": 1800,
        "max_tokens": 131072,
        "thinking_budget": 8192,
        "attempts": 1,
        "difficulty": {"R": 8, "H": 0.0, "S": 100, "level": "L0"},
        "seed": 42,
    }


def valid_manifest() -> dict:
    """Валидный Run Manifest v2 (ADR-058)."""
    return {
        "schema_version": schemas.MANIFEST_SCHEMA,
        "run_id": "run-1",
        "task_id": "corruption-l0-00",
        "env_version": "environment-v1",
        "victory_axis": "победа по оси: фитнес-гейт на задачах K3",
        "gates_version": {
            "arch_ml_build": "buildhash",
            "constraints_sha256": HASH,
            "hidden_constraints_sha256": HASH,
            "excluded_infra_rules": ["C-032", "C-045"],
        },
        "workspace_sha256": HASH,
        "model": {"snapshot_sha256": HASH, "base": "net-test", "is_stub": False},
        "baseline": {"model": "kimi-k3", "snapshot_sha256": HASH},
        "decoding": {"temperature": 0.7, "top_p": 0.95, "seed": 42},
        "budget": {"max_tokens": 131072, "thinking_budget": 8192, "attempts_allowed": 1, "attempts_used": 1},
        "usage": {"tokens_in": 0, "tokens_out": 10, "cost_usd": 0.0, "host": "net-test"},
        "timing": {"started": "2026-09-12T00:00:00Z", "finished": "2026-09-12T00:01:00Z", "resume_count": 0},
        "verdict": {
            "pass": True,
            "reward": {"pass_component": 1.0, "soft": 0.5, "new_violations": 0, "effort_penalty": 0.0},
            "issues_warn": [],
        },
    }


def valid_manifest_v1() -> dict:
    """Исторический манифест v1: только для аудита с allow_legacy=True."""
    m = valid_manifest()
    for key in ("schema_version", "victory_axis", "baseline"):
        m.pop(key)
    m["model"].pop("is_stub")
    return m


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


# ── (г) H-слой §10: схемы валидируют наличие/отсутствие gates_sha256 ─────────
def test_task_spec_missing_gates_sha256():
    s = valid_task_spec()
    del s["gates_sha256"]
    errs = schemas.validate_task_spec(s)
    assert any("gates_sha256" in e for e in errs), errs


def test_task_spec_gates_sha256_bad_hash():
    s = valid_task_spec()
    s["gates_sha256"]["spine"] = "not-a-hash"
    errs = schemas.validate_task_spec(s)
    assert any("gates_sha256.spine" in e for e in errs), errs


def test_task_spec_gates_sha256_incomplete():
    s = valid_task_spec()
    del s["gates_sha256"]["constraints"]
    errs = schemas.validate_task_spec(s)
    assert any("gates_sha256.constraints" in e for e in errs), errs


def test_task_spec_gates_sha256_unknown_key():
    s = valid_task_spec()
    s["gates_sha256"]["extra"] = HASH
    errs = schemas.validate_task_spec(s)
    assert any("gates_sha256" in e and "extra" in e for e in errs), errs


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


# ── v2 (ADR-058): ось, baseline, запрет заглушки ────────────────────────────
def test_manifest_empty_missing_pinning_names_v2_fields():
    missing = schemas.missing_pinning({})
    for name in ("schema_version", "victory_axis", "baseline.snapshot_sha256", "model.is_stub"):
        assert name in missing, missing


def test_manifest_requires_victory_axis():
    m = valid_manifest()
    del m["victory_axis"]
    errs = schemas.validate_manifest(m)
    assert any("victory_axis" in e for e in errs)


def test_manifest_requires_baseline():
    m = valid_manifest()
    del m["baseline"]
    errs = schemas.validate_manifest(m)
    assert any("baseline" in e for e in errs)


def test_manifest_requires_is_stub_bool():
    m = valid_manifest()
    del m["model"]["is_stub"]
    errs = schemas.validate_manifest(m)
    assert any("is_stub" in e for e in errs)


def test_manifest_stub_is_not_pinning():
    m = valid_manifest()
    m["model"]["is_stub"] = True
    # Структурно bool валиден, но доказательством не является (ADR-058).
    assert schemas.validate_manifest(m) == []
    assert "model.is_stub" in schemas.missing_pinning(m)
    # отсутствие поля — тоже неполный пин (не доказано, что не заглушка)
    del m["model"]["is_stub"]
    assert "model.is_stub" in schemas.missing_pinning(m)


def test_manifest_v1_requires_allow_legacy():
    v1 = valid_manifest_v1()
    assert schemas.validate_manifest(v1)  # v2-поля обязательны без послабления
    assert schemas.validate_manifest(v1, allow_legacy=True) == []
    assert schemas.missing_pinning(v1)  # v2-набор пиннинга
    assert schemas.missing_pinning(v1, allow_legacy=True) == []


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
