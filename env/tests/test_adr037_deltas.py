"""ADR-037 дельты D (атом drift_config_value, версии атомов, H-слой v2, SFT D7)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from env import corruption
from env.schemas import validate_task_spec
from env.util import copy_case_snapshot, is_sha256_hex, sha256_file
from env.verifier import EXCLUDED_INFRA_RULES, detect_excluded_infra_rules, gate_integrity

ROOT = Path(__file__).resolve().parents[2]


def _clean_case(tmp_path: Path) -> Path:
    """Мини-кейс: adr + config + claims с одним config_binding."""
    case = tmp_path / "case"
    (case / "docs" / "adr").mkdir(parents=True)
    (case / "model").mkdir()
    (case / "net").mkdir()
    (case / "tools").mkdir()
    (case / "docs" / "adr" / "ADR-001-x.md").write_text(
        "# ADR-001\n## Alternatives Considered\n### Negative\n## Reversibility\n",
        encoding="utf-8",
    )
    (case / "net" / "config.json").write_text(
        json.dumps({"num_kda_layers": 18, "moe_top_k": 2}, indent=2) + "\n", encoding="utf-8"
    )
    (case / "model" / "claims.yaml").write_text(
        "- id: CL-1\n"
        "  source: {file: docs/adr/ADR-001-x.md, anchor: \"## Alternatives Considered\"}\n"
        "  statement: s\n"
        "  kind: config_binding\n"
        "  sensor: S-001\n"
        "  fact: num_kda_layers\n"
        "  predicate: {op: \"==\", value: 18, unit: count, tolerance: null}\n"
        "  subject_match: []\n"
        "  rule: null\n"
        "  binding: {file: net/config.json, path: num_kda_layers, value: 18}\n",
        encoding="utf-8",
    )
    return case


def test_level_atoms_v2_adds_drift_only_on_l1_l3():
    assert "drift_config_value" not in corruption.LEVEL_ATOMS["L0"]
    assert "drift_config_value" not in corruption.LEVEL_ATOMS["L1"]
    assert "drift_config_value" not in corruption.LEVEL_ATOMS_V2["L0"]
    for level in ("L1", "L2", "L3"):
        assert "drift_config_value" in corruption.LEVEL_ATOMS_V2[level]
        assert "drift_config_value" not in corruption.LEVEL_ATOMS[level]


def test_v1_plan_is_byte_stable(tmp_path):
    case = _clean_case(tmp_path)
    a = [d.as_dict() for d in corruption.plan_damages(case, 7, "L1", atoms_version="v1")]
    b = [d.as_dict() for d in corruption.plan_damages(case, 7, "L1", atoms_version="v1")]
    assert a == b  # детерминизм по сиду
    assert all(d["kind"] != "drift_config_value" for d in a)


def test_drift_apply_revert_deterministic(tmp_path):
    case = _clean_case(tmp_path)
    planned = [d for d in corruption.plan_damages(case, 3, "L1", atoms_version="v2")
               if d.kind == "drift_config_value"]
    assert planned, "drift-атом должен планироваться на L1 v2"
    ws = tmp_path / "ws"
    copy_case_snapshot(case, ws)
    for d in planned:
        corruption.apply_damage(ws, d)
    after = json.loads((ws / "net" / "config.json").read_text(encoding="utf-8"))  # JSON валиден
    assert any(after[k] != json.loads((case / "net" / "config.json").read_text())[k]
               for k in after)
    for d in planned:
        corruption.revert_damage(ws, case, d)
    assert (ws / "net" / "config.json").read_text(encoding="utf-8") == \
        (case / "net" / "config.json").read_text(encoding="utf-8")


def test_c047_flow_corrupt_fail_then_restore_pass(tmp_path):
    case = _clean_case(tmp_path)
    ws = tmp_path / "ws"
    copy_case_snapshot(case, ws)
    d = next(d for d in corruption.plan_damages(case, 5, "L1", atoms_version="v2")
             if d.kind == "drift_config_value")
    corruption.apply_damage(ws, d)
    bad = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "check_claims.py"),
         "--mode", "config-bindings", "--root", str(ws)],
        capture_output=True, text=True,
    )
    assert bad.returncode == 1
    assert (d and "CL-1" in bad.stdout), bad.stdout  # сообщение называет CL-id
    assert "expected" not in bad.stdout  # ожидаемое значение не подсказано
    corruption.revert_damage(ws, case, d)
    good = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "check_claims.py"),
         "--mode", "config-bindings", "--root", str(ws)],
        capture_output=True, text=True,
    )
    assert good.returncode == 0, good.stdout + good.stderr


def _spec(gates, *, atoms_version="v1"):
    return {
        "id": "t", "source": "corruption", "prompt": "p",
        "workspace": {"format": "git-bundle", "bundle_sha256": "a" * 64, "base_commit": "b" * 40},
        "objective": {"kind": "restore-gates", "tests_cmd": "true"},
        "verifier": {"constraints": "CONSTRAINTS.yaml", "spine": True, "trace": True,
                     "hidden_constraints_sha256": "c" * 64},
        "gates_sha256": gates, "atoms_version": atoms_version,
        "budget_seconds": 10, "max_tokens": 10, "thinking_budget": 1, "attempts": 1,
        "difficulty": {"R": 1, "H": 0.0, "S": 1, "level": "L1"}, "seed": 0,
    }


def test_schema_v1_and_v2_gate_keys():
    base = {"constraints": "a" * 64, "spine": "b" * 64}
    assert validate_task_spec(_spec(base)) == []
    v2 = {**base, "claims": "c" * 64, "claims_checker": "d" * 64}
    assert validate_task_spec(_spec(v2, atoms_version="v2")) == []
    # v2 без claims-пары → ошибка
    assert any("claims" in e for e in validate_task_spec(_spec(base, atoms_version="v2")))
    # лишний ключ у v1 → ошибка
    assert any("неизвестные" in e for e in validate_task_spec(_spec(v2)))


def test_excluded_infra_rules_include_c048_not_c047(tmp_path):
    case = tmp_path / "c"
    case.mkdir()
    (case / "CONSTRAINTS.yaml").write_text(
        "constraints:\n"
        "  - id: C-040\n    type: command_succeeds\n    kind: behavioural\n"
        "  - id: C-049\n    type: command_succeeds\n    kind: structural\n    infra: false\n"
        "  - id: C-050\n    type: command_succeeds\n    kind: behavioural\n",
        encoding="utf-8",
    )
    detected = detect_excluded_infra_rules(case / "CONSTRAINTS.yaml")
    assert "C-040" in detected and "C-050" in detected and "C-049" not in detected
    assert len(EXCLUDED_INFRA_RULES) == 16
    assert "C-050" in EXCLUDED_INFRA_RULES and "C-049" not in EXCLUDED_INFRA_RULES


def test_h_layer_v2_pins_claims_and_checker(tmp_path):
    ws = tmp_path / "ws"
    (ws / "model").mkdir(parents=True)
    (ws / "tools").mkdir()
    (ws / "CONSTRAINTS.yaml").write_text("constraints: []\n", encoding="utf-8")
    (ws / "ARCHITECTURE-SPINE.md").write_text("# spine\n", encoding="utf-8")
    (ws / "model" / "claims.yaml").write_text("- id: CL-1\n", encoding="utf-8")
    (ws / "tools" / "check_claims.py").write_text("# checker\n", encoding="utf-8")
    pins = {
        "constraints": sha256_file(ws / "CONSTRAINTS.yaml"),
        "spine": sha256_file(ws / "ARCHITECTURE-SPINE.md"),
        "claims": sha256_file(ws / "model" / "claims.yaml"),
        "claims_checker": sha256_file(ws / "tools" / "check_claims.py"),
    }
    spec = _spec(pins, atoms_version="v2")
    clean = gate_integrity(ws, task_spec=spec)
    assert clean.passed, [e["message"] for e in clean.errors]
    # правка claims.yaml → hacking
    (ws / "model" / "claims.yaml").write_text("- id: CL-1 # tampered\n", encoding="utf-8")
    hacked = gate_integrity(ws, task_spec=spec)
    assert not hacked.passed
    assert any(e.get("class") == "hacking" and e["file"] == "model/claims.yaml" for e in hacked.errors)


def test_sft_d7_reference_trajectory_uses_workspace_adr(tmp_path):
    from env import sft_block

    ws = tmp_path / "ws"
    (ws / "docs").mkdir(parents=True)
    (ws / "net").mkdir()
    (ws / "net" / "config.json").write_text(
        json.dumps({"vocab_size": 160001, "moe_top_k": 2}, indent=2) + "\n", encoding="utf-8"
    )
    (ws / "docs" / "ADR-004.md").write_text(
        "Канонический словарь — 160000 токенов.\n", encoding="utf-8"
    )

    class _Noop:
        def run_gates(self, root):
            return {"passed": True}

    claim = {
        "id": "CL-X", "kind": "config_binding",
        "binding": {"file": "net/config.json", "path": "vocab_size", "value": 160000},
        "source": {"file": "docs/ADR-004.md", "anchor": "Канонический словарь"},
    }
    messages = sft_block.build_config_drift_trajectory(
        ws, _Noop(), claim=claim, adr_rel="docs/ADR-004.md"
    )
    # system+user (2) + 5 ходов инструментов (по 2 сообщения) + finish (1) = 13.
    assert len(messages) == 13
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    assert len(tool_msgs) == 5
    restored = json.loads((ws / "net" / "config.json").read_text(encoding="utf-8"))
    assert restored["vocab_size"] == 160000
    assert restored["moe_top_k"] == 2  # остальное не тронуто
    assert sft_block.CONFIG_DRIFT_SEQUENCE == (
        "run_gates", "read_file", "read_file", "edit_file", "run_gates", "finish"
    )
