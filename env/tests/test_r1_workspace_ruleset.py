"""R-1' (§7): снапшот несёт ПОЛНЫЙ кейсовый ruleset; вердикт машинонезависим
через фильтр инфраструктурных правил (:mod:`env.verifier`).

Ревизия R-1 (E-2.5): подмена CONSTRAINTS.yaml на workspace-ruleset отклонена —
ломала trace (12× ad-not-verified: 12 model/AD ссылаются на case-скоуп правила,
отсутствовавшие в 5-правильном наборе; §9: редукция R<10 запрещена) и делала
атомы порчи невидимыми песочнице. R-1' возвращает полный ruleset в снапшот, а
машинонезависимость даёт фильтр: ``command_succeeds``-правила (C-032…C-046)
проверяют контур ВНЕ воркспейса (tools/, evidence/, .arch-handoff, стенд GB10)
и в изоляции неисполнимы.

Тесты (а)–(д) handoff E-2.6. Требуют бинаря arch-ml (``arch_ml`` fixture →
skip с явной причиной при отсутствии).

Оговорка о кейсе: тесты исполняются на состоянии кейса в границах задачи
(baseline) — фикстура ``case_dir`` (conftest) исключает посторонние
пост-baseline файлы (ADR-030), нарушающие C-001/C-002 вне зоны задачи E-2.6.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from env import corruption
from env.util import EMPTY_HIDDEN_SHA256, copy_case_snapshot
from env.verifier import (
    EXCLUDED_INFRA_RULES,
    detect_excluded_infra_rules,
    filter_fitness_report,
    verify,
)

#: Спека restore-gates: задачных тестов нет (``true``), проверяются только гейты.
SPEC = {
    "id": "probe-r1",
    "source": "corruption",
    "objective": {"kind": "restore-gates", "tests_cmd": "true"},
    "verifier": {
        "constraints": "CONSTRAINTS.yaml",
        "spine": True,
        "trace": True,
        "hidden_constraints_sha256": EMPTY_HIDDEN_SHA256,
    },
    "max_tokens": 131072,
}


def _snapshot(case_dir: Path, tmp_path: Path) -> Path:
    """Изолированный снапшот кейса baseline (для порчи/восстановления)."""
    dst = tmp_path / "case"
    copy_case_snapshot(case_dir, dst)
    return dst


def test_snapshot_carries_full_case_ruleset(case_dir, tmp_path):
    """Снапшот несёт ПОЛНЫЙ ruleset (байт-в-байт кейсовый), не редукцию."""
    dst = tmp_path / "snap"
    copy_case_snapshot(case_dir, dst)
    assert (dst / "CONSTRAINTS.yaml").read_bytes() == (case_dir / "CONSTRAINTS.yaml").read_bytes()
    data = yaml.safe_load((dst / "CONSTRAINTS.yaml").read_text(encoding="utf-8"))
    assert len(data["constraints"]) > 10, "редукция R<10 запрещена (§9)"


def test_excluded_infra_rules_detected_dynamically(case_dir):
    """Динамический детектор инфраструктурных ``command_succeeds`` == реестровый пин (16).

    ADR-036 (дельта D2): C-048 (``--evaluate``, факты в ``evidence/``) —
    инфраструктурное правило; C-047 (сверка деклараций, без сети/GPU) помечено
    ``infra: false`` и в вердикте остаётся.
    """
    detected = detect_excluded_infra_rules(case_dir / "CONSTRAINTS.yaml")
    assert detected == EXCLUDED_INFRA_RULES
    assert len(detected) == 16
    assert "C-048" in EXCLUDED_INFRA_RULES
    assert "C-047" not in EXCLUDED_INFRA_RULES


def test_clean_case_verdict_passes(case_dir, tmp_path, arch_ml):
    """(а) чистый кейс → verdict.passed ПОЛНОСТЬЮ (fitness после фильтра ∧ spine ∧ trace)."""
    ws = _snapshot(case_dir, tmp_path)
    v = verify(SPEC, ws, bin=arch_ml)
    assert v.fitness.passed, [it["rule"] for it in v.fitness.errors]
    assert v.spine.passed, [it["rule"] for it in v.spine.errors]
    assert v.trace.passed, [it["rule"] for it in v.trace.errors]
    assert v.passed, {
        "fitness": v.fitness.passed,
        "spine": v.spine.passed,
        "trace": v.trace.passed,
        "excluded": len(v.excluded_violations),
    }
    # Инфра-находки не пусты (стражи вне песочницы краснеют) и отфильтрованы.
    assert v.excluded_violations, "фильтр должен был исключить инфра-правила"
    assert all(
        it["rule"] in EXCLUDED_INFRA_RULES
        or it["rule"] in {c["name"] for c in _rules_by_id(case_dir)}
        for it in v.excluded_violations
    )


def test_corrupted_case_verdict_fails(case_dir, tmp_path, arch_ml):
    """(б) порченный кейс (remove_adr_section ∩ break_affects) → вердикт красный."""
    clean = _snapshot(case_dir, tmp_path)
    ws = tmp_path / "corrupted"
    damages = corruption.corrupt(clean, ws, seed=42, level="L1")
    kinds = {d.kind for d in damages}
    assert "remove_adr_section" in kinds and "break_affects" in kinds, kinds
    v = verify(SPEC, ws, bin=arch_ml)
    assert not v.passed


def test_restored_case_verdict_passes(case_dir, tmp_path, arch_ml):
    """(в) восстановление по Damage-листу → вердикт зелёный."""
    clean = _snapshot(case_dir, tmp_path)
    ws = tmp_path / "restored"
    damages = corruption.corrupt(clean, ws, seed=42, level="L1")
    for d in damages:
        corruption.revert_damage(ws, clean, d)
    v = verify(SPEC, ws, bin=arch_ml)
    assert v.passed, {
        "fitness": v.fitness.passed,
        "spine": v.spine.passed,
        "trace": v.trace.passed,
        "fitness_errors": [it["rule"] for it in v.fitness.errors][:5],
    }


def test_filter_removes_infra_violation_but_keeps_it_transparent(case_dir):
    """(г) инфра-нарушение (rule C-040) исчезает из вердикта, видно в excluded_violations."""
    report = {
        "passed": False,
        "issues": [
            {"rule": "C-040", "file": "ws", "line": 0, "message": "stand", "severity": "error"},
            {"rule": "C-007", "file": "docs/x.md", "line": 1, "message": "TODO", "severity": "error"},
        ],
        "summary": "x",
    }
    filtered, excluded = filter_fitness_report(report, case_dir / "CONSTRAINTS.yaml")
    # C-040 удалён из issues и не влияет на passed; C-007 остаётся (content-правило).
    assert [it["rule"] for it in filtered["issues"]] == ["C-007"]
    assert filtered["passed"] is False  # C-007 — error
    assert [it["rule"] for it in excluded] == ["C-040"]
    assert filtered["excluded_violations"] == excluded
    assert "C-040" in filtered["excluded_infra_rules"]
    assert EXCLUDED_INFRA_RULES == tuple(filtered["excluded_infra_rules"])

    # Только инфра-нарушение → после фильтра passed True (raw_passed=False сохранён).
    only_infra = {"passed": False, "issues": [dict(report["issues"][0])]}
    f2, _ = filter_fitness_report(only_infra, case_dir / "CONSTRAINTS.yaml")
    assert f2["passed"] is True
    assert f2["raw_passed"] is False


def test_filter_neutralises_performance_roofline_rule_c046(case_dir):
    """C-046 (performance-roofline, стенд/KPI) — нейтрален в песочнице: не в вердикте."""
    assert "C-046" in EXCLUDED_INFRA_RULES
    report = {
        "passed": False,
        "issues": [
            {"rule": "C-046", "file": "ws", "line": 0, "message": "roofline",
             "severity": "error"},
        ],
    }
    filtered, excluded = filter_fitness_report(report, case_dir / "CONSTRAINTS.yaml")

    assert filtered["issues"] == []
    assert filtered["passed"] is True
    assert filtered["raw_passed"] is False
    assert [it["rule"] for it in excluded] == ["C-046"]


def test_filter_is_deterministic(case_dir):
    """(д) фильтр детерминирован: одинаковый вход → одинаковый выход (байт-в-байт)."""
    report = {
        "passed": False,
        "issues": [
            {"rule": "C-040", "file": "ws", "line": 0, "message": "m", "severity": "error"},
            {"rule": "C-007", "file": "a.md", "line": 1, "message": "m", "severity": "error"},
        ],
    }
    c = case_dir / "CONSTRAINTS.yaml"
    a, _ = filter_fitness_report(report, c)
    b, _ = filter_fitness_report(report, c)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_verifier_report_excludes_infra_on_clean(case_dir, tmp_path, arch_ml):
    """Интеграционно: _run_fitness на чистом кейсе даёт passed и excluded_violations."""
    from env.verifier import _run_fitness

    ws = _snapshot(case_dir, tmp_path)
    gate, report = _run_fitness(ws, ws / "CONSTRAINTS.yaml", arch_ml)
    assert gate.passed is True
    assert report["excluded_violations"], "инфра-нарушения видны в excluded_violations"
    assert set(report["excluded_infra_rules"]) == set(EXCLUDED_INFRA_RULES)


def _rules_by_id(case_dir: Path) -> list[dict]:
    data = yaml.safe_load((case_dir / "CONSTRAINTS.yaml").read_text(encoding="utf-8"))
    return data["constraints"]
