"""Экспортёр S-031 spine_coverage: две доли как факты (ADR-038, дельта H2).

Доли — наблюдение, а не цель: порогов нет (ADR-037, отклонённая альтернатива
целевой пропорции). ``share_measurable_claims_with_sensor`` = утверждения с
датчиком / (утверждения с датчиком + кандидаты ``measurable_pending``);
``share_incidents_guarded`` = инциденты со стражем из rule|test / все инциденты;
``untriaged_candidates`` = кандидаты ``--scan-adr`` без решения в триаже.
Качество ``derived``: происхождение — ``inputs`` (sha256 реестров).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from tools.sensors.protocol import BaseExporter, Fact, SensorSpec

REGISTRY_FILES = (
    "model/claims.yaml",
    "model/claims-triage.yaml",
    "evidence/incidents.yaml",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(root: Path) -> dict[str, Any]:
    """Считает две доли и счётчики по реестрам. Без порогов."""
    from tools import check_claims as cc

    root = Path(root)
    claims = cc.load_claims(root)
    triage = cc.load_triage(root) or []
    with_sensor = sum(1 for c in claims if c.get("sensor"))
    measurable_pending = sum(1 for t in triage if t.get("disposition") == "measurable_pending")
    denom_claims = with_sensor + measurable_pending
    incidents = cc.load_incidents(root)
    guarded = sum(
        1 for i in incidents
        if isinstance(i.get("guard"), dict) and ("rule" in i["guard"] or "test" in i["guard"])
    )
    untriaged = cc.scan_adr(root)["unresolved_count"]
    return {
        "share_measurable_claims_with_sensor": (with_sensor / denom_claims if denom_claims else None),
        "share_incidents_guarded": (guarded / len(incidents) if incidents else None),
        "untriaged_candidates": untriaged,
        "measurable_claims_numerator": with_sensor,
        "measurable_claims_denominator": denom_claims,
        "incidents_guarded_numerator": guarded,
        "incidents_denominator": len(incidents),
    }


def _inputs(root: Path) -> list[dict[str, Any]]:
    return [
        {"file": rel, "sha256": _sha256(root / rel)}
        for rel in REGISTRY_FILES
        if (root / rel).is_file()
    ]


class SpineCoverageExporter(BaseExporter):
    SPEC = SensorSpec(
        id="S-031",
        facts=(
            "share_measurable_claims_with_sensor", "share_incidents_guarded", "untriaged_candidates",
            "measurable_claims_numerator", "measurable_claims_denominator",
            "incidents_guarded_numerator", "incidents_denominator",
        ),
        schema={
            "share_measurable_claims_with_sensor": {"unit": "fraction", "quality": "derived", "level": "end_to_end"},
            "share_incidents_guarded": {"unit": "fraction", "quality": "derived", "level": "end_to_end"},
            "untriaged_candidates": {"unit": "count", "quality": "derived", "level": "diagnostic"},
            "measurable_claims_numerator": {"unit": "count", "quality": "derived", "level": "diagnostic"},
            "measurable_claims_denominator": {"unit": "count", "quality": "derived", "level": "diagnostic"},
            "incidents_guarded_numerator": {"unit": "count", "quality": "derived", "level": "diagnostic"},
            "incidents_denominator": {"unit": "count", "quality": "derived", "level": "diagnostic"},
        },
        level="end_to_end",
        raw={"note": "derived: происхождение даёт inputs (sha256 реестров)"},
        context=("repo",),
        pack="spine",
    )

    def collect(self, subject: dict[str, Any], *, root: Any = None, **_: Any) -> list[Fact]:
        repo_root = Path(root) if root is not None else Path(__file__).resolve().parents[4]
        values = measure(repo_root)
        inputs = _inputs(repo_root)
        facts: list[Fact] = []
        for fact in self.SPEC.facts:
            value = values.get(fact)
            if value is None:
                facts.append(self.unavailable(fact, subject, "нет данных для доли (пустой знаменатель)"))
            else:
                facts.append(self.fact(fact, value, subject=subject, inputs=inputs,
                                       method="check_claims: реестры claims/triage/incidents (ADR-038, дельта H2)"))
        return facts


EXPORTER = SpineCoverageExporter()
