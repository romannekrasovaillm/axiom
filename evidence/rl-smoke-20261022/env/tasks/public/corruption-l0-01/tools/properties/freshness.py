"""Шаблон ``freshness`` — запись факта не старше допуска.

``max_age_h`` — часы. Метка времени не разбирается или запись отсутствует →
``unverified``; устаревшая запись — ``fail``.
"""

from __future__ import annotations

from .base import Facts, Mutant, Property, Verdict, age_hours, register


@register
class Freshness(Property):
    name = "freshness"
    param_schema = {
        "type": "object",
        "required": ["fact", "max_age_h"],
        "properties": {
            "fact": {"type": "string"},
            "max_age_h": {"type": "number"},
        },
    }
    levels = {"end_to_end", "component", "diagnostic"}

    def evaluate(self, params: dict, facts: Facts) -> Verdict:
        spec = str(params.get("fact", ""))
        record = facts.latest(spec)
        if record is None:
            return Verdict.unverified(f"нет факта {spec}")
        if record.get("status") != "ok":
            return Verdict.unverified("факт unverified: " + str(record.get("note", "")))
        age = age_hours(record.get("ts"), facts.now)
        if age is None:
            return Verdict.unverified(f"{spec}: метка времени не разбирается")
        limit = float(params.get("max_age_h"))
        if age > limit:
            return Verdict.failed(f"{spec}: возраст {age:.2f}ч > {limit}ч", [spec], record.get("value"))
        return Verdict.passed(f"{spec}: возраст {age:.2f}ч ≤ {limit}ч", [spec], record.get("value"))

    def mutants(self, params: dict, facts: Facts) -> list[Mutant]:
        spec = str(params.get("fact", ""))
        if facts.latest(spec) is None:
            return []
        limit = float(params.get("max_age_h"))

        def _stale(recs: list[dict]) -> list[dict]:
            from datetime import datetime, timedelta, timezone

            recs = [dict(r) for r in recs]
            last = dict(recs[-1])
            old = datetime.now(timezone.utc) - timedelta(hours=limit * 2.0 + 1.0)
            last["ts"] = old.isoformat(timespec="seconds")
            recs[-1] = last
            return recs

        return [self._mutant(params, facts, spec, "устаревшая запись", _stale)]
