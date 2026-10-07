"""Шаблон ``determinism`` — один предмет даёт одинаковое значение в разных прогонах.

Меньше ``min_runs`` записей по предмету → ``unverified``: о детерминизме нельзя
судить по одному замеру (ADR-037, окна и допуски).
"""

from __future__ import annotations

from .base import Facts, Mutant, Property, Verdict, register


@register
class Determinism(Property):
    name = "determinism"
    param_schema = {
        "type": "object",
        "required": ["fact"],
        "properties": {
            "fact": {"type": "string"},
            "subject_match": {"type": ["array", "null"]},
            "min_runs": {"type": "integer"},
        },
    }
    levels = {"end_to_end", "component"}

    def evaluate(self, params: dict, facts: Facts) -> Verdict:
        spec = str(params.get("fact", ""))
        min_runs = int(params.get("min_runs") or 2)
        records = [r for r in facts.all_subjects(spec) if r.get("status") == "ok"]
        if len(records) < min_runs:
            return Verdict.unverified(f"прогонов по предмету {spec}: {len(records)} < {min_runs}")
        values = [r.get("value") for r in records]
        if any(v != values[0] for v in values[1:]):
            return Verdict.failed(f"значения одного предмета разошлись: {values}", [spec], values)
        return Verdict.passed(f"{len(records)} прогонов дали одно значение", [spec], values[0])

    def mutants(self, params: dict, facts: Facts) -> list[Mutant]:
        spec = str(params.get("fact", ""))
        records = [r for r in facts.all_subjects(spec) if r.get("status") == "ok"]
        if not records:
            return []
        base = {k: v for k, v in records[-1].items() if not k.startswith("_")}
        first = dict(base)
        second = dict(base)
        first["value"] = 1
        second["value"] = 2

        def _transform(recs: list[dict]) -> list[dict]:
            return [first, second]

        return [self._mutant(params, facts, spec, "два прогона одного предмета разошлись", _transform)]
