"""Шаблон ``liveness`` — счётчик меняется, а не стоит дольше допуска.

Менее двух записей с меткой времени → ``unverified``. Стоящий счётчик или шаг
между записями больше ``max_stall_s`` — нарушение живости.
"""

from __future__ import annotations

from .base import Facts, Mutant, Property, Verdict, register

from datetime import datetime


@register
class Liveness(Property):
    name = "liveness"
    param_schema = {
        "type": "object",
        "required": ["counter", "max_stall_s"],
        "properties": {
            "counter": {"type": "object", "required": ["fact"]},
            "max_stall_s": {"type": "number"},
        },
    }
    levels = {"end_to_end", "component", "diagnostic"}

    def _stamp(self, ts):
        if not isinstance(ts, str):
            return None
        try:
            return datetime.fromisoformat(ts)
        except ValueError:
            return None

    def evaluate(self, params: dict, facts: Facts) -> Verdict:
        spec = str((params.get("counter") or {}).get("fact", ""))
        records = [r for r in facts.series(spec) if r.get("status") == "ok"]
        if len(records) < 2:
            return Verdict.unverified(f"записей счётчика {spec}: {len(records)} < 2")
        prev, last = records[-2], records[-1]
        if prev.get("value") == last.get("value"):
            return Verdict.failed(f"счётчик {spec} не менялся", [spec], last.get("value"))
        t_prev, t_last = self._stamp(prev.get("ts")), self._stamp(last.get("ts"))
        if t_prev and t_last:
            gap = (t_last - t_prev).total_seconds()
            if gap > float(params.get("max_stall_s")):
                return Verdict.failed(
                    f"шаг счётчика {spec} = {gap:.0f}с > {params.get('max_stall_s')}с",
                    [spec], last.get("value"),
                )
        return Verdict.passed("счётчик жив", [spec], last.get("value"))

    def mutants(self, params: dict, facts: Facts) -> list[Mutant]:
        spec = str((params.get("counter") or {}).get("fact", ""))
        records = [r for r in facts.series(spec) if r.get("status") == "ok"]
        if not records:
            return []
        base = {k: v for k, v in records[-1].items() if not k.startswith("_")}
        first = dict(base)
        second = dict(base)
        first["value"] = 1
        second["value"] = 1

        def _stall(recs: list[dict]) -> list[dict]:
            return [first, second]

        return [self._mutant(params, facts, spec, "счётчик не меняется (стой)", _stall)]
