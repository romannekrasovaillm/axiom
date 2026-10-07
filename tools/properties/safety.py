"""Шаблон ``safety`` — запрещённое состояние отсутствует.

``forbidden`` = ``{fact, op, value}``: если измеренное значение удовлетворяет
условию, состояние запрещено → ``fail``. Отсутствие факта → ``unverified``
(нельзя объявить безопасность без замера).
"""

from __future__ import annotations

from .base import Facts, Mutant, Property, Verdict, ok_value, register


@register
class Safety(Property):
    name = "safety"
    param_schema = {
        "type": "object",
        "required": ["forbidden"],
        "properties": {
            "forbidden": {
                "type": "object",
                "required": ["fact", "op", "value"],
                "properties": {
                    "fact": {"type": "string"},
                    "op": {"type": "string", "enum": ["==", "!=", ">", ">=", "<", "<="]},
                    "value": {},
                },
            }
        },
    }
    levels = {"end_to_end", "component"}

    def _holds(self, op: str, value, target) -> bool:
        try:
            if op == "==":
                return value == target
            if op == "!=":
                return value != target
            if op == ">":
                return value > target
            if op == ">=":
                return value >= target
            if op == "<":
                return value < target
            if op == "<=":
                return value <= target
        except TypeError:
            return False
        return False

    def evaluate(self, params: dict, facts: Facts) -> Verdict:
        forbidden = params.get("forbidden") or {}
        spec = str(forbidden.get("fact", ""))
        value, reason = ok_value(facts.latest(spec))
        if reason:
            return Verdict.unverified(f"{spec}: {reason}")
        if self._holds(str(forbidden.get("op")), value, forbidden.get("value")):
            return Verdict.failed(
                f"запрещённое состояние: {spec} {forbidden.get('op')} {forbidden.get('value')!r}",
                [spec], value,
            )
        return Verdict.passed(f"запрещённого состояния нет ({spec})", [spec], value)

    def mutants(self, params: dict, facts: Facts) -> list[Mutant]:
        forbidden = params.get("forbidden") or {}
        spec = str(forbidden.get("fact", ""))
        op = str(forbidden.get("op"))
        target = forbidden.get("value")

        def _violate(recs: list[dict]) -> list[dict]:
            recs = [dict(r) for r in recs]
            last = dict(recs[-1])
            if op == "==":
                last["value"] = target
            elif op == "!=":
                last["value"] = (float(target) + 1.0) if isinstance(target, (int, float)) else "__mutant__"
            elif op == ">":
                last["value"] = (float(target) + 1.0) if isinstance(target, (int, float)) else target
            elif op == ">=":
                last["value"] = target
            elif op == "<":
                last["value"] = (float(target) - 1.0) if isinstance(target, (int, float)) else target
            else:  # "<="
                last["value"] = target
            recs[-1] = last
            return recs

        if facts.latest(spec) is None:
            return []
        return [self._mutant(params, facts, spec, "запрещённое состояние присутствует", _violate)]
