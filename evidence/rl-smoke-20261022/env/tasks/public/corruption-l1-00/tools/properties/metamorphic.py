"""Шаблон ``metamorphic`` — соотношение между базой и преобразованным.

Пример: ``t == b * k`` (ускорение, масштаб). Домен задаёт отношение, ``k`` и
допуск. Нечисловые стороны или отсутствие факта → ``unverified``.
"""

from __future__ import annotations

from .base import Facts, Mutant, Property, Verdict, ok_value, register


@register
class Metamorphic(Property):
    name = "metamorphic"
    param_schema = {
        "type": "object",
        "required": ["base", "transformed", "relation", "k"],
        "properties": {
            "base": {"type": "object", "required": ["fact"]},
            "transformed": {"type": "object", "required": ["fact"]},
            "relation": {"type": "string"},
            "k": {"type": "number"},
            "tolerance": {"type": ["number", "null"]},
        },
    }
    levels = {"end_to_end", "component"}

    def _expected(self, base: float, k: float, relation: str) -> float:
        rel = relation.replace(" ", "")
        if rel not in ("t==b*k", "t==k*b", "b*k==t", "k*b==t"):
            raise ValueError(f"неизвестное отношение {relation!r}")
        return base * k

    def evaluate(self, params: dict, facts: Facts) -> Verdict:
        base, reason = ok_value(facts.latest(str((params.get("base") or {}).get("fact", ""))))
        if reason:
            return Verdict.unverified(f"base: {reason}")
        transformed, reason = ok_value(facts.latest(str((params.get("transformed") or {}).get("fact", ""))))
        if reason:
            return Verdict.unverified(f"transformed: {reason}")
        if isinstance(base, bool) or isinstance(transformed, bool) \
                or not isinstance(base, (int, float)) or not isinstance(transformed, (int, float)):
            return Verdict.unverified("metamorphic: стороны не числовые")
        try:
            expected = self._expected(float(base), float(params.get("k")), str(params.get("relation")))
        except ValueError as exc:
            return Verdict.unverified(str(exc))
        tol = float(params.get("tolerance") or 0.0)
        ev = [str((params.get("base") or {}).get("fact", "")), str((params.get("transformed") or {}).get("fact", ""))]
        if abs(float(transformed) - expected) <= tol:
            return Verdict.passed(f"{transformed} ≈ {expected} (±{tol})", ev, transformed)
        return Verdict.failed(f"{transformed} ≉ {expected} (±{tol})", ev, transformed)

    def mutants(self, params: dict, facts: Facts) -> list[Mutant]:
        spec = str((params.get("transformed") or {}).get("fact", ""))
        record = facts.latest(spec)
        if record is None:
            return []
        tol = float(params.get("tolerance") or 0.0)

        def _break(recs: list[dict]) -> list[dict]:
            recs = [dict(r) for r in recs]
            last = dict(recs[-1])
            v = last.get("value")
            base = facts.latest(str((params.get("base") or {}).get("fact", "")))
            if isinstance(v, (int, float)) and base is not None \
                    and isinstance(base.get("value"), (int, float)):
                last["value"] = float(base["value"]) * float(params.get("k")) + tol + 1.0
            else:
                last["value"] = "мутант"
            recs[-1] = last
            return recs

        return [self._mutant(params, facts, spec, "нарушение соотношения t == b*k", _break)]
