"""Шаблон ``conservation`` — сумма левой стороны сохраняется в правой.

``left``/``right`` — списки ссылок на факты; ``tolerance`` — абсолютный допуск.
Недостача хотя бы одного слагаемого → ``unverified`` (не молчаливый ``pass``).
"""

from __future__ import annotations

from typing import Optional

from .base import Facts, Mutant, Property, Verdict, ok_value, register


@register
class Conservation(Property):
    name = "conservation"
    param_schema = {
        "type": "object",
        "required": ["left", "right"],
        "properties": {
            "left": {"type": "array", "items": {"type": "string"}},
            "right": {"type": "array", "items": {"type": "string"}},
            "tolerance": {"type": ["number", "null"]},
        },
    }
    levels = {"end_to_end", "component"}

    def _sum(self, specs: list, facts: Facts) -> tuple[Optional[float], Optional[str]]:
        total = 0.0
        for spec in specs:
            value, reason = ok_value(facts.latest(str(spec)))
            if reason:
                return None, f"{spec}: {reason}"
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None, f"{spec}: значение не числовое"
            total += float(value)
        return total, None

    def evaluate(self, params: dict, facts: Facts) -> Verdict:
        left, reason = self._sum(params.get("left") or [], facts)
        if reason:
            return Verdict.unverified(f"левая сторона: {reason}")
        right, reason = self._sum(params.get("right") or [], facts)
        if reason:
            return Verdict.unverified(f"правая сторона: {reason}")
        tol = float(params.get("tolerance") or 0.0)
        evidence = list(params.get("left") or []) + list(params.get("right") or [])
        if abs(left - right) <= tol:
            return Verdict.passed(f"{left} == {right} (±{tol})", evidence, left - right)
        return Verdict.failed(f"{left} != {right} (±{tol})", evidence, left - right)

    def mutants(self, params: dict, facts: Facts) -> list[Mutant]:
        left = [str(s) for s in (params.get("left") or [])]
        right = [str(s) for s in (params.get("right") or [])]
        tol = float(params.get("tolerance") or 0.0)
        out: list[Mutant] = []
        if left:
            spec = left[0]

            def _shift(recs: list[dict], spec: str = spec, tol: float = tol) -> list[dict]:
                recs = [dict(r) for r in recs]
                last = dict(recs[-1])
                v = last.get("value")
                last["value"] = (float(v) if isinstance(v, (int, float)) else 0.0) + max(tol * 2.0, tol + 1.0)
                recs[-1] = last
                return recs

            out.append(self._mutant(params, facts, spec, "сдвиг стороны за допуском", _shift))
            out.append(self._mutant(params, facts, spec, "потеря одного слагаемого",
                                    lambda recs: self._zero(recs)))
        return out

    @staticmethod
    def _zero(recs: list[dict]) -> list[dict]:
        recs = [dict(r) for r in recs]
        last = dict(recs[-1])
        last["value"] = 0.0
        recs[-1] = last
        return recs
