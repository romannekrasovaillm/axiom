"""Шаблон ``differential`` — две величины совпадают в пределах допуска.

Используется и для паритетных тестов (скан/WY/UT, FLA), и для теневого сравнения
стражей (дельта R2): ``a`` — исходный страж, ``b`` — экземпляр шаблона. Одна
сторона отсутствует → ``unverified`` (не ``pass``).
"""

from __future__ import annotations

from typing import Any, Optional

from .base import Facts, Mutant, Property, Verdict, ok_value, register


@register
class Differential(Property):
    name = "differential"
    param_schema = {
        "type": "object",
        "required": ["a", "b"],
        "properties": {
            "a": {"type": "object", "required": ["fact"]},
            "b": {"type": "object", "required": ["fact"]},
            "tolerance": {"type": ["number", "null"]},
        },
    }
    levels = {"end_to_end", "component"}

    def evaluate(self, params: dict, facts: Facts) -> Verdict:
        a_spec = str((params.get("a") or {}).get("fact", ""))
        b_spec = str((params.get("b") or {}).get("fact", ""))
        a, reason = ok_value(facts.latest(a_spec))
        if reason:
            return Verdict.unverified(f"сторона a ({a_spec}): {reason}")
        b, reason = ok_value(facts.latest(b_spec))
        if reason:
            return Verdict.unverified(f"сторона b ({b_spec}): {reason}")
        tol = params.get("tolerance")
        evidence = [a_spec, b_spec]
        if tol is not None and isinstance(a, (int, float)) and isinstance(b, (int, float)) \
                and not isinstance(a, bool) and not isinstance(b, bool):
            if abs(float(a) - float(b)) <= float(tol):
                return Verdict.passed(f"{a} ≈ {b} (±{tol})", evidence, abs(float(a) - float(b)))
            return Verdict.failed(f"{a} ≉ {b} (±{tol})", evidence, abs(float(a) - float(b)))
        if a == b:
            return Verdict.passed(f"{a!r} == {b!r}", evidence, a)
        return Verdict.failed(f"{a!r} != {b!r}", evidence, a)

    def mutants(self, params: dict, facts: Facts) -> list[Mutant]:
        a_spec = str((params.get("a") or {}).get("fact", ""))
        b_spec = str((params.get("b") or {}).get("fact", ""))
        tol = params.get("tolerance")
        out: list[Mutant] = []
        record = facts.latest(a_spec)
        if record is not None:
            value = record.get("value")
            delta = (float(tol) + 1.0) if isinstance(tol, (int, float)) else None

            def _diverge(recs: list[dict], value: Any = value, delta: Optional[float] = delta) -> list[dict]:
                recs = [dict(r) for r in recs]
                last = dict(recs[-1])
                if isinstance(value, (int, float)) and not isinstance(value, bool) and delta is not None:
                    last["value"] = float(value) + delta
                else:
                    last["value"] = "мутант"
                recs[-1] = last
                return recs

            out.append(self._mutant(params, facts, a_spec, "расхождение за допуском", _diverge))
            out.append(self._mutant(params, facts, b_spec, "одна сторона отсутствует (unverified)",
                                    lambda recs: [], expected="unverified"))
        return out
