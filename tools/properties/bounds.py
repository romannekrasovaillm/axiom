"""Шаблон ``bounds`` — измеренное значение внутри границ.

Границы ``min``/``max`` — доменные; окно/допуск с происхождением — из ADR-037/
ADR-039. Отсутствие факта → ``unverified`` (E-3.5). Значение ровно на границе —
не нарушение.
"""

from __future__ import annotations

from typing import Optional

from .base import Facts, Mutant, Property, Verdict, ok_value, register, windowed_value

_BOUND = {"type": ["number", "null"]}


@register
class Bounds(Property):
    name = "bounds"
    param_schema = {
        "type": "object",
        "required": ["fact"],
        "properties": {
            "fact": {"type": "string"},
            "min": _BOUND,
            "max": _BOUND,
            "window": {"type": ["object", "null"]},
            "tolerance": {"type": ["number", "null"]},
            "tolerance_source": {"type": ["object", "null"]},
        },
    }
    levels = {"end_to_end", "component"}

    def _value(self, spec: str, params: dict, facts: Facts) -> tuple[Optional[float], Optional[str]]:
        records = facts.series(spec)
        if not records:
            return None, f"нет факта {spec}"
        if records[-1].get("status") != "ok":
            return None, "факт unverified: " + str(records[-1].get("note", ""))
        value = windowed_value(records, params.get("window"))
        if value is None:
            return None, f"значение окна {spec} не числовое"
        return value, None

    def evaluate(self, params: dict, facts: Facts) -> Verdict:
        spec = str(params.get("fact", ""))
        value, reason = self._value(spec, params, facts)
        if reason:
            return Verdict.unverified(reason)
        lo = params.get("min")
        hi = params.get("max")
        if lo is None and hi is None:
            return Verdict.unverified("bounds: не заданы min/max")
        if lo is not None and value < float(lo):
            return Verdict.failed(f"{value!r} < min {lo!r}", [spec], value)
        if hi is not None and value > float(hi):
            return Verdict.failed(f"{value!r} > max {hi!r}", [spec], value)
        return Verdict.passed(f"{lo!r} <= {value!r} <= {hi!r}", [spec], value)

    def mutants(self, params: dict, facts: Facts) -> list[Mutant]:
        spec = str(params.get("fact", ""))
        out: list[Mutant] = []
        lo, hi = params.get("min"), params.get("max")
        if lo is not None:
            out.append(self._mutant(params, facts, spec, f"ниже min на 1 ({lo})",
                                    self._make(lo - 1.0)))
            out.append(self._mutant(params, facts, spec, f"ниже min на 10% ({lo})",
                                    self._make(float(lo) - max(abs(float(lo)) * 0.10, 1e-9))))
        if hi is not None:
            out.append(self._mutant(params, facts, spec, f"выше max на 1 ({hi})",
                                    self._make(hi + 1.0)))
            out.append(self._mutant(params, facts, spec, f"выше max на 10% ({hi})",
                                    self._make(float(hi) + max(abs(float(hi)) * 0.10, 1e-9))))
        return out

    @staticmethod
    def _make(value: float):
        def _transform(recs: list[dict]) -> list[dict]:
            recs = [dict(r) for r in recs]
            last = dict(recs[-1])
            last["value"] = value
            recs[-1] = last
            return recs

        return _transform
