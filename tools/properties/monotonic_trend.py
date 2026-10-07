"""Шаблон ``monotonic_trend`` — ряд не разворачивается против направления.

``up``: значения не убывают (в пределах допуска); ``down``: не возрастают.
Менее двух точек в окне → ``unverified``.
"""

from __future__ import annotations

from .base import Facts, Mutant, Property, Verdict, register, windowed_value


@register
class MonotonicTrend(Property):
    name = "monotonic_trend"
    param_schema = {
        "type": "object",
        "required": ["series", "direction"],
        "properties": {
            "series": {"type": "object", "required": ["fact"]},
            "direction": {"type": "string", "enum": ["up", "down"]},
            "window": {"type": ["object", "null"]},
            "tolerance": {"type": ["number", "null"]},
        },
    }
    levels = {"end_to_end", "component"}

    def _values(self, params: dict, facts: Facts):
        spec = str((params.get("series") or {}).get("fact", ""))
        recs = facts.series(spec)
        vals = [r.get("value") for r in recs
                if isinstance(r.get("value"), (int, float)) and not isinstance(r.get("value"), bool)]
        window = params.get("window")
        if isinstance(window, dict):
            n = window.get("n")
            if n:
                vals = vals[-int(n):]
        return spec, [float(v) for v in vals]

    def evaluate(self, params: dict, facts: Facts) -> Verdict:
        spec, vals = self._values(params, facts)
        if len(vals) < 2:
            return Verdict.unverified(f"ряд {spec}: точек {len(vals)} < 2")
        tol = float(params.get("tolerance") or 0.0)
        direction = str(params.get("direction"))
        for prev, cur in zip(vals, vals[1:]):
            if direction == "up" and cur < prev - tol:
                return Verdict.failed(f"тренд вверх нарушен: {prev} → {cur}", [spec], vals)
            if direction == "down" and cur > prev + tol:
                return Verdict.failed(f"тренд вниз нарушен: {prev} → {cur}", [spec], vals)
        return Verdict.passed(f"тренд {direction} держится на {len(vals)} точках", [spec], vals)

    def mutants(self, params: dict, facts: Facts) -> list[Mutant]:
        spec, _ = self._values(params, facts)
        if not facts.series(spec):
            return []
        tol = float(params.get("tolerance") or 0.0)

        def _reverse(recs: list[dict]) -> list[dict]:
            out = [dict(r) for r in recs]
            numeric = [i for i, r in enumerate(out)
                       if isinstance(r.get("value"), (int, float)) and not isinstance(r.get("value"), bool)]
            if len(numeric) < 2:
                return out
            i, j = numeric[-2], numeric[-1]
            hi = max(out[i]["value"], out[j]["value"])
            lo = min(out[i]["value"], out[j]["value"])
            direction = str(params.get("direction"))
            if direction == "up":
                out[i]["value"], out[j]["value"] = hi + tol + 1.0, lo
            else:
                out[i]["value"], out[j]["value"] = lo, hi + tol + 1.0
            return out

        return [self._mutant(params, facts, spec, "развёрнутый тренд в последнем окне", _reverse)]
