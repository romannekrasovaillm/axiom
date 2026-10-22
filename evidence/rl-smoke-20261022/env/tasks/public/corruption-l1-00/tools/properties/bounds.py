"""Шаблон ``bounds`` — измеренное значение внутри границ.

Границы ``min``/``max`` — доменные; окно/допуск с происхождением — из ADR-037/
ADR-039. Отсутствие факта → ``unverified`` (E-3.5). Значение ровно на границе —
не нарушение.
"""

from __future__ import annotations

from typing import Optional

from .base import Facts, Mutant, Property, Verdict, mutant_fingerprint, register, windowed_value

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

    # -- мутанты: серия достаточной длины, чтобы окно дало значение ----------
    def _baseline(self, params: dict, records: list[dict]) -> float:
        lo, hi = params.get("min"), params.get("max")
        if lo is not None and hi is not None:
            return (float(lo) + float(hi)) / 2.0
        if lo is not None:
            return float(lo) + max(1.0, abs(float(lo)) * 0.1)
        if hi is not None:
            return float(hi) - max(1.0, abs(float(hi)) * 0.1)
        return 0.0

    def _series(self, params: dict, facts: Facts, violating: float) -> list[dict]:
        spec = str(params.get("fact", ""))
        window = params.get("window") if isinstance(params.get("window"), dict) else {}
        n = int(window.get("n") or 0)
        skip = int(window.get("skip_warmup") or 0)
        existing = facts.series(spec)
        template = dict(existing[-1]) if existing else {}
        template.pop("_lineno", None)
        template.pop("_raw", None)
        template.setdefault("fact", spec.split(".", 1)[1] if "." in spec else spec)
        template.setdefault("unit", "")
        template.setdefault("ts", "2026-10-07T00:00:00+00:00")
        template.setdefault("subject", {})
        base = self._baseline(params, existing)
        length = max(len(existing), skip + max(n, 1))
        series = []
        for _ in range(length):
            rec = dict(template)
            rec["value"] = base
            rec["status"] = "ok"
            series.append(rec)
        for i in range(max(n, 1)):
            series[-(i + 1)]["value"] = violating
        return series

    def mutants(self, params: dict, facts: Facts) -> list[Mutant]:
        lo, hi = params.get("min"), params.get("max")
        out: list[Mutant] = []

        def _add(value: float, description: str) -> None:
            spec = str(params.get("fact", ""))
            out.append(Mutant(
                fingerprint=mutant_fingerprint(self.name, params, description),
                description=f"{self.name}: {description}",
                overlay={spec: self._series(params, facts, value)},
            ))

        if lo is not None:
            _add(float(lo) - 1.0, f"ниже min на 1 ({lo})")
            _add(float(lo) - max(abs(float(lo)) * 0.10, 1e-9), f"ниже min на 10% ({lo})")
        if hi is not None:
            _add(float(hi) + 1.0, f"выше max на 1 ({hi})")
            _add(float(hi) + max(abs(float(hi)) * 0.10, 1e-9), f"выше max на 10% ({hi})")
        return out
