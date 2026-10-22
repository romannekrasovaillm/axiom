"""Шаблон ``declared_equals_actual`` — декларация совпадает с измеренным.

Универсальное свойство: то, что система *объявляет* (ключ файла, другой факт или
литерал), совпадает с тем, что *измерено* фактом. Домен задаёт пару и допуск.
"""

from __future__ import annotations

from typing import Any, Optional

from .base import Facts, Mutant, Property, Verdict, register

_DECLARED = {
    "anyOf": [
        {"type": "object", "required": ["file", "key"]},
        {"type": "object", "required": ["fact"]},
        {"type": "object", "required": ["value"]},
    ]
}


@register
class DeclaredEqualsActual(Property):
    name = "declared_equals_actual"
    param_schema = {
        "type": "object",
        "required": ["declared", "actual"],
        "properties": {
            "declared": _DECLARED,
            "actual": {"type": "object", "required": ["fact"]},
            "tolerance": {"type": ["number", "null"]},
        },
    }
    levels = {"end_to_end", "component"}

    def _declared(self, declared: dict, facts: Facts) -> tuple[Optional[Any], Optional[str]]:
        if not isinstance(declared, dict):
            return None, "declared: ожидался объект"
        if "file" in declared and "key" in declared:
            import json

            path = facts.root / str(declared["file"])
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None, f"декларация недоступна: {declared['file']}"
            cur: Any = data
            for part in str(declared["key"]).split("."):
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    return None, f"декларация {declared['key']} не найдена в {declared['file']}"
            return cur, None
        if "fact" in declared:
            from .base import ok_value

            return ok_value(facts.latest(str(declared["fact"])))
        if "value" in declared:
            return declared["value"], None
        return None, "declared: нет file/key, fact или value"

    def evaluate(self, params: dict, facts: Facts) -> Verdict:
        declared, reason = self._declared(params.get("declared") or {}, facts)
        if reason:
            return Verdict.unverified(reason)
        from .base import ok_value, parse_spec

        actual_spec = str((params.get("actual") or {}).get("fact", ""))
        try:
            parse_spec(actual_spec)
        except Exception as exc:  # noqa: BLE001
            return Verdict.unverified(f"actual: {exc}")
        actual, reason = ok_value(facts.latest(actual_spec))
        if reason:
            return Verdict.unverified(f"actual {actual_spec}: {reason}")
        tol = params.get("tolerance")
        if tol is not None and isinstance(declared, (int, float)) and isinstance(actual, (int, float)) \
                and not isinstance(declared, bool) and not isinstance(actual, bool):
            ok = abs(float(declared) - float(actual)) <= float(tol)
        else:
            ok = declared == actual
        ev = [actual_spec]
        if ok:
            return Verdict.passed(f"declared={declared!r} == actual={actual!r}", ev, actual)
        return Verdict.failed(f"declared={declared!r} != actual={actual!r}", ev, actual)

    def mutants(self, params: dict, facts: Facts) -> list[Mutant]:
        actual_spec = str((params.get("actual") or {}).get("fact", ""))
        out: list[Mutant] = []
        try:
            last = facts.latest(actual_spec)
        except Exception:  # noqa: BLE001
            return out
        if last is None:
            return out
        value = last.get("value")

        def _relace(recs: list[dict]) -> list[dict]:
            recs = [dict(r) for r in recs]
            r = dict(recs[-1])
            r["value"] = (value + 1) if isinstance(value, (int, float)) and not isinstance(value, bool) else "мутант"
            recs[-1] = r
            return recs

        def _type(recs: list[dict]) -> list[dict]:
            recs = [dict(r) for r in recs]
            r = dict(recs[-1])
            r["value"] = str(value)
            recs[-1] = r
            return recs

        out.append(self._mutant(params, facts, actual_spec, "actual: соседнее значение", _relace))
        out.append(self._mutant(params, facts, actual_spec, "actual: смена типа", _type))
        return out
