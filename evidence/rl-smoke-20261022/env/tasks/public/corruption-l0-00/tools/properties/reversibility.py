"""Шаблон ``reversibility`` — после отката состояние совпадает с исходным.

``before`` и ``after_revert`` — ссылки на факты-хеши. Неполный откат (байт
отличается) — нарушение; отсутствие любой стороны → ``unverified``.
"""

from __future__ import annotations

from .base import Facts, Mutant, Property, Verdict, ok_value, register


@register
class Reversibility(Property):
    name = "reversibility"
    param_schema = {
        "type": "object",
        "required": ["before", "after_revert"],
        "properties": {
            "before": {"type": "object", "required": ["fact"]},
            "after_revert": {"type": "object", "required": ["fact"]},
        },
    }
    levels = {"end_to_end", "component"}

    def evaluate(self, params: dict, facts: Facts) -> Verdict:
        before_spec = str((params.get("before") or {}).get("fact", ""))
        after_spec = str((params.get("after_revert") or {}).get("fact", ""))
        before, reason = ok_value(facts.latest(before_spec))
        if reason:
            return Verdict.unverified(f"before {before_spec}: {reason}")
        after, reason = ok_value(facts.latest(after_spec))
        if reason:
            return Verdict.unverified(f"after_revert {after_spec}: {reason}")
        evidence = [before_spec, after_spec]
        if str(before) == str(after):
            return Verdict.passed("состояние после отката совпадает с исходным", evidence, before)
        return Verdict.failed("неполный откат: состояние отличается от исходного", evidence, after)

    def mutants(self, params: dict, facts: Facts) -> list[Mutant]:
        after_spec = str((params.get("after_revert") or {}).get("fact", ""))
        record = facts.latest(after_spec)
        if record is None:
            return []
        value = str(record.get("value") or "")

        def _one_byte(recs: list[dict]) -> list[dict]:
            recs = [dict(r) for r in recs]
            last = dict(recs[-1])
            v = str(last.get("value") or value)
            ch = "0" if (v[-1:] != "0") else "1"
            last["value"] = (v[:-1] + ch) if v else "1"
            recs[-1] = last
            return recs

        return [self._mutant(params, facts, after_spec, "неполный откат (один байт отличается)", _one_byte)]
