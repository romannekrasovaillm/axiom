"""Шаблон ``identity`` — хеш артефакта совпадает с объявленным.

Сравниваются **полные** строки хешей: усечённый хеш (класс INC обрезанного
хеша) — явное нарушение, а не «почти равенство». Отсутствие любой стороны →
``unverified``.
"""

from __future__ import annotations

from typing import Any, Optional

from .base import Facts, Mutant, Property, Verdict, register

_DECLARED = {
    "anyOf": [
        {"type": "object", "required": ["file", "key"]},
        {"type": "object", "required": ["fact"]},
    ]
}


@register
class Identity(Property):
    name = "identity"
    param_schema = {
        "type": "object",
        "required": ["artifact", "declared"],
        "properties": {
            "artifact": {"type": "object", "required": ["fact"]},
            "declared": _DECLARED,
        },
    }
    levels = {"end_to_end", "component"}

    def _declared(self, declared: dict, facts: Facts) -> tuple[Optional[str], Optional[str]]:
        if "file" in declared and "key" in declared:
            import json

            try:
                data = json.loads((facts.root / str(declared["file"])).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None, f"объявление недоступно: {declared['file']}"
            cur: Any = data
            for part in str(declared["key"]).split("."):
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    return None, f"ключ {declared['key']} не найден в {declared['file']}"
            return str(cur), None
        if "fact" in declared:
            from .base import ok_value

            value, reason = ok_value(facts.latest(str(declared["fact"])))
            return (None, reason) if reason else (str(value), None)
        return None, "declared: нет file/key или fact"

    def evaluate(self, params: dict, facts: Facts) -> Verdict:
        from .base import ok_value

        artifact_spec = str((params.get("artifact") or {}).get("fact", ""))
        artifact, reason = ok_value(facts.latest(artifact_spec))
        if reason:
            return Verdict.unverified(f"artifact {artifact_spec}: {reason}")
        declared, reason = self._declared(params.get("declared") or {}, facts)
        if reason:
            return Verdict.unverified(reason)
        artifact, declared = str(artifact), str(declared)
        if len(artifact) != len(declared):
            return Verdict.failed(
                f"усечённый хеш: {len(artifact)} символов против объявленных {len(declared)}",
                [artifact_spec], artifact,
            )
        if artifact != declared:
            return Verdict.failed("хеш артефакта не совпал с объявленным", [artifact_spec], artifact)
        return Verdict.passed("хеш артефакта совпал с объявленным", [artifact_spec], artifact)

    def mutants(self, params: dict, facts: Facts) -> list[Mutant]:
        spec = str((params.get("artifact") or {}).get("fact", ""))
        out: list[Mutant] = []
        if facts.latest(spec) is None:
            return out

        def _one_char(recs: list[dict]) -> list[dict]:
            recs = [dict(r) for r in recs]
            last = dict(recs[-1])
            value = str(last.get("value") or "")
            if not value:
                return recs
            ch = "0" if value[-1] != "0" else "1"
            last["value"] = value[:-1] + ch
            recs[-1] = last
            return recs

        def _truncate(recs: list[dict]) -> list[dict]:
            recs = [dict(r) for r in recs]
            last = dict(recs[-1])
            value = str(last.get("value") or "")
            last["value"] = value[: max(1, len(value) // 2)]
            recs[-1] = last
            return recs

        out.append(self._mutant(params, facts, spec, "замена одного hex-символа", _one_char))
        out.append(self._mutant(params, facts, spec, "усечение хеша", _truncate))
        return out
