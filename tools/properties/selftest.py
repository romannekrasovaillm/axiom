"""Selftest каталога свойств (дельта P2).

Канонический прогон каждого шаблона: эталон не краснеет, свои мутанты убиваются
(вердикт ``fail``), «нет фактов» даёт ``unverified``, параметры проходят
``param_schema``. Используется тестами ``tools/tests/test_property_<name>.py`` и
исполняемым прогоном ``python3 -m tools.properties.selftest``.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from . import TEMPLATE_NAMES, assert_catalog, get_property
from .base import Facts, validate_params

_TS = "2026-10-07T00:00:00+00:00"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _rec(value: Any, **extra: Any) -> dict:
    rec = {"value": value, "status": "ok", "unit": "", "ts": _TS, "subject": {}}
    rec.update(extra)
    return rec


def write_fixture(root: Path, records: dict, files: dict | None = None) -> Path:
    """Пишет ``<root>/evidence/facts/<S>.jsonl`` и вспомогательные файлы."""
    facts_dir = root / "evidence" / "facts"
    facts_dir.mkdir(parents=True, exist_ok=True)
    by_sensor: dict[str, list[dict]] = {}
    for spec, recs in records.items():
        sensor, fact = spec.split(".", 1)
        for rec in recs:
            r = dict(rec)
            r.setdefault("fact", fact)
            by_sensor.setdefault(sensor, []).append(r)
    for sensor, recs in by_sensor.items():
        text = "\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n"
        (facts_dir / f"{sensor}.jsonl").write_text(text, encoding="utf-8")
    for rel, content in (files or {}).items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return facts_dir


def _canonical() -> dict[str, dict]:
    now = _now()
    return {
        "declared_equals_actual": {
            "params": {"declared": {"file": "cfg.json", "key": "n"}, "actual": {"fact": "S-902.params_total"}},
            "records": {"S-902.params_total": [_rec(100)]},
            "files": {"cfg.json": '{"n": 100}'},
            "expected": "pass",
        },
        "bounds": {
            "params": {"fact": "S-903.snapshot_bytes", "min": 10, "max": 100},
            "records": {"S-903.snapshot_bytes": [_rec(50)]},
            "expected": "pass",
            "boundary": 10,  # граница вплотную не должна краснеть
        },
        "conservation": {
            "params": {"left": ["S-904.a"], "right": ["S-904.b"], "tolerance": 0},
            "records": {"S-904.a": [_rec(30)], "S-904.b": [_rec(30)]},
            "expected": "pass",
        },
        "identity": {
            "params": {"artifact": {"fact": "S-905.tokenizer_sha256"}, "declared": {"file": "man.json", "key": "h"}},
            "records": {"S-905.tokenizer_sha256": [_rec("a" * 64)]},
            "files": {"man.json": json.dumps({"h": "a" * 64})},
            "expected": "pass",
        },
        "determinism": {
            "params": {"fact": "S-906.verdict_identical", "min_runs": 2},
            "records": {"S-906.verdict_identical": [_rec(True), _rec(True)]},
            "expected": "pass",
        },
        "reversibility": {
            "params": {"before": {"fact": "S-907.h_before"}, "after_revert": {"fact": "S-907.h_after"}},
            "records": {"S-907.h_before": [_rec("b" * 64)], "S-907.h_after": [_rec("b" * 64)]},
            "expected": "pass",
        },
        "differential": {
            "params": {"a": {"fact": "S-908.ta"}, "b": {"fact": "S-908.tb"}, "tolerance": None},
            "records": {"S-908.ta": [_rec(5)], "S-908.tb": [_rec(5)]},
            "expected": "pass",
        },
        "metamorphic": {
            "params": {"base": {"fact": "S-909.b"}, "transformed": {"fact": "S-909.t"},
                       "relation": "t == b*k", "k": 3, "tolerance": 0},
            "records": {"S-909.b": [_rec(10)], "S-909.t": [_rec(30)]},
            "expected": "pass",
        },
        "monotonic_trend": {
            "params": {"series": {"fact": "S-910.s"}, "direction": "up", "tolerance": 0},
            "records": {"S-910.s": [_rec(1), _rec(2), _rec(3)]},
            "expected": "pass",
        },
        "liveness": {
            "params": {"counter": {"fact": "S-911.c"}, "max_stall_s": 60},
            "records": {
                "S-911.c": [
                    _rec(1, ts=(now - timedelta(seconds=5)).isoformat(timespec="seconds")),
                    _rec(2, ts=now.isoformat(timespec="seconds")),
                ]
            },
            "expected": "pass",
        },
        "safety": {
            "params": {"forbidden": {"fact": "S-912.n", "op": ">", "value": 1}},
            "records": {"S-912.n": [_rec(0)]},
            "expected": "pass",
        },
        "freshness": {
            "params": {"fact": "S-913.f", "max_age_h": 1},
            "records": {"S-913.f": [_rec(7, ts=now.isoformat(timespec="seconds"))]},
            "expected": "pass",
        },
    }


def selftest_one(name: str) -> list[tuple[str, bool]]:
    """Прогон одного шаблона. Возвращает список ``(метка, прошло)``."""
    template = get_property(name)
    if template is None:
        return [(f"{name}: шаблон не найден", False)]
    spec = _canonical()[name]
    checks: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory(prefix=f"prop-{name}-") as tmp:
        root = Path(tmp)
        facts_dir = write_fixture(root, spec["records"], spec.get("files"))
        facts = Facts(root, out_dir=facts_dir, subject_match=[], subject={}, now=_now())
        checks.append(("params по param_schema", validate_params(template.param_schema, spec["params"]) == []))
        verdict = template.evaluate(spec["params"], facts)
        checks.append((f"эталон: {spec['expected']}", verdict.cls == spec["expected"]))
        if "boundary" in spec:
            record = {"value": spec["boundary"], "status": "ok", "unit": "", "ts": _TS, "subject": {}}
            sub = spec["params"]["fact"]
            boundary_facts = Facts(root, out_dir=facts_dir, subject_match=[], subject={},
                                   overlay={sub: [record]}, now=_now())
            checks.append(("граница вплотную не краснеет", template.evaluate(spec["params"], boundary_facts).cls == "pass"))
        empty = Facts(root, out_dir=root / "empty-facts", subject_match=[], subject={}, now=_now())
        checks.append(("нет фактов → unverified", template.evaluate(spec["params"], empty).cls == "unverified"))
        mutants = template.mutants(spec["params"], facts)
        checks.append(("есть хотя бы один мутант", len(mutants) >= 1))
        for mutant in mutants:
            result = template.evaluate(spec["params"], mutant.apply(facts))
            if mutant.expected == "unverified":
                checks.append((f"мутант «{mutant.description}» → unverified", result.cls == "unverified"))
            else:
                checks.append((f"мутант «{mutant.description}» убит", result.cls == "fail"))
    return checks


def run_all() -> int:
    assert_catalog()
    ok = True
    for name in TEMPLATE_NAMES:
        checks = selftest_one(name)
        passed = all(p for _, p in checks)
        ok = ok and passed
        print(f"[selftest] {'PASS' if passed else 'FAIL'}: шаблон {name}")
        for label, p in checks:
            if not p:
                print(f"    [FAIL] {label}")
    print(f"[selftest] {'PASS' if ok else 'FAIL'}: каталог свойств (12 шаблонов)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(run_all())
