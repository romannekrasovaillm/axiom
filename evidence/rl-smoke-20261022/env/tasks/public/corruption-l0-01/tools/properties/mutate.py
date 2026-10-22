"""Мутационное тестирование стражей (ADR-039, дельта M).

Для каждого утверждения с ``property`` генерируются мутанты шаблона над текущими
фактами (overlay, без записи в ``evidence/``). Исходы: мутант **убит** — вердикт
``fail``; **выжил** — ``pass``; **ушёл в unverified** — учитывается отдельно и
убийством не считается. Мутационный счёт = убитые / все неэквивалентные мутанты
(эквивалентные — раздел ``equivalent_mutants`` в ``model/properties.yaml``).

Для ``wrapped``-стражей используются их собственные ``--selftest``-мутанты; если
страж не публикует мутационный счёт — ``unverified`` с пометкой.

Факты: S-039 ``mutation_score`` (по утверждению и правилу), S-040
``mutation_summary`` (распределение по шаблонам, список выживших). Мета-правило
M4: утверждение из ``requires`` гейта или правила ``severity ≥ high`` обязано
иметь ``mutation_score = 1,0``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from tools.properties import Facts, get_property
from tools.properties.base import load_properties_registry


def _cc():
    import tools.check_claims as cc  # ленивый импорт: без цикла на импорте модуля

    return cc


def equivalent_fingerprints(root: str | Path) -> dict[str, str]:
    data = load_properties_registry(root)
    out: dict[str, str] = {}
    for entry in data.get("equivalent_mutants") or []:
        if isinstance(entry, dict) and entry.get("fingerprint"):
            out[str(entry["fingerprint"])] = str(entry.get("reason") or "")
    return out


def claim_mutations(claim: dict, root: Path, out_dir: Optional[Path], equivalent: dict[str, str]) -> Optional[dict]:
    template = get_property(str(claim.get("property")))
    if template is None:
        return None
    # Мутант проверяет предикат шаблона, а не привязку предмета: фильтр предмета
    # не участвует (иначе утверждение с несовпавшим предметом не тестируемо).
    facts = Facts(root, out_dir=out_dir, subject_match=[], subject={})
    mutants = template.mutants(claim.get("params") or {}, facts)
    killed = survived = unverified = equiv = 0
    survivors: list[dict] = []
    for mutant in mutants:
        if mutant.fingerprint in equivalent:
            equiv += 1
            continue
        verdict = template.evaluate(claim.get("params") or {}, mutant.apply(facts))
        if verdict.cls == "fail":
            killed += 1
        elif verdict.cls == "pass":
            survived += 1
            survivors.append({"fingerprint": mutant.fingerprint, "mutation": mutant.description})
        else:
            unverified += 1
    non_equivalent = len(mutants) - equiv
    score = (killed / non_equivalent) if non_equivalent else None
    return {
        "claim": claim.get("id"), "rule": claim.get("rule"), "property": template.name,
        "killed": killed, "survived": survived, "unverified": unverified,
        "equivalent": equiv, "non_equivalent": non_equivalent,
        "mutation_score": score, "survivors": survivors,
    }


def wrapped_guard_mutations(root: str | Path) -> dict[str, dict]:
    """Мутационный счёт ``wrapped``-стражей по их собственным ``--selftest`` (M1).

    Страж запускается как есть; из вывода берётся опубликованный счёт вида
    ``N/N`` (убито/всего). Если страж счёт не публикует — ``unverified`` с
    пометкой, а не выдуманное число.
    """
    import re
    import subprocess
    import sys

    root = Path(root).resolve()
    cc = _cc()
    rules = cc.load_rule_properties(root) or []
    out: dict[str, dict] = {}
    for entry in rules:
        if entry.get("migration") != "wrapped":
            continue
        rule = str(entry.get("rule"))
        command = str(entry.get("probe") or "")
        # Команду берём из CONSTRAINTS (реестр правил), как у C-047/C-048.
        text = (root / "CONSTRAINTS.yaml").read_text(encoding="utf-8")
        m = re.search(rf"-\s*id:\s*{re.escape(rule)}\b(.*?)(?=\n\s*-\s*id:|\Z)", text, re.S)
        if m:
            mc = re.search(r"command:\s*'(.+)'", m.group(1))
            if mc:
                command = mc.group(1)
        if not command:
            out[rule] = {"verdict": "unverified", "reason": "команда стража не найдена"}
            continue
        cmd = command if not command.startswith("python3 ") else sys.executable + " " + command[len("python3 "):]
        try:
            proc = subprocess.run(cmd, shell=True, cwd=str(root), capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            out[rule] = {"verdict": "unverified", "reason": "таймаут selftest"}
            continue
        found = re.findall(r"(\d+)\s*/\s*(\d+)", (proc.stdout or "") + (proc.stderr or ""))
        if found:
            killed, total = (int(x) for x in found[-1])
            out[rule] = {"verdict": "ok", "killed": killed, "total": total,
                         "mutation_score": (killed / total) if total else None}
        else:
            out[rule] = {"verdict": "unverified",
                         "reason": "selftest не публикует мутационный счёт"}
    return out


def mutation_report(root: str | Path, out_dir: Optional[Path] = None) -> dict[str, Any]:
    root = Path(root).resolve()
    cc = _cc()
    equivalent = equivalent_fingerprints(root)
    by_claim: dict[str, dict] = {}
    by_template: dict[str, dict] = {}
    survivors: list[dict] = []
    for claim in cc.load_claims(root):
        if not claim.get("property"):
            continue
        entry = claim_mutations(claim, root, out_dir, equivalent)
        if entry is None:
            continue
        by_claim[str(entry["claim"])] = entry
        agg = by_template.setdefault(entry["property"], {"killed": 0, "non_equivalent": 0, "claims": 0})
        agg["killed"] += entry["killed"]
        agg["non_equivalent"] += entry["non_equivalent"]
        agg["claims"] += 1
        survivors.extend({"claim": entry["claim"], **s} for s in entry["survivors"])
    # По правилам: агрегируем утверждения, ссылающиеся на правило.
    by_rule: dict[str, dict] = {}
    for entry in by_claim.values():
        rule = entry.get("rule")
        if not rule:
            continue
        agg = by_rule.setdefault(str(rule), {"killed": 0, "non_equivalent": 0, "claims": []})
        agg["killed"] += entry["killed"]
        agg["non_equivalent"] += entry["non_equivalent"]
        agg["claims"].append(entry["claim"])
    for rule, agg in by_rule.items():
        agg["mutation_score"] = (agg["killed"] / agg["non_equivalent"]) if agg["non_equivalent"] else None
    for name, agg in by_template.items():
        agg["mutation_score"] = (agg["killed"] / agg["non_equivalent"]) if agg["non_equivalent"] else None
    return {
        "by_claim": by_claim,
        "by_rule": by_rule,
        "by_template": by_template,
        "survivors": survivors,
        "equivalent": list(equivalent),
        "wrapped": wrapped_guard_mutations(root),
    }


def gate_violations(root: str | Path, out_dir: Optional[Path] = None) -> list[str]:
    """Мета-правило M4: гейтовые/высокие утверждения обязаны иметь score 1.0."""
    root = Path(root).resolve()
    cc = _cc()
    report = mutation_report(root, out_dir)
    gated = cc.gate_claim_ids(root)
    severities = cc.rule_severities(root)
    errors: list[str] = []
    for claim in cc.load_claims(root):
        cid = str(claim.get("id"))
        rule = claim.get("rule")
        high = bool(rule and str(severities.get(str(rule), "")).lower() in ("high", "critical"))
        if cid not in gated and not high:
            continue
        if not claim.get("property"):
            # Утверждение без шаблона (presence) — мутационный счёт неприменим,
            # находка выносится в open_questions, а не в красный M4.
            continue
        entry = report["by_claim"].get(cid)
        if entry is None:
            errors.append(f"{cid}: мутационный счёт не посчитан (гейтовое/высокое утверждение)")
            continue
        if entry["mutation_score"] != 1.0:
            fmt = ", ".join(f"{s['fingerprint'][:12]}…" for s in entry["survivors"]) or "unverified"
            errors.append(
                f"{cid}: mutation_score={entry['mutation_score']} ≠ 1.0 (выжившие/неубитые: {fmt})"
            )
    return errors


def write_facts(root: Path, report: dict, out_dir: Optional[Path] = None) -> None:
    from tools.sensors.fact import write_fact
    from tools.sensors.subject import build_subject

    subject = build_subject(repo_root=root, config_path=root / "net" / "config.json", device="cpu")
    for sensor, fact, value in (
        ("S-039", "mutation_score", {"by_claim": report["by_claim"], "by_rule": report["by_rule"]}),
        ("S-040", "mutation_summary", {"by_template": report["by_template"],
                                       "survivors": report["survivors"],
                                       "equivalent": report["equivalent"],
                                       "wrapped": report.get("wrapped", {})}),
    ):
        write_fact(sensor, fact, value, unit="", quality="derived",
                   method="tools.properties.mutate (ADR-039, дельта M)",
                   subject=subject, out_dir=out_dir)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="M: мутационное тестирование стражей")
    parser.add_argument("--root", default=".")
    parser.add_argument("--no-fact", action="store_true")
    parser.add_argument("--json", dest="json_path", default=None)
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    report = mutation_report(root)
    if not args.no_fact:
        write_facts(root, report)
    if args.json_path:
        Path(args.json_path).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for cid, entry in report["by_claim"].items():
        print(f"  {cid} [{entry['property']}]: score={entry['mutation_score']} "
              f"killed={entry['killed']} survived={entry['survived']} unverified={entry['unverified']}")
    violations = gate_violations(root)
    for v in violations:
        print(f"[M4] FAIL: {v}")
    print(f"мутационный счёт: утверждений {len(report['by_claim'])}, выживших {len(report['survivors'])}, "
          f"нарушений M4 {len(violations)}")
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
