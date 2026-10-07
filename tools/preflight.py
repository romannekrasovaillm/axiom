#!/usr/bin/env python3
"""Единый preflight открывающих гейтов (ADR-038, дельта K2).

    python3 tools/preflight.py <gate> [--override-preflight REASON]

Читает ``model/opening-gates.yaml``, проверяет каждое требование гейта:
``{claim: CL-NNN}`` — через ``check_claims --evaluate``; ``{guard: <имя>, run:
<run-ref>}`` — через указанный страж в режиме «недоказанное не открывает»
(``--require-verified``). ``pass`` — только если **все** требования ``pass``;
любое ``unverified`` или ``fail`` — отказ с перечнем причин. Отчёт пишется
фактом S-033 ``gate_preflight`` (виден в отчёте, дельта K3/K4).

``check_budget_gate.py --preflight`` остаётся и вызывается как частный случай
(гейт ``rental-pretrain-l3``). Аварийный обход ``--override-preflight <reason>``
разрешает продолжить, но обход фиксируется фактом (не молчит).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools import check_claims as cc  # noqa: E402
from tools import check_performance_roofline as roofline  # noqa: E402
from tools.miniyaml import MiniYamlError, load_file  # noqa: E402
from tools.sensors.fact import write_fact  # noqa: E402
from tools.sensors.subject import build_subject  # noqa: E402

GATES_FILE = "model/opening-gates.yaml"
FACT_SENSOR = "S-033"
FACT_NAME = "gate_preflight"


class GateError(ValueError):
    """Реестр гейтов не читается или гейт не найден."""


def load_gates(root: str | Path) -> list[dict[str, Any]]:
    path = Path(root) / GATES_FILE
    if not path.is_file():
        raise GateError(f"нет реестра гейтов: {path}")
    try:
        data = load_file(path)
    except MiniYamlError as exc:
        raise GateError(f"{path}: невалидный YAML — {exc}") from exc
    if isinstance(data, dict):
        data = data.get("gates")
    if not isinstance(data, list):
        raise GateError(f"{path}: ожидался список гейтов")
    return [g for g in data if isinstance(g, dict)]


def find_gate(root: str | Path, name: str) -> dict[str, Any]:
    for gate in load_gates(root):
        if gate.get("gate") == name:
            return gate
    raise GateError(f"гейт {name!r} не объявлен в {GATES_FILE}")


def _claim_cache(root: Path) -> dict[str, dict[str, Any]]:
    return {r["id"]: r for r in cc.evaluate(root)}


def _flapping_map(root: Path) -> dict[str, Any]:
    """Карта флапания утверждений из факта S-037 (пусто, если истории нет)."""
    from tools.sensors.fact import read_latest

    try:
        record = read_latest("S-037", "flapping")
        if record and record.get("status") == "ok" and isinstance(record.get("value"), dict):
            return record["value"]
    except Exception:  # noqa: BLE001 — нет факта = нечего сверять
        return {}
    return {}


def evaluate_requirement(
    requirement: dict[str, Any],
    root: Path,
    claims: dict[str, dict[str, Any]],
    *,
    facts_dir: Optional[str] = None,
    flapping: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Вердикт одного требования: ``pass | fail | unverified`` + причина."""
    entry: dict[str, Any] = {"requirement": requirement}
    if isinstance(requirement.get("claim"), str):
        cid = requirement["claim"]
        result = claims.get(cid)
        if result is None:
            return {**entry, "verdict": "fail", "reason": f"утверждение {cid} не найдено"}
        if flapping and flapping.get(cid) is True:
            return {
                **entry, "verdict": "unverified", "claim": cid,
                "reason": f"{cid}: вердикт флапает — допуск требует перепиннинга, гейт не открыт (M4)",
            }
        return {
            **entry, "verdict": result["verdict"], "claim": cid,
            "reason": result.get("reason") or f"{cid}: {result['verdict']}",
        }
    if isinstance(requirement.get("guard"), str):
        guard = requirement["guard"]
        if guard == "performance-roofline":
            run = requirement.get("run")
            if not isinstance(run, str):
                return {**entry, "verdict": "fail", "reason": "performance-roofline без run"}
            code, report = roofline.run_check(run, None, require_verified=True, facts_dir=facts_dir)
            verdict = "pass" if code == roofline.EXIT_OK else (
                "unverified" if code == roofline.EXIT_UNVERIFIED else "fail"
            )
            return {
                **entry, "verdict": verdict, "guard": guard, "run": run,
                "reason": f"{guard}/{run}: {report.get('verdict')} (exit {code})",
            }
        return {**entry, "verdict": "unverified", "guard": guard,
                "reason": f"страж {guard} не подключён к единому preflight"}
    return {**entry, "verdict": "fail", "reason": f"неизвестное требование: {requirement!r}"}


def preflight(
    root: str | Path,
    gate_name: str,
    *,
    facts_dir: Optional[str] = None,
    out_dir: Optional[str] = None,
    override: Optional[str] = None,
) -> tuple[int, dict[str, Any]]:
    root = Path(root).resolve()
    gate = find_gate(root, gate_name)
    claims = _claim_cache(root)
    flapping = _flapping_map(root)
    requirements = gate.get("requires") or []
    results = [
        evaluate_requirement(req, root, claims, facts_dir=facts_dir, flapping=flapping)
        for req in requirements
        if isinstance(req, dict)
    ]
    blockers = [r for r in results if r["verdict"] != "pass"]
    status = "pass" if not blockers else "refused"
    report = {
        "gate": gate_name,
        "opens": gate.get("opens"),
        "verdict": status,
        "requirements": results,
        "blockers": blockers,
        "override": override,
        "invocation": gate.get("invocation"),
    }
    _write_fact(root, report, out_dir=out_dir)
    if blockers and override:
        report["verdict"] = "overridden"
        return 0, report
    return (0 if not blockers else 1), report


def _write_fact(root: Path, report: dict[str, Any], *, out_dir: Optional[str] = None) -> None:
    subject = build_subject(
        repo_root=root, config_path=root / "net" / "config.json", device="cpu"
    )
    write_fact(
        FACT_SENSOR, FACT_NAME, report, unit="", quality="measured",
        method="tools/preflight.py (единый preflight, ADR-038 дельта K2)",
        subject=subject, out_dir=out_dir,
    )


def _print_report(report: dict[str, Any]) -> None:
    print(f"Preflight «{report['gate']}» (opens={report.get('opens')}): {report['verdict'].upper()}")
    for req in report["requirements"]:
        mark = {"pass": "PASS", "fail": "FAIL", "unverified": "UNVERIFIED"}[req["verdict"]]
        print(f"  [{mark}] {req['reason']}")
    unverified = [r for r in report["requirements"] if r["verdict"] == "unverified"]
    if unverified:
        print("\n  НЕ ПРОВЕРЕНО (unverified) — дверь не открыта:")
        for req in unverified:
            print(f"    - {req['reason']}")
    if report["blockers"]:
        print("\nОтказ: unverified/fail — гейт не открыт (ADR-038):")
        for req in report["blockers"]:
            print(f"    - {req['reason']}")
    if report.get("override"):
        print(f"\nВНИМАНИЕ: аварийный обход гейта: {report['override']}")


def _fact_override(report: dict[str, Any]) -> None:
    if report.get("override") and report["verdict"] == "overridden":
        print("[preflight] обход зафиксирован фактом S-033.gate_preflight (виден в отчёте)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Единый preflight открывающих гейтов (ADR-038)")
    parser.add_argument("gate", nargs="?", help="имя гейта из model/opening-gates.yaml")
    parser.add_argument("--root", default=".", help="корень кейса (по умолчанию '.')")
    parser.add_argument("--facts-dir", default=None, help="каталог фактов (evidence/facts)")
    parser.add_argument("--out-dir", default=None, help="куда писать факт S-033")
    parser.add_argument("--json", dest="json_path", default=None, help="файл отчёта JSON")
    parser.add_argument("--override-preflight", default=None,
                        help="аварийный обход гейта с причиной (пишется фактом)")
    parser.add_argument("--list", action="store_true", help="перечислить гейты")
    parser.add_argument("--selftest", action="store_true")
    return parser


def run_selftest() -> int:
    import tempfile

    checks: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory(prefix="preflight-selftest-") as tmp:
        root = Path(tmp)
        (root / "model").mkdir()
        (root / "model" / "opening-gates.yaml").write_text(
            "- gate: demo\n  opens: money\n  requires:\n"
            "    - {claim: CL-1}\n"
            "  invocation: \"x\"\n",
            encoding="utf-8",
        )
        (root / "model" / "sensors.yaml").write_text("- id: S-001\n  facts: [v]\n", encoding="utf-8")
        (root / "model" / "claims.yaml").write_text("[]\n", encoding="utf-8")
        (root / "CONSTRAINTS.yaml").write_text("- id: C-001\n", encoding="utf-8")
        gate = find_gate(root, "demo")
        checks.append(("гейт найден", gate["gate"] == "demo"))
        claims = {"CL-1": {"id": "CL-1", "verdict": "unverified", "reason": "нет факта"}}
        res = evaluate_requirement({"claim": "CL-1"}, root, claims)
        checks.append(("unverified-утверждение → unverified", res["verdict"] == "unverified"))
        missing = evaluate_requirement({"claim": "CL-нет"}, root, claims)
        checks.append(("неизвестное утверждение → fail", missing["verdict"] == "fail"))
        unknown_guard = evaluate_requirement({"guard": "нет-такого"}, root, claims)
        checks.append(("неизвестный страж → unverified", unknown_guard["verdict"] == "unverified"))
    ok = all(p for _, p in checks)
    for label, passed in checks:
        print(f"[selftest] {'PASS' if passed else 'FAIL'}: {label}")
    print(f"[selftest] {'PASS' if ok else 'FAIL'}: preflight")
    return 0 if ok else 1


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.selftest:
        return run_selftest()
    if args.list:
        for gate in load_gates(args.root):
            print(f"  {gate.get('gate')} (opens={gate.get('opens')}) → {gate.get('invocation')}")
        return 0
    if not args.gate:
        build_parser().print_usage(sys.stderr)
        print("укажите <gate> или --list/--selftest", file=sys.stderr)
        return 2
    try:
        code, report = preflight(
            args.root, args.gate, facts_dir=args.facts_dir, out_dir=args.out_dir,
            override=args.override_preflight,
        )
    except GateError as exc:
        print(f"preflight: {exc}", file=sys.stderr)
        return 2
    _print_report(report)
    _fact_override(report)
    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
