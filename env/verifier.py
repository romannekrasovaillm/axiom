"""Обвязка CLI ``arch-ml``: механические верификаторы (§4).

- ``control check <ws> --constraints <c> --json`` → FitnessReport (fitness);
- ``control spine <ws>/ARCHITECTURE-SPINE.md`` → текст, exit code (spine);
- ``trace check <ws>`` → markdown, exit code (trace);
- ``objective.tests_cmd`` → exit code (задачные тесты).

Вердикт учитывает только ``severity: error``; ``warn`` пишется в манифест без
влияния на награду (§4). Сигнатура нарушения — пара ``(rule, file)`` (§3);
строка ``line`` в сигнатуру не входит.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .util import run_cmd, sha256_file

SPINE_FILE = "ARCHITECTURE-SPINE.md"


class ArchMlUnavailable(RuntimeError):
    """Бинарь arch-ml недоступен (нет ENV_ARCH_ML_BIN / пути в PATH)."""


def arch_ml_bin() -> str:
    return os.environ.get("ENV_ARCH_ML_BIN", "arch-ml")


def _which(bin: str) -> bool:
    if os.path.sep in bin or bin.startswith("."):
        return Path(bin).is_file()
    for d in os.environ.get("PATH", "").split(os.pathsep):
        if d and (Path(d) / bin).is_file():
            return True
    return False


def arch_ml_available(bin: Optional[str] = None) -> bool:
    return _which(bin or arch_ml_bin())


def arch_ml_build_hash(bin: Optional[str] = None) -> str:
    """Хеш сборки CLI (пиннинг ``gates_version.arch_ml_build``, AD-4)."""
    b = bin or arch_ml_bin()
    path = Path(b) if (os.path.sep in b or b.startswith(".")) else None
    if path is None:
        for d in os.environ.get("PATH", "").split(os.pathsep):
            cand = Path(d) / b if d else None
            if cand and cand.is_file():
                path = cand
                break
    if path is None or not path.is_file():
        raise ArchMlUnavailable(f"arch-ml бинарь не найден: {b}")
    return sha256_file(path)


@dataclass
class GateResult:
    passed: bool
    errors: list[dict] = field(default_factory=list)
    warns: list[dict] = field(default_factory=list)

    def signature_set(self, file_override: Optional[str] = None) -> frozenset[tuple[str, str]]:
        out = set()
        for it in self.errors:
            f = file_override if file_override is not None else it.get("file", "")
            out.add((it["rule"], f))
        return frozenset(out)


@dataclass
class Verdict:
    passed: bool
    objective_kind: str
    fitness: GateResult
    spine: GateResult
    trace: GateResult
    hidden: Optional[GateResult]
    tests_passed: bool
    violations: frozenset = frozenset()  # (rule, file) error-сигнатуры
    warn_issues: list = field(default_factory=list)
    fitness_report: Optional[dict] = None

    def gates(self) -> dict[str, bool]:
        g = {
            "fitness": self.fitness.passed,
            "spine": self.spine.passed,
            "trace": self.trace.passed,
            "tests": self.tests_passed,
        }
        if self.hidden is not None:
            g["hidden"] = self.hidden.passed
        return g


def _run_fitness(ws_dir: Path, constraints: Path, bin: str) -> tuple[GateResult, Optional[dict]]:
    """``control check --json`` → (GateResult, сырой FitnessReport)."""
    proc = run_cmd([bin, "control", "check", str(ws_dir), "--constraints", str(constraints), "--json"])
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"control check упал (code {proc.returncode}): {proc.stderr.strip()}")
    try:
        report = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"control check: не JSON ({exc}): {proc.stdout[:200]}") from exc
    errors, warns = _split_issues(report.get("issues", []))
    return GateResult(passed=bool(report.get("passed")), errors=errors, warns=warns), report


def _split_issues(issues: list[dict]) -> tuple[list[dict], list[dict]]:
    errors, warns = [], []
    for it in issues:
        if it.get("severity") == "error":
            errors.append(it)
        elif it.get("severity") == "warn":
            warns.append(it)
    return errors, warns


_SPINE_RE = re.compile(r"^(.+):(\d+)\s+(\S+)$")


def _run_spine(ws_dir: Path, bin: str) -> GateResult:
    spine = ws_dir / SPINE_FILE
    proc = run_cmd([bin, "control", "spine", str(spine)])
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"control spine упал (code {proc.returncode}): {proc.stderr.strip()}")
    errors, warns = [], []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("[") or "]" not in line:
            continue
        sev, rest = line[1:].split("]", 1)
        sev = sev.strip()
        if sev not in ("error", "warn"):
            continue
        rest = rest.strip()
        head, _, msg = rest.partition(" — ")
        m = _SPINE_RE.match(head)
        rule = m.group(3) if m else "spine"
        item = {"rule": rule, "file": SPINE_FILE, "line": int(m.group(2)) if m else 0,
                "message": msg, "severity": sev}
        (errors if sev == "error" else warns).append(item)
    return GateResult(passed=len(errors) == 0, errors=errors, warns=warns)


_TRACE_ISSUE_RE = re.compile(r"^- \[(error|warn)\] ([^:]+): (.*)$")


def _run_trace(ws_dir: Path, bin: str) -> GateResult:
    proc = run_cmd([bin, "trace", "check", str(ws_dir)])
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"trace check упал (code {proc.returncode}): {proc.stderr.strip()}")
    errors, warns = [], []
    for line in proc.stdout.splitlines():
        m = _TRACE_ISSUE_RE.match(line.strip())
        if not m:
            continue
        sev, rule, msg = m.group(1), m.group(2).strip(), m.group(3)
        item = {"rule": rule, "file": "", "line": 0, "message": msg, "severity": sev}
        (errors if sev == "error" else warns).append(item)
    return GateResult(passed=len(errors) == 0, errors=errors, warns=warns)


def _run_hidden(ws_dir: Path, hidden_constraints: Path, bin: str) -> GateResult:
    gate, _ = _run_fitness(ws_dir, hidden_constraints, bin)
    return gate


def _require_bin(bin: Optional[str]) -> str:
    b = bin or arch_ml_bin()
    if not arch_ml_available(b):
        raise ArchMlUnavailable(f"arch-ml бинарь недоступен: {b}")
    return b


def collect_gates(
    ws_dir: Path,
    task_spec: dict,
    bin: Optional[str] = None,
    hidden_constraints: Optional[Path] = None,
) -> dict[str, GateResult]:
    """Прогон всех заявленных верификаторов по финальному состоянию."""
    b = _require_bin(bin)
    ver = task_spec.get("verifier", {})
    constraints = ws_dir / ver.get("constraints", "CONSTRAINTS.yaml")

    fitness, report = _run_fitness(ws_dir, constraints, b)
    gates: dict[str, GateResult] = {"fitness": fitness}
    gates["_fitness_report"] = report  # type: ignore[assignment]

    if ver.get("spine", True):
        gates["spine"] = _run_spine(ws_dir, b)
    if ver.get("trace", True):
        gates["trace"] = _run_trace(ws_dir, b)
    if hidden_constraints is not None and hidden_constraints.exists():
        gates["hidden"] = _run_hidden(ws_dir, hidden_constraints, b)
    return gates


def _violations_from_gates(gates: dict[str, GateResult]) -> frozenset[tuple[str, str]]:
    sigs: set[tuple[str, str]] = set()
    for name, gate in gates.items():
        if name.startswith("_") or name == "fitness_report":
            continue
        sigs |= set(gate.signature_set())
    return frozenset(sigs)


def _warns_from_gates(gates: dict[str, GateResult]) -> list[dict]:
    out: list[dict] = []
    for name, gate in gates.items():
        if name.startswith("_"):
            continue
        out.extend(gate.warns)
    return out


def run_tests(ws_dir: Path, task_spec: dict) -> bool:
    """Исполняет ``objective.tests_cmd`` в workspace (exit 0 = успех)."""
    cmd = task_spec.get("objective", {}).get("tests_cmd", "true")
    proc = run_cmd(["bash", "-c", cmd], cwd=ws_dir, timeout=300)
    return proc.returncode == 0


def verify(
    task_spec: dict,
    final_ws: Path,
    bin: Optional[str] = None,
    hidden_constraints: Optional[Path] = None,
    run_task_tests: bool = True,
) -> Verdict:
    """Вердикт по финальному состоянию (§5). Детерминированная функция состояния."""
    b = _require_bin(bin)
    gates = collect_gates(final_ws, task_spec, bin=b, hidden_constraints=hidden_constraints)
    fitness = gates["fitness"]
    spine = gates.get("spine", GateResult(True))
    trace = gates.get("trace", GateResult(True))
    hidden = gates.get("hidden")

    tests_passed = run_tests(final_ws, task_spec) if run_task_tests else True

    kind = task_spec.get("objective", {}).get("kind", "restore-gates")
    gate_pass = fitness.passed and spine.passed and trace.passed and (hidden.passed if hidden else True)
    if kind == "keep-gates-implement":
        passed = gate_pass and tests_passed
    else:
        passed = gate_pass and tests_passed

    violations = _violations_from_gates(gates)
    warns = _warns_from_gates(gates)
    return Verdict(
        passed=passed,
        objective_kind=kind,
        fitness=fitness,
        spine=spine,
        trace=trace,
        hidden=hidden,
        tests_passed=tests_passed,
        violations=violations,
        warn_issues=warns,
        fitness_report=gates.get("_fitness_report"),  # type: ignore[arg-type]
    )


def collect_violations(
    ws_dir: Path,
    task_spec: dict,
    bin: Optional[str] = None,
    hidden_constraints: Optional[Path] = None,
) -> frozenset[tuple[str, str]]:
    """Только error-сигнатуры ``(rule, file)`` состояния (для reward §5)."""
    gates = collect_gates(ws_dir, task_spec, bin=bin, hidden_constraints=hidden_constraints)
    return _violations_from_gates(gates)
