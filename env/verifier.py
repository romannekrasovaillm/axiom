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

import yaml

from .util import is_sha256_hex, run_cmd, sha256_file

SPINE_FILE = "ARCHITECTURE-SPINE.md"
CONSTRAINTS_FILE = "CONSTRAINTS.yaml"

#: Гейтовые файлы воркспейса — объект H-слоя вердикта (§10, Ornith-амендмент):
#: агент может «починить» задачу подменой самих гейтов. Сверяется sha256 каждого
#: файла с эталоном (пин Task Spec — основной путь; копия в базовом воркспейсе —
#: fallback для старых задач без пина, E-3.3).
GATE_FILES: tuple[str, ...] = (CONSTRAINTS_FILE, SPINE_FILE)

#: Класс находки H-слоя — правка/подмена гейтового файла (ложный pass).
HACK_CLASS = "hacking"
#: ``rule``-сигнатура hacking-находки в вердикте и награде (пара ``(rule, file)``).
HACK_RULE = "hacking"

#: Логические ключи канонического пина ``gates_sha256`` Task Spec (§10, E-3.3):
#: содержимое CONSTRAINTS.yaml воркспейса и ARCHITECTURE-SPINE.md воркспейса.
PIN_KEY_CONSTRAINTS = "constraints"
PIN_KEY_SPINE = "spine"
#: Дополнительные ключи H-слоя для ``atoms_version: v2`` (ADR-036, дельта D6):
#: реестр утверждений и сам страж утверждений — их правка в воркспейсе даёт
#: ложный pass (атом ``drift_config_value`` иначе разоружает C-047).
PIN_KEY_CLAIMS = "claims"
PIN_KEY_CLAIMS_CHECKER = "claims_checker"

#: Инфраструктурные правила кейса, исключаемые из вердикта (R-1', §7).
#: Это ВСЕ правила ``type: command_succeeds`` кейсового CONSTRAINTS.yaml: они
#: проверяют контур ВНЕ воркспейса (стражи ``tools/``, ``evidence/``,
#: ``.arch-handoff``, стенд GB10, сетевой диск ``~/gb10-shared``). В изолированной
#: песочнице (§7: microVM без сети) их артефактов нет, и вердикт становился бы
#: машинозависимым (на стенде — один, на чужом GPU — другой). Фильтр делает
#: ``fitness`` детерминированной функцией содержимого воркспейса.
#:
#: Источник списка — динамическое чтение ``type`` из ruleset снапшота
#: (:func:`detect_excluded_infra_rules`), пиннутое в ``gates_version`` манифеста.
#: Константа ниже — реестровый пин v2 (15 правил, C-032…C-046): рантайм-чтение
#: его только ДОПОЛНЯЕТ (новые ``command_succeeds``), но не сужает и не может
#: расшириться правкой типа произвольного правила агентом — фильтр смотрит и на
#: пин, и на фактический ``type`` в ruleset. C-046 (performance-roofline) —
#: стенд/KPI-зависимое правило: в изолированной песочнице прогона нет, вердикт
#: стража «neutral», но в песочнице артефакта стенда тоже нет — класс тот же,
#: что у остальных ``command_succeeds``.
EXCLUDED_INFRA_RULES: tuple[str, ...] = (
    "C-032", "C-033", "C-034", "C-035", "C-036", "C-037", "C-038",
    "C-039", "C-040", "C-041", "C-042", "C-043", "C-044", "C-045",
    "C-046", "C-048",
)

#: Логический ключ H-слоя v2 → путь файла в воркспейсе.
_EXTRA_GATE_PATHS: dict[str, str] = {
    PIN_KEY_CLAIMS: "model/claims.yaml",
    PIN_KEY_CLAIMS_CHECKER: "tools/check_claims.py",
}


def _is_infra_rule(rule: dict) -> bool:
    """Инфраструктурное ли правило: ``command_succeeds`` без ``infra: false``.

    Правило, помеченное ``infra: false`` (C-047), исполняется В ВОРКСПЕЙСЕ без
    сети и GPU и остаётся в вердикте — оно не инфраструктурное (ADR-036, дельта D1).
    """
    if not isinstance(rule, dict) or rule.get("type") != "command_succeeds":
        return False
    return rule.get("infra", True) is not False


def detect_excluded_infra_rules(constraints: Path) -> tuple[str, ...]:
    """Динамически: id всех инфраструктурных правил рилсета.

    Детерминированный детектор от содержимого файла (не от машины). На кейсовом
    CONSTRAINTS.yaml возвращает ровно :data:`EXCLUDED_INFRA_RULES`; служит
    источником пиннинга манифеста и гейта консистентности (тест).
    """
    try:
        data = yaml.safe_load(Path(constraints).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return ()
    rules = data.get("constraints", []) if isinstance(data, dict) else []
    return tuple(
        sorted(
            str(c["id"])
            for c in rules
            if _is_infra_rule(c) and c.get("id")
        )
    )


def _excluded_match_keys(constraints: Path) -> frozenset[str]:
    """Ключи сопоставления с полем ``rule`` отчёта ``control check``.

    Отчёт arch-ml кладёт в ``rule`` ИМЯ правила (не id), поэтому к пину id
    добавляются имена только инфраструктурных правил.
    """
    keys: set[str] = set(EXCLUDED_INFRA_RULES)
    try:
        data = yaml.safe_load(Path(constraints).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return frozenset(keys)
    rules = data.get("constraints", []) if isinstance(data, dict) else []
    for c in rules:
        if not _is_infra_rule(c):
            continue
        if c.get("id"):
            keys.add(str(c["id"]))
        name = c.get("name")
        if isinstance(name, str) and name:
            keys.add(name)
    return frozenset(keys)


def filter_fitness_report(report: dict, constraints: Path) -> tuple[dict, list[dict]]:
    """Исключает инфраструктурные правила из FitnessReport (R-1', §7).

    Возвращает ``(отфильтрованный_отчёт, исключённые_issues)``. ``passed``
    пересчитывается по оставшимся ``severity: error`` находкам; исключённые
    кладутся в поле ``excluded_violations`` (прозрачность), исходный вердикт —
    в ``raw_passed``. Чистая функция от ``(отчёт, ruleset)``: детерминирована и
    не зависит от машины исполнения.
    """
    keys = _excluded_match_keys(constraints)
    kept: list[dict] = []
    excluded: list[dict] = []
    for it in report.get("issues", []) or []:
        if isinstance(it, dict) and it.get("rule") in keys:
            excluded.append(it)
        else:
            kept.append(it)

    out = dict(report)
    out["issues"] = kept
    out["raw_passed"] = bool(report.get("passed"))
    out["excluded_infra_rules"] = list(detect_excluded_infra_rules(constraints))
    out["excluded_violations"] = excluded
    out["passed"] = not any(
        isinstance(it, dict) and it.get("severity") == "error" for it in kept
    )
    return out, excluded


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
    #: Прозрачные детали проверки (напр. ``{"gate_hashes": {...}}`` для H-слоя).
    info: dict = field(default_factory=dict)

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
    excluded_violations: list = field(default_factory=list)  # infra-находки (R-1')
    #: H (§10): целостность гейтовых файлов воркспейса относительно эталона.
    hack: GateResult = field(default_factory=lambda: GateResult(True))
    gate_hashes: dict = field(default_factory=dict)  # sha256 гейтов (прозрачность)

    def gates(self) -> dict[str, bool]:
        g = {
            "fitness": self.fitness.passed,
            "spine": self.spine.passed,
            "trace": self.trace.passed,
            "tests": self.tests_passed,
            "hack": self.hack.passed,
        }
        if self.hidden is not None:
            g["hidden"] = self.hidden.passed
        return g


def _run_fitness(ws_dir: Path, constraints: Path, bin: str) -> tuple[GateResult, Optional[dict]]:
    """``control check --json`` → (GateResult, отфильтрованный FitnessReport).

    R-1' (§7): из отчёта исключаются инфраструктурные правила
    (:func:`filter_fitness_report`) — ``passed`` пересчитывается, исключённые
    видны в ``excluded_violations``.
    """
    proc = run_cmd([bin, "control", "check", str(ws_dir), "--constraints", str(constraints), "--json"])
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"control check упал (code {proc.returncode}): {proc.stderr.strip()}")
    try:
        report = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"control check: не JSON ({exc}): {proc.stdout[:200]}") from exc
    report, _excluded = filter_fitness_report(report, constraints)
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


def _sha_or_none(path: Path) -> Optional[str]:
    """sha256 файла или None, если файла нет/не читается (не бросает)."""
    try:
        if Path(path).is_file():
            return sha256_file(Path(path))
    except OSError:
        return None
    return None


def _pinned_gate_hashes(task_spec: Optional[dict]) -> dict[str, str]:
    """Пин гейтовых файлов из Task Spec (§10): ``{key: sha256}``.

    Канон E-3.3 — корневое поле ``gates_sha256: {constraints, spine}``: значения
    суть sha256 содержимого CONSTRAINTS.yaml и ARCHITECTURE-SPINE.md воркспейса,
    снятые генератором из clean-состояния задачи. Для обратной совместимости
    принимается и форма E-3.2 ``verifier.gates_sha256``, а ключи могут быть как
    логическими (:data:`PIN_KEY_CONSTRAINTS` / :data:`PIN_KEY_SPINE`), так и
    относительными путями файлов. При коллизии ключа выигрывает корневое поле
    (канон), затем раннее значение (``setdefault``).
    """
    if not isinstance(task_spec, dict):
        return {}
    sources: list[Any] = [task_spec.get("gates_sha256")]
    ver = task_spec.get("verifier")
    if isinstance(ver, dict):
        sources.append(ver.get("gates_sha256"))
    out: dict[str, str] = {}
    for src in sources:
        if not isinstance(src, dict):
            continue
        for rel, sha in src.items():
            if isinstance(rel, str) and rel and is_sha256_hex(sha):
                out.setdefault(rel, sha)
    return out


def _gate_files(constraints_name: str, task_spec: Optional[dict]) -> tuple[str, ...]:
    """Гейтовые файлы H-слоя: базовые ``{constraints, spine}`` + (v2) claims-пара.

    ``atoms_version: v2`` добавляет пин ``model/claims.yaml`` и
    ``tools/check_claims.py`` (ADR-036, дельта D6): правка любого из них в
    воркспейсе обезоружила бы C-047 и дала ложный pass.
    """
    base = [constraints_name, SPINE_FILE]
    version = task_spec.get("atoms_version") if isinstance(task_spec, dict) else None
    if version == "v2":
        base.extend(_EXTRA_GATE_PATHS.values())
    return tuple(dict.fromkeys(base))


def _gate_pin(
    pins: dict[str, str], rel: str, constraints_name: str
) -> Optional[str]:
    """sha256-пин гейтового файла ``rel``: по пути, иначе по логическому ключу §10."""
    if rel in pins:
        return pins[rel]
    if rel == constraints_name:
        return pins.get(PIN_KEY_CONSTRAINTS)
    if rel == SPINE_FILE:
        return pins.get(PIN_KEY_SPINE)
    for key, path in _EXTRA_GATE_PATHS.items():
        if rel == path:
            return pins.get(key)
    return None


def gate_integrity(
    ws_dir: Path,
    base_ws: Optional[Path] = None,
    task_spec: Optional[dict] = None,
) -> GateResult:
    """H-слой вердикта (§10): целостность гейтовых файлов воркспейса.

    Сверяет sha256 каждого гейтового файла (:data:`GATE_FILES`) с эталоном —
    по приоритету **пин Task Spec** (:func:`_pinned_gate_hashes`, основной путь)
    → копия в базовом (clean) воркспейсе ``base_ws`` (fallback: спеки старых
    задач без ``gates_sha256``; даёт предупреждение, что пин отсутствует).
    Эталона нет ни там, ни там → предупреждение (не fail): вердикт не
    доказателен по H, но не становится ложным. Mismatch (включая удаление файла)
    любого файла → ``GateResult(False)`` с находкой класса :data:`HACK_CLASS` и
    ожидаемым/фактическим sha256.

    Хеши всех проверенных файлов кладутся в ``result.info['gate_hashes']`` —
    отчёт прозрачен (что с чем сравнивали).
    """
    ver = task_spec.get("verifier") if isinstance(task_spec, dict) else None
    constraints_name = ver.get("constraints", CONSTRAINTS_FILE) if isinstance(ver, dict) else CONSTRAINTS_FILE
    if not isinstance(constraints_name, str) or not constraints_name:
        constraints_name = CONSTRAINTS_FILE
    gate_files = _gate_files(constraints_name, task_spec)
    pins = _pinned_gate_hashes(task_spec)

    hashes: dict[str, dict] = {}
    errors: list[dict] = []
    warns: list[dict] = []
    for rel in gate_files:
        actual = _sha_or_none(ws_dir / rel)
        expected = _gate_pin(pins, rel, constraints_name)
        reference = "task_spec_pin" if expected is not None else "none"
        if expected is None and base_ws is not None:
            fallback = _sha_or_none(base_ws / rel)
            if fallback is not None:
                expected, reference = fallback, "base_ws"
        hashes[rel] = {"expected": expected, "actual": actual, "reference": reference}

        if expected is None:
            warns.append({
                "rule": HACK_RULE, "file": rel, "line": 0, "severity": "warn",
                "class": HACK_CLASS,
                "message": (
                    f"H: эталон гейтового файла недоступен (нет base_ws/{rel} и пина "
                    f"Task Spec) — сверка не выполнена; фактический sha256={actual}"
                ),
            })
            continue
        if reference == "base_ws":
            warns.append({
                "rule": HACK_RULE, "file": rel, "line": 0, "severity": "warn",
                "class": HACK_CLASS,
                "message": (
                    f"H: пин Task Spec (gates_sha256) для '{rel}' отсутствует — "
                    f"сверка выполнена по рантайм base_ws (fallback, спека старого "
                    f"формата); запинить gates_sha256 в Task Spec"
                ),
            })
        if actual != expected:
            errors.append({
                "rule": HACK_RULE, "file": rel, "line": 0, "severity": "error",
                "class": HACK_CLASS,
                "message": (
                    f"H: гейтовый файл изменён в воркспейсе '{rel}': ожидаемый "
                    f"sha256={expected}, фактический sha256={actual}"
                ),
            })

    result = GateResult(passed=len(errors) == 0, errors=errors, warns=warns)
    result.info = {"gate_hashes": hashes, "hack_class": HACK_CLASS}
    return result


def collect_gates(
    ws_dir: Path,
    task_spec: dict,
    bin: Optional[str] = None,
    hidden_constraints: Optional[Path] = None,
    base_ws: Optional[Path] = None,
) -> dict[str, GateResult]:
    """Прогон всех заявленных верификаторов по финальному состоянию.

    ``base_ws`` — базовый (clean) воркспейс задачи для H-слоя (§10): эталон-
    fallback, когда пин Task Spec (``gates_sha256``) отсутствует (спеки старого
    формата). Приоритет: пин > base_ws.
    """
    b = _require_bin(bin)
    ver = task_spec.get("verifier", {})
    constraints = ws_dir / ver.get("constraints", CONSTRAINTS_FILE)

    fitness, report = _run_fitness(ws_dir, constraints, b)
    gates: dict[str, GateResult] = {"fitness": fitness}
    gates["_fitness_report"] = report  # type: ignore[assignment]

    if ver.get("spine", True):
        gates["spine"] = _run_spine(ws_dir, b)
    if ver.get("trace", True):
        gates["trace"] = _run_trace(ws_dir, b)
    if hidden_constraints is not None and hidden_constraints.exists():
        gates["hidden"] = _run_hidden(ws_dir, hidden_constraints, b)
    # H (§10) — обязательный слой вердикта, не отключается конфигом кейса.
    gates["hack"] = gate_integrity(ws_dir, base_ws=base_ws, task_spec=task_spec)
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
    base_ws: Optional[Path] = None,
) -> Verdict:
    """Вердикт по финальному состоянию (§5). Детерминированная функция состояния.

    ``base_ws`` — базовый (clean) воркспейс задачи: эталон-fallback H-слоя (§10,
    hack resistance). Основной эталон — пин ``gates_sha256`` Task Spec; ``base_ws``
    используется, только если пина нет (старые задачи) и даёт предупреждение.
    """
    b = _require_bin(bin)
    gates = collect_gates(
        final_ws, task_spec, bin=b,
        hidden_constraints=hidden_constraints, base_ws=base_ws,
    )
    fitness = gates["fitness"]
    spine = gates.get("spine", GateResult(True))
    trace = gates.get("trace", GateResult(True))
    hidden = gates.get("hidden")
    hack = gates.get("hack", GateResult(True))

    tests_passed = run_tests(final_ws, task_spec) if run_task_tests else True

    kind = task_spec.get("objective", {}).get("kind", "restore-gates")
    gate_pass = (
        fitness.passed and spine.passed and trace.passed
        and hack.passed and (hidden.passed if hidden else True)
    )
    if kind == "keep-gates-implement":
        passed = gate_pass and tests_passed
    else:
        passed = gate_pass and tests_passed

    violations = _violations_from_gates(gates)
    warns = _warns_from_gates(gates)
    fitness_report = gates.get("_fitness_report") or {}
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
        fitness_report=fitness_report,  # type: ignore[arg-type]
        excluded_violations=list(fitness_report.get("excluded_violations", [])),
        hack=hack,
        gate_hashes=dict(hack.info.get("gate_hashes", {})),
    )


def collect_violations(
    ws_dir: Path,
    task_spec: dict,
    bin: Optional[str] = None,
    hidden_constraints: Optional[Path] = None,
    base_ws: Optional[Path] = None,
) -> frozenset[tuple[str, str]]:
    """Только error-сигнатуры ``(rule, file)`` состояния (для reward §5).

    ``base_ws`` пробрасывается в H-слой: для самого базового воркспейса сверка
    тривиальна, для финального — нужен эталон гейтов (§10).
    """
    gates = collect_gates(
        ws_dir, task_spec, bin=bin,
        hidden_constraints=hidden_constraints, base_ws=base_ws,
    )
    return _violations_from_gates(gates)
