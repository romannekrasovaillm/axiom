#!/usr/bin/env python3
"""C-039 — поведенческий страж AD-2: LLM-судья вне контура награды RL.

Проверяет **по коду**, а не по прозе:

* (a) модули контура награды не импортируют и не вызывают судейские модули,
  клиентов LLM и внешние API (LLM-SDK, HTTP-клиенты, сетевые CLI в запуске
  процессов, динамические импорты, ключи провайдеров);
* (b) судейский модуль (если он есть в репозитории) не импортируется из пути
  награды — судья допустим только как отдельная калиброванная метрика вне
  контура (ADR-002, ADR-005);
* (c) печатает, какие файлы считаются контуром награды и на каком основании
  (маркеры сидов + импорт-замыкание).

Границы контура (объявлены, не подразумеваются)
-----------------------------------------------

AD-2 связывает «среда ↔ verifier'ы ↔ награда RL-контура»: награда — функция
вердикта, вердикт собирают механические гейты. Поэтому контуром принят
**пакет, в котором определена награда** (в кейсе — ``env/``: ``reward.py``,
``verifier.py``, ``run.py``), а не отдельный «файл с функцией reward»:
правило «судьи нет в пути награды» должно ловить и обвязку, где вердикт
превращается в награду. Пакет определяется маркерами (имя модуля ``reward``,
класс ``Reward``, функция с ``reward`` в имени, ``def compute(...)`` рядом с
упоминанием reward) — сработавшие маркеры печатаются.

Дополнительно берётся импорт-замыкание по локальным модулям (зависимости
контура): если награда ходит в помощника, судейский вызов не должен
прятаться за ним. Замыкание печатается отдельным списком — это **не** контур,
а то, что он вызывает.

Что заведомо вне проверки (граница, а не пробел): транзитивные зависимости
вне дерева репозитория (пакеты из site-packages); вызовы, собранные из
нестроковых выражений (например, URL из переменной окружения); намеренная
обфускация (``eval``/``exec`` над собранной строкой). Инструмент не сканирует
сам себя (``tools/check_reward_isolation.py``): он не контур награды, а его
текст содержит маркеры, которые он ищет.

Ложный PASS запрещён: если контур награды не найден, скрипт возвращает
``EXIT_NOT_VERIFIED`` и печатает «НЕ ПРОВЕРЕНО: контур награды не найден»
(это же сообщение уходит в гейт кейса как красный вердикт).

Запуск::

    python3 tools/check_reward_isolation.py [--root <каталог кейса>]

Коды возврата: ``0`` — контур найден и чист (PASS); ``1`` — в пути награды
найдены судья/LLM-клиент/внешний API (FAIL); ``2`` — контур награды не
найден (НЕ ПРОВЕРЕНО, ложный PASS запрещён).
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

#: Файл инструмента: не сканируется (см. границы в докстринге).
TOOL_PATH = Path(__file__).resolve()
#: Корень кейса по умолчанию — каталог, в котором лежит ``tools/``.
DEFAULT_ROOT = TOOL_PATH.parent.parent

EXIT_PASS = 0
EXIT_VIOLATION = 1
EXIT_NOT_VERIFIED = 2

#: Текст, обязанный попасть в вывод при ненайденном контуре (ложный PASS запрещён).
NOT_VERIFIED_MESSAGE = "НЕ ПРОВЕРЕНО: контур награды не найден"


# --- что не является контуром награды ---------------------------------------

_EXCLUDED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".venv",
        "venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".arch-handoff",
        "node_modules",
        "tests",
    }
)
_EXCLUDED_FILE_RE = re.compile(r"^(test_.*\.py|.*_test\.py|conftest\.py)$")


# --- маркеры сидов: где определена награда ----------------------------------

_NAME_SEED_RE = re.compile(r"reward", re.IGNORECASE)
_CLASS_REWARD_RE = re.compile(r"(?m)^\s*class\s+\w*Reward\w*\s*[\(:]")
_DEF_REWARD_RE = re.compile(r"(?m)^\s*def\s+\w*[Rr]eward\w*\s*\(")
_DEF_COMPUTE_RE = re.compile(r"(?m)^\s*def\s+compute\s*\(")
_WORD_REWARD_RE = re.compile(r"(?i)\breward\b")


# --- запрещённое в пути награды ---------------------------------------------

_LLM_SDK_PREFIXES = (
    "openai",
    "anthropic",
    "litellm",
    "cohere",
    "mistralai",
    "ollama",
    "google.generativeai",
    "google.genai",
    "vertexai",
    "replicate",
    "together",
    "groq",
    "deepseek",
    "dashscope",
    "zhipuai",
    "moonshot",
    "portkey_ai",
    "dspy",
)
_HTTP_CLIENT_PREFIXES = (
    "requests",
    "httpx",
    "aiohttp",
    "urllib.request",
    "urllib3",
    "http.client",
    "websockets",
    "websocket",
)

#: Имена функций запуска процессов: сетевой CLI в аргументах — внешний вызов.
_PROCESS_CALL_NAMES = frozenset(
    {
        "run",
        "call",
        "check_call",
        "check_output",
        "Popen",
        "system",
        "popen",
        "spawnv",
        "execv",
        "run_cmd",
    }
)
_NETWORK_CLI_RE = re.compile(
    r"(?i)(^|[\s\"'/\\])(curl|wget|ncat|nc|socat|telnet|ssh|scp)([\s\"']|$)"
)
_URL_RE = re.compile(r"https?://")

#: Провайдер LLM + ключ/эндпоинт в одной строке — обращение к внешнему API.
_PROVIDER_RE = re.compile(
    r"(?i)\b(openai|anthropic|azure[_-]?openai|gemini|cohere|mistral|litellm"
    r"|dashscope|zhipu|moonshot|deepseek)[a-z0-9_]*"
)
_CREDENTIAL_RE = re.compile(r"(?i)(api[_-]?key|apikey|access[_-]?token|secret|base[_-]?url)")

#: Судейский модуль: имя или содержимое объявляет LLM-судью.
_JUDGE_NAME_RE = re.compile(
    r"(?i)(^|[._\-/])(llm[_\-]?judge|judge[_\-]?llm|model[_\-]?judge"
    r"|ai[_\-]?judge|judge)([._\-/]|$)"
)
_JUDGE_CONTENT_RES = (
    (re.compile(r"(?i)\bllm[\s\-]*судь"), "объявлен LLM-судья (llm-судья)"),
    (re.compile(r"(?i)\bllm[\s\-_]*judge\b"), "объявлен LLM-судья (llm judge)"),
    (re.compile(r"(?i)\bмодель[\s\-]*судь"), "объявлена модель-судья"),
    (re.compile(r"(?i)\bjudge\s+model\b"), "объявлена judge model"),
    (
        re.compile(
            r"(?i)you\s+are\s+(an?\s+)?(impartial\s+|fair\s+|expert\s+)?"
            r"(judge|evaluator|grader)\b"
        ),
        "промпт-шаблон судьи",
    ),
)


class ScanError(Exception):
    """Вход не читается (не Python-файл / битый AST)."""


# --- структуры ---------------------------------------------------------------


@dataclass(frozen=True)
class ImportRef:
    """Один оператор импорта: модуль, имена, относительность, строка."""

    module: str | None
    names: tuple[str, ...]
    level: int
    line: int
    text: str

    def candidates(self, package: str) -> list[str]:
        """Абсолютные имена, под которыми импорт может быть локальным модулем."""
        if self.level:
            parts = package.split(".") if package else []
            up = self.level - 1
            if up:
                parts = parts[:-up] if up <= len(parts) else []
            base = ".".join(parts + ([self.module] if self.module else []))
        else:
            base = self.module or ""
        out = [base] if base else []
        out.extend(f"{base}.{name}" if base else name for name in self.names)
        return [name for name in out if name]


@dataclass(frozen=True)
class SourceFile:
    """Разобранный модуль: путь, модульное имя, AST и импорты."""

    path: Path
    rel: str
    module: str
    text: str
    tree: ast.Module
    imports: tuple[ImportRef, ...]

    @property
    def package(self) -> str:
        return self.module.rpartition(".")[0]


@dataclass(frozen=True)
class CircuitFile:
    """Файл контура награды с основанием включения."""

    rel: str
    basis: str


@dataclass(frozen=True)
class Finding:
    """Нарушение AD-2 в контуре награды — красный гейт."""

    code: str
    rel: str
    line: int
    message: str


@dataclass(frozen=True)
class Warning:
    """Замечание, гейт не красит: судья вне пути награды допустим."""

    code: str
    rel: str
    message: str


@dataclass
class Report:
    """Результат прогона стража."""

    root: Path
    seeds: list[tuple[str, str]] = field(default_factory=list)
    core: list[CircuitFile] = field(default_factory=list)
    dependencies: list[CircuitFile] = field(default_factory=list)
    judge_modules: list[tuple[str, str]] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    warnings: list[Warning] = field(default_factory=list)

    @property
    def verified(self) -> bool:
        """Контур награды найден — иначе вердикт «НЕ ПРОВЕРЕНО»."""
        return bool(self.seeds)

    @property
    def ok(self) -> bool:
        return self.verified and not self.findings

    @property
    def exit_code(self) -> int:
        if not self.verified:
            return EXIT_NOT_VERIFIED
        return EXIT_PASS if not self.findings else EXIT_VIOLATION


# --- разбор кода -------------------------------------------------------------


def _module_name(rel: Path) -> str:
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _render_import(node: ast.stmt) -> str:
    return re.sub(r"\s+", " ", ast.unparse(node)).strip()


def _iter_imports(tree: ast.Module) -> tuple[ImportRef, ...]:
    refs: list[ImportRef] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                refs.append(
                    ImportRef(alias.name, (), 0, node.lineno, _render_import(node))
                )
        elif isinstance(node, ast.ImportFrom):
            refs.append(
                ImportRef(
                    node.module,
                    tuple(alias.name for alias in node.names),
                    node.level,
                    node.lineno,
                    _render_import(node),
                )
            )
    return tuple(refs)


def scan_candidates(root: Path) -> list[Path]:
    """Все ``*.py`` кейса, кроме служебных каталогов, тестов и самого инструмента."""
    out: list[Path] = []
    for path in sorted(root.rglob("*.py")):
        try:
            resolved = path.resolve()
        except OSError:  # pragma: no cover — битый симлинк
            continue
        if resolved == TOOL_PATH:
            continue
        rel = path.relative_to(root)
        if any(part in _EXCLUDED_DIRS for part in rel.parts[:-1]):
            continue
        if _EXCLUDED_FILE_RE.match(path.name):
            continue
        out.append(path)
    return out


def load_source(path: Path, root: Path) -> SourceFile:
    rel = path.relative_to(root)
    text = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text, filename=str(rel))
    except SyntaxError as exc:
        raise ScanError(f"{rel}: не разбирается как Python — {exc}") from exc
    return SourceFile(
        path=path,
        rel=rel.as_posix(),
        module=_module_name(rel),
        text=text,
        tree=tree,
        imports=_iter_imports(tree),
    )


def seed_reasons(source: SourceFile) -> list[str]:
    """Маркеры, по которым модуль признаётся определением награды."""
    reasons: list[str] = []
    if _NAME_SEED_RE.search(source.path.stem):
        reasons.append(f"маркер имени: '{source.path.name}' содержит 'reward'")
    if _CLASS_REWARD_RE.search(source.text):
        reasons.append("маркер контента: объявлен класс Reward")
    if _DEF_REWARD_RE.search(source.text):
        reasons.append("маркер контента: объявлена функция с 'reward' в имени")
    if _DEF_COMPUTE_RE.search(source.text) and _WORD_REWARD_RE.search(source.text):
        reasons.append("маркер контента: def compute(...) рядом с упоминанием reward")
    return reasons


def package_dir(path: Path, root: Path) -> Path:
    """Верхний каталог-пакет, содержащий файл (для ``env/sub/x.py`` — ``env/``)."""
    directory = path.parent
    while directory.parent != root and (directory.parent / "__init__.py").exists():
        directory = directory.parent
    return directory


def judge_reasons(source: SourceFile) -> list[str]:
    """Маркеры судейского модуля: имя или объявление в содержимом."""
    reasons: list[str] = []
    if _JUDGE_NAME_RE.search(source.path.stem):
        reasons.append(f"маркер имени: '{source.path.name}' называет судью")
    for pattern, reason in _JUDGE_CONTENT_RES:
        if pattern.search(source.text):
            reasons.append(reason)
            break
    return reasons


def _dotted(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None


def _literal_strings(node: ast.expr) -> list[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        out: list[str] = []
        for element in node.elts:
            out.extend(_literal_strings(element))
        return out
    return []


def _is_prohibited(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == p or name.startswith(f"{p}.") for p in prefixes)


# --- проверки ----------------------------------------------------------------


def _check_imports(
    source: SourceFile,
    index: dict[str, SourceFile],
    judges: dict[str, str],
    imported_judges: set[str],
) -> list[Finding]:
    """(a)+(b) Импорты контура: LLM-клиенты, внешние API, судейские модули."""
    findings: list[Finding] = []
    for ref in source.imports:
        names = ref.candidates(source.package)
        for name in names:
            if _is_prohibited(name, _LLM_SDK_PREFIXES):
                findings.append(
                    Finding(
                        "REWARD-PATH-LLM-IMPORT",
                        source.rel,
                        ref.line,
                        f"импорт LLM-клиента '{name}' в пути награды — вердикт "
                        "перестаёт быть детерминированной функцией артефактов "
                        f"(«{ref.text}»)",
                    )
                )
            if _is_prohibited(name, _HTTP_CLIENT_PREFIXES):
                findings.append(
                    Finding(
                        "REWARD-PATH-EXTERNAL-API",
                        source.rel,
                        ref.line,
                        f"импорт внешнего API-клиента '{name}' в пути награды "
                        f"(«{ref.text}»)",
                    )
                )
        targets = _resolve(ref, source, index)
        judge_targets = [target for target in targets if target.rel in judges]
        for target in judge_targets:
            imported_judges.add(target.rel)
            findings.append(
                Finding(
                    "REWARD-PATH-JUDGE-IMPORT",
                    source.rel,
                    ref.line,
                    f"импорт судейского модуля '{target.rel}' "
                    f"({judges[target.rel]}) из пути награды",
                )
            )
        if not judge_targets:
            for name in names:
                if _JUDGE_NAME_RE.search(name):
                    findings.append(
                        Finding(
                            "REWARD-PATH-JUDGE-IMPORT",
                            source.rel,
                            ref.line,
                            f"импорт судейского модуля '{name}' из пути награды — "
                            "LLM-судья допустим только вне контура (ADR-002)",
                        )
                    )
                    break
    return findings


def _check_calls(source: SourceFile) -> list[Finding]:
    """(a) Вызовы: динамический импорт запрещённого модуля, сетевой CLI, ключи."""
    findings: list[Finding] = []
    for node in ast.walk(source.tree):
        if isinstance(node, ast.Call):
            name = _dotted(node.func) or ""
            last = name.rpartition(".")[2]
            if last == "import_module" or name == "__import__":
                for arg in node.args:
                    for literal in _literal_strings(arg):
                        if _is_prohibited(literal, _LLM_SDK_PREFIXES) or _is_prohibited(
                            literal, _HTTP_CLIENT_PREFIXES
                        ):
                            findings.append(
                                Finding(
                                    "REWARD-PATH-DYNAMIC-IMPORT",
                                    source.rel,
                                    node.lineno,
                                    f"динамический импорт '{literal}' в пути награды "
                                    f"через {name}()",
                                )
                            )
            if last in _PROCESS_CALL_NAMES:
                for arg in node.args:
                    for literal in _literal_strings(arg):
                        if _NETWORK_CLI_RE.search(literal) or _URL_RE.search(literal):
                            findings.append(
                                Finding(
                                    "REWARD-PATH-NETWORK-CALL",
                                    source.rel,
                                    node.lineno,
                                    "запуск процесса с сетевым адресом/CLI "
                                    f"('{literal[:80]}') в пути награды",
                                )
                            )
    for number, line in enumerate(source.text.splitlines(), start=1):
        if _PROVIDER_RE.search(line) and _CREDENTIAL_RE.search(line):
            findings.append(
                Finding(
                    "REWARD-PATH-PROVIDER-CREDENTIAL",
                    source.rel,
                    number,
                    "упоминание провайдера LLM и ключа/эндпоинта в пути награды: "
                    f"«{line.strip()[:100]}»",
                )
            )
    return findings


def _resolve(
    ref: ImportRef, source: SourceFile, index: dict[str, SourceFile]
) -> list[SourceFile]:
    """Локальные модули, на которые указывает импорт (пусто для внешних)."""
    found: list[SourceFile] = []
    for name in ref.candidates(source.package):
        target = index.get(name)
        if target is not None and target.rel != source.rel and target not in found:
            found.append(target)
    return found


def build_report(root: Path) -> Report:
    """Полный прогон: сиды → пакет контура → импорт-замыкание → проверки."""
    report = Report(root=root)
    sources = [load_source(path, root) for path in scan_candidates(root)]
    index: dict[str, SourceFile] = {}
    for source in sources:
        index.setdefault(source.module, source)

    seeds: list[SourceFile] = []
    for source in sources:
        reasons = seed_reasons(source)
        if reasons:
            seeds.append(source)
            report.seeds.append((source.rel, "; ".join(reasons)))
    if not seeds:
        return report

    judges: dict[str, str] = {}
    for source in sources:
        reasons = judge_reasons(source)
        if reasons:
            judges[source.rel] = "; ".join(reasons)
    report.judge_modules = sorted(judges.items())

    packages = {package_dir(seed.path, root) for seed in seeds}
    seed_basis = {seed.rel: report.seeds[i][1] for i, seed in enumerate(seeds)}
    core: dict[str, SourceFile] = {}
    for source in sources:
        if package_dir(source.path, root) in packages:
            core[source.rel] = source
            if source.rel in seed_basis:
                basis = f"сид: {seed_basis[source.rel]}"
            else:
                package = package_dir(source.path, root).relative_to(root).as_posix()
                basis = f"пакет контура {package}/ (сиды: {', '.join(sorted(seed_basis))})"
            report.core.append(CircuitFile(source.rel, basis))

    # Импорт-замыкание: то, что контур вызывает (не контур, но в пути награды).
    dependencies: dict[str, SourceFile] = {}
    dependency_basis: dict[str, str] = {}
    queue = list(core.values())
    while queue:
        current = queue.pop()
        for ref in current.imports:
            for target in _resolve(ref, current, index):
                if target.rel in core or target.rel in dependencies:
                    continue
                dependencies[target.rel] = target
                dependency_basis[target.rel] = f"← {current.rel}: {ref.text}"
                queue.append(target)
    report.dependencies = [
        CircuitFile(rel, dependency_basis[rel]) for rel in sorted(dependencies)
    ]

    checked = list(core.values()) + [dependencies[rel] for rel in sorted(dependencies)]
    imported_judges: set[str] = set()
    for source in checked:
        report.findings.extend(_check_imports(source, index, judges, imported_judges))
        report.findings.extend(_check_calls(source))

    for rel in sorted(set(judges) & set(core) - imported_judges):
        report.warnings.append(
            Warning(
                "JUDGE-MODULE-IN-CIRCUIT-PACKAGE",
                rel,
                f"судейский модуль '{rel}' лежит в пакете контура награды и не "
                "импортируется из него — допустимо только как отдельная метрика; "
                "держите его вне пакета, чтобы граница была видна",
            )
        )

    report.findings.sort(key=lambda finding: (finding.rel, finding.line, finding.code))
    return report


# --- вывод -------------------------------------------------------------------


def render(report: Report) -> str:
    out = [
        "C-039 · контур награды RL (AD-2): LLM-судья и внешние API вне пути награды",
        f"Каталог: {report.root}",
        "Область сканирования: **/*.py (кроме .git, __pycache__, tests/, test_*.py, conftest.py, самого инструмента)",
    ]

    if not report.verified:
        out.append("")
        out.append(
            f"{NOT_VERIFIED_MESSAGE} — ни один модуль не объявляет награду "
            "(маркеры: имя 'reward', класс Reward, функция с 'reward', "
            "def compute(...) рядом с reward)."
        )
        out.append(
            "Проверка не выполнена: PASS не выдаётся (ложный PASS запрещён)."
        )
        return "\n".join(out)

    out.append("")
    out.append(f"Сиды контура ({len(report.seeds)}) — на каком основании:")
    out.extend(f"  {rel} — {basis}" for rel, basis in report.seeds)

    out.append("")
    out.append(f"Файлы контура награды ({len(report.core)}):")
    out.extend(f"  {item.rel} — {item.basis}" for item in report.core)

    out.append("")
    if report.dependencies:
        out.append(
            f"Импорт-зависимости контура ({len(report.dependencies)}) — "
            "не контур, но вызываются из него, поэтому проверяются:"
        )
        out.extend(f"  {item.rel} {item.basis}" for item in report.dependencies)
    else:
        out.append("Импорт-зависимости контура: нет (контур замкнут в своём пакете)")

    out.append("")
    if report.judge_modules:
        out.append(f"Судейские модули в репозитории ({len(report.judge_modules)}):")
        out.extend(f"  {rel} — {basis}" for rel, basis in report.judge_modules)
    else:
        out.append("Судейские модули в репозитории: не найдены")

    out.append("")
    if report.findings:
        out.append(f"Нарушения ({len(report.findings)}):")
        out.extend(
            f"  [{finding.code}] {finding.rel}:{finding.line} — {finding.message}"
            for finding in report.findings
        )
    else:
        out.append(
            "Нарушения (0): обращений к LLM-судье, LLM-клиентам и внешним API "
            "в пути награды нет"
        )

    if report.warnings:
        out.append("")
        out.append(f"Замечания ({len(report.warnings)}) — гейт не красят:")
        out.extend(
            f"  [{warning.code}] {warning.rel} — {warning.message}"
            for warning in report.warnings
        )

    out.append("")
    out.append(
        "Итог: PASS — награда считается из артефактов, судья вне контура"
        if report.ok
        else "Итог: FAIL — судья/LLM-клиент/внешний API в пути награды (AD-2)"
    )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="C-039: LLM-судья вне контура награды RL (AD-2)"
    )
    parser.add_argument(
        "--root",
        default=str(DEFAULT_ROOT),
        help="каталог кейса (по умолчанию — каталог, содержащий tools/)",
    )
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()

    try:
        report = build_report(root)
    except ScanError as exc:
        print(f"{NOT_VERIFIED_MESSAGE}: {exc}")
        print("Проверка не выполнена: PASS не выдаётся (ложный PASS запрещён).")
        print(f"check_reward_isolation: {exc}", file=sys.stderr)
        return EXIT_NOT_VERIFIED

    print(render(report))
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
