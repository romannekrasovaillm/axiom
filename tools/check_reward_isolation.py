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
  (объявленные корни + импорт-замыкание), а какие — нет.

Границы контура (объявлены, не подразумеваются)
-----------------------------------------------

AD-2 связывает «среда ↔ verifier'ы ↔ награда RL-контура»: награда — функция
вердикта, вердикт собирают механические гейты. Контур пути награды объявлен
**списком корней** :data:`REWARD_PATH_ROOTS` (``env/reward.py``,
``env/verifier.py``, ``env/run.py``) — награда, вердикт и обвязка запуска;
так он и записан в SPEC AD-2. Корень проверяется вместе с пакетным
``__init__`` (он исполняется при импорте корня, значит лежит в пути награды) и
транзитивным импорт-замыканием: если награда ходит в помощника, судейский
вызов не должен прятаться за ним.

Импорты, ведущие за пределы пакета корней (``env/``) — например, helper в
другом пакете, — идут отдельным списком «импорт-зависимости» и проверяются
наравне с контуром: в них тоже нельзя прятать судью или LLM-клиента.

Что заведомо вне проверки (граница, а не пробел): модули, **не достижимые из
корней по импортам**, — приборы пакета среды (``env/eval_*.py``,
``env/calibrate.py``, ``env/main.py``, ``env/openai_adapter.py`` — порт без
HTTP) и пакет ``clients/`` (httpx-транспорт, отдельный домен: механизм
внешнего endpoint'а, а не контур награды). Достижимый помощник проверяется,
где бы он ни лежал; недостижимый — нет. Транзитивные зависимости вне дерева
репозитория (site-packages); вызовы, собранные из нестроковых выражений
(URL из переменной окружения); намеренная обфускация (``eval``/``exec``) —
тоже граница. Инструмент не сканирует сам себя
(``tools/check_reward_isolation.py``).

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

#: Объявленные корни пути награды (ADR-033 амендмент; SPEC AD-2 — «пакет среды
#: env/: reward.py, verifier.py, run.py»). Контур — import-замыкание от них, а
#: не весь пакет ``env/``: приборы (``eval_*``, ``calibrate``, ``main``,
#: ``openai_adapter``) контуром награды не являются, и httpx-транспорт в
#: ``clients/`` по этой границе вне проверки. Пути — от корня кейса.
REWARD_PATH_ROOTS: tuple[str, ...] = (
    "env/reward.py",
    "env/verifier.py",
    "env/run.py",
)


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
    roots: list[CircuitFile] = field(default_factory=list)
    core: list[CircuitFile] = field(default_factory=list)
    dependencies: list[CircuitFile] = field(default_factory=list)
    judge_modules: list[tuple[str, str]] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    warnings: list[Warning] = field(default_factory=list)

    @property
    def verified(self) -> bool:
        """Хотя бы один объявленный корень есть — иначе «НЕ ПРОВЕРЕНО»."""
        return bool(self.roots)

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
    """Полный прогон: объявленные корни → пакетный ``__init__`` → импорт-замыкание → проверки."""
    report = Report(root=root)
    sources = [load_source(path, root) for path in scan_candidates(root)]
    index: dict[str, SourceFile] = {}
    by_rel: dict[str, SourceFile] = {}
    for source in sources:
        index.setdefault(source.module, source)
        by_rel.setdefault(source.rel, source)

    # Корни пути награды объявлены (:data:`REWARD_PATH_ROOTS`), а не выведены
    # маркерами: «весь пакет env/» — шире контура (приборы в него не входят).
    core: dict[str, SourceFile] = {}
    basis: dict[str, str] = {}
    for rel in REWARD_PATH_ROOTS:
        source = by_rel.get(rel)
        if source is None:
            continue
        core[rel] = source
        basis[rel] = "объявленный корень пути награды (env/reward.py, env/verifier.py, env/run.py)"
        report.roots.append(CircuitFile(rel, basis[rel]))
    if not report.roots:
        return report

    judges: dict[str, str] = {}
    for source in sources:
        reasons = judge_reasons(source)
        if reasons:
            judges[source.rel] = "; ".join(reasons)
    report.judge_modules = sorted(judges.items())

    # Пакетный ``__init__`` исполняется при импорте корня — тоже путь награды.
    root_packages = {package_dir(source.path, root) for source in core.values()}
    for package in sorted(root_packages):
        init = package / "__init__.py"
        try:
            rel = init.relative_to(root).as_posix()
        except ValueError:  # pragma: no cover — пакет вне корня кейса
            continue
        source = by_rel.get(rel)
        if source is not None and rel not in core:
            core[rel] = source
            basis[rel] = (
                f"__init__ пакета контура {package.relative_to(root).as_posix()}/ "
                "(исполняется при импорте корня)"
            )

    # Импорт-замыкание: что вызывают корни. Модули внутри пакета корней — часть
    # контура; ведущие наружу (helper в другом пакете) — импорт-зависимости.
    dependencies: dict[str, SourceFile] = {}
    queue = list(core.values())
    while queue:
        current = queue.pop()
        for ref in current.imports:
            for target in _resolve(ref, current, index):
                if target.rel in core or target.rel in dependencies:
                    continue
                if package_dir(target.path, root) in root_packages:
                    core[target.rel] = target
                    basis[target.rel] = f"← {current.rel}: {ref.text}"
                else:
                    dependencies[target.rel] = target
                    basis[target.rel] = f"← {current.rel}: {ref.text}"
                queue.append(target)

    report.core = [CircuitFile(rel, basis[rel]) for rel in sorted(core)]
    report.dependencies = [
        CircuitFile(rel, basis[rel]) for rel in sorted(dependencies)
    ]

    checked = [core[rel] for rel in sorted(core)] + [
        dependencies[rel] for rel in sorted(dependencies)
    ]
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
        "Область сканирования: **/*.py; проверяются корни пути награды и их импорт-замыкание",
    ]

    if not report.verified:
        out.append("")
        out.append(
            f"{NOT_VERIFIED_MESSAGE} — ни один объявленный корень пути награды "
            f"не найден ({', '.join(REWARD_PATH_ROOTS)}); пакет env/ без них "
            "контуром не считается."
        )
        out.append(
            "Проверка не выполнена: PASS не выдаётся (ложный PASS запрещён)."
        )
        return "\n".join(out)

    out.append("")
    out.append(
        f"Корни пути награды ({len(report.roots)}) — объявлены, не подразумеваются:"
    )
    out.extend(f"  {item.rel} — {item.basis}" for item in report.roots)

    out.append("")
    out.append(
        f"Файлы контура награды ({len(report.core)}) — корни, пакетный __init__ "
        "и транзитивные импорты внутри пакета:"
    )
    out.extend(f"  {item.rel} — {item.basis}" for item in report.core)

    out.append("")
    if report.dependencies:
        out.append(
            f"Импорт-зависимости контура ({len(report.dependencies)}) — "
            "импортируются корнями за пределами пакета, поэтому проверяются:"
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
