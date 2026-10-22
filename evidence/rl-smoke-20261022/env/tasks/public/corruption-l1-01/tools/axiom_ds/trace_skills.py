"""Потоки T (трейсы) и S (скиллы) для доменного корпуса токенизатора (ADR-043, ADR-044).

Модуль — только **извлечение текста** из источников: он отдаёт последовательность
текстовых единиц (сообщение сессии, тело методики), а решение о том, что считать
документом (абзац, контракт ``\\n\\n``), принимает билдер
(``tools/build_domain_corpus.py``) — как и для потока R. Скраб сюда не входит:
это ступень билдера, до записи корпуса (ADR-020, ADR-043 п.3).

Отбор источника — по **разбору строк, а не по имени файла** (ADR-043 п.2):

* ``~/.claude/projects``, ``~/.qwen/**/chats`` (формат Claude Code) — записи
  ``type ∈ {user, assistant}``: берутся блоки ``text`` и ``tool_result.content``,
  служебные (``tool_use``, ``thinking``) игнорируются;
* ``~/.arch-ml/sessions`` — ``kind ∈ {user, assistant}``: ``content`` и
  ``reasoning_content``; ``system`` (системный промпт + память контура —
  повторяется в каждой сессии), ``usage``/``event``/``tool`` исключаются целиком;
* ``~/.openclaw`` — только записи с ключом ``conversation`` (готовые траектории);
* ``~/spine-wow-live``, ``~/spine-archive-*``, ``~/.arch-ml/reports`` — текстовые
  отчёты как есть; ``*.json`` под каталогом ``reports/`` — тоже текст отчёта
  прогона (ADR-043 Amendment п.4).

Ничего не открывается у запрещённых путей и суффиксов (``venv/``,
``node_modules/``, ``site-packages/``, ``evals/``, ``*.so``, ``*.db``, ``*.pack``,
архивы, изображения) — deny-list проверяется до чтения.

Структурные данные (``*.yaml``/``*.yml``, а также ``*.json`` вне ``reports/``) в
поток T не входят: это служебные конфиги и метрики, а не трейсы (засоряют
словарь, ADR-043 п.2 показывает чем). Отдельно: ``*.json``/``*.yaml``/``*.yml``
внутри spine-каталогов (``spine-*/**``) исключены всегда — это конфиги и данные
модели, а не текст (ADR-043 Amendment п.4). Решения зафиксированы в манифесте
корпуса (``streams.t.notes``).

YAML-фронтматтер ``SKILL.md`` (12.4% объёма библиотеки) в поток S **не входит**:
структурные ключи срезаются, ``description`` (триггеры и синонимы запросов —
доменный текст) сохраняется отдельной единицей, тело методики — целиком
(ADR-044 п.3).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

#: Метки потоков (совпадают с ключами отчёта в манифесте корпуса).
TRACES = "t"
SKILLS = "s"

#: Компоненты пути, содержимое которых не читается никогда (ADR-043 п.2).
EXCLUDE_DIR_COMPONENTS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "venv",
        ".venv",
        "node_modules",
        "site-packages",
        "dist-packages",
        "evals",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".cache",
    }
)

#: Каталоги со служебными метриками сессий: ``~/.qwen/usage`` (token-usage-*.jsonl).
USAGE_DIR_COMPONENTS = frozenset({"usage"})

#: Файлы-метрики по имени: не читаются, где бы ни лежали.
USAGE_FILE_NAMES = frozenset({"usage_record.jsonl", "token-usage.jsonl"})

#: Суффиксы, которые не открываются: библиотеки, базы, архивы, изображения, бинарь.
EXCLUDE_SUFFIXES = (
    ".so", ".dylib", ".dll", ".a", ".o", ".obj", ".lib", ".exe", ".bin", ".wasm",
    ".db", ".sqlite", ".sqlite3", ".db-wal", ".db-shm",
    ".pack", ".idx", ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".pyc", ".pyo", ".class", ".jar",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".ico", ".svgz", ".tiff",
    ".mp3", ".mp4", ".wav", ".ogg", ".webm", ".mov",
    ".pdf", ".woff", ".woff2", ".ttf", ".otf", ".eot",
)

#: Текстовые расширения потока T, читаемые как есть (отчёты, экспорты сессий).
TRACES_TEXT_SUFFIXES = (".md", ".txt", ".log", ".rst", ".csv")

#: Каталог отчётов прогонов: ``*.json`` под ним — содержательный текст, а не
#: конфиг (ADR-043 Amendment п.4: ``~/.arch-ml/reports/**/*.json`` включаются).
REPORTS_DIR_NAME = "reports"

#: Spine-каталоги (``spine-wow-live``, ``spine-archive-*``): их ``*.json``/``*.yaml``
#: — структурные данные модели, а не текст; исключаются всегда (ADR-043 Amend. п.4).
SPINE_DIR_RE = re.compile(r"^spine-")

#: Шаблоны обхода по умолчанию: T — всё поддерево (фильтр по суффиксу/формату),
#: S — только markdown-методики (ADR-044 п.2).
TRACE_PATTERNS = ("**/*",)
SKILL_PATTERNS = ("**/*.md",)

#: Источники T по умолчанию: (корень, метка). SQLite-сессии (openclaw/hermes)
#: здесь отсутствуют сознательно — они ждут отдельной дельты (ADR-043 п.7).
DEFAULT_TRACE_SOURCES: tuple[tuple[str, str], ...] = (
    ("~/.claude/projects", "claude-projects"),
    ("~/.arch-ml/sessions", "arch-ml-sessions"),
    ("~/.openclaw", "openclaw"),
    ("~/.qwen", "qwen"),
    ("~/.arch-ml/reports", "arch-ml-reports"),
    ("~/spine-wow-live", "spine-wow-live"),
    ("~/spine-archive-20260927", "spine-archive"),
)

#: Источники S по умолчанию: (корень, метка).
DEFAULT_SKILL_SOURCES: tuple[tuple[str, str], ...] = (
    ("~/experiments/agents/0710-ariadna/plugins", "plugins-library"),
    ("~/.arch-ml/plugins", "arch-ml-plugins"),
)

#: Записи сессий Claude Code / Qwen, из которых берётся текст.
_SESSION_TYPES = frozenset({"user", "assistant"})

#: Сессии Spine: берутся только реплики; системный промпт и служебное — нет.
_SESSION_KINDS = frozenset({"user", "assistant"})


class StreamError(Exception):
    """Отказ потока: неизвестный режим, неразбираемый шаблон обхода."""


@dataclass(frozen=True)
class StreamSource:
    """Источник потока T или S: корень, режим разбора, маски файлов и метка."""

    root: str
    mode: str
    patterns: tuple[str, ...]
    label: str

    @property
    def path(self) -> Path:
        return Path(os.path.expanduser(self.root))


# --------------------------------------------------------------------------- #
# Отбор файлов
# --------------------------------------------------------------------------- #


def is_excluded(path: str | os.PathLike[str]) -> bool:
    """True, если файл не читается: deny-list каталога, имени или суффикса.

    Проверяются все компоненты пути, поэтому ``--traces-source`` не может
    «затащить» внутрь запрещённый каталог.
    """
    p = Path(os.path.expanduser(os.fspath(path)))
    if p.name in USAGE_FILE_NAMES:
        return True
    for part in p.parts:
        if part in EXCLUDE_DIR_COMPONENTS or part in USAGE_DIR_COMPONENTS:
            return True
    return p.suffix.lower() in EXCLUDE_SUFFIXES


def _glob_regex(pattern: str) -> re.Pattern[str]:
    """Glob → регулярка с честной семантикой ``**`` (``*`` не пересекает ``/``).

    ``pathlib`` до 3.13 разбирает ``**`` как обычную звёздочку, а ``fnmatch``
    пускает ``*`` через разделитель — оба дают неверный отбор вложенных файлов.
    Нужен ровно один предсказуемый смысл, поэтому трансляция своя.
    """
    out: list[str] = ["(?s:"]
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == "*":
            if pattern[index : index + 2] == "**":
                index += 2
                if pattern[index : index + 1] == "/":  # «**/» — любая глубина
                    out.append("(?:.*/)?")
                    index += 1
                else:
                    out.append(".*")
            else:
                out.append("[^/]*")
                index += 1
        elif char == "?":
            out.append("[^/]")
            index += 1
        else:
            out.append(re.escape(char))
            index += 1
    out.append(r")\Z")
    return re.compile("".join(out))


def _matches(relative: str, patterns: Sequence[str]) -> bool:
    return any(_glob_regex(pattern).match(relative) for pattern in patterns)


def stream_files(spec: StreamSource) -> list[Path]:
    """Файлы источника: обход с отсечением запрещённых каталогов, детерминированный.

    Порядок — сортировка по posix-относительному пути (``readdir`` порядка не
    даёт, а корпус обязан воспроизводиться). Каталоги из deny-list не обходятся
    вовсе — в них нечего читать, а обход ``~/.openclaw`` стоит гигабайты.
    """
    for pattern in spec.patterns:  # шаблон, который не транслируется, — отказ
        _glob_regex(pattern)
    root = spec.path
    if not root.is_dir():
        return []
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(
            name
            for name in dirnames
            if name not in EXCLUDE_DIR_COMPONENTS and name not in USAGE_DIR_COMPONENTS
        )
        base = Path(dirpath)
        for name in sorted(filenames):
            path = base / name
            if is_excluded(path):
                continue
            relative = path.relative_to(root).as_posix()
            if _matches(relative, spec.patterns):
                found.append(path)
    return sorted(found, key=lambda path: path.relative_to(root).as_posix())


def parse_stream_source(raw: str, *, mode: str) -> StreamSource:
    """``ROOT[::GLOB[,GLOB...]]`` → :class:`StreamSource`.

    Без явных масок берутся маски потока (``**/*`` для T, ``**/*.md`` для S).
    """
    if mode not in (TRACES, SKILLS):
        raise StreamError(f"неизвестный режим потока: {mode!r}")
    root, _, patterns_raw = raw.partition("::")
    root = root.strip()
    if not root:
        raise StreamError(f"пустой источник потока: {raw!r}")
    patterns = tuple(p.strip() for p in patterns_raw.split(",") if p.strip())
    if not patterns:
        patterns = TRACE_PATTERNS if mode == TRACES else SKILL_PATTERNS
    stem = Path(os.path.expanduser(root)).name
    return StreamSource(root=root, mode=mode, patterns=patterns, label=stem or root)


def default_sources(mode: str) -> list[StreamSource]:
    """Источники потока по умолчанию (T или S) в объявленном порядке."""
    if mode == TRACES:
        declared, patterns = DEFAULT_TRACE_SOURCES, TRACE_PATTERNS
    elif mode == SKILLS:
        declared, patterns = DEFAULT_SKILL_SOURCES, SKILL_PATTERNS
    else:
        raise StreamError(f"неизвестный режим потока: {mode!r}")
    return [
        StreamSource(root=root, mode=mode, patterns=patterns, label=label)
        for root, label in declared
    ]


# --------------------------------------------------------------------------- #
# Разбор форматов
# --------------------------------------------------------------------------- #


def _safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:  # правая защита: файл исчез между обходом и чтением
        return ""


def _blocks_text(content: object) -> Iterator[str]:
    """Текст блоков ``message.content`` формата Claude Code (строка или список)."""
    if isinstance(content, str):
        if content.strip():
            yield content
        return
    if not isinstance(content, list):
        return
    for block in content:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                yield text
        elif kind == "tool_result":
            # Результат инструмента — тоже реплика трейса (ADR-043, карта форматов).
            yield from _tool_result_text(block.get("content"))


def _tool_result_text(content: object) -> Iterator[str]:
    if isinstance(content, str):
        if content.strip():
            yield content
        return
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    yield text


def _record_texts(record: dict) -> Iterator[str]:
    """Тексты одной записи jsonl — по разбору полей, а не по имени файла.

    Порядок распознавания фиксирован: ``conversation`` (openclaw) → ``kind``
    (сессии Spine) → ``type`` (Claude Code / Qwen). Запись, не похожая ни на один
    формат, не даёт текста.
    """
    conversation = record.get("conversation")
    if isinstance(conversation, str) and conversation.strip():
        yield conversation
        return
    kind = record.get("kind")
    if isinstance(kind, str):
        if kind not in _SESSION_KINDS:
            return  # system/usage/event/tool — служебное, в корпус не идёт
        for key in ("content", "reasoning_content"):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                yield value
            elif isinstance(value, list):
                yield from _blocks_text(value)
        return
    rtype = record.get("type")
    if isinstance(rtype, str) and rtype in _SESSION_TYPES:
        message = record.get("message")
        if isinstance(message, dict):
            yield from _blocks_text(message.get("content"))
        else:
            yield from _blocks_text(record.get("content"))


def iter_jsonl_texts(path: Path) -> Iterator[str]:
    """Тексты сессионного jsonl. Битые строки пропускаются молча (баннер сессии)."""
    try:
        handle = open(path, "r", encoding="utf-8", errors="replace")
    except OSError:
        return
    with handle:
        for line in handle:
            line = line.strip()
            if not line or line[0] != "{":
                continue
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(record, dict):
                yield from _record_texts(record)


# --------------------------------------------------------------------------- #
# YAML-фронтматтер методик
# --------------------------------------------------------------------------- #

_FRONTMATTER_KEY = re.compile(r"^(?P<key>[A-Za-z0-9_.\-]+)\s*:\s*(?P<value>.*)$")


def _frontmatter_description(lines: Sequence[str]) -> str:
    """``description`` из фронтматтера: скаляр, многострочный или блочный.

    Структурные ключи (``name``, ``allowed-tools``, ``license``…) не возвращаются:
    это служебные поля, а не доменный текст (ADR-044 п.3).
    """
    for index, line in enumerate(lines):
        match = _FRONTMATTER_KEY.match(line.strip())
        if not match or match.group("key") != "description":
            continue
        value = match.group("value").strip()
        indent = len(line) - len(line.lstrip())
        if value in ("", "|", ">", "|-", ">-", "|+", ">+"):
            block = "|" in value or value.startswith("|")
            collected: list[str] = []
            for follow in lines[index + 1 :]:
                if not follow.strip():
                    collected.append("")
                    continue
                follow_indent = len(follow) - len(follow.lstrip())
                if follow_indent <= indent:
                    break
                collected.append(follow.strip())
            text = ("\n" if block else " ").join(part for part in collected).strip()
            return text
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        continuation: list[str] = [value]
        for follow in lines[index + 1 :]:
            if not follow.strip() or _FRONTMATTER_KEY.match(follow.strip()):
                break
            if (len(follow) - len(follow.lstrip())) <= indent:
                break
            continuation.append(follow.strip())
        return " ".join(part for part in continuation if part).strip()
    return ""


def strip_frontmatter(text: str) -> tuple[str, str]:
    """``(тело, description)``: структурные ключи срезаются, ``description`` — текст.

    Незакрытый фронтматтер (файл не по контракту) не режется: угадывать границу
    там, где её нет, — значит потерять тело методики.
    """
    if not text.startswith("---"):
        return text, ""
    lines = text.splitlines(keepends=True)
    if lines[0].strip() != "---":
        return text, ""
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            front = [line.rstrip("\n") for line in lines[1:index]]
            body = "".join(lines[index + 1 :])
            return body, _frontmatter_description(front)
    return text, ""


# --------------------------------------------------------------------------- #
# Единицы текста по файлу
# --------------------------------------------------------------------------- #


def _has_component(path: Path, name: str) -> bool:
    return any(part == name for part in path.parts)


def is_structural_spine(path: str | os.PathLike[str]) -> bool:
    """True, если путь лежит внутри spine-каталога (``spine-*/**``).

    Там ``*.json``/``*.yaml`` — структурные данные модели, а не текст
    (ADR-043 Amendment п.4), поэтому такие файлы в поток T не входят. Проверяются
    только каталоги пути: имя файла ``spine-…`` каталогом не является.
    """
    return any(SPINE_DIR_RE.match(part) for part in Path(os.fspath(path)).parts[:-1])


def json_report_allowed(path: str | os.PathLike[str]) -> bool:
    """True, если ``*.json`` — отчёт прогона (``reports/**``), а не конфиг.

    Включаются только json под каталогом ``reports/`` и только вне
    spine-каталогов; всё прочее ``*.json`` — служебный конфиг/метрики
    (ADR-043 Amendment п.4).
    """
    p = Path(os.fspath(path))
    if is_structural_spine(p):
        return False
    return _has_component(p, REPORTS_DIR_NAME)


def iter_texts(path: Path, *, mode: str) -> Iterator[str]:
    """Текстовые единицы файла: сообщения сессии либо (description, тело) методики.

    Поток T: ``*.jsonl`` разбирается как события сессии, текстовые расширения
    и ``*.json`` отчётов (``reports/**``) читаются как есть. Поток S: только
    ``*.md``, фронтматтер срезан.
    """
    suffix = path.suffix.lower()
    if mode == TRACES:
        if suffix == ".jsonl":
            yield from iter_jsonl_texts(path)
        elif suffix == ".json":
            if not json_report_allowed(path):
                return  # конфиг/метрика, не отчёт: в корпус не идёт
            text = _safe_read(path)
            if text.strip():
                yield text
        elif suffix in TRACES_TEXT_SUFFIXES:
            text = _safe_read(path)
            if text.strip():
                yield text
        return
    if mode == SKILLS:
        if suffix != ".md":
            return
        text = _safe_read(path)
        body, description = strip_frontmatter(text)
        for part in (description, body):
            if part.strip():
                yield part
        return
    raise StreamError(f"неизвестный режим потока: {mode!r}")


#: Служебные источники T, ждущие отдельной дельты извлечения из БД (ADR-043 п.7).
#: В манифесте фиксируются как ``EMPTY-PENDING`` с причиной — не имитируются.
PENDING_TRACE_SOURCES: tuple[dict[str, str], ...] = (
    {
        "id": "openclaw-sessions-sqlite",
        "path": "~/.openclaw/agents/main/agent/openclaw-agent.sqlite",
        "status": "EMPTY-PENDING",
        "reason": "сессии openclaw лежат в SQLite (~672 МБ) — требуется отдельная "
                  "дельта извлечения из БД с явным контрактом (ADR-043 п.7)",
    },
    {
        "id": "hermes-sessions-sqlite",
        "path": "~/.hermes/state.db",
        "status": "EMPTY-PENDING",
        "reason": "реальные сессии hermes лежат в SQLite (~719 МБ); jsonl-соседи "
                  "несут только метрики — требуется отдельная дельта (ADR-043 п.7)",
    },
)


def pending_entries() -> list[dict[str, str]]:
    """Записи ``EMPTY-PENDING`` с отметкой, есть ли источник на этой машине."""
    out: list[dict[str, str]] = []
    for declared in PENDING_TRACE_SOURCES:
        entry = dict(declared)
        entry["present"] = "true" if Path(os.path.expanduser(entry["path"])).exists() else "false"
        out.append(entry)
    return out
