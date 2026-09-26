"""Ступень 1 пайплайна axiom-domain-ds-v1: deny-list каталогов + скраб секретов.

Контракт (ADR-020, дельта-1):

* ``is_denied_path(path)`` — путь запрещён к чтению целиком; вызывающий обязан
  пропустить файл, **не открывая** его (не «прочитать и вычистить»);
* ``scrub_text(text)`` — вычищенный текст, секреты заменены на ``<REDACTED>``;
* ``scrub_text_stats(text, stats)`` — то же, но с учётом сработавших правил;
* ``scrub_session(obj, stats=None)`` — рекурсивная копия объекта сессии с
  вычищенными строками (исходный объект не мутируется);
* ``scrub_inplace(obj, stats)`` — тот же обход без копии, для потокового прогона.

Ступень стоит ДО эпизодизации и дедупа: ни один секрет не должен дожить до
выходного jsonl, отчёта или лога. Отчёт получает только счётчики (число замен
по правилам), не значения.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

REDACTED = "<REDACTED>"

# --------------------------------------------------------------------------- #
# Deny-list: то, что не читается вовсе
# --------------------------------------------------------------------------- #

#: Компоненты пути (каталоги), содержимое которых не читается никогда.
DENY_DIR_COMPONENTS = frozenset(
    {
        "API-keys",
        ".ssh",
        ".aws",
        ".gnupg",
    }
)

#: Имена файлов, которые не открываются, где бы они ни лежали.
DENY_FILE_NAMES = frozenset(
    {
        ".netrc",
        ".env",
        "credentials",
        "id_rsa",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
    }
)

#: Расширения файлов-ключей.
DENY_FILE_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".keystore", ".jks")


def is_denied_path(path: str | os.PathLike[str]) -> bool:
    """True, если путь запрещено читать (deny-list каталогов/файлов-ключей).

    Проверяются все компоненты пути, поэтому параметр ``--source`` не может
    «затащить» внутрь запретный каталог.
    """
    try:
        raw = os.fspath(path)
    except TypeError:
        return False
    p = Path(raw).expanduser()
    try:
        parts = p.resolve().parts
    except OSError:  # pragma: no cover - сломанные симлинки/права
        parts = p.absolute().parts
    for part in parts:
        if part in DENY_DIR_COMPONENTS or part in DENY_FILE_NAMES:
            return True
        if part.startswith(".env."):
            return True
        if part.lower().endswith(DENY_FILE_SUFFIXES):
            return True
    return False


# --------------------------------------------------------------------------- #
# Правила скраба
# --------------------------------------------------------------------------- #

#: Порядок важен: сначала «широкие» блочные правила, потом точечные.
#: Все группы внутри правил — необязательные (``(?:...)``), поэтому в сборной
#: регулярке номер сработавшей ветки равен ``m.lastindex``.
_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        "private_key",
        r"-----BEGIN(?:[^-]{0,40})PRIVATE KEY-----[\s\S]{0,65536}?-----END(?:[^-]{0,40})PRIVATE KEY-----",
    ),
    ("openai", r"\bsk-[A-Za-z0-9_\-]{16,}"),
    ("aws", r"\bAKIA[0-9A-Z]{16}\b"),
    ("github", r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    ("slack", r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b"),
    ("jwt", r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"),
    ("bearer", r"(?i)\bbearer\s+[A-Za-z0-9\-._~+/=]{16,}"),
    # URL с паролем: ://user:pass@
    ("url_password", r"(?<=://)[^/\s:@]{1,64}:[^/\s:@]{1,64}(?=@)"),
    # hex/base64 длиной >=32 в контексте присваивания (env-контекст)
    ("hex_env", r"\b[A-Za-z_][A-Za-z0-9_]{2,40}\s*=\s*[\"']?[0-9a-fA-F]{32,}[\"']?"),
)

#: Сборная регулярка для дешёвой разведки: какие правила вообще срабатывают.
_COMBINED = re.compile("|".join(f"({pat})" for _, pat in _PATTERNS))

#: Замена целиком.
_SUB_REDACT = re.compile(REDACTED)

_NAME_BY_INDEX = {i + 1: name for i, (name, _) in enumerate(_PATTERNS)}

#: Присваивание «секретного» имени: ключ сохраняем, значение вычищаем.
_ENV_ASSIGN = re.compile(
    r"(?P<head>(?i:\b[A-Z_][A-Z0-9_]{0,60}(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD"
    r"|CREDENTIAL|CRED|AUTH|SALT|SIGNATURE|API)[A-Z0-9_]{0,20})\s*[=:]\s*[\"']?)"
    r"(?P<value>[^\s\"'`]{16,})"
)
_ENV_ASSIGN_NAME = "env_assignment"


@dataclass
class ScrubStats:
    """Счётчики скраба: сколько замен сделано и какими правилами."""

    total: int = 0
    by_pattern: Counter = field(default_factory=Counter)

    def add(self, name: str, count: int = 1) -> None:
        if count <= 0:
            return
        self.total += count
        self.by_pattern[name] += count

    def merge(self, other: "ScrubStats") -> "ScrubStats":
        self.total += other.total
        self.by_pattern.update(other.by_pattern)
        return self

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "by_pattern": {k: int(v) for k, v in sorted(self.by_pattern.items())},
        }


def _redact(match: re.Match) -> str:
    return REDACTED


def _redact_env(match: re.Match) -> str:
    return match.group("head") + REDACTED


def _touched_patterns(text: str) -> set[str]:
    """Дешёвая разведка: какие правила встречаются в тексте (один проход)."""
    found: set[str] = set()
    for match in _COMBINED.finditer(text):
        name = _NAME_BY_INDEX.get(match.lastindex or 0)
        if name:
            found.add(name)
        if len(found) == len(_PATTERNS):
            break
    if _ENV_ASSIGN.search(text):
        found.add(_ENV_ASSIGN_NAME)
    return found


def scrub_text_stats(text: str, stats: ScrubStats | None = None) -> str:
    """Вычистить текст, пополнив ``stats`` счётчиками по правилам."""
    if not text:
        return text
    touched = _touched_patterns(text)
    if not touched:
        return text
    out = text
    for name, pattern in _PATTERNS:
        if name not in touched:
            continue
        out, count = re.subn(pattern, _redact, out)
        if stats is not None:
            stats.add(name, count)
    if _ENV_ASSIGN_NAME in touched:
        out, count = _ENV_ASSIGN.subn(_redact_env, out)
        if stats is not None:
            stats.add(_ENV_ASSIGN_NAME, count)
    return out


def scrub_text(text: str) -> str:
    """Вычищенный текст: секреты заменены на ``<REDACTED>``."""
    return scrub_text_stats(text, None)


def scrub_session(obj: Any, stats: ScrubStats | None = None) -> Any:
    """Рекурсивная копия объекта сессии с вычищенными строками."""
    return _walk(obj, stats, inplace=False)


def scrub_inplace(obj: Any, stats: ScrubStats | None = None) -> Any:
    """То же, что ``scrub_session``, но без копии — для потокового прогона."""
    return _walk(obj, stats, inplace=True)


def _walk(obj: Any, stats: ScrubStats | None, inplace: bool) -> Any:
    if isinstance(obj, str):
        return scrub_text_stats(obj, stats)
    if isinstance(obj, Mapping):
        target = obj if inplace else {}
        for key, value in obj.items():
            walked = _walk(value, stats, inplace)
            if inplace:
                if walked is not value:
                    target[key] = walked
            else:
                target[key] = walked
        return target
    if isinstance(obj, list):
        target = obj if inplace else []
        for index, value in enumerate(obj):
            walked = _walk(value, stats, inplace)
            if inplace:
                if walked is not value:
                    target[index] = walked
            else:
                target.append(walked)
        return target
    if isinstance(obj, tuple):
        return tuple(_walk(item, stats, inplace) for item in obj)
    return obj


def scrub_many(lines: Iterable[str], stats: ScrubStats | None = None) -> Iterable[str]:
    """Потоковый скраб строк (для логов и вспомогательных текстов)."""
    for line in lines:
        yield scrub_text_stats(line, stats)
