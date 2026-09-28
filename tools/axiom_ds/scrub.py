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

Границы правил (ADR-020, дельты-3/3b): правила ``hex_env`` (длинное hex-значение
в контексте присваивания) и ``env_assignment`` (присваивание «секретного» имени)
**не применяются** к ключам с hash-именами — подстроки :data:`HASH_KEY_MARKERS`
в имени ключа; значения таких ключей суть хеши/соли доменных данных (sha256
дистиллята, соль сплита), их замена портит корпус, а не защищает. Для
credential-имён (key/token/secret/password и родственных) оба правила работают
как прежде.
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

#: Подстроки имени ключа, при которых hex_env и env_assignment НЕ применяются:
#: хеши и соли — не credential, а их значения — доменные данные (sha256
#: дистиллята, соль сплита), которые замена портит (ADR-020, дельты-3/3b).
HASH_KEY_MARKERS = ("sha256", "sha1", "md5", "hash", "salt", "checksum")

#: Имя ключа целиком до hash-маркера — подстрока в любом месте имени, включая
#: суффикс (``SALT_KEY``, ``DOC_SHA256``): hash-семейство имён исключено из обоих
#: правил полностью, а не только по суффиксу.
_HASH_NAME_SCAN = r"[A-Za-z0-9_]*(?i:" + "|".join(HASH_KEY_MARKERS) + r")"

#: Заслон по имени ключа: правило не применяется, если имя содержит hash-подстроку.
#: Lookahead нулевой ширины — своей группы захвата не добавляет, поэтому нумерация
#: ``m.lastindex`` в сборной регулярке не сдвигается. ``\b`` (в варианте для
#: hex_env) ставит скан на границу слова, чтобы lookahead не начал с середины
#: имени: у env_assignment ту же роль играет lookbehind ``(?<![A-Za-z0-9_])``.
_HASH_KEY_GUARD = r"(?!" + _HASH_NAME_SCAN + r")"
_HEX_KEY_GUARD = r"\b" + _HASH_KEY_GUARD

#: Порядок важен: сначала «широкие» блочные правила, потом точечные.
#: Все группы внутри правил — необязательные (``(?:...)``), поэтому в сборной
#: регулярке номер сработавшей ветки равен ``m.lastindex``.
_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        "private_key",
        r"-----BEGIN(?:[^-]{0,40})PRIVATE KEY-----[\s\S]{0,65536}?-----END(?:[^-]{0,40})PRIVATE KEY-----",
    ),
    # Обрезанный блок: чтение с --limit или выдача поиска по файлу показывают
    # заголовок и тело, но не футер. Правило целого блока такое пропускает, а
    # ключевой материал утекает. Берём заголовок вместе со строками тела: строка
    # обязана оканчиваться переводом строки, поэтому проза после «заголовка в
    # тексте» не съедается, а тело реального ключа (≈4 КБ) — съедается целиком.
    (
        "private_key_dangling",
        r"-----BEGIN(?:[^-]{0,40})PRIVATE KEY-----(?:[ \t]*[A-Za-z0-9+/=]*[ \t]*\r?\n){0,2000}",
    ),
    # Осиротевшие маркеры (после снятия тела) — чтобы «BEGIN … PRIVATE KEY» не
    # переживал скраб ни в каком виде.
    (
        "private_key_marker",
        r"-----BEGIN(?:[^-]{0,40})PRIVATE KEY-----|-----END(?:[^-]{0,40})PRIVATE KEY-----",
    ),
    ("openai", r"\bsk-[A-Za-z0-9_\-]{16,}"),
    ("aws", r"\bAKIA[0-9A-Z]{16}\b"),
    ("github", r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    ("slack", r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b"),
    ("jwt", r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"),
    # (?i:…) — локальный флаг: глобальный (?i) запрещён внутри сборной регулярки
    ("bearer", r"(?i:\bbearer\s+[A-Za-z0-9\-._~+/=]{16,})"),
    # URL с паролем: ://user:pass@
    ("url_password", r"(?<=://)[^/\s:@]{1,64}:[^/\s:@]{1,64}(?=@)"),
    # hex длиной >=32 в контексте присваивания (env-контекст), но только для
    # credential-имён: ключи с hash-именами (sha256/salt/…) не трогаются — их
    # значения суть хеши доменных данных, а не секреты (ADR-020, дельта-3).
    ("hex_env", _HEX_KEY_GUARD + r"[A-Za-z_][A-Za-z0-9_]{2,40}\s*=\s*[\"']?[0-9a-fA-F]{32,}[\"']?"),
)

#: Сборная регулярка для дешёвой разведки: какие правила вообще срабатывают.
_COMBINED = re.compile("|".join(f"({pat})" for _, pat in _PATTERNS))

_NAME_BY_INDEX = {i + 1: name for i, (name, _) in enumerate(_PATTERNS)}

#: Присваивание «секретного» имени: ключ сохраняем, значение вычищаем.
#:
#: Ключ обязан быть похож на переменную окружения — UPPER_SNAKE
#: (``OPENAI_API_KEY``) или lower_snake (``db_password``), и суффикс обязан
#: *завершать* имя. Без этого правила ловится код: ``ExperimentalMaterial3Api``,
#: ``Unauthorized``, ``_cached_key``, ``SPECIAL_KEYS`` — имена, а не секреты.
#: ``SALT`` из списка суффиксов снят дельтой-3b: соли — хеш-семейство, и заслон
#: :data:`_HASH_KEY_GUARD` исключает их из правила целиком (как и ``sha256``/
#: ``md5``/``hash``/``checksum`` в любом месте имени).
_ENV_ASSIGN = re.compile(
    _HASH_KEY_GUARD
    + r"(?P<head>(?:"
    r"(?<![A-Za-z0-9_])[A-Z][A-Z0-9_]{0,60}"
    r"(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|CREDS|AUTH|SIGNATURE|API)"
    r"|(?<![A-Za-z0-9_])[a-z][a-z0-9_]*"
    r"(?:key|token|secret|password|passwd|credential|creds|auth|signature|api)"
    r")\s*[=:]\s*[\"']?)"
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
