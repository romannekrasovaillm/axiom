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
**не применяются** к ключам hash-семейства — имя режется по ``_``, и если ЛЮБОЙ
сегмент (после lowercase) равен маркеру :data:`HASH_KEY_MARKERS`, присваивание
пропускается целиком. Значения таких ключей суть хеши/соли доменных данных (sha256
дистиллята, соль сплита), их замена портит корпус, а не защищает: ``SALT_KEY`` и
``sha256_HASH`` остаются нетронутыми, а ``HASHICORP_VAULT_TOKEN`` (сегмент
``hashicorp`` — не маркер) ловится. Оба правила **сохраняют имя ключа** — заменяется
значение (``KEY=<REDACTED>``), а не присваивание целиком. Для credential-имён
(key/token/secret/password и родственных) оба правила работают как прежде.
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

#: Маркеры hash-семейства — СЕГМЕНТЫ имени ключа (split по ``_``, после lowercase),
#: при которых hex_env и env_assignment НЕ применяются: хеши и соли — не credential,
#: а их значения — доменные данные (sha256 дистиллята, соль сплита), которые замена
#: портит (ADR-020, дельты-3/3b).
HASH_KEY_MARKERS = ("sha256", "sha1", "md5", "hash", "salt", "checksum")

#: Тот же список множеством: заслон ищет равенство сегмента, а не подстроку.
_HASH_SEGMENTS = frozenset(HASH_KEY_MARKERS)


def _is_hash_key(name: str) -> bool:
    """Имя ключа принадлежит hash-семейству: ЛЮБОЙ его сегмент — маркер.

    Семантика сегментная, а не подстрочная (дельта-3b): ``SALT_KEY`` и
    ``doc_sha256`` исключаются целиком — в том числе когда имя кончается
    credential-словом (``KEY``/``TOKEN``/``SECRET``/``PASSWORD``), — а
    ``HASHICORP_VAULT_TOKEN`` ловится: подстрока ``hash`` внутри сегмента
    ``hashicorp`` маркером не является.
    """
    return any(segment in _HASH_SEGMENTS for segment in name.lower().split("_"))


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
)

#: Сборная регулярка для дешёвой разведки: какие правила вообще срабатывают.
#: Правила присваиваний (``hex_env``, ``env_assignment``) сюда не входят: они несут
#: именованные группы (``head``/``name``), а разведка опознаёт сработавшую ветку по
#: ``m.lastindex`` — именованная группа внутри ветки сдвинула бы его. Их наличие
#: проверяется отдельными поисками в :func:`_touched_patterns`.
_COMBINED = re.compile("|".join(f"({pat})" for _, pat in _PATTERNS))

_NAME_BY_INDEX = {i + 1: name for i, (name, _) in enumerate(_PATTERNS)}

#: Присваивание с длинным hex-значением (≥32) при имени без hash-сегмента: имя
#: сохраняется, значение вычищается — ``KEY=<REDACTED>`` (дельта-3b; прежняя
#: семантика дельты-3 заменяла совпадение целиком вместе с именем). ``\b`` держит
#: скан на границе слова: имя не подхватывается с середины длинного идентификатора.
#: Замыкающая кавычка значения в совпадение не входит и остаётся на месте —
#: ``KEY="<REDACTED>"`` читается как присваивание, а не как рваная строка.
_HEX_ENV = re.compile(
    r"\b(?P<head>(?P<name>[A-Za-z_][A-Za-z0-9_]{2,40})\s*=\s*[\"']?)"
    r"(?P<value>[0-9a-fA-F]{32,})"
)
_HEX_ENV_NAME = "hex_env"

#: Присваивание «секретного» имени: ключ сохраняем, значение вычищаем.
#:
#: Ключ обязан быть похож на переменную окружения — UPPER_SNAKE
#: (``OPENAI_API_KEY``) или lower_snake (``db_password``), и суффикс обязан
#: *завершать* имя. Без этого правила ловится код: ``ExperimentalMaterial3Api``,
#: ``Unauthorized``, ``_cached_key``, ``SPECIAL_KEYS`` — имена, а не секреты.
#: ``SALT`` из списка суффиксов снят дельтой-3b: соли — hash-семейство, и заслон
#: :func:`_is_hash_key` исключает их из правила по сегменту имени (как и ``sha256``/
#: ``md5``/``hash``/``checksum`` в любом сегменте).
_ENV_ASSIGN = re.compile(
    r"(?P<head>(?P<name>"
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


def _redact_assignment(
    pattern: re.Pattern, rule: str, text: str, stats: ScrubStats | None
) -> str:
    """Вычистить значения присваиваний, сохранив имена ключей.

    Имя hash-семейства (:func:`_is_hash_key`) не трогается вовсе: реплейсер
    возвращает совпадение как есть, поэтому текст и счётчики правила остаются
    нетронутыми — срабатывание считается по числу замен, а не по числу найденных
    присваиваний.
    """
    replaced = 0

    def repl(match: re.Match) -> str:
        nonlocal replaced
        if _is_hash_key(match.group("name")):
            return match.group(0)
        replaced += 1
        return match.group("head") + REDACTED

    out = pattern.sub(repl, text)
    if stats is not None:
        stats.add(rule, replaced)
    return out


def _touched_patterns(text: str) -> set[str]:
    """Дешёвая разведка: какие правила встречаются в тексте (один проход)."""
    found: set[str] = set()
    for match in _COMBINED.finditer(text):
        name = _NAME_BY_INDEX.get(match.lastindex or 0)
        if name:
            found.add(name)
        if len(found) == len(_PATTERNS):
            break
    if _HEX_ENV.search(text):
        found.add(_HEX_ENV_NAME)
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
    # Правила присваиваний идут после блочных: их значение — остаток строки, уже
    # очищенный от ключевого материала (token/bearer/…). Оба сохраняют имя ключа и
    # оба пропускают hash-семейство имён (см. :func:`_is_hash_key`).
    if _HEX_ENV_NAME in touched:
        out = _redact_assignment(_HEX_ENV, _HEX_ENV_NAME, out, stats)
    if _ENV_ASSIGN_NAME in touched:
        out = _redact_assignment(_ENV_ASSIGN, _ENV_ASSIGN_NAME, out, stats)
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
