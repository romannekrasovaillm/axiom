"""Минимальный загрузчик YAML-подмножества (stdlib-only).

Зачем свой разбор: правило C-047 исполняется в песочнице воркспейса без сети и
GPU, где сторонних зависимостей (PyYAML) может не быть. Реестры поведенческого
слоя (``model/claims.yaml``, ``model/sensors.yaml``, ``evidence/incidents.yaml``)
пишутся в строгом подмножестве YAML, а его достаточно прочитать стандартной
библиотекой.

Поддерживается ровно то, что нужно реестрам, и не больше:

* блочные отображения (``key: value``) и блочные последовательности (``- ...``);
* вложенность по отступам (пробелы);
* скаляры: строка (в кавычках или bare), число, ``true/false``, ``null``/``~``;
* flow-значения: список ``[a, b]`` и отображение ``{k: v}``;
* комментарии ``#`` вне кавычек/скобок.

Это не общий YAML: неподдерживаемая конструкция поднимает :class:`MiniYamlError`,
а не угадывается. Так честнее: реестр, который не читается, должен быть виден.
"""

from __future__ import annotations

import re
from typing import Any

_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?$")


class MiniYamlError(ValueError):
    """Текст вне поддерживаемого подмножества (не угадывается)."""


def load(text: str) -> Any:
    """Разбирает документ. Пустой текст → ``None``."""
    lines = _preprocess(text)
    if not lines:
        return None
    value, index = _parse_block(lines, 0, lines[0][0])
    if index != len(lines):
        line_no, content = lines[index][2], lines[index][1]
        raise MiniYamlError(f"строка {line_no}: неожиданный отступ/content {content!r}")
    return value


def load_file(path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return load(fh.read())


# ── подготовка строк ────────────────────────────────────────────────────────


def _preprocess(text: str) -> list[tuple[int, str, int]]:
    """Возвращает список ``(indent, content, line_no)`` без пустых строк/комментариев."""
    out: list[tuple[int, str, int]] = []
    for line_no, raw in enumerate(text.splitlines(), start=1):
        if "\t" in raw[: len(raw) - len(raw.lstrip())]:
            raise MiniYamlError(f"строка {line_no}: табуляция в отступе не поддерживается")
        stripped = _strip_comment(raw)
        if not stripped.strip():
            continue
        indent = len(stripped) - len(stripped.lstrip(" "))
        out.append((indent, stripped.strip(), line_no))
    return out


def _strip_comment(line: str) -> str:
    quote: str | None = None
    depth = 0
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        elif ch == "#" and depth == 0 and (i == 0 or line[i - 1] in " \t"):
            return line[:i]
    return line


# ── блочный разбор ──────────────────────────────────────────────────────────


def _parse_block(lines, index, indent):
    if index >= len(lines):
        return None, index
    if lines[index][0] != indent:
        raise MiniYamlError(
            f"строка {lines[index][2]}: ожидался отступ {indent}, получен {lines[index][0]}"
        )
    if lines[index][1].startswith("-"):
        return _parse_seq(lines, index, indent)
    return _parse_map(lines, index, indent)


def _parse_seq(lines, index, indent):
    items: list[Any] = []
    while index < len(lines) and lines[index][0] == indent and lines[index][1].startswith("-"):
        raw = lines[index][1]
        rest = raw[1:]
        lead = len(rest) - len(rest.lstrip(" "))
        item_indent = indent + 1 + lead
        content = rest.strip()
        if content == "":
            index += 1
            if index < len(lines) and lines[index][0] > indent:
                value, index = _parse_block(lines, index, lines[index][0])
            else:
                value = None
            items.append(value)
        elif _is_map_entry(content):
            lines[index] = (item_indent, content, lines[index][2])
            value, index = _parse_map(lines, index, item_indent)
            items.append(value)
        else:
            items.append(_parse_scalar(content, lines[index][2]))
            index += 1
    return items, index


def _parse_map(lines, index, indent):
    result: dict[str, Any] = {}
    while index < len(lines) and lines[index][0] == indent and not lines[index][1].startswith("-"):
        content = lines[index][1]
        line_no = lines[index][2]
        key, value_text = _split_key(content, line_no)
        if value_text == "":
            index += 1
            if index < len(lines) and lines[index][0] > indent:
                value, index = _parse_block(lines, index, lines[index][0])
            elif index < len(lines) and lines[index][0] == indent and lines[index][1].startswith("-"):
                value, index = _parse_seq(lines, index, indent)
            else:
                value = None
            result[key] = value
        else:
            result[key] = _parse_scalar(value_text, line_no)
            index += 1
    return result, index


def _is_map_entry(content: str) -> bool:
    try:
        _split_key(content, 0)
    except MiniYamlError:
        return False
    return True


def _split_key(content: str, line_no: int) -> tuple[str, str]:
    quote: str | None = None
    depth = 0
    for i, ch in enumerate(content):
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        elif ch == ":" and depth == 0:
            key = content[:i].strip()
            if not key:
                raise MiniYamlError(f"строка {line_no}: пустой ключ")
            return _unquote(key), content[i + 1 :].strip()
    raise MiniYamlError(f"строка {line_no}: ожидалась пара key: value, получено {content!r}")


# ── скаляры и flow ──────────────────────────────────────────────────────────


def _parse_scalar(text: str, line_no: int) -> Any:
    if text.startswith("["):
        inner, rest = _take_flow(text, "[", "]", line_no)
        return [_parse_scalar(part.strip(), line_no) for part in _split_top(inner) if part.strip() != ""]
    if text.startswith("{"):
        inner, rest = _take_flow(text, "{", "}", line_no)
        mapping: dict[str, Any] = {}
        for part in _split_top(inner):
            if not part.strip():
                continue
            key, value_text = _split_key(part.strip(), line_no)
            mapping[key] = _parse_scalar(value_text, line_no)
        return mapping
    return _atom(text)


def _take_flow(text: str, open_ch: str, close_ch: str, line_no: int) -> tuple[str, str]:
    quote: str | None = None
    depth = 0
    for i, ch in enumerate(text):
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[1:i], text[i + 1 :].strip()
    raise MiniYamlError(f"строка {line_no}: незакрытый flow {text!r}")


def _split_top(inner: str) -> list[str]:
    parts: list[str] = []
    quote: str | None = None
    depth = 0
    current = ""
    for ch in inner:
        if quote:
            current += ch
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
            current += ch
        elif ch in "[{":
            depth += 1
            current += ch
        elif ch in "]}":
            depth -= 1
            current += ch
        elif ch == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += ch
    if current.strip():
        parts.append(current)
    return parts


def _atom(text: str) -> Any:
    if len(text) >= 2 and text[0] in "\"'" and text[-1] == text[0]:
        return _unquote(text)
    low = text.lower()
    if low in ("null", "~", ""):
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    if _INT_RE.match(text):
        return int(text)
    if _FLOAT_RE.match(text):
        return float(text)
    return text


def _unquote(text: str) -> str:
    if len(text) >= 2 and text[0] in "\"'" and text[-1] == text[0]:
        body = text[1:-1]
        if text[0] == '"':
            return body.replace('\\"', '"').replace("\\\\", "\\").replace("\\n", "\n")
        return body.replace("''", "'")
    return text
