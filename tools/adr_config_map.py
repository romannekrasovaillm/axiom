"""Загрузчик декларативной карты согласованности ADR↔config (C-047).

Карта — данные (:mod:`tools/adr_config_map.yaml`), этот модуль — только их
чтение: страж (`tools/check_adr_config_consistency.py`) и мутатор порчи
(`env/corruption.py`) обязаны видеть ОДНУ и ту же карту, иначе «порча,
детектируемая стражем» перестаёт быть проверяемым утверждением.

Формат — ограниченное подмножество YAML, поддержанное здесь намеренно узко и
без внешних зависимостей (доктрина стражей: stdlib-only, см.
``tools/check_declarative_context.py`` — в среде гейта PyYAML может не быть, а
падать на импорте страж не имеет права):

* ``key: value`` на нулевом отступе — скаляр верхнего уровня;
* ``key:`` без значения — блок-список;
* ``  - key: value`` — элемент-словарь списка (отступ ровно 2);
* ``  - value`` — элемент-скаляр списка (без ``:`` в теле);
* ``    key: value`` — поле элемента (отступ ≥ 4);
* значение в двойных кавычках парсится как JSON-строка (``\\d`` → ``\\d``),
  ``[a, b]`` — как JSON-список, ``true``/``false``/``null`` и целые — по типу;
* полнострочные комментарии (``#``) игнорируются; значение, содержащее ``:``,
  обязано быть в кавычках.

Всё остальное (многострочные скаляры, якоря, вложенные блоки глубже двух
уровней) — ошибка :class:`MapError`, а не молчаливая интерпретация.

API::

    spec = load()                       # или load(Path("tools/adr_config_map.yaml"))
    for m in get_mappings(spec): ...    # {id, adr_pattern, config_field, drift, note}
    adr_files = resolve_adr(case_dir, spec)     # список файлов под glob (0|1|n)
    rel = resolve_adr_strict(case_dir, spec)    # ровно один, иначе MapError
    problems = validate(spec)           # структурные дефекты самой карты
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

#: Карта по умолчанию — рядом с модулем (карта едет вместе со стражем).
DEFAULT_MAP = Path(__file__).resolve().parent / "adr_config_map.yaml"

#: Поля элемента ``mappings``, обязательные к заполнению.
REQUIRED_MAPPING_KEYS = ("id", "adr_pattern", "config_field")

#: Допустимые типы значений верхнего уровня.
_TOP_LEVEL_KEYS = ("version", "adr_glob", "config", "mappings", "swaps", "frozen_config_fields")


class MapError(ValueError):
    """Карта не читается или нарушает собственный контракт."""


def _parse_scalar(raw: str, where: str) -> Any:
    """Скаляр ограниченного подмножества YAML. ``where`` — «файл:строка»."""
    raw = raw.strip()
    if not raw:
        return ""
    if raw[0] in "\"[":
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise MapError(f"{where}: значение не разбирается как JSON-скаляр: {raw!r} ({exc})") from exc
    if raw in ("true", "false"):
        return raw == "true"
    if raw in ("null", "~"):
        return None
    if re.fullmatch(r"-?\d+", raw):
        return int(raw)
    if raw.startswith("'") or raw.endswith("'"):
        raise MapError(f"{where}: одиночные кавычки не поддержаны — используйте двойные")
    return raw


def load(path: Path | str = DEFAULT_MAP) -> dict[str, Any]:
    """Читает карту. Ошибка чтения/разбора — :class:`MapError`, не тихий ``{}``."""
    path = Path(path)
    if not path.is_file():
        raise MapError(f"карта не найдена: {path.as_posix()}")
    doc: dict[str, Any] = {}
    block: str | None = None
    item: dict[str, Any] | None = None
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        where = f"{path.as_posix()}:{lineno}"
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if indent == 0:
            key, sep, raw = stripped.partition(":")
            if not sep or not key.strip():
                raise MapError(f"{where}: ожидалось «key: value» верхнего уровня")
            key = key.strip()
            if raw.strip() == "":
                block, item = key, None
                doc[key] = []
            else:
                doc[key] = _parse_scalar(raw, where)
                block, item = None, None
        elif indent == 2 and stripped.startswith("- "):
            if block is None:
                raise MapError(f"{where}: элемент списка вне блока-списка")
            body = stripped[2:]
            if ":" not in body:
                # элемент-скаляр («- \"tokenizer_hash\"»); значение с «:» обязано
                # быть словарным элементом либо уйти в кавычки-как-словарь
                item = None
                doc[block].append(_parse_scalar(body, where))
                continue
            item = {}
            doc[block].append(item)
            key, sep, raw = body.partition(":")
            if not sep or not key.strip():
                raise MapError(f"{where}: ожидалось «- key: value»")
            item[key.strip()] = _parse_scalar(raw, where)
        elif indent >= 4:
            if item is None:
                raise MapError(f"{where}: поле вне элемента списка")
            key, sep, raw = stripped.partition(":")
            if not sep or not key.strip():
                raise MapError(f"{where}: ожидалось «key: value»")
            item[key.strip()] = _parse_scalar(raw, where)
        else:
            raise MapError(f"{where}: неожиданный отступ {indent}")
    return doc


def get_mappings(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Соответствия карты; не-список — :class:`MapError`."""
    mappings = spec.get("mappings")
    if not isinstance(mappings, list):
        raise MapError("карта: секция «mappings» отсутствует или не список")
    by_id: set[str] = set()
    for entry in mappings:
        if not isinstance(entry, dict):
            raise MapError("карта: элемент «mappings» — не словарь")
        for key in REQUIRED_MAPPING_KEYS:
            if not entry.get(key):
                raise MapError(f"карта: у соответствия {entry.get('id')!r} не заполнено «{key}»")
        if entry["id"] in by_id:
            raise MapError(f"карта: дубль id соответствия {entry['id']!r}")
        by_id.add(entry["id"])
    return mappings


def mapping_by_id(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Индекс соответствий по id."""
    return {m["id"]: m for m in get_mappings(spec)}


def get_swaps(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Свопы term_swap (может отсутствовать)."""
    swaps = spec.get("swaps") or []
    if not isinstance(swaps, list):
        raise MapError("карта: секция «swaps» — не список")
    return swaps


def frozen_config_fields(spec: dict[str, Any]) -> set[str]:
    """Поля config.json, которые смысловая порча не трогает никогда."""
    frozen = spec.get("frozen_config_fields") or []
    if not isinstance(frozen, list):
        raise MapError("карта: секция «frozen_config_fields» — не список")
    return {str(f) for f in frozen}


def config_rel(spec: dict[str, Any]) -> str:
    """Путь конфига относительно корня кейса."""
    value = spec.get("config")
    if not isinstance(value, str) or not value:
        raise MapError("карта: не заполнено поле «config»")
    return value


def resolve_adr(case_dir: Path | str, spec: dict[str, Any]) -> list[str]:
    """Файлы ADR под glob карты — относительные posix-пути, отсортированы.

    Возвращает список (0, 1 или больше): решение «сколько их должно быть»
    принимает вызывающий — страж краснеет, мутатор отказывается работать.
    """
    pattern = spec.get("adr_glob")
    if not isinstance(pattern, str) or not pattern:
        raise MapError("карта: не заполнено поле «adr_glob»")
    root = Path(case_dir)
    return sorted(p.relative_to(root).as_posix() for p in root.glob(pattern) if p.is_file())


def resolve_adr_strict(case_dir: Path | str, spec: dict[str, Any]) -> str:
    """Ровно один файл ADR под glob; иначе :class:`MapError` (порча не гадает)."""
    found = resolve_adr(case_dir, spec)
    if len(found) != 1:
        raise MapError(
            f"карта: glob {spec.get('adr_glob')!r} разрешается в {len(found)} файлов "
            f"({', '.join(found) or 'пусто'}) — нужен ровно один"
        )
    return found[0]


def validate(spec: dict[str, Any]) -> list[str]:
    """Структурные дефекты карты (пусто = карта состоятельна).

    Проверяется то, на чём страж мог бы молча соврать: неизвестные секции,
    некомпилируемые паттерны, отсутствие группы-числа, свопы на несуществующие
    или «дрейфующие» поля, поля из frozen-списка в mappings.
    """
    problems: list[str] = []
    try:
        mappings = get_mappings(spec)
    except MapError as exc:
        return [str(exc)]
    try:
        swaps = get_swaps(spec)
    except MapError as exc:
        return [str(exc)]
    try:
        frozen = frozen_config_fields(spec)
    except MapError as exc:
        return [str(exc)]

    for key in spec:
        if key not in _TOP_LEVEL_KEYS:
            problems.append(f"карта: неизвестная секция «{key}»")

    seen_fields: set[str] = set()
    for entry in mappings:
        where = f"соответствие {entry['id']!r}"
        try:
            pattern = re.compile(entry["adr_pattern"])
        except re.error as exc:
            problems.append(f"{where}: паттерн не компилируется ({exc})")
            continue
        if pattern.groups < 1:
            problems.append(f"{where}: в паттерне нет группы с числом — извлекать нечего")
        field = entry["config_field"]
        if field in seen_fields:
            problems.append(f"{where}: поле config «{field}» уже занято другим соответствием")
        seen_fields.add(field)
        if field.split(".")[0] in frozen:
            problems.append(f"{where}: поле config «{field}» стоит в frozen_config_fields — оно неприкосновенно")

    known = {m["id"] for m in mappings}
    for swap in swaps:
        sid = swap.get("id")
        between = swap.get("between")
        if not isinstance(between, list) or len(between) != 2:
            problems.append(f"своп {sid!r}: поле «between» — список ровно из двух id")
            continue
        for ref in between:
            if ref not in known:
                problems.append(f"своп {sid!r}: неизвестный id соответствия {ref!r}")
            elif mapping_by_id(spec).get(ref, {}).get("drift"):
                problems.append(
                    f"своп {sid!r}: {ref!r} помечен drift: true — один атом переписал бы "
                    "цель другого (numeric_drift и term_swap пересеклись бы)"
                )
    return problems
