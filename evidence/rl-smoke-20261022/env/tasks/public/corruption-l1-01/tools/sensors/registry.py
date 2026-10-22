"""Реестр датчиков ``model/sensors.yaml`` и проверка его исполнения (дельта C1).

Реестр объявляет каталог датчиков по контекстам исполнения. :func:`probe_sensor`
для каждого ``active`` датчика смотрит на его файл фактов: существует ли,
парсится ли последняя запись по контракту, цела ли цепочка ``prev_sha256``,
свежесть в пределах ``freshness_max_h``. Исход — ``pass | fail | unverified``,
где ``unverified`` — честное «датчик не может измерить» (``pending`` или
последний факт со статусом ``unverified``), а ``fail`` — нарушение контракта.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..miniyaml import MiniYamlError, load_file

from .fact import (
    QUALITIES,
    age_hours,
    facts_path,
    load_contract_check,
    read_records,
    verify_chain,
)
from .subject import REPO_ROOT

#: Реестр датчиков по умолчанию.
DEFAULT_REGISTRY = REPO_ROOT / "model" / "sensors.yaml"

#: Обязательные поля записи реестра.
REQUIRED_FIELDS: tuple[str, ...] = (
    "id",
    "name",
    "facts",
    "context",
    "producer",
    "output",
    "quality",
    "freshness_max_h",
    "zone",
    "status",
)

#: Допустимые статусы датчика.
STATUSES: tuple[str, ...] = ("active", "pending")

#: Допустимые зоны (ADR-025/028).
ZONES: tuple[str, ...] = ("line-1", "line-2")

#: Уровни измерения факта (дельта L, ADR-038).
LEVELS: tuple[str, ...] = ("end_to_end", "component", "diagnostic")


class RegistryError(ValueError):
    """Реестр не читается или нарушает схему."""


def load_sensors(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Читает реестр датчиков (список или ``{"sensors": [...]}``)."""
    registry_path = Path(path) if path is not None else DEFAULT_REGISTRY
    try:
        raw = load_file(registry_path)
    except FileNotFoundError as exc:
        raise RegistryError(f"нет реестра датчиков: {registry_path}") from exc
    except MiniYamlError as exc:
        raise RegistryError(f"{registry_path}: невалидный YAML — {exc}") from exc
    if isinstance(raw, dict):
        raw = raw.get("sensors")
    if not isinstance(raw, list):
        raise RegistryError(
            f"{registry_path}: ожидался список датчиков (или объект с ключом sensors)"
        )
    sensors: list[dict[str, Any]] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise RegistryError(f"{registry_path}: датчик #{index} не объект")
        sensors.append(entry)
    return sensors


def validate_sensors(sensors: list[dict[str, Any]]) -> list[str]:
    """Схема реестра. Возвращает список нарушений (пусто = валидно)."""
    errors: list[str] = []
    seen: set[str] = set()
    for index, entry in enumerate(sensors):
        sid = entry.get("id")
        label = sid if isinstance(sid, str) and sid else f"#{index}"
        for field in REQUIRED_FIELDS:
            if field not in entry:
                errors.append(f"{label}: нет поля {field!r}")
        if not isinstance(sid, str) or not sid.strip():
            errors.append(f"{label}: id — непустая строка")
            continue
        if sid in seen:
            errors.append(f"{label}: дубль id")
        seen.add(sid)

        facts = entry.get("facts")
        if not isinstance(facts, list) or not facts or not all(
            isinstance(f, str) and f for f in facts
        ):
            errors.append(f"{label}: facts — непустой список непустых строк")

        if entry.get("quality") not in QUALITIES:
            errors.append(f"{label}: quality={entry.get('quality')!r} вне {QUALITIES}")

        status = entry.get("status")
        if status not in STATUSES:
            errors.append(f"{label}: status={status!r} вне {STATUSES}")

        zone = entry.get("zone")
        if not isinstance(zone, str) or not zone.strip():
            errors.append(f"{label}: zone — непустая строка (line-1/line-2)")
        else:
            for token in (t.strip() for t in zone.split("|")):
                if token and token not in ZONES:
                    errors.append(f"{label}: zone={token!r} вне {ZONES}")

        fresh = entry.get("freshness_max_h")
        if fresh is not None:
            if isinstance(fresh, bool) or not isinstance(fresh, (int, float)) or fresh <= 0:
                errors.append(f"{label}: freshness_max_h — null или число > 0")

        level = entry.get("level")
        if level is not None and level not in LEVELS:
            errors.append(f"{label}: level={level!r} вне {LEVELS}")
        fact_levels = entry.get("fact_levels")
        if fact_levels is not None:
            if not isinstance(fact_levels, dict):
                errors.append(f"{label}: fact_levels — объект {{факт: уровень}}")
            else:
                for fact, lvl in fact_levels.items():
                    if lvl not in LEVELS:
                        errors.append(f"{label}: fact_levels[{fact}]={lvl!r} вне {LEVELS}")
                    elif isinstance(facts, list) and fact not in facts:
                        errors.append(f"{label}: fact_levels[{fact}] не объявлен в facts")

        for field in ("name", "context", "producer", "output"):
            value = entry.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{label}: {field} — непустая строка")
    return errors


def sensor_fact_index(sensors: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Карта ``факт → [id датчиков]`` (для сверки объявлений утверждений)."""
    index: dict[str, list[str]] = {}
    for entry in sensors:
        sid = str(entry.get("id", ""))
        for fact in entry.get("facts", []) or []:
            if isinstance(fact, str):
                index.setdefault(fact, []).append(sid)
    return index


def find_sensor(sensors: list[dict[str, Any]], sensor_id: str) -> Optional[dict[str, Any]]:
    for entry in sensors:
        if entry.get("id") == sensor_id:
            return entry
    return None


def probe_sensor(
    sensor: dict[str, Any],
    *,
    out_dir: str | Path | None = None,
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    """Исход по одному датчику: ``pass | fail | unverified`` + причина."""
    sid = str(sensor.get("id", "?"))
    status = sensor.get("status")
    result: dict[str, Any] = {
        "id": sid,
        "status": status,
        "verdict": "unverified",
        "reason": "",
        "records": 0,
        "last_ts": None,
        "last_status": None,
    }
    if status == "pending":
        result["reason"] = (
            "датчик pending: не исполнялся (нет источника/окна) — проверять нечего"
        )
        return result

    ok, reason, last = load_contract_check(sid, out_dir)
    if not ok or last is None:
        result["verdict"] = "fail"
        result["reason"] = reason
        return result

    chain_ok, chain_reason = verify_chain(sid, out_dir)
    if not chain_ok:
        result["verdict"] = "fail"
        result["reason"] = chain_reason
        return result

    facts = sensor.get("facts") or []
    records = read_records(sid, out_dir)
    result["records"] = len(records)
    result["last_ts"] = last.get("ts")
    result["last_status"] = last.get("status")

    declared = set(facts)
    if declared and not any(r.get("fact") in declared for r in records):
        result["verdict"] = "fail"
        result["reason"] = (
            f"{sid}: в файле нет ни одного объявленного факта {sorted(declared)} "
            "(реестр разошёлся с датчиком)"
        )
        return result

    fresh = sensor.get("freshness_max_h")
    if fresh is not None and isinstance(fresh, (int, float)):
        age = age_hours(str(last.get("ts") or ""), now)
        if age is None:
            result["verdict"] = "fail"
            result["reason"] = f"{sid}: ts последней записи не разбирается ({last.get('ts')!r})"
            return result
        if age > float(fresh):
            result["verdict"] = "fail"
            result["reason"] = (
                f"{sid}: факт устарел ({age:.1f} ч > freshness_max_h={fresh} ч)"
            )
            return result

    if last.get("status") == "unverified":
        result["verdict"] = "unverified"
        result["reason"] = f"{sid}: последний факт unverified — {last.get('note') or 'причина не указана'}"
        return result

    result["verdict"] = "pass"
    result["reason"] = f"{sid}: последний факт по контракту, цепочка цела"
    return result


def probe_all(
    sensors: list[dict[str, Any]],
    *,
    out_dir: str | Path | None = None,
    now: Optional[datetime] = None,
) -> list[dict[str, Any]]:
    return [probe_sensor(s, out_dir=out_dir, now=now) for s in sensors]


__all__ = [
    "DEFAULT_REGISTRY",
    "REQUIRED_FIELDS",
    "STATUSES",
    "ZONES",
    "RegistryError",
    "load_sensors",
    "validate_sensors",
    "sensor_fact_index",
    "find_sensor",
    "probe_sensor",
    "probe_all",
    "facts_path",
]
