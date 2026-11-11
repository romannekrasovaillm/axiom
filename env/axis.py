"""Ось победы (AD-1, ADR-058): единственный путь в пакете решения.

Ось — не свободный текст прогона: она объявлена заранее в пакете решения
``docs/DECISION-PACKAGE.json`` на пути ``choice.victory_axis`` и копируется
в Run Manifest v2 (``victory_axis``).  Манифест с другой осью (или с осью
на верхнем уровне пакета) доказательством не является: сравнение манифеста
с пакетом идёт по нормализованной строке, поэтому переформулировка оси в
одном месте без другого даёт красный гейт, а не ложное совпадение.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: Путь оси в пакете решения: ``docs/DECISION-PACKAGE.json#/choice/victory_axis``.
AXIS_PATH = ("choice", "victory_axis")

_AXIS_POINTER = "#/" + "/".join(AXIS_PATH)


class AxisError(ValueError):
    """Ось не читается: нет файла, нет ключа, пусто или не строка."""


def load_axis(package_path: Path) -> str:
    """Вернуть непустую ось победы из пакета решения.

    Ошибки (нет файла, не JSON, нет ключа по :data:`AXIS_PATH`, пустое или
    нестроковое значение) — :class:`AxisError` с путём файла и JSON-pointer.
    """
    path = Path(package_path)
    try:
        doc: Any = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AxisError(f"{path}: пакет решения не найден") from exc
    except OSError as exc:
        raise AxisError(f"{path}: пакет решения не читается: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise AxisError(f"{path}: пакет решения не JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise AxisError(f"{path}: пакет решения — не JSON-объект")

    current: Any = doc
    for key in AXIS_PATH:
        if not isinstance(current, dict) or key not in current:
            raise AxisError(f"{path}{_AXIS_POINTER}: ключ оси отсутствует")
        current = current[key]
    if not isinstance(current, str) or not current.strip():
        raise AxisError(f"{path}{_AXIS_POINTER}: ось пуста или не строка")
    return current


def normalize_axis(s: str) -> str:
    """Нормализация оси: пробельные последовательности → один пробел."""
    return " ".join(s.split())


def axis_matches(a: str, b: str) -> bool:
    """Совпадение осей по нормализованным строкам (ADR-058)."""
    return normalize_axis(a) == normalize_axis(b)


def has_top_level_axis(doc: dict) -> bool:
    """Есть ли ``victory_axis`` на верхнем уровне (двусмысленность — ошибка)."""
    return isinstance(doc, dict) and "victory_axis" in doc
