"""Внешний источник механической верификации: отчёты турникета харнесса.

Читаются два формата, оба — про контракт результата ``status``:

* ``result.json`` — машинный контракт прогона (``{"status": "complete", …}``);
* ``*.log`` прогона харнесса — строка «Контракт результата: status=…».

Привязка отчёта к сессии возможна только по времени (в отчётах нет id сессии),
поэтому она **консервативна**: пара принимается, только если временное окно
отчёта пересекается с окном эпизода, и если кандидат ровно один. Неоднозначная
или отсутствующая пара — это ``None`` (unverified), а не догадка: ложный
``verified-complete`` отправил бы в SFT эпизод без подтверждённого исхода.
"""

from __future__ import annotations

import glob as globlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

#: Строка контракта в логе прогона харнесса.
CONTRACT_LINE = re.compile(r"Контракт результата:\s*status=([A-Za-z_-]+)")

#: Метка времени в имени лога: hr-YYYYMMDD-HHMMSS-NN.log
LOG_STAMP = re.compile(r"(\d{8})-(\d{6})")

#: Ширина окна привязки отчёта к эпизоду (секунды).
DEFAULT_WINDOW_SEC = 3600.0


def _parse_iso(value: str | None) -> float | None:
    if not value or not isinstance(value, str):
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


@dataclass
class HarnessReport:
    path: str
    status: str
    timestamp: float | None
    origin: str  # "result.json" | "log"


def _stamp_from_name(path: str) -> float | None:
    match = LOG_STAMP.search(Path(path).name)
    if not match:
        return None
    try:
        parsed = datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc).timestamp()


def _read_result_json(path: str) -> HarnessReport | None:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    status = payload.get("status")
    if not isinstance(status, str):
        return None
    timestamp = _stamp_from_name(path)
    if timestamp is None:
        try:
            timestamp = os.path.getmtime(path)
        except OSError:
            timestamp = None
    return HarnessReport(path=path, status=status, timestamp=timestamp, origin="result.json")


def _read_log(path: str) -> HarnessReport | None:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                match = CONTRACT_LINE.search(line)
                if match:
                    timestamp = _stamp_from_name(path)
                    if timestamp is None:
                        try:
                            timestamp = os.path.getmtime(path)
                        except OSError:
                            timestamp = None
                    return HarnessReport(
                        path=path,
                        status=match.group(1).lower(),
                        timestamp=timestamp,
                        origin="log",
                    )
    except OSError:
        return None
    return None


class HarnessIndex:
    """Индекс отчётов турникета: статусы + временные окна для привязки."""

    def __init__(self, window_sec: float = DEFAULT_WINDOW_SEC) -> None:
        self.window_sec = float(window_sec)
        self._reports: list[HarnessReport] = []
        self._ambiguous_matches = 0

    # -- сборка ----------------------------------------------------------- #

    @classmethod
    def from_globs(cls, patterns: Iterable[str], window_sec: float = DEFAULT_WINDOW_SEC):
        index = cls(window_sec=window_sec)
        for pattern in patterns:
            index.add_glob(pattern)
        return index

    @classmethod
    def from_glob(cls, pattern: str, window_sec: float = DEFAULT_WINDOW_SEC):
        index = cls(window_sec=window_sec)
        index.add_glob(pattern)
        return index

    def add_glob(self, pattern: str) -> int:
        """Добавить отчёты по glob-шаблону; вернуть число разобранных."""
        added = 0
        for path in sorted(globlib.glob(os.path.expanduser(pattern), recursive=True)):
            if not os.path.isfile(path):
                continue
            report = _read_result_json(path) if path.endswith(".json") else _read_log(path)
            if report is None:
                continue
            self._reports.append(report)
            added += 1
        return added

    # -- доступ ------------------------------------------------------------ #

    @property
    def size(self) -> int:
        return len(self._reports)

    @property
    def ambiguous_matches(self) -> int:
        return self._ambiguous_matches

    def statuses(self) -> list[str]:
        return [report.status for report in self._reports]

    def by_status(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for report in self._reports:
            counts[report.status] = counts.get(report.status, 0) + 1
        return dict(sorted(counts.items()))

    def match(self, started_at: str | None, ended_at: str | None) -> str | None:
        """Статус единственного отчёта, чьё окно пересекается с окном эпизода."""
        start = _parse_iso(started_at)
        end = _parse_iso(ended_at)
        if start is None:
            return None
        if end is None or end < start:
            end = start
        low = start - self.window_sec
        high = end + self.window_sec

        candidates: list[HarnessReport] = []
        for report in self._reports:
            if report.timestamp is None:
                continue
            if low <= report.timestamp <= high:
                candidates.append(report)

        unique_statuses = {report.status for report in candidates}
        if len(unique_statuses) > 1 or len(candidates) > 1:
            self._ambiguous_matches += 1
            return None
        return candidates[0].status if candidates else None

    def to_dict(self) -> dict:
        return {
            "reports": self.size,
            "by_status": self.by_status(),
            "ambiguous_matches": self.ambiguous_matches,
            "window_sec": self.window_sec,
        }
