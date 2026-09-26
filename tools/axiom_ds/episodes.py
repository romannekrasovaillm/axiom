"""Ступень 2 пайплайна axiom-domain-ds-v1: эпизодизация сессии по tool-циклам.

Эпизод — это одна задача пользователя и её отработка агентом:

    user-запрос → ассистент (текст и/или tool_use) → tool_result → … → финальный ответ

Границы: новый эпизод открывает **пользовательский запрос** (событие ``type=user``
с текстовым содержимым). События с ``tool_result`` — это ответ среды, они не
открывают эпизод, а продолжают текущий. События до первого запроса (системные,
мета, снапшоты) в эпизоды не попадают.

Формат эпизода (контракт задачи, дельта-1 ADR-020)::

    {"id": str, "source_session": str, "class": str,
     "turns": [{"role": user|assistant|tool, "kind": text|tool_call|tool_result,
                "content": str}],
     "started_at": iso8601, "ended_at": iso8601}

``content`` у ``tool_call`` — JSON-строка ``{"name": …, "input": …}``: имя
инструмента без аргументов теряет агентную траекторию, поэтому аргументы
сохраняются (см. отчёт дельты, допущение A-3).

Чтение — потоковое: сессия не поднимается в память целиком ни как список строк,
ни как список эпизодов (``iter_episodes`` — генератор).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from . import scrub as scrub_mod

#: Максимальный размер одной строки jsonl (защита от «одной огромной строки»).
MAX_LINE_BYTES = 32 * 1024 * 1024

#: Максимальная длина текста одного хода (защита от раздувания обучающих блоков).
MAX_TURN_CHARS = 200_000

TURN_ROLES = ("user", "assistant", "tool")
TURN_KINDS = ("text", "tool_call", "tool_result")


@dataclass
class Turn:
    role: str
    kind: str
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "kind": self.kind, "content": self.content}


@dataclass
class Episode:
    id: str
    source_session: str
    turns: list[Turn]
    started_at: str
    ended_at: str
    cls: str = "unverified"
    evidence: str = "none"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source_session": self.source_session,
            "class": self.cls,
            "turns": [turn.to_dict() for turn in self.turns],
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "evidence": self.evidence,
        }

    def text(self) -> str:
        return "\n".join(turn.content for turn in self.turns)


@dataclass
class SessionStats:
    """Счётчики чтения одной сессии (числа, без содержимого)."""

    events: int = 0
    parse_errors: int = 0
    oversized_lines: int = 0
    empty_sessions: int = 0

    def to_dict(self) -> dict:
        return {
            "events": self.events,
            "parse_errors": self.parse_errors,
            "oversized_lines": self.oversized_lines,
            "empty_sessions": self.empty_sessions,
        }


# --------------------------------------------------------------------------- #
# Чтение jsonl
# --------------------------------------------------------------------------- #


def iter_jsonl(path: str | os.PathLike[str], stats: SessionStats | None = None) -> Iterator[dict]:
    """Потоково отдать события сессии; битые строки считаются, а не роняют прогон."""
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if len(line) > MAX_LINE_BYTES:
                if stats is not None:
                    stats.oversized_lines += 1
                continue
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, ValueError, RecursionError):
                if stats is not None:
                    stats.parse_errors += 1
                continue
            if not isinstance(event, dict):
                if stats is not None:
                    stats.parse_errors += 1
                continue
            if stats is not None:
                stats.events += 1
            yield event


# --------------------------------------------------------------------------- #
# Разбор события в ходы
# --------------------------------------------------------------------------- #


def _block_text(block: Any) -> str:
    if isinstance(block, str):
        return block
    if not isinstance(block, dict):
        return ""
    for key in ("text", "content"):
        value = block.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts = [_block_text(item) for item in value]
            joined = "\n".join(part for part in parts if part)
            if joined:
                return joined
    return ""


def _truncate(text: str) -> str:
    if len(text) <= MAX_TURN_CHARS:
        return text
    return text[:MAX_TURN_CHARS]


def _message_content(event: dict) -> Any:
    message = event.get("message")
    if isinstance(message, dict):
        return message.get("content")
    return None


def is_user_prompt(event: dict) -> bool:
    """Пользовательский запрос (открывает эпизод), а не ответ среды."""
    if event.get("type") != "user":
        return False
    if event.get("isMeta"):
        return False
    content = _message_content(event)
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        has_text = False
        has_tool_result = False
        for block in content:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "tool_result":
                has_tool_result = True
            elif kind == "text" and str(block.get("text", "")).strip():
                has_text = True
        return has_text and not has_tool_result
    return False


def event_turns(event: dict) -> list[Turn]:
    """Ходы, которые событие добавляет в текущий эпизод."""
    if not isinstance(event, dict):
        return []
    etype = event.get("type")
    content = _message_content(event)
    turns: list[Turn] = []

    if etype == "user":
        if isinstance(content, str):
            if content.strip():
                turns.append(Turn("user", "text", _truncate(content)))
            return turns
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_result":
                    text = _block_text(block.get("content"))
                    turns.append(Turn("tool", "tool_result", _truncate(text)))
                elif block.get("type") == "text" and str(block.get("text", "")).strip():
                    turns.append(Turn("user", "text", _truncate(str(block.get("text", "")))))
        return turns

    if etype == "assistant" and isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "text":
                text = str(block.get("text", ""))
                if text.strip():
                    turns.append(Turn("assistant", "text", _truncate(text)))
            elif kind == "tool_use":
                payload = {
                    "name": str(block.get("name", "")),
                    "input": block.get("input", {}),
                }
                turns.append(
                    Turn(
                        "assistant",
                        "tool_call",
                        _truncate(json.dumps(payload, ensure_ascii=False, sort_keys=True)),
                    )
                )
            # thinking и прочие блоки в контракт ходов не входят (допущение A-4)
        return turns

    return turns


def _episode_id(session_id: str, index: int) -> str:
    digest = hashlib.sha256(f"{session_id}\x00{index}".encode("utf-8")).hexdigest()
    return f"ep-{digest[:16]}"


def _session_id(event: dict, fallback: str) -> str:
    value = event.get("sessionId")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


def _timestamp(event: dict) -> str:
    value = event.get("timestamp")
    return value if isinstance(value, str) else ""


def _fallback_session_id(path: str | os.PathLike[str]) -> str:
    stem = Path(path).stem or "session"
    return f"anon-{hashlib.sha256(stem.encode('utf-8')).hexdigest()[:12]}"


@dataclass
class _EpisodeBuffer:
    session_id: str
    turns: list[Turn] = field(default_factory=list)
    started_at: str = ""
    ended_at: str = ""


# --------------------------------------------------------------------------- #
# Публичный API
# --------------------------------------------------------------------------- #


def iter_episodes(
    path: str | os.PathLike[str],
    stats: SessionStats | None = None,
    scrub_stats: scrub_mod.ScrubStats | None = None,
    source_session: str | None = None,
    start_index: int = 0,
) -> Iterator[Episode]:
    """Потоково отдать эпизоды сессии: сессия не поднимается в память целиком.

    Каждое событие вычищается скрабом (ступень 1) до попадания в ходы.
    """
    fallback = source_session or _fallback_session_id(path)
    buffer: _EpisodeBuffer | None = None
    index = start_index

    for raw_event in iter_jsonl(path, stats):
        session_id = _session_id(raw_event, fallback)
        event = scrub_mod.scrub_inplace(raw_event, scrub_stats)
        timestamp = _timestamp(event)

        if is_user_prompt(event):
            if buffer is not None and buffer.turns:
                yield _flush(buffer, index)
                index += 1
            buffer = _EpisodeBuffer(session_id=session_id, started_at=timestamp, ended_at=timestamp)
            buffer.turns.extend(event_turns(event))
            continue

        if buffer is None:
            continue  # события до первого запроса в эпизод не входят

        turns = event_turns(event)
        if not turns:
            continue
        buffer.turns.extend(turns)
        if timestamp:
            buffer.ended_at = timestamp

    if buffer is not None and buffer.turns:
        yield _flush(buffer, index)


def _flush(buffer: _EpisodeBuffer, index: int) -> Episode:
    return Episode(
        id=_episode_id(buffer.session_id, index),
        source_session=buffer.session_id,
        turns=list(buffer.turns),
        started_at=buffer.started_at,
        ended_at=buffer.ended_at or buffer.started_at,
    )


def load_episodes(
    path: str | os.PathLike[str],
    stats: SessionStats | None = None,
    scrub_stats: scrub_mod.ScrubStats | None = None,
    source_session: str | None = None,
) -> list[Episode]:
    """Список эпизодов файла (удобная обёртка над ``iter_episodes``)."""
    return list(iter_episodes(path, stats, scrub_stats, source_session))


def split_episodes(
    events: Sequence[dict],
    source_session: str | None = None,
    start_index: int = 0,
    scrub_stats: scrub_mod.ScrubStats | None = None,
) -> list[Episode]:
    """Разрезать список событий на эпизоды (для тестов и внешних вызывающих)."""
    fallback = source_session or "session"
    episodes: list[Episode] = []
    buffer: _EpisodeBuffer | None = None
    index = start_index

    for raw_event in events:
        if not isinstance(raw_event, dict):
            continue
        session_id = _session_id(raw_event, fallback)
        event = scrub_mod.scrub_session(raw_event, scrub_stats)
        timestamp = _timestamp(event)

        if is_user_prompt(event):
            if buffer is not None and buffer.turns:
                episodes.append(_flush(buffer, index))
                index += 1
            buffer = _EpisodeBuffer(session_id=session_id, started_at=timestamp, ended_at=timestamp)
            buffer.turns.extend(event_turns(event))
            continue

        if buffer is None:
            continue
        turns = event_turns(event)
        if not turns:
            continue
        buffer.turns.extend(turns)
        if timestamp:
            buffer.ended_at = timestamp

    if buffer is not None and buffer.turns:
        episodes.append(_flush(buffer, index))
    return episodes


def turn_kind_counts(episodes: Iterable[Episode]) -> dict[str, int]:
    counts = {kind: 0 for kind in TURN_KINDS}
    for episode in episodes:
        for turn in episode.turns:
            counts[turn.kind] = counts.get(turn.kind, 0) + 1
    return counts
