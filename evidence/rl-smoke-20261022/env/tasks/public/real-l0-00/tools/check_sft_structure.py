#!/usr/bin/env python3
"""C-044 / SFT-STAGE.delta §8.1 — структурный аудитор SFT-набора: гейт до стадии.

Урок Лагуны (LAG-ADR-042/043): ~14 ч SFT выброшено на структурных дефектах
набора — незакрытых ``<think>``, ``<tool_call>`` внутри ``<think>``, записях без
ответа.  Модель воспроизводит дефект дословно (16.86% / 11.24% / 10.67% у
Лагуны), поэтому аудит выполняется **до** старта стадии, а не постфактум.

Формат входа (§2 спеки; снят с фактического набора
``~/gb10-shared/datasets/sft_train_v12.jsonl``)
--------------------------------------------------------------------------
Каждая строка — объект с полем ``messages``; траектория агента живёт целиком в
**одном** сообщении ``assistant`` и состоит из повторяющихся шагов::

    <think>…</think>
    <tool_call>{"name": …}</tool_call>
    <tool_response>…</tool_response>
    <think>…</think>
    …
    финальный текстовый ответ

Маркеров конца хода (chat-сентинелей) в наборе **нет** — их роль выполняет
структура тегов: естественный конец хода — закрытый ``</think>``/``</tool_call>``/
``</tool_response>``.  Поэтому ход (turn) здесь = одно assistant-сообщение: у
офлайн-набора траектория сгенерирована одним проходом, значит «ход» и
«генерация» совпадают, а бюджет пробы обрезает именно её.

Проверки (два класса хода + дефект эпизода, §8.1)
-------------------------------------------------
* ``unclosed_think`` (ход) — ``<think>`` без парного ``</think>``;
* ``tool_call_in_think`` (ход) — ``<tool_call>`` внутри ``<think>…</think>``;
* ``no_answer`` (эпизод) — **последний** assistant-ход эпизода отсутствует,
  пуст или содержит только блоки ``<think>``/``<tool_call>``/``<tool_response>``
  (после их удаления нет содержательного текста).

**Эпизод и промежуточные ходы.**  Эпизод = одна запись (строка) jsonl.
Промежуточные assistant-ходы (голый ``<tool_call>``, за которым в эпизоде
следует результат ``user``/``tool``) дефектом **не являются** — они выносятся в
отдельное информационное поле ``intermediate_tool_turns`` (count+share) и в
классы дефектов не попадают.

**Границы суждения.**  Дефекты считаются только среди **естественно
завершённых** ходов (LAG-ADR-042/043).  Ход считается усечённым (обрезанным
бюджетом пробы), если после последнего структурного закрывающего тега остался
открывающий ``<think>``/``<tool_call>`` без пары: такой хвост **не** дефект, а
усечение (мутант на пере-подсчёт).  ``no_answer`` оценивается только для
естественно завершённого последнего хода — у усечённого хода ответа нет по
определению.

**Ложные субстроки.**  Строки набора цитируют теги в прозе (например,
пересказывают системную инструкцию).  Чтобы не считать цитату дефектом,
открывающие теги распознаются только в структурной позиции — первыми
непробельными символами строки.  Закрывающие теги ``</tool_call>`` в наборе
идут в той же строке, что и открывающий, и распознаются в любом месте.

**Сатурация** (§8.2.2): при ``truncated_share ≥ 20 %`` прибор сатурирован —
вердикт по формату не выносится (exit 2), а не «чисто».

**Сопутствующий критерий** (§8.2.1): ``unfinished_tool_call`` — открытый
``<tool_call>`` без ``</tool_call>``; учитывается как отдельная метрика стадии
(≤ 5 % на пробе ``n ≥ 100``).

Коды возврата
-------------
* ``0`` — допустимо (дефектов нет; либо дефекты есть, но режим не ``--strict``);
* ``1`` — дефекты в ``--strict``;
* ``2`` — проверить нельзя: нет входа, объявленный путь отсутствует, вход
  нечитаем (fail-closed) или прибор сатурирован.

Нормализация (``--normalize``)
------------------------------
Исходник **не трогается**, результат — новый файл; неизменённые записи
переносятся байт-в-байт.  Журнал обратим: на каждую изменённую строку — список
правок ``{start, end, old, new, class}`` в координатах текста до правки
(``source_sha256`` даёт исходник).  Непочинимое (``no_answer`` — ответа нет
механически; усечение — не дефект) остаётся как есть и помечается ``unfixable``.
Правка на лету запрещена.

Запуск::

    python3 tools/check_sft_structure.py --selftest
    python3 tools/check_sft_structure.py --input sft_train_v12.jsonl --strict --json audit.json
    python3 tools/check_sft_structure.py --input in.jsonl --normalize \
        --out out.jsonl --journal out.journal.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

#: Схема отчёта аудита.
REPORT_SCHEMA = "axiom-sft-structure/3"
#: Схема журнала нормализации.
JOURNAL_SCHEMA = "axiom-sft-normalize/3"

EXIT_OK = 0
EXIT_DEFECT = 1
EXIT_CANNOT = 2

#: Классы дефектов **хода** (§8.1): считаются по сообщениям-assistant.
TURN_CLASSES = ("unclosed_think", "tool_call_in_think")
#: Все три класса дефектов §8.1 (гейт): два ходовых + ``no_answer`` (эпизод).
CLASSES = ("unclosed_think", "tool_call_in_think", "no_answer")
#: Порог сатурации (§8.2.2): доля усечённых ходов, выше — вердикт не выносится.
SATURATION_SHARE = 0.20
#: Порог незавершённых вызовов инструмента (§8.2.1): ≤ 5 % на пробе n ≥ 100.
UNFINISHED_TOOL_CALL_MAX = 0.05
UNFINISHED_MIN_PROBE = 100
#: Сколько примеров на класс кладётся в отчёт.
MAX_EXAMPLES = 5
SNIPPET_LEN = 240

#: Открывающие теги — только в структурной позиции (первый непробельный символ
#: строки); цитаты тегов внутри прозы так не считаются дефектом.
_OPEN_RE = re.compile(r"(?m)^[ \t]*(<think>|<tool_call>|<tool_response>)")
#: Закрывающие теги — в любой позиции (``</tool_call>`` стоит в строке открытия).
_CLOSE_RE = re.compile(r"(</think>|</tool_call>|</tool_response>)")
#: Закрытые блоки (для проверки «есть ли ответ»).
_BLOCK_RE = re.compile(
    r"<think>.*?</think>|<tool_call>.*?</tool_call>|<tool_response>.*?</tool_response>",
    re.S,
)
_THINK_PAIR_RE = re.compile(r"<think>.*?</think>", re.S)
_TOOL_CALL_PAIR_RE = re.compile(r"<tool_call>.*?</tool_call>", re.S)

#: Максимум итераций механической починки одной записи (защита от цикла).
_MAX_FIX_STEPS = 10_000


class InputError(Exception):
    """Вход существует, но нечитаем/непригоден: fail-closed (exit 2)."""


# --------------------------------------------------------------------------- #
# Структурный разбор
# --------------------------------------------------------------------------- #


def structural_tokens(text: str) -> list[tuple[int, str, bool]]:
    """Структурные теги как ``(позиция, тег, открывающий?)`` в порядке текста."""
    tokens = [(m.start(1), m.group(1), True) for m in _OPEN_RE.finditer(text)]
    tokens += [(m.start(1), m.group(1), False) for m in _CLOSE_RE.finditer(text)]
    tokens.sort(key=lambda item: (item[0], not item[2]))
    return tokens


def _next_token_after(tokens: list[tuple[int, str, bool]], position: int) -> int | None:
    for pos, _tag, _is_open in tokens:
        if pos > position:
            return pos
    return None


def analyze_content(text: str) -> dict[str, Any]:
    """Классы дефектов одного assistant-сообщения (хода).

    Возвращает счётчики классов, признак усечения и позиции незакрытых
    ``<think>`` (в координатах ``text``), по которым работает нормализация.
    """
    tokens = structural_tokens(text)
    think_stack: list[int] = []
    call_stack: list[int] = []
    tool_call_in_think = 0
    last_close = -1
    for position, tag, is_open in tokens:
        if is_open:
            if tag == "<think>":
                think_stack.append(position)
            elif tag == "<tool_call>":
                if think_stack:
                    tool_call_in_think += 1
                call_stack.append(position)
        else:
            last_close = position
            if tag == "</think>":
                if think_stack:
                    think_stack.pop()
            elif tag == "</tool_call>":
                if call_stack:
                    call_stack.pop()

    trailing_opens = [tag for pos, tag, is_open in tokens if pos > last_close and is_open]
    truncated = any(tag in ("<think>", "<tool_call>") for tag in trailing_opens)
    if truncated:
        # Хвостовые открытые теги — усечение бюджетом, а не структурный дефект.
        think_stack = [pos for pos in think_stack if pos <= last_close]
        call_stack = [pos for pos in call_stack if pos <= last_close]

    return {
        "unclosed_think": len(think_stack),
        "tool_call_in_think": tool_call_in_think,
        "unfinished_tool_call": len(call_stack),
        "truncated": truncated,
        "unclosed_positions": sorted(think_stack),
    }


def defect_counts(verdict: dict[str, Any]) -> dict[str, int]:
    """Счётчики классов **хода** из вердикта (bool → 0/1)."""
    return {name: int(verdict[name]) for name in TURN_CLASSES}


# --------------------------------------------------------------------------- #
# Эпизод: дефект ответа и промежуточные ходы
# --------------------------------------------------------------------------- #

#: Роли, завершающие промежуточный ход агента (результат инструмента/вопрос).
_RESULT_ROLES = ("user", "tool")


def _substantive_text(text: str) -> str:
    """Текст после удаления блоков ``<think>``/``<tool_call>``/``<tool_response>``."""
    return _BLOCK_RE.sub("", text).strip()


def _is_bare_tool_turn(text: str) -> bool:
    """Ход-«голый вызов»: есть ``<tool_call>`` и нет содержательного текста."""
    return bool(_TOOL_CALL_PAIR_RE.search(text)) and _substantive_text(text) == ""


def _record_messages(record: Any) -> list[Any]:
    """Сообщения записи; fallback на ``content``/``text`` как один ход."""
    if not isinstance(record, dict):
        raise InputError("запись не является объектом JSON")
    messages = record.get("messages")
    if isinstance(messages, list) and messages:
        return messages
    for key in ("content", "text"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return [{"role": "assistant", "content": value}]
    raise InputError("запись без messages и без content/text — формат не распознан")


def _message_parts(message: Any) -> tuple[str | None, str | None]:
    """``(роль, текст)`` сообщения, если оно объект со строковым ``content``."""
    if not isinstance(message, dict):
        return None, None
    role = message.get("role")
    content = message.get("content")
    return (
        role if isinstance(role, str) else None,
        content if isinstance(content, str) else None,
    )


def analyze_episode(record: Any) -> dict[str, Any]:
    """Классы дефектов **эпизода** (записи) по §8.1.

    Ходовые классы (``unclosed_think``/``tool_call_in_think``) агрегируются по
    assistant-сообщениям; ``no_answer`` — дефект эпизода: последний assistant-ход
    отсутствует, пуст или содержит только блоки.  Промежуточные «голые» вызовы
    (за которыми в эпизоде следует результат ``user``/``tool``) дефектом не
    считаются — их число идёт в ``intermediate_tool_turns``.
    """
    messages = _record_messages(record)
    roles = [_message_parts(message) for message in messages]
    assistant_indices = [
        index
        for index, (role, content) in enumerate(roles)
        if role == "assistant" and content is not None
    ]

    turn_counts = {name: 0 for name in TURN_CLASSES}
    turn_texts: dict[str, str | None] = {name: None for name in TURN_CLASSES}
    unfinished = completed = truncated = 0
    for index in assistant_indices:
        text = roles[index][1] or ""
        verdict = analyze_content(text)
        if verdict["truncated"]:
            truncated += 1
            continue
        completed += 1
        for name in TURN_CLASSES:
            if verdict[name]:
                turn_counts[name] += verdict[name]
                if turn_texts[name] is None:
                    turn_texts[name] = text
        unfinished += verdict["unfinished_tool_call"]

    intermediate = 0
    for order, index in enumerate(assistant_indices):
        if order == len(assistant_indices) - 1:
            break  # последний assistant-ход не «промежуточный»
        text = roles[index][1] or ""
        if not _is_bare_tool_turn(text):
            continue
        next_role = roles[index + 1][0] if index + 1 < len(messages) else None
        if next_role in _RESULT_ROLES:
            intermediate += 1

    if assistant_indices:
        last_text = roles[assistant_indices[-1]][1] or ""
        episode_completed = not analyze_content(last_text)["truncated"]
        no_answer = episode_completed and _substantive_text(last_text) == ""
    else:
        last_text = ""
        episode_completed = False
        no_answer = True  # последний assistant-ход эпизода отсутствует

    return {
        "assistant_turns": len(assistant_indices),
        "completed_turns": completed,
        "truncated_turns": truncated,
        "turn_counts": turn_counts,
        "turn_texts": turn_texts,
        "unfinished_tool_call": unfinished,
        "intermediate_tool_turns": intermediate,
        "no_answer": no_answer,
        "last_text": last_text,
        "episode_completed": episode_completed,
    }


# --------------------------------------------------------------------------- #
# Механическая починка (нормализация новым файлом)
# --------------------------------------------------------------------------- #


class _Editor:
    """Последовательные правки текста с журналом, обратимым в обратном порядке.

    Каждая правка записывается в координатах состояния **до** неё; откат идёт от
    последней правки к первой, поэтому координаты всегда актуальны.
    """

    def __init__(self, text: str) -> None:
        self.text = text
        self.edits: list[dict[str, Any]] = []

    def insert(self, position: int, insertion: str, defect_class: str) -> None:
        self.text = self.text[:position] + insertion + self.text[position:]
        self.edits.append(
            {"start": position, "end": position, "old": "", "new": insertion,
             "class": defect_class}
        )

    def replace(self, start: int, end: int, replacement: str, defect_class: str) -> None:
        old = self.text[start:end]
        if old == replacement:
            return
        self.text = self.text[:start] + replacement + self.text[end:]
        self.edits.append(
            {"start": start, "end": end, "old": old, "new": replacement,
             "class": defect_class}
        )


def _is_anchored(text: str, position: int) -> bool:
    """Открывающий тег в начале строки (первый непробельный символ) или в позиции 0."""
    line_start = text.rfind("\n", 0, position) + 1
    return text[line_start:position].strip(" \t") == ""


def _first_call_in_think(text: str) -> tuple[int, int, int] | None:
    """Первая пара ``<tool_call>…</tool_call>`` внутри ``<think>…</think>``.

    Возвращает ``(начало_пары, конец_пары, позиция_за_закрывающим </think>)``.
    """
    for think in _THINK_PAIR_RE.finditer(text):
        for call in _TOOL_CALL_PAIR_RE.finditer(think.group(0)):
            if not _is_anchored(text, think.start() + call.start()):
                continue
            return (
                think.start() + call.start(),
                think.start() + call.end(),
                think.end(),
            )
    return None


def normalize_content(
    text: str,
) -> tuple[str, list[dict[str, Any]], dict[str, int], dict[str, int], bool]:
    """Нормализовать одно assistant-сообщение (ход).

    Возвращает ``(новый_текст, правки, найденные_классы, непочиненные_классы,
    усечён)``.  Классы здесь — ходовые (``TURN_CLASSES``); ``no_answer`` —
    дефект эпизода, механически непочиним (ответа нет — не выдумывать) и
    помечается в ``_normalize_record``.  Усечённый хвост не трогается.
    """
    initial = analyze_content(text)
    present = defect_counts(initial)
    if initial["unfinished_tool_call"]:
        present["unfinished_tool_call"] = initial["unfinished_tool_call"]
    editor = _Editor(text)

    for _ in range(_MAX_FIX_STEPS):
        state = analyze_content(editor.text)
        if state["truncated"]:
            break
        if state["unclosed_positions"]:
            position = state["unclosed_positions"][0]
            target = _next_token_after(structural_tokens(editor.text), position)
            if target is None:
                break
            editor.insert(target, "</think>", "unclosed_think")
            continue
        if state["tool_call_in_think"] > 0:
            found = _first_call_in_think(editor.text)
            if found is None:
                break
            start, end, think_end = found
            call_text = editor.text[start:end]
            # Сначала вставка за ``</think>`` (координата справа), затем удаление
            # пары (координата слева) — так обе позиции верны в момент правки.
            editor.insert(think_end, call_text, "tool_call_in_think")
            editor.replace(start, end, "", "tool_call_in_think")
            continue
        break

    final = analyze_content(editor.text)
    remaining = defect_counts(final)
    unfixable = {name: count for name, count in remaining.items() if count}
    if final["unfinished_tool_call"]:
        unfixable["unfinished_tool_call"] = final["unfinished_tool_call"]
    return editor.text, editor.edits, present, unfixable, initial["truncated"]


# --------------------------------------------------------------------------- #
# Ввод jsonl
# --------------------------------------------------------------------------- #


def _iter_jsonl(path: Path) -> Iterator[tuple[int, bytes, Any]]:
    """Строки jsonl: ``(номер, сырые_байты, разобранная_запись)``; сбой — fail-closed."""
    try:
        with path.open("rb") as handle:
            for lineno, raw in enumerate(handle, start=1):
                if not raw.strip():
                    continue
                try:
                    record = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise InputError(
                        f"{path}:{lineno}: строка не разбирается как JSON: {exc}"
                    ) from exc
                yield lineno, raw, record
    except OSError as exc:
        raise InputError(
            f"{path}: файл не читается ({type(exc).__name__}): {exc}"
        ) from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snippet(text: str) -> str:
    snippet = " ".join(text.strip().split())
    return snippet[:SNIPPET_LEN] + ("…" if len(snippet) > SNIPPET_LEN else "")


# --------------------------------------------------------------------------- #
# Аудит
# --------------------------------------------------------------------------- #


def run_check(
    input_paths: Iterable[str | Path],
    *,
    strict: bool = False,
    limit: int = 0,
) -> tuple[int, dict[str, Any]]:
    """Структурный аудит набора.  Возвращает ``(код, отчёт)``."""
    paths = [Path(p) for p in input_paths]
    if not paths:
        return EXIT_CANNOT, _cannot("не переданы входные файлы — аудировать нечего", paths)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        return EXIT_CANNOT, _cannot(
            "объявленные пути отсутствуют: " + ", ".join(missing), paths
        )

    counts: dict[str, int] = {name: 0 for name in CLASSES}
    affected: dict[str, int] = {name: 0 for name in CLASSES}
    unfinished = 0
    intermediate = 0
    records = 0
    assistant_messages = 0
    truncated = 0
    episodes_completed = 0
    per_file: list[dict[str, Any]] = []
    examples: dict[str, list[dict[str, Any]]] = {name: [] for name in CLASSES}

    def note(name: str, path: Path, lineno: int, text: str, count: int) -> None:
        counts[name] += count
        affected[name] += 1
        if len(examples[name]) < MAX_EXAMPLES:
            examples[name].append(
                {"file": str(path), "line": lineno, "snippet": _snippet(text)}
            )

    try:
        for path in paths:
            file_counts = {name: 0 for name in CLASSES}
            file_records = file_completed = file_truncated = file_intermediate = 0
            stop = False
            for lineno, _raw, record in _iter_jsonl(path):
                if limit and records >= limit:
                    stop = True
                    break
                records += 1
                file_records += 1
                episode = analyze_episode(record)
                assistant_messages += episode["assistant_turns"]
                file_completed += episode["completed_turns"]
                file_truncated += episode["truncated_turns"]
                truncated += episode["truncated_turns"]
                unfinished += episode["unfinished_tool_call"]
                intermediate += episode["intermediate_tool_turns"]
                file_intermediate += episode["intermediate_tool_turns"]
                if episode["episode_completed"]:
                    episodes_completed += 1
                for name in TURN_CLASSES:
                    count = episode["turn_counts"][name]
                    if count:
                        file_counts[name] += count
                        note(name, path, lineno, episode["turn_texts"][name] or "", count)
                if episode["no_answer"]:
                    file_counts["no_answer"] += 1
                    note("no_answer", path, lineno, episode["last_text"], 1)
            per_file.append(
                {
                    "path": str(path),
                    "records": file_records,
                    "completed_turns": file_completed,
                    "truncated_turns": file_truncated,
                    "intermediate_tool_turns": file_intermediate,
                    "classes": dict(file_counts),
                }
            )
            if stop:
                break
    except InputError as exc:
        return EXIT_CANNOT, _cannot(f"fail-closed: {exc}", paths)

    completed = assistant_messages - truncated
    truncated_share = (truncated / assistant_messages) if assistant_messages else 0.0
    classes = {
        name: {
            "count": counts[name],
            "affected_turns": affected[name],
            "share": (
                round(counts[name] / episodes_completed, 6)
                if name == "no_answer" and episodes_completed
                else round(counts[name] / completed, 6) if completed else 0.0
            ),
        }
        for name in CLASSES
    }
    intermediate_share = (intermediate / completed) if completed else 0.0
    unfinished_share = (unfinished / completed) if completed else 0.0
    gate_unfinished = (
        completed >= UNFINISHED_MIN_PROBE and unfinished_share > UNFINISHED_TOOL_CALL_MAX
    )
    defects_found = any(counts[name] > 0 for name in CLASSES) or gate_unfinished
    saturated = assistant_messages > 0 and truncated_share >= SATURATION_SHARE

    if saturated:
        code, verdict = EXIT_CANNOT, "saturated"
    elif defects_found and strict:
        code, verdict = EXIT_DEFECT, "defects"
    else:
        code = EXIT_OK
        verdict = "defects" if defects_found else "admissible"

    return code, {
        "schema": REPORT_SCHEMA,
        "verdict": verdict,
        "input_files": [str(path) for path in paths],
        "records": records,
        "episodes_completed": episodes_completed,
        "assistant_messages": assistant_messages,
        "completed_turns": completed,
        "truncated_turns": truncated,
        "truncated_share": round(truncated_share, 6),
        "saturated": saturated,
        "strict": strict,
        "classes": classes,
        "intermediate_tool_turns": {
            "count": intermediate,
            "share": round(intermediate_share, 6),
        },
        "unfinished_tool_call": {
            "count": unfinished,
            "share": round(unfinished_share, 6),
            "threshold": UNFINISHED_TOOL_CALL_MAX,
            "min_probe": UNFINISHED_MIN_PROBE,
            "gated": gate_unfinished,
        },
        "defects_found": defects_found,
        "examples": examples,
        "files": per_file,
    }


def _cannot(reason: str, paths: list[Path]) -> dict[str, Any]:
    return {
        "schema": REPORT_SCHEMA,
        "verdict": "cannot-check",
        "reason": reason,
        "input_files": [str(path) for path in paths],
        "records": 0,
        "episodes_completed": 0,
        "assistant_messages": 0,
        "completed_turns": 0,
        "truncated_turns": 0,
        "truncated_share": 0.0,
        "saturated": False,
        "classes": {name: {"count": 0, "affected_turns": 0, "share": 0.0} for name in CLASSES},
        "intermediate_tool_turns": {"count": 0, "share": 0.0},
        "unfinished_tool_call": {
            "count": 0,
            "share": 0.0,
            "threshold": UNFINISHED_TOOL_CALL_MAX,
            "min_probe": UNFINISHED_MIN_PROBE,
            "gated": False,
        },
        "defects_found": False,
        "examples": {name: [] for name in CLASSES},
        "files": [],
    }


# --------------------------------------------------------------------------- #
# Нормализация
# --------------------------------------------------------------------------- #


def _normalize_record(
    record: Any,
) -> tuple[Any, bool, dict[str, Any]]:
    """Нормализовать запись; вернуть ``(запись, изменена?, след)``."""
    if not isinstance(record, dict) or not isinstance(record.get("messages"), list):
        raise InputError("запись без messages — нормализация невозможна")
    present: dict[str, int] = {name: 0 for name in CLASSES}
    unfixable: dict[str, int] = {name: 0 for name in CLASSES}
    messages: list[Any] = []
    changed = False
    message_journal: list[dict[str, Any]] = []
    for index, message in enumerate(record["messages"]):
        if (
            isinstance(message, dict)
            and message.get("role") == "assistant"
            and isinstance(message.get("content"), str)
        ):
            new_text, edits, found, bad, truncated = normalize_content(message["content"])
            for name in TURN_CLASSES:
                present[name] += found[name]
                unfixable[name] += bad.get(name, 0)
            if found.get("unfinished_tool_call"):
                present["unfinished_tool_call"] = present.get("unfinished_tool_call", 0) + found["unfinished_tool_call"]
            if bad.get("unfinished_tool_call"):
                unfixable["unfinished_tool_call"] = unfixable.get("unfinished_tool_call", 0) + bad["unfinished_tool_call"]
            if new_text != message["content"]:
                changed = True
                message_journal.append(
                    {
                        "message_index": index,
                        "before_sha256": hashlib.sha256(
                            message["content"].encode("utf-8")
                        ).hexdigest(),
                        "after_sha256": hashlib.sha256(new_text.encode("utf-8")).hexdigest(),
                        "edits": edits,
                    }
                )
                message = {**message, "content": new_text}
        messages.append(message)
    # ``no_answer`` — дефект эпизода: ответ не выдумывается механически.
    if analyze_episode(record)["no_answer"]:
        present["no_answer"] += 1
        unfixable["no_answer"] += 1
    return {**record, "messages": messages}, changed, {
        "classes": {name: count for name, count in present.items() if count},
        "unfixable": {name: count for name, count in unfixable.items() if count},
        "messages": message_journal,
    }


def run_normalize(
    input_path: str | Path,
    out_path: str | Path,
    journal_path: str | Path | None = None,
    *,
    limit: int = 0,
) -> tuple[int, dict[str, Any]]:
    """Механическая нормализация **новым файлом**: исходник не трогается."""
    source = Path(input_path)
    out = Path(out_path)
    journal_out = Path(journal_path) if journal_path else Path(str(out) + ".journal.json")
    if not source.exists():
        return EXIT_CANNOT, _cannot(f"объявленный путь отсутствует: {source}", [source])

    source_sha_before = _sha256(source)
    transformations: list[dict[str, Any]] = []
    records = changed = 0
    try:
        raw_lines = source.read_bytes().splitlines(keepends=True)
    except OSError as exc:
        return EXIT_CANNOT, _cannot(
            f"fail-closed: {source}: файл не читается ({type(exc).__name__}): {exc}",
            [source],
        )

    out_lines: list[bytes] = []
    try:
        for lineno, raw in enumerate(raw_lines, start=1):
            if not raw.strip():
                out_lines.append(raw)
                continue
            if limit and records >= limit:
                out_lines.append(raw)
                continue
            records += 1
            try:
                record = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise InputError(
                    f"{source}:{lineno}: строка не разбирается как JSON: {exc}"
                ) from exc
            new_record, record_changed, trail = _normalize_record(record)
            if not record_changed:
                out_lines.append(raw)
                if trail["classes"] or trail["unfixable"]:
                    # Дефект есть, но механически не чинится (нет ответа/усечение):
                    # строка переносится байт-в-байт, след остаётся в журнале.
                    transformations.append(
                        {"line": lineno, "changed": False, "classes": trail["classes"],
                         "unfixable": trail["unfixable"], "messages": []}
                    )
                continue
            changed += 1
            out_lines.append(
                (json.dumps(new_record, ensure_ascii=False) + "\n").encode("utf-8")
            )
            transformations.append(
                {
                    "line": lineno,
                    "changed": True,
                    "classes": trail["classes"],
                    "unfixable": trail["unfixable"],
                    "messages": trail["messages"],
                }
            )
    except InputError as exc:
        return EXIT_CANNOT, _cannot(f"fail-closed: {exc}", [source])

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"".join(out_lines))
    source_sha_after = _sha256(source)

    journal = {
        "schema": JOURNAL_SCHEMA,
        "source": str(source),
        "source_sha256": source_sha_before,
        "output": str(out),
        "output_sha256": _sha256(out),
        "source_unchanged": source_sha_before == source_sha_after,
        "records_total": records,
        "records_changed": changed,
        "reversal": (
            "к строке исходника применить edits в обратном порядке, заменяя new на old"
        ),
        "transformations": transformations,
    }
    journal_out.parent.mkdir(parents=True, exist_ok=True)
    journal_out.write_text(
        json.dumps(journal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    journal["journal"] = str(journal_out)
    return EXIT_OK, journal


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _emit(report: dict[str, Any], json_path: str | None, quiet: bool) -> None:
    if json_path:
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(json_path).write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if not quiet:
        print(json.dumps(report, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="C-044/SFT-STAGE §8.1: структурный аудит набора"
    )
    parser.add_argument("--input", dest="input_paths", action="append", default=[],
                        help="jsonl-набор; повторяется")
    parser.add_argument("--out", dest="out_path", default=None,
                        help="куда писать нормализованный набор (--normalize)")
    parser.add_argument("--journal", dest="journal_path", default=None,
                        help="куда писать журнал нормализации (по умолчанию рядом с --out)")
    parser.add_argument("--strict", action="store_true", help="дефекты → exit 1")
    parser.add_argument("--normalize", action="store_true",
                        help="механическая нормализация новым файлом")
    parser.add_argument("--limit", type=int, default=0, help="пробовать только первые N записей")
    parser.add_argument("--json", dest="json_path", default=None, help="куда записать отчёт")
    parser.add_argument("--quiet", action="store_true", help="не печатать отчёт в stdout")
    parser.add_argument("--selftest", action="store_true",
                        help="синтетический selftest с мутантами (tmp; реальные данные не трогаются)")
    args = parser.parse_args(argv)

    if args.selftest:
        return run_selftest()

    if args.normalize:
        if not args.input_paths:
            report = _cannot("не передан --input для нормализации", [])
            report["exit_code"] = EXIT_CANNOT
            _emit(report, args.json_path, args.quiet)
            return EXIT_CANNOT
        source = Path(args.input_paths[0])
        out = Path(args.out_path) if args.out_path else source.with_name(
            source.stem + ".normalized.jsonl"
        )
        code, report = run_normalize(source, out, args.journal_path, limit=args.limit)
        report["exit_code"] = code
        _emit(report, args.json_path, args.quiet)
        return code

    code, report = run_check(args.input_paths, strict=args.strict, limit=args.limit)
    report["exit_code"] = code
    _emit(report, args.json_path, args.quiet)
    return code


# --------------------------------------------------------------------------- #
# Selftest (C-044: мутанты краснеют, чистая пара зелёная, усечение не дефект)
# --------------------------------------------------------------------------- #

_CLEAN = (
    "<think>\nкраткое рассуждение\n</think>\n"
    '<tool_call>{"name": "search_concepts", "query": "x"}</tool_call>\n'
    "<tool_response>\nslug: x\ntype: t\n</tool_response>\n"
    "Финальный ответ по концептам."
)
_UNCLOSED = (
    "<think>\nпервое рассуждение без закрытия\n"
    "<think>\nвторое рассуждение\n</think>\n"
    "Финальный ответ."
)
_TOOL_IN_THINK = (
    "<think>\nрассуждение\n"
    '<tool_call>{"name": "search_concepts", "query": "y"}</tool_call>\n'
    "продолжение рассуждения\n</think>\n"
    "<tool_response>\nslug: y\n</tool_response>\n"
    "Финальный ответ."
)
_NO_ANSWER = (
    "<think>\nрассуждение\n</think>\n"
    '<tool_call>{"name": "search_concepts", "query": "z"}</tool_call>\n'
    "<tool_response>\nslug: z\n</tool_response>"
)
_TRUNCATED = "<think>\nрассуждение\n</think>\nначало ответа\n<think>\nоборвано бюджетом"

#: Многошаговый эпизод: «голый» вызов (без текста ответа) + результат среды.
_CLEAN_ANSWER = "<think>\nрассуждение\n</think>\nФинальный ответ по концептам."
_BARE_CALL = (
    "<think>\nрассуждение\n</think>\n"
    '<tool_call>{"name": "search_concepts", "query": "q"}</tool_call>'
)
_TOOL_RESULT = "<tool_response>\nslug: q\ntype: t\n</tool_response>"


def _episode_messages(*, pauses: int, final: bool) -> list[dict[str, str]]:
    """Эпизод с ``pauses`` промежуточными вызовами (и, если ``final``, ответом)."""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": "системная инструкция"},
        {"role": "user", "content": "вопрос"},
    ]
    for _ in range(pauses):
        messages.append({"role": "assistant", "content": _BARE_CALL})
        messages.append({"role": "user", "content": _TOOL_RESULT})
    messages.append({"role": "assistant", "content": _CLEAN_ANSWER if final else _BARE_CALL})
    return messages


def _write_episodes(path: Path, episodes: list[list[dict[str, str]]]) -> None:
    lines = [
        json.dumps({"messages": episode, "task_type": "explain_relation"},
                   ensure_ascii=False) + "\n"
        for episode in episodes
    ]
    path.write_text("".join(lines), encoding="utf-8")


def _write_records(path: Path, assistants: list[str]) -> None:
    lines = []
    for text in assistants:
        record = {
            "messages": [
                {"role": "system", "content": "системная инструкция с <tool_call> в прозе"},
                {"role": "user", "content": "вопрос"},
                {"role": "assistant", "content": text},
            ],
            "task_type": "explain_relation",
        }
        lines.append(json.dumps(record, ensure_ascii=False) + "\n")
    path.write_text("".join(lines), encoding="utf-8")


def _replay_edits(source_text: str, edits: list[dict[str, Any]]) -> str:
    """Откатить правки журнала: из текста-результата получить исходный.

    Правка ``{start, end, old, new}`` заменяет ``[start, end)`` на ``new`` в
    состоянии **до** неё; откат идёт от последней правки к первой, поэтому
    вставленный фрагмент ``new`` лежит в ``[start, start+len(new))``.
    """
    text = source_text
    for edit in reversed(edits):
        start = edit["start"]
        new = edit["new"]
        assert text[start:start + len(new)] == new, (
            f"журнал необратим: ожидалось {new!r}, найдено {text[start:start + len(new)]!r}"
        )
        text = text[:start] + edit["old"] + text[start + len(new):]
    return text


def run_selftest() -> int:
    """Синтетика в ``tmp``: чистая зелёная, мутанты классов и усечения краснеют."""
    import tempfile

    checks: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory(prefix="sft-structure-selftest-") as tmp:
        root = Path(tmp)

        # Чистый набор — зелёный.
        clean = root / "clean.jsonl"
        _write_records(clean, [_CLEAN for _ in range(5)])
        code_clean, report_clean = run_check([clean], strict=True)
        checks.append(("чистый набор → exit 0", code_clean == EXIT_OK))
        checks.append(
            ("чистый набор → verdict admissible, дефектов нет",
             report_clean["verdict"] == "admissible" and not report_clean["defects_found"])
        )

        # Мутант класса 1: незакрытый <think> в завершённом ходу.
        unclosed = root / "unclosed.jsonl"
        _write_records(unclosed, [_CLEAN for _ in range(4)] + [_UNCLOSED])
        code_u, report_u = run_check([unclosed], strict=True)
        checks.append(("мутант «незакрытый think» → exit 1 (strict)", code_u == EXIT_DEFECT))
        checks.append(
            ("мутант «незакрытый think» → класс unclosed_think > 0",
             report_u["classes"]["unclosed_think"]["count"] > 0)
        )
        checks.append(
            ("мутант «незакрытый think» → других классов нет",
             report_u["classes"]["tool_call_in_think"]["count"] == 0
             and report_u["classes"]["no_answer"]["count"] == 0)
        )

        # Мутант класса 2: <tool_call> внутри <think>.
        in_think = root / "in_think.jsonl"
        _write_records(in_think, [_CLEAN for _ in range(4)] + [_TOOL_IN_THINK])
        code_t, report_t = run_check([in_think], strict=True)
        checks.append(("мутант «tool_call в think» → exit 1 (strict)", code_t == EXIT_DEFECT))
        checks.append(
            ("мутант «tool_call в think» → класс tool_call_in_think > 0",
             report_t["classes"]["tool_call_in_think"]["count"] > 0)
        )

        # Дефект эпизода: последний ход без содержательного текста (только блоки).
        no_answer = root / "no_answer.jsonl"
        _write_records(no_answer, [_CLEAN for _ in range(4)] + [_NO_ANSWER])
        code_n, report_n = run_check([no_answer], strict=True)
        checks.append(("эпизод без ответа → exit 1 (strict)", code_n == EXIT_DEFECT))
        checks.append(
            ("эпизод без ответа → класс no_answer > 0",
             report_n["classes"]["no_answer"]["count"] > 0)
        )

        # Многошаговый эпизод: промежуточные голые вызовы — НЕ дефект, в отдельном поле.
        multi = root / "multi_pause.jsonl"
        _write_episodes(multi, [_episode_messages(pauses=3, final=True)])
        code_m, report_m = run_check([multi], strict=True)
        checks.append(
            ("многошаговый эпизод → exit 0 (промежуточные вызовы не дефект)",
             code_m == EXIT_OK and report_m["verdict"] == "admissible")
        )
        checks.append(
            ("многошаговый эпизод → intermediate_tool_turns == 3, no_answer == 0",
             report_m["intermediate_tool_turns"]["count"] == 3
             and report_m["classes"]["no_answer"]["count"] == 0)
        )

        # Эпизод, оборвавшийся на голом вызове → дефект эпизода (нет ответа).
        cut = root / "ended_in_call.jsonl"
        _write_episodes(cut, [_episode_messages(pauses=2, final=False)])
        code_c, report_c = run_check([cut], strict=True)
        checks.append(("эпизод закончился голым tool_call → exit 1 (strict)", code_c == EXIT_DEFECT))
        checks.append(
            ("эпизод закончился голым tool_call → no_answer == 1, intermediate == 2",
             report_c["classes"]["no_answer"]["count"] == 1
             and report_c["intermediate_tool_turns"]["count"] == 2)
        )

        # Мутант усечения: оборванный бюджетом ход НЕ считается незакрытым think.
        trunc = root / "truncated.jsonl"
        _write_records(trunc, [_CLEAN for _ in range(10)] + [_TRUNCATED])
        code_tr, report_tr = run_check([trunc], strict=True)
        checks.append(
            ("усечённый ход → не незакрытый think, exit 0",
             code_tr == EXIT_OK and report_tr["classes"]["unclosed_think"]["count"] == 0)
        )
        checks.append(("усечённый ход → truncated_turns == 1", report_tr["truncated_turns"] == 1))

        # Сатурация: доля усечённых ≥ 20 % → вердикт по формату не выносится (exit 2).
        saturated = root / "saturated.jsonl"
        _write_records(saturated, [_CLEAN] + [_TRUNCATED] * 3)
        code_s, report_s = run_check([saturated], strict=True)
        checks.append(
            ("сатурация (≥ 20 % усечённых) → exit 2",
             code_s == EXIT_CANNOT and report_s["saturated"] is True)
        )

        # Нет входа / отсутствующий путь → exit 2.
        checks.append(("нет входных файлов → exit 2", run_check([])[0] == EXIT_CANNOT))
        checks.append(
            ("отсутствующий путь → exit 2",
             run_check([root / "нет-такого.jsonl"])[0] == EXIT_CANNOT)
        )

        # Нечитаемый вход → exit 2 (fail-closed).
        broken = root / "broken.jsonl"
        broken.write_bytes(b"\xff\xfe not json\n")
        checks.append(("нечитаемый вход → exit 2 (fail-closed)", run_check([broken])[0] == EXIT_CANNOT))

        # Дефекты без --strict → exit 0 (вердикт «defects», не красный).
        code_lax, report_lax = run_check([unclosed], strict=False)
        checks.append(
            ("дефекты без --strict → exit 0, verdict defects",
             code_lax == EXIT_OK and report_lax["verdict"] == "defects")
        )

        # Отчёт несёт примеры записей.
        checks.append(
            ("отчёт несёт примеры записей",
             bool(report_u["examples"]["unclosed_think"])
             and report_u["examples"]["unclosed_think"][0]["line"] == 5)
        )

        # Нормализация: исходник не тронут, новый файл, неизменённые байт-в-байт.
        src = root / "to_normalize.jsonl"
        _write_records(src, [_CLEAN, _UNCLOSED, _TOOL_IN_THINK, _NO_ANSWER])
        src_before = src.read_bytes()
        out = root / "normalized.jsonl"
        journal_path = root / "normalized.journal.json"
        code_norm, journal = run_normalize(src, out, journal_path)
        src_lines = src_before.splitlines(keepends=True)
        out_lines = out.read_bytes().splitlines(keepends=True)
        checks.append(("нормализация → exit 0", code_norm == EXIT_OK))
        checks.append(("нормализация → исходник не изменён", src.read_bytes() == src_before))
        checks.append(("нормализация → журнал: source_unchanged", journal["source_unchanged"] is True))
        checks.append(("нормализация → новый файл существует", out.exists()))
        checks.append(("нормализация → неизменённая запись байт-в-байт", out_lines[0] == src_lines[0]))
        checks.append(
            ("нормализация меняет только дефектные записи",
             2 <= journal["records_changed"] <= 3)
        )
        # Журнал обратим: правки, применённые в обратном порядке, дают исходник.
        reversible = True
        for transformation in journal["transformations"]:
            line = transformation["line"]
            original_record = json.loads(src_lines[line - 1].decode("utf-8"))
            normalized_record = json.loads(out_lines[line - 1].decode("utf-8"))
            for message in transformation["messages"]:
                index = message["message_index"]
                normalized_text = normalized_record["messages"][index]["content"]
                if _replay_edits(normalized_text, message["edits"]) != original_record["messages"][index]["content"]:
                    reversible = False
        checks.append(("нормализация → журнал обратим (что и на что заменено)", reversible))
        # Незакрытый think починен.
        norm_fix = run_check([out], strict=True)
        checks.append(
            ("нормализация → дефектные классы сняты (остался только no_answer)",
             norm_fix[1]["classes"]["unclosed_think"]["count"] == 0
             and norm_fix[1]["classes"]["tool_call_in_think"]["count"] == 0)
        )
        # no_answer не чинится механически — помечен unfixable.
        checks.append(
            ("нормализация → no_answer помечен непочинимым",
             any(t["unfixable"].get("no_answer") for t in journal["transformations"]))
        )

    ok = all(passed for _, passed in checks)
    for label, passed in checks:
        print(f"[selftest] {'PASS' if passed else 'FAIL'}: {label}")
    print(
        f"[selftest] {'PASS' if ok else 'FAIL'}: прибор ловит два класса хода и "
        "дефект эпизода, не путает усечение с дефектом, журнал обратим"
    )
    return EXIT_OK if ok else EXIT_DEFECT


if __name__ == "__main__":
    raise SystemExit(main())
