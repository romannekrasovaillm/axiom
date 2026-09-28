"""Ступень 3 пайплайна axiom-domain-ds-v1: механическая верификация исхода.

Вердикт берётся не «по смыслу ответа», а из машинного артефакта (AD-2:
вердикт механический, не модельный). Два источника:

1. **Завершающий JSON-контракт сессии** — последний текстовый ход ассистента,
   который оканчивается JSON-объектом со строковым полем ``status`` (после него
   допустимы только пробелы и закрывающая ``` фенса). Контракт в середине
   ответа завершающим не считается.
2. **Парный harness-отчёт** (``result.json`` турникета или строка
   «Контракт результата: status=…» в логе прогона) — источник, внешний сессии.

Классы:

* ``verified-complete`` — ``status=complete``; идут в SFT-компонент;
* ``verified-partial`` — ``status=partial`` у источника **и** механический признак
  зелёного сьюта в последних tool-результатах эпизода
  (:data:`SUITE_GREEN_RE` / :data:`SUITE_RED_RE`, окно :data:`GREEN_SUITE_WINDOW`);
  в SFT-компонент допускается **с флагом** ``verification: partial-green``;
* ``verified-failed`` — ``status`` из ``{blocked, conflicts, failed}``; negative-пул RL;
* ``unverified`` — всё остальное, включая ``partial`` без зелёного сьюта,
  отсутствие контракта и неоднозначную пару с harness-отчётом; в SFT не попадает,
  считается счётчиком.

Признак зелёного сьюта — **механический**, по последним ``GREEN_SUITE_WINDOW``
tool-результатам эпизода: сьют зелёный, если в их текстах есть строка вида
«N passed»/«passed» (:data:`SUITE_GREEN_RE`) и нет красного признака
(:data:`SUITE_RED_RE`). Красный признак — и верхний регистр pytest
(«FAILED»/«ERROR»), и строчные итоги того же сьюта («1 failed, 5 passed»,
«2 errors») — дельта-3b: строка «1 failed, 5 passed» красная, а не зелёная.
Проза и рассуждения модели признак не дают: берутся только ``tool_result``-ходы,
то есть вывод среды.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterator

VERIFIED_COMPLETE = "verified-complete"
VERIFIED_PARTIAL = "verified-partial"
VERIFIED_FAILED = "verified-failed"
UNVERIFIED = "unverified"

CLASSES = (VERIFIED_COMPLETE, VERIFIED_PARTIAL, VERIFIED_FAILED, UNVERIFIED)

#: Классы, допускаемые в SFT-компонент (ADR-020, дельты-1/3).
SFT_CLASSES = (VERIFIED_COMPLETE, VERIFIED_PARTIAL)

#: Словарь статусов — ровно контракт результата харнесса (complete|partial|blocked)
#: плюс статусы, названные в дельте-1 ADR-020. Никаких синонимов: неизвестный
#: статус — это unverified, а не догадка о его смысле.
COMPLETE_STATUSES = frozenset({"complete"})
FAILED_STATUSES = frozenset({"blocked", "conflicts", "failed"})
#: ``partial`` — исход объявлен неполным; сам по себе в SFT не пускает, но в паре
#: с зелёным сьютом даёт ``verified-partial``.
PARTIAL_STATUSES = frozenset({"partial"})

#: Значения evidence в эпизоде — чем именно подтверждён класс.
EVIDENCE_CONTRACT = "in-session-contract"
EVIDENCE_HARNESS = "harness-report"
EVIDENCE_PARTIAL_GREEN = "partial-green"
EVIDENCE_NONE = "none"

#: Флаг записи: эпизод допущен в SFT как **частично** верифицированный. Прочие
#: записи флага не несут (``verification: null``): их допуск решает класс.
VERIFICATION_PARTIAL_GREEN = "partial-green"

#: Окно поиска признака зелёного сьюта: последние N tool-результатов эпизода.
GREEN_SUITE_WINDOW = 5

#: Зелёный сьют: строка вида «N passed»/«passed» (pytest-подобный итог).
SUITE_GREEN_RE = re.compile(r"\b(?:\d+\s+)?passed\b")
#: Красный сьют: верхний регистр pytest («FAILED»/«ERROR») **и** строчные итоги того
#: же сьюта («1 failed, 5 passed», «2 errors») — bare-слова ``failed``/``error`` в
#: последних tool-результатах. Регистр не значим намеренно (дельта-3b): до этого
#: строка «1 failed, 5 passed» проходила как зелёный сьют (``passed``) и давала
#: ``verified-partial`` красному прогону. Ложное срабатывание стоит одного
#: неверифицированного эпизода, пропуск красного — целого класса брака в SFT.
SUITE_RED_RE = re.compile(r"(?i)\b(?:failed|error)s?\b")

#: Окно поиска JSON-объекта: хвост ответа, а не весь текст (защита от O(n^2)).
TAIL_WINDOW = 64_000

_FENCE_TAIL = re.compile(r"```[A-Za-z0-9_+-]*")


def _iter_json_objects(text: str) -> Iterator[tuple[Any, int]]:
    """Отдать (объект, позиция конца) для всех JSON-объектов верхнего уровня."""
    length = len(text)
    index = 0
    while index < length:
        start = text.find("{", index)
        if start < 0:
            return
        depth = 0
        in_string = False
        escaped = False
        end = -1
        for pos in range(start, length):
            char = text[pos]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = pos + 1
                    break
        if end < 0:
            return
        try:
            candidate = json.loads(text[start:end])
        except (json.JSONDecodeError, ValueError, RecursionError):
            index = start + 1
            continue
        if isinstance(candidate, dict):
            yield candidate, end
        index = end


def _is_trailing(text: str, end: int) -> bool:
    suffix = text[end:].strip()
    if not suffix:
        return True
    return bool(_FENCE_TAIL.fullmatch(suffix))


def extract_final_contract(episode: Any) -> dict | None:
    """Завершающий JSON-контракт эпизода или None.

    Смотрится последний текстовый ход ассистента: контракт обязан быть в конце
    ответа (после него — только пробелы и, возможно, закрывающая фенса).
    """
    turns = getattr(episode, "turns", None)
    if not turns:
        return None
    final_text = None
    for turn in reversed(turns):
        if getattr(turn, "role", None) == "assistant" and getattr(turn, "kind", None) == "text":
            final_text = getattr(turn, "content", "") or ""
            break
    if not final_text:
        return None
    window = final_text[-TAIL_WINDOW:] if len(final_text) > TAIL_WINDOW else final_text
    offset = len(final_text) - len(window)

    last: tuple[dict, int] | None = None
    for candidate, end in _iter_json_objects(window):
        if isinstance(candidate.get("status"), str):
            last = (candidate, end + offset)
    if last is None:
        return None
    candidate, end = last
    if not _is_trailing(final_text, end):
        return None
    return candidate


def _status_class(status: str | None) -> str | None:
    if not isinstance(status, str):
        return None
    normalized = status.strip().lower()
    if normalized in COMPLETE_STATUSES:
        return VERIFIED_COMPLETE
    if normalized in FAILED_STATUSES:
        return VERIFIED_FAILED
    return None


def _normalized_status(status: str | None) -> str:
    return status.strip().lower() if isinstance(status, str) else ""


def green_suite_evidence(episode: Any) -> bool:
    """True, если последние ``GREEN_SUITE_WINDOW`` tool-результатов дают зелёный сьют.

    Признак строго механический (AD-2) и смотрит только на вывод среды:

    * берутся последние ``GREEN_SUITE_WINDOW`` ходов ``kind=tool_result``
      (рассуждения и текст ассистента в признак не входят — модель может
      написать «тесты прошли» и без тестов);
    * сьют зелёный, если в их текстах срабатывает :data:`SUITE_GREEN_RE`
      («N passed»/«passed») и **не** срабатывает :data:`SUITE_RED_RE`
      («FAILED»/«ERROR» верхним регистром и строчные итоги «1 failed, 5 passed»,
      «2 errors») — красная строка в тех же результатах отменяет признак.

    Отсутствие tool-результатов — не зелёный сьют (``False``), а не «нет данных
    против»: без вывода среды механического подтверждения нет.
    """
    turns = getattr(episode, "turns", None)
    if not turns:
        return False
    tail = [
        str(getattr(turn, "content", "") or "")
        for turn in turns
        if getattr(turn, "kind", None) == "tool_result"
    ][-GREEN_SUITE_WINDOW:]
    if not tail:
        return False
    joined = "\n".join(tail)
    return bool(SUITE_GREEN_RE.search(joined)) and not SUITE_RED_RE.search(joined)


def classify_with_evidence(
    episode: Any, harness_status: str | None = None
) -> tuple[str, str]:
    """Класс верификации эпизода и источник вердикта.

    Правило разрешения противоречий — **fail-closed**: в SFT эпизод попадает,
    только если ни один из источников не говорит «провал».

    1. контракт сессии ``failed/blocked/conflicts`` → ``verified-failed``;
    2. harness-отчёт ``failed/blocked/conflicts`` → ``verified-failed``
       (в том числе когда контракт сессии объявил ``complete``: расхождение
       источников трактуется против эпизода, а не в его пользу);
    3. иначе ``complete`` от любого источника → ``verified-complete``;
    4. иначе ``partial`` у источника **и** зелёный сьют в последних
       tool-результатах (:func:`green_suite_evidence`) → ``verified-partial``;
    5. иначе ``unverified`` — включая неполный статус контракта (``partial``)
       без механического подтверждения: harness-отчёт по времени может только
       подтвердить полный успех, но не переписать «сессия сама сказала, что не
       закончила».

    ``verified-partial`` — снисхождение ровно на один шаг и только по
    механическому свидетельству: неполный статус (``partial``) плюс вывод среды
    с зелёным сьютом. Полного класса он не даёт (``partial`` ≠ ``complete``), а
    без зелёного сьюта остаётся ``unverified``.
    """
    contract = extract_final_contract(episode)
    contract_class = _status_class(contract.get("status")) if contract is not None else None
    contract_status = _normalized_status(contract.get("status")) if contract is not None else ""

    if contract_class == VERIFIED_FAILED:
        return VERIFIED_FAILED, EVIDENCE_CONTRACT

    harness_class = _status_class(harness_status)
    if harness_class == VERIFIED_FAILED:
        return VERIFIED_FAILED, EVIDENCE_HARNESS

    if contract is not None:
        if contract_class == VERIFIED_COMPLETE:
            return VERIFIED_COMPLETE, EVIDENCE_CONTRACT
        if contract_status in PARTIAL_STATUSES and green_suite_evidence(episode):
            return VERIFIED_PARTIAL, EVIDENCE_PARTIAL_GREEN
        return UNVERIFIED, EVIDENCE_CONTRACT

    if harness_class == VERIFIED_COMPLETE:
        return VERIFIED_COMPLETE, EVIDENCE_HARNESS
    if _normalized_status(harness_status) in PARTIAL_STATUSES and green_suite_evidence(episode):
        return VERIFIED_PARTIAL, EVIDENCE_PARTIAL_GREEN
    return UNVERIFIED, EVIDENCE_NONE


def classify_episode(episode: Any, harness_status: str | None = None) -> str:
    """Класс верификации: verified-complete | verified-partial | verified-failed | unverified."""
    return classify_with_evidence(episode, harness_status)[0]


def classify(episode: Any, harness_status: str | None = None) -> str:
    """Синоним ``classify_episode`` (короткая форма для отчётов)."""
    return classify_episode(episode, harness_status)


def apply_class(episode: Any, harness_status: str | None = None) -> str:
    """Проставить ``episode.cls`` и ``episode.evidence``; вернуть класс."""
    cls, evidence = classify_with_evidence(episode, harness_status)
    episode.cls = cls
    episode.evidence = evidence
    return cls
