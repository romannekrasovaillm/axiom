#!/usr/bin/env python3
"""C-058/C-059 — детектор точки изменения (changepoint) кривой loss.

Относительный дрейф (медиана последних 1000 шагов, +3%) ловит резкую деградацию,
но не видит **устойчивую ступень**: после сдвига окно само впитывает новый
уровень.  Этот инструмент сравнивает «до» и «после» на блочных медианах
(ADR-060, changes/loss-changepoint-guard/DELTA.md):

* вход — строки схемы ``pretrain-metrics/v1`` (``step``, ``loss``) из файлов
  ``metrics-seg-*.jsonl``; каталог разворачивается в файлы, сегменты
  сортируются по номеру (``seg-2`` раньше ``seg-10``), дубли шага
  схлопываются — побеждает последнее вхождение (перекрытие сегментов);
* блок ``B=25`` шагов → медиана loss блока, хвост короче B отбрасывается;
* для каждого блока ``k`` с историей ``>= N+G+M`` блоков:
  ``recent = med(блоки k-N+1..k)``, ``reference = med(блоки k-N-G-M+1..k-N-G)``,
  ``delta = recent - reference``,
  ``z = delta / (sigma_b * sqrt(1/N + 1/M))``,
  где ``sigma_b = 1.4826 * MAD`` по **всем** блочным медианам;
* событие — ``|delta| >= min_abs`` и ``|z| >= z_threshold``, удержание
  ``>= persist`` блоков подряд; ``step`` события — медианный шаг первого блока
  удержания, ``delta``/``z`` — значения того же первого блока, ``blocks`` —
  длина удержания; события, начала которых отстоят менее чем на 3 блока,
  сливаются (остаётся раннее);
* реестр ``--ledger`` подавляет событие при ``|detected_step - step| <= tol``;
  запись без ``owner``/``reason``/``ref``/``review_by`` недействительна и
  не подавляет (предупреждение в stderr).

Коды возврата: **0** — неподавленных событий нет (в том числе нет данных,
``status=neutral``); **1** — есть неподавленное событие; **2** — некорректный
вход (битая JSON-строка, нечисловой ``step``, NaN/Inf ``loss``, пустой набор
строк, ``sigma_b == 0``, отсутствующий/битый ledger).

Зависимости — только стандартная библиотека; PyYAML используется, если он
установлен, иначе включается минимальный встроенный парсер подмножества YAML
(список отображений с плоскими скалярами и блочными строками ``>-``/``|``).
Сеть, JAX и numpy не используются: результат детерминирован и воспроизводим на
любой машине.

Запуск::

    python3 tools/check_loss_changepoint.py --metrics <каталог|файл> [...] \\
        [--ledger evidence/monitoring/changepoints.yaml] [--json] [--quiet]
    python3 tools/check_loss_changepoint.py --selftest

``rows`` в JSON — число уникальных шагов после схлопывания дублей (именно этот
ряд поступает в блоки); ``blocks`` — число полных блоков по B шагов.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

try:  # PyYAML — опциональная зависимость; без неё работает встроенный парсер.
    import yaml as _yaml
except Exception:  # pragma: no cover - зависит от окружения
    _yaml = None

HAVE_PYYAML = _yaml is not None

# Коды возврата (DELTA A7).
EXIT_OK = 0
EXIT_ALERT = 1
EXIT_INPUT = 2

# Статусы JSON-отчёта.
STATUS_OK = "ok"
STATUS_ALERT = "alert"
STATUS_NEUTRAL = "neutral"

# Схема строк живого прогона (ADR-057) и шаблон имени сегмента.
METRICS_SCHEMA = "pretrain-metrics/v1"
SEGMENT_RE = re.compile(r"^metrics-seg-(\d+)\.jsonl$")

# Параметры по умолчанию — откалиброваны на long-jax2-ab14050 (ADR-060).
DEFAULT_BLOCK = 25
DEFAULT_RECENT = 6
DEFAULT_REFERENCE = 12
DEFAULT_GAP = 2
DEFAULT_MIN_ABS = 0.10
DEFAULT_Z_THRESHOLD = 3.5
DEFAULT_PERSIST = 3
DEFAULT_TOL = 400

#: События, отстоящие менее чем на столько блоков, сливаются (DELTA A5).
MERGE_DISTANCE_BLOCKS = 3

#: Обязательные поля записи реестра, без которых она не подавляет событие.
LEDGER_REQUIRED_FIELDS = ("owner", "reason", "ref", "review_by")


class InputError(Exception):
    """Некорректный вход: CLI обязан вернуть exit 2 с сообщением в stderr."""


# ---------------------------------------------------------------------------
# Параметры и структуры данных
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Params:
    """Параметры детектора; все вынесены в CLI (ADR-060, Consequences)."""

    block: int = DEFAULT_BLOCK
    recent: int = DEFAULT_RECENT
    reference: int = DEFAULT_REFERENCE
    gap: int = DEFAULT_GAP
    min_abs: float = DEFAULT_MIN_ABS
    z_threshold: float = DEFAULT_Z_THRESHOLD
    persist: int = DEFAULT_PERSIST
    tol: int = DEFAULT_TOL

    def validate(self) -> None:
        if self.block < 1:
            raise InputError("--block должен быть >= 1")
        if self.recent < 1:
            raise InputError("--recent должен быть >= 1")
        if self.reference < 1:
            raise InputError("--reference должен быть >= 1")
        if self.gap < 0:
            raise InputError("--gap должен быть >= 0")
        if self.min_abs < 0:
            raise InputError("--min-abs должен быть >= 0")
        if self.z_threshold < 0:
            raise InputError("--z-threshold должен быть >= 0")
        if self.persist < 1:
            raise InputError("--persist должен быть >= 1")
        if self.tol < 0:
            raise InputError("--tol должен быть >= 0")

    def to_dict(self, sigma_b: float | None = None) -> dict[str, Any]:
        return {
            "block": self.block,
            "recent": self.recent,
            "reference": self.reference,
            "gap": self.gap,
            "min_abs": self.min_abs,
            "z_threshold": self.z_threshold,
            "persist": self.persist,
            "tol": self.tol,
            "sigma_b": None if sigma_b is None else round(sigma_b, 9),
        }


@dataclass(frozen=True)
class Block:
    """Полный блок из B подряд идущих шагов (хвост короче B отброшен)."""

    index: int
    first_step: int
    last_step: int
    median_step: float
    median_loss: float


@dataclass(frozen=True)
class Event:
    """Обнаруженный changepoint."""

    step: float
    direction: str
    delta: float
    z: float
    blocks: int
    start_block: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "direction": self.direction,
            "delta": round(self.delta, 6),
            "z": round(self.z, 6),
            "blocks": self.blocks,
        }


@dataclass(frozen=True)
class _Run:
    """Внутреннее состояние удержания условия подряд по блокам."""

    start_block: int
    last_block: int
    direction: str
    delta: float
    z: float
    blocks: int


@dataclass
class Analysis:
    """Результат статистической части (до подавления реестром)."""

    events: list[Event]
    blocks: list[Block]
    sigma_b: float | None
    evaluated_blocks: int


@dataclass(frozen=True)
class LedgerRecord:
    """Действительная запись реестра известных сдвигов."""

    index: int
    step: int
    direction: str | None
    owner: str
    reason: str
    ref: str
    review_by: str


@dataclass(frozen=True)
class InvalidLedgerRecord:
    """Запись, которая не подавляет событие; причина сообщается."""

    index: int
    reason: str


@dataclass
class Ledger:
    records: list[LedgerRecord] = field(default_factory=list)
    invalid: list[InvalidLedgerRecord] = field(default_factory=list)
    source: str | None = None


@dataclass(frozen=True)
class SuppressedEvent:
    event: Event
    record: LedgerRecord

    def to_dict(self) -> dict[str, Any]:
        data = self.event.to_dict()
        data.update(
            {
                "ledger_step": self.record.step,
                "ledger_direction": self.record.direction,
                "owner": self.record.owner,
                "reason": self.record.reason,
                "ref": self.record.ref,
                "review_by": self.record.review_by,
            }
        )
        return data


@dataclass
class LoadedMetrics:
    """Загруженный ряд: уникальные шаги, отсортированные по возрастанию."""

    series: list[tuple[int, float]]
    rows_read: int
    files: list[Path]


@dataclass
class CheckResult:
    """Итог проверки — ровно поля JSON-контракта DELTA A9."""

    status: str
    events: list[Event]
    suppressed: list[SuppressedEvent]
    params: dict[str, Any]
    rows: int
    blocks: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "events": [event.to_dict() for event in self.events],
            "suppressed": [item.to_dict() for item in self.suppressed],
            "params": self.params,
            "rows": self.rows,
            "blocks": self.blocks,
        }


# ---------------------------------------------------------------------------
# Загрузка метрик
# ---------------------------------------------------------------------------


def resolve_metrics_files(paths: Sequence[str | Path]) -> list[Path]:
    """Развернуть пути ``--metrics`` в список файлов.

    Каталог → все ``metrics-seg-*.jsonl``, отсортированные по **номеру**
    сегмента (``seg-2`` раньше ``seg-10``), а не лексикографически.  Файл
    берётся как есть.  Несуществующий путь источника данных не даёт: это
    состояние ``neutral`` (C-059: отсутствие метрик — не красный гейт).
    Повторно указанный файл читается один раз.
    """

    files: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_dir():
            found: list[tuple[int, str, Path]] = []
            for candidate in path.glob("metrics-seg-*.jsonl"):
                match = SEGMENT_RE.match(candidate.name)
                if match:
                    found.append((int(match.group(1)), candidate.name, candidate))
            found.sort(key=lambda item: (item[0], item[1]))
            files.extend(item[2] for item in found)
        elif path.is_file():
            files.append(path)

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in files:
        try:
            key = path.resolve()
        except OSError:  # pragma: no cover - разрешение пути почти всегда работает
            key = path.absolute()
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _coerce_step(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputError(f"{location}: нечисловой step: {value!r}")
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise InputError(f"{location}: нечисловой step: {value!r}")
        return int(value)
    return value


def _coerce_loss(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputError(f"{location}: нечисловой loss: {value!r}")
    loss = float(value)
    if not math.isfinite(loss):
        raise InputError(f"{location}: NaN/Inf loss: {value!r}")
    return loss


def load_metrics(paths: Sequence[str | Path]) -> LoadedMetrics:
    """Прочитать ``metrics-seg-*.jsonl``, проверить строки и схлопнуть дубли.

    Ошибки входа (битая JSON-строка, нечисловой ``step``, NaN/Inf ``loss``,
    пустой набор строк) поднимают :class:`InputError` → exit 2.  Дубли шага
    схлопываются: остаётся последнее вхождение в порядке чтения (сегменты уже
    упорядочены по номеру).
    """

    files = resolve_metrics_files(paths)
    latest: dict[int, float] = {}
    rows_read = 0

    for path in files:
        try:
            handle = path.open("r", encoding="utf-8")
        except OSError as exc:
            raise InputError(f"не удалось открыть {path}: {exc}") from exc
        try:
            with handle:
                for lineno, line in enumerate(handle, 1):
                    text = line.strip()
                    if not text:
                        continue
                    location = f"{path}:{lineno}"
                    try:
                        row = json.loads(text)
                    except json.JSONDecodeError as exc:
                        raise InputError(
                            f"{location}: битая JSON-строка: {exc.msg}"
                        ) from exc
                    if not isinstance(row, dict):
                        raise InputError(
                            f"{location}: строка не является JSON-объектом"
                        )
                    step = _coerce_step(row.get("step"), location)
                    loss = _coerce_loss(row.get("loss"), location)
                    # Последнее вхождение шага побеждает (перекрытие сегментов).
                    latest[step] = loss
                    rows_read += 1
        except UnicodeDecodeError as exc:
            raise InputError(f"{path}: файл не в UTF-8: {exc}") from exc

    if files and rows_read == 0:
        raise InputError(
            "пустой набор строк: в найденных файлах нет ни одной строки "
            "схемы pretrain-metrics/v1"
        )

    return LoadedMetrics(
        series=sorted(latest.items()),
        rows_read=rows_read,
        files=files,
    )


# ---------------------------------------------------------------------------
# Блочная статистика
# ---------------------------------------------------------------------------


def build_blocks(
    series: Sequence[tuple[int, float]], block: int = DEFAULT_BLOCK
) -> list[Block]:
    """Нарезать отсортированный ряд на блоки по ``block`` шагов.

    Значение блока — медиана loss.  Хвост короче ``block`` отбрасывается.
    """

    if block < 1:
        raise InputError("--block должен быть >= 1")
    blocks: list[Block] = []
    for start in range(0, len(series) - block + 1, block):
        chunk = series[start : start + block]
        steps = [step for step, _ in chunk]
        losses = [loss for _, loss in chunk]
        blocks.append(
            Block(
                index=len(blocks),
                first_step=steps[0],
                last_step=steps[-1],
                median_step=statistics.median(steps),
                median_loss=statistics.median(losses),
            )
        )
    return blocks


def robust_sigma(block_medians: Sequence[float]) -> float:
    """``sigma_b = 1.4826 * MAD`` по всем блочным медианам (ADR-060)."""

    if not block_medians:
        raise InputError("sigma_b не определена: нет блоков")
    center = statistics.median(block_medians)
    mad = statistics.median([abs(value - center) for value in block_medians])
    return 1.4826 * mad


def _window_medians(
    medians: Sequence[float], k: int, params: Params
) -> tuple[float, float]:
    recent = medians[k - params.recent + 1 : k + 1]
    reference_end = k - params.recent - params.gap
    reference_start = reference_end - params.reference + 1
    reference = medians[reference_start : reference_end + 1]
    return statistics.median(recent), statistics.median(reference)


def detect_changepoints(
    series: Sequence[tuple[int, float]], params: Params | None = None
) -> Analysis:
    """Найти точки изменения loss по двухоконному робастному тесту.

    ``series`` — уникальные шаги, отсортированные по возрастанию.  Условие
    события должно удерживаться ``persist`` блоков подряд; направление
    удержания не меняется (смена знака ``delta`` закрывает удержание).
    """

    params = params or Params()
    params.validate()
    blocks = build_blocks(series, params.block)
    if not blocks:
        return Analysis([], blocks, None, 0)

    medians = [block.median_loss for block in blocks]
    sigma_b = robust_sigma(medians)

    history = params.recent + params.gap + params.reference
    evaluated = len(blocks) - history + 1
    if evaluated <= 0:
        # Истории не хватает: окна не сравниваются, z не вычисляется.
        return Analysis([], blocks, sigma_b, 0)
    if sigma_b == 0.0:
        raise InputError(
            "sigma_b == 0: все блочные медианы совпадают (вырожденный вход)"
        )

    z_denominator = sigma_b * math.sqrt(
        1.0 / params.recent + 1.0 / params.reference
    )

    runs: list[_Run] = []
    current: _Run | None = None
    for k in range(history - 1, len(blocks)):
        recent_median, reference_median = _window_medians(medians, k, params)
        delta = recent_median - reference_median
        z = delta / z_denominator
        qualifies = (
            abs(delta) >= params.min_abs and abs(z) >= params.z_threshold
        )
        direction = "up" if delta > 0 else "down"

        if qualifies:
            if (
                current is not None
                and k == current.last_block + 1
                and direction == current.direction
            ):
                current = _Run(
                    start_block=current.start_block,
                    last_block=k,
                    direction=current.direction,
                    delta=current.delta,
                    z=current.z,
                    blocks=current.blocks + 1,
                )
            else:
                if current is not None:
                    runs.append(current)
                current = _Run(
                    start_block=k,
                    last_block=k,
                    direction=direction,
                    delta=delta,
                    z=z,
                    blocks=1,
                )
        elif current is not None:
            runs.append(current)
            current = None
    if current is not None:
        runs.append(current)

    events = [
        Event(
            step=blocks[run.start_block].median_step,
            direction=run.direction,
            delta=run.delta,
            z=run.z,
            blocks=run.blocks,
            start_block=run.start_block,
        )
        for run in runs
        if run.blocks >= params.persist
    ]
    return Analysis(
        events=merge_events(events),
        blocks=blocks,
        sigma_b=sigma_b,
        evaluated_blocks=evaluated,
    )


def merge_events(events: Sequence[Event]) -> list[Event]:
    """Слить события, отстоящие менее чем на 3 блока; остаётся раннее."""

    merged: list[Event] = []
    for event in events:
        if (
            merged
            and event.start_block - merged[-1].start_block
            < MERGE_DISTANCE_BLOCKS
        ):
            continue
        merged.append(event)
    return merged


# ---------------------------------------------------------------------------
# Реестр известных сдвигов
# ---------------------------------------------------------------------------


class _LedgerSyntaxError(ValueError):
    """Синтаксис подмножества YAML не разобран встроенным парсером."""


@dataclass(frozen=True)
class _YamlLine:
    indent: int
    text: str
    number: int


def _strip_yaml_comment(line: str) -> str:
    """Убрать комментарий ``#`` вне кавычек (встроенный парсер)."""

    out: list[str] = []
    quote: str | None = None
    previous = ""
    for char in line:
        if quote is not None:
            out.append(char)
            if char == quote and previous != "\\":
                quote = None
        elif char in ('"', "'"):
            quote = char
            out.append(char)
        elif char == "#" and (not out or previous.isspace()):
            break
        else:
            out.append(char)
        previous = char
    return "".join(out).rstrip()


def _yaml_lines(text: str) -> list[_YamlLine]:
    lines: list[_YamlLine] = []
    for number, raw in enumerate(text.splitlines(), 1):
        cleaned = _strip_yaml_comment(raw)
        if not cleaned.strip():
            continue
        stripped = cleaned.lstrip(" ")
        indent = len(cleaned) - len(stripped)
        if "\t" in cleaned[:indent]:
            raise _LedgerSyntaxError(f"строка {number}: табуляция в отступе")
        if stripped in ("---", "..."):
            continue
        lines.append(_YamlLine(indent=indent, text=stripped, number=number))
    return lines


def _parse_yaml_scalar(raw: str) -> Any:
    text = raw.strip()
    if text == "":
        return None
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
        if text[0] == '"':
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text[1:-1]
        return text[1:-1].replace("''", "'")
    lowered = text.lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if lowered in ("null", "~", "none"):
        return None
    if re.fullmatch(r"[+-]?\d+", text):
        return int(text)
    if re.fullmatch(
        r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?", text
    ):
        return float(text)
    return text


def _is_block_marker(value: str) -> bool:
    return value in ("|", "|-", "|+", ">", ">-", ">+")


def _parse_block_scalar(
    lines: Sequence[_YamlLine], index: int, key_indent: int, marker: str
) -> tuple[str, int]:
    """Собрать блочную строку (``>-``/``|``) до ключа с меньшим отступом."""

    collected: list[str] = []
    while index < len(lines) and lines[index].indent > key_indent:
        collected.append(lines[index].text)
        index += 1
    if marker.startswith(">"):
        value = " ".join(collected)  # folded: строки складываются пробелом
    else:
        value = "\n".join(collected)  # literal
    if marker.endswith("-"):
        value = value.rstrip()
    elif value:
        value += "\n"
    return value, index


def _split_key(text: str, number: int) -> tuple[str, str]:
    key, separator, rest = text.partition(":")
    if not separator or not key.strip():
        raise _LedgerSyntaxError(
            f"строка {number}: ожидалось 'key: value', получено {text!r}"
        )
    return key.strip(), rest.strip()


def _parse_mapping(
    lines: Sequence[_YamlLine], index: int, indent: int
) -> tuple[dict[str, Any], int]:
    result: dict[str, Any] = {}
    while index < len(lines):
        line = lines[index]
        if line.indent < indent:
            break
        if line.text.startswith("-"):
            break
        if line.indent > indent:
            raise _LedgerSyntaxError(
                f"строка {line.number}: неожиданный отступ"
            )
        key, value = _split_key(line.text, line.number)
        if _is_block_marker(value):
            block, index = _parse_block_scalar(lines, index + 1, indent, value)
            result[key] = block
        elif value == "":
            index += 1
            if index < len(lines) and lines[index].indent > indent:
                child, index = _parse_block(
                    lines, index, lines[index].indent
                )
                result[key] = child
            else:
                result[key] = None
        else:
            result[key] = _parse_yaml_scalar(value)
            index += 1
    return result, index


def _parse_sequence(
    lines: Sequence[_YamlLine], index: int, indent: int
) -> tuple[list[Any], int]:
    items: list[Any] = []
    while index < len(lines):
        line = lines[index]
        if line.indent != indent or not line.text.startswith("-"):
            break
        if line.text == "-":
            index += 1
            if index < len(lines) and lines[index].indent > indent:
                child, index = _parse_block(
                    lines, index, lines[index].indent
                )
                items.append(child)
            else:
                items.append(None)
            continue
        if not line.text.startswith("- "):
            raise _LedgerSyntaxError(
                f"строка {line.number}: ожидался элемент списка '- '"
            )
        rest = line.text[2:]
        if rest.startswith("- "):
            raise _LedgerSyntaxError(
                f"строка {line.number}: вложенные списки не поддерживаются"
            )
        if ":" not in rest:
            items.append(_parse_yaml_scalar(rest))
            index += 1
            continue
        # Отображение внутри списка: первая пара key: value стоит после '- '.
        content_indent = line.indent + 2
        virtual = _YamlLine(
            indent=content_indent, text=rest, number=line.number
        )
        virtual_lines = list(lines[:index]) + [virtual] + list(lines[index + 1 :])
        item, index = _parse_mapping(virtual_lines, index, content_indent)
        items.append(item)
    return items, index


def _parse_block(
    lines: Sequence[_YamlLine], index: int, indent: int
) -> tuple[Any, int]:
    if lines[index].text.startswith("-"):
        return _parse_sequence(lines, index, indent)
    return _parse_mapping(lines, index, indent)


def parse_yaml_subset(text: str) -> Any:
    """Минимальный YAML-парсер нужного подмножества (без PyYAML).

    Поддерживает: отображения, списки отображений, плоские скаляры
    (int/float/bool/null/строки в кавычках) и блочные строки ``|``/``>``
    с chomping-индикаторами ``-``/``+``.  Комментарии ``#`` вне кавычек
    отбрасываются.  Всё остальное — :class:`_LedgerSyntaxError` → exit 2.
    """

    lines = _yaml_lines(text)
    if not lines:
        return None
    value, index = _parse_block(lines, 0, lines[0].indent)
    if index != len(lines):
        raise _LedgerSyntaxError(
            f"строка {lines[index].number}: неожиданное продолжение документа"
        )
    return value


def _nonempty_text(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _ledger_record(item: Any, index: int) -> tuple[LedgerRecord | None, InvalidLedgerRecord | None]:
    """Проверить запись реестра; недействительная не подавляет событие."""

    if not isinstance(item, dict):
        return None, InvalidLedgerRecord(index, "запись не является отображением")

    step_value = item.get("step")
    if (
        isinstance(step_value, bool)
        or not isinstance(step_value, (int, float))
        or (
            isinstance(step_value, float)
            and (not math.isfinite(step_value) or not step_value.is_integer())
        )
    ):
        return None, InvalidLedgerRecord(
            index, f"нечисловой step: {step_value!r}"
        )

    missing = [
        name for name in LEDGER_REQUIRED_FIELDS if _nonempty_text(item.get(name)) is None
    ]
    direction_raw = item.get("direction")
    direction: str | None = None
    if direction_raw is not None:
        direction_text = _nonempty_text(direction_raw)
        if direction_text is None or direction_text.lower() not in ("up", "down"):
            missing.append("direction (up|down)")
        else:
            direction = direction_text.lower()
    if missing:
        return None, InvalidLedgerRecord(
            index, "нет обязательных полей: " + ", ".join(missing)
        )

    return (
        LedgerRecord(
            index=index,
            step=int(step_value),
            direction=direction,
            owner=_nonempty_text(item.get("owner")) or "",
            reason=_nonempty_text(item.get("reason")) or "",
            ref=_nonempty_text(item.get("ref")) or "",
            review_by=_nonempty_text(item.get("review_by")) or "",
        ),
        None,
    )


def ledger_from_raw(raw: Any, source: str | None = None) -> Ledger:
    """Нормализовать разобранный YAML в реестр с валидацией записей."""

    if raw is None:
        entries: Any = []
    elif isinstance(raw, dict):
        entries = raw.get("changepoints", [])
    elif isinstance(raw, list):
        entries = raw
    else:
        raise InputError(
            "ledger: ожидался список changepoints или отображение с ним"
        )
    if entries is None:
        entries = []
    if not isinstance(entries, list):
        raise InputError("ledger: changepoints должен быть списком записей")

    ledger = Ledger(source=source)
    for index, item in enumerate(entries):
        record, invalid = _ledger_record(item, index)
        if record is not None:
            ledger.records.append(record)
        if invalid is not None:
            ledger.invalid.append(invalid)
    return ledger


def parse_ledger_text(
    text: str, *, prefer_pyyaml: bool | None = None, source: str | None = None
) -> Ledger:
    """Разобрать YAML-реестр: PyYAML, если доступен, иначе встроенный парсер."""

    use_pyyaml = HAVE_PYYAML if prefer_pyyaml is None else prefer_pyyaml
    if use_pyyaml:
        if _yaml is None:
            raise InputError(
                "PyYAML недоступен, а встроенный парсер не разрешён "
                "(prefer_pyyaml=True)"
            )
        try:
            raw = _yaml.safe_load(text)
        except Exception as exc:  # PyYAML кидает разные типы исключений
            raise InputError(f"YAML ledger не разобран: {exc}") from exc
    else:
        try:
            raw = parse_yaml_subset(text)
        except _LedgerSyntaxError as exc:
            raise InputError(
                f"YAML ledger не разобран встроенным парсером: {exc}"
            ) from exc
    return ledger_from_raw(raw, source=source)


def load_ledger(path: str | Path) -> Ledger:
    """Прочитать реестр; отсутствие явно указанного файла — exit 2."""

    ledger_path = Path(path).expanduser()
    if not ledger_path.is_file():
        raise InputError(f"ledger не найден: {ledger_path}")
    try:
        text = ledger_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InputError(f"не удалось прочитать ledger {ledger_path}: {exc}") from exc
    return parse_ledger_text(text, source=str(ledger_path))


def apply_ledger(
    events: Sequence[Event], ledger: Ledger, tol: int = DEFAULT_TOL
) -> tuple[list[SuppressedEvent], list[Event]]:
    """Разделить события на подавленные реестром и неподавленные.

    Подавление — по расстоянию шагов (``|detected_step - step| <= tol``);
    направление записи не участвует в сопоставлении (DELTA A6).
    """

    suppressed: list[SuppressedEvent] = []
    unsuppressed: list[Event] = []
    for event in events:
        match = next(
            (
                record
                for record in ledger.records
                if abs(event.step - record.step) <= tol
            ),
            None,
        )
        if match is None:
            unsuppressed.append(event)
        else:
            suppressed.append(SuppressedEvent(event=event, record=match))
    return suppressed, unsuppressed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="check_loss_changepoint.py",
        description=(
            "Детектор точки изменения кривой loss по блочным медианам "
            "(ADR-060, C-058/C-059)"
        ),
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        action="extend",
        metavar="PATH",
        help="каталог с metrics-seg-*.jsonl или отдельный файл (можно несколько)",
    )
    parser.add_argument(
        "--ledger",
        metavar="YAML",
        help="реестр известных сдвигов (evidence/monitoring/changepoints.yaml)",
    )
    parser.add_argument("--block", type=int, default=DEFAULT_BLOCK, metavar="B")
    parser.add_argument("--recent", type=int, default=DEFAULT_RECENT, metavar="N")
    parser.add_argument(
        "--reference", type=int, default=DEFAULT_REFERENCE, metavar="M"
    )
    parser.add_argument("--gap", type=int, default=DEFAULT_GAP, metavar="G")
    parser.add_argument(
        "--min-abs", type=float, default=DEFAULT_MIN_ABS, metavar="X"
    )
    parser.add_argument(
        "--z-threshold", type=float, default=DEFAULT_Z_THRESHOLD, metavar="Z"
    )
    parser.add_argument(
        "--persist", type=int, default=DEFAULT_PERSIST, metavar="K"
    )
    parser.add_argument("--tol", type=int, default=DEFAULT_TOL, metavar="STEPS")
    parser.add_argument(
        "--json",
        action="store_true",
        help="один JSON-объект в stdout (status, events, suppressed, params, rows, blocks)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="только итоговая строка; при exit 0 в неинтерактивном режиме — ничего",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="встроенные детерминированные кейсы (fixed seed); exit 0 только при PASS",
    )
    return parser


def _format_event(prefix: str, event: Event) -> str:
    return (
        f"{prefix} step={event.step} direction={event.direction} "
        f"delta={event.delta:+.4f} z={event.z:+.2f} blocks={event.blocks}"
    )


def _format_suppressed(item: SuppressedEvent) -> str:
    base = _format_event("SUPPRESSED", item.event)
    return (
        f"{base} ledger_step={item.record.step} "
        f"owner={item.record.owner!r} review_by={item.record.review_by!r}"
    )


def _emit(result: CheckResult, args: argparse.Namespace, exit_code: int) -> int:
    if args.json:
        payload = json.dumps(
            result.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        sys.stdout.write(payload + "\n")
        return exit_code

    if args.quiet:
        if exit_code == EXIT_ALERT:
            print(f"НЕПОДАВЛЕННЫХ CHANGEPOINT: {len(result.events)}")
        elif sys.stdout.isatty():
            if result.status == STATUS_NEUTRAL:
                print("NEUTRAL: метрик не найдено")
            elif result.suppressed and not result.events:
                print(f"OK: все события подавлены реестром ({len(result.suppressed)})")
            else:
                print(
                    f"OK: changepoints не обнаружены "
                    f"(rows={result.rows}, blocks={result.blocks})"
                )
        return exit_code

    if result.status == STATUS_NEUTRAL:
        print("NEUTRAL: метрик не найдено")
        return exit_code
    if not result.events and not result.suppressed:
        print(
            f"OK: changepoints не обнаружены "
            f"(rows={result.rows}, blocks={result.blocks})"
        )
        return exit_code
    for event in result.events:
        print(_format_event("CHANGEPOINT", event))
    for item in result.suppressed:
        print(_format_suppressed(item))
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа CLI; возвращает код 0/1/2 (не вызывает ``sys.exit``)."""

    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse: --help → 0, ошибка аргументов → 2
        return int(exc.code) if exc.code is not None else EXIT_INPUT

    if args.selftest:
        return run_selftest()

    if not args.metrics:
        print(
            "ОШИБКА: нужен --metrics <каталог|файл> [...] или --selftest",
            file=sys.stderr,
        )
        return EXIT_INPUT

    params = Params(
        block=args.block,
        recent=args.recent,
        reference=args.reference,
        gap=args.gap,
        min_abs=args.min_abs,
        z_threshold=args.z_threshold,
        persist=args.persist,
        tol=args.tol,
    )
    try:
        params.validate()
    except InputError as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return EXIT_INPUT

    try:
        loaded = load_metrics(args.metrics)
        ledger = load_ledger(args.ledger) if args.ledger else Ledger()
    except InputError as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return EXIT_INPUT

    for invalid in ledger.invalid:
        print(
            f"ПРЕДУПРЕЖДЕНИЕ: ledger[{invalid.index}]: {invalid.reason}; "
            "запись недействительна и не подавляет",
            file=sys.stderr,
        )

    if not loaded.files:
        result = CheckResult(
            status=STATUS_NEUTRAL,
            events=[],
            suppressed=[],
            params=params.to_dict(None),
            rows=0,
            blocks=0,
        )
        return _emit(result, args, EXIT_OK)

    try:
        analysis = detect_changepoints(loaded.series, params)
    except InputError as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return EXIT_INPUT

    suppressed, unsuppressed = apply_ledger(analysis.events, ledger, params.tol)
    exit_code = EXIT_ALERT if unsuppressed else EXIT_OK
    result = CheckResult(
        status=STATUS_ALERT if unsuppressed else STATUS_OK,
        events=unsuppressed,
        suppressed=suppressed,
        params=params.to_dict(analysis.sigma_b),
        rows=len(loaded.series),
        blocks=len(analysis.blocks),
    )
    return _emit(result, args, exit_code)


# ---------------------------------------------------------------------------
# Встроенный selftest (C-058) — детерминированные синтетические кейсы
# ---------------------------------------------------------------------------


def _series_stationary(
    count: int, sigma: float, seed: int, base: float = 5.0
) -> list[tuple[int, float]]:
    rng = random.Random(seed)
    return [(step, base + rng.gauss(0.0, sigma)) for step in range(count)]


def _series_trend(
    count: int, total_delta: float, sigma: float, seed: int, base: float = 5.0
) -> list[tuple[int, float]]:
    rng = random.Random(seed)
    span = max(1, count - 1)
    return [
        (step, base + total_delta * step / span + rng.gauss(0.0, sigma))
        for step in range(count)
    ]


def _series_step(
    count: int,
    at: int,
    delta: float,
    sigma: float,
    seed: int,
    base: float = 5.0,
) -> list[tuple[int, float]]:
    rng = random.Random(seed)
    return [
        (
            step,
            base + (delta if step >= at else 0.0) + rng.gauss(0.0, sigma),
        )
        for step in range(count)
    ]


def _series_student_t3(
    count: int, sigma: float, seed: int, base: float = 5.0
) -> list[tuple[int, float]]:
    """Стационарный ряд с тяжёлыми хвостами: t с 3 степенями свободы."""

    rng = random.Random(seed)
    series: list[tuple[int, float]] = []
    for step in range(count):
        numerator = rng.gauss(0.0, 1.0)
        chi_square = sum(rng.gauss(0.0, 1.0) ** 2 for _ in range(3))
        series.append(
            (step, base + sigma * numerator / math.sqrt(chi_square / 3.0))
        )
    return series


def _write_metrics(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _case_expect_no_events(
    name: str, series: Sequence[tuple[int, float]]
) -> str | None:
    analysis = detect_changepoints(series)
    if analysis.events:
        found = [
            (event.step, event.direction, round(event.delta, 3))
            for event in analysis.events
        ]
        return f"{name}: ожидалось 0 событий, получено {len(found)}: {found}"
    return None


def _case_step_detected(
    name: str,
    series: Sequence[tuple[int, float]],
    expected_step: int,
    expected_direction: str,
    tolerance: int,
) -> str | None:
    analysis = detect_changepoints(series)
    if not analysis.events:
        return f"{name}: событие не обнаружено"
    event = analysis.events[0]
    if event.direction != expected_direction:
        return f"{name}: direction={event.direction}, ожидалось {expected_direction}"
    if abs(event.step - expected_step) > tolerance:
        return (
            f"{name}: detected step={event.step}, ожидалось "
            f"{expected_step} ± {tolerance}"
        )
    return None


def _selftest_case_nan_exit2() -> str | None:
    import contextlib
    import io
    import tempfile

    with tempfile.TemporaryDirectory(prefix="loss-changepoint-selftest-") as tmp:
        path = Path(tmp) / "metrics-seg-0.jsonl"
        path.write_text(
            json.dumps({"schema": METRICS_SCHEMA, "step": 0, "loss": 5.0})
            + "\n"
            + '{"schema": "pretrain-metrics/v1", "step": 1, "loss": NaN}\n',
            encoding="utf-8",
        )
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            code = main(["--metrics", str(path), "--quiet"])
    if code != EXIT_INPUT:
        return f"NaN в loss: exit {code}, ожидалось {EXIT_INPUT}"
    return None


def _selftest_case_duplicates_and_gap() -> str | None:
    import tempfile

    base = _series_stationary(20_000, 0.25, seed=2)
    with tempfile.TemporaryDirectory(prefix="loss-changepoint-selftest-") as tmp:
        directory = Path(tmp)
        first: list[dict[str, Any]] = []
        second: list[dict[str, Any]] = []
        for step, loss in base:
            if 7_000 <= step < 7_300:
                continue  # разрыв в нумерации шагов
            first.append({"schema": METRICS_SCHEMA, "step": step, "loss": loss})
            if step % 500 == 0:
                # дубль шага с чуть иным loss: побеждает последнее вхождение
                first.append(
                    {
                        "schema": METRICS_SCHEMA,
                        "step": step,
                        "loss": loss + 0.001,
                    }
                )
            if step >= 18_000:
                second.append(
                    {
                        "schema": METRICS_SCHEMA,
                        "step": step,
                        "loss": loss - 0.001,
                    }
                )
        _write_metrics(directory / "metrics-seg-1.jsonl", first)
        _write_metrics(directory / "metrics-seg-2.jsonl", second)
        loaded = load_metrics([str(directory)])
    if len(loaded.series) != 20_000 - 300:
        return (
            f"дубли/разрыв: уникальных шагов {len(loaded.series)}, "
            f"ожидалось {20_000 - 300}"
        )
    analysis = detect_changepoints(loaded.series)
    if analysis.events:
        return f"дубли/разрыв: ложное событие {analysis.events[0].to_dict()}"
    return None


def _selftest_case_ledger_suppression() -> str | None:
    series = _series_step(30_000, at=10_000, delta=0.17, sigma=0.25, seed=7)
    analysis = detect_changepoints(series)
    if not analysis.events:
        return "подавление реестром: ступень не обнаружена, подавлять нечего"
    ledger = Ledger(
        records=[
            LedgerRecord(
                index=index,
                step=int(event.step),
                direction=event.direction,
                owner="selftest",
                reason="встроенный кейс: известный сдвиг",
                ref="tools/check_loss_changepoint.py --selftest",
                review_by="2027-01-31",
            )
            for index, event in enumerate(analysis.events)
        ]
    )
    _, unsuppressed = apply_ledger(analysis.events, ledger, DEFAULT_TOL)
    if unsuppressed:
        return f"подавление реестром: осталось {len(unsuppressed)} неподавленных"
    return None


def run_selftest() -> int:
    """Прогнать обязательные кейсы (a)–(i); exit 0 только если все PASS."""

    # Шаговые кейсы используют 30 000 шагов (1/3 до сдвига, 2/3 после).
    # При равных половинах глобальная MAD впитывает саму ступень: для Δ=0.17
    # sigma_b ≈ 0.74·Δ, и z сверху ограничен 2.70 < 3.5 — по определению
    # ALGORITHM такая ступень недетектируема.  Асимметрия ряда даёт sigma_b
    # из «фоновой» части и сохраняет требование «ступень +0.17 на шаге 10000».
    cases: list[tuple[str, Any]] = [
        (
            "a_stationary_noise_no_events",
            lambda: _case_expect_no_events(
                "стационарный шум σ=0.25", _series_stationary(20_000, 0.25, seed=2)
            ),
        ),
        (
            "b_linear_trend_no_events",
            lambda: _case_expect_no_events(
                "линейный тренд -0.3", _series_trend(20_000, -0.3, 0.25, seed=2)
            ),
        ),
        (
            "c_step_up_detected",
            lambda: _case_step_detected(
                "ступень +0.17 на 10000",
                _series_step(30_000, at=10_000, delta=0.17, sigma=0.25, seed=7),
                expected_step=10_000,
                expected_direction="up",
                tolerance=150,
            ),
        ),
        (
            "d_step_down_direction",
            lambda: _case_step_detected(
                "ступень -0.17 на 10000",
                _series_step(30_000, at=10_000, delta=-0.17, sigma=0.25, seed=12),
                expected_step=10_000,
                expected_direction="down",
                tolerance=150,
            ),
        ),
        (
            "e_heavy_tails_no_events",
            lambda: _case_expect_no_events(
                "тяжёлые хвосты t3", _series_student_t3(20_000, 0.25, seed=2)
            ),
        ),
        ("f_nan_exit2", _selftest_case_nan_exit2),
        ("g_duplicates_and_gap", _selftest_case_duplicates_and_gap),
        (
            "h_non_multiple_step_detected",
            lambda: _case_step_detected(
                "ступень +0.17 на 10013 (некратный блок)",
                _series_step(30_000, at=10_013, delta=0.17, sigma=0.25, seed=7),
                expected_step=10_013,
                expected_direction="up",
                tolerance=150,
            ),
        ),
        ("i_ledger_suppression_exit0", _selftest_case_ledger_suppression),
    ]

    failed = 0
    for name, case in cases:
        try:
            problem = case()
        except Exception as exc:  # кейс не должен ронять весь selftest
            problem = f"{type(exc).__name__}: {exc}"
        if problem is None:
            print(f"PASS {name}")
        else:
            failed += 1
            print(f"FAIL {name}: {problem}")

    print(
        "SELFTEST: PASS" if failed == 0 else f"SELFTEST: FAIL ({failed} из {len(cases)})"
    )
    return EXIT_OK if failed == 0 else EXIT_ALERT


if __name__ == "__main__":
    sys.exit(main())
