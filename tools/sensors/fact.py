"""Контракт записи факта (ADR-036, дельта C1).

Запись факта — одна строка JSONL в ``evidence/facts/<S-id>.jsonl``: что
измерено, значение, единицы, когда, каким методом и **на каком предмете**
(:mod:`tools.sensors.subject`). Файл — только дозапись; каждая запись несёт
``prev_sha256`` предыдущей строки, поэтому правка истории обнаружима (цепочка).

Качество факта: ``measured`` (прямой замер), ``derived`` (вычислен из других
фактов, со ссылками в ``inputs``), ``wrapped`` (переупаковка выхода
существующего инструмента). Статус: ``ok`` | ``unverified`` — датчик, который
не может измерить, пишет ``unverified`` с причиной в ``note``, а не заглушку
(C-007). Вердикта здесь нет: его выносит предикат утверждения.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from .subject import REPO_ROOT, subject_is_pinned

#: Каталог фактов по умолчанию.
DEFAULT_FACTS_DIR = REPO_ROOT / "evidence" / "facts"

#: Обязательные ключи записи факта (контракт C1).
FACT_KEYS: tuple[str, ...] = (
    "sensor",
    "fact",
    "value",
    "unit",
    "quality",
    "method",
    "ts",
    "subject",
    "inputs",
    "status",
    "note",
    "prev_sha256",
)

#: Классы качества факта.
QUALITIES: tuple[str, ...] = ("measured", "derived", "wrapped")

#: Статусы записи.
STATUSES: tuple[str, ...] = ("ok", "unverified")


class FactError(ValueError):
    """Запись факта нарушает контракт (пишется честно, а не молча)."""


def now_iso() -> str:
    """Текущее время с таймзоной (ISO-8601, секунды)."""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def canonical(record: dict[str, Any]) -> str:
    """Канонический вид записи: компактный JSON, ключи отсортированы."""
    return json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def line_sha256(raw: str) -> str:
    """sha256 текста строки (включая завершающий перевод строки)."""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def facts_path(sensor: str, out_dir: str | Path | None = None) -> Path:
    """Путь файла фактов датчика ``<out_dir>/<sensor>.jsonl``."""
    base = Path(out_dir) if out_dir is not None else DEFAULT_FACTS_DIR
    return base / f"{sensor}.jsonl"


def _existing_raw_lines(path: Path) -> list[str]:
    """Строки файла с переводами строк (пустые игнорируются)."""
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")
    return [ln for ln in text.splitlines(keepends=True) if ln.strip()]


def _last_raw_line(path: Path) -> Optional[str]:
    lines = _existing_raw_lines(path)
    return lines[-1] if lines else None


def _validate(
    sensor: str,
    fact: str,
    value: Any,
    unit: str,
    quality: str,
    method: str,
    subject: Any,
    inputs: Any,
    status: str,
    note: str,
) -> None:
    if not isinstance(sensor, str) or not sensor.strip():
        raise FactError("sensor: непустая строка обязательна")
    if not isinstance(fact, str) or not fact.strip():
        raise FactError("fact: непустая строка обязательна")
    if quality not in QUALITIES:
        raise FactError(f"quality: {quality!r} вне {QUALITIES}")
    if status not in STATUSES:
        raise FactError(f"status: {status!r} вне {STATUSES}")
    if not isinstance(method, str) or not method.strip():
        raise FactError("method: непустая строка обязательна (чем измерено)")
    if not isinstance(unit, str):
        raise FactError("unit: строка (пустая допустима)")
    if not subject_is_pinned(subject):
        raise FactError(
            "subject: пин предмета пуст — факт не привязывается к утверждению "
            "(ADR-036); неизвестные поля пишутся как null"
        )
    if inputs is not None and not isinstance(inputs, list):
        raise FactError("inputs: список ссылок на исходные записи")
    if status == "ok" and value is None:
        raise FactError("status=ok: значение обязательно (None = unverified)")
    if status == "unverified" and not (note or "").strip():
        raise FactError("status=unverified: нужна причина в note (не молчать)")


def write_fact(
    sensor: str,
    fact: str,
    value: Any,
    *,
    unit: str,
    quality: str,
    method: str,
    subject: Any,
    out_dir: str | Path | None = None,
    inputs: Optional[list[dict[str, Any]]] = None,
    status: str = "ok",
    note: str = "",
    ts: Optional[str] = None,
) -> dict[str, Any]:
    """Дописывает одну запись факта и возвращает её.

    ``prev_sha256`` — sha256 предыдущей строки того же файла (``None`` для
    первой), цепочка делает правку истории обнаружимой. Только дозапись:
    существующие строки не переписываются.
    """
    _validate(sensor, fact, value, unit, quality, method, subject, inputs, status, note)

    path = facts_path(sensor, out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    prev = _last_raw_line(path)
    record: dict[str, Any] = {
        "sensor": sensor,
        "fact": fact,
        "value": value,
        "unit": unit,
        "quality": quality,
        "method": method,
        "ts": ts or now_iso(),
        "subject": subject,
        "inputs": list(inputs or []),
        "status": status,
        "note": note,
        "prev_sha256": line_sha256(prev) if prev is not None else None,
    }
    raw = canonical(record) + "\n"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(raw)
    return record


def read_records(
    sensor: str, out_dir: str | Path | None = None
) -> list[dict[str, Any]]:
    """Все записи факта датчика (разобранные), в порядке файла.

    Битую строку молча не глотаем: выбрасывается :class:`FactError` — реестр
    обязан её увидеть, а не «потерять».
    """
    path = facts_path(sensor, out_dir)
    records: list[dict[str, Any]] = []
    for lineno, raw in enumerate(_existing_raw_lines(path), start=1):
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise FactError(f"{path}:{lineno}: строка не JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise FactError(f"{path}:{lineno}: строка не объект JSON")
        record["_lineno"] = lineno
        record["_raw"] = raw
        records.append(record)
    return records


def verify_chain(sensor: str, out_dir: str | Path | None = None) -> tuple[bool, str]:
    """Проверка цепочки ``prev_sha256``. Возвращает ``(ok, причина)``."""
    path = facts_path(sensor, out_dir)
    lines = _existing_raw_lines(path)
    if not lines:
        return False, f"{path}: нет записей"
    try:
        records = read_records(sensor, out_dir)
    except FactError as exc:
        return False, str(exc)
    for index, record in enumerate(records):
        expected = None if index == 0 else line_sha256(lines[index - 1])
        actual = record.get("prev_sha256")
        if actual != expected:
            return (
                False,
                f"{path}:{index + 1}: prev_sha256={actual!r}, ожидалось {expected!r} "
                "(цепочка разорвана — история правилась)",
            )
    return True, f"{path}: цепочка цела ({len(records)} записей)"


def load_contract_check(sensor: str, out_dir: str | Path | None = None) -> tuple[bool, str, Optional[dict[str, Any]]]:
    """Проверяет последнюю запись по контракту. ``(ok, причина, запись)``."""
    path = facts_path(sensor, out_dir)
    if not path.is_file():
        return False, f"{path}: файл фактов отсутствует", None
    try:
        records = read_records(sensor, out_dir)
    except FactError as exc:
        return False, str(exc), None
    if not records:
        return False, f"{path}: файл пуст", None
    last = records[-1]
    for key in FACT_KEYS:
        if key not in last:
            return False, f"{path}: в записи нет поля {key!r}", last
    if last.get("quality") not in QUALITIES:
        return False, f"{path}: quality={last.get('quality')!r} вне {QUALITIES}", last
    if last.get("status") not in STATUSES:
        return False, f"{path}: status={last.get('status')!r} вне {STATUSES}", last
    if not subject_is_pinned(last.get("subject")):
        return False, f"{path}: запись без пина предмета (subject пуст)", last
    return True, f"{path}: последняя запись по контракту", last


def read_latest(
    sensor: str,
    fact: str,
    subject_filter: Any = None,
    *,
    out_dir: str | Path | None = None,
    subject: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Последняя запись факта ``fact`` датчика ``sensor``.

    ``subject_filter`` — какие поля пина обязаны совпасть: список имён полей
    (тогда сравнивается с ``subject``) или словарь ``{поле: значение}`` (прямое
    сравнение). ``None`` — фильтра нет. Несовпадение предмета — запись не
    подходит (лучше ``unverified``, чем ложная привязка).
    """
    try:
        records = read_records(sensor, out_dir)
    except FactError:
        return None

    def matches(record: dict[str, Any]) -> bool:
        if record.get("fact") != fact:
            return False
        if not subject_filter:
            return True
        rec_subject = record.get("subject") or {}
        if isinstance(subject_filter, dict):
            items = subject_filter.items()
            return all(rec_subject.get(k) == v for k, v in items)
        if isinstance(subject_filter, (list, tuple)):
            if subject is None:
                return True
            for key in subject_filter:
                ref_value = (subject or {}).get(key)
                if ref_value is None:
                    continue  # неизвестный конец фильтра не объявляем несовпадением
                if rec_subject.get(key) != ref_value:
                    return False
            return True
        return True

    for record in reversed(records):
        if matches(record):
            return record
    return None


def latest_ts(records: Iterable[dict[str, Any]]) -> Optional[str]:
    """Последний непустой ``ts`` среди записей."""
    last: Optional[str] = None
    for record in records:
        ts = record.get("ts")
        if isinstance(ts, str) and ts:
            last = ts
    return last


def age_hours(ts: str, now: Optional[datetime] = None) -> Optional[float]:
    """Возраст метки времени в часах; ``None``, если метка не разбирается."""
    try:
        parsed = datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    moment = now or datetime.now(timezone.utc)
    return (moment - parsed).total_seconds() / 3600.0
