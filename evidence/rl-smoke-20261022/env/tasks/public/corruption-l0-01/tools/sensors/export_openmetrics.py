"""Экспорт последних фактов в текстовый формат OpenMetrics (ADR-038, дельта G4).

Протокол фактов (ADR-037) совместим с внешними системами наблюдаемости без
зависимости от них: последний факт каждого датчика становится gauge'ем, а пин
предмета и датчик — метками. Формат — OpenMetrics (Prometheus text). Файл
``stdlib-only``; selftest разбирает собственный вывод.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from .fact import DEFAULT_FACTS_DIR, read_records

#: Имя метрики-контейнера факта.
METRIC_NAME = "axiom_fact"

#: Поля пина предмета, попадающие в метки (остальные не размножаем).
SUBJECT_LABEL_KEYS = ("git_sha", "config_sha256", "tokenizer_sha256", "checkpoint_sha256", "dataset_ref", "run_ref", "device_kind")

_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


def _sanitize_label(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def latest_records(facts_dir: str | Path | None = None, sensors: Optional[Iterable[str]] = None) -> list[dict[str, Any]]:
    """Последняя запись каждого факта каждого датчика (только с числовым значением)."""
    base = Path(facts_dir) if facts_dir is not None else DEFAULT_FACTS_DIR
    if not base.is_dir():
        return []
    wanted = set(sensors) if sensors else None
    out: list[dict[str, Any]] = []
    for path in sorted(base.glob("S-*.jsonl")):
        sensor = path.stem
        if wanted is not None and sensor not in wanted:
            continue
        try:
            records = read_records(sensor, base)
        except Exception:  # noqa: BLE001 — битый файл пропускается, не роняет экспорт
            continue
        last_by_fact: dict[str, dict[str, Any]] = {}
        for record in records:
            last_by_fact[str(record.get("fact"))] = record
        for record in last_by_fact.values():
            value = record.get("value")
            if record.get("status") != "ok" or not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            out.append(record)
    return out


def _labels(record: dict[str, Any]) -> list[tuple[str, str]]:
    labels: list[tuple[str, str]] = [
        ("sensor", str(record.get("sensor"))),
        ("fact", str(record.get("fact"))),
    ]
    subject = record.get("subject") or {}
    for key in SUBJECT_LABEL_KEYS:
        value = subject.get(key)
        if value not in (None, "", [], {}):
            labels.append((key, _sanitize_label(value)))
    quality = record.get("quality")
    if quality:
        labels.append(("quality", str(quality)))
    return labels


def render(records: list[dict[str, Any]], *, timestamp: Optional[str] = None) -> str:
    """Текстовый формат OpenMetrics: шапка, gauge-строки, ``# EOF``."""
    lines = [
        "# HELP axiom_fact Последний факт поведенческого слоя Spine (ADR-037/ADR-038)",
        "# TYPE axiom_fact gauge",
    ]
    ts_ms: Optional[int] = None
    if timestamp:
        try:
            parsed = datetime.fromisoformat(timestamp)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            ts_ms = int(parsed.timestamp() * 1000)
        except ValueError:
            ts_ms = None
    for record in sorted(records, key=lambda r: (str(r.get("sensor")), str(r.get("fact")))):
        value = record.get("value")
        if record.get("status") != "ok" or not isinstance(value, (int, float)) or isinstance(value, bool):
            continue  # gauge — только подтверждённое числовое значение
        label_text = ",".join(f'{key}="{value}"' for key, value in _labels(record))
        line = f"{METRIC_NAME}{{{label_text}}} {value}"
        if ts_ms is not None:
            line += f" {ts_ms}"
        lines.append(line)
    lines.append("# EOF")
    return "\n".join(lines) + "\n"


def parse(text: str) -> list[dict[str, Any]]:
    """Разбор OpenMetrics-текста обратно в сэмплы (для selftest и приёмки)."""
    samples: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        head, _, tail = line.partition(" ")
        if not head.startswith(METRIC_NAME):
            continue
        name, _, label_text = head.partition("{")
        labels = {m.group(1): m.group(2) for m in _LABEL_RE.finditer(label_text)}
        parts = tail.split()
        try:
            value = float(parts[0]) if parts else None
        except (ValueError, TypeError):
            continue  # нечисловая строка gauge'ем не является — пропускаем
        samples.append({"metric": name, "labels": labels, "value": value})
    return samples


def run_selftest() -> int:
    """Мутанты: разбор собственного вывода, пропуск нечисловых/не-ok фактов."""
    checks: list[tuple[str, bool]] = []
    records = [
        {"sensor": "S-001", "fact": "num_kda_layers", "value": 18, "status": "ok", "quality": "measured",
         "subject": {"config_sha256": "ab" * 32, "run_ref": None}},
        {"sensor": "S-012", "fact": "tok_s_median_window", "value": 800.5, "status": "ok", "quality": "wrapped",
         "subject": {"run_ref": "kda-wyut-delta", "config_sha256": None}},
        {"sensor": "S-012", "fact": "loss_median_window", "value": None, "status": "unverified", "quality": "wrapped",
         "subject": {}},
    ]
    text = render(records, timestamp="2026-10-07T12:00:00+03:00")
    parsed = parse(text)
    checks.append(("разбор собственного вывода", len(parsed) == 2))
    checks.append(("значения сохранены", {s["value"] for s in parsed} == {18.0, 800.5}))
    checks.append(("метки датчика и факта на месте",
                   all("sensor" in s["labels"] and "fact" in s["labels"] for s in parsed)))
    checks.append(("пин предмета стал меткой",
                   any(s["labels"].get("run_ref") == "kda-wyut-delta" for s in parsed)))
    checks.append(("нечисловой факт не экспортируется", not any(s["labels"].get("fact") == "loss_median_window" for s in parsed)))
    checks.append(("конец потока # EOF", text.strip().endswith("# EOF")))
    ok = all(p for _, p in checks)
    for label, passed in checks:
        print(f"[selftest] {'PASS' if passed else 'FAIL'}: {label}")
    print(f"[selftest] {'PASS' if ok else 'FAIL'}: export_openmetrics")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Экспорт последних фактов в OpenMetrics (ADR-038)")
    parser.add_argument("--facts-dir", default=None, help="каталог фактов (evidence/facts)")
    parser.add_argument("--sensor", action="append", default=None, help="ограничить датчиком (можно несколько)")
    parser.add_argument("--out", default=None, help="файл вывода (по умолчанию stdout)")
    parser.add_argument("--selftest", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.selftest:
        return run_selftest()
    records = latest_records(args.facts_dir, args.sensor)
    text = render(records)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
