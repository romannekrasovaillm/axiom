"""Протокол экспортёра датчиков (ADR-038, дельта G).

Один интерфейс для всех датчиков и доменных пакетов (по образцу
Prometheus/OpenTelemetry): экспортёр объявляет себя (:meth:`describe` →
:class:`SensorSpec`) и собирает факты (:meth:`collect` → ``list[Fact]``).
Тест соответствия протоколу (:func:`check_conformance`) прогоняется по всем
экспортёрам: ``describe`` валиден, ``collect`` возвращает записи по контракту,
а при недоступном источнике — ``unverified``, не исключение.

Ядро кейса (S-001…S-030) остаётся на прямой регистрации в ``model/sensors.yaml``
(``pack: ml``); новые экспортёры живут в ``tools/sensors/packs/<domain>/`` и
подхватываются :func:`tools.sensors.packs.discover_exporters`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Protocol, runtime_checkable

from .fact import FACT_KEYS, QUALITIES, STATUSES, validate_raw_ref

#: Уровни измерения (дельта L, ADR-038).
LEVELS: tuple[str, ...] = ("end_to_end", "component", "diagnostic")

#: Допустимые качества факта (переиспользуем контракт ядра).
_QUALITIES = QUALITIES
_STATUSES = STATUSES


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


@dataclass(frozen=True)
class SensorSpec:
    """Паспорт экспортёра: что он умеет, из чего и с какой свежестью."""

    id: str
    facts: tuple[str, ...]
    #: факт → {unit, quality, level}
    schema: dict[str, dict[str, Any]]
    level: str = "diagnostic"
    raw: Optional[dict[str, Any]] = None
    freshness_max_h: Optional[float] = None
    context: tuple[str, ...] = ()
    pack: str = ""

    def validate(self) -> list[str]:
        """Схема спецификации: список нарушений (пусто = валидно)."""
        errs: list[str] = []
        if not isinstance(self.id, str) or not self.id.strip():
            errs.append("describe.id: непустая строка обязательна")
        if not self.facts or not all(isinstance(f, str) and f for f in self.facts):
            errs.append("describe.facts: непустой список непустых имён")
        if self.level not in LEVELS:
            errs.append(f"describe.level={self.level!r} вне {LEVELS}")
        if not isinstance(self.schema, dict) or set(self.schema) != set(self.facts):
            errs.append("describe.schema: ключи обязаны совпасть с facts")
        for fact, meta in (self.schema or {}).items():
            if not isinstance(meta, dict):
                errs.append(f"describe.schema[{fact}]: объект обязателен")
                continue
            if meta.get("quality") not in _QUALITIES:
                errs.append(f"describe.schema[{fact}].quality вне {_QUALITIES}")
            if meta.get("level", self.level) not in LEVELS:
                errs.append(f"describe.schema[{fact}].level вне {LEVELS}")
            if "unit" not in meta:
                errs.append(f"describe.schema[{fact}].unit обязателен (пустая строка допустима)")
        if self.raw is not None:
            if not isinstance(self.raw, dict):
                errs.append("describe.raw: объект или null")
            elif "path" in self.raw:
                if not isinstance(self.raw["path"], str) or not self.raw["path"].strip():
                    errs.append("describe.raw.path: непустая строка")
            elif not self.raw.get("note"):
                errs.append("describe.raw: либо path, либо note")
        if self.freshness_max_h is not None:
            if isinstance(self.freshness_max_h, bool) or not isinstance(self.freshness_max_h, (int, float)):
                errs.append("describe.freshness_max_h: число или null")
            elif self.freshness_max_h <= 0:
                errs.append("describe.freshness_max_h: > 0")
        return errs

    def fact_meta(self, fact: str) -> dict[str, Any]:
        meta = dict(self.schema.get(fact, {}))
        meta.setdefault("quality", "measured")
        meta.setdefault("level", self.level)
        meta.setdefault("unit", "")
        return meta


@dataclass(frozen=True)
class Fact:
    """Запись факта в форме контракта C1 (см. :mod:`tools.sensors.fact`)."""

    sensor: str
    fact: str
    value: Any
    unit: str = ""
    quality: str = "measured"
    method: str = ""
    subject: dict[str, Any] = field(default_factory=dict)
    inputs: tuple[dict[str, Any], ...] = ()
    status: str = "ok"
    note: str = ""
    level: str = "diagnostic"
    raw_ref: Optional[dict[str, Any]] = None

    def to_record(self, ts: Optional[str] = None) -> dict[str, Any]:
        return {
            "sensor": self.sensor,
            "fact": self.fact,
            "value": self.value,
            "unit": self.unit,
            "quality": self.quality,
            "method": self.method,
            "ts": ts or _now_iso(),
            "subject": self.subject,
            "inputs": list(self.inputs),
            "status": self.status,
            "note": self.note,
            "prev_sha256": None,
            "raw_ref": self.raw_ref,
        }

    def validate(self) -> list[str]:
        errs: list[str] = []
        if self.quality not in _QUALITIES:
            errs.append(f"quality={self.quality!r} вне {_QUALITIES}")
        if self.status not in _STATUSES:
            errs.append(f"status={self.status!r} вне {_STATUSES}")
        if self.level not in LEVELS:
            errs.append(f"level={self.level!r} вне {LEVELS}")
        if self.status == "ok" and self.value is None:
            errs.append("status=ok без значения")
        if self.status == "unverified" and not self.note.strip():
            errs.append("status=unverified без причины в note")
        errs.extend(validate_raw_ref(self.raw_ref))
        return errs

    def write(self, out_dir: Any = None) -> dict[str, Any]:
        from .fact import write_fact

        return write_fact(
            self.sensor, self.fact, self.value, unit=self.unit, quality=self.quality,
            method=self.method, subject=self.subject, out_dir=out_dir,
            inputs=list(self.inputs), status=self.status, note=self.note,
            raw_ref=self.raw_ref,
        )


@runtime_checkable
class Exporter(Protocol):
    """Интерфейс экспортёра: ``describe()`` и ``collect(subject, **inputs)``."""

    def describe(self) -> SensorSpec:  # pragma: no cover — протокол
        ...

    def collect(self, subject: dict[str, Any], **inputs: Any) -> list[Fact]:  # pragma: no cover
        ...


class BaseExporter:
    """База экспортёра: ``describe`` из :attr:`SPEC` и сборка ``Fact`` по схеме."""

    SPEC: SensorSpec

    def describe(self) -> SensorSpec:  # noqa: D102
        return self.SPEC

    def fact(
        self,
        fact: str,
        value: Any,
        *,
        subject: dict[str, Any],
        status: str = "ok",
        note: str = "",
        method: str = "",
        raw_ref: Optional[dict[str, Any]] = None,
        inputs: Optional[list[dict[str, Any]]] = None,
        unit: Optional[str] = None,
    ) -> Fact:
        meta = self.SPEC.fact_meta(fact)
        return Fact(
            sensor=self.SPEC.id, fact=fact, value=value,
            unit=meta["unit"] if unit is None else unit,
            quality=meta["quality"], level=meta["level"],
            method=method or f"экспортёр {self.SPEC.id}",
            subject=subject, inputs=tuple(inputs or []),
            status=status, note=note, raw_ref=raw_ref,
        )

    def unavailable(self, fact: str, subject: dict[str, Any], reason: str) -> Fact:
        return self.fact(fact, None, subject=subject, status="unverified", note=reason)


def collect_all(exporter: Exporter, subject: dict[str, Any], **inputs: Any) -> list[Fact]:
    """Собирает факты экспортёра, гарантируя список ``Fact`` (не исключение)."""
    try:
        facts = exporter.collect(subject, **inputs)
    except Exception as exc:  # noqa: BLE001 — недоступный источник = unverified, не падение
        spec = exporter.describe()
        return [
            Fact(sensor=spec.id, fact=f, value=None,
                 unit=spec.fact_meta(f)["unit"], quality=spec.fact_meta(f)["quality"],
                 level=spec.fact_meta(f)["level"], method=f"экспортёр {spec.id}",
                 subject=subject, status="unverified",
                 note=f"исключение при сборе: {type(exc).__name__}: {exc}")
            for f in spec.facts
        ]
    return [f for f in facts if isinstance(f, Fact)]


def check_conformance(exporters: list[Exporter], *, subject: Optional[dict[str, Any]] = None) -> list[str]:
    """Тест соответствия протоколу. Возвращает список нарушений (пусто = OK)."""
    errs: list[str] = []
    ref_subject = subject or {"git_sha": "0" * 40, "git_dirty": False, "device_kind": "cpu"}
    seen: set[str] = set()
    for exporter in exporters:
        try:
            spec = exporter.describe()
        except Exception as exc:  # noqa: BLE001
            errs.append(f"{exporter!r}: describe() упал — {type(exc).__name__}: {exc}")
            continue
        if not isinstance(spec, SensorSpec):
            errs.append(f"{spec!r}: describe() вернул не SensorSpec")
            continue
        errs.extend(f"{spec.id}: {e}" for e in spec.validate())
        if spec.id in seen:
            errs.append(f"{spec.id}: дубль id среди экспортёров")
        seen.add(spec.id)

        facts = collect_all(exporter, ref_subject)
        if not isinstance(facts, list):
            errs.append(f"{spec.id}: collect() вернул не список")
            continue
        returned = {f.fact for f in facts}
        missing = set(spec.facts) - returned
        if missing:
            errs.append(f"{spec.id}: collect() не вернул объявленные факты {sorted(missing)}")
        for fact in facts:
            if fact.sensor != spec.id:
                errs.append(f"{spec.id}: факт {fact.fact} с чужим sensor={fact.sensor}")
            for e in fact.validate():
                errs.append(f"{spec.id}/{fact.fact}: {e}")
            record = fact.to_record()
            if not set(FACT_KEYS) <= set(record):
                errs.append(f"{spec.id}/{fact.fact}: запись не по контракту C1")
    return errs


__all__ = [
    "LEVELS",
    "SensorSpec",
    "Fact",
    "Exporter",
    "BaseExporter",
    "collect_all",
    "check_conformance",
]
