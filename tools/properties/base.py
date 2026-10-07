"""Каталог свойств-шаблонов (ADR-039, дельта P1).

Поведенческая проверка — это пара «шаблон свойства + привязка»: шаблон верен для
любой системы (границы, идентичность, сохранение, детерминизм, …), привязка —
доменна (какой факт, какой порог, чей порог). Здесь живёт универсальная часть:
интерфейс :class:`Property`, читатель фактов :class:`Facts` с наложением
(overlay) для мутантов и :class:`Verdict` (`pass | fail | unverified`).

Инварианты каталога:

* отсутствие или недостаточность фактов даёт ``unverified`` с причиной — никогда
  не ``pass`` («нет метрик ≠ норма», E-3.5);
* мутанты подменяют записи фактов **в памяти** (overlay) и никогда не пишут в
  ``evidence/``;
* окна, допуски, условия замера и уровни берутся из ADR-037/ADR-039, а не
  зашиваются в шаблон.

Реестр описаний — ``model/properties.yaml``; привязки правил-стражей —
``model/rule-properties.yaml``; кандидаты — ``model/candidates.yaml``.
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]

_SPEC_RE = re.compile(r"^(S-\d+)\.([A-Za-z0-9_]+)$")

#: Допустимые уровни фактов (ADR-038, дельта L).
LEVELS: tuple[str, ...] = ("end_to_end", "component", "diagnostic")

#: Классы вердикта (ADR-037).
VERDICT_CLASSES: tuple[str, ...] = ("pass", "fail", "unverified")


class PropertyError(ValueError):
    """Шаблон или его параметры нарушают контракт (не угадывается)."""


# ── вердикт и мутант ────────────────────────────────────────────────────────


@dataclass
class Verdict:
    """Исход применения свойства: класс + причина + ссылки на записи фактов."""

    cls: str
    reason: str = ""
    evidence: list = field(default_factory=list)
    value: Any = None

    def __post_init__(self) -> None:
        if self.cls not in VERDICT_CLASSES:
            raise PropertyError(f"класс вердикта {self.cls!r} вне {VERDICT_CLASSES}")

    @classmethod
    def passed(cls, reason: str = "", evidence: Optional[list] = None, value: Any = None) -> "Verdict":
        return cls("pass", reason, list(evidence or []), value)

    @classmethod
    def failed(cls, reason: str = "", evidence: Optional[list] = None, value: Any = None) -> "Verdict":
        return cls("fail", reason, list(evidence or []), value)

    @classmethod
    def unverified(cls, reason: str = "", evidence: Optional[list] = None) -> "Verdict":
        return cls("unverified", reason, list(evidence or []), None)


@dataclass
class Mutant:
    """Оператор мутации: подмена записей фактов в памяти.

    ``overlay`` — ``{spec: [записи факта]}``: полностью заменяет набор записей
    факта на время прогона. Неэквивалентный мутант обязан убиваться (вердикт
    ``fail``); выживший — находка мутационного тестирования (дельта M).
    """

    fingerprint: str
    description: str
    overlay: dict
    expected: str = "fail"  # какой класс ожидается у убитого мутанта

    def apply(self, facts: "Facts") -> "Facts":
        return facts.with_overlay(self.overlay)


def mutant_fingerprint(property_name: str, params: dict, description: str) -> str:
    """Отпечаток мутанта: sha256(шаблон + канонизированные параметры + описание)."""
    payload = json.dumps(
        {"property": property_name, "params": canonical_params(params), "mutation": description},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_params(params: Any) -> Any:
    """Канонизирует параметры для отпечатка/дедупликации (сортировка ключей)."""
    if isinstance(params, dict):
        return {str(k): canonical_params(v) for k, v in sorted(params.items())}
    if isinstance(params, list):
        return [canonical_params(v) for v in params]
    return params


def property_fingerprint(property_name: str, params: dict) -> str:
    """Отпечаток кандидата: sha256(шаблон + канонизированные параметры) (дельта G)."""
    payload = json.dumps(
        {"property": property_name, "params": canonical_params(params)},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ── читатель фактов с наложением ────────────────────────────────────────────


def parse_spec(spec: Any) -> tuple[str, str]:
    """``"S-012.tok_s_median_window"`` → ``("S-012", "tok_s_median_window")``."""
    if not isinstance(spec, str):
        raise PropertyError(f"ссылка на факт должна быть строкой 'S-id.fact', получено {spec!r}")
    m = _SPEC_RE.match(spec.strip())
    if not m:
        raise PropertyError(f"ссылка на факт {spec!r} не вида 'S-id.fact'")
    return m.group(1), m.group(2)


def _ref_subject(root: Path) -> dict:
    try:
        from tools.sensors.subject import build_subject

        return build_subject(repo_root=root, config_path=root / "net" / "config.json", device="cpu")
    except Exception:  # noqa: BLE001 — без предмета фильтр не применяется, но факт читается
        return {}


class Facts:
    """Читатель фактов ``evidence/facts/*.jsonl`` с фильтром предмета и overlay.

    ``overlay`` — ``{spec: [записи]}``; записи overlay возвращаются как есть
    (после фильтра предмета) и никогда не пишутся на диск.
    """

    def __init__(
        self,
        root: str | Path,
        out_dir: str | Path | None = None,
        overlay: Optional[dict] = None,
        subject_match: Optional[list] = None,
        subject: Optional[dict] = None,
        now: Any = None,
    ) -> None:
        self.root = Path(root)
        self.out_dir = Path(out_dir) if out_dir else None
        self.overlay = dict(overlay or {})
        self.subject_match = list(subject_match or [])
        self.subject = subject if subject is not None else _ref_subject(self.root)
        self.now = now

    # -- чтение ----------------------------------------------------------
    def _raw(self, spec: str) -> list[dict]:
        if spec in self.overlay:
            return [dict(r) for r in self.overlay[spec]]
        sensor, fact = parse_spec(spec)
        try:
            from tools.sensors.fact import read_records

            records = read_records(sensor, self.out_dir)
        except Exception:  # noqa: BLE001 — битый/отсутствующий файл = нет факта (unverified)
            return []
        return [r for r in records if r.get("fact") == fact]

    def series(self, spec: str) -> list[dict]:
        """Записи факта ``spec`` с совпавшим предметом, в порядке файла."""
        out = []
        for record in self._raw(spec):
            if self._subject_ok(record):
                out.append(record)
        return out

    def _subject_ok(self, record: dict) -> bool:
        if not self.subject_match:
            return True
        rec = record.get("subject") or {}
        for key in self.subject_match:
            ref = (self.subject or {}).get(key)
            if ref is None:
                continue
            if rec.get(key) != ref:
                return False
        return True

    def latest(self, spec: str) -> Optional[dict]:
        recs = self.series(spec)
        return recs[-1] if recs else None

    def all_subjects(self, spec: str) -> list[dict]:
        return self._raw(spec)

    # -- мутация ---------------------------------------------------------
    def with_overlay(self, extra: dict) -> "Facts":
        merged = {**self.overlay, **extra}
        return Facts(self.root, self.out_dir, merged, self.subject_match, self.subject, self.now)

    def clone_last(self, spec: str, **changes: Any) -> dict:
        """Копия последней записи факта с правками (для мутантов)."""
        last = self.latest(spec)
        if last is None:
            raise PropertyError(f"нет записи факта {spec} для мутации")
        rec = {k: v for k, v in last.items() if not k.startswith("_")}
        rec.update(changes)
        return rec


def ok_value(record: Optional[dict]) -> tuple[Optional[Any], Optional[str]]:
    """``(значение, причина)`` для записи: причина непустая, если значение негодно."""
    if record is None:
        return None, "нет факта"
    if record.get("status") != "ok":
        return None, "факт unverified: " + str(record.get("note", ""))
    value = record.get("value")
    if value is None:
        return None, "значение отсутствует"
    return value, None


# ── окно и статистика ───────────────────────────────────────────────────────


def _numbers(records: list[dict]) -> list[float]:
    vals = []
    for r in records:
        v = r.get("value")
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            vals.append(float(v))
    return vals


def stat_value(values: list[float], stat: str) -> Optional[float]:
    if not values:
        return None
    if stat in ("last", None, ""):
        return values[-1]
    if stat == "median":
        return float(statistics.median(values))
    if stat == "mean":
        return sum(values) / len(values)
    if stat == "max":
        return max(values)
    if stat == "min":
        return min(values)
    if stat in ("p90", "p95"):
        ordered = sorted(values)
        idx = int(round((0.90 if stat == "p90" else 0.95) * (len(ordered) - 1)))
        return ordered[idx]
    raise PropertyError(f"неизвестная статистика окна: {stat!r}")


def windowed_value(records: list[dict], window: Any) -> Optional[float]:
    """Значение по окну ``{stat, n, skip_warmup}`` (или последнее значение)."""
    if not records:
        return None
    if not isinstance(window, dict):
        # Без окна — последнее значение (совпадает с предикатом ADR-037).
        return stat_value(_numbers(records[-1:]), "last")
    vals = _numbers(records)
    if not vals:
        return None
    skip = int(window.get("skip_warmup") or 0)
    if skip:
        vals = vals[skip:]
    n = window.get("n")
    if n:
        vals = vals[-int(n):]
    return stat_value(vals, str(window.get("stat") or "last"))


def age_hours(ts: Any, now: Any = None) -> Optional[float]:
    from datetime import datetime, timezone

    if not isinstance(ts, str):
        return None
    try:
        parsed = datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    moment = now or datetime.now(timezone.utc)
    return (moment - parsed).total_seconds() / 3600.0


# ── валидация параметров (подмножество JSON Schema) ─────────────────────────

_TYPES: dict[str, Any] = {
    "object": dict,
    "array": list,
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "null": type(None),
}


def _type_ok(expected: Any, value: Any) -> bool:
    types = expected if isinstance(expected, list) else [expected]
    for t in types:
        if t == "number" and isinstance(value, bool):
            continue
        py = _TYPES.get(t)
        if py is None:
            continue
        if isinstance(value, py):
            if t == "integer" and isinstance(value, bool):
                continue
            return True
    return False


def validate_params(schema: Any, params: Any, path: str = "params") -> list[str]:
    """Проверка ``params`` по подмножеству JSON Schema. Список нарушений."""
    errors: list[str] = []
    if not isinstance(schema, dict):
        return errors
    if "anyOf" in schema:
        branches = schema["anyOf"]
        if not any(not validate_params(b, params, path) for b in branches):
            errors.append(f"{path}: не подходит ни одна из схем anyOf")
        return errors
    expected = schema.get("type")
    if expected is not None and not _type_ok(expected, params):
        errors.append(f"{path}: ожидался тип {expected!r}, получено {type(params).__name__}")
        return errors
    if isinstance(params, dict):
        for req in schema.get("required") or []:
            if req not in params:
                errors.append(f"{path}.{req}: обязательное поле отсутствует")
        for key, sub in (schema.get("properties") or {}).items():
            if key in params:
                errors.extend(validate_params(sub, params[key], f"{path}.{key}"))
        if schema.get("additionalProperties") is False:
            known = set((schema.get("properties") or {}).keys())
            for key in params:
                if key not in known:
                    errors.append(f"{path}.{key}: неизвестное поле")
    if isinstance(params, list) and isinstance(schema.get("items"), dict):
        for i, item in enumerate(params):
            errors.extend(validate_params(schema["items"], item, f"{path}[{i}]"))
    enum = schema.get("enum")
    if enum is not None and params not in enum:
        errors.append(f"{path}: значение {params!r} вне {enum}")
    return errors


# ── базовый шаблон и реестр ─────────────────────────────────────────────────

_REGISTRY: dict[str, "Property"] = {}


class Property:
    """Шаблон поведенческого свойства.

    Наследники объявляют ``name``, ``param_schema``, ``levels`` и реализуют
    :meth:`evaluate` / :meth:`mutants`. Отсутствие фактов → ``unverified``.
    """

    name: str = ""
    param_schema: dict = {}
    levels: set = set(LEVELS)

    def evaluate(self, params: dict, facts: Facts) -> Verdict:  # pragma: no cover - интерфейс
        raise NotImplementedError

    def mutants(self, params: dict, facts: Facts) -> list[Mutant]:  # pragma: no cover - интерфейс
        return []

    # -- вспомогательное -------------------------------------------------
    def _mutant(self, params: dict, facts: Facts, spec: str, description: str,
                transform: Callable[[list[dict]], list[dict]], expected: str = "fail") -> Mutant:
        records = [dict(r) for r in facts.series(spec)]
        mutated = transform(records)
        # Мутант инъектирует измерение: запись с подменённым значением считается
        # снятой (status ok), иначе проверка не дойдёт до предиката и мутант уйдёт
        # в unverified независимо от того, ловит ли его шаблон.
        for record in mutated:
            if record.get("status") != "ok":
                record["status"] = "ok"
        return Mutant(
            fingerprint=mutant_fingerprint(self.name, params, description),
            description=f"{self.name}: {description}",
            overlay={spec: mutated},
            expected=expected,
        )


def register(cls: type) -> type:
    instance = cls()
    if not instance.name:
        raise PropertyError(f"{cls.__name__}: не задано имя шаблона")
    if instance.name in _REGISTRY:
        raise PropertyError(f"шаблон {instance.name!r} зарегистрирован дважды")
    _REGISTRY[instance.name] = instance
    return cls


def get_property(name: str) -> Optional[Property]:
    return _REGISTRY.get(name)


def all_properties() -> dict[str, Property]:
    return dict(_REGISTRY)


def load_properties_registry(root: str | Path) -> dict:
    """Читает ``model/properties.yaml`` как словарь (templates + equivalent_mutants)."""
    path = Path(root) / "model" / "properties.yaml"
    if not path.is_file():
        return {"templates": [], "equivalent_mutants": []}
    from tools.miniyaml import load_file

    data = load_file(path)
    if not isinstance(data, dict):
        raise PropertyError(f"{path}: ожидался объект с ключами templates/equivalent_mutants")
    return data
