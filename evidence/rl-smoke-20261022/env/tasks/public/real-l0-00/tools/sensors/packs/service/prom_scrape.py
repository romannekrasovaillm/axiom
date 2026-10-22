"""Экспортёр SVC-002: экспозиция Prometheus (дельта G3, ADR-038).

Читает текстовый формат экспозиции Prometheus: квантили задержки из гистограммы
(``name_bucket{le=...}`` + ``name_count``) и долю ошибок по метке ``code``.
Без сети и живых сервисов — только файл.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from tools.sensors.protocol import BaseExporter, Fact, SensorSpec

_SAMPLE_RE = re.compile(r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?P<labels>\{.*\})?\s+(?P<value>[0-9eE.+-]+)\s*$")
_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')
_ERROR_CODE_RE = re.compile(r"^[45]\d\d$")


def _parse_labels(text: Optional[str]) -> dict[str, str]:
    if not text:
        return {}
    return {m.group(1): m.group(2) for m in _LABEL_RE.finditer(text)}


def _quantile(buckets: list[tuple[float, float]], total: float, q: float) -> Optional[float]:
    """Квантиль гистограммы линейной интерполяцией по кумулятивным бакетам."""
    if not buckets or total <= 0:
        return None
    target = q * total
    prev_le = 0.0
    prev_cum = 0.0
    for le, cum in buckets:
        if cum >= target:
            if cum == prev_cum:
                return le
            frac = (target - prev_cum) / (cum - prev_cum)
            return prev_le + (le - prev_le) * frac
        prev_le, prev_cum = le, cum
    return buckets[-1][0]


class PromScrapeExporter(BaseExporter):
    SPEC = SensorSpec(
        id="SVC-002",
        facts=("latency_p50_seconds", "latency_p99_seconds", "error_share"),
        schema={
            "latency_p50_seconds": {"unit": "seconds", "quality": "wrapped", "level": "end_to_end"},
            "latency_p99_seconds": {"unit": "seconds", "quality": "wrapped", "level": "end_to_end"},
            "error_share": {"unit": "fraction", "quality": "wrapped", "level": "end_to_end"},
        },
        level="end_to_end",
        raw={"path": "<prometheus-exposition.txt>", "format": "text", "retention_days": None, "in_git": False},
        context=("repo",),
        pack="service",
    )

    def collect(self, subject: dict[str, Any], *, input_path: Any = None, histogram: str = "", **_: Any) -> list[Fact]:
        if not input_path:
            return [self.unavailable(f, subject, "не передан файл экспозиции (input_path)") for f in self.SPEC.facts]
        path = Path(input_path)
        if not path.is_file():
            return [self.unavailable(f, subject, f"файл экспозиции не найден: {path}") for f in self.SPEC.facts]

        buckets: dict[str, list[tuple[float, float]]] = {}
        counts: dict[str, float] = {}
        error_sum = 0.0
        code_total = 0.0

        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            m = _SAMPLE_RE.match(line)
            if not m:
                continue
            name = m.group("name")
            labels = _parse_labels(m.group("labels"))
            value = float(m.group("value"))
            if name.endswith("_bucket") and "le" in labels:
                try:
                    le = float(labels["le"])
                except ValueError:
                    continue
                base = name[: -len("_bucket")]
                buckets.setdefault(base, []).append((le, value))
            elif name.endswith("_count"):
                counts[name[: -len("_count")]] = value
            elif "code" in labels:
                code_total += value
                if _ERROR_CODE_RE.match(labels["code"]):
                    error_sum += value

        base = histogram or (sorted(buckets)[0] if buckets else "")
        series = sorted(buckets.get(base, []))
        total = counts.get(base)
        if total is None:
            total = max((c for _, c in series), default=0.0)
        p50 = _quantile(series, total, 0.5)
        p99 = _quantile(series, total, 0.99)

        facts: list[Fact] = []
        if p50 is None or p99 is None:
            reason = f"гистограмма задержки не найдена (base={base!r})"
            facts.extend(self.unavailable(f, subject, reason) for f in self.SPEC.facts[:2])
        else:
            facts.append(self.fact("latency_p50_seconds", round(p50, 6), subject=subject, method=f"гистограмма {base} из {path.name}"))
            facts.append(self.fact("latency_p99_seconds", round(p99, 6), subject=subject, method=f"гистограмма {base} из {path.name}"))
        if code_total <= 0:
            facts.append(self.unavailable("error_share", subject, "нет счётчиков с меткой code"))
        else:
            facts.append(self.fact("error_share", round(error_sum / code_total, 6), subject=subject, method=f"сумма по code из {path.name}"))
        return facts


EXPORTER = PromScrapeExporter()
