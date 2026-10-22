"""Экспортёр S-035 noise_baseline: шум железа как источник допуска (ADR-038, M2).

Коэффициент вариации (CV) ряда ток/с между окнами и между прогонами и
рекомендованный допуск ``k·σ``. Допуск получает происхождение-факт: где нет
ряда — честное ``unverified``, а не подставленная константа. Источник —
факты S-012 (окна метрик претрейна).
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any, Optional

from tools.sensors.fact import read_records
from tools.sensors.protocol import BaseExporter, Fact, SensorSpec

#: Множитель сигмы для рекомендованного допуска (решение, не измерение).
K_SIGMA = 2.0
#: Длина окна для оценки шума.
WINDOW_N = 50


def _series(sensor: str, fact: str, out_dir: Any) -> list[float]:
    try:
        records = read_records(sensor, out_dir)
    except Exception:  # noqa: BLE001
        return []
    values = [
        float(r["value"]) for r in records
        if r.get("fact") == fact and r.get("status") == "ok"
        and isinstance(r.get("value"), (int, float)) and not isinstance(r.get("value"), bool)
    ]
    return values[-WINDOW_N:]


def _cv(values: list[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    mean = statistics.fmean(values)
    if mean == 0:
        return None
    return statistics.pstdev(values) / abs(mean)


def measure(out_dir: Any = None) -> dict[str, Any]:
    tok_s = _series("S-012", "tok_s_median_window", out_dir)
    step_s = _series("S-012", "step_seconds_median", out_dir)
    cv_tok = _cv(tok_s)
    cv_step = _cv(step_s)
    recommended = None
    if cv_tok is not None:
        recommended = round(K_SIGMA * cv_tok, 6)
    return {
        "cv_tok_s": cv_tok,
        "cv_step_seconds": cv_step,
        "recommended_tolerance": recommended,
        "window_n": WINDOW_N,
        "k_sigma": K_SIGMA,
    }


class NoiseBaselineExporter(BaseExporter):
    SPEC = SensorSpec(
        id="S-035",
        facts=("cv_tok_s", "cv_step_seconds", "recommended_tolerance", "window_n", "k_sigma"),
        schema={
            "cv_tok_s": {"unit": "fraction", "quality": "measured", "level": "diagnostic"},
            "cv_step_seconds": {"unit": "fraction", "quality": "measured", "level": "diagnostic"},
            "recommended_tolerance": {"unit": "fraction", "quality": "derived", "level": "diagnostic"},
            "window_n": {"unit": "count", "quality": "measured", "level": "diagnostic"},
            "k_sigma": {"unit": "ratio", "quality": "derived", "level": "diagnostic"},
        },
        level="diagnostic",
        raw={"path": "evidence/kda-wyut/metrics.jsonl", "format": "jsonl", "retention_days": None, "in_git": True},
        context=("repo",),
        pack="ml",
    )

    def collect(self, subject: dict[str, Any], *, out_dir: Any = None, **_: Any) -> list[Fact]:
        data = measure(out_dir)
        method = "CV рядов S-012 (окна метрик претрейна); допуск = k·σ, k — решение ADR-038"
        facts: list[Fact] = []
        for fact in ("cv_tok_s", "cv_step_seconds", "recommended_tolerance"):
            value = data[fact]
            if value is None:
                facts.append(self.unavailable(
                    fact, subject,
                    "ряда S-012 с числами нет (метрики не сняты) — допуск не подставляется константой",
                ))
            else:
                facts.append(self.fact(fact, value, subject=subject, method=method))
        facts.append(self.fact("window_n", data["window_n"], subject=subject, method=method))
        facts.append(self.fact("k_sigma", data["k_sigma"], subject=subject, method=method))
        return facts


EXPORTER = NoiseBaselineExporter()
