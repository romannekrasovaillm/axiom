"""Экспортёр S-037 flapping: флапание вердиктов (ADR-038, дельта M4).

По истории S-036 (запись на каждое утверждение с каждого прогона ``--evaluate``)
считает число смен исхода за последние ``window_n`` оценок. Превышение порога
помечает утверждение ``flapping``: оно не открывает гейт (preflight даёт
``unverified``). Порог и окно — решение (ADR-038), не измерение.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from tools.sensors.fact import read_records
from tools.sensors.protocol import BaseExporter, Fact, SensorSpec

#: Окно истории и порог смен (решение, не измерение).
WINDOW_N = 10
THRESHOLD = 3


def measure(out_dir: Any = None, *, window_n: int = WINDOW_N, threshold: int = THRESHOLD) -> dict[str, Any]:
    try:
        records = read_records("S-036", out_dir)
    except Exception:  # noqa: BLE001
        records = []
    history: dict[str, list[str]] = {}
    for record in records:
        value = record.get("value")
        if not isinstance(value, dict):
            continue
        claim = value.get("claim")
        verdict = value.get("verdict")
        if isinstance(claim, str) and isinstance(verdict, str):
            history.setdefault(claim, []).append(verdict)

    flapping: dict[str, bool] = {}
    switches: dict[str, int] = {}
    for claim, verdicts in history.items():
        tail = verdicts[-window_n:]
        changes = sum(1 for a, b in zip(tail, tail[1:]) if a != b)
        switches[claim] = changes
        flapping[claim] = changes >= threshold
    return {
        "flapping": flapping,
        "switches": switches,
        "window_n": window_n,
        "threshold": threshold,
        "n_claims_with_history": len(history),
    }


class FlappingExporter(BaseExporter):
    SPEC = SensorSpec(
        id="S-037",
        facts=("flapping", "switches", "window_n", "threshold", "n_claims_with_history"),
        schema={
            "flapping": {"unit": "", "quality": "derived", "level": "diagnostic"},
            "switches": {"unit": "", "quality": "derived", "level": "diagnostic"},
            "window_n": {"unit": "count", "quality": "measured", "level": "diagnostic"},
            "threshold": {"unit": "count", "quality": "measured", "level": "diagnostic"},
            "n_claims_with_history": {"unit": "count", "quality": "derived", "level": "diagnostic"},
        },
        level="diagnostic",
        raw={"note": "derived: история вердиктов S-036 (происхождение даёт inputs)"},
        context=("repo",),
        pack="spine",
    )

    def collect(self, subject: dict[str, Any], *, out_dir: Any = None,
                window_n: int = WINDOW_N, threshold: int = THRESHOLD, **_: Any) -> list[Fact]:
        data = measure(out_dir, window_n=window_n, threshold=threshold)
        method = f"история S-036, окно {window_n}, порог смен {threshold} (решение ADR-038)"
        return [
            self.fact("flapping", data["flapping"], subject=subject, method=method),
            self.fact("switches", data["switches"], subject=subject, method=method),
            self.fact("window_n", data["window_n"], subject=subject, method=method),
            self.fact("threshold", data["threshold"], subject=subject, method=method),
            self.fact("n_claims_with_history", data["n_claims_with_history"], subject=subject, method=method),
        ]


EXPORTER = FlappingExporter()
