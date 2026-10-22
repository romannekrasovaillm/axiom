"""Экспортёр S-032 piles_trend: понедельные доли кучек (ADR-038, дельта H3).

Ряд долей «документы / поведение / работа» из классификатора S-010
(``tools/sensors/process_piles.py``) по неделям, начиная с 25.09. Это **тренд
для наблюдения**, не KPI: порогов нет. Классификатор переиспользуется, второй
реализации нет.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from tools.sensors.process_piles import (
    BEHAVIOUR,
    DOCS,
    WORK,
    _classify_paths,
    _git,
    _rule_touched,
)
from tools.sensors.protocol import BaseExporter, Fact, SensorSpec

DEFAULT_SINCE = "2026-09-25"


def _parse_log_with_dates(text: str) -> list[dict[str, Any]]:
    commits: list[dict[str, Any]] = []
    current: Optional[dict[str, Any]] = None
    for line in text.split("\n"):
        if line.startswith("\x1e"):
            if current is not None:
                commits.append(current)
            sha, _, rest = line[1:].partition("\x1f")
            subject, _, date = rest.partition("\x1f")
            current = {"sha": sha.strip(), "subject": subject.strip(), "date": date.strip(), "files": []}
        elif current is not None and line.strip():
            parts = line.split("\t")
            if len(parts) >= 2:
                current["files"].append(parts[-1])
    if current is not None:
        commits.append(current)
    return commits


def _week_label(date_text: str) -> Optional[str]:
    try:
        parsed = datetime.fromisoformat(date_text)
    except (TypeError, ValueError):
        return None
    iso = parsed.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def measure(
    repo: str | Path,
    *,
    since: str = DEFAULT_SINCE,
    until: Optional[str] = None,
) -> dict[str, Any]:
    repo = Path(repo)
    args = ["log", "--no-merges", "--name-status", "--format=%x1e%H%x1f%s%x1f%cI", f"--since={since}"]
    if until:
        args.append(f"--until={until}")
    commits = _parse_log_with_dates(_git(repo, args))

    weeks: dict[str, dict[str, int]] = {}
    for commit in commits:
        label = _week_label(commit["date"])
        if label is None:
            continue
        bucket = weeks.setdefault(label, {DOCS: 0, BEHAVIOUR: 0, WORK: 0})
        paths = commit["files"]
        if not paths:
            bucket[WORK] += 1
            continue
        rule_touched = _rule_touched(repo, commit["sha"]) if any(
            p.startswith("model/AD-") for p in paths
        ) else False
        bucket[_classify_paths(paths, rule_touched)] += 1

    ordered = sorted(weeks)
    docs: list[Optional[float]] = []
    behaviour: list[Optional[float]] = []
    work: list[Optional[float]] = []
    for label in ordered:
        counts = weeks[label]
        total = sum(counts.values())
        for bucket, series in ((DOCS, docs), (BEHAVIOUR, behaviour), (WORK, work)):
            series.append(round(counts[bucket] / total, 6) if total else None)
    return {"weeks": ordered, "docs": docs, "behaviour": behaviour, "work": work, "n_weeks": len(ordered)}


class PilesTrendExporter(BaseExporter):
    SPEC = SensorSpec(
        id="S-032",
        facts=("weekly_docs_share", "weekly_behaviour_share", "weekly_work_share", "weeks", "n_weeks"),
        schema={
            "weekly_docs_share": {"unit": "fraction", "quality": "measured", "level": "component"},
            "weekly_behaviour_share": {"unit": "fraction", "quality": "measured", "level": "component"},
            "weekly_work_share": {"unit": "fraction", "quality": "measured", "level": "component"},
            "weeks": {"unit": "", "quality": "measured", "level": "diagnostic"},
            "n_weeks": {"unit": "count", "quality": "measured", "level": "diagnostic"},
        },
        level="component",
        raw={"path": ".git", "format": "text", "retention_days": None, "in_git": True},
        context=("repo",),
        pack="spine",
    )

    def collect(self, subject: dict[str, Any], *, root: Any = None, since: str = DEFAULT_SINCE, **_: Any) -> list[Fact]:
        repo_root = Path(root) if root is not None else Path(__file__).resolve().parents[4]
        data = measure(repo_root, since=since)
        method = f"git log по неделям с {since}; классификатор S-010 (process_piles)"
        facts = [
            self.fact("weeks", data["weeks"], subject=subject, method=method),
            self.fact("weekly_docs_share", data["docs"], subject=subject, method=method),
            self.fact("weekly_behaviour_share", data["behaviour"], subject=subject, method=method),
            self.fact("weekly_work_share", data["work"], subject=subject, method=method),
            self.fact("n_weeks", data["n_weeks"], subject=subject, method=method),
        ]
        return facts


EXPORTER = PilesTrendExporter()
