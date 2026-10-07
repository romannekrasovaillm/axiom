"""Реестр кандидатов (ADR-039, дельта G).

Кандидат — предложенная привязка «шаблон → предмет → порог → источник порога».
Исполнитель кандидатов **не принимает**: ``accepted`` ставит только архитектор.
Дедупликация — по отпечатку ``fp = sha256(property + канонизированные params)``;
отклонённые отпечатки повторно не предлагаются.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from tools.properties.base import canonical_params, property_fingerprint

CANDIDATES_FILE = Path("model/candidates.yaml")
SOURCES: tuple[str, ...] = ("declared", "observed", "telemetry", "incident", "llm")
STATUSES: tuple[str, ...] = ("proposed", "accepted", "rejected", "deferred")


def make_candidate(
    source: str,
    property_name: str,
    params: dict,
    *,
    evidence: Optional[list] = None,
    support: Optional[int] = None,
    status: str = "proposed",
    ref: Optional[str] = None,
    reason: Optional[str] = None,
) -> dict:
    if source not in SOURCES:
        raise ValueError(f"источник кандидата {source!r} вне {SOURCES}")
    if status not in STATUSES:
        raise ValueError(f"статус кандидата {status!r} вне {STATUSES}")
    return {
        "fp": property_fingerprint(property_name, params),
        "source": source,
        "property": property_name,
        "params": canonical_params(params),
        "evidence": list(evidence or []),
        "support": support,
        "status": status,
        "ref": ref,
        "reason": reason,
    }


def load_candidates(root: str | Path) -> list[dict]:
    path = Path(root) / CANDIDATES_FILE
    if not path.is_file():
        return []
    from tools.miniyaml import load_file

    data = load_file(path)
    if isinstance(data, dict):
        data = data.get("candidates")
    return [c for c in data if isinstance(c, dict)] if isinstance(data, list) else []


def _emit(entries: list[dict]) -> str:
    lines = [
        "# model/candidates.yaml — реестр кандидатов-привязок (ADR-039, дельта G).",
        "#",
        "# Кандидат предложен генератором (source ∈ declared|observed|telemetry|incident|llm);",
        "# принимает кандидата ТОЛЬКО архитектор (status accepted + ref на CL-NNN).",
        "# Дедупликация по fp; отклонённые/deferred несут reason и повторно не предлагаются.",
        "# Перегенерация: python3 -m tools.properties.generate_candidates",
        "",
    ]
    for e in entries:
        lines.append(f"- fp: {e['fp']}")
        lines.append(f"  source: {e['source']}")
        lines.append(f"  property: {e['property']}")
        lines.append(f"  params: {json.dumps(e.get('params') or {}, ensure_ascii=False, sort_keys=True)}")
        lines.append(f"  evidence: {json.dumps(e.get('evidence') or [], ensure_ascii=False)}")
        if e.get("support") is not None:
            lines.append(f"  support: {e['support']}")
        lines.append(f"  status: {e['status']}")
        lines.append(f"  ref: {e['ref'] if e.get('ref') else 'null'}")
        lines.append(f"  reason: {json.dumps(e['reason'], ensure_ascii=False) if e.get('reason') else 'null'}")
    return "\n".join(lines) + "\n"


def save_candidates(root: str | Path, entries: list[dict]) -> None:
    path = Path(root) / CANDIDATES_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_emit(entries), encoding="utf-8")


def merge(existing: list[dict], incoming: list[dict]) -> tuple[list[dict], int, int]:
    """Сливает кандидатов по fp: новые добавляются, известные не дублируются.

    Возвращает ``(реестр, добавлено, пропущено)``. Отклонённый отпечаток
    повторно не предлагается (пропущен).
    """
    seen = {str(e.get("fp")): e for e in existing}
    added = skipped = 0
    out = list(existing)
    for cand in incoming:
        fp = str(cand.get("fp"))
        if fp in seen:
            skipped += 1
            continue
        out.append(cand)
        seen[fp] = cand
        added += 1
    return out, added, skipped
