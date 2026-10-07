#!/usr/bin/env python3
"""check_claims.py — страж реестра утверждений и реестра инцидентов (ADR-037, дельта A).

Утверждение (`model/claims.yaml`) — дословный anchor первоисточника + предикат
над фактом датчика. Реестр инцидентов — `evidence/incidents.yaml`.

Режимы::

    --verify                 схема, anchor, датчик/факт, правило, уникальность id,
                             реестр инцидентов (rule-kind, test-файл)
    --evaluate [--fail-on X] последний факт с совпавшим предметом → предикат →
                             класс pass | fail | unverified (нет факта/устарел/
                             предмет не совпал/факт unverified)
    --mode config-bindings   сверка binding ↔ <root>/net/config.json (песочница)
    --scan-adr               информационно: числа/формулировки ADR без anchor (exit 0)
    --report                 счётчики реестров (без порогов)
    --selftest               мутанты

Разбор YAML — stdlib-only (`tools/miniyaml.py`): правило C-049 исполняется в
песочнице без сети и сторонних зависимостей.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.miniyaml import MiniYamlError, load_file  # noqa: E402
from tools.sensors.fact import DEFAULT_FACTS_DIR, read_latest  # noqa: E402

CLAIMS_FILE = "model/claims.yaml"
INCIDENTS_FILE = "evidence/incidents.yaml"
SENSORS_FILE = "model/sensors.yaml"
CONSTRAINTS_FILE = "CONSTRAINTS.yaml"

CLAIM_KEYS = ("id", "source", "statement", "kind", "subject_match")
KINDS = ("number", "bound", "obligation", "config_binding")
GUARD_KEYS = ("rule", "test", "pending", "unguardable")
_RULE_ID_RE = re.compile(r"^\s*-\s*id:\s*(C-\d+)\s*$")
_ADR_KEYWORDS_RE = re.compile(r"долж|не более|не менее|цель|порог|≥|≤", re.IGNORECASE)


class ClaimsError(Exception):
    """Вход не читается/не является ожидаемым артефактом."""


# ── загрузка ────────────────────────────────────────────────────────────────


def _load(path: Path) -> Any:
    try:
        return load_file(path)
    except FileNotFoundError as exc:
        raise ClaimsError(f"нет файла: {path}") from exc
    except MiniYamlError as exc:
        raise ClaimsError(f"{path}: невалидный YAML — {exc}") from exc


def load_claims(root: Path) -> list[dict[str, Any]]:
    data = _load(root / CLAIMS_FILE)
    if not isinstance(data, list):
        raise ClaimsError(f"{CLAIMS_FILE}: ожидался список утверждений")
    return [c for c in data if isinstance(c, dict)]


def load_incidents(root: Path) -> list[dict[str, Any]]:
    path = root / INCIDENTS_FILE
    if not path.is_file():
        return []
    data = _load(path)
    if not isinstance(data, list):
        raise ClaimsError(f"{INCIDENTS_FILE}: ожидался список инцидентов")
    return [c for c in data if isinstance(c, dict)]


def load_sensors(root: Path) -> list[dict[str, Any]]:
    data = _load(root / SENSORS_FILE)
    if isinstance(data, dict):
        data = data.get("sensors")
    return [s for s in data if isinstance(s, dict)] if isinstance(data, list) else []


def rule_ids_and_kinds(root: Path) -> dict[str, str]:
    """id → kind из CONSTRAINTS.yaml (regex: YAML кейса может быть вне подмножества)."""
    text = (root / CONSTRAINTS_FILE).read_text(encoding="utf-8")
    ids = [m.group(1) for m in _RULE_ID_RE.finditer(text)]
    kinds: dict[str, str] = {}
    lines = text.splitlines()
    current: Optional[str] = None
    for line in lines:
        m = _RULE_ID_RE.match(line)
        if m:
            current = m.group(1)
            kinds.setdefault(current, "")
            continue
        if current and re.match(r"^\s+kind:\s*(\w+)", line):
            kinds[current] = re.match(r"^\s+kind:\s*(\w+)", line).group(1)
    return kinds


# ── verify ──────────────────────────────────────────────────────────────────


def verify(root: Path, *, with_incidents: bool = True) -> list[str]:
    errors: list[str] = []
    claims = load_claims(root)
    sensors = load_sensors(root)
    sensor_facts = {s.get("id"): set(s.get("facts") or []) for s in sensors}
    rules = rule_ids_and_kinds(root)

    seen: set[str] = set()
    for claim in claims:
        cid = str(claim.get("id", "?"))
        for key in CLAIM_KEYS:
            if key not in claim:
                errors.append(f"{cid}: нет поля {key!r}")
        if cid in seen:
            errors.append(f"{cid}: дубль id")
        seen.add(cid)

        source = claim.get("source")
        if not isinstance(source, dict) or "file" not in source or "anchor" not in source:
            errors.append(f"{cid}: source — {file, anchor} обязательны")
        else:
            fpath = root / str(source["file"])
            anchor = str(source["anchor"])
            if not fpath.is_file():
                errors.append(f"{cid}: source.file не существует: {source['file']}")
            else:
                text = fpath.read_text(encoding="utf-8")
                if anchor not in text:
                    errors.append(f"{cid}: anchor не найден дословно в {source['file']}")
            if not (20 <= len(anchor) <= 200):
                errors.append(f"{cid}: длина anchor {len(anchor)} вне 20–200")

        if claim.get("kind") not in KINDS:
            errors.append(f"{cid}: kind={claim.get('kind')!r} вне {KINDS}")

        has_sensor = claim.get("sensor") is not None
        has_pending = claim.get("pending") is not None
        if has_sensor == has_pending:
            errors.append(f"{cid}: ровно одно из sensor | pending")
        if has_sensor:
            sid = claim.get("sensor")
            fact = claim.get("fact")
            if "fact" not in claim:
                errors.append(f"{cid}: sensor без fact")
            if "predicate" not in claim:
                errors.append(f"{cid}: sensor без predicate")
            if sid not in sensor_facts:
                errors.append(f"{cid}: датчик {sid} не объявлен в {SENSORS_FILE}")
            elif fact not in sensor_facts.get(sid, set()):
                errors.append(f"{cid}: факт {fact!r} не объявлен у датчика {sid}")

        rule = claim.get("rule")
        if rule is not None and rule not in rules:
            errors.append(f"{cid}: правило {rule} не существует в {CONSTRAINTS_FILE}")

        if claim.get("kind") == "config_binding":
            binding = claim.get("binding")
            if not isinstance(binding, dict) or "file" not in binding or "path" not in binding:
                errors.append(f"{cid}: config_binding требует binding {{file, path, value}}")

    if with_incidents:
        errors.extend(_verify_incidents(root, sensor_facts, rules))
    return errors


def _verify_incidents(root: Path, sensor_facts: dict, rules: dict[str, str]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for inc in load_incidents(root):
        iid = str(inc.get("id", "?"))
        if iid in seen:
            errors.append(f"{iid}: дубль id")
        seen.add(iid)
        for key in ("id", "date", "title", "evidence", "sensor", "guard"):
            if key not in inc:
                errors.append(f"{iid}: нет поля {key!r}")
        for ev in inc.get("evidence") or []:
            if not (root / str(ev)).is_file():
                errors.append(f"{iid}: evidence не существует: {ev}")
        sensor = inc.get("sensor")
        if sensor is not None and sensor not in sensor_facts:
            errors.append(f"{iid}: датчик {sensor} не объявлен в {SENSORS_FILE}")
        guard = inc.get("guard")
        if not isinstance(guard, dict):
            errors.append(f"{iid}: guard обязателен")
            continue
        present = [k for k in GUARD_KEYS if k in guard and guard[k] is not None]
        if len(present) != 1:
            errors.append(f"{iid}: guard — ровно один ключ из {GUARD_KEYS}, найдено {present}")
            continue
        key = present[0]
        if key == "rule":
            rule = guard["rule"]
            if rule not in rules:
                errors.append(f"{iid}: правило-страж {rule} не существует")
            elif rules.get(rule) not in ("behavioural", "structural"):
                errors.append(f"{iid}: страж инцидента {rule} имеет kind={rules.get(rule)!r} (documentary не считается)")
        elif key == "test":
            spec = str(guard["test"])
            path_str, _, name = spec.partition("::")
            tpath = root / path_str
            if not tpath.is_file():
                errors.append(f"{iid}: test-файл не существует: {path_str}")
            elif not name or f"def {name}(" not in tpath.read_text(encoding="utf-8"):
                errors.append(f"{iid}: def {name} не найден в {path_str}")
        elif key == "pending":
            pending = guard["pending"]
            if not isinstance(pending, dict) or not pending.get("reason"):
                errors.append(f"{iid}: pending требует reason")
        elif key == "unguardable":
            if not isinstance(guard["unguardable"], dict) or not guard["unguardable"].get("reason"):
                errors.append(f"{iid}: unguardable требует reason")
    return errors


# ── предмет и предикат ──────────────────────────────────────────────────────


def reference_subject(root: Path) -> dict[str, Any]:
    from tools.sensors.subject import build_subject

    return build_subject(repo_root=root, config_path=root / "net" / "config.json", device="cpu")


def _subject_matches(record: dict[str, Any], keys: Any, ref: dict[str, Any]) -> bool:
    if not keys:
        return True
    rec = record.get("subject") or {}
    for key in keys:
        ref_value = ref.get(key)
        if ref_value is None:
            continue  # неизвестный конец не объявляем несовпадением
        if rec.get(key) != ref_value:
            return False
    return True


def apply_predicate(predicate: dict[str, Any], value: Any) -> Optional[bool]:
    """Применяет предикат. ``None`` — не применим (значение отсутствует/не тот тип)."""
    if not isinstance(predicate, dict):
        return None
    path = predicate.get("path")
    if path and isinstance(value, dict):
        value = value.get(path)
    if value is None:
        return None
    op = predicate.get("op")
    target = predicate.get("value")
    tol = predicate.get("tolerance")
    try:
        if op == "==":
            return value == target
        if op == "!=":
            return value != target
        if op == ">=":
            return value >= target
        if op == ">":
            return value > target
        if op == "<=":
            return value <= target
        if op == "<":
            return value < target
        if op == "approx":
            return abs(float(value) - float(target)) <= float(tol or 0.0)
        if op == "between":
            lo, hi = target
            return float(lo) <= float(value) <= float(hi)
    except (TypeError, ValueError):
        return None
    return None


def evaluate_claim(claim: dict[str, Any], root: Path, out_dir: Optional[Path]) -> dict[str, Any]:
    cid = claim.get("id")
    base = {"id": cid, "statement": claim.get("statement")}
    if claim.get("pending") is not None:
        return {**base, "verdict": "unverified", "reason": "pending: " + str((claim["pending"] or {}).get("reason", ""))}
    sensor = claim.get("sensor")
    fact = claim.get("fact")
    keys = claim.get("subject_match") or []
    ref = reference_subject(root)
    record = read_latest(sensor, fact, keys, out_dir=out_dir, subject=ref) if sensor and fact else None
    if record is None:
        return {**base, "verdict": "unverified", "sensor": sensor, "fact": fact,
                "reason": "нет факта с совпавшим предметом"}
    if record.get("status") == "unverified":
        return {**base, "verdict": "unverified", "sensor": sensor, "fact": fact,
                "reason": "факт unverified: " + str(record.get("note", ""))}
    ok = apply_predicate(claim.get("predicate") or {}, record.get("value"))
    if ok is None:
        return {**base, "verdict": "unverified", "sensor": sensor, "fact": fact,
                "reason": "предикат не применим к значению", "value": record.get("value")}
    return {**base, "verdict": "pass" if ok else "fail", "sensor": sensor, "fact": fact,
            "value": record.get("value"), "ts": record.get("ts")}


def evaluate(root: Path, out_dir: Optional[Path] = None) -> list[dict[str, Any]]:
    return [evaluate_claim(c, root, out_dir) for c in load_claims(root)]


# ── config-bindings ─────────────────────────────────────────────────────────


def _dig(data: Any, path: str) -> Any:
    cur = data
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _adr_id(claim: dict[str, Any]) -> Optional[str]:
    import re as _re

    m = _re.search(r"(ADR-\d+)", str((claim.get("source") or {}).get("file", "")))
    return m.group(1) if m else None


def config_bindings(root: Path) -> list[dict[str, Any]]:
    """Сверка ``config_binding`` ↔ ``<root>/net/config.json``.

    Сообщение называет CL-id, файл, ключ и ADR, но **не ожидаемое значение**
    (урок DEF-1: артефакт сообщён — решение не подсказано, ADR-037 дельта D4).
    """
    results: list[dict[str, Any]] = []
    for claim in load_claims(root):
        if claim.get("kind") != "config_binding":
            continue
        binding = claim.get("binding") or {}
        rel = binding.get("file")
        path = binding.get("path")
        expected = binding.get("value")
        actual = None
        ok = False
        try:
            data = json.loads((root / str(rel)).read_text(encoding="utf-8"))
            actual = _dig(data, str(path))
            ok = actual == expected
        except (OSError, json.JSONDecodeError):
            actual = None
        entry: dict[str, Any] = {
            "id": claim.get("id"), "adr": _adr_id(claim),
            "binding": {"file": rel, "path": path},
            "verdict": "pass" if ok else "fail",
        }
        if not ok:
            entry["message"] = (
                f"C-049: привязка {claim.get('id')} не совпала с {rel}:{path} "
                f"(ADR {_adr_id(claim)}) — значение не сообщается намеренно"
            )
        results.append(entry)
    return results


# ── реестры слоя (дельта H/I/J, ADR-038) ────────────────────────────────────

TRIAGE_FILE = "model/claims-triage.yaml"
LINEAGE_FILE = "model/rule-lineage.yaml"
GATES_FILE = "model/opening-gates.yaml"

#: Решения кандидата-триажа (дельта H1).
DISPOSITIONS: tuple[str, ...] = ("claim", "measurable_pending", "not_measurable", "not_a_claim")

#: Ставки утверждения (дельта J1): цена ошибки.
STAKES: tuple[str, ...] = ("money", "irreversible", "public", "internal")
_STAKE_ORDER = {"money": 0, "irreversible": 1, "public": 2, "internal": 3}


def _load_registry(root: Path, rel: str) -> Optional[list[dict[str, Any]]]:
    """Читает список-реестр; отсутствие файла — ``None``, пустой/не список — ошибка."""
    path = root / rel
    if not path.is_file():
        return None
    data = _load(path)
    if isinstance(data, dict):
        for key in (Path(rel).stem, "entries", "items"):
            candidate = data.get(key)
            if isinstance(candidate, list):
                data = candidate
                break
    if data is None:
        return []
    if not isinstance(data, list):
        raise ClaimsError(f"{rel}: ожидался список")
    return [item for item in data if isinstance(item, dict)]


def load_triage(root: Path) -> Optional[list[dict[str, Any]]]:
    return _load_registry(root, TRIAGE_FILE)


def load_lineage(root: Path) -> Optional[list[dict[str, Any]]]:
    return _load_registry(root, LINEAGE_FILE)


def adr_candidates(root: Path) -> list[dict[str, Any]]:
    """Кандидаты-утверждения из ADR: строки с маркерами долга/порога (дельта H1)."""
    out: list[dict[str, Any]] = []
    adr_dir = root / "docs" / "adr"
    if not adr_dir.is_dir():
        return out
    for path in sorted(adr_dir.glob("*.md")):
        rel = path.relative_to(root).as_posix()
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if text and _ADR_KEYWORDS_RE.search(text) and 20 <= len(text) <= 200:
                out.append({"file": rel, "anchor": text})
    return out


def scan_adr(root: Path) -> dict[str, Any]:
    """Кандидаты ADR и доля нерешённых (число — информационный факт, порога нет)."""
    anchors_files = {str(c.get("source", {}).get("file")) for c in load_claims(root)}
    files = []
    for path in sorted((root / "docs" / "adr").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        hits = [ln.strip() for ln in text.splitlines() if _ADR_KEYWORDS_RE.search(ln)]
        rel = path.relative_to(root).as_posix()
        files.append({"file": rel, "hits": len(hits), "covered": rel in anchors_files})
    decided: dict[tuple[str, str], Any] = {}
    for entry in load_triage(root) or []:
        if entry.get("file") and entry.get("anchor"):
            decided[(str(entry["file"]), str(entry["anchor"]))] = entry.get("disposition")
    candidates = [
        {**cand, "disposition": decided.get((cand["file"], cand["anchor"]))}
        for cand in adr_candidates(root)
    ]
    unresolved = [c for c in candidates if c["disposition"] is None]
    return {
        "files": files,
        "candidates": candidates,
        "unresolved": unresolved,
        "unresolved_count": len(unresolved),
    }


def shares(root: Path) -> dict[str, Any]:
    """Две доли и их числители/знаменатели (дельта H2/H4). Порогов нет."""
    claims = load_claims(root)
    triage = load_triage(root) or []
    with_sensor = sum(1 for c in claims if c.get("sensor"))
    measurable_pending = sum(1 for t in triage if t.get("disposition") == "measurable_pending")
    denom_claims = with_sensor + measurable_pending
    incidents = load_incidents(root)
    guarded = sum(
        1 for i in incidents
        if isinstance(i.get("guard"), dict) and ("rule" in i["guard"] or "test" in i["guard"])
    )
    scan = scan_adr(root)
    return {
        "share_measurable_claims_with_sensor": {
            "value": (with_sensor / denom_claims if denom_claims else None),
            "numerator": with_sensor,
            "denominator": denom_claims,
        },
        "share_incidents_guarded": {
            "value": (guarded / len(incidents) if incidents else None),
            "numerator": guarded,
            "denominator": len(incidents),
        },
        "measurable_pending_candidates": measurable_pending,
        "untriaged_candidates": scan["unresolved_count"],
    }


def report(root: Path, out_dir: Optional[Path] = None) -> dict[str, Any]:
    claims = load_claims(root)
    incidents = load_incidents(root)
    sensors = load_sensors(root)
    results = evaluate(root, out_dir)
    by_class = {"pass": 0, "fail": 0, "unverified": 0}
    for r in results:
        by_class[r["verdict"]] = by_class.get(r["verdict"], 0) + 1
    guarded = sum(
        1 for i in incidents
        if isinstance(i.get("guard"), dict) and ("rule" in i["guard"] or "test" in i["guard"])
    )
    payload = {
        "claims_total": len(claims),
        "claims_with_sensor": sum(1 for c in claims if c.get("sensor")),
        "claims_pending": sum(1 for c in claims if c.get("pending")),
        "verdicts": by_class,
        "incidents_total": len(incidents),
        "incidents_guarded": guarded,
        "sensors_active": sum(1 for s in sensors if s.get("status") == "active"),
        "sensors_pending": sum(1 for s in sensors if s.get("status") == "pending"),
    }
    payload.update(shares(root))
    # K6: «не проверено» не сворачивается в зелёное — отдельный блок с причиной.
    payload["unverified_block"] = [
        {"id": r["id"], "sensor": r.get("sensor"), "fact": r.get("fact"), "reason": r.get("reason")}
        for r in results
        if r["verdict"] == "unverified"
    ]
    return payload


def format_shares(payload: dict[str, Any]) -> list[str]:
    """Две доли первыми строками — с числителем и знаменателем (дельта H4)."""
    lines = []
    for key, title in (
        ("share_measurable_claims_with_sensor", "доля утверждений с датчиком"),
        ("share_incidents_guarded", "доля инцидентов под стражем"),
    ):
        block = payload.get(key) or {}
        value = block.get("value")
        shown = "n/a" if value is None else f"{value:.4f}"
        lines.append(
            f"{key} = {shown} ({block.get('numerator')}/{block.get('denominator')}) — {title}"
        )
    lines.append(f"untriaged_candidates = {payload.get('untriaged_candidates')} (информационно, порога нет)")
    return lines


# ── verify-layer: согласованность реестров (дельта I/J, ADR-038) ─────────────


def _verify_lineage(root: Path) -> list[str]:
    errors: list[str] = []
    kinds = rule_ids_and_kinds(root)
    documentary = {rid for rid, kind in kinds.items() if kind == "documentary"}
    lineage = load_lineage(root)
    if lineage is None:
        return [f"{LINEAGE_FILE}: реестр наследования отсутствует"]
    seen: set[str] = set()
    for entry in lineage:
        rid = str(entry.get("rule", "?"))
        if rid in seen:
            errors.append(f"{rid}: дубль в {LINEAGE_FILE}")
        seen.add(rid)
        if rid not in kinds:
            errors.append(f"{rid}: правила нет в CONSTRAINTS.yaml")
            continue
        present = [k for k in ("successor", "none", "pending") if entry.get(k) is not None]
        if len(present) != 1:
            errors.append(f"{rid}: ровно одно из successor|none|pending, найдено {present}")
            continue
        key = present[0]
        if key == "successor":
            succ = str(entry["successor"])
            if succ not in kinds:
                errors.append(f"{rid}: наследник {succ} не существует в CONSTRAINTS.yaml")
            elif kinds.get(succ) == "documentary":
                errors.append(f"{rid}: наследник {succ} — documentary (нужен поведенческий/структурный)")
        elif key == "none":
            reason = entry["none"]
            if not (isinstance(reason, dict) and str(reason.get("reason", "")).strip()):
                errors.append(f"{rid}: none требует непустую причину")
        else:
            pending = entry["pending"]
            if not (
                isinstance(pending, dict)
                and str(pending.get("reason", "")).strip()
                and pending.get("owner")
                and pending.get("since")
            ):
                errors.append(f"{rid}: pending требует reason, owner, since")
    for rid in sorted(documentary - seen):
        errors.append(f"{rid}: documentary-правило не покрыто реестром {LINEAGE_FILE}")
    return errors


def _verify_claims_stake(root: Path) -> list[str]:
    errors: list[str] = []
    for claim in load_claims(root):
        cid = str(claim.get("id", "?"))
        stake = claim.get("stake")
        if stake is None:
            errors.append(f"{cid}: нет поля stake (дельта J)")
            continue
        if stake not in STAKES:
            errors.append(f"{cid}: stake={stake!r} вне {STAKES}")
            continue
        if stake in ("money", "irreversible", "public"):
            has_sensor = claim.get("sensor") is not None
            pending = claim.get("pending")
            pending_ok = isinstance(pending, dict) and pending.get("owner") and pending.get("since")
            if not has_sensor and not pending_ok:
                errors.append(
                    f"{cid}: ставка {stake} без датчика и без pending с owner/since"
                )
    return errors


def _verify_triage(root: Path) -> list[str]:
    errors: list[str] = []
    claims_ids = {str(c.get("id")) for c in load_claims(root)}
    triage = load_triage(root)
    if triage is None:
        return [f"{TRIAGE_FILE}: реестр триажа отсутствует"]
    seen: set[tuple[str, str]] = set()
    for entry in triage:
        rel = entry.get("file")
        anchor = entry.get("anchor")
        if not rel or not anchor:
            errors.append("triage: file и anchor обязательны")
            continue
        rel, anchor = str(rel), str(anchor)
        key = (rel, anchor)
        if key in seen:
            errors.append(f"triage: дубль записи {rel}: {anchor[:40]}")
        seen.add(key)
        disposition = entry.get("disposition")
        if disposition not in DISPOSITIONS:
            errors.append(f"triage {rel}: disposition={disposition!r} вне {DISPOSITIONS}")
            continue
        path = root / rel
        if not path.is_file():
            errors.append(f"triage: файл не существует: {rel}")
            continue
        if anchor not in path.read_text(encoding="utf-8"):
            errors.append(f"triage: anchor не найден дословно в {rel}: {anchor[:50]}")
        if not (20 <= len(anchor) <= 200):
            errors.append(f"triage: длина anchor {len(anchor)} вне 20–200")
        if disposition == "claim":
            if str(entry.get("ref")) not in claims_ids:
                errors.append(f"triage {rel}: claim без ref на существующее утверждение")
        elif disposition == "measurable_pending":
            if not (entry.get("owner") and entry.get("since")):
                errors.append(f"triage {rel}: measurable_pending требует owner и since")
        elif not str(entry.get("reason", "")).strip():
            errors.append(f"triage {rel}: {disposition} требует reason")
    return errors


def load_gates(root: Path) -> Optional[list[dict[str, Any]]]:
    return _load_registry(root, GATES_FILE)


def _fail_open_map(root: Path) -> dict[str, Any]:
    """Карта fail-open стражей из факта S-034 (пусто, если аудита нет)."""
    try:
        from tools.sensors.fact import read_latest

        record = read_latest("S-034", "fail_open")
        if record and record.get("status") == "ok" and isinstance(record.get("value"), dict):
            return record["value"]
    except Exception:  # noqa: BLE001 — нет факта = нечего сверять
        return {}
    return {}


def _verify_gates(root: Path) -> list[str]:
    """Реестр открывающих гейтов: состав, opens, и запрет fail-open без verified (K5)."""
    errors: list[str] = []
    gates = load_gates(root)
    if gates is None:
        return [f"{GATES_FILE}: реестр открывающих гейтов отсутствует"]
    seen: set[str] = set()
    fail_open = _fail_open_map(root)
    for gate in gates:
        name = gate.get("gate")
        if not isinstance(name, str) or not name.strip():
            errors.append("gate: имя обязательно")
            continue
        if name in seen:
            errors.append(f"gate {name}: дубль")
        seen.add(name)
        if gate.get("opens") not in ("money", "stage", "public"):
            errors.append(f"gate {name}: opens вне money|stage|public")
        requires = gate.get("requires")
        if not isinstance(requires, list) or not requires:
            errors.append(f"gate {name}: requires — непустой список")
            continue
        for req in requires:
            if not isinstance(req, dict):
                errors.append(f"gate {name}: требование не объект")
                continue
            if isinstance(req.get("guard"), str):
                guard = req["guard"]
                if fail_open.get(guard) is True and not req.get("require_verified"):
                    errors.append(
                        f"gate {name}: страж {guard} fail-open — требуется require_verified (K5)"
                    )
            elif not isinstance(req.get("claim"), str):
                errors.append(f"gate {name}: требование без claim|guard")
    return errors


def verify_layer(root: Path) -> list[str]:
    """Согласованность реестров слоя (дельты I/J/K; L/M добавляются далее)."""
    errors: list[str] = []
    errors.extend(_verify_lineage(root))
    errors.extend(_verify_claims_stake(root))
    errors.extend(_verify_triage(root))
    errors.extend(_verify_gates(root))
    return errors


# ── очередь работ (дельта J3) ───────────────────────────────────────────────


def queue(root: Path) -> list[dict[str, Any]]:
    """``pending`` и ``measurable_pending``, отсортированные по ставке и возрасту."""
    items: list[dict[str, Any]] = []
    for claim in load_claims(root):
        pending = claim.get("pending")
        if not isinstance(pending, dict):
            continue
        items.append({
            "kind": "claim",
            "id": str(claim.get("id")),
            "stake": str(claim.get("stake", "internal")),
            "since": str(pending.get("since") or ""),
            "owner": str(pending.get("owner") or ""),
            "reason": str(pending.get("reason") or ""),
        })
    for entry in load_triage(root) or []:
        if entry.get("disposition") != "measurable_pending":
            continue
        items.append({
            "kind": "triage",
            "id": str(entry.get("file")) + ": " + str(entry.get("anchor", ""))[:60],
            "stake": str(entry.get("stake", "internal")),
            "since": str(entry.get("since") or ""),
            "owner": str(entry.get("owner") or ""),
            "reason": "measurable_pending",
        })
    items.sort(key=lambda item: (_STAKE_ORDER.get(item["stake"], 9), item["since"]))
    return items



# ── selftest (мутанты) ──────────────────────────────────────────────────────


def _write_fixture(tmp: Path, subject: dict, out_dir: Path, *, pin: Optional[str] = None, ts: Optional[str] = None) -> None:
    from tools.sensors.fact import write_fact

    subj = dict(subject)
    if pin is not None:
        subj["config_sha256"] = pin
    write_fact("S-001", "num_kda_layers", 18, unit="count", quality="measured",
               method="fixture", subject=subj, out_dir=out_dir, ts=ts)


def run_selftest() -> int:
    checks: list[tuple[str, bool]] = []

    def _write_claims(path: Path, claims: list[dict[str, Any]]) -> None:
        import json as _json
        # JSON — подмножество YAML: мини-YAML прочитает оба.
        path.write_text(_json.dumps(claims, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    with tempfile.TemporaryDirectory(prefix="claims-selftest-") as tmp:
        root = Path(tmp)
        (root / "docs").mkdir()
        src = root / "docs" / "src.md"
        anchor = "целевой порог приёмки не менее 800 ток/с на окне"
        src.write_text(f"# doc\n{anchor}\n", encoding="utf-8")
        (root / "model").mkdir()
        (root / "net").mkdir()
        (root / "net" / "config.json").write_text('{"num_kda_layers": 18}', encoding="utf-8")
        (root / "model" / "sensors.yaml").write_text(
            json.dumps([{"id": "S-001", "facts": ["num_kda_layers"]}], ensure_ascii=False),
            encoding="utf-8",
        )
        (root / "CONSTRAINTS.yaml").write_text(
            "constraints:\n  - id: C-049\n    kind: structural\n    severity: high\n",
            encoding="utf-8",
        )
        from tools.sensors.subject import build_subject

        subject = build_subject(
            repo_root=root, git_sha="a" * 40, dirty=False,
            config_path=root / "net" / "config.json", device="cpu",
        )
        facts = root / "facts"
        base = {
            "id": "CL-900",
            "source": {"file": "docs/src.md", "anchor": anchor},
            "statement": "порог 800",
            "kind": "bound",
            "sensor": "S-001",
            "fact": "num_kda_layers",
            "predicate": {"op": ">=", "value": 18, "unit": "count", "tolerance": None},
            "subject_match": ["config_sha256"],
            "rule": None,
            "binding": None,
        }

        # Эталон: anchor на месте, датчик/факт объявлены, предмет совпал, предикат pass.
        _write_claims(root / "model" / "claims.yaml", [base])
        _write_fixture(root, subject, facts)
        errors = verify(root, with_incidents=False)
        checks.append(("эталон: verify без ошибок", errors == []))
        good = evaluate_claim(base, root, facts)
        checks.append(("эталон: предикат pass", good["verdict"] == "pass"))

        # Мутант: дрейф anchor.
        drift = dict(base)
        drift["source"] = {"file": "docs/src.md", "anchor": "другой текст про порог 800"}
        _write_claims(root / "model" / "claims.yaml", [drift])
        checks.append(("дрейф anchor → ошибка", any("anchor" in e for e in verify(root, with_incidents=False))))

        # Мутант: предикат против unverified-факта.
        _write_claims(root / "model" / "claims.yaml", [base])
        from tools.sensors.fact import write_fact

        write_fact("S-001", "num_kda_layers", None, unit="count", quality="measured",
                   method="fixture", subject=subject, out_dir=facts, status="unverified", note="нет источника")
        checks.append(("предикат против unverified → unverified",
                       evaluate_claim(base, root, facts)["verdict"] == "unverified"))

    # Мутанты на реальном репозитории (tmp-факты): чужой пин, просроченный факт, дубль id.
    with tempfile.TemporaryDirectory(prefix="claims-mut-") as tmp:
        root = _REPO_ROOT
        facts = Path(tmp) / "facts"
        from tools.sensors.fact import write_fact

        ref = reference_subject(root)
        write_fact("S-001", "num_kda_layers", 18, unit="count", quality="measured",
                   method="fixture", subject={**ref, "config_sha256": "dead" * 16}, out_dir=facts)
        claim = {
            "id": "CL-901", "kind": "number", "sensor": "S-001", "fact": "num_kda_layers",
            "predicate": {"op": "==", "value": 18}, "subject_match": ["config_sha256"],
            "statement": "x", "source": {"file": "net/config.json", "anchor": '"num_kda_layers": 18,'},
            "pending": None, "rule": None, "binding": None,
        }
        checks.append(("факт с чужим пином → unverified",
                       evaluate_claim(claim, root, facts)["verdict"] == "unverified"))
        empty_dir = Path(tmp) / "empty_facts"
        missing = evaluate_claim({**claim, "subject_match": []}, Path(tmp), empty_dir)
        checks.append(("нет фактов в каталоге → unverified", missing["verdict"] == "unverified"))

    checks.append(("config-bindings на репозитории зелёные", all(r["verdict"] == "pass" for r in config_bindings(_REPO_ROOT))))
    ok = all(p for _, p in checks)
    for label, passed in checks:
        print(f"[selftest] {'PASS' if passed else 'FAIL'}: {label}")
    print(f"[selftest] {'PASS' if ok else 'FAIL'}: check_claims")
    return 0 if ok else 1


# ── CLI ─────────────────────────────────────────────────────────────────────


def _print_summary(results: list[dict[str, Any]]) -> None:
    counts = {"pass": 0, "fail": 0, "unverified": 0}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    print(f"Утверждений: {len(results)}; pass {counts['pass']}, fail {counts['fail']}, unverified {counts['unverified']}")
    for r in results:
        mark = r["verdict"].upper()
        print(f"  [{r['id']}] {mark}: {r.get('reason') or (str(r.get('value')) + ' ' + str((r.get('statement') or '')[:40]))}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Страж реестра утверждений и инцидентов (ADR-037)")
    parser.add_argument("--root", default=".", help="корень кейса (по умолчанию '.')")
    parser.add_argument("--out-dir", default=None, help="каталог фактов (evidence/facts)")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--mode", choices=["config-bindings"], default=None)
    parser.add_argument("--scan-adr", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--verify-layer", action="store_true",
                        help="согласованность реестров слоя (ADR-038, дельта I/J)")
    parser.add_argument("--queue", action="store_true",
                        help="очередь работ: pending и measurable_pending по ставке (дельта J3)")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--fail-on", choices=["fail", "unverified"], default=None)
    parser.add_argument("--json", dest="json_path", default=None)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir) if args.out_dir else None

    if args.selftest:
        return run_selftest()
    try:
        if args.verify_layer:
            errors = verify_layer(root)
            for err in errors:
                print(f"[claims] LAYER-FAIL: {err}")
            print(f"verify-layer: {'OK' if not errors else f'{len(errors)} нарушений'}")
            if args.json_path:
                Path(args.json_path).write_text(
                    json.dumps({"mode": "verify-layer", "errors": errors}, ensure_ascii=False, indent=1) + "\n",
                    encoding="utf-8",
                )
            return 1 if errors else 0
        if args.queue:
            items = queue(root)
            payload = {"mode": "queue", "items": items}
            if args.json_path:
                Path(args.json_path).write_text(
                    json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
                )
            print(f"Очередь работ ({len(items)}): ставка → возраст")
            for item in items:
                print(f"  [{item['stake']}] {item['id']} (owner={item['owner']}, since={item['since']}): {item['reason'][:80]}")
            return 0
        if args.verify or args.mode == "config-bindings":
            failed = False
            payload: dict[str, Any] = {"mode": args.mode or "verify"}
            if args.verify:
                errors = verify(root)
                for err in errors:
                    print(f"[claims] FAIL: {err}")
                print(f"verify: {'OK' if not errors else f'{len(errors)} нарушений'}")
                payload["verify"] = {"errors": errors, "passed": not errors}
                failed = failed or bool(errors)
            if args.mode == "config-bindings":
                results = config_bindings(root)
                bad = [r for r in results if r["verdict"] != "pass"]
                payload["bindings"] = results
                payload["fail"] = len(bad)
                payload["passed"] = not bad
                print(json.dumps(
                    {"mode": "config-bindings", "bindings": results,
                     "fail": len(bad), "passed": not bad},
                    ensure_ascii=False,
                ))
                failed = failed or bool(bad)
            if args.json_path:
                Path(args.json_path).write_text(
                    json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
                )
            return 1 if failed else 0
        if args.evaluate:
            results = evaluate(root, out_dir)
            payload = {"claims": results}
            if args.json_path:
                Path(args.json_path).write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            _print_summary(results)
            if args.fail_on:
                fails = [r for r in results if r["verdict"] == "fail"]
                if args.fail_on == "fail" and fails:
                    return 1
                if args.fail_on == "unverified" and any(r["verdict"] != "pass" for r in results):
                    return 1
            return 0
        if args.scan_adr:
            payload = scan_adr(root)
            print(json.dumps(payload, ensure_ascii=False, indent=1))
            return 0
        if args.report:
            payload = report(root, out_dir)
            for line in format_shares(payload):
                print(line)
            block = payload.get("unverified_block") or []
            if block:
                print("\nНЕ ПРОВЕРЕНО (unverified) — дверь не открывает (K6, ADR-038):")
                for item in block:
                    print(f"  [{item['id']}] {item.get('sensor')}/{item.get('fact')}: {item.get('reason')}")
            print(json.dumps(payload, ensure_ascii=False, indent=1))
            return 0
    except ClaimsError as exc:
        print(f"check_claims: {exc}", file=sys.stderr)
        return 1
    build_parser().print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
