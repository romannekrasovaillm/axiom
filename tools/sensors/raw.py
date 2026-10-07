"""Происхождение факта до сырья (ADR-038, дельта F).

Факт ``measured``/``wrapped`` обязан прослеживаться до сырья: у датчика в
``model/sensors.yaml`` объявлен блок ``raw`` (``path``/``format``/
``retention_days``/``in_git``), а у записи факта — ``raw_ref`` ``{path, sha256,
selector}``. ``python3 -m tools.sensors.probe --raw`` сверяет:

* сырьё существует (недоступное внешнее — ``unverified``, не ``fail``);
* sha256 выбранного фрагмента совпадает с ``raw_ref.sha256``;
* сырьё изменено после факта (sha не совпал) → ``fail`` — подмена истории.

Здесь живут разрешение пути (в т.ч. glob и ``~``), селектор фрагмента
(``lines A-B`` | ``key <dotted>`` | пустой = весь файл) и сама проверка. Файл —
``stdlib-only``: исполняется в воркспейсе без сети и GPU.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .fact import read_records, validate_raw_ref
from .subject import REPO_ROOT

_LINES_RE = re.compile(r"^lines\s+(\d+)\s*-\s*(\d+)$", re.IGNORECASE)


def parse_selector(selector: Any) -> tuple[str, Any]:
    """Селектор фрагмента сырья → ``(вид, параметр)``.

    Поддержано: ``''``/``None`` (весь файл), ``lines A-B`` (1-based, включительно),
    ``key a.b.c`` (значение JSON по пути). Неизвестный селектор — ``('unknown', text)``
    (проверка обязана это заметить, а не угадать фрагмент).
    """
    if selector is None or (isinstance(selector, str) and not selector.strip()):
        return ("file", None)
    text = str(selector).strip()
    m = _LINES_RE.match(text)
    if m:
        start, end = int(m.group(1)), int(m.group(2))
        if start >= 1 and end >= start:
            return ("lines", (start, end))
        return ("invalid", text)
    if text.lower().startswith("key "):
        return ("key", text[4:].strip())
    return ("unknown", text)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fingerprint(path: Path, selector: Any) -> Optional[str]:
    """sha256 выбранного фрагмента. ``None`` — селектор неприменим/файл не читается."""
    kind, param = parse_selector(selector)
    try:
        if kind == "file":
            return _sha256_bytes(path.read_bytes())
        raw = path.read_text(encoding="utf-8")
        if kind == "lines":
            start, end = param
            lines = raw.splitlines()
            if end > len(lines):
                return None
            chunk = "\n".join(lines[start - 1 : end]) + "\n"
            return _sha256_bytes(chunk.encode("utf-8"))
        if kind == "key":
            data = json.loads(raw)
            cur: Any = data
            for part in str(param).split("."):
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    return None
            canonical = json.dumps(cur, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            return _sha256_bytes(canonical.encode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return None


def resolve_paths(raw_path: str, repo_root: str | Path = REPO_ROOT) -> list[Path]:
    """Разрешает путь/glob сырья (относительно корня репозитория; ``~`` и ``**`` учтены).

    Разбор через :mod:`glob` (``recursive=True``): permission-denied на сетевом
    монтировании не роняет проверку, а даёт пустой список (честное unverified).
    """
    text = str(raw_path).strip()
    if not text:
        return []
    expanded = os.path.expanduser(text)
    if not os.path.isabs(expanded):
        expanded = str(Path(repo_root) / expanded)
    if any(ch in expanded for ch in "*?["):
        return sorted(Path(p) for p in glob.glob(expanded, recursive=True))
    return [Path(expanded)]


def existing_paths(raw_path: str, repo_root: str | Path = REPO_ROOT) -> list[Path]:
    found: list[Path] = []
    for candidate in resolve_paths(raw_path, repo_root):
        try:
            if candidate.exists():
                found.append(candidate)
        except OSError:
            continue  # permission denied / мёртвое монтирование — не красный, а unverified
    return found


def probe_records(sensor_id: str, out_dir: Any = None) -> list[dict[str, Any]]:
    """Записи факта датчика, несущие непустой ``raw_ref``."""
    try:
        records = read_records(sensor_id, out_dir)
    except Exception:  # noqa: BLE001 — битый файл ловит probe контракта, не этот режим
        return []
    return [r for r in records if isinstance(r.get("raw_ref"), dict)]


def probe_raw_sensor(
    sensor: dict[str, Any],
    *,
    out_dir: Any = None,
    repo_root: str | Path = REPO_ROOT,
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    """Исход проверки происхождения по одному датчику: ``pass | fail | unverified``."""
    sid = str(sensor.get("id", "?"))
    raw = sensor.get("raw")
    row: dict[str, Any] = {
        "id": sid,
        "verdict": "unverified",
        "reason": "",
        "raw_path": None,
        "raw_present": False,
        "in_git": None,
        "records_with_raw_ref": 0,
        "records_total": 0,
    }
    if not isinstance(raw, dict) or not raw.get("path"):
        note = ""
        if isinstance(raw, dict):
            note = str(raw.get("note") or "")
        row["reason"] = "raw не применим (нет отдельного сырья)" + (f": {note}" if note else "")
        return row

    raw_path = str(raw["path"])
    row["raw_path"] = raw_path
    row["in_git"] = bool(raw.get("in_git"))
    paths = existing_paths(raw_path, repo_root)
    row["raw_present"] = bool(paths)
    if not paths:
        where = "в git" if raw.get("in_git") else "вне git"
        row["reason"] = (
            f"сырьё недоступно на этой машине ({where}): {raw_path} — "
            "прослеживание невозможно (честное unverified)"
        )
        return row

    # Подмена истории: у факта зафиксирован sha256 фрагмента — он обязан совпасть.
    mismatches: list[str] = []
    for record in probe_records(sid, out_dir):
        ref = record["raw_ref"]
        errs = validate_raw_ref(ref)
        if errs:
            mismatches.append(f"{sid}/{record.get('fact')}: " + "; ".join(errs))
            continue
        ref_paths = existing_paths(str(ref.get("path")), repo_root)
        if not ref_paths:
            row["reason"] = (
                f"{sid}/{record.get('fact')}: сырьё из raw_ref недоступно: {ref.get('path')} "
                "(честное unverified)"
            )
            return row
        actual = fingerprint(ref_paths[0], ref.get("selector"))
        if actual is None:
            row["reason"] = (
                f"{sid}/{record.get('fact')}: селектор {ref.get('selector')!r} "
                "неприменим к сырью (unverified)"
            )
            return row
        if actual != str(ref.get("sha256")):
            mismatches.append(
                f"{sid}/{record.get('fact')}: sha256 фрагмента {actual[:12]}… ≠ "
                f"зафиксированного {str(ref.get('sha256'))[:12]}… (сырьё изменено после факта)"
            )

    try:
        row["records_total"] = len(read_records(sid, out_dir))
    except Exception:  # noqa: BLE001
        row["records_total"] = 0
    row["records_with_raw_ref"] = len(probe_records(sid, out_dir))

    if mismatches:
        row["verdict"] = "fail"
        row["reason"] = "подмена истории сырья: " + "; ".join(mismatches)
        return row

    retention = raw.get("retention_days")
    note = ""
    if isinstance(retention, (int, float)) and row["records_with_raw_ref"] == 0:
        note = f"; retention_days={retention}"
    row["verdict"] = "pass"
    row["reason"] = (
        f"сырьё на месте: {paths[0]}"
        + (f" (glob: {len(paths)} файлов)" if len(paths) > 1 else "")
        + f"; raw_ref зафиксирован в {row['records_with_raw_ref']}/{row['records_total']} записях"
        + (" (факты до дельты F — происхождение не записано)" if row["records_with_raw_ref"] == 0 else "")
        + note
    )
    return row


def probe_raw_all(
    sensors: list[dict[str, Any]],
    *,
    out_dir: Any = None,
    repo_root: str | Path = REPO_ROOT,
    now: Optional[datetime] = None,
) -> list[dict[str, Any]]:
    return [probe_raw_sensor(s, out_dir=out_dir, repo_root=repo_root, now=now) for s in sensors]


def run_selftest() -> int:
    """Мутанты: подмена сырья краснеет, недоступное внешнее — unverified, raw null — unverified."""
    import tempfile

    from .fact import write_fact
    from .subject import build_subject

    checks: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory(prefix="sensors-raw-selftest-") as tmp:
        root = Path(tmp)
        repo = root / "repo"
        repo.mkdir()
        raw = repo / "metrics.jsonl"
        raw.write_text(
            "".join(json.dumps({"step": i, "tok_s": 100 + i}) + "\n" for i in range(30)),
            encoding="utf-8",
        )
        out = root / "facts"
        subject = build_subject(repo_root=repo, git_sha="a" * 40, dirty=False, device="cpu")
        selector = "lines 11-30"
        good_sha = fingerprint(raw, selector)
        assert good_sha is not None

        write_fact(
            "S-950", "tok_s_median_window", 114.5, unit="tok_s", quality="wrapped",
            method="selftest", subject=subject, out_dir=out,
            raw_ref={"path": "metrics.jsonl", "sha256": good_sha, "selector": selector},
        )
        sensor = {"id": "S-950", "raw": {"path": "metrics.jsonl", "format": "jsonl", "in_git": True, "retention_days": None}}
        ok = probe_raw_sensor(sensor, out_dir=out, repo_root=repo)
        checks.append(("сырьё и sha совпали → pass", ok["verdict"] == "pass"))
        checks.append(("raw_ref учтён", ok["records_with_raw_ref"] == 1))

        # Подмена истории: сырьё изменилось после факта → fail.
        raw.write_text(
            "".join(json.dumps({"step": i, "tok_s": 999 + i}) + "\n" for i in range(30)),
            encoding="utf-8",
        )
        tampered = probe_raw_sensor(sensor, out_dir=out, repo_root=repo)
        checks.append(("сырьё изменено после факта → fail", tampered["verdict"] == "fail"))
        checks.append(("причина называет подмену", "подмена истории" in tampered["reason"]))

        # Недоступное внешнее сырьё → unverified, не fail.
        missing = probe_raw_sensor(
            {"id": "S-951", "raw": {"path": "/нет/такого/файла.jsonl", "format": "jsonl", "in_git": False, "retention_days": 90}},
            out_dir=out, repo_root=repo,
        )
        checks.append(("внешнее сырьё недоступно → unverified", missing["verdict"] == "unverified"))
        checks.append(("недоступное не red", "fail" not in missing["reason"]))

        # raw null (чистая арифметика) → unverified, но не ошибка.
        null_raw = probe_raw_sensor(
            {"id": "S-003", "raw": None}, out_dir=out, repo_root=repo
        )
        checks.append(("raw null → unverified «не применим»", null_raw["verdict"] == "unverified"))
        checks.append(("raw null назван в причине", "не применим" in null_raw["reason"]))

        # Селектор key по JSON.
        cfg = repo / "config.json"
        cfg.write_text(json.dumps({"model": {"vocab_size": 160000}}), encoding="utf-8")
        key_sha = fingerprint(cfg, "key model.vocab_size")
        checks.append(("селектор key даёт детерминированный sha", isinstance(key_sha, str) and len(key_sha) == 64))
        checks.append(("другой ключ — другой sha", fingerprint(cfg, "key model.vocab_size") != fingerprint(cfg, "")))

    ok_all = all(passed for _, passed in checks)
    for label, passed in checks:
        print(f"[selftest] {'PASS' if passed else 'FAIL'}: {label}")
    print(f"[selftest] {'PASS' if ok_all else 'FAIL'}: raw")
    return 0 if ok_all else 1


__all__ = [
    "parse_selector",
    "fingerprint",
    "resolve_paths",
    "existing_paths",
    "probe_records",
    "probe_raw_sensor",
    "probe_raw_all",
    "run_selftest",
]
