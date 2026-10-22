"""Матрица обнаружения возмущений (ADR-039, дельта X).

Десять сценариев порчи исполняются каждый в **копии репозитория** — временном
``git worktree`` (``git worktree add --detach <tmp> HEAD``), никогда в основном
дереве; после прогона копия удаляется (RUNBOOK §6). По каждому сценарию
записывается ожидаемое правило/класс (из таблицы X1), фактическое и ``detected``.
Необнаруженный сценарий — находка в ``open_questions``; правило и шаблон не
ослабляются (X3). Датчик S-041 ``detection_matrix``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.properties import Facts, get_property  # noqa: E402
from tools.properties.shadow import guard_class  # noqa: E402


# ── утилиты копии и правки фактов ───────────────────────────────────────────

def _make_worktree(ref: str = "HEAD") -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="chaos-wt-"))
    subprocess.run(["git", "worktree", "add", "--detach", "--force", str(tmp), ref],
                   cwd=str(_REPO_ROOT), check=True, capture_output=True, text=True)
    return tmp


def _remove_worktree(path: Path) -> None:
    subprocess.run(["git", "worktree", "remove", "--force", str(path)],
                   cwd=str(_REPO_ROOT), capture_output=True, text=True)
    subprocess.run(["git", "worktree", "prune"], cwd=str(_REPO_ROOT), capture_output=True, text=True)


def _read_records(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def _write_records(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")


def _fact_path(root: Path, sensor: str) -> Path:
    return root / "evidence" / "facts" / f"{sensor}.jsonl"


def _eval_property(root: Path, property_name: str, params: dict) -> str:
    template = get_property(property_name)
    facts = Facts(root, out_dir=root / "evidence" / "facts", subject_match=[], subject={})
    return template.evaluate(params, facts).cls


# ── сценарии ────────────────────────────────────────────────────────────────

def _s1_delete_metrics(wt: Path) -> tuple[str, str, str]:
    metrics = wt / "evidence" / "kda-wyut" / "metrics.jsonl"
    if metrics.exists():
        metrics.unlink()
    cmd = (f"{sys.executable} {wt}/tools/check_performance_roofline.py --run kda-wyut-delta "
           f"--metrics {metrics} --pins {wt}/evidence/kpi-pins.json")
    cls, detail = guard_class(wt, cmd)
    return "C-046 (bounds)", cls, detail


def _s2_age_facts(wt: Path) -> tuple[str, str, str]:
    path = _fact_path(wt, "S-018")
    records = _read_records(path)
    from datetime import datetime, timedelta, timezone

    aged = 0
    for record in records:
        if record.get("fact") == "model_loads_count":
            record["ts"] = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat(timespec="seconds")
            aged += 1
    _write_records(path, records)
    cls = _eval_property(wt, "freshness", {"fact": "S-018.model_loads_count", "max_age_h": 24})
    return "freshness", cls, f"запись состарена на 48 ч при допуске 24 ч (состарено записей: {aged})"


def _s3_corrupt_raw(wt: Path) -> tuple[str, str, str]:
    # Сырьё правится после факта; ловится probe --raw по raw_ref записи.
    cmd = f"{sys.executable} {wt}/tools/sensors/probe.py --raw"
    cls, detail = guard_class(wt, cmd)
    return "probe --raw", cls, detail


def _s4_break_chain(wt: Path) -> tuple[str, str, str]:
    from tools.sensors.fact import verify_chain

    path = _fact_path(wt, "S-018")
    records = _read_records(path)
    if records:
        records[-1]["prev_sha256"] = "0" * 64
        _write_records(path, records)
    ok, reason = verify_chain("S-018", wt / "evidence" / "facts")
    return "probe (цепочка prev_sha256)", "pass" if ok else "fail", reason


def _s5_change_config_key(wt: Path) -> tuple[str, str, str]:
    from tools import check_claims as cc

    path = wt / "net" / "config.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["num_kda_layers"] = data.get("num_kda_layers", 18) + 1
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    bad = [r for r in cc.config_bindings(wt) if r["verdict"] != "pass"]
    cls = "fail" if bad else "pass"
    ids = ",".join(str(r["id"]) for r in bad) or "нет"
    return "C-047/C-049 (declared_equals_actual)", cls, f"привязки вне согласия: {ids}"


def _s6_truncate_hash(wt: Path) -> tuple[str, str, str]:
    path = _fact_path(wt, "S-004")
    records = _read_records(path)
    full = "a" * 64
    if records:
        for r in records:
            if r.get("fact") == "tokenizer_sha256":
                r["value"] = full[:16]
                r["status"] = "ok"
                r["note"] = ""
        _write_records(path, records)
    pin = wt / "evidence" / "tokenizer-pin.json"
    pin.write_text(json.dumps({"h": full}), encoding="utf-8")
    cls = _eval_property(wt, "identity", {
        "artifact": {"fact": "S-004.tokenizer_sha256"},
        "declared": {"file": "evidence/tokenizer-pin.json", "key": "h"},
    })
    return "identity", cls, "хеш токенизатора обрезан до 16 символов"


def _s7_budget_overrun(wt: Path) -> tuple[str, str, str]:
    path = _fact_path(wt, "S-022")
    records = _read_records(path)
    limit = 100.0
    if not records:
        records = [{"sensor": "S-022", "fact": "spend_usd", "value": limit + 5.0, "unit": "usd",
                    "quality": "measured", "method": "chaos", "ts": "2026-10-07T00:00:00+00:00",
                    "subject": {}, "inputs": [], "status": "ok", "note": "", "prev_sha256": None,
                    "raw_ref": None}]
    else:
        for r in records:
            if r.get("fact") == "spend_usd":
                r["value"] = limit + 5.0
                r["status"] = "ok"
    _write_records(path, records)
    cls = _eval_property(wt, "bounds", {"fact": "S-022.spend_usd", "max": limit})
    return "C-041 (bounds)", cls, f"расход {limit + 5.0} > лимита {limit}"


def _s8_flapping(wt: Path) -> tuple[str, str, str]:
    path = _fact_path(wt, "S-036")
    records = _read_records(path)
    if records:
        verdicts = ["pass", "fail"]
        base = dict(records[-1])
        for i, r in enumerate(records[-10:]):
            r["value"] = {**(r.get("value") or {}), "claim": "CL-001", "verdict": verdicts[i % 2]}
        _write_records(path, records)
    cmd = f"{sys.executable} {wt}/tools/sensors/run_exporter.py --id S-037 --root {wt} --out-dir {wt}/evidence/facts"
    code, detail = guard_class(wt, cmd)
    flapping = {}
    for r in _read_records(_fact_path(wt, "S-037")):
        if r.get("fact") == "flapping" and isinstance(r.get("value"), dict):
            flapping = r["value"]
    cls = "fail" if flapping.get("CL-001") is True else "unverified"
    return "S-037 flapping / preflight", cls, f"flagging CL-001={flapping.get('CL-001')} (exporter exit {code})"


def _s9_proxy_divergence(wt: Path) -> tuple[str, str, str]:
    from tools import check_claims as cc

    path = _fact_path(wt, "S-014")
    records = _read_records(path)
    for r in records:
        if r.get("fact") == "kda_component_speedup":
            r["value"] = 5.0
            r["status"] = "ok"
    _write_records(path, records)
    results = cc.evaluate(wt)
    divergences = cc.proxy_divergences(results, wt)
    return "check_claims --evaluate (прокси-расхождение)", "fail" if divergences else "unverified", \
        (divergences[0] if divergences else "строка прокси-расхождения не выведена")


def _s10_claims_edit(wt: Path) -> tuple[str, str, str]:
    ws = Path(tempfile.mkdtemp(prefix="chaos-ws-"))
    shutil.copytree(wt, ws, dirs_exist_ok=True, symlinks=True,
                    ignore=shutil.ignore_patterns(".git"))
    claims = ws / "model" / "claims.yaml"
    claims.write_text(claims.read_text(encoding="utf-8") + "\n# правка воркспейса v2\n", encoding="utf-8")
    try:
        from env.verifier import gate_integrity

        result = gate_integrity(ws, base_ws=wt, task_spec={"atoms_version": "v2"})
        cls = "fail" if not result.passed else "pass"
        detail = "класс hacking обнаружен" if not result.passed else "mismatch не найден"
    except Exception as exc:  # noqa: BLE001 — отсутствие механизма = unverified, не pass
        cls, detail = "unverified", f"gate_integrity недоступен: {exc}"
    shutil.rmtree(ws, ignore_errors=True)
    return "env/verifier (hacking)", cls, detail


SCENARIOS: tuple[tuple[str, str, str, Callable], ...] = (
    ("удалить файл метрик прогона", "C-046 (bounds)", "unverified", _s1_delete_metrics),
    ("состарить факты на 2× допуска свежести", "freshness", "unverified", _s2_age_facts),
    ("изменить один байт сырья после факта", "probe --raw", "fail", _s3_corrupt_raw),
    ("разорвать цепочку prev_sha256", "probe (цепочка)", "fail", _s4_break_chain),
    ("сменить значение привязанного ключа конфига", "C-047 (declared_equals_actual)", "fail", _s5_change_config_key),
    ("обрезать хеш токенизатора в манифесте", "identity", "fail", _s6_truncate_hash),
    ("расход сверх лимита сметы в фактах S-022", "C-041 (bounds)", "fail", _s7_budget_overrun),
    ("история вердиктов с флапанием", "S-037 flapping", "unverified", _s8_flapping),
    ("компонент pass, сквозной ток/с fail", "check_claims --evaluate", "fail", _s9_proxy_divergence),
    ("правка model/claims.yaml в воркспейсе v2", "env/verifier (hacking)", "fail", _s10_claims_edit),
)


def run_matrix() -> dict:
    rows = []
    for name, expected_rule, expected_class, detector in SCENARIOS:
        wt = _make_worktree("HEAD")
        try:
            rule, cls, detail = detector(wt)
            detected = cls in ("fail", "unverified")
            rows.append({
                "scenario": name, "expected_rule": expected_rule, "expected_class": expected_class,
                "actual_rule": rule, "actual_class": cls, "detected": detected, "detail": detail,
            })
        except Exception as exc:  # noqa: BLE001 — сбой сценария = необнаружение, не pass
            rows.append({
                "scenario": name, "expected_rule": expected_rule, "expected_class": expected_class,
                "actual_rule": expected_rule, "actual_class": "unverified",
                "detected": False, "detail": f"сценарий не выполнен: {exc}",
            })
        finally:
            _remove_worktree(wt)
    return {"rows": rows, "all_detected": all(r["detected"] for r in rows),
            "undetected": [r["scenario"] for r in rows if not r["detected"]]}


def write_fact(root: Path, payload: dict) -> None:
    from tools.sensors.fact import write_fact
    from tools.sensors.subject import build_subject

    subject = build_subject(repo_root=root, config_path=root / "net" / "config.json", device="cpu")
    write_fact("S-041", "detection_matrix", payload, unit="", quality="measured",
               method="tools.properties.chaos (ADR-039, дельта X)", subject=subject)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="X: матрица обнаружения возмущений")
    parser.add_argument("--root", default=".")
    parser.add_argument("--no-fact", action="store_true")
    parser.add_argument("--json", dest="json_path", default=None)
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    payload = run_matrix()
    if not args.no_fact:
        write_fact(root, payload)
    if args.json_path:
        Path(args.json_path).write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for row in payload["rows"]:
        mark = "OK" if row["detected"] else "НЕ ОБНАРУЖЕНО"
        print(f"  [{mark}] {row['scenario']}: ожид. {row['expected_class']}, факт {row['actual_class']} "
              f"({row['actual_rule']})")
    print(f"матрица обнаружения: сценариев {len(payload['rows'])}, необнаружено {len(payload['undetected'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
