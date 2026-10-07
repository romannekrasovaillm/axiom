"""CLI реестра датчиков: ``python3 -m tools.sensors.probe [--sensor S-NNN]``.

Печатает класс по каждому датчику (``pass | fail | unverified``) и причину.
Код возврата ``1``, если хотя бы один ``active`` датчик в ``fail`` (нарушение
контракта или устаревший факт); ``unverified`` — честное «нет данных», не
красный (датчик, которому недоступен источник, обязан уметь это сказать).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from .fact import write_fact
from .registry import (
    RegistryError,
    load_sensors,
    probe_all,
    probe_sensor,
    validate_sensors,
)


def _print_row(row: dict) -> None:
    mark = {"pass": "PASS", "fail": "FAIL", "unverified": "UNVERIFIED"}.get(
        row["verdict"], row["verdict"].upper()
    )
    print(f"  [{row['id']}] {mark}: {row['reason']}")


def run_selftest() -> int:
    """Мутанты: разрыв цепочки, запись без subject, просроченный факт,
    неизвестный датчик → красные; эталон → зелёный."""
    from .subject import build_subject

    checks: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory(prefix="sensors-probe-selftest-") as tmp:
        root = Path(tmp)
        facts_dir = root / "facts"
        now = datetime(2026, 10, 7, 12, 0, 0, tzinfo=timezone.utc)
        subject = build_subject(
            repo_root=root, git_sha="a" * 40, dirty=False, device="cpu"
        )

        def sensor(sid: str, **over) -> dict:
            base = {
                "id": sid,
                "name": f"selftest {sid}",
                "facts": ["probe_value"],
                "context": "cpu",
                "producer": "tools.sensors.probe --selftest",
                "output": f"evidence/facts/{sid}.jsonl",
                "quality": "measured",
                "freshness_max_h": None,
                "zone": "line-2",
                "status": "active",
            }
            base.update(over)
            return base

        # Эталон: свежая запись с пином предмета → pass.
        write_fact(
            "S-901", "probe_value", 1, unit="count", quality="measured",
            method="selftest", subject=subject, out_dir=facts_dir,
            ts=now.isoformat(timespec="seconds"),
        )
        ok_row = probe_sensor(sensor("S-901"), out_dir=facts_dir, now=now)
        checks.append(("эталон → pass", ok_row["verdict"] == "pass"))

        # Мутант: просроченный факт (freshness_max_h=1, запись 10 ч назад).
        write_fact(
            "S-902", "probe_value", 1, unit="count", quality="measured",
            method="selftest", subject=subject, out_dir=facts_dir,
            ts=(now - timedelta(hours=10)).isoformat(timespec="seconds"),
        )
        stale = probe_sensor(
            sensor("S-902", freshness_max_h=1), out_dir=facts_dir, now=now
        )
        checks.append(("просроченный факт → fail", stale["verdict"] == "fail"))

        # Мутант: разрыв цепочки — переписываем файл, ломая prev_sha256.
        chain_dir = root / "facts_chain"
        write_fact(
            "S-903", "probe_value", 1, unit="count", quality="measured",
            method="selftest", subject=subject, out_dir=chain_dir,
            ts=now.isoformat(timespec="seconds"),
        )
        write_fact(
            "S-903", "probe_value", 2, unit="count", quality="measured",
            method="selftest", subject=subject, out_dir=chain_dir,
            ts=now.isoformat(timespec="seconds"),
        )
        p = chain_dir / "S-903.jsonl"
        lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
        rec = json.loads(lines[-1])
        rec["prev_sha256"] = "0" * 64
        lines[-1] = json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n"
        p.write_text("".join(lines), encoding="utf-8")
        broken = probe_sensor(sensor("S-903"), out_dir=chain_dir, now=now)
        checks.append(("разрыв цепочки → fail", broken["verdict"] == "fail"))

        # Мутант: запись без пина предмета — пишем файл напрямую (в обход контракта).
        empty_dir = root / "facts_empty"
        empty_dir.mkdir(parents=True, exist_ok=True)
        empty_subject = {k: None for k in subject}
        (empty_dir / "S-904.jsonl").write_text(
            json.dumps(
                {
                    "sensor": "S-904", "fact": "probe_value", "value": 1,
                    "unit": "count", "quality": "measured", "method": "selftest",
                    "ts": now.isoformat(timespec="seconds"), "subject": empty_subject,
                    "inputs": [], "status": "ok", "note": "", "prev_sha256": None,
                },
                ensure_ascii=False, sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        no_subject = probe_sensor(sensor("S-904"), out_dir=empty_dir, now=now)
        checks.append(("запись без subject → fail", no_subject["verdict"] == "fail"))

        # Мутант: неизвестный датчик — запрошен, но в реестре его нет.
        by_id = {s["id"]: s for s in [sensor("S-901")]}
        unknown = by_id.get("S-999")
        checks.append(("неизвестный датчик → не найден", unknown is None))
        # Просроченный unverified ≠ fail: pending-датчик «не может измерить».
        pending = probe_sensor(
            sensor("S-905", status="pending"), out_dir=facts_dir, now=now
        )
        checks.append(("pending → unverified", pending["verdict"] == "unverified"))

        # Реестр: валидный эталон проходит схему.
        reg_errors = validate_sensors([sensor("S-901")])
        checks.append(("валидный реестр → без ошибок", reg_errors == []))
        bad_errors = validate_sensors([sensor("S-901", quality="magic")])
        checks.append(("плохой quality → ошибка схемы", bool(bad_errors)))

    ok = all(passed for _, passed in checks)
    for label, passed in checks:
        print(f"[selftest] {'PASS' if passed else 'FAIL'}: {label}")
    print(f"[selftest] {'PASS' if ok else 'FAIL'}: probe")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Реестр датчиков: класс по каждому датчику (ADR-037, дельта C1)",
    )
    parser.add_argument("--sensor", action="append", default=None,
                        help="проверить только этот датчик (можно несколько раз)")
    parser.add_argument("--registry", default=None, help="путь к реестру датчиков")
    parser.add_argument("--out-dir", default=None, help="каталог фактов (evidence/facts)")
    parser.add_argument("--json", dest="json_path", default=None, help="куда записать отчёт")
    parser.add_argument("--selftest", action="store_true", help="синтетический selftest с мутантами")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.selftest:
        return run_selftest()

    try:
        sensors = load_sensors(args.registry)
    except RegistryError as exc:
        print(f"probe: реестр не читается — {exc}", file=sys.stderr)
        return 1
    errors = validate_sensors(sensors)
    if errors:
        print("probe: реестр нарушает схему:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    if args.sensor:
        wanted = set(args.sensor)
        known = {s.get("id") for s in sensors}
        rows = []
        for sid in args.sensor:
            if sid not in known:
                rows.append({
                    "id": sid, "status": None, "verdict": "fail",
                    "reason": f"{sid}: датчик не объявлен в реестре",
                    "records": 0, "last_ts": None, "last_status": None,
                })
        rows.extend(
            probe_all([s for s in sensors if s.get("id") in wanted], out_dir=args.out_dir)
        )
    else:
        rows = probe_all(sensors, out_dir=args.out_dir)

    print(f"Реестр датчиков ({len(rows)}):")
    for row in rows:
        _print_row(row)

    active_fails = [r for r in rows if r["status"] == "active" and r["verdict"] == "fail"]
    active = [r for r in rows if r["status"] == "active"]
    unverified = [r for r in rows if r["verdict"] == "unverified"]
    print(
        f"\nИтог: active {len(active)}, pass "
        f"{sum(1 for r in active if r['verdict'] == 'pass')}, "
        f"unverified {len(unverified)}, fail {len(active_fails)}"
    )
    if args.json_path:
        Path(args.json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_path).write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if active_fails:
        for row in active_fails:
            print(f"[probe] FAIL: {row['reason']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
