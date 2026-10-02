#!/usr/bin/env python3
"""Внешний килл-свитч претрейна на vast.ai (слой защиты №4, ADR-008/AD-8).

Независим от лупа: работает на хосте-операторе, читает состояние инстанса через
vast CLI и останавливает инстанс при выходе за смету — даже если сам луп завис.

Триггеры (первый сработавший):
  --spend-cap USD       суммарный spend инстанса (compute+storage+bw по vast) → create stop-файл
  --hard-cap USD        → vast stop instance (жёсткая остановка)
  --balance-min USD     баланс аккаунта vast ниже → vast stop (нуль кредитов = удаление данных!)
Запуск: nohup python3 tools/vast_watchdog.py --instance-id <id> --smeta evidence/budget/pretreain-l3.json
Требует установленный vast CLI и залогиненную сессию (vastai set api-key).
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

def sh(*args):
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[watchdog] CLI error: {r.stderr[:200]}", flush=True)
    return r.stdout

def instance_spend(instance_id: str) -> tuple[float | None, str]:
    """Суммарный spend инстанса в USD (compute+storage+bandwidth по данным vast)."""
    out = sh("vastai", "show", "instances", "--raw")
    try:
        for inst in json.loads(out or "[]"):
            if str(inst.get("id")) == str(instance_id):
                per = {
                    "compute": float(inst.get("gpu_cost_perhr", inst.get("dph_total", 0)) or 0),
                    "storage": float(inst.get("storage_cost_perday", inst.get("storage_gpu_cost_perhr", 0)) or 0),
                }
                # vast отдаёт накопленный счёт по-разному; консервативно: uptime * rate
                uptime_h = float(inst.get("uptime_sec", inst.get("cur_state", 0)) or 0) / 3600.0
                spend = per["compute"] * uptime_h
                return spend, json.dumps(per)
    except json.JSONDecodeError:
        pass
    return None, "instance not found"

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance-id", required=True)
    ap.add_argument("--smeta", default="evidence/budget/pretreain-l3.json")
    ap.add_argument("--spend-cap", type=float, required=True, help="USD: создать stop-файл")
    ap.add_argument("--hard-cap", type=float, required=True, help="USD: vast stop")
    ap.add_argument("--balance-min", type=float, default=20.0, help="USD: ниже — stop (нуль кредитов = удаление)")
    ap.add_argument("--stop-file", default="/tmp/axiom-pretrain-stop/stop")
    ap.add_argument("--interval", type=int, default=600)
    a = ap.parse_args()

    stop_file = Path(a.stop_file)
    print(f"[watchdog] инстанс {a.instance_id}: spend-cap {a.spend-cap if False else a.spend_cap}, hard-cap {a.hard_cap}, balance-min {a.balance_min}", flush=True)
    while True:
        spend, info = instance_spend(a.instance_id)
        if spend is not None:
            print(f"[watchdog] spend ≈ ${spend:.2f} ({info})", flush=True)
            if spend >= a.hard_cap:
                print(f"[watchdog] HARD CAP ${a.hard_cap} — vast stop", flush=True)
                sh("vastai", "stop", "instance", "--raw", a.instance_id)
                return 2
            if spend >= a.spend_cap and not stop_file.exists():
                stop_file.parent.mkdir(parents=True, exist_ok=True)
                stop_file.write_text(f"spend ${spend:.2f} >= cap ${a.spend_cap}")
                print(f"[watchdog] SPEND CAP — stop-файл создан {stop_file}", flush=True)
        # баланс аккаунта (нуль кредитов = auto-stop + риск удаления данных)
        bal = sh("vastai", "show", "user", "--raw")
        try:
            user = json.loads(bal or "{}")
            credit = float(user.get("credit", 0) or 0)
            if credit < a.balance_min:
                print(f"[watchdog] БАЛАНС ${credit:.2f} < ${a.balance_min} — vast stop (защита от удаления)", flush=True)
                sh("vastai", "stop", "instance", "--raw", a.instance_id)
                return 3
        except (json.JSONDecodeError, ValueError):
            print("[watchdog] баланс не прочитан (CLI?)", flush=True)
        time.sleep(a.interval)

if __name__ == "__main__":
    sys.exit(main())
