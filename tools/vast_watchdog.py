#!/usr/bin/env python3
"""Внешний килл-свитч претрейна на vast.ai (слой защиты №4, ADR-008/AD-8).

Независим от лупа: останавливает прогон при выходе за смету — даже если сам луп
завис.  Два режима:

* ``--host`` (рекомендуется) — на **домашнем хосте**: читает ставку/баланс/state
  инстанса через vast CLI и **синхронизированную копию** ``metrics.jsonl``
  (pull-модель, runbook §4.5), а не путь на инстансе.  Здесь же жёсткий стоп.
* без ``--host`` — на инстансе: ключа vast там нет (D4, риск угона), поэтому
  читается только локальный ``metrics.jsonl``, ставка задаётся явно
  ``--usd-per-gpu-hour``, и при выходе за spend-cap создаётся stop-файл.

Триггеры (первый сработавший):
  --spend-cap USD       суммарный spend инстанса → создать stop-файл (луп встанет сам)
  --hard-cap USD        → vast stop instance (только хост-режим)
  --balance-min USD     баланс аккаунта vast ниже → vast stop (только хост-режим)

Запуск (хост):
    nohup python3 tools/vast_watchdog.py --host --instance-id <id> \
      --metrics ~/gb10-shared/runs/pretreain-l3/metrics.jsonl \
      --spend-cap 225 --hard-cap 260 --balance-min 20 > /tmp/watchdog.log 2>&1 &
Требует установленный vast CLI и залогиненную сессию (vastai set api-key).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

#: Хост-дефолт: синхронизированная копия метрик (runbook §4.5, pull с инстанса).
DEFAULT_HOST_METRICS = str(Path.home() / "gb10-shared" / "runs" / "pretreain-l3" / "metrics.jsonl")


def sh(*args):
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[watchdog] CLI error: {r.stderr[:200]}", flush=True)
    return r.stdout


def instance_info(instance_id: str) -> dict | None:
    """Запись инстанса из vast (dph_total, cur_state, id) или None, если не найден.

    None — это не «нулевой spend», а «сторожить нечего»: инстанс исчез из выдачи
    (destroy, смена id после пересоздания, чужой API-ключ).  Различать эти случаи
    обязан вызывающий — молчаливое продолжение с нулём выдало бы зелёный вердикт
    при отсутствующем стороже.
    """
    out = sh("vastai", "show", "instances", "--raw")
    try:
        instances = json.loads(out or "[]")
    except json.JSONDecodeError:
        return None
    for inst in instances:
        if str(inst.get("id")) == str(instance_id):
            return inst
    return None


def instance_rate(info: dict | None) -> float | None:
    """Ставка инстанса USD/ч (dph_total). None = unknown → watchdog НЕ молчит."""
    if not info:
        return None
    v = info.get("dph_total")
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def metrics_gpu_hours(metrics_path: str) -> float:
    """Фактические GPU-часы из метрик лупа (schema pretrain-metrics/v1 — верифицированный источник).

    ``gpu_hours`` в метриках накопительный по прогону (К3: сумма по ногам), поэтому
    spend считается от всего прогона, а не от последней ноги.
    """
    last = 0.0
    try:
        with open(metrics_path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    value = d.get("gpu_hours")
                    if value is not None:
                        last = float(value)
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
    except FileNotFoundError:
        return 0.0
    return last


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance-id", required=True)
    ap.add_argument("--host", action="store_true",
                    help="режим домашнего хоста: vast CLI + синхронизированные метрики")
    ap.add_argument("--smeta", default="evidence/budget/pretreain-l3.json")
    ap.add_argument("--spend-cap", type=float, required=True, help="USD: создать stop-файл")
    ap.add_argument("--hard-cap", type=float, default=None, help="USD: vast stop (хост-режим)")
    ap.add_argument("--balance-min", type=float, default=20.0,
                    help="USD: ниже — stop (нуль кредитов = удаление; хост-режим)")
    ap.add_argument("--stop-file", default="/tmp/axiom-pretrain-stop/stop")
    ap.add_argument("--metrics", default=None,
                    help="метрики лупа: в хост-режиме — синхронизированная копия "
                         f"(по умолчанию {DEFAULT_HOST_METRICS}), иначе локальный файл инстанса")
    ap.add_argument("--usd-per-gpu-hour", type=float, default=None,
                    help="ставка аренды (обязательна без --host: ключа vast на инстансе нет)")
    ap.add_argument("--interval", type=int, default=600)
    return ap.parse_args()


def main() -> int:
    a = _parse_args()
    stop_file = Path(a.stop_file)

    if a.host:
        metrics_path = a.metrics or DEFAULT_HOST_METRICS
    else:
        if a.usd_per_gpu_hour is None:
            print("[watchdog] ОТКАЗ: без --host нужна явная --usd-per-gpu-hour "
                  "(на инстансе ключа vast нет — D4)", flush=True)
            return 5
        if a.metrics is None:
            print("[watchdog] ОТКАЗ: без --host укажите --metrics (локальный файл инстанса)", flush=True)
            return 5
        metrics_path = a.metrics

    print(
        f"[watchdog] режим {'ХОСТ' if a.host else 'ИНСТАНС'}, инстанс {a.instance_id}: "
        f"spend-cap {a.spend_cap}, hard-cap {a.hard_cap}, метрики {metrics_path}",
        flush=True,
    )

    while True:
        info = instance_info(a.instance_id) if a.host else None
        if a.host and info is None:
            # Stale-инстанс: сторожить нечего (destroy / смена id).  Это НЕ зелёный
            # вердикт — это отказ сторожа, и он обязан быть громким (D2/D4).
            print(
                f"[watchdog] STALE: инстанс {a.instance_id} не найден в `vastai show instances` — "
                "id сменился или инстанс удалён; сторож больше ничего не охраняет, выход",
                flush=True,
            )
            return 4
        state = (info or {}).get("cur_state")
        rate = instance_rate(info) if a.host else float(a.usd_per_gpu_hour)
        gh = metrics_gpu_hours(metrics_path)

        if rate is None:
            print("[watchdog] СТАВКА UNKNOWN (dph_total недоступен) — НЕ считаю spend молча", flush=True)
        else:
            spend = gh * rate
            print(
                f"[watchdog] state={state} spend ≈ ${spend:.2f} ({gh:.2f} ч × ${rate:.2f}/ч)",
                flush=True,
            )
            if a.host and a.hard_cap is not None and spend >= a.hard_cap:
                print(f"[watchdog] HARD CAP ${a.hard_cap} — vast stop", flush=True)
                sh("vastai", "stop", "instance", "--raw", a.instance_id)
                return 2
            if spend >= a.spend_cap and not stop_file.exists():
                stop_file.parent.mkdir(parents=True, exist_ok=True)
                stop_file.write_text(f"spend ${spend:.2f} >= cap ${a.spend_cap}")
                print(f"[watchdog] SPEND CAP — stop-файл создан {stop_file}", flush=True)

        # баланс аккаунта — только хост-режим (на инстансе ключа нет, D4)
        if a.host:
            bal = sh("vastai", "show", "user", "--raw")
            try:
                user = json.loads(bal or "{}")
                if "credit" not in user:
                    print("[watchdog] поле credit недоступно — пропуск проверки баланса (fail-loud)", flush=True)
                    time.sleep(a.interval)
                    continue
                credit = float(user["credit"] or 0)
                if credit < a.balance_min:
                    print(f"[watchdog] БАЛАНС ${credit:.2f} < ${a.balance_min} — vast stop (защита от удаления)", flush=True)
                    sh("vastai", "stop", "instance", "--raw", a.instance_id)
                    return 3
            except (json.JSONDecodeError, ValueError, TypeError):
                print("[watchdog] баланс не прочитан (CLI?)", flush=True)

        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
