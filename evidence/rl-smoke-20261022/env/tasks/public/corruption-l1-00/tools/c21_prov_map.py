#!/usr/bin/env python3
"""c21_prov_map.py — тонкая провенанс-карта шага (цикл-21, ADR-054 Am.33).

Отличия от c19_prov_map.py:
  * op_name берутся из ДВУХ модулей дампа: `jit_loss_fn` (модель+loss+backward)
    и `jit_step` (оптимизатор); ядра, найденные только в модуле шага,
    помечаются `opt` (метка probe_opt на jit-границе может не пронести имя);
  * компоненты: forward / ntp_ce / mtp / kda / mla / moe / mlp / attnres / opt;
  * печатает также топ-ядра внутри каждого компонента (для выбора кернела G2).

Запуск: python3 c21_prov_map.py <trace.sqlite> <dump_dir> [t_min t_max]
"""
import collections
import glob
import os
import re
import sqlite3
import sys


def hlo_name(kernel: str) -> str:
    m = re.match(r"^(.*?)_(\d+)$", kernel)
    return f"{m.group(1)}.{m.group(2)}" if m else kernel


def module_map(dump: str, pattern: str):
    cands = glob.glob(f"{dump}/{pattern}*after_optimizations.txt")
    out = {}
    for path in cands:
        pat = re.compile(r"^\s*%([A-Za-z0-9_.]+)\s*=\s*.*?op_name=\"([^\"]*)\"", re.MULTILINE)
        text = open(path, encoding="utf-8", errors="replace").read()
        for m in pat.finditer(text):
            out.setdefault(m.group(1), m.group(2))
    return out


PROBES = ("ntp_ce", "mtp", "kda", "mla_dense", "mla", "moe", "mlp", "attnres", "forward")


def probe_of(op: str, in_step: bool) -> str:
    for name in PROBES:
        if f"probe_{name}" in op:
            return name
    return "opt" if in_step else "—"


def main():
    trace, dump = sys.argv[1], sys.argv[2]
    lo = int(float(sys.argv[3]) * 1e9) if len(sys.argv) > 3 else 0
    hi = int(float(sys.argv[4]) * 1e9) if len(sys.argv) > 4 else 2**62
    names = {i: v for i, v in sqlite3.connect(trace).execute("SELECT id,value FROM StringIds")}
    loss_map = module_map(dump, "*jit_loss_fn.")
    step_map = module_map(dump, "*jit_step.")
    print(f"# оп-имена: loss_fn {len(loss_map)}, jit_step {len(step_map)}, "
          f"пересечение {len(set(loss_map) & set(step_map))}")

    con = sqlite3.connect(trace)
    spans = [(st, en, txt) for st, en, txt in con.execute(
        "SELECT start,end,text FROM NVTX_EVENTS WHERE text LIKE 'Thunk:%' AND end IS NOT NULL "
        "AND start>=? AND end<=?", (lo, hi))]
    spans.sort()

    def span_at(t):
        best = None
        for st, en, txt in spans:
            if st <= t <= en and (best is None or (en - st) < (best[1] - best[0])):
                best = (st, en, txt)
        return best[2].replace("Thunk:#hlo_op=", "").rstrip("#") if best else "вне Thunk"

    launches = {}
    for cid, st, en in con.execute(
        "SELECT r.correlationId,r.start,r.end FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
        "JOIN StringIds s ON s.id=r.nameId WHERE s.value='cuGraphLaunch' "
        "AND r.start>=? AND r.end<=?", (lo, hi)
    ):
        launches[cid] = (st, en)

    per = collections.Counter()
    per_n = collections.Counter()
    top = collections.defaultdict(collections.Counter)
    names_k = {}
    total = 0
    for cid, st, en, nid in con.execute(
        "SELECT correlationId,start,end,demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL "
        "WHERE start>=? AND end<=?", (lo, hi)
    ):
        d = en - st
        total += d
        k = names.get(nid, str(nid))
        key = hlo_name(k)
        if key in loss_map:
            p = probe_of(loss_map[key], False)
        elif key in step_map:
            p = probe_of(step_map[key], True)
        else:
            p = "—"
        per[p] += d
        per_n[p] += 1
        top[p][k] += d
        names_k[nid] = d
    print(f"\n# GPU всего {total/1e9:.2f}s, ядер {sum(per_n.values())}")
    for p, d in per.most_common():
        print(f"{d/1e9:7.2f}s  {per_n[p]:7d} яд  {100*d/total:5.1f}%  {p}")
    print("\n== топ-ядра по компонентам")
    for p, d in per.most_common():
        if p in ("—", "opt"):
            continue
        items = ", ".join(f"{k[:44]} {v/1e6:.0f}ms" for k, v in top[p].most_common(4))
        print(f"[{p}] {items}")
    print("\n== неотнесённое (топ-8)")
    for k, d in top["—"].most_common(8):
        print(f"{d/1e6:8.0f}ms  {k[:100]}")


if __name__ == "__main__":
    raise SystemExit(main())
