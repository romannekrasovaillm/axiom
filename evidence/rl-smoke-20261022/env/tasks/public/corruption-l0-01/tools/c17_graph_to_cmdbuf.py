#!/usr/bin/env python3
"""graph_to_cmdbuf.py — привязка CUDA-графов к NVTX-спанам (цикл-17).

В nsys-sqlite с --cuda-graph-trace=node ядра внутри графа несут correlationId
запуска cuGraphLaunch. Группируем ядра по correlationId, находим объемлющий
NVTX-спан Thunk (command_buffer_NNNN) и печатаем «спан → GPU-время, ядер,
графов, топ-классы». Это даёт ранжированные адреса кернел-войны на клетке B=4.

Запуск: python3 graph_to_cmdbuf.py <trace.sqlite> [t_min t_max]
"""
import collections
import sqlite3
import sys


def main():
    con = sqlite3.connect(sys.argv[1])
    tmin = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    tmax = float(sys.argv[3]) if len(sys.argv) > 3 else 1e18
    lo, hi = int(tmin * 1e9), int(tmax * 1e9)
    names = {i: v for i, v in con.execute("SELECT id, value FROM StringIds")}

    # запуски графов (runtime) в окне
    launches = {}
    for cid, st, en in con.execute(
        "SELECT r.correlationId, r.start, r.end FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
        "JOIN StringIds s ON s.id=r.nameId WHERE s.value='cuGraphLaunch' "
        "AND r.start>=? AND r.end<=?", (lo, hi)
    ):
        launches[cid] = (st, en)

    # ядра в окне, сгруппированные по correlationId
    per = collections.defaultdict(lambda: {"n": 0, "dur": 0, "kern": collections.Counter()})
    for cid, st, en, nid in con.execute(
        "SELECT correlationId, start, end, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL "
        "WHERE start>=? AND end<=?", (lo, hi)
    ):
        if cid not in launches:
            continue
        p = per[cid]
        p["n"] += 1
        p["dur"] += en - st
        p["kern"][nid] += en - st

    spans = [(st, en, txt) for st, en, txt in con.execute(
        "SELECT start,end,text FROM NVTX_EVENTS WHERE text LIKE 'Thunk:%' AND end IS NOT NULL "
        "AND start>=? AND end<=?", (lo, hi))]
    spans.sort()

    def label_at(t):
        best = None
        for st, en, txt in spans:
            if st <= t <= en and (best is None or (en - st) < (best[1] - best[0])):
                best = (st, en, txt)
        return best[2].replace("Thunk:#hlo_op=", "").rstrip("#") if best else "вне Thunk-спана"

    agg = collections.defaultdict(lambda: {"n": 0, "dur": 0, "kern": collections.Counter(),
                                           "graphs": 0, "launch_ms": 0.0})
    for cid, p in per.items():
        st, en = launches[cid]
        a = agg[label_at(st)]
        a["n"] += p["n"]
        a["dur"] += p["dur"]
        a["graphs"] += 1
        a["launch_ms"] += (en - st) / 1e6
        a["kern"].update(p["kern"])

    print("спан -> GPU-время, графов, ядер, хост-время запусков, топ-3 ядра")
    for label, a in sorted(agg.items(), key=lambda kv: -kv[1]["dur"]):
        top = ", ".join(f"{names.get(k, k)[:42]} {v/1e6:.0f}ms" for k, v in a["kern"].most_common(3))
        print(f"{a['dur']/1e9:7.2f}s  {a['graphs']:4d} гр  {a['n']:6d} яд  "
              f"host {a['launch_ms']/1e3:6.1f}s  {label[:46]:46s} | {top}")
    print(f"ИТОГО GPU в графах: {sum(a['dur'] for a in agg.values())/1e9:.2f}s; "
          f"графов {sum(a['graphs'] for a in agg.values())}")


if __name__ == "__main__":
    raise SystemExit(main())
