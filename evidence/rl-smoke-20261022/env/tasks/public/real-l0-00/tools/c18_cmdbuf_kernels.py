#!/usr/bin/env python3
"""cmdbuf_kernels.py — полный список ядер одного NVTX-спана command_buffer_NNNN.

Вход: node-level nsys-sqlite (--cuda-graph-trace=node), метка спана и окно времени.
Ядра связываются с запуском cuGraphLaunch по correlationId; запуск относится к
спану, если его start попадает в интервал спана.

Запуск: python3 cmdbuf_kernels.py <trace.sqlite> command_buffer_1229 [t_min t_max]
"""
import collections
import sqlite3
import sys


def main():
    con = sqlite3.connect(sys.argv[1])
    label = sys.argv[2]
    lo = int(float(sys.argv[3]) * 1e9) if len(sys.argv) > 3 else 0
    hi = int(float(sys.argv[4]) * 1e9) if len(sys.argv) > 4 else 10**19
    names = {i: v for i, v in con.execute("SELECT id, value FROM StringIds")}

    spans = [(st, en) for st, en in con.execute(
        "SELECT start,end FROM NVTX_EVENTS WHERE text=? AND end IS NOT NULL "
        "AND start>=? AND end<=?", (f"Thunk:#hlo_op={label}#", lo, hi))]
    if not spans:
        # допускаем передачу полного текста
        spans = [(st, en) for st, en in con.execute(
            "SELECT start,end FROM NVTX_EVENTS WHERE text LIKE ? AND end IS NOT NULL "
            "AND start>=? AND end<=?", (f"%{label}%", lo, hi))]
    spans.sort()
    print(f"# спанов {label}: {len(spans)}; окно [{lo/1e9:.1f},{hi/1e9:.1f}]")

    def in_span(t):
        for st, en in spans:
            if st <= t <= en:
                return True
        return False

    launches = {}
    for cid, st, en in con.execute(
        "SELECT r.correlationId,r.start,r.end FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
        "JOIN StringIds s ON s.id=r.nameId WHERE s.value='cuGraphLaunch' "
        "AND r.start>=? AND r.end<=?", (lo, hi)
    ):
        if in_span(st):
            launches[cid] = (st, en)
    agg = collections.Counter()
    cnt = collections.Counter()
    for cid, st, en, nid in con.execute(
        "SELECT correlationId,start,end,demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL "
        "WHERE start>=? AND end<=?", (lo, hi)
    ):
        if cid in launches:
            agg[nid] += en - st
            cnt[nid] += 1
    tot = sum(agg.values())
    print(f"# графов {len(launches)}, ядер {sum(cnt.values())}, GPU {tot/1e9:.2f}s")
    for nid, d in agg.most_common(40):
        print(f"{d/1e6:9.1f} ms  {cnt[nid]:6d}  {100*d/tot:5.1f}%  {names.get(nid, str(nid))[:110]}")


if __name__ == "__main__":
    raise SystemExit(main())
