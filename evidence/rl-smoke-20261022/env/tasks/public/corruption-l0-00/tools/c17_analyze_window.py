#!/usr/bin/env python3
"""Атрибуция окна шага в nsys-sqlite (цикл-17, ADR-054 Am.28).

Считает по окну [a,b] секунд трейса:
  - GPU-время ядер, хост-API CUDA, синхронизации, NVTX-спаны Thunk;
  - union каждого класса и общее покрытие (пересечения не считаются дважды);
  - неатрибутированный остаток (CPU вне CUDA API/NVTX).
Запуск: python3 analyze_window.py <trace.sqlite> <a> <b> [<a2> <b2> ...]
"""
import sqlite3
import sys


def intervals(con, sql):
    return [(a / 1e9, b / 1e9) for a, b in con.execute(sql) if b > a]


def union(ivs):
    ivs = sorted(ivs)
    out = []
    for a, b in ivs:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def dur(u):
    return sum(b - a for a, b in u)


def intersect(u, v):
    r = []
    i = j = 0
    while i < len(u) and j < len(v):
        a = max(u[i][0], v[j][0])
        b = min(u[i][1], v[j][1])
        if b > a:
            r.append((a, b))
        if u[i][1] < v[j][1]:
            i += 1
        else:
            j += 1
    return r


def main():
    con = sqlite3.connect(sys.argv[1])
    wins = [(float(sys.argv[i]), float(sys.argv[i + 1])) for i in range(2, len(sys.argv), 2)]
    queries = {
        "gpu_kernel": "SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE end>start",
        "gpu_graph": "SELECT start,end FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE WHERE end>start",
        "cuda_api": "SELECT start,end FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE end>start",
        "sync": "SELECT start,end FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION WHERE end>start",
        "nvtx_thunk": "SELECT start,end FROM NVTX_EVENTS WHERE text LIKE 'Thunk:%' AND end IS NOT NULL",
        "nvtx_compile": "SELECT start,end FROM NVTX_EVENTS WHERE text LIKE 'Xla%' AND end IS NOT NULL",
        "osrt_mutex": (
            "SELECT start,end FROM OSRT_API WHERE nameId="
            "(SELECT id FROM StringIds WHERE value='pthread_mutex_lock')"
        ),
    }
    raw = {k: intervals(con, q) for k, q in queries.items()}
    for a, b in wins:
        win = [(a, b)]
        print(f"== окно [{a:.1f}, {b:.1f}]: wall {b - a:.2f} с")
        u = {}
        for k in raw:
            sel = intersect(raw[k], win)
            u[k] = union(sel)
            print(f"  {k:14s} n={len(sel):6d} total={dur(sel):7.2f} union={dur(u[k]):7.2f}")
        allu = union(sum((u[k] for k in u), []))
        print(f"  UNION(все)={dur(allu):.2f}  НЕАТРИБУТИРОВАНО={b - a - dur(allu):.2f} с")
        th = u["nvtx_thunk"]
        rest = union(u["cuda_api"] + u["sync"] + u["gpu_kernel"] + u["osrt_mutex"])
        inside = dur(intersect(th, rest))
        print(f"  thunk: всего {dur(th):.2f}, из них пересекается с api/sync/gpu/mutex {inside:.2f}, "
              f"чистый хост {dur(th) - inside:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
