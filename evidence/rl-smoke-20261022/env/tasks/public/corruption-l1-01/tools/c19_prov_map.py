#!/usr/bin/env python3
"""c19_prov_map.py — провенанс-карта хвоста (цикл-19, ADR-054 Am.31).

Вход:
  trace.sqlite — nsys node-level (--cuda-graph-trace=node) прогона с
                 AXIOM_TRACE_MARKERS=1;
  dump_dir     — XLA-дамп того же графа (после оптимизаций, jit_loss_fn).

Что делает:
  1) находит NVTX-спаны Thunk и запуски cuGraphLaunch, привязывает ядра к
     спанам (как c17_graph_to_cmdbuf);
  2) для каждого имени ядра находит HLO-строку `%name = ...` и вытаскивает
     op_name (там ожидаются сегменты probe_kda / probe_mla / probe_moe /
     probe_attnres / probe_ce / probe_mlp);
  3) агрегирует GPU-время и число ядер по probe-компоненту и по спанам.

Запуск: python3 c19_prov_map.py <trace.sqlite> <dump_dir> [t_min t_max]
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


def load_op_names(dump: str):
    cands = glob.glob(f"{dump}/*jit_loss_fn*after_optimizations.txt")
    if not cands:
        cands = glob.glob(f"{dump}/*after_optimizations.txt")
    if not cands:
        raise SystemExit(f"нет after_optimizations.txt в {dump}")
    path = max(cands, key=os.path.getsize)
    pat = re.compile(r"^\s*%([A-Za-z0-9_.]+)\s*=\s*.*?op_name=\"([^\"]*)\"", re.MULTILINE)
    text = open(path, encoding="utf-8", errors="replace").read()
    mapping = {m.group(1): m.group(2) for m in pat.finditer(text)}
    print(f"# HLO {path} ({len(mapping)} имён с op_name)")
    return mapping


def probe_of(op: str) -> str:
    for name in ("kda", "mla_dense", "mla", "moe", "mlp", "attnres", "ce"):
        if f"probe_{name}" in op:
            return name
    return "—"


def main():
    trace, dump = sys.argv[1], sys.argv[2]
    lo = int(float(sys.argv[3]) * 1e9) if len(sys.argv) > 3 else 0
    hi = int(float(sys.argv[4]) * 1e9) if len(sys.argv) > 4 else 2**62
    names = {i: v for i, v in sqlite3.connect(trace).execute("SELECT id,value FROM StringIds")}
    opmap = load_op_names(dump)

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

    per_probe = collections.Counter()
    n_probe = collections.Counter()
    per_span = collections.Counter()
    per_span_probe = collections.defaultdict(collections.Counter)
    unknown = collections.Counter()
    total = 0
    for cid, st, en, nid in con.execute(
        "SELECT correlationId,start,end,demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL "
        "WHERE start>=? AND end<=?", (lo, hi)
    ):
        d = en - st
        total += d
        kernel = names.get(nid, str(nid))
        op = opmap.get(hlo_name(kernel), "")
        p = probe_of(op)
        per_probe[p] += d
        n_probe[p] += 1
        if p == "—":
            unknown[op or kernel] += d
        if cid in launches:
            label = span_at(launches[cid][0])
            per_span[label] += d
            per_span_probe[label][p] += d

    print(f"\n# GPU всего {total/1e9:.2f}s, ядер {sum(n_probe.values())}")
    print("== по probe-компоненту")
    for p, d in per_probe.most_common():
        print(f"{d/1e9:7.2f}s  {n_probe[p]:7d} яд  {100*d/total:5.1f}%  probe_{p}")
    print("\n== топ-спаны command_buffer_*")
    for label, d in per_span.most_common(12):
        mix = ", ".join(f"{k} {100*v/d:.0f}%" for k, v in per_span_probe[label].most_common(4))
        print(f"{d/1e9:7.2f}s  {label[:40]:40s} | {mix}")
    print("\n== неотнесённое (топ-10)")
    for name, d in unknown.most_common(10):
        print(f"{d/1e9:7.2f}s  {name[:110]}")


if __name__ == "__main__":
    raise SystemExit(main())
