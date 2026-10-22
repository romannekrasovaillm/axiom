#!/usr/bin/env python3
"""map_kernels.py — карта kernel → HLO для цикла-18 (ADR-054 Am.29).

Вход: каталог XLA-дампа (--xla_dump_hlo_as_text) и список имён ядер nsys
(например, из cmdbuf_kernels.py). Для каждого имени ищет в
*.jit_loss_fn.sm_12.1a_gpu_after_optimizations.txt строку определения фьюжена
`%<name> = ...` и печатает: выходную форму, op_name (слой/фаза), deduplicated_name,
вид backend_config (triton nested gemm / native emitter) и ключевые параметры
(num_warps, num_stages, output_tiles).

Имена ядер nsys: `gemm_fusion_dot_1231` → HLO `%gemm_fusion_dot.1231`;
`input_add_reduce_fusion` → `%input_add_reduce_fusion` (без номера).

Запуск: python3 map_kernels.py <dump_dir> [kernel ...]
Без списка — читает kernel из stdin (по одному в строке).
"""
import glob
import json
import re
import sys


def find_hlo_file(dump):
    cands = glob.glob(f"{dump}/*jit_loss_fn*after_optimizations.txt")
    if not cands:
        # в дампе могут быть только .txt без суффикса модуля
        cands = glob.glob(f"{dump}/*after_optimizations.txt")
    if not cands:
        raise SystemExit("не найден after_optimizations.txt")
    return max(cands, key=lambda p: __import__("os").path.getsize(p))


def hlo_name(kernel):
    m = re.match(r"^(.*?)_(\d+)$", kernel)
    if m:
        return f"{m.group(1)}.{m.group(2)}"
    return kernel


def main():
    dump = sys.argv[1]
    kernels = sys.argv[2:] or [l.strip() for l in sys.stdin if l.strip()]
    path = find_hlo_file(dump)
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    print(f"# HLO: {path} ({len(text)/1e6:.0f} MB), ядер в списке: {len(kernels)}")
    hdr = f"{'kernel':34s} {'shape':26s} {'cfg':22s} {'warps/stages/tiles':24s} op_name"
    print(hdr)
    for k in kernels:
        h = hlo_name(k)
        pat = re.compile(r"^\s*%" + re.escape(h) + r"\s*=\s*(.*)$", re.MULTILINE)
        m = pat.search(text)
        if not m:
            print(f"{k:34s} {'—':26s} {'—':22s} {'—':24s} НЕ НАЙДЕН")
            continue
        line = m.group(1)
        shape_m = re.match(r"[^ ]+", line)
        shape = shape_m.group(0) if shape_m else "?"
        op_m = re.search(r'op_name="([^"]*)"', line)
        op = op_m.group(1) if op_m else "—"
        op = op.replace("jit(loss_fn)/", "")
        ded_m = re.search(r'deduplicated_name="([^"]*)"', line)
        ded = ded_m.group(1) if ded_m else ""
        kind_m = re.search(r'"kind":"([^"]+)"', line)
        kind = kind_m.group(1) if kind_m else "—"
        warps = re.search(r'"num_warps":"?(\d+)', line)
        stages = re.search(r'"num_stages":"?(\d+)', line)
        tiles = re.search(r'"output_tiles":\[([^\]]*)\]', line)
        t = f"{warps.group(1) if warps else '?'}/{stages.group(1) if stages else '?'}/" + \
            (tiles.group(1).replace('"', '') if tiles else "?")
        print(f"{k:34s} {shape[:26]:26s} {kind[:22]:22s} {t[:24]:24s} {op[:90]}"
              + (f"  [{ded}]" if ded and ded != h else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
