#!/usr/bin/env python3
"""Approximate liveness/peak analysis of an XLA HLO text dump.

Sequential model: within one computation, each instruction's results are live
from their def to their last use (operands of later instructions / ROOT /
operands of nested computation calls at the call site). Nested computations are
analyzed separately (as if the call were one instruction).
"""
import re
import sys
from collections import defaultdict

SHAPE_RE = re.compile(r'^([a-z]+[0-9]*)\[([0-9,]*)\]')

def shape_bytes(shape: str):
    m = SHAPE_RE.match(shape)
    if not m:
        return None
    dims = [int(x) for x in m.group(2).split(',') if x != '']
    n = 1
    for d in dims:
        n *= d
    dtype = m.group(1)
    width = {'f32': 4, 'f64': 8, 'bf16': 2, 'f16': 2, 's32': 4, 's8': 1, 'pred': 1,
             'u32': 4, 's64': 8, 'u8': 1, 'c64': 8, 'f8e4m3fn': 1, 'f8e4m3': 1, 'f8e5m2': 1,
             's16': 2, 'u16': 2}.get(dtype)
    if width is None:
        return None
    return n * width

VAL_RE = re.compile(r'%[A-Za-z0-9_.\-]+(?:#[0-9]+)?')

def parse(path):
    with open(path) as f:
        text = f.read()
    lines = text.split('\n')
    comps = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r'^(?:ENTRY\s+)?(%?[A-Za-z0-9_.\-]+)\s*(\(.*\))?\s*\{$', line.strip())
        if m and not line.startswith(' ') and not line.startswith('\t'):
            name = m.group(1)
            depth = line.count('{') - line.count('}')
            body = [line]
            j = i + 1
            while depth > 0 and j < len(lines):
                body.append(lines[j])
                depth += lines[j].count('{') - lines[j].count('}')
                j += 1
            comps.setdefault(name, (i, body))
            i = j
        else:
            i += 1
    return comps

def analyze_comp(body, big_threshold=4 * 1024**3):
    defs = {}
    defs_at = defaultdict(list)
    last_use = {}
    instr_lines = []
    n_instr = 0
    for line in body:
        s = line.strip()
        if not s or s.startswith('//') or s.startswith('}'):
            continue
        if '=' not in s:
            continue
        n_instr += 1
        lhs, rhs = s.split('=', 1)
        lhs = lhs.strip()
        if lhs.startswith('ROOT '):
            lhs = lhs[5:].strip()
        names = VAL_RE.findall(lhs)
        if not names:
            continue
        rhs_s = rhs.strip()
        if rhs_s.startswith('('):
            shapes = re.findall(r'[a-z]+[0-9]*\[[0-9,]*\]', rhs_s.split(')', 1)[0] + ')')
        else:
            m = SHAPE_RE.match(rhs_s)
            shapes = [m.group(0)] if m else []
        b = 0
        for sh in shapes:
            nb = shape_bytes(sh)
            if nb is not None:
                b += nb
        idx = n_instr
        for nm in names:
            defs[nm] = (idx, b, tuple(shapes))
            defs_at[idx].append(nm)
        for nm in VAL_RE.findall(rhs):
            last_use[nm.split('#')[0]] = idx
        instr_lines.append((idx, s))
    live = {}
    peak = 0
    peak_idx = -1
    peak_big = {}
    for idx, s in instr_lines:
        for nm in defs_at.get(idx, ()):
            live[nm] = defs[nm][1]
        tot = 0
        for nm in list(live):
            if last_use.get(nm, 0) < idx:
                del live[nm]
                continue
            tot += live[nm]
        if tot > peak:
            peak = tot
            peak_idx = idx
            big = {}
            for nm, b in live.items():
                if b >= big_threshold:
                    key = defs[nm][2]
                    big[key] = big.get(key, 0) + 1
            peak_big = big
    return peak, peak_idx, peak_big

if __name__ == '__main__':
    path = sys.argv[1]
    pattern = sys.argv[2] if len(sys.argv) > 2 else None
    comps = parse(path)
    print(f"# computations: {len(comps)}")
    sel = [(n, v) for n, v in comps.items() if pattern is None or pattern in n]
    sel_sorted = sorted(sel, key=lambda kv: -len(kv[1][1]))
    for name, (start, body) in sel_sorted[:8]:
        peak, pidx, big = analyze_comp(body)
        print(f"== {name} @line {start+1}: lines={len(body)} peak={peak/2**30:.2f} GiB (instr {pidx})")
        for sh, cnt in sorted(big.items(), key=lambda kv: -kv[1])[:12]:
            nb = shape_bytes(sh[0]) if sh else 0
            print(f"   live big: {cnt} x {sh} = {cnt*nb/2**30:.1f} GiB")
