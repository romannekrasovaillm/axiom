#!/usr/bin/env python3
"""Peak-liveness decomposition of one HLO computation (who is alive at the peak)."""
import re
import sys

NAME = r'%[\w.\-]+'
SHAPE = r'([a-z]+[0-9]*)\[([0-9,]*)\]'
WIDTH = {'f32': 4, 'bf16': 2, 'f16': 2, 's32': 4, 'pred': 1, 's8': 1, 'u32': 4, 's64': 8, 'u8': 1}


def nbytes(dt, dims):
    n = 1
    for d in dims.split(','):
        if d:
            n *= int(d)
    w = WIDTH.get(dt)
    return None if w is None else n * w


def parse_comps(lines):
    comps = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r'^(?:ENTRY\s+)?(%[\w.\-]+|[\w.\-]+)\s*(\(.*\))?\s*\{$', line.strip())
        if m and not line.startswith((' ', '\t')):
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


def analyze(body, top=12):
    defs = {}
    last_use = {}
    order = []  # (idx, names, bytes, line)
    n = 0
    for line in body:
        s = line.strip()
        if not s or s.startswith('//') or '=' not in s:
            continue
        n += 1
        lhs, rhs = s.split('=', 1)
        lhs, rhs = lhs.strip(), rhs.strip()
        if lhs.startswith('ROOT '):
            lhs = lhs[5:].strip()
        names = re.findall(NAME, lhs)
        if not names:
            continue
        if rhs.startswith('('):
            shapes = re.findall(SHAPE, rhs.split(')', 1)[0] + ')')
        else:
            m = re.match(SHAPE, rhs)
            shapes = [m.groups()] if m else []
        b = 0
        for dt, dims in shapes:
            nb = nbytes(dt, dims)
            if nb:
                b += nb
        for nm in names:
            defs[nm] = (n, b, tuple(f"{dt}[{dims}]" for dt, dims in shapes))
        for nm in re.findall(NAME, rhs):
            last_use[nm.split('#')[0]] = n
        order.append((n, names, b, s))
    live = {}
    peak = 0
    peak_snapshot = []
    peak_idx = -1
    for idx, names, b, s in order:
        for nm in names:
            if nm in defs and defs[nm][0] == idx:
                live[nm] = defs[nm][1]
        for nm in list(live):
            if last_use.get(nm, 0) < idx:
                del live[nm]
        tot = sum(live.values())
        if tot > peak:
            peak = tot
            peak_idx = idx
            snap = sorted(((v, k) for k, v in live.items()), reverse=True)[:top]
            peak_snapshot = snap
    return peak, peak_idx, peak_snapshot, defs


def main():
    path = sys.argv[1]
    pattern = sys.argv[2] if len(sys.argv) > 2 else None
    lines = open(path).read().split('\n')
    comps = parse_comps(lines)
    best = []
    for name, (start, body) in comps.items():
        if pattern and pattern not in name:
            continue
        if len(body) < 30:
            continue
        peak, pidx, snap, defs = analyze(body)
        best.append((peak, name, start, pidx, snap))
    best.sort(key=lambda r: -r[0])
    for peak, name, start, pidx, snap in best[:5]:
        print(f"== {name} @line {start+1}: peak={peak/2**30:.2f} GiB at instr {pidx}")
        for v, k in snap:
            d, _, shapes = defs.get(k, (-1, 0, ()))
            sh = ",".join(shapes)[:80]
            print(f"     {v/2**30:8.2f} GiB  {k} (def instr {d}) {sh}")


if __name__ == '__main__':
    main()
