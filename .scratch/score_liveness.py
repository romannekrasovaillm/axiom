#!/usr/bin/env python3
"""Score-tensor liveness census of an XLA HLO text (executable form).

Finds intermediate values whose shape is `attention-score-like` (the last two
dimensions equal => (B,H,T,T) family) and reports, per computation,
* the maximum number of such values simultaneously live (sequential liveness),
* their total bytes at that point,
* the longest-lived instances.
"""
import re
import sys
import collections

NAME = r'%?[A-Za-z0-9_.\-]+(?:#[0-9]+)?'
SHAPE = r'([a-z]+[0-9]*)\[([0-9,]*)\]'
VAL_RE = re.compile(NAME)


def shape_bytes(shape, dtype):
    n = 1
    for d in shape.split(','):
        if d:
            n *= int(d)
    w = {'f32': 4, 'f64': 8, 'bf16': 2, 'f16': 2, 'pred': 1, 's32': 4, 's8': 1,
         'u32': 4, 's64': 8, 'u8': 1}.get(dtype, None)
    return None if w is None else n * w


def is_score_like(dtype, dims):
    if dtype not in ('f32', 'bf16'):
        return False
    if len(dims) < 2:
        return False
    return dims[-1] == dims[-2] and dims[-1] >= 32


def parse_comps(text):
    lines = text.split('\n')
    comps = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r'^(?:ENTRY\s+)?(%[A-Za-z0-9_.\-]+|(?:ENTRY )?[A-Za-z0-9_.\-]+)\s*(\(.*\))?\s*\{$', line.strip())
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


def analyze(body):
    defs = {}
    last_use = {}
    instr = []  # (idx, line)
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
        names = VAL_RE.findall(lhs)
        if not names:
            continue
        if rhs.startswith('('):
            shapes = re.findall(SHAPE, rhs.split(')', 1)[0] + ')')
        else:
            m = re.match(SHAPE, rhs)
            shapes = [m.groups()] if m else []
        b = 0
        big = []
        for dt, dims in shapes:
            ds = [int(x) for x in dims.split(',') if x]
            nb = shape_bytes(dims, dt)
            if nb:
                b += nb
            if is_score_like(dt, ds):
                big.append(nb)
        for nm in names:
            defs[nm] = (n, b, big)
        for nm in VAL_RE.findall(rhs):
            last_use[nm.split('#')[0]] = n
        instr.append((n, s, names, b, big))
    # liveness over score-like components only
    live = {}
    peak_cnt = 0
    peak_bytes = 0
    peak_idx = -1
    longest = []
    for idx, s, names, b, big in instr:
        for nm in names:
            if nm in defs and defs[nm][0] == idx:
                lv = defs[nm][2]
                if lv:
                    live[nm] = (sum(lv), idx)
                    longest.append([nm, idx, None, sum(lv), s[:120]])
        tot = 0
        cnt = 0
        for nm in list(live):
            lu = last_use.get(nm, 0)
            if lu < idx:
                del live[nm]
                continue
            tot += live[nm][0]
            cnt += 1
        if tot > peak_bytes:
            peak_bytes = tot
            peak_cnt = cnt
            peak_idx = idx
    # fill last-use for live ranges
    for rec in longest:
        rec[2] = last_use.get(rec[0], rec[1])
    longest.sort(key=lambda r: -(r[3]))
    return peak_cnt, peak_bytes, peak_idx, longest, n


def main():
    path = sys.argv[1]
    text = open(path).read()
    comps = parse_comps(text)
    rows = []
    for name, (start, body) in comps.items():
        if len(body) < 20:
            continue
        peak_cnt, peak_bytes, peak_idx, longest, n = analyze(body)
        if peak_cnt:
            rows.append((peak_bytes, peak_cnt, name, start, peak_idx, longest, n))
    rows.sort(key=lambda r: -r[0])
    for peak_bytes, peak_cnt, name, start, peak_idx, longest, n in rows[:10]:
        print(f"== {name} @line {start+1}: instrs={n} peak score-live: {peak_cnt} x {peak_bytes/2**20:.2f} MiB at instr {peak_idx}")
        for nm, d, lu, b, s in longest[:5]:
            print(f"     live[{d}..{lu}] ({lu-d} instrs, {b/2**20:.2f} MiB) {nm}: {s[:100]}")


if __name__ == '__main__':
    main()
