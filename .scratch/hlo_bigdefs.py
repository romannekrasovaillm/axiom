#!/usr/bin/env python3
"""Attribute every >8GiB buffer-def instruction of an HLO dump to its enclosing computation."""
import re, sys, collections

sys.path.insert(0, '/home/roman/.arch-ml/worktrees/axiom-43e236d789188868/theseus-20261014113809/.scratch')

path = sys.argv[1]
with open(path) as f:
    lines = f.read().split('\n')

# find computation ranges: a line matching '^%name... {' at col 0, ends at matching brace
comp_ranges = []  # (name, start_line_idx, end_line_idx)
i = 0
while i < len(lines):
    line = lines[i]
    m = re.match(r'^(?:ENTRY\s+)?(%?[A-Za-z0-9_.\-]+)\s*\(', line)
    if m and line.rstrip().endswith('{') and not line.startswith(' '):
        depth = line.count('{') - line.count('}')
        j = i
        while depth > 0 and j < len(lines):
            j += 1
            depth += lines[j].count('{') - lines[j].count('}')
        comp_ranges.append((m.group(1), i, j))
        i = j
    else:
        i += 1

def comp_of(lineno):
    best = None
    for name, s, e in comp_ranges:
        if s <= lineno <= e:
            if best is None or s > best[1]:
                best = (name, s)
    return best[0] if best else '<none>'

SHAPE_LA = re.compile(r'= (?:\(?)(?:f32|bf16)\[([0-9,]*)\]')
BIG = 8 * 2**30  # 8 GiB fp32 count limit; use fp32 4 bytes assumption for reporting

def nbytes(shape, dtype):
    n = 1
    for d in shape.split(','):
        if d == '':
            continue
        n *= int(d)
    return n * (4 if dtype == 'f32' else 2)

counts = collections.Counter()
meta_counts = collections.Counter()
big_defs = []
for idx, line in enumerate(lines):
    m = re.search(r'= \(?((?:f32|bf16)\[[0-9,]*\])', line)
    if not m:
        continue
    full = re.findall(r'(f32|bf16)\[([0-9,]*)\]', line[:400])
    # only take the leading shape(s) of the result
    mm = re.match(r'\s*(?:ROOT )?%?[A-Za-z0-9_.\-]*\s*=\s*(.+)$', line)
    if not mm:
        continue
    rhs = mm.group(1)
    if rhs.startswith('('):
        shapes = re.findall(r'(f32|bf16)\[([0-9,]*)\]', rhs.split(')', 1)[0] + ')')
        if not shapes:
            continue
        total = sum(nbytes(s, d) for d, s in shapes)
    else:
        m2 = re.match(r'(f32|bf16)\[([0-9,]*)\]', rhs)
        if not m2:
            continue
        total = nbytes(m2.group(2), m2.group(1))
    if total >= BIG:
        c = comp_of(idx)
        counts[c] += 1
        big_defs.append((idx + 1, c, total))
        meta = re.search(r'op_name="([^"]*)"', line)
        if meta:
            # strip the leading jit(loss_fn) and stack_frame_id noise
            op = meta.group(1)
            op = op.replace('jit(loss_fn)', '')
            meta_counts[op.split('/')[1] if '/' in op else op] += 1

print('== big-buffer defs per computation:')
for name, cnt in counts.most_common(15):
    print(f'  {cnt:5d}  {name}')
print('== big-buffer defs per top op segment:')
for name, cnt in meta_counts.most_common(25):
    print(f'  {cnt:5d}  {name}')
print('== biggest computations by big-defs:')
rng = {name: (s, e) for name, s, e in comp_ranges}
for name, cnt in counts.most_common(6):
    s, e = rng[name]
    print(f'  {name}: lines {s+1}..{e+1}  ({e-s} lines)')
