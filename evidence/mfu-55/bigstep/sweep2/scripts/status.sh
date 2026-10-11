#!/usr/bin/env bash
ST=/home/roman/axiom/evidence/mfu-55/bigstep
echo "=== $(date '+%d.%m %H:%M:%S') (стенд) ==="
CUR=$(pgrep -af "pretrain_run[.]py" | head -1 | grep -oE "run-ref [a-zA-Z0-9_-]+" | cut -d' ' -f2)
GPU=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
echo "прогон: ${CUR:-—} | GPU mem ${GPU:-0}"
echo "--- sweep2/sweep2b (последние клетки):"
tail -n 6 "$ST/sweep2/sweep2.log" 2>/dev/null | grep -E "===|---" | tail -4
tail -n 4 "$ST/sweep2/sweep2b.log" 2>/dev/null
echo "--- сводка (успешные клетки):"
awk -F'\t' 'NR==1 || ($4!="-" && $4!="tok_s")' "$ST/sweep2/sweep2-summary.tsv" 2>/dev/null | awk -F'\t' '{printf "%-18s B=%s chunk=%-5s %8s ток/с  mfu=%s%%\n", $1,$2,$3,$4,$5}'
echo "--- nsys2:"
tail -n 2 "$ST/sweep2/nsys2.log" 2>/dev/null
