#!/bin/bash
# Регресс вехи-11 по одному файлу за прогон (heartbeat в каждом).
cd /home/roman/.arch-ml/worktrees/axiom-43e236d789188868/theseus-20261015183027
FILES="$@"
RC_ALL=0
for f in $FILES; do
  echo "########## $f ##########"
  bash .scratch/run_tests.sh "$f" | tail -n 4
  rc=${PIPESTATUS[0]}
  echo "########## $f -> rc=$rc"
  [ "$rc" -ne 0 ] && RC_ALL=1
done
echo "== regression rc_all=$RC_ALL =="
exit $RC_ALL
