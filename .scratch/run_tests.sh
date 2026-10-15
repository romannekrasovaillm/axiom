#!/bin/bash
# Heartbeat-обёртка прогона тестов: pytest в фоне, пульс каждые 30 с.
# Приёмное окружение (форма вехи-10): venv-axiom-0112cpu-py312 (jax 0.11.2),
# NET_JAX_BACKEND=cpu, AXIOM_KDA_CC_IMPL=jax (persistent-кернел кампании
# отказывает CPU-узлу).  Override: TEST_PY=... EXTRA_ENV="K=V ..."
# Usage: run_tests.sh <pytest args...>
cd /home/roman/.arch-ml/worktrees/axiom-43e236d789188868/theseus-20261015183027
TEST_PY=${TEST_PY:-/home/roman/venv-axiom-0112cpu-py312/bin/python}
export NET_JAX_BACKEND=${NET_JAX_BACKEND:-cpu}
export AXIOM_KDA_CC_IMPL=${AXIOM_KDA_CC_IMPL:-jax}
NAME=$(basename "${1%.py}")
LOG=".scratch/testlog-${NAME}-$(date +%H%M%S).log"
mkdir -p .scratch
echo "[start] $(date +%H:%M:%S) py=$TEST_PY pytest $* -> $LOG"
"$TEST_PY" -m pytest -q -s -p no:cacheprovider "$@" > "$LOG" 2>&1 &
PID=$!
while kill -0 "$PID" 2>/dev/null; do
  sleep 30
  echo "[heartbeat $(date +%H:%M:%S)] $(tail -c 600 "$LOG" | tr '\n' '|' | tail -c 220)"
done
wait "$PID"
RC=$?
echo "[done $RC] $(date +%H:%M:%S)"
echo "===== log: $LOG ====="
cat "$LOG"
exit "$RC"
