#!/usr/bin/env bash
# nsys2.sh — финал цепочки: nsys-профиль ЛУЧШЕЙ клетки серий sweep2/sweep2b (замена шага (3) очереди).
# Ждёт ОБЕ серии (sweep2.sh и sweep2b.sh), берёт глобальный максимум mfu из общего TSV (awk-max,
# без sort — локаль ru_RU ломает sort -k/-n), восстанавливает конфигурацию клетки (B, chunk, ce,
# CB-флаги из cbmax.txt) и снимает профиль структуры шага для вердикта G1.
set -uo pipefail
ROOT=/home/roman/axiom
OUT=$ROOT/evidence/mfu-55/bigstep/sweep2
PY=/home/roman/venv-axiom-0112/bin/python
mkdir -p "$OUT"; cd "$ROOT" || exit 1
log() { printf '[nsys2 %s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$OUT/nsys2.log"; }

# (1) ждать обе серии (не более 8 ч суммарно)
w=0
while pgrep -f "sweep2[b]*[.]sh" >/dev/null 2>&1; do
  [ "$w" -ge 28800 ] && { log "серии идут >8ч — профилирую лучшее из готового"; break; }
  sleep 60; w=$((w+60))
done
log "серии завершены; сводка:"
cat "$OUT/sweep2-summary.txt" 2>/dev/null | tee -a "$OUT/nsys2.log" || tail -20 "$OUT/sweep2-summary.tsv" | tee -a "$OUT/nsys2.log"

# (2) лучшая клетка по mfu_pct — awk-max (sort в ru_RU локали ненадёжен)
BESTROW=$(awk -F'\t' 'NR>1 && $5!="-" && $5+0>m {m=$5+0; row=$0} END {print row}' "$OUT/sweep2-summary.tsv" 2>/dev/null)
if [ -z "$BESTROW" ]; then log "нет успешных клеток — профилировать нечего"; exit 1; fi
NAME=$(printf '%s\n' "$BESTROW" | cut -f1); B=$(printf '%s\n' "$BESTROW" | cut -f2); CS=$(printf '%s\n' "$BESTROW" | cut -f3)
log "лучшая клетка: $NAME (B=$B chunk=$CS) — снимаю профиль"

# (3) конфигурация клетки
CE=4096; [ "$NAME" = "b4_ce2048" ] && CE=2048
[ "$NAME" = "c128_B4" ] && CE=2048
CBMAX=$(cat "$OUT/cbmax.txt" 2>/dev/null || echo "FUSION,CUBLAS,CUDNN")
cbflag=""
case "$NAME" in
  cbunroll_*) cbflag="--xla_gpu_enable_command_buffer=$CBMAX --xla_gpu_command_buffer_unroll_loops=true" ;;
  cb_*)       cbflag="--xla_gpu_enable_command_buffer=$CBMAX" ;;
esac
log "конфигурация профиля: B=$B chunk=$CS ce=$CE cb='$cbflag'"

PROF=$OUT/profile-best; mkdir -p "$PROF"
(
  export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
  export JAX_COMPILATION_CACHE_DIR=$HOME/.cache/jax_cc_0112
  export TF_GPU_ALLOCATOR=cuda_malloc_async AXIOM_KDA_SOLVE=jax
  export XLA_FLAGS="--xla_gpu_autotune_level=2 $cbflag"
  exec timeout -k 60 2400 nsys profile -o "$PROF/nsys-best" --force-overwrite true \
    --trace=cuda,nvtx,osrt --cuda-memory-usage=false --sample=none \
    "$PY" tools/pretrain_run.py \
    --model-preset l3-full --grad-checkpointing --grad-checkpointing-policy full \
    --steps 6 --total-steps 6 --seq-len 8192 --batch-size "$B" --micro-batch "$B" \
    --budget-limit-usd 5 --peak-tflops 98.2 --ce-chunk-tokens "$CE" --remat-policy none \
    --chunk-size "$CS" \
    --shard-root /home/roman/axiom-run/raw-v2 \
    --tokens-root /home/roman/gb10-shared/datasets/axiom-pretrain-l3/tokens-v2 \
    --streams W --run-ref "sweep2-profile-$NAME" --out "$PROF" --metrics "$PROF/metrics.jsonl"
) > "$PROF/profile.log" 2>&1
rc=$?
log "профиль: rc=$rc"
if [ -f "$PROF/nsys-best.nsys-rep" ]; then
  nsys stats --report cuda_api_sum,cuda_gpu_kern_sum --format table \
    "$PROF/nsys-best.nsys-rep" > "$PROF/api-summary.txt" 2>&1 || true
  log "готово: $PROF/api-summary.txt (запуски ядер и API-время для вердикта G1)"
fi
date -Is > "$OUT/nsys2-heartbeat"
log "NSYS2 ЗАВЕРШЁН"
