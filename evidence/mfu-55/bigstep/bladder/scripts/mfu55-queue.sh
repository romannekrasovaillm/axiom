#!/usr/bin/env bash
# mfu55-queue.sh — автономный конвейер после лестницы батчей:
#   (1) дождаться конца b-ladder.sh
#   (2) флаговые клетки: число итераций KDA-скана (chunk 256), CUDA-графы (command_buffer)
#   (3) nsys-профиль лучшей конфигурации → структура шага (запуски, API-время, доля GPU)
#   (4) сводка в evidence/mfu-55/bigstep/queue/
set -uo pipefail
ROOT=/home/roman/axiom
OUT=$ROOT/evidence/mfu-55/bigstep/queue
LAD=$ROOT/evidence/mfu-55/bigstep/bladder
PY=/home/roman/venv-axiom-0112/bin/python
TMO=${CELL_TIMEOUT:-2400}
mkdir -p "$OUT"; cd "$ROOT" || exit 1
log() { printf '[queue %s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$OUT/queue.log"; }

# (1) ждём лестницу (не более 4 часов)
w=0
while pgrep -f "b-ladder[.]sh" >/dev/null 2>&1; do
  [ "$w" -ge 14400 ] && { log "лестница идёт >4ч — продолжаю без неё"; break; }
  sleep 60; w=$((w+60))
done
log "лестница завершена (или ожидание прекращено); сводка:"
cat "$LAD/ladder-summary.txt" 2>/dev/null | tee -a "$OUT/queue.log"

# лучший батч лестницы (по mfu) → берём для флаговых клеток
best=$(awk -F'\t' 'NR>1 && $4!="-1" && $4>0 {print $2"\t"$4}' "$LAD/ladder-summary.tsv" 2>/dev/null | sort -k2 -nr | head -1 | cut -f1)
best=${best:-2}
log "лучший батч по лестнице: B=$best"

run_cell() {  # name batch flags... 
  local name="$1" B="$2"; shift 2
  local d="$OUT/$name"; mkdir -p "$d"
  log "=== $name: B=$B $*"
  (
    export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.96
    export JAX_COMPILATION_CACHE_DIR=$HOME/.cache/jax_cc_0112
    export TF_GPU_ALLOCATOR=cuda_malloc_async AXIOM_KDA_SOLVE=jax
    exec timeout -k 60 "$TMO" "$PY" tools/pretrain_run.py \
      --model-preset l3-full --grad-checkpointing --grad-checkpointing-policy full \
      --steps 4 --total-steps 4 --seq-len 8192 --batch-size "$B" --micro-batch "$B" \
      --budget-limit-usd 5 --peak-tflops 98.2 --ce-chunk-tokens 4096 --remat-policy none \
      --shard-root /home/roman/axiom-run/raw-v2 \
      --tokens-root /home/roman/gb10-shared/datasets/axiom-pretrain-l3/tokens-v2 \
      --streams W --run-ref "queue-$name" --out "$d" --metrics "$d/metrics.jsonl" "$@"
  ) > "$d/run.log" 2>&1
  local rc=$?
  local m; m=$("$PY" "$HOME/axiom-run/night-metrics.py" "$d/metrics.jsonl" 2>/dev/null || echo "- - - - -")
  local tok mfu; read -r tok mfu _ <<< "$m"
  log "--- $name: rc=$rc tok/s=$tok mfu=$mfu"
  printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$B" "$tok" "$mfu" "$rc" >> "$OUT/queue-summary.tsv"
}

# (2) флаговые клетки: чанк KDA (число итераций скана) и CUDA-графы
XLA_FLAGS="--xla_gpu_autotune_level=2" run_cell chunk256_B$best "$best" --chunk-size 256
XLA_FLAGS="--xla_gpu_autotune_level=2 --xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUDNN" \
  run_cell cmdbuf_B$best "$best"

# (3) nsys-профиль лучшей конфигурации
best_cell=$(awk -F'\t' 'NR>1 && $4!="-1" {print $3"\t"$1}' "$OUT/queue-summary.tsv" 2>/dev/null | sort -nr | head -1 | cut -f2)
[ -z "$best_cell" ] && best_cell="B${best}_real"
log "профилирую лучшую клетку: $best_cell"
PROF=$OUT/profile
mkdir -p "$PROF"
(
  export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
  export JAX_COMPILATION_CACHE_DIR=$HOME/.cache/jax_cc_0112 TF_GPU_ALLOCATOR=cuda_malloc_async AXIOM_KDA_SOLVE=jax
  export XLA_FLAGS="--xla_gpu_autotune_level=2"
  cd "$ROOT"
  timeout -k 60 2400 nsys profile -o "$PROF/nsys-best" --force-overwrite true \
    --trace=cuda,nvtx,osrt --cuda-memory-usage=false --sample=none \
    "$PY" tools/pretrain_run.py \
      --model-preset l3-full --grad-checkpointing --grad-checkpointing-policy full \
      --steps 6 --total-steps 6 --seq-len 8192 --batch-size "$best" --micro-batch "$best" \
      --budget-limit-usd 5 --peak-tflops 98.2 --ce-chunk-tokens 4096 --remat-policy none \
      --shard-root /home/roman/axiom-run/raw-v2 \
      --tokens-root /home/roman/gb10-shared/datasets/axiom-pretrain-l3/tokens-v2 \
      --streams W --run-ref "queue-profile" --out "$PROF" --metrics "$PROF/metrics.jsonl"
) > "$PROF/profile.log" 2>&1
if [ -f "$PROF/nsys-best.nsys-rep" ]; then
  nsys stats --report cuda_api_sum,cuda_gpu_kern_sum --format table \
    "$PROF/nsys-best.nsys-rep" > "$PROF/api-summary.txt" 2>&1 || true
  log "профиль готов: $PROF/api-summary.txt"
fi

log "КОНВЕЙЕР ЗАВЕРШЁН"
{ printf 'cell\tbatch\ttok_s\tmfu\trc\n'; cat "$OUT/queue-summary.tsv" 2>/dev/null; } > "$OUT/queue-summary.txt"
cat "$OUT/queue-summary.txt" | tee -a "$OUT/queue.log"
date -Is > "$OUT/heartbeat"
