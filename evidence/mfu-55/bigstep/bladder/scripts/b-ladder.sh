#!/usr/bin/env bash
# b-ladder.sh — решающая проверка рычага «укрупнение форм»: реальный микро-батч 2/4/8
# Конфигурация: remat none (снимает OOM 88.91 ГиБ, проверено смоуком S1), ce4096, autotune 2.
# Расчётный след (скилл llm-vram-calc): B=2 → 28 ГБ, B=4 → 47 ГБ, B=8 → 86 ГБ из 115.
set -uo pipefail
ROOT=/home/roman/axiom
OUT=$ROOT/evidence/mfu-55/bigstep/bladder
PY=/home/roman/venv-axiom-0112/bin/python
TMO=${CELL_TIMEOUT:-2400}
mkdir -p "$OUT"; cd "$ROOT" || exit 1
log() { printf '[bladder %s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$OUT/ladder.log"; }

CELLS=$(cat <<'CEIL'
B2_real|2|4|MDV>=1.55%
B4_real|4|4|MDV>=1.80%
B8_real|8|4|MDV>=2.00%
CEIL
)

for row in $CELLS; do
  [ -z "$row" ] && continue
  IFS='|' read -r name B steps hyp <<< "$row"
  d="$OUT/$name"; mkdir -p "$d"
  log "=== $name: batch=$B, $steps шагов, remat none"
  (
    export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.96
    export JAX_COMPILATION_CACHE_DIR=$HOME/.cache/jax_cc_0112
    export TF_GPU_ALLOCATOR=cuda_malloc_async AXIOM_KDA_SOLVE=jax
    export XLA_FLAGS="--xla_gpu_autotune_level=2"
    exec timeout -k 60 "$TMO" "$PY" tools/pretrain_run.py \
      --model-preset l3-full --grad-checkpointing --grad-checkpointing-policy full \
      --steps "$steps" --total-steps "$steps" --seq-len 8192 \
      --batch-size "$B" --micro-batch "$B" \
      --budget-limit-usd 5 --peak-tflops 98.2 \
      --ce-chunk-tokens 4096 --remat-policy none \
      --shard-root /home/roman/axiom-run/raw-v2 \
      --tokens-root /home/roman/gb10-shared/datasets/axiom-pretrain-l3/tokens-v2 \
      --streams W --run-ref "bladder-$name" --out "$d" --metrics "$d/metrics.jsonl"
  ) > "$d/run.log" 2>&1
  rc=$?
  oom=0; grep -q 'RESOURCE_EXHAUSTED' "$d/run.log" 2>/dev/null && oom=1
  m=$("$PY" "$HOME/axiom-run/night-metrics.py" "$d/metrics.jsonl" 2>/dev/null || echo "- - - - -")
  read -r tok mfu l0 l1 nst <<< "$m"
  log "--- $name: rc=$rc oom=$oom tok/s=$tok mfu=$mfu loss $l0→$l1 ($nst шагов) | $hyp"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$B" "$tok" "$mfu" "$rc" "$oom" "$hyp" "$nst" >> "$OUT/ladder-summary.tsv"
  date -Is > "$OUT/heartbeat"
done
log "лестница завершена"
{ printf 'cell\tbatch\ttok_s\tmfu\trc\toom\thypothesis\tsteps\n'; cat "$OUT/ladder-summary.tsv"; } > "$OUT/ladder-summary.txt"
cat "$OUT/ladder-summary.txt" | tee -a "$OUT/ladder.log"
