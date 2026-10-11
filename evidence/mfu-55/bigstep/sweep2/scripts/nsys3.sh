#!/usr/bin/env bash
# nsys3.sh — профиль НОВОГО лидера серий (если он появился; c64_B2 уже профилирован nsys2).
# Ждёт sweep2c, берёт глобальный максимум mfu (awk-max), пропускает уже профилированные клетки.
set -uo pipefail
ROOT=/home/roman/axiom
OUT=$ROOT/evidence/mfu-55/bigstep/sweep2
PY=/home/roman/venv-axiom-0112/bin/python
cd "$ROOT" || exit 1
log() { printf '[nsys3 %s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$OUT/nsys3.log"; }

w=0
while pgrep -f "sweep2c[.]sh" >/dev/null 2>&1; do
  [ "$w" -ge 14400 ] && { log "sweep2c идёт >4ч — выхожу"; exit 1; }
  sleep 60; w=$((w+60))
done
log "sweep2c завершена"

BESTROW=$(awk -F'\t' 'NR>1 && $5!="-" && $5+0>m {m=$5+0; row=$0} END {print row}' "$OUT/sweep2-summary.tsv" 2>/dev/null)
NAME=$(printf '%s\n' "$BESTROW" | cut -f1)
if [ -z "$NAME" ]; then log "нет успешных клеток"; exit 1; fi
case "$NAME" in
  c64_B2) log "лидер не сменился ($NAME) — профиль уже снят nsys2, выхожу"; exit 0 ;;
esac
B=$(printf '%s\n' "$BESTROW" | cut -f2); CS=$(printf '%s\n' "$BESTROW" | cut -f3)
log "новый лидер: $NAME (B=$B chunk=$CS) — снимаю профиль"

CE=4096
case "$NAME" in *_ce2048*|*B4*) CE=2048 ;; esac
CBMAX=$(cat "$OUT/cbmax.txt" 2>/dev/null || echo "FUSION,CUBLAS,CUDNN")
cbflag=""; cbdir=""
case "$NAME" in
  cbru_*) cbflag="--xla_gpu_enable_command_buffer=$CBMAX --xla_gpu_command_buffer_unroll_loops=true"; cbdir="$HOME/.cache/jax_cc_0112_cbun" ;;
  cbr_*)  cbflag="--xla_gpu_enable_command_buffer=$CBMAX"; cbdir="$HOME/.cache/jax_cc_0112_cb" ;;
  cbunroll_*) cbflag="--xla_gpu_enable_command_buffer=$CBMAX --xla_gpu_command_buffer_unroll_loops=true" ;;
  cb_*)       cbflag="--xla_gpu_enable_command_buffer=$CBMAX" ;;
esac
log "конфигурация: B=$B chunk=$CS ce=$CE cb='$cbflag' cache=${cbdir:-default}"

PROF=$OUT/profile-best-$NAME; mkdir -p "$PROF"
(
  export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
  export JAX_COMPILATION_CACHE_DIR="${cbdir:-$HOME/.cache/jax_cc_0112}"
  export TF_GPU_ALLOCATOR=cuda_malloc_async AXIOM_KDA_SOLVE=jax
  export XLA_FLAGS="--xla_gpu_autotune_level=2 $cbflag"
  exec timeout -k 60 2400 nsys profile -o "$PROF/nsys-best" --force-overwrite true \
    --trace=cuda,nvtx,osrt --cuda-memory-usage=false --sample=none \
    "$PY" tools/pretrain_run.py \
    --model-preset l3-full --grad-checkpointing --grad-checkpointing-policy full \
    --steps 8 --total-steps 8 --seq-len 8192 --batch-size "$B" --micro-batch "$B" \
    --budget-limit-usd 5 --peak-tflops 98.2 --ce-chunk-tokens "$CE" --remat-policy none \
    --chunk-size "$CS" \
    --shard-root /home/roman/axiom-run/raw-v2 \
    --tokens-root /home/roman/gb10-shared/datasets/axiom-pretrain-l3/tokens-v2 \
    --streams W --run-ref "sweep2c-profile-$NAME" --out "$PROF" --metrics "$PROF/metrics.jsonl"
) > "$PROF/profile.log" 2>&1
rc=$?
log "профиль: rc=$rc (8 шагов: стационар = шаги 4..8, warmup в начале)"
if [ -f "$PROF/nsys-best.nsys-rep" ]; then
  nsys stats --report cuda_api_sum,cuda_gpu_kern_sum --format table \
    "$PROF/nsys-best.nsys-rep" > "$PROF/api-summary.txt" 2>&1 || true
  log "готово: $PROF/api-summary.txt"
fi
date -Is > "$OUT/nsys3-heartbeat"
log "NSYS3 ЗАВЕРШЁН"
