#!/usr/bin/env bash
# sweep2c.sh — серия-2c: чистый тест command buffers (фикс контаминации кэшем) + ретрай c128_B2.
# Причина: JAX persistent compilation cache НЕ включает XLA_FLAGS в ключ — cb-клетки серии-2b
# переиспользовали бинарник c064_B1 (412/410 против 414.7 = шум того же экзешника).
# Фикс: cb-клетки идут со СВЕЖИМИ каталогами кэша (полная перекомпиляция с флагами).
# Набор CB по discovery: FUSION,CUBLAS,CUDNN,WHILE,CONDITIONAL (COMPUTE/COLLECTIVE/PARAMETER/DOT отвергнуты).
# Клетки (деньги вперёд): cbr_c64_B2 (CB на лучшей), cbru_c64_B2 (+unroll), cbr_c064_B1 (изоляция CB), c128_B2_retry.
set -uo pipefail
ROOT=/home/roman/axiom
OUT=$ROOT/evidence/mfu-55/bigstep/sweep2
PY=/home/roman/venv-axiom-0112/bin/python
TMO=${CELL_TIMEOUT:-2400}
CBMAX="FUSION,CUBLAS,CUDNN,WHILE,CONDITIONAL"
cd "$ROOT" || exit 1
log() { printf '[sweep2c %s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$OUT/sweep2c.log"; }

fails=0
run_cell() {  # name batch chunk ce mode cachedir
  local name="$1" B="$2" CS="$3" CE="$4" MODE="$5" CD="$6"
  local d="$OUT/$name"; mkdir -p "$d" "$CD"
  local cbflag=""
  case "$MODE" in
    cb)       cbflag="--xla_gpu_enable_command_buffer=$CBMAX" ;;
    cbunroll) cbflag="--xla_gpu_enable_command_buffer=$CBMAX --xla_gpu_command_buffer_unroll_loops=true" ;;
  esac
  log "=== $name: B=$B chunk=$CS ce=$CE cb=$MODE cache=$CD"
  (
    export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.96
    export JAX_COMPILATION_CACHE_DIR="$CD"
    export TF_GPU_ALLOCATOR=cuda_malloc_async AXIOM_KDA_SOLVE=jax
    export XLA_FLAGS="--xla_gpu_autotune_level=2 $cbflag"
    exec timeout -k 60 "$TMO" "$PY" tools/pretrain_run.py \
      --model-preset l3-full --grad-checkpointing --grad-checkpointing-policy full \
      --steps 4 --total-steps 4 --seq-len 8192 --batch-size "$B" --micro-batch "$B" \
      --budget-limit-usd 5 --peak-tflops 98.2 --ce-chunk-tokens "$CE" --remat-policy none \
      --chunk-size "$CS" \
      --shard-root /home/roman/axiom-run/raw-v2 \
      --tokens-root /home/roman/gb10-shared/datasets/axiom-pretrain-l3/tokens-v2 \
      --streams W --run-ref "sweep2c-$name" --out "$d" --metrics "$d/metrics.jsonl"
  ) > "$d/run.log" 2>&1
  local rc=$?
  local tok mfu lf ll
  read -r tok mfu lf ll <<< "$("$PY" - "$d/metrics.jsonl" <<'PYEOF'
import json,sys,statistics
rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
ok=[r for r in rows if r.get("step",0)>=2]
if ok:
    med=statistics.median(r["tokens_per_sec"] for r in ok)
    print(f"{med:.1f}\t{med*6*502399236/98.2e12*100:.3f}\t{rows[0].get('loss','-')}\t{rows[-1].get('loss','-')}")
else:
    print("-\t-\t-\t-")
PYEOF
)"
  log "--- $name: rc=$rc tok/s=$tok mfu=$mfu% loss=$lf->$ll"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$B" "$CS" "$tok" "$mfu" "$rc" "$lf" "$ll" >> "$OUT/sweep2-summary.tsv"
  date -Is > "$OUT/heartbeat"
  if [ $rc -ne 0 ]; then fails=$((fails+1)); else fails=0; fi
  [ $fails -ge 3 ] && { log "СТОП: 3 отказа подряд"; exit 2; }
}

# (1) деньги вперёд: CB на лучшей конфигурации (свежий кэш — флаги реально применятся)
run_cell cbr_c64_B2  2 64 4096 cb       "$HOME/.cache/jax_cc_0112_cb"
# (2) CB + unroll while-лупов (отдельный свежий кэш — флаги разные, кэш общий нельзя)
run_cell cbru_c64_B2 2 64 4096 cbunroll "$HOME/.cache/jax_cc_0112_cbun"
# (3) изоляция вклада CB на базе B1 (кэш cb уже прогрет той же парой флагов — новый HLO, честная компиляция)
run_cell cbr_c064_B1 1 64 4096 cb       "$HOME/.cache/jax_cc_0112_cb"
# (4) ретрай c128_B2 (основной кэш; если снова 137 — вердикт: host-OOM компилятора на chunk>=128 x B2)
run_cell c128_B2_retry 2 128 4096 none  "$HOME/.cache/jax_cc_0112"

log "СЕРИЯ-2c ЗАВЕРШЕНА"
BEST=$(awk -F'\t' 'NR>1 && $5!="-" && $5+0>m {m=$5+0; n=$1} END {print n"\t"m}' "$OUT/sweep2-summary.tsv")
log "лучший результат всех серий: $BEST (mfu %)"
{ printf 'cell\tbatch\tchunk\ttok_s\tmfu_pct\trc\tloss_first\tloss_last\n'; tail -n +2 "$OUT/sweep2-summary.tsv"; } > "$OUT/sweep2-summary.txt"
date -Is > "$OUT/heartbeat"
