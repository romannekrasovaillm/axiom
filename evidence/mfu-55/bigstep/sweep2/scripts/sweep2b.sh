#!/usr/bin/env bash
# sweep2b.sh — серия-2b: адаптивный discovery CB-значений + пропущенные комбинации серии-2.
# Причина: probe серии-2 отверг 'DOT' (Illegal value for --xla_gpu_enable_command_buffer),
# а выбор лучшего чанка сломался на sort в локали ru_RU (ушёл в лексикографику) — комбинация
# c128_B2 и CB-клетки не отработали. Здесь: (0) ждать sweep2; (1) дешёвый discovery
# допустимых значений флага (import jax умирает за ~1 с на нелегальном значении);
# (2) клетки: c128_B2, CB-изоляция на B1, комбинации на B2, условно c128_B4.
# Результаты ДОПИСЫВАЮТСЯ в общий sweep2-summary.tsv (nsys2 берёт глобальный максимум).
set -uo pipefail
ROOT=/home/roman/axiom
OUT=$ROOT/evidence/mfu-55/bigstep/sweep2
PY=/home/roman/venv-axiom-0112/bin/python
TMO=${CELL_TIMEOUT:-2400}
cd "$ROOT" || exit 1
log() { printf '[sweep2b %s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$OUT/sweep2b.log"; }

# (0) ждать sweep2.sh (не более 6 ч); паттерн не матчит сам sweep2b
w=0
while pgrep -f "sweep2[.]sh" >/dev/null 2>&1; do
  [ "$w" -ge 21600 ] && { log "sweep2 идёт >6ч — продолжаю"; break; }
  sleep 60; w=$((w+60))
done
log "sweep2 завершена (или ожидание прекращено)"

# (1) discovery: какие значения флага принимает парсер 0.11.2 (база FUSION,CUBLAS,CUDNN легальна)
CB_BASE="FUSION,CUBLAS,CUDNN"
CBMAX="$CB_BASE"
for V in WHILE CONDITIONAL COMPUTE COLLECTIVE PARAMETER; do
  if XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 \
     XLA_FLAGS="--xla_gpu_enable_command_buffer=$CB_BASE,$V" \
     timeout 90 "$PY" -c "import jax; import jax.numpy as jnp; jnp.zeros(8).block_until_ready()" >/dev/null 2>&1; then
    CBMAX="$CBMAX,$V"; log "значение $V: принято"
  else
    log "значение $V: отвергнуто"
  fi
done
printf '%s' "$CBMAX" > "$OUT/cbmax.txt"
log "итоговый набор CB: $CBMAX"
HAS_WHILE=0; case "$CBMAX" in *WHILE*) HAS_WHILE=1;; esac

fails=0
run_cell() {  # name batch chunk ce cbmode
  local name="$1" B="$2" CS="$3" CE="$4" MODE="$5"
  local d="$OUT/$name"; mkdir -p "$d"
  local cbflag=""
  case "$MODE" in
    cb)       cbflag="--xla_gpu_enable_command_buffer=$CBMAX" ;;
    cbunroll) cbflag="--xla_gpu_enable_command_buffer=$CBMAX --xla_gpu_command_buffer_unroll_loops=true" ;;
  esac
  log "=== $name: B=$B chunk=$CS ce=$CE cb=$MODE"
  (
    export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.96
    export JAX_COMPILATION_CACHE_DIR=$HOME/.cache/jax_cc_0112
    export TF_GPU_ALLOCATOR=cuda_malloc_async AXIOM_KDA_SOLVE=jax
    export XLA_FLAGS="--xla_gpu_autotune_level=2 $cbflag"
    exec timeout -k 60 "$TMO" "$PY" tools/pretrain_run.py \
      --model-preset l3-full --grad-checkpointing --grad-checkpointing-policy full \
      --steps 4 --total-steps 4 --seq-len 8192 --batch-size "$B" --micro-batch "$B" \
      --budget-limit-usd 5 --peak-tflops 98.2 --ce-chunk-tokens "$CE" --remat-policy none \
      --chunk-size "$CS" \
      --shard-root /home/roman/axiom-run/raw-v2 \
      --tokens-root /home/roman/gb10-shared/datasets/axiom-pretrain-l3/tokens-v2 \
      --streams W --run-ref "sweep2b-$name" --out "$d" --metrics "$d/metrics.jsonl"
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

# (2a) пропущенная комбинация серии-2: лучший чанк (128) × B2
run_cell c128_B2 2 128 4096 none      # MDV ≥ 1.55% (1.469 × chunk-бонус)

# (2b) CB-изоляция на базе B1 (база легальна всегда)
run_cell cb_c064_B1 1 64 4096 cb      # MDV ≥ 1.35%
[ "$HAS_WHILE" = "1" ] && run_cell cbunroll_c064_B1 1 64 4096 cbunroll  # MDV ≥ 1.45%

# (2c) комбинации на B2 — цель S1 ≥ 3% (977 ток/с)
run_cell cb_c128_B2 2 128 4096 cb
[ "$HAS_WHILE" = "1" ] && run_cell cbunroll_c128_B2 2 128 4096 cbunroll

# (2d) если b4_ce2048 прошёл — продолжить лестницу батча с лучшим чанком
B4RC=$(awk -F'\t' '$1=="b4_ce2048" {print $6}' "$OUT/sweep2-summary.tsv" | head -1)
if [ "$B4RC" = "0" ]; then
  log "b4_ce2048 прошёл (rc=0) — добавляю c128_B4"
  run_cell c128_B4 4 128 2048 none
else
  log "b4_ce2048 не прошёл (rc=${B4RC:-нет}) — лестница батча остаётся на B2"
fi

# лучший результат серии (awk-max, без sort — локале не доверяем)
BEST=$(awk -F'\t' 'NR>1 && $5!="-" && $5+0>m {m=$5+0; n=$1} END {print n"\t"m}' "$OUT/sweep2-summary.tsv")
log "СЕРИЯ-2b ЗАВЕРШЕНА; лучший результат: $BEST (mfu %)"
{ printf 'cell\tbatch\tchunk\ttok_s\tmfu_pct\trc\tloss_first\tloss_last\n'; tail -n +2 "$OUT/sweep2-summary.tsv"; } > "$OUT/sweep2-summary.txt"
date -Is > "$OUT/heartbeat"
