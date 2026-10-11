#!/usr/bin/env bash
# sweep2.sh — серия-2 кампании MFU-55 (ADR-050/053, Am.2): диспетчеризация + укрупнение чанка.
# Контекст (доказанные адреса):
#   - 95% времени шага — хост-диспетчеризация (85k запусков ядер/шаг, GPU-вычислений ~1 с);
#   - дефолт --chunk-size 64 → 128 итераций KDA-скана × 18 слоёв × 3 прохода;
#   - XLA 0.11.2: command buffers для WHILE/CONDITIONAL/DOT/COMPUTE +
#     --xla_gpu_command_buffer_unroll_loops (unroll при известном счётчике);
#   - B=4 OOM: HLO-пик 109.38 GiB при gc full (bladder) — ремедио-клетка ce2048.
# Порядок: (1) ждать mfu55-queue.sh; (2) probe CB-enum; (3) c-лестница B=1;
#   (4) CB-клетки B=1; (5) комбинации B=2; (6) ремедио B4; (7) сводка.
# Метрика: mfu = tokens_per_sec × 6 × N_active(502399236) / 98.2e12, медиана шагов ≥ 2; loss-контроль.
set -uo pipefail
ROOT=/home/roman/axiom
OUT=$ROOT/evidence/mfu-55/bigstep/sweep2
PY=/home/roman/venv-axiom-0112/bin/python
TMO=${CELL_TIMEOUT:-2400}
mkdir -p "$OUT"; cd "$ROOT" || exit 1
log() { printf '[sweep2 %s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$OUT/sweep2.log"; }
printf 'cell\tbatch\tchunk\ttok_s\tmfu_pct\trc\tloss_first\tloss_last\n' > "$OUT/sweep2-summary.tsv"

# (1) ждём очередь серии-1 (не более 6 ч)
w=0
while pgrep -f "mfu55-queue[.]sh" >/dev/null 2>&1; do
  [ "$w" -ge 21600 ] && { log "очередь идёт >6ч — продолжаю без неё"; break; }
  sleep 60; w=$((w+60))
done
log "очередь завершена (или ожидание прекращено); сводка queue:"
cat "$ROOT/evidence/mfu-55/bigstep/queue/queue-summary.txt" 2>/dev/null | tee -a "$OUT/sweep2.log"

# (2) probe: регистр enum command buffer (быстрый 1-шаговый прогон, B=1, дефолт чанк)
CBVALS_UP="FUSION,CUBLAS,CUDNN,DOT,WHILE,CONDITIONAL,COMPUTE"
CBVALS_LO="fusion,cublas,cudnn,dot,while,conditional,compute"
probe_cell() {  # $1=значения флага; stdout: rc
  local vals="$1" d="$OUT/probe_cb"; rm -rf "$d"; mkdir -p "$d"
  (
    export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.96
    export JAX_COMPILATION_CACHE_DIR=$HOME/.cache/jax_cc_0112
    export TF_GPU_ALLOCATOR=cuda_malloc_async AXIOM_KDA_SOLVE=jax
    export XLA_FLAGS="--xla_gpu_autotune_level=2 --xla_gpu_enable_command_buffer=$vals"
    exec timeout -k 60 900 "$PY" tools/pretrain_run.py \
      --model-preset l3-full --grad-checkpointing --grad-checkpointing-policy full \
      --steps 1 --total-steps 1 --seq-len 8192 --batch-size 1 --micro-batch 1 \
      --budget-limit-usd 5 --peak-tflops 98.2 --ce-chunk-tokens 4096 --remat-policy none \
      --chunk-size 64 \
      --shard-root /home/roman/axiom-run/raw-v2 \
      --tokens-root /home/roman/gb10-shared/datasets/axiom-pretrain-l3/tokens-v2 --streams W \
      --run-ref sweep2-probe-cb --out "$d" --metrics "$d/metrics.jsonl"
  ) > "$d/run.log" 2>&1
  return $?
}
log "=== probe_cb (uppercase enum)"
if probe_cell "$CBVALS_UP"; then CBVALS="$CBVALS_UP"; log "probe: uppercase принят (rc=0)"
elif grep -qiE "unrecognized|invalid value|Illegal.*value|Unknown.*command" "$OUT/probe_cb/run.log" 2>/dev/null; then
  log "probe: uppercase отвергнут — пробую lowercase"
  if probe_cell "$CBVALS_LO"; then CBVALS="$CBVALS_LO"; log "probe: lowercase принят (rc=0)"
  else CBVALS=""; log "probe: оба регистра отвергнуты — CB-клетки пропускаются"; fi
else
  CBVALS="$CBVALS_UP"; log "probe: rc!=0 не из-за enum (смотреть probe_cb/run.log); uppercase считается принятым"
fi

fails=0
run_cell() {  # name batch chunk ce  [XLA-доп.флаги через CBFLAVOUR]
  local name="$1" B="$2" CS="$3" CE="$4"
  local d="$OUT/$name"; mkdir -p "$d"
  local cbflag=""
  case "${CBFLAVOUR:-none}" in
    cb)       cbflag="--xla_gpu_enable_command_buffer=$CBVALS" ;;
    cbunroll) cbflag="--xla_gpu_enable_command_buffer=$CBVALS --xla_gpu_command_buffer_unroll_loops=true" ;;
  esac
  log "=== $name: B=$B chunk=$CS ce=$CE cb=${CBFLAVOUR:-none}"
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
      --tokens-root /home/roman/gb10-shared/datasets/axiom-pretrain-l3/tokens-v2 --streams W \
      --run-ref "sweep2-$name" --out "$d" --metrics "$d/metrics.jsonl"
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

# (3) c-лестница на B=1: сокращение итераций KDA-скана (дефолт 64 → 128 итераций на 8192)
CBFLAVOUR=none
run_cell c064_B1  1 64   4096  # контроль = кампанийная база remat none (ожидание ~1.30%)
run_cell c128_B1  1 128  4096  # MDV ≥ 1.45%
run_cell c256_B1  1 256  4096  # MDV ≥ 1.55%
run_cell c512_B1  1 512  4096  # MDV ≥ 1.65% (или фиксация спада — потолок чанка)
run_cell c1024_B1 1 1024 4096  # MDV ≥ 1.70% / память C×C: контроль OOM

# лучший чанк по B=1 для комбинаций
BESTCS=$(awk -F'\t' 'NR>1 && $5!="-" && $5+0>0 {print $3"\t"$5}' "$OUT/sweep2-summary.tsv" | sort -k2 -nr | head -1 | cut -f1)
BESTCS=${BESTCS:-64}
log "лучший chunk по B=1: $BESTCS"

# (4) CB-клетки на B=1 (chunk дефолт 64 — изоляция вклада CB)
if [ -n "$CBVALS" ]; then
  CBFLAVOUR=cbunroll; run_cell cbunroll_c064_B1 1 64 4096   # MDV ≥ 1.6%: while-сканы в командных буферах
  CBFLAVOUR=cb;        run_cell cb_c064_B1 1 64 4096        # разделение вклада unroll_loops
  # (5) комбинации на B=2 — кандидаты в кампанийную конфигурацию; цель S1 ≥ 3% (977 ток/с)
  CBFLAVOUR=cbunroll; run_cell "cbunroll_c${BESTCS}_B2" 2 "$BESTCS" 4096
  CBFLAVOUR=cb;        run_cell "cb_c${BESTCS}_B2" 2 "$BESTCS" 4096
fi
# комбинация лучший chunk без CB на B=2 (контроль вклада CB на том же чанке)
CBFLAVOUR=none; run_cell "c${BESTCS}_B2" 2 "$BESTCS" 4096

# (6) ремедио B4: адрес HLO-пика 109 GiB — CE-чанк мельче (2048), контроль OOM
CBFLAVOUR=none; run_cell b4_ce2048 4 64 2048

log "СЕРИЯ-2 ЗАВЕРШЕНА"
{ printf 'cell\tbatch\tchunk\ttok_s\tmfu_pct\trc\tloss_first\tloss_last\n'; tail -n +2 "$OUT/sweep2-summary.tsv"; } > "$OUT/sweep2-summary.txt"
cat "$OUT/sweep2-summary.txt" | tee -a "$OUT/sweep2.log"
date -Is > "$OUT/heartbeat"
