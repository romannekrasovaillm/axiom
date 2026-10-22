#!/usr/bin/env bash
# c17-profile.sh — цикл-17 кампании MFU-55 (ADR-054 Am.28): nsys+NVTX атрибуция
# стационарного шага B=4 на актуальной клетке лидера.
#
# Зачем: вся существующая атрибуция (Am.27) снята с трейса цикла-6 на B=2 и на
# стеке десяти циклов назад. Правило Am.25 запрещает фьюзинг не-KDA части без
# NVTX-атрибуции на той клетке, которую оптимизируем. Этот прогон даёт:
#   - GPU-время по классам ядер (cuda_gpu_kern_sum),
#   - хост-API и синхронизации (cuda_api_sum),
#   - NVTX-спаны шага (Thunk/XlaCompile/while/remat),
#   - разложение 39.28 с/шаг с покрытием.
#
# Две ноги: leader-стек flash-bwd (818.9 в c15) и persist-on (834.2 в c16).
# Кэш компиляции переиспользуется от c16 — холодной компиляции (24 мин) нет
# у persist-on; persist-off компилируется заново (его кэш был убит rc=137).
#
# Требование стенда: ~80 ГиБ unified memory (jit_loss_fn 70.02 + jit_step 15.50
# по HLO-дампу c15). Чужие нагрузки (llama-server) должны быть остановлены
# владельцем: AD-7/C-040, урок INC-009. Лок: ~/gb10-shared/.locks/mfu-55.lock.
#
# Кодовых правок не делает: read-only измерение.
set -u
cd /home/roman/axiom || exit 1
OUT=${OUT:-evidence/mfu-55/cycle17-profile}
STEPS=${STEPS:-4}
mkdir -p "$OUT"
NSYS=$(command -v nsys || true)
if [ -z "$NSYS" ]; then
  for c in /usr/local/bin/nsys /usr/local/cuda/bin/nsys /opt/nvidia/nsight-systems/bin/nsys; do
    [ -x "$c" ] && NSYS=$c && break
  done
fi
if [ -z "$NSYS" ]; then echo "[c17] ОТКАЗ: nsys не найден" >&2; exit 3; fi
echo "[c17] nsys=$NSYS steps=$STEPS out=$OUT"

run_cell () {
  local name=$1; shift
  local D=$OUT/$name; mkdir -p "$D"
  (
    export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.96
    export JAX_COMPILATION_CACHE_DIR=$HOME/.cache/jax_cc_0112_c16_$name
    export TF_GPU_ALLOCATOR=cuda_malloc_async AXIOM_KDA_SOLVE=jax AXIOM_MLA_DENSE_FLASH=1
    export XLA_FLAGS="--xla_gpu_autotune_level=1"
    export AXIOM_KDA_CC_IMPL=persistent AXIOM_GRAD_CKPT_POLICY=per_layer_block
    export AXIOM_MASTER_DTYPE=bf16 AXIOM_MASTER_COMP=kahan
    export AXIOM_MLA_BWD_BF16=1 AXIOM_CE_BF16=1 AXIOM_COMPUTE_DTYPE=bf16
    # GPU-сэмплер под nsys: занятость/мощность, чтобы сверить таймлайн с dmon.
    ( nvidia-smi --query-gpu=timestamp,utilization.gpu,power.draw,temperature.gpu \
        --format=csv,noheader -lms 500 > "$D/gpu-sampler.csv" 2>&1 & echo $! > "$D/sampler.pid" )
    exec env "$@" "$NSYS" profile --stats=true --force-overwrite=true \
      --trace=cuda,nvtx,osrt --osrt-threshold=1000 --capture-range=none --sample=none \
      --output "$D/$name" \
      timeout -k 60 6900 /home/roman/venv-axiom-0112/bin/python tools/pretrain_run.py \
        --model-preset l3-full --grad-checkpointing --grad-checkpointing-policy full \
        --seq-len 8192 --batch-size 4 --micro-batch 4 \
        --steps "$STEPS" --total-steps "$STEPS" \
        --budget-limit-usd 5 --peak-tflops 98.2 --ce-chunk-tokens 2048 \
        --remat-policy none --chunk-size 32 \
        --shard-root /home/roman/axiom-run/raw-v2 \
        --tokens-root /home/roman/gb10-shared/datasets/axiom-pretrain-l3/tokens-v2 \
        --streams W --run-ref c17-$name --out "$D" --metrics "$D/metrics.jsonl"
  ) > "$D/run.log" 2>&1
  local RC=$?
  [ -f "$D/sampler.pid" ] && kill "$(cat "$D/sampler.pid")" 2>/dev/null
  echo "[c17 $name] rc=$RC" | tee -a "$OUT/summary.txt"
  /home/roman/venv-axiom-0112/bin/python - "$D/metrics.jsonl" "$name" <<'PYEOF' | tee -a "$OUT/summary.txt"
import json,sys,statistics
path,name = sys.argv[1], sys.argv[2]
try:
    rows=[json.loads(l) for l in open(path) if l.strip()]
except FileNotFoundError:
    print(f"[c17 {name}] нет metrics"); raise SystemExit
st=[r for r in rows if r.get("step",0)>=2]
if not st:
    print(f"[c17 {name}] нет стационара (шагов {len(rows)})"); raise SystemExit
print("[c17 %s] стационар: %.1f ток/с, шаг %.3f с, MFU %.3f%%, nan %d, n=%d" % (
    name, statistics.median(r["tokens_per_sec"] for r in st),
    statistics.median(r["step_seconds"] for r in st),
    statistics.median(r["mfu"] for r in st)*100,
    sum(1 for r in rows if r.get("loss")!=r.get("loss")), len(st)))
PYEOF
}

# persist-on: кэш c16 тёплый, база 834.2; persist-off: база 818.9, компиляция заново.
run_cell persist-on AXIOM_MLA_PERSIST=1 AXIOM_MLA_DENSE_FLASH=0 AXIOM_MLA_FLASH_BWD=0
run_cell persist-off AXIOM_MLA_DENSE_FLASH=1 AXIOM_MLA_FLASH_BWD=1
echo "[c17] ЗАВЕРШЕНО"
