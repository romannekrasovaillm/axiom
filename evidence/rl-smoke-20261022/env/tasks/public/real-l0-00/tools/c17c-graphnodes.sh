#!/usr/bin/env bash
# c17c-graphnodes.sh — цикл-17, шаг 2: nsys с РАСКРЫТИЕМ CUDA-графов по узлам.
#
# Почему: трейс c17-profile снят с дефолтным --cuda-graph-trace=graph. В нём
# видны 2612 cuGraphLaunch (103.17 с GPU-времени) и только 6885 отдельных ядер
# (12.57 с) — то есть 89% GPU-времени спрятано внутри графов, а `cuda_gpu_kern_sum`
# его не видит. Отсюда ложный вывод «GPU пуст 92%» (он же мог стоять за Am.15).
# С --cuda-graph-trace=node каждое ядро внутри графа становится событием, и
# cuda_gpu_kern_sum даёт настоящий ranked-список адресов кернел-войны.
#
# Клетка: persist-on (лидер, 834.2), 2 шага (кэш c16 тёплый), nsys + NVTX.
# Кодовых правок нет.
set -u
cd /home/roman/axiom || exit 1
OUT=${OUT:-evidence/mfu-55/cycle17-profile}
STEPS=${STEPS:-2}
D=$OUT/persist-off-graphnodes
mkdir -p "$D"
NSYS=$(command -v nsys || true)
if [ -z "$NSYS" ]; then
  for c in /usr/local/bin/nsys /usr/local/cuda/bin/nsys /opt/nvidia/nsight-systems/bin/nsys; do
    [ -x "$c" ] && NSYS=$c && break
  done
fi
if [ -z "$NSYS" ]; then echo "[c17c] ОТКАЗ: nsys не найден" >&2; exit 3; fi
echo "[c17c] nsys=$NSYS steps=$STEPS out=$D"

(
  export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.96
  export JAX_COMPILATION_CACHE_DIR=$HOME/.cache/jax_cc_0112_c16_persist-off
  export TF_GPU_ALLOCATOR=cuda_malloc_async AXIOM_KDA_SOLVE=jax AXIOM_MLA_DENSE_FLASH=1
  export XLA_FLAGS="--xla_gpu_autotune_level=1"
  export AXIOM_KDA_CC_IMPL=persistent AXIOM_GRAD_CKPT_POLICY=per_layer_block
  export AXIOM_MASTER_DTYPE=bf16 AXIOM_MASTER_COMP=kahan
  export AXIOM_MLA_BWD_BF16=1 AXIOM_CE_BF16=1 AXIOM_COMPUTE_DTYPE=bf16 AXIOM_MLA_FLASH_BWD=1
  exec "$NSYS" profile --stats=true --force-overwrite=true \
    --trace=cuda,nvtx,osrt --osrt-threshold=1000 \
    --cuda-graph-trace=node --capture-range=none --sample=none \
    --output "$D/persist-off-graphnodes" \
    timeout -k 60 6900 /home/roman/venv-axiom-0112/bin/python tools/pretrain_run.py \
      --model-preset l3-full --grad-checkpointing --grad-checkpointing-policy full \
      --seq-len 8192 --batch-size 4 --micro-batch 4 \
      --steps "$STEPS" --total-steps "$STEPS" \
      --budget-limit-usd 5 --peak-tflops 98.2 --ce-chunk-tokens 2048 \
      --remat-policy none --chunk-size 32 \
      --shard-root /home/roman/axiom-run/raw-v2 \
      --tokens-root /home/roman/gb10-shared/datasets/axiom-pretrain-l3/tokens-v2 \
      --streams W --run-ref c17c-graphnodes --out "$D" --metrics "$D/metrics.jsonl"
) > "$D/run.log" 2>&1
RC=$?
echo "[c17c] rc=$RC" | tee -a "$OUT/summary.txt"
/home/roman/venv-axiom-0112/bin/python - "$D/metrics.jsonl" <<'PYEOF' | tee -a "$OUT/summary.txt"
import json,sys,statistics
try:
    rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
except FileNotFoundError:
    print("[c17c] нет metrics"); raise SystemExit
st=[r for r in rows if r.get("step",0)>=2]
if st:
    print("[c17c] стационар: %.1f ток/с, шаг %.3f с, MFU %.3f%%, n=%d" % (
        statistics.median(r["tokens_per_sec"] for r in st),
        statistics.median(r["step_seconds"] for r in st),
        statistics.median(r["mfu"] for r in st)*100, len(st)))
else:
    print("[c17c] нет стационара (шагов %d)" % len(rows))
PYEOF
