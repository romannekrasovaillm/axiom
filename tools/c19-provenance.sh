#!/usr/bin/env bash
# c19-provenance.sh — цикл-19 (ADR-054 Am.31): nsys node-level + HLO-дамп клетки
# B=4 persist-on с включёнными провенанс-метками (AXIOM_TRACE_MARKERS=1).
#
# Цель: отнести command_buffer_1229/985 (хвост ~9.75 с GPU) к компоненту
# (probe_kda / probe_mla / probe_moe / probe_attnres / probe_ce / probe_mlp).
# Свежий JAX-кэш — иначе HLO-дампа не будет; markers меняют имена, поэтому
# старый кэш всё равно не подошёл бы.
#
# Требования: стенд свободен (B=4 ~82.5 ГиБ), ~1.1 ГБ под дамп, ~200 МБ под trace.
set -u
cd "${REPO:-/home/roman/axiom}" || exit 1
OUT=${OUT:-evidence/mfu-55/cycle19-provenance}
DUMP=${DUMP:-/home/roman/axiom-run/xladump-c19}
STEPS=${STEPS:-2}
CACHE=${CACHE:-$HOME/.cache/jax_cc_0112_c19_prov}
rm -rf "$DUMP" "$CACHE" && mkdir -p "$DUMP" "$OUT"
NSYS=$(command -v nsys || true)
[ -z "$NSYS" ] && for c in /usr/local/bin/nsys /usr/local/cuda/bin/nsys; do [ -x "$c" ] && NSYS=$c && break; done
[ -z "$NSYS" ] && { echo "[c19] ОТКАЗ: nsys не найден" >&2; exit 3; }

(
  export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.96
  export JAX_COMPILATION_CACHE_DIR=$CACHE
  export TF_GPU_ALLOCATOR=cuda_malloc_async AXIOM_KDA_SOLVE=jax AXIOM_MLA_DENSE_FLASH=0
  export AXIOM_KDA_CC_IMPL=persistent AXIOM_GRAD_CKPT_POLICY=per_layer_block
  export AXIOM_MASTER_DTYPE=bf16 AXIOM_MASTER_COMP=kahan
  export AXIOM_MLA_BWD_BF16=1 AXIOM_CE_BF16=1 AXIOM_COMPUTE_DTYPE=bf16 AXIOM_MLA_PERSIST=1
  export AXIOM_TRACE_MARKERS=1
  export XLA_FLAGS="--xla_dump_to=$DUMP --xla_dump_include_timestamp=false --xla_dump_hlo_as_text --xla_gpu_autotune_level=1"
  exec "$NSYS" profile --stats=true --force-overwrite=true \
    --trace=cuda,nvtx,osrt --osrt-threshold=1000 --cuda-graph-trace=node \
    --capture-range=none --sample=none --output "$OUT/prov" \
    timeout -k 60 6900 /home/roman/venv-axiom-0112/bin/python tools/pretrain_run.py \
      --model-preset l3-full --grad-checkpointing --grad-checkpointing-policy full \
      --seq-len 8192 --batch-size 4 --micro-batch 4 --steps "$STEPS" --total-steps "$STEPS" \
      --budget-limit-usd 5 --peak-tflops 98.2 --ce-chunk-tokens 2048 \
      --remat-policy none --chunk-size 32 \
      --shard-root /home/roman/axiom-run/raw-v2 \
      --tokens-root /home/roman/gb10-shared/datasets/axiom-pretrain-l3/tokens-v2 \
      --streams W --run-ref c19-prov --out "$OUT" --metrics "$OUT/metrics.jsonl"
) > "$OUT/run.log" 2>&1
RC=$?
echo "[c19] rc=$RC" | tee -a "$OUT/summary.txt"
du -sh "$DUMP" 2>/dev/null | tee -a "$OUT/summary.txt"
/home/roman/venv-axiom-0112/bin/python - "$OUT/metrics.jsonl" <<'PYEOF' | tee -a "$OUT/summary.txt"
import json,sys,statistics
try:
    rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
except FileNotFoundError:
    print("[c19] нет metrics"); raise SystemExit
st=[r for r in rows if r.get("step",0)>=2]
print("[c19] стационар: %s" % ("%.1f ток/с, шаг %.3f с, nan %d" % (
    statistics.median(r["tokens_per_sec"] for r in st),
    statistics.median(r["step_seconds"] for r in st),
    sum(1 for r in rows if r.get("loss")!=r.get("loss"))) if st else "нет"))
PYEOF
echo "[c19] ЗАВЕРШЕНО"
