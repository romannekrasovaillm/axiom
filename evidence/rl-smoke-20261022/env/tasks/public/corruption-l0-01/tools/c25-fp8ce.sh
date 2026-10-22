#!/usr/bin/env bash
# c25-fp8ce.sh — цикл-23 (Am.36): A/B размера CE-чанка на клетке G1 (B=4).
#
# Мотив: пофазовый профиль показал CE/lm_head ~3.15 с (fwd 0.89 + bwd 2.27)
# при ~14% пика; оба CE-головы (NTP и MTP) идут одним chunked-путём с
# ce_chunk_tokens=2048 (16 чанков на голову) — мелкие GEMM. Память больше не
# связывает (пик 82.5 ГиБ из 121; ce2048 выбирался под старый пол Am.20).
# Проверяем ce=8192 (4 чанка) против базы 2048 (база уже измерена: 1092.5 ток/с,
# 29.99 с). Численный результат тот же (иная редукция), паритет лосса в допуске.
#
# ВНИМАНИЕ (инцидент цикла-22): ce_chunk_tokens — статический аргумент графа,
# поэтому это ХОЛОДНАЯ компиляция fused-шага (~20–25 мин). Холодные компиляции
# самого шага раньше проходили; инцидент был на фазовых ногах. Наблюдать память,
# при swap-трэше — timeout сам убьёт (6900 с).
set -u
cd "${REPO:-/home/roman/axiom}" || exit 1
CE=${CE:-2048}
OUT=${OUT:-evidence/mfu-55/cycle25-fp8ce}
STEPS=${STEPS:-10}
CACHE=${CACHE:-$HOME/.cache/jax_cc_0112_c25_fp8ce}
mkdir -p "$OUT"
(
  export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.96
  export JAX_COMPILATION_CACHE_DIR=$CACHE
  export TF_GPU_ALLOCATOR=cuda_malloc_async AXIOM_KDA_SOLVE=jax AXIOM_MLA_DENSE_FLASH=0
  export AXIOM_KDA_CC_IMPL=persistent AXIOM_GRAD_CKPT_POLICY=per_layer_block
  export AXIOM_MASTER_DTYPE=bf16 AXIOM_MASTER_COMP=kahan
  export AXIOM_MLA_BWD_BF16=1 AXIOM_CE_BF16=1 AXIOM_COMPUTE_DTYPE=bf16 AXIOM_MLA_PERSIST=1
  export AXIOM_MTP_KDA_IMPL=campaign AXIOM_CE_FP8=1
  export XLA_FLAGS="--xla_gpu_autotune_level=1"
  exec timeout -k 60 6900 /home/roman/venv-axiom-0112/bin/python tools/pretrain_run.py \
    --model-preset l3-full --grad-checkpointing --grad-checkpointing-policy full \
    --seq-len 8192 --batch-size 4 --micro-batch 4 --steps "$STEPS" --total-steps "$STEPS" \
    --budget-limit-usd 5 --peak-tflops 98.2 --ce-chunk-tokens "$CE" \
    --remat-policy none --chunk-size 32 \
    --shard-root /home/roman/axiom-run/raw-v2 \
    --tokens-root /home/roman/gb10-shared/datasets/axiom-pretrain-l3/tokens-v2 \
    --streams W --run-ref c23-cefused --out "$OUT" --metrics "$OUT/metrics.jsonl"
) > "$OUT/run.log" 2>&1
RC=$?
echo "[c25 ce=$CE] rc=$RC" | tee -a "$OUT/summary.txt"
/home/roman/venv-axiom-0112/bin/python - "$OUT/metrics.jsonl" "$CE" <<'PYEOF' | tee -a "$OUT/summary.txt"
import json,sys,statistics
try:
    rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
except FileNotFoundError:
    print("[c25] нет metrics"); raise SystemExit
st=[r for r in rows if r.get("step",0)>=2]
if st:
    print("[c25 ce=%s] стационар n=%d: %.1f ток/с, шаг %.3f с, MFU %.3f%%, nan %d" % (
        sys.argv[2], len(st), statistics.median(r["tokens_per_sec"] for r in st),
        statistics.median(r["step_seconds"] for r in st),
        statistics.median(r["mfu"] for r in st)*100,
        sum(1 for r in rows if r.get("loss")!=r.get("loss"))))
    print("[c25 losses]", [round(r["loss"],5) for r in rows])
PYEOF
echo "[c25] ЗАВЕРШЕНО"
