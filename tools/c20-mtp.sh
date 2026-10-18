#!/usr/bin/env bash
# c20-mtp.sh — цикл-20 (ADR-054 Am.32): стендовый A/B MTP KDA на chunked_cc.
#
# База: c16/c17 persist-on B=4 — 834.2–840.8 ток/с, шаг 39.0–39.3 с.
# Клетка: AXIOM_MTP_KDA_IMPL=campaign (MTP KDA через apply_kda -> chunked_cc,
# персистентный кернел) + AXIOM_KDA_CC_IMPL=persistent; 10 шагов.
# Ожидание: −7…−9 с/шаг (снятие 256-графового jnp-хвоста) ⇒ 30–32 с, G1 (33.53 с).
# Приёмка: nan 0, loss-траектория в допуске (вес MTP 0.1), шаг ≤33.5 с.
#
# Стенд: B=4 ~82.5 ГиБ; свежий кэш (граф меняется) — компиляция ~25 мин.
set -u
cd "${REPO:-/home/roman/axiom}" || exit 1
OUT=${OUT:-evidence/mfu-55/cycle20-mtp}
STEPS=${STEPS:-10}
CACHE=${CACHE:-$HOME/.cache/jax_cc_0112_c20_mtp}
mkdir -p "$OUT"
(
  export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.96
  export JAX_COMPILATION_CACHE_DIR=$CACHE
  export TF_GPU_ALLOCATOR=cuda_malloc_async AXIOM_KDA_SOLVE=jax AXIOM_MLA_DENSE_FLASH=0
  export AXIOM_KDA_CC_IMPL=persistent AXIOM_GRAD_CKPT_POLICY=per_layer_block
  export AXIOM_MASTER_DTYPE=bf16 AXIOM_MASTER_COMP=kahan
  export AXIOM_MLA_BWD_BF16=1 AXIOM_CE_BF16=1 AXIOM_COMPUTE_DTYPE=bf16 AXIOM_MLA_PERSIST=1
  export AXIOM_MTP_KDA_IMPL=campaign
  export XLA_FLAGS="--xla_gpu_autotune_level=1"
  exec timeout -k 60 6900 /home/roman/venv-axiom-0112/bin/python tools/pretrain_run.py \
    --model-preset l3-full --grad-checkpointing --grad-checkpointing-policy full \
    --seq-len 8192 --batch-size 4 --micro-batch 4 --steps "$STEPS" --total-steps "$STEPS" \
    --budget-limit-usd 5 --peak-tflops 98.2 --ce-chunk-tokens 2048 \
    --remat-policy none --chunk-size 32 \
    --shard-root /home/roman/axiom-run/raw-v2 \
    --tokens-root /home/roman/gb10-shared/datasets/axiom-pretrain-l3/tokens-v2 \
    --streams W --run-ref c20-mtp --out "$OUT" --metrics "$OUT/metrics.jsonl"
) > "$OUT/run.log" 2>&1
RC=$?
echo "[c20] rc=$RC" | tee -a "$OUT/summary.txt"
/home/roman/venv-axiom-0112/bin/python - "$OUT/metrics.jsonl" <<'PYEOF' | tee -a "$OUT/summary.txt"
import json,sys,statistics
try:
    rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
except FileNotFoundError:
    print("[c20] нет metrics"); raise SystemExit
st=[r for r in rows if r.get("step",0)>=2]
if st:
    print("[c20] стационар n=%d: %.1f ток/с, шаг %.3f с, MFU %.3f%%, nan %d" % (
        len(st), statistics.median(r["tokens_per_sec"] for r in st),
        statistics.median(r["step_seconds"] for r in st),
        statistics.median(r["mfu"] for r in st)*100,
        sum(1 for r in rows if r.get("loss")!=r.get("loss"))))
else:
    print("[c20] нет стационара (шагов %d)" % len(rows))
print("[c20] losses:", [round(r["loss"], 5) for r in rows])
PYEOF
echo "[c20] ЗАВЕРШЕНО"
