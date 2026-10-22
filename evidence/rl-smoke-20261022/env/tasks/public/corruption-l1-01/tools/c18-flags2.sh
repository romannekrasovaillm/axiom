#!/usr/bin/env bash
# c18-flags.sh — цикл-18, шаг 2 (Am.29): скрининг XLA-флагов GEMM-пути на клетке
# persist-on B=4. Адрес: remat-хвост (command_buffer_1229/985) собран из мелких
# f32 Triton nested GEMM (`__triton_nested_gemm_fusion`, 1 стадия, 4 warps,
# тайл 1x64x32) — 1024+ запусков на класс. Проверяем, перекладывается ли этот
# класс на batched cuBLAS / новый fusion-путь / warp specialization.
#
# Клетки (3 шага, без nsys; метрики — шаг/ток-с):
#   ctrl        — базовый стек c16 (кэш тёплый, компиляции нет)
#   no_triton   — --xla_gpu_enable_triton_gemm=false (batched cuBLAS)
#   gemm_v2     — --xla_gpu_experimental_gemm_fusion_v2=true
#   warp_spec   — warp specialization + triton_gemm_any
#
# Приёмка клетки: rc=0, nan 0, mediana шагов 2-3; выигрыш ≥3% шага — кандидат
# на повторный проход (шум кампании ~1-2%). Кодовых правок нет.
set -u
cd /home/roman/axiom || exit 1
OUT=${OUT:-evidence/mfu-55/cycle18-flags2}
STEPS=${STEPS:-3}
mkdir -p "$OUT"
run_cell () {
  local name=$1; shift
  local cache=$1; shift
  local xla="$1"; shift
  local D=$OUT/$name; mkdir -p "$D"
  (
    export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.96
    export JAX_COMPILATION_CACHE_DIR=$cache
    export TF_GPU_ALLOCATOR=cuda_malloc_async AXIOM_KDA_SOLVE=jax AXIOM_MLA_DENSE_FLASH=0
    export AXIOM_KDA_CC_IMPL=persistent AXIOM_GRAD_CKPT_POLICY=per_layer_block
    export AXIOM_MASTER_DTYPE=bf16 AXIOM_MASTER_COMP=kahan
    export AXIOM_MLA_BWD_BF16=1 AXIOM_CE_BF16=1 AXIOM_COMPUTE_DTYPE=bf16 AXIOM_MLA_PERSIST=1
    export XLA_FLAGS="$xla"
    exec timeout -k 60 6900 /home/roman/venv-axiom-0112/bin/python tools/pretrain_run.py \
      --model-preset l3-full --grad-checkpointing --grad-checkpointing-policy full \
      --seq-len 8192 --batch-size 4 --micro-batch 4 --steps "$STEPS" --total-steps "$STEPS" \
      --budget-limit-usd 5 --peak-tflops 98.2 --ce-chunk-tokens 2048 \
      --remat-policy none --chunk-size 32 \
      --shard-root /home/roman/axiom-run/raw-v2 \
      --tokens-root /home/roman/gb10-shared/datasets/axiom-pretrain-l3/tokens-v2 \
      --streams W --run-ref c18-$name --out "$D" --metrics "$D/metrics.jsonl"
  ) > "$D/run.log" 2>&1
  local rc=$?
  /home/roman/venv-axiom-0112/bin/python - "$D/metrics.jsonl" "$name" "$rc" <<'PYEOF' | tee -a "$OUT/summary.txt"
import json,sys,statistics
path,name,rc = sys.argv[1],sys.argv[2],sys.argv[3]
try:
    rows=[json.loads(l) for l in open(path) if l.strip()]
except FileNotFoundError:
    print(f"[c18 {name}] rc={rc} нет metrics"); raise SystemExit
st=[r for r in rows if r.get("step",0)>=2]
if st:
    print("[c18 %s] rc=%s med %.1f ток/с, шаг %.3f с, MFU %.3f%%, nan %d, n=%d, loss %s" % (
        name, rc, statistics.median(r["tokens_per_sec"] for r in st),
        statistics.median(r["step_seconds"] for r in st),
        statistics.median(r["mfu"] for r in st)*100,
        sum(1 for r in rows if r.get("loss")!=r.get("loss")), len(st),
        [round(r["loss"],4) for r in rows]))
else:
    print(f"[c18 {name}] rc={rc} нет стационара (шагов {len(rows)})")
PYEOF
}




echo "[c18-flags2] ЗАВЕРШЕНО"
run_cell ragged_gg  "$HOME/.cache/jax_cc_0112_c18_ragged" "--xla_gpu_autotune_level=1 --xla_gpu_experimental_use_ragged_dot_grouped_gemm=true"
run_cell ragged_ws  "$HOME/.cache/jax_cc_0112_c18_raggedws" "--xla_gpu_autotune_level=1 --xla_gpu_experimental_use_ragged_dot_grouped_gemm=true --xla_gpu_experimental_enable_triton_warp_specialization=true"
run_cell v2_ws      "$HOME/.cache/jax_cc_0112_c18_v2ws" "--xla_gpu_autotune_level=1 --xla_gpu_experimental_gemm_fusion_v2=true --xla_gpu_experimental_enable_triton_warp_specialization=true --xla_gpu_triton_gemm_any=true"
echo "[c18-flags2] ЗАВЕРШЕНО"
