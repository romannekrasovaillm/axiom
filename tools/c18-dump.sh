#!/usr/bin/env bash
# c18-dump.sh — цикл-18 кампании MFU-55 (ADR-054 Am.29 п.1): HLO-дамп клетки
# persist-on B=4 для карты kernel→HLO (command_buffer_1229/985 в первую очередь).
#
# Свежий JAX-кэш намеренно: дампы пишутся только при компиляции; тёплый кэш
# отдал бы готовый executable и не дал HLO. Компиляция ~20–25 мин, исполнение
# 1 шага (пик ~82.5 ГиБ; стенд освобождён по решению владельца 18.10).
#
# Кодовых правок нет. Выход: ~/axiom-run/xladump-c18 (HLO/LLVM/thunk_metadata).
set -u
cd /home/roman/axiom || exit 1
DUMP=${DUMP:-/home/roman/axiom-run/xladump-c18}
OUT=${OUT:-evidence/mfu-55/cycle18-hlomap}
rm -rf "$DUMP" && mkdir -p "$DUMP" "$OUT"
(
  export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.96
  export JAX_COMPILATION_CACHE_DIR=$HOME/.cache/jax_cc_0112_c18_dump
  export TF_GPU_ALLOCATOR=cuda_malloc_async AXIOM_KDA_SOLVE=jax AXIOM_MLA_DENSE_FLASH=0
  export AXIOM_KDA_CC_IMPL=persistent AXIOM_GRAD_CKPT_POLICY=per_layer_block
  export AXIOM_MASTER_DTYPE=bf16 AXIOM_MASTER_COMP=kahan
  export AXIOM_MLA_BWD_BF16=1 AXIOM_CE_BF16=1 AXIOM_COMPUTE_DTYPE=bf16 AXIOM_MLA_PERSIST=1
  export XLA_FLAGS="--xla_dump_to=$DUMP --xla_dump_include_timestamp=false --xla_dump_hlo_as_text --xla_gpu_autotune_level=1"
  exec timeout -k 60 6900 /home/roman/venv-axiom-0112/bin/python tools/pretrain_run.py \
    --model-preset l3-full --grad-checkpointing --grad-checkpointing-policy full \
    --seq-len 8192 --batch-size 4 --micro-batch 4 --steps 1 --total-steps 1 \
    --budget-limit-usd 5 --peak-tflops 98.2 --ce-chunk-tokens 2048 \
    --remat-policy none --chunk-size 32 \
    --shard-root /home/roman/axiom-run/raw-v2 \
    --tokens-root /home/roman/gb10-shared/datasets/axiom-pretrain-l3/tokens-v2 \
    --streams W --run-ref c18-dump --out "$OUT" --metrics "$OUT/metrics.jsonl"
) > "$OUT/run.log" 2>&1
RC=$?
echo "[c18-dump] rc=$RC" | tee -a "$OUT/summary.txt"
du -sh "$DUMP" | tee -a "$OUT/summary.txt"
ls "$DUMP" | wc -l | tee -a "$OUT/summary.txt"
