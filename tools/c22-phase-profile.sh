#!/usr/bin/env bash
# c22-phase-profile.sh — цикл-22 (Am.34): пофазовый профиль шага B=4 на main.
#
# Зачем: провенанс-карта цикла-21 оставила 51.5% (15.2 с) неразмеченными
# (backward + библиотечные GEMM) и 1.91 с «оптимизатора» по модулю. Встроенный
# opt-in пофазовый профиль (``--phase-profile``) меряет sec_forward /
# sec_backward / sec_backopt / sec_other хост-таймерами с block_until_ready —
# это прямая атрибуция оптимизатора и backward без новых меток.
#
# Профиль срабатывает на kpi-интервале (kpi_every=20) -> 20 шагов; фьюзнутый
# шаг идёт штатно, профильные ноги — дополнительный декомпозированный прогон.
# Тёплый кэш c20: фьюзнутый шаг компиляции не требует; ноги профиля — да.
# Стенд: B=4 ~82.5 ГиБ; риск OOM профильных ног — при OOM фиксируем и идём B=2.
set -u
cd "${REPO:-/home/roman/axiom}" || exit 1
OUT=${OUT:-evidence/mfu-55/cycle22-phase}
STEPS=${STEPS:-20}
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
    --remat-policy none --chunk-size 32 --phase-profile \
    --shard-root /home/roman/axiom-run/raw-v2 \
    --tokens-root /home/roman/gb10-shared/datasets/axiom-pretrain-l3/tokens-v2 \
    --streams W --run-ref c22-phase --out "$OUT" --metrics "$OUT/metrics.jsonl"
) > "$OUT/run.log" 2>&1
RC=$?
echo "[c22] rc=$RC" | tee -a "$OUT/summary.txt"
/home/roman/venv-axiom-0112/bin/python - "$OUT/metrics.jsonl" <<'PYEOF' | tee -a "$OUT/summary.txt"
import json,sys,statistics
try:
    rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
except FileNotFoundError:
    print("[c22] нет metrics"); raise SystemExit
st=[r for r in rows if r.get("step",0)>=2]
if st:
    print("[c22] стационар n=%d: %.1f ток/с, шаг %.3f с, nan %d" % (
        len(st), statistics.median(r["tokens_per_sec"] for r in st),
        statistics.median(r["step_seconds"] for r in st),
        sum(1 for r in rows if r.get("loss")!=r.get("loss"))))
for r in rows:
    if r.get("phase_profile"):
        keys=["step","step_seconds","sec_forward","sec_backward","sec_backopt","sec_other",
              "sec_kda","sec_mla","sec_moe","sec_ce","sec_bwd_kda","sec_bwd_mla",
              "sec_bwd_moe","sec_bwd_ce","sec_bwd_other","reconciliation_pct","sec_loader"]
        print("[c22 phase]", {k: (round(r[k],3) if isinstance(r.get(k),(int,float)) else r.get(k)) for k in keys})
PYEOF
echo "[c22] ЗАВЕРШЕНО"
