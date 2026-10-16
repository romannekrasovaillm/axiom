#!/bin/bash
# Регресс вехи-12 по одному файлу за прогон (heartbeat в каждом).
# test_15_cost_8k_64k НЕ гоняем (указание задачи).
cd /home/roman/.arch-ml/worktrees/axiom-43e236d789188868/theseus-20261016050103
FILES="net/tests/test_03_mla_parity.py net/tests/test_29_compute_dtype.py net/tests/test_30_mla_flash_parity.py net/tests/test_31_mla_dense_compute_dtype.py net/tests/test_31_remat_policy.py net/tests/test_35_bf16_master.py net/tests/test_36_accum_jit.py net/tests/test_37_per_layer_block.py net/tests/test_38_grad_ckpt_env.py net/tests/test_39_bwd_bf16_storage.py net/tests/test_24_grad_checkpoint.py"
if [ -n "$1" ]; then FILES="$@"; fi
RC_ALL=0
for f in $FILES; do
  echo "########## $f ##########"
  bash .scratch/run_tests.sh "$f" | tail -n 3
  rc=${PIPESTATUS[0]}
  echo "########## $f -> rc=$rc"
  [ "$rc" -ne 0 ] && RC_ALL=1
done
echo "== regression rc_all=$RC_ALL =="
exit $RC_ALL
