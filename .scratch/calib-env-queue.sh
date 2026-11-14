#!/usr/bin/env bash
# Очередь калибровки среды (P-7a / C-056, ADR-058): ждёт освобождения GPU
# по AD-7/C-040 (одна модельная нагрузка за раз), берёт лок, запускает прогон.
#
# ИСПРАВЛЕНО 2026-11-14 после инцидента 08:13Z (глобальный OOM + reboot GB10):
#  1) занятость определяется НЕ только по nvidia-smi: между сегментами обучения
#     есть окно ~12 с, в которое GPU «свободен» — прежняя проверка попадала в него;
#     добавлена проверка процессов pretrain_run.py / long_jax2_ab_full.sh;
#  2) двойная проверка + стабилизационное окно 180 с (нагрузка не появилась);
#  3) требуется явное разрешение владельца: файл .locks/axiom-calib-env.approved;
#  4) обязательный XLA_PYTHON_CLIENT_MEM_FRACTION (урок 2026-09-26: дефолтные
#     75% JAX = OOM-каскад на GB10);
#  5) финальная TOCTOU-проверка непосредственно перед запуском python.
set -euo pipefail
cd "$HOME/axiom"

RUN_BASE="$HOME/gb10-shared/runs/axiom-long-pretrain-20261023"
TRAIN_PROCS='tools/pretrain_run.py|long_jax2_ab_full.sh'
APPROVAL="$HOME/gb10-shared/.locks/axiom-calib-env.approved"
LOCK="$HOME/gb10-shared/.locks/axiom-calib-env.lock"
WAIT_SECS="${WAIT_SECS:-300}"
STABLE_SECS="${STABLE_SECS:-180}"

LOG_DIR="$HOME/axiom-run/calib-env-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOG_DIR"
exec >>"$LOG_DIR/queue.log" 2>&1
echo "[$(date -Is)] очередь калибровки запущена; wait=${WAIT_SECS}s stable=${STABLE_SECS}s"

busy_reason() {
  local procs apps
  procs=$(pgrep -af "$TRAIN_PROCS" 2>/dev/null | grep -v -E "pgrep|calib-env-queue" | head -3 || true)
  if [ -n "$procs" ]; then
    echo "обучение: $(printf '%s' "$procs" | head -1 | cut -c1-140)"
    return 0
  fi
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
  if [ "${apps:-0}" -gt 0 ]; then
    echo "GPU compute-apps: $apps"
    return 0
  fi
  return 1
}

if [ ! -f "$APPROVAL" ]; then
  echo "[$(date -Is)] ОТКАЗ: нет разрешения владельца ($APPROVAL) — выходим"
  exit 4
fi
echo "[$(date -Is)] разрешение владельца: $(head -1 "$APPROVAL")"

while :; do
  if r=$(busy_reason); then
    echo "[$(date -Is)] занято ($r); ждём ${WAIT_SECS}s"
    sleep "$WAIT_SECS"
    continue
  fi
  echo "[$(date -Is)] свободно; стабилизационное окно ${STABLE_SECS}s"
  sleep "$STABLE_SECS"
  if r=$(busy_reason); then
    echo "[$(date -Is)] в окне появилась нагрузка ($r); продолжаем ждать"
    sleep "$WAIT_SECS"
    continue
  fi
  break
done

if ! mkdir "$LOCK" 2>/dev/null; then
  echo "[$(date -Is)] лок занят: $LOCK — выходим (другой прогон уже в очереди)"
  exit 3
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

# TOCTOU: финальная проверка прямо перед запуском python
if r=$(busy_reason); then
  echo "[$(date -Is)] перед стартом появилась нагрузка ($r) — отмена"
  exit 5
fi

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.5}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export AXIOM_KDA_CC_IMPL=jax AXIOM_KDA_SOLVE=jax

echo "[$(date -Is)] старт калибровки среды (P-7a); XLA_MEM_FRACTION=$XLA_PYTHON_CLIENT_MEM_FRACTION"
CKPT="$HOME/axiom-run/pilot-l3full-endurance/checkpoints/step-00000900"
NAME="l3-endurance-step900"
TOK="$HOME/axiom-run/raw/tokenizer/tokenizer-manifest.json"
FINAL="$RUN_BASE/long-jax2-ab14050/pretrain/active/step-00050000"
if [ -d "$FINAL" ]; then
  CKPT="$FINAL"; NAME="l3-long-jax2-step50000"
  TOK="$HOME/gb10-shared/datasets/axiom-pretrain-l3/tokenizer-v2/tokenizer-manifest.json"
fi
echo "[$(date -Is)] чекпоинт: $CKPT; токенизатор: $TOK"
"$HOME/venv-axiom/bin/python" -m env.main calibrate \
  --tasks env/data --case . --out "$LOG_DIR" \
  --adapter jaxlm \
  --checkpoint "$CKPT" \
  --config .scratch/config-calib-chunk32.json \
  --tokenizer-path "$TOK" \
  --model-name "$NAME" --model-seed 7
echo "[$(date -Is)] калибровка завершена; отчёт: $LOG_DIR/calibration-$NAME.json"
