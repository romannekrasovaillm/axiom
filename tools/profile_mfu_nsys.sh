#!/usr/bin/env bash
#
# tools/profile_mfu_nsys.sh — MFU стадия 2, прибор nsys.
#
# Тот же короткий прогон, что и tools/profile_mfu.py, но под
# `nsys profile --stats=true`: таймлайн CUDA-ядер и CUDA-API хоста.  В каталог
# evidence/mfu-profile/nsys-<name>/ ложатся:
#
#   <name>.nsys-rep        — бинарный отчёт nsys (носитель для nsys stats)
#   <name>-nsys.log        — stdout/stderr прогона под --stats=true
#   cuda_gpu_kern_sum.txt  — сводка по CUDA-ядрам (nsys stats)
#   cuda_api_sum.txt       — сводка по CUDA-API (nsys stats)
#   <name>-profile.json    — отчёт profile_mfu.py без jax-трейса (--no-trace)
#
# jax.profiler.trace намеренно отключён (--no-trace): профилировщик внутри
# профилировщика смешал бы окна.  Оба прибора независимы и калибруются друг
# против друга.
#
# Прогоны на стенде выполняет архитектор (AD-7/C-040: одна нагрузка за раз —
# перед запуском взять лок на ~/gb10-shared/.locks).  nsys без GPU/стенда не
# проходит: скрипт падает внятной ошибкой, а не «тихо ничего не делает».
#
# Флаги (те же имена, что у profile_mfu.py):
#   --config --batch --seq --steps --warmup --mode --name --out-dir --trace-dir
#   --all       — прогнать все три клетки кампании (план берётся у profile_mfu.py)
#   --dry-run   — напечатать команды nsys, ничего не исполняя (nsys не требуется)
#
# Обнаружение nsys: $NSYS (если задан — берётся как есть), иначе PATH,
# затем /usr/local/bin/nsys, /usr/local/cuda/bin/nsys.  Ничего не нашлось —
# ошибка и exit 3.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASE_DIR="$(cd "$HERE/.." && pwd)"
PY="${PYTHON:-python3}"

NAME="dense124m-b1"
CONFIG="net/config-dense124m.json"
BATCH=1
SEQ=8192
STEPS=6
WARMUP=2
MODE="fp32"
OUT_DIR="evidence/mfu-profile"
TRACE_DIR=""
ALL=0
DRY_RUN=0

usage() {
  sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)    CONFIG="$2"; shift 2 ;;
    --batch)     BATCH="$2"; shift 2 ;;
    --seq)       SEQ="$2"; shift 2 ;;
    --steps)     STEPS="$2"; shift 2 ;;
    --warmup)    WARMUP="$2"; shift 2 ;;
    --mode)      MODE="$2"; shift 2 ;;
    --name)      NAME="$2"; shift 2 ;;
    --out-dir)   OUT_DIR="$2"; shift 2 ;;
    --trace-dir) TRACE_DIR="$2"; shift 2 ;;
    --all)       ALL=1; shift ;;
    --dry-run)   DRY_RUN=1; shift ;;
    -h|--help)   usage; exit 0 ;;
    *) echo "неизвестный флаг: $1" >&2; usage >&2; exit 2 ;;
  esac
done

resolve_nsys() {
  if [[ -n "${NSYS:-}" ]]; then
    if [[ -x "$NSYS" ]]; then echo "$NSYS"; return 0; fi
    echo "ОШИБКА: nsys не найден: NSYS=$NSYS не исполняем." >&2
    return 3
  fi
  local found
  if found="$(command -v nsys 2>/dev/null)" && [[ -n "$found" ]]; then
    echo "$found"; return 0
  fi
  local candidate
  for candidate in /usr/local/bin/nsys /usr/local/cuda/bin/nsys; do
    if [[ -x "$candidate" ]]; then echo "$candidate"; return 0; fi
  done
  echo "ОШИБКА: nsys не найден ни в PATH, ни в /usr/local/bin, ни в /usr/local/cuda/bin." >&2
  echo "Установите Nsight Systems или задайте путь: NSYS=/path/to/nsys $0 ..." >&2
  return 3
}

# one cell: profile under nsys, then export kernel/API summaries
run_cell() {
  local nsys="$1" name="$2" config="$3" batch="$4" seq="$5"
  local cell_dir="$OUT_DIR/nsys-$name"
  local rep="$cell_dir/$name"
  local profile_json="$cell_dir/$name-profile.json"
  local log="$cell_dir/$name-nsys.log"

  if [[ -n "$TRACE_DIR" ]]; then
    local tdir="$TRACE_DIR"
  else
    tdir="$cell_dir/trace"
  fi

  local pycmd=("$PY" "$HERE/profile_mfu.py" --no-trace
    --name "$name" --config "$config" --batch "$batch" --seq "$seq"
    --steps "$STEPS" --warmup "$WARMUP" --mode "$MODE"
    --out "$profile_json" --trace-dir "$tdir")

  local prof=(profile --stats=true --force-overwrite=true --output "$rep"
    "${pycmd[@]}")

  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '%s %q' "$nsys" "${prof[0]}"
    local a
    for a in "${prof[@]:1}"; do printf ' %q' "$a"; done
    printf ' > %q\n' "$log"
    printf '%s stats --report cuda_gpu_kern_sum --format table --force-export=true %q > %q\n' \
      "$nsys" "$rep.nsys-rep" "$cell_dir/cuda_gpu_kern_sum.txt"
    printf '%s stats --report cuda_api_sum --format table --force-export=true %q > %q\n' \
      "$nsys" "$rep.nsys-rep" "$cell_dir/cuda_api_sum.txt"
    return 0
  fi

  mkdir -p "$cell_dir"
  echo "[profile-mfu-nsys] $name → $cell_dir" >&2
  if ! "$nsys" "${prof[@]}" > "$log" 2>&1; then
    echo "ОШИБКА: nsys profile упал для клетки $name — смотри $log" >&2
    return 1
  fi
  if [[ ! -f "$rep.nsys-rep" ]]; then
    echo "ОШИБКА: после nsys profile нет $rep.nsys-rep (лог: $log)" >&2
    return 1
  fi

  local report
  for report in cuda_gpu_kern_sum cuda_api_sum; do
    if ! "$nsys" stats --report "$report" --format table --force-export=true \
        "$rep.nsys-rep" > "$cell_dir/$report.txt" 2>&1; then
      echo "ОШИБКА: nsys stats --report $report упал для $name" >&2
      return 1
    fi
  done
  echo "[profile-mfu-nsys] $name: готово ($rep.nsys-rep + 2 сводки)" >&2
  return 0
}

# --- plan / execution ------------------------------------------------------
if [[ "$DRY_RUN" -eq 1 ]]; then
  nsys="nsys"  # в dry-run путь не важен
else
  if ! nsys="$(resolve_nsys)"; then
    exit 3
  fi
fi

if [[ "$ALL" -eq 1 ]]; then
  # план клеток — единый источник: profile_mfu.py --plan (TSV)
  while IFS=$'\t' read -r p_name p_config p_batch p_seq; do
    [[ -z "$p_name" ]] && continue
    run_cell "$nsys" "$p_name" "$p_config" "$p_batch" "$p_seq"
  done < <("$PY" "$HERE/profile_mfu.py" --plan)
else
  run_cell "$nsys" "$NAME" "$CONFIG" "$BATCH" "$SEQ"
fi
