"""Контракт smoke-скрипта Mosaic MMA — проверки на CPU (без GPU).

Проверяется ровно то, что можно проверить без ускорителя:

* CLI: ``--output PATH`` обязателен, отчёт реально пишется;
* структура JSON: оговорённый набор ключей и допустимое значение ``status``;
* на машине без GPU исход обязан быть ``blocked`` с причиной (``no-gpu``), а не
  ``correctness-pass`` — подмена проверки запрещена (C-007): ``hasattr``/импорт/CPU-прогон
  доказательством GPU-компиляции не считаются;
* коды возврата: 0 — успех/``blocked`` (отсутствие GPU не ошибка скрипта), 2 — нет ``--output``.

Исполнение на GPU сюда не входит и в CI не требуется: его делает архитектор на стенде.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tools" / "mosaic" / "mma_smoke.py"

REQUIRED_KEYS = {"status", "shape", "dtype", "max_abs", "rel", "seconds",
                 "error_type", "error", "stage", "cases"}
ALLOWED_STATUS = {"correctness-pass", "correctness-fail", "blocked"}


def _run(tmp_path: Path, *extra: str) -> tuple[int, dict, Path]:
    out = tmp_path / "mma.json"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(out), *extra],
        capture_output=True, text=True, timeout=600, cwd=str(REPO),
    )
    assert out.is_file(), f"отчёт не записан; stderr={proc.stderr[-800:]}"
    return proc.returncode, json.loads(out.read_text(encoding="utf-8")), out


def test_script_exists():
    assert SCRIPT.is_file(), f"нет скрипта {SCRIPT}"


def test_cli_requires_output():
    """Без ``--output`` argparse обязан отказать (код 2), ничего не выдумывая."""
    proc = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 2, proc.stderr[-400:]


def test_report_structure_and_status(tmp_path):
    code, doc, _ = _run(tmp_path)
    missing = REQUIRED_KEYS - set(doc)
    assert not missing, f"в отчёте нет ключей: {sorted(missing)}"
    assert doc["status"] in ALLOWED_STATUS, doc["status"]
    assert isinstance(doc["cases"], list) and doc["cases"], "список кейсов пуст"
    for case in doc["cases"]:
        assert case["status"] in ALLOWED_STATUS
        assert set(case) >= {"shape", "dtype", "status", "stage", "error_type"}
    assert code in (0, 1, 2), code


def test_without_gpu_status_is_blocked(tmp_path):
    """На CPU исход — ``blocked`` с причиной; «прошло» без GPU запрещено."""
    try:
        import jax

        has_gpu = any(d.platform == "gpu" for d in jax.devices())
    except Exception:  # noqa: BLE001 — нет jax вовсе: тем более blocked
        has_gpu = False

    if has_gpu:  # pragma: no cover — на исполнителе GPU нет (AD-7)
        pytest.skip("на этой машине есть GPU: проверка blocked неприменима")

    code, doc, _ = _run(tmp_path)
    assert doc["status"] == "blocked", doc
    assert doc["error_type"] in ("no-gpu", "ModuleNotFoundError", "ImportError"), doc
    assert doc["error"], "у blocked обязана быть причина"
    assert doc["max_abs"] is None and doc["rel"] is None
    assert code == 0, "отсутствие GPU — не ошибка скрипта (код 0)"


def test_cases_flag_and_dtype_reporting(tmp_path):
    """``--cases`` расширяет набор, ``--dtype`` фиксируется в отчёте по каждому кейсу."""
    code, doc, _ = _run(tmp_path, "--cases", "--dtype", "bfloat16")
    assert code in (0, 1)
    assert all(c["dtype"] in ("bfloat16", None) for c in doc["cases"])
