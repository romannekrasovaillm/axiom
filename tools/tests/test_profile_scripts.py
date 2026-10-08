"""MFU стадия 2 — профилировочные приборы (``tools/profile_mfu.py``,
``tools/profile_mfu_nsys.sh``).

Прогоны приборов — на стенде архитектора (GPU занят parity-ногой), поэтому в
отчётах статус ``EMPTY-PENDING``, а механически проверяется здесь то, что можно
проверить без GPU и без jax:

* **арифметика разбора traceEvents** — на мок-фикстуре с детерминированными
  длительностями ожидаемые доли и gap считаются точно;
* **fail-closed** — пустой/битый трейс даёт ``ProfileError`` и статус
  ``TRACE-ERROR`` без выдуманных чисел (никаких ``top_ops``, ``gap``/``op_total``
  при отсутствии данных);
* **план клеток** — те же три формы кампании, что в стадии 1;
* **nsys-обвязка** — dry-run печатает команды ``nsys profile --stats=true`` и
  ``nsys stats`` по обеим сводкам на каждую клетку; отсутствие nsys — внятная
  ошибка (exit 3), а не тихий проход.

Модуль импортируется без jax (на этой машине его нет): профильные импорты
``net/*``/``jax`` живут внутри ``run_cell``, а разбор трейса — чистые функции.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1]
CASE_DIR = TOOLS_DIR.parent
for _p in (str(TOOLS_DIR), str(CASE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import profile_mfu as prof  # noqa: E402

NSYS_SCRIPT = TOOLS_DIR / "profile_mfu_nsys.sh"
PROFILE_SCRIPT = TOOLS_DIR / "profile_mfu.py"


def run_cli(script: Path, *args: str, env: dict | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        cwd=str(CASE_DIR),
        env=env,
    )


def run_sh(*args: str, env: dict | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(NSYS_SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=str(CASE_DIR),
        env=env,
    )


#: Детерминированная мок-фикстура трейса (микросекунды).
#: einsum 2×10, reduce 5, copy 15 → op_total 40; прочий регион 100 → окно 100,
#: покрытие 40, gap 60.  Разбор не зависит от jax — это чистый traceEvents.
MOCK_EVENTS = [
    {"ph": "X", "name": "einsum.1", "cat": "XLA Ops", "ts": 0, "dur": 10, "pid": 1, "tid": 1},
    {"ph": "X", "name": "einsum.1", "cat": "XLA Ops", "ts": 10, "dur": 10, "pid": 1, "tid": 1},
    {"ph": "X", "name": "reduce.2", "cat": "XLA Ops", "ts": 20, "dur": 5, "pid": 1, "tid": 1},
    {"ph": "X", "name": "copy.3", "cat": "XLA Ops", "ts": 25, "dur": 15, "pid": 1, "tid": 1},
    {"ph": "X", "name": "jit-step", "cat": "user_annotation", "ts": 0, "dur": 100, "pid": 1, "tid": 0},
    {"ph": "B", "name": "einsum.1", "ts": 0, "pid": 1, "tid": 1},
]


def write_trace(dirpath: Path, events, *, filename: str = "trace.json") -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / filename).write_text(
        json.dumps({"traceEvents": events, "displayTimeUnit": "us"}),
        encoding="utf-8",
    )
    return dirpath


# --------------------------------------------------------------------------- #
# Разбор traceEvents — арифметика на мок-фикстуре
# --------------------------------------------------------------------------- #


def test_selftest_is_green() -> None:
    result = run_cli(PROFILE_SCRIPT, "--selftest")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FAIL" not in result.stdout
    assert "PASS" in result.stdout


def test_aggregation_is_deterministic_and_durations_are_exact(tmp_path: Path) -> None:
    trace = write_trace(tmp_path / "trace", MOCK_EVENTS)
    analysis = prof.analyze_trace(trace)

    # Только X-события (6 → 5): B-событие без dur отброшено.
    assert analysis["trace"]["complete_event_count"] == 5
    assert abs(analysis["op_total_seconds"] - 40e-6) < 1e-12

    names = [op["name"] for op in analysis["top_ops"]]
    assert names == ["einsum.1", "copy.3", "reduce.2"]  # отсортировано по времени
    by_name = {op["name"]: op for op in analysis["top_ops"]}
    assert abs(by_name["einsum.1"]["share"] - 0.5) < 1e-12
    assert abs(by_name["copy.3"]["share"] - 0.375) < 1e-12
    assert abs(by_name["reduce.2"]["share"] - 0.125) < 1e-12
    assert by_name["einsum.1"]["count"] == 2
    assert abs(by_name["einsum.1"]["mean_seconds"] - 10e-6) < 1e-12
    # Доли top-N суммируются к 1 (share считается от op_total).
    assert abs(sum(op["share"] for op in analysis["top_ops"]) - 1.0) < 1e-12

    # Прочий регион (jit-step) — не операция: ушёл в unclassified, не в op_total.
    assert analysis["unclassified_count"] == 1
    assert abs(analysis["unclassified_seconds"] - 100e-6) < 1e-12


def test_gap_is_window_minus_operation_union(tmp_path: Path) -> None:
    trace = write_trace(tmp_path / "trace", MOCK_EVENTS)
    gap = prof.analyze_trace(trace)["gap"]
    assert abs(gap["window_seconds"] - 100e-6) < 1e-12
    assert abs(gap["covered_seconds"] - 40e-6) < 1e-12
    assert abs(gap["gap_seconds"] - 60e-6) < 1e-12
    assert abs(gap["gap_share"] - 0.6) < 1e-12


def test_gap_union_does_not_double_count_nesting() -> None:
    nested = prof.complete_events(
        [
            {"ph": "X", "name": "einsum.1", "ts": 0, "dur": 100},
            {"ph": "X", "name": "reduce.2", "ts": 10, "dur": 20},  # внутри einsum
        ]
    )
    assert abs(prof.compute_gap(nested)["covered_seconds"] - 100e-6) < 1e-12


def test_classification_precedence_collective_before_reduce() -> None:
    assert prof.classify_op("all-reduce.1") == "collective"
    assert prof.classify_op("reduce-scatter.0") == "collective"
    assert prof.classify_op("reduce.3") == "reduce"
    assert prof.classify_op("einsum.42") == "einsum"
    assert prof.classify_op("jit-step") == "other"
    assert prof.is_op("jit-step") is False


def test_trace_source_with_most_events_wins(tmp_path: Path) -> None:
    """JAX кладёт копию трейса рядом: берём один источник, а не суммируем."""
    trace = write_trace(tmp_path / "trace", MOCK_EVENTS[:1], filename="a.json")
    write_trace(trace, MOCK_EVENTS[:3], filename="b.json")
    events, meta = prof.load_trace_events(trace)
    assert len(events) == 3
    assert meta["file_used"].endswith("b.json")
    assert len(meta["duplicate_trace_files"]) == 1


# --------------------------------------------------------------------------- #
# Fail-closed: нет данных — нет чисел
# --------------------------------------------------------------------------- #


def test_empty_trace_dir_raises_no_numbers(tmp_path: Path) -> None:
    empty = tmp_path / "trace"
    empty.mkdir()
    (empty / "summary.json").write_text(json.dumps({"note": "no events"}), encoding="utf-8")
    with pytest.raises(prof.ProfileError):
        prof.load_trace_events(empty)
    with pytest.raises(prof.ProfileError):
        prof.analyze_trace(empty)


def test_broken_trace_raises(tmp_path: Path) -> None:
    broken = tmp_path / "trace"
    broken.mkdir()
    (broken / "trace.json").write_text("{ это не json", encoding="utf-8")
    with pytest.raises(prof.ProfileError):
        prof.analyze_trace(broken)


def test_missing_trace_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(prof.ProfileError):
        prof.load_trace_events(tmp_path / "absent")


def test_cli_parse_broken_trace_is_fail_closed(tmp_path: Path) -> None:
    broken = tmp_path / "trace"
    broken.mkdir()
    (broken / "trace.json").write_text("{ битый", encoding="utf-8")
    out = tmp_path / "report.json"
    result = run_cli(
        PROFILE_SCRIPT, "--parse-trace", str(broken), "--name", "t", "--out", str(out)
    )
    assert result.returncode == 2, result.stdout + result.stderr
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["status"] == "TRACE-ERROR"
    assert report["top_ops"] == [] and report["gap"] is None
    assert "op_total_seconds" not in report  # чисел профиля нет вовсе
    assert report["error"]


def test_cli_parse_complete_trace_writes_numbers(tmp_path: Path) -> None:
    trace = write_trace(tmp_path / "trace", MOCK_EVENTS)
    out = tmp_path / "report.json"
    result = run_cli(
        PROFILE_SCRIPT, "--parse-trace", str(trace), "--name", "t", "--out", str(out)
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "einsum.1" in result.stdout  # таблица напечатана
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["status"] == "COMPLETE"
    assert report["schema"] == prof.REPORT_SCHEMA
    assert report["version"] == prof.REPORT_VERSION
    assert [op["name"] for op in report["top_ops"]] == ["einsum.1", "copy.3", "reduce.2"]
    assert abs(report["gap"]["gap_share"] - 0.6) < 1e-12
    assert any("эвристика" in c for c in report["caveats"])


# --------------------------------------------------------------------------- #
# План клеток и EMPTY-PENDING без GPU
# --------------------------------------------------------------------------- #


def test_plan_is_the_three_campaign_cells() -> None:
    cells = prof.plan()
    assert {(c["name"], c["config"], c["batch"], c["seq"]) for c in cells} == {
        ("dense124m-b1", "net/config-dense124m.json", 1, 8192),
        ("dense124m-b4", "net/config-dense124m.json", 4, 8192),
        ("l3full-b1", "net/config.json", 1, 8192),
    }


def test_plan_cli_is_tsv_for_bash() -> None:
    result = run_cli(PROFILE_SCRIPT, "--plan")
    assert result.returncode == 0
    rows = [line.split("\t") for line in result.stdout.strip().splitlines()]
    assert len(rows) == 3
    assert all(len(row) == 4 for row in rows)


def test_cli_without_gpu_is_pending_without_numbers(tmp_path: Path) -> None:
    if prof.gpu_available():
        pytest.skip("на машине есть GPU — контракт EMPTY-PENDING проверяется без него")
    out = tmp_path / "profile.json"
    result = run_cli(PROFILE_SCRIPT, "--name", "empty", "--out", str(out))
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["status"] == "EMPTY-PENDING"
    assert report["gap"] is None and report["top_ops"] == []
    assert report["steps"] == prof.DEFAULT_STEPS
    assert report["warmup"] == prof.DEFAULT_WARMUP
    assert "op_total_seconds" not in report


def test_module_imports_without_jax() -> None:
    """Прибор импортируется без jax: профильные импорты — внутри run_cell."""
    probe = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, %r); import profile_mfu; "
         "assert 'jax' not in sys.modules, 'импорт прибора потянул jax'; print('ok')"
         % str(TOOLS_DIR)],
        capture_output=True, text=True, cwd=str(CASE_DIR),
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr
    assert probe.stdout.strip() == "ok"


# --------------------------------------------------------------------------- #
# nsys-обвязка: план (dry-run) и внятная ошибка без nsys
# --------------------------------------------------------------------------- #


def test_nsys_dry_run_all_lists_every_cell_with_both_summaries() -> None:
    result = run_sh("--dry-run", "--all")
    assert result.returncode == 0, result.stdout + result.stderr
    out = result.stdout

    # Каждая клетка плана — своим каталогом nsys-<name>.
    for cell in prof.plan():
        assert f"nsys-{cell['name']}" in out
        assert f"--name {cell['name']}" in out
        assert f"--config {cell['config']}" in out
    assert out.count("nsys profile") == len(prof.plan())
    assert "--stats=true" in out
    # Обе текстовые сводки запрашиваются явно.
    assert "--report cuda_gpu_kern_sum" in out
    assert "--report cuda_api_sum" in out
    assert out.count(".nsys-rep") >= len(prof.plan())


def test_nsys_dry_run_honours_the_flags() -> None:
    result = run_sh(
        "--dry-run", "--name", "custom", "--config", "net/config.json",
        "--batch", "2", "--seq", "4096", "--steps", "3", "--warmup", "1",
        "--mode", "bf16",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    out = result.stdout
    assert "--name custom" in out
    assert "--batch 2" in out and "--seq 4096" in out
    assert "--steps 3" in out and "--warmup 1" in out
    assert "--mode bf16" in out
    assert "--no-trace" in out  # под nsys jax-трейс выключен


def test_nsys_missing_is_a_clear_error() -> None:
    env = dict(os.environ)
    env["NSYS"] = "/nonexistent/nsys"
    result = run_sh("--name", "x", env=env)
    assert result.returncode == 3
    combined = result.stdout + result.stderr
    assert "nsys не найден" in combined
    # Никакого тихого «успеха» без профиля.
    assert not (CASE_DIR / "evidence" / "mfu-profile" / "nsys-x" / "x.nsys-rep").exists()


def test_nsys_script_is_executable() -> None:
    assert os.access(NSYS_SCRIPT, os.X_OK)
    assert os.access(PROFILE_SCRIPT, os.X_OK)
