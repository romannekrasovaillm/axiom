"""Tests of ``tools/check_performance_roofline.py``.

The guard closes the fitness blind spot named by
``axiom_fitness_blind_spot_assessment``: no rule compared measured speed against
the declared KPI.  It must redden when a KPI-pinned run comes in below the pinned
threshold, stay neutral for runs that carry no pin, pass when the median clears
the bar, and fail closed (never silently pass) when the metrics are absent,
empty, or unreadable.  The median window is deterministic (≥10 last records).

The four mutants of the task are exercised individually, plus determinism of the
median and the window semantics.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1]
CASE_DIR = TOOLS_DIR.parent
for _p in (str(TOOLS_DIR), str(CASE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import check_performance_roofline as perf  # noqa: E402

TOOL = TOOLS_DIR / "check_performance_roofline.py"
DEFAULT_PINS = CASE_DIR / "evidence" / "kpi-pins.json"
RUN = "kda-wyut-delta"
BASELINE = 86.0
THRESHOLD = 800.0


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


def write_pins(path: Path, runs: list[dict[str, float]] | None = None) -> Path:
    pins = runs if runs is not None else [
        {"run": RUN, "kpi_tok_s_baseline": BASELINE, "threshold": THRESHOLD}
    ]
    path.write_text(json.dumps(pins), encoding="utf-8")
    return path


def write_metrics(path: Path, values: list[float]) -> Path:
    path.write_text(
        "".join(
            json.dumps({"step": step, "tok_s": value}) + "\n"
            for step, value in enumerate(values)
        ),
        encoding="utf-8",
    )
    return path


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        capture_output=True,
        text=True,
        cwd=str(CASE_DIR),
    )


# --------------------------------------------------------------------------- #
# Scenario (в): median clears the threshold → PASS
# --------------------------------------------------------------------------- #


def test_median_above_threshold_passes(tmp_path: Path) -> None:
    pins = write_pins(tmp_path / "pins.json")
    metrics = write_metrics(tmp_path / "metrics.jsonl", [900.0 + i for i in range(20)])
    code, report = perf.run_check(RUN, metrics, pins_path=pins)
    assert code == perf.EXIT_OK
    assert report["verdict"] == "ok"
    assert report["tok_s_median"] >= THRESHOLD
    assert report["shortfall_factor"] is None
    assert report["pin"]["threshold"] == THRESHOLD


def test_median_exactly_at_threshold_passes(tmp_path: Path) -> None:
    """The comparison is `>=`: the bar itself is a pass."""
    pins = write_pins(tmp_path / "pins.json")
    metrics = write_metrics(tmp_path / "metrics.jsonl", [THRESHOLD] * 12)
    code, report = perf.run_check(RUN, metrics, pins_path=pins)
    assert code == perf.EXIT_OK
    assert report["tok_s_median"] == THRESHOLD


# --------------------------------------------------------------------------- #
# Scenario (а): median below threshold → FAIL with the shortfall factor
# --------------------------------------------------------------------------- #


def test_below_threshold_fails(tmp_path: Path) -> None:
    pins = write_pins(tmp_path / "pins.json")
    metrics = write_metrics(tmp_path / "metrics.jsonl", [BASELINE] * 20)
    code, report = perf.run_check(RUN, metrics, pins_path=pins)
    assert code == perf.EXIT_FAIL
    assert report["verdict"] == "regression"
    assert report["tok_s_median"] == BASELINE
    assert report["shortfall_factor"] == pytest.approx(THRESHOLD / BASELINE, rel=1e-6)
    # The printed diagnostics name the pin, the fact and the shortfall.
    assert report["pin"]["run"] == RUN
    assert report["threshold"] == THRESHOLD


def test_zero_speed_is_a_regression_without_crashing(tmp_path: Path) -> None:
    pins = write_pins(tmp_path / "pins.json")
    metrics = write_metrics(tmp_path / "metrics.jsonl", [0.0] * 20)
    code, report = perf.run_check(RUN, metrics, pins_path=pins)
    assert code == perf.EXIT_FAIL
    assert report["verdict"] == "regression"
    assert report["tok_s_median"] == 0.0
    assert "регрессия" in report["message"]


def test_below_threshold_reports_speedup_against_baseline(tmp_path: Path) -> None:
    pins = write_pins(tmp_path / "pins.json")
    metrics = write_metrics(tmp_path / "metrics.jsonl", [86.0] * 20)
    _code, report = perf.run_check(RUN, metrics, pins_path=pins)
    assert report["speedup_vs_baseline"] == pytest.approx(1.0, rel=1e-9)


# --------------------------------------------------------------------------- #
# Scenario (б): run carries no pin → neutral PASS (metrics not even required)
# --------------------------------------------------------------------------- #


def test_unpinned_run_is_neutral_pass(tmp_path: Path) -> None:
    pins = write_pins(tmp_path / "pins.json")
    metrics = write_metrics(tmp_path / "metrics.jsonl", [BASELINE] * 20)
    code, report = perf.run_check("some-other-run", metrics, pins_path=pins)
    assert code == perf.EXIT_OK
    assert report["verdict"] == "neutral"
    assert report["pin"] is None


def test_unpinned_run_passes_even_without_metrics(tmp_path: Path) -> None:
    """A run outside the KPI contour must not be dragged down by its metrics."""
    pins = write_pins(tmp_path / "pins.json")
    code, report = perf.run_check("not-declared", tmp_path / "absent.jsonl", pins_path=pins)
    assert code == perf.EXIT_OK
    assert report["verdict"] == "neutral"


def test_pins_file_without_the_run_is_neutral(tmp_path: Path) -> None:
    pins = write_pins(
        tmp_path / "pins.json",
        [{"run": "another-run", "kpi_tok_s_baseline": 10.0, "threshold": 100.0}],
    )
    metrics = write_metrics(tmp_path / "metrics.jsonl", [1.0] * 20)
    code, report = perf.run_check(RUN, metrics, pins_path=pins)
    assert code == perf.EXIT_OK
    assert report["verdict"] == "neutral"


# --------------------------------------------------------------------------- #
# Scenario (г): empty metrics → fail-closed «нет данных»
# --------------------------------------------------------------------------- #


def test_empty_metrics_fails_closed(tmp_path: Path) -> None:
    pins = write_pins(tmp_path / "pins.json")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n\n", encoding="utf-8")
    code, report = perf.run_check(RUN, empty, pins_path=pins)
    assert code == perf.EXIT_FAIL
    assert report["verdict"] == "no-data"
    assert "нет данных" in report["message"]


def test_missing_metrics_file_fails_closed(tmp_path: Path) -> None:
    pins = write_pins(tmp_path / "pins.json")
    code, report = perf.run_check(RUN, tmp_path / "absent.jsonl", pins_path=pins)
    assert code == perf.EXIT_FAIL
    assert report["verdict"] == "no-data"


def test_no_metrics_argument_fails_closed(tmp_path: Path) -> None:
    pins = write_pins(tmp_path / "pins.json")
    code, report = perf.run_check(RUN, None, pins_path=pins)
    assert code == perf.EXIT_FAIL
    assert report["verdict"] == "no-data"


def test_garbage_metrics_line_fails_closed(tmp_path: Path) -> None:
    pins = write_pins(tmp_path / "pins.json")
    broken = tmp_path / "broken.jsonl"
    broken.write_text("не json\n", encoding="utf-8")
    code, report = perf.run_check(RUN, broken, pins_path=pins)
    assert code == perf.EXIT_FAIL
    assert report["verdict"] == "no-data"


def test_metrics_line_without_tok_s_fails_closed(tmp_path: Path) -> None:
    pins = write_pins(tmp_path / "pins.json")
    metrics = tmp_path / "missing-field.jsonl"
    metrics.write_text(json.dumps({"step": 0}) + "\n", encoding="utf-8")
    code, report = perf.run_check(RUN, metrics, pins_path=pins)
    assert code == perf.EXIT_FAIL
    assert report["verdict"] == "no-data"


def test_missing_pins_file_fails_closed(tmp_path: Path) -> None:
    """A deleted pins file must not disarm the guard silently."""
    metrics = write_metrics(tmp_path / "metrics.jsonl", [9000.0] * 20)
    code, report = perf.run_check(RUN, metrics, pins_path=tmp_path / "no-pins.json")
    assert code == perf.EXIT_FAIL
    assert report["verdict"] == "no-data"


def test_malformed_pin_fails_closed(tmp_path: Path) -> None:
    pins = tmp_path / "pins.json"
    pins.write_text(json.dumps([{"run": RUN}]), encoding="utf-8")
    metrics = write_metrics(tmp_path / "metrics.jsonl", [9000.0] * 20)
    code, report = perf.run_check(RUN, metrics, pins_path=pins)
    assert code == perf.EXIT_FAIL
    assert report["verdict"] == "no-data"


# --------------------------------------------------------------------------- #
# Determinism of the median
# --------------------------------------------------------------------------- #


def test_median_odd_count() -> None:
    assert perf.median([300.0, 100.0, 200.0]) == 200.0


def test_median_even_count_is_mean_of_central_pair() -> None:
    assert perf.median([400.0, 100.0, 300.0, 200.0]) == 250.0


def test_median_is_invariant_to_order(tmp_path: Path) -> None:
    pins = write_pins(tmp_path / "pins.json")
    plain = write_metrics(tmp_path / "plain.jsonl", [100.0, 200.0, 300.0, 400.0])
    shuffled = write_metrics(tmp_path / "shuffled.jsonl", [200.0, 400.0, 100.0, 300.0])
    m_plain = perf.run_check(RUN, plain, pins_path=pins)[1]["tok_s_median"]
    m_shuffled = perf.run_check(RUN, shuffled, pins_path=pins)[1]["tok_s_median"]
    assert m_plain == m_shuffled == 250.0


def test_repeated_runs_are_identical(tmp_path: Path) -> None:
    pins = write_pins(tmp_path / "pins.json")
    metrics = write_metrics(tmp_path / "metrics.jsonl", [850.0, 810.0, 900.0, 790.0])
    first = perf.run_check(RUN, metrics, pins_path=pins)
    second = perf.run_check(RUN, metrics, pins_path=pins)
    assert first == second


# --------------------------------------------------------------------------- #
# Window semantics: the last ≥10 records decide
# --------------------------------------------------------------------------- #


def test_late_regression_is_caught_by_the_last_window(tmp_path: Path) -> None:
    """A fast start followed by a collapse must fail — the tail is what counts."""
    pins = write_pins(tmp_path / "pins.json")
    metrics = write_metrics(tmp_path / "metrics.jsonl", [9000.0] * 30 + [86.0] * 30)
    code, report = perf.run_check(RUN, metrics, pins_path=pins)
    assert code == perf.EXIT_FAIL
    assert report["verdict"] == "regression"
    assert report["tok_s_median"] == 86.0


def test_late_recovery_passes(tmp_path: Path) -> None:
    pins = write_pins(tmp_path / "pins.json")
    metrics = write_metrics(tmp_path / "metrics.jsonl", [86.0] * 30 + [9000.0] * 30)
    code, report = perf.run_check(RUN, metrics, pins_path=pins)
    assert code == perf.EXIT_OK
    assert report["samples"] == perf.MIN_WINDOW
    assert report["records"] == 60


def test_window_is_floored_to_ten_records(tmp_path: Path) -> None:
    """A sub-10 window request is clamped: determinism needs ≥10 records."""
    pins = write_pins(tmp_path / "pins.json")
    metrics = write_metrics(tmp_path / "metrics.jsonl", [9000.0] * 5 + [86.0] * 15)
    _code, report = perf.run_check(RUN, metrics, pins_path=pins, window=3)
    assert report["window"] == perf.MIN_WINDOW
    assert report["samples"] == perf.MIN_WINDOW


def test_wider_window_is_honoured(tmp_path: Path) -> None:
    pins = write_pins(tmp_path / "pins.json")
    metrics = write_metrics(tmp_path / "metrics.jsonl", [9000.0] * 20 + [86.0] * 20)
    _code, report = perf.run_check(RUN, metrics, pins_path=pins, window=25)
    assert report["window"] == 25
    assert report["samples"] == 25


# --------------------------------------------------------------------------- #
# Shipped default pin (ADR-032)
# --------------------------------------------------------------------------- #


def test_shipped_default_pins_declare_adr032_kpi() -> None:
    assert DEFAULT_PINS.exists(), f"нет дефолтного пин-файла: {DEFAULT_PINS}"
    pins = perf.load_pins(DEFAULT_PINS)
    pin = next((p for p in pins if p["run"] == "kda-wyut-delta"), None)
    assert pin is not None
    assert pin["kpi_tok_s_baseline"] == 86.0
    assert pin["threshold"] == 800.0


def test_default_pins_path_is_repo_anchored() -> None:
    """The default must resolve to the repo's evidence/ regardless of cwd."""
    assert perf.DEFAULT_PINS == CASE_DIR / "evidence" / "kpi-pins.json"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_cli_selftest_is_green() -> None:
    result = run_cli("--selftest")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FAIL" not in result.stdout
    assert "PASS" in result.stdout


def test_cli_regression_exits_one_and_prints_shortfall(tmp_path: Path) -> None:
    pins = write_pins(tmp_path / "pins.json")
    metrics = write_metrics(tmp_path / "metrics.jsonl", [BASELINE] * 20)
    result = run_cli(
        "--run", RUN, "--metrics", str(metrics), "--pins", str(pins)
    )
    assert result.returncode == 1
    assert "FAIL" in result.stderr
    assert "регрессия" in result.stderr
    assert "недобор" in result.stderr


def test_cli_writes_report(tmp_path: Path) -> None:
    pins = write_pins(tmp_path / "pins.json")
    metrics = write_metrics(tmp_path / "metrics.jsonl", [9000.0] * 20)
    report_path = tmp_path / "roofline.json"
    result = run_cli(
        "--run", RUN, "--metrics", str(metrics), "--pins", str(pins),
        "--json", str(report_path), "--quiet",
    )
    assert result.returncode == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["verdict"] == "ok"
    assert report["exit_code"] == 0
    assert report["schema"] == perf.REPORT_SCHEMA


def test_cli_requires_run_outside_selftest() -> None:
    result = run_cli("--metrics", "whatever.jsonl")
    assert result.returncode != 0
