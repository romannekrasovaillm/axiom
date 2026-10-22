"""Tests of ``tools/check_rl_health.py`` (RL-STAGE.delta §8.1, C-045).

The guard must redden on a degenerate reward row (each stop class
independently), stay green on a healthy row, treat warnings as warnings rather
than stops, and fail closed on unreadable/absent metrics.  Both evaluation
points (first 50 steps and the current window) are exercised.
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

import check_rl_health as rl  # noqa: E402

TOOL = TOOLS_DIR / "check_rl_health.py"


def healthy_row(step: int, **overrides: float) -> dict[str, float]:
    row = {
        "step": step,
        "pass_rate": 0.39,
        "reward_mean": 0.38,
        "zero_reward_share": 0.51,
        "adv_nonzero_share": 0.90,
        "entropy_mean": 1.70,
        "ngram_repeat_max": 3.0,
        "clip_frac": 0.04,
    }
    row.update(overrides)
    return row


def write_rows(path: Path, rows: list[dict[str, float]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        capture_output=True,
        text=True,
        cwd=str(CASE_DIR),
    )


# --------------------------------------------------------------------------- #
# Healthy / warnings
# --------------------------------------------------------------------------- #


def test_healthy_row_is_green(tmp_path: Path) -> None:
    path = tmp_path / "healthy.jsonl"
    write_rows(path, [healthy_row(i) for i in range(60)])
    code, report = rl.run_check([path])
    assert code == rl.EXIT_OK
    assert report["verdict"] == "ok"
    assert report["stops"] == []
    assert report["warnings"] == []
    assert report["steps"] == 60


def test_warn_only_row_is_green_with_warnings(tmp_path: Path) -> None:
    path = tmp_path / "warn.jsonl"
    write_rows(path, [healthy_row(i, ngram_repeat_max=9.0, clip_frac=0.25) for i in range(60)])
    code, report = rl.run_check([path])
    assert code == rl.EXIT_OK
    assert report["stops"] == []
    assert set(report["warnings"]) == {"template_collapse", "clip_frac"}


# --------------------------------------------------------------------------- #
# Stop classes, each independently
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("name", "overrides"),
    [
        ("zero_pass_zero_reward", {"pass_rate": 0.0, "reward_mean": 0.02}),
        ("zero_reward_share", {"zero_reward_share": 0.95}),
        ("no_group_variance", {"adv_nonzero_share": 0.0}),
    ],
)
def test_stop_class_reddens(tmp_path: Path, name: str, overrides: dict[str, float]) -> None:
    path = tmp_path / f"{name}.jsonl"
    write_rows(path, [healthy_row(i, **overrides) for i in range(60)])
    code, report = rl.run_check([path])
    assert code == rl.EXIT_DEGENERATE
    assert name in report["stops"]


def test_entropy_collapse_two_consecutive_steps(tmp_path: Path) -> None:
    rows = [healthy_row(i) for i in range(60)]
    rows[10]["entropy_mean"] = 0.30
    rows[11]["entropy_mean"] = 0.30
    path = tmp_path / "entropy.jsonl"
    write_rows(path, rows)
    code, report = rl.run_check([path])
    assert code == rl.EXIT_DEGENERATE
    assert "entropy_collapse" in report["stops"]


def test_single_entropy_dip_is_not_a_stop(tmp_path: Path) -> None:
    rows = [healthy_row(i) for i in range(60)]
    rows[10]["entropy_mean"] = 0.30
    path = tmp_path / "entropy_single.jsonl"
    write_rows(path, rows)
    code, report = rl.run_check([path])
    assert code == rl.EXIT_OK
    assert "entropy_collapse" not in report["stops"]


# --------------------------------------------------------------------------- #
# Windows
# --------------------------------------------------------------------------- #


def test_both_evaluation_windows_are_reported(tmp_path: Path) -> None:
    path = tmp_path / "healthy.jsonl"
    write_rows(path, [healthy_row(i) for i in range(60)])
    _code, report = rl.run_check([path])
    labels = [window["window"] for window in report["windows"]]
    assert labels == ["first_50", "current"]
    assert report["windows"][0]["steps"] == 50


def test_degeneracy_only_late_is_caught_by_current_window(tmp_path: Path) -> None:
    """A healthy first 50 that collapses for the whole current window must stop."""
    rows = [healthy_row(i) for i in range(50)]
    rows += [healthy_row(i, pass_rate=0.0, reward_mean=0.01) for i in range(50, 100)]
    path = tmp_path / "late.jsonl"
    write_rows(path, rows)
    code, report = rl.run_check([path])
    assert code == rl.EXIT_DEGENERATE
    assert "zero_pass_zero_reward" in report["stops"]
    # The collapse is invisible in the first window — the current window is what
    # catches it ("живая строка лога ≠ свод окна").
    assert report["windows"][0]["stops"]["zero_pass_zero_reward"] is False
    assert report["windows"][1]["stops"]["zero_pass_zero_reward"] is True


# --------------------------------------------------------------------------- #
# Share normalization
# --------------------------------------------------------------------------- #


def test_percent_shares_are_accepted(tmp_path: Path) -> None:
    path = tmp_path / "percent.jsonl"
    write_rows(path, [healthy_row(i, zero_reward_share=51.0) for i in range(60)])
    assert rl.run_check([path])[0] == rl.EXIT_OK


def test_percent_above_threshold_still_stops(tmp_path: Path) -> None:
    path = tmp_path / "percent_bad.jsonl"
    write_rows(path, [healthy_row(i, zero_reward_share=91.0) for i in range(60)])
    code, report = rl.run_check([path])
    assert code == rl.EXIT_DEGENERATE
    assert "zero_reward_share" in report["stops"]


# --------------------------------------------------------------------------- #
# Fail-closed edges
# --------------------------------------------------------------------------- #


def test_no_input_is_cannot_check() -> None:
    assert rl.run_check([])[0] == rl.EXIT_CANNOT


def test_missing_path_is_cannot_check(tmp_path: Path) -> None:
    assert rl.run_check([tmp_path / "нет.jsonl"])[0] == rl.EXIT_CANNOT


def test_garbage_input_is_cannot_check(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text("не json\n", encoding="utf-8")
    assert rl.run_check([path])[0] == rl.EXIT_CANNOT


def test_empty_input_is_cannot_check(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("\n", encoding="utf-8")
    assert rl.run_check([path])[0] == rl.EXIT_CANNOT


def test_missing_field_is_cannot_check(tmp_path: Path) -> None:
    path = tmp_path / "missing.jsonl"
    write_rows(path, [{"step": 0, "pass_rate": 0.4}])
    assert rl.run_check([path])[0] == rl.EXIT_CANNOT


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_cli_selftest_is_green() -> None:
    result = run_cli("--selftest")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FAIL" not in result.stdout
    assert "PASS" in result.stdout


def test_cli_writes_report(tmp_path: Path) -> None:
    path = tmp_path / "healthy.jsonl"
    write_rows(path, [healthy_row(i) for i in range(60)])
    report_path = tmp_path / "health.json"
    result = run_cli("--input", str(path), "--json", str(report_path), "--quiet")
    assert result.returncode == 0
    report = json.loads(report_path.read_text())
    assert report["verdict"] == "ok"
    assert report["exit_code"] == 0
