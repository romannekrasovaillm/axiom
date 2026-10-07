"""Tests of ``tools/check_rental_budget_gate.py`` (guard C-048).

The guard answers one institutional question: may an *rental budget* artifact be
approved while a performance pin that declares ``blocks_rental`` has not been
shown to clear its threshold?  It must redden when a blocking pin has no metrics
file, an empty/unreadable metrics file, or a median below the threshold — and
stay green only when every blocking pin has demonstrably cleared its bar, or no
pin declares ``blocks_rental`` at all.

Deliberate difference from C-046 (``check_performance_roofline``): there an
*absent* metrics file is neutral (the run has not happened yet).  Here absence is
a blocker — a budget cannot be unlocked by unproven efficiency.  The mutant
classes (а)–(е) of the task are exercised individually, plus determinism, the
metrics-path inference convention, and the CLI boundary cases.
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

import check_rental_budget_gate as gate  # noqa: E402
import check_performance_roofline as roofline  # noqa: E402

TOOL = TOOLS_DIR / "check_rental_budget_gate.py"
DEFAULT_PINS = CASE_DIR / "evidence" / "kpi-pins.json"
RUN = "kda-wyut-delta"
BASELINE = 86.0
THRESHOLD = 800.0


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


def write_pins(path: Path, entries: list[dict]) -> Path:
    path.write_text(json.dumps({"pins": entries}, ensure_ascii=False), encoding="utf-8")
    return path


def blocking_pin(**overrides) -> dict:
    pin = {
        "run": RUN,
        "kpi_tok_s_baseline": BASELINE,
        "threshold": THRESHOLD,
        "blocks_rental": True,
    }
    pin.update(overrides)
    return pin


def write_metrics(path: Path, values: list[float]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps({"step": step, "tok_s": value}) + "\n"
            for step, value in enumerate(values)
        ),
        encoding="utf-8",
    )
    return path


def make_budget(path: Path) -> Path:
    path.write_text('{"run_ref": "l3", "usd_estimate": 0}', encoding="utf-8")
    return path


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        capture_output=True,
        text=True,
        cwd=str(CASE_DIR),
    )


# --------------------------------------------------------------------------- #
# Scenario (а): unreached blocker + budget present → FAIL naming the blockers
# --------------------------------------------------------------------------- #


def test_absent_metrics_with_budget_blocks(tmp_path: Path) -> None:
    """A blocking pin with no metrics file is *not* excused — the budget is held."""
    pins = write_pins(tmp_path / "pins.json", [blocking_pin()])
    budget = make_budget(tmp_path / "rental-l3.json")
    code, report = gate.run_check(
        pins_path=pins, rental_budget=budget, repo_root=tmp_path
    )
    assert code == gate.EXIT_FAIL
    assert report["verdict"] == "blocked"
    assert [b["run"] for b in report["blockers"]] == [RUN]
    assert report["blockers"][0]["reason"] == gate.REASON_NO_METRICS_FILE
    assert report["blockers"][0]["tok_s_median"] is None
    assert "арендная смета заблокирована" in report["message"]
    assert RUN in report["message"]


def test_median_below_threshold_with_budget_blocks(tmp_path: Path) -> None:
    pins = write_pins(tmp_path / "pins.json", [blocking_pin()])
    budget = make_budget(tmp_path / "rental-l3.json")
    write_metrics(tmp_path / "evidence" / "kda-wyut" / "metrics.jsonl", [BASELINE] * 20)
    code, report = gate.run_check(
        pins_path=pins, rental_budget=budget, repo_root=tmp_path
    )
    assert code == gate.EXIT_FAIL
    assert report["verdict"] == "blocked"
    blocker = report["blockers"][0]
    assert blocker["reason"] == gate.REASON_BELOW_THRESHOLD
    assert blocker["tok_s_median"] == BASELINE
    assert blocker["shortfall_factor"] == pytest.approx(THRESHOLD / BASELINE, rel=1e-6)


def test_blockers_are_listed_in_pin_order(tmp_path: Path) -> None:
    pins = write_pins(tmp_path / "pins.json", [
        blocking_pin(run="run-a"),
        blocking_pin(run="run-b"),
    ])
    budget = make_budget(tmp_path / "rental.json")
    _code, report = gate.run_check(
        pins_path=pins, rental_budget=budget, repo_root=tmp_path
    )
    assert [b["run"] for b in report["blockers"]] == ["run-a", "run-b"]
    assert report["message"].endswith("run-a, run-b")


def test_blocked_names_every_blocker_in_the_message(tmp_path: Path) -> None:
    pins = write_pins(tmp_path / "pins.json", [
        blocking_pin(run="run-a"),
        blocking_pin(run="run-b"),
    ])
    budget = make_budget(tmp_path / "rental.json")
    code, report = gate.run_check(
        pins_path=pins, rental_budget=budget, repo_root=tmp_path
    )
    assert code == gate.EXIT_FAIL
    assert "run-a" in report["message"] and "run-b" in report["message"]


# --------------------------------------------------------------------------- #
# Scenario (б): the blocker is lifted (mock metrics clear the bar) → PASS
# --------------------------------------------------------------------------- #


def test_cleared_blocker_passes(tmp_path: Path) -> None:
    pins = write_pins(tmp_path / "pins.json", [blocking_pin()])
    budget = make_budget(tmp_path / "rental-l3.json")
    write_metrics(
        tmp_path / "evidence" / "kda-wyut" / "metrics.jsonl",
        [900.0 + i for i in range(20)],
    )
    code, report = gate.run_check(
        pins_path=pins, rental_budget=budget, repo_root=tmp_path
    )
    assert code == gate.EXIT_OK
    assert report["verdict"] == "pass"
    assert report["blockers"] == []
    assert report["blocking_pins"] == 1


def test_median_exactly_at_threshold_passes(tmp_path: Path) -> None:
    """The comparison is `>=`: the bar itself unlocks the budget."""
    pins = write_pins(tmp_path / "pins.json", [blocking_pin()])
    budget = make_budget(tmp_path / "rental.json")
    write_metrics(tmp_path / "evidence" / "kda-wyut" / "metrics.jsonl", [THRESHOLD] * 12)
    code, _report = gate.run_check(
        pins_path=pins, rental_budget=budget, repo_root=tmp_path
    )
    assert code == gate.EXIT_OK


def test_late_regression_is_caught_by_the_last_window(tmp_path: Path) -> None:
    """A fast start followed by a collapse must still hold the budget."""
    pins = write_pins(tmp_path / "pins.json", [blocking_pin()])
    budget = make_budget(tmp_path / "rental.json")
    write_metrics(
        tmp_path / "evidence" / "kda-wyut" / "metrics.jsonl",
        [9000.0] * 30 + [86.0] * 30,
    )
    code, report = gate.run_check(
        pins_path=pins, rental_budget=budget, repo_root=tmp_path
    )
    assert code == gate.EXIT_FAIL
    assert report["blockers"][0]["tok_s_median"] == 86.0


# --------------------------------------------------------------------------- #
# Scenario (в): no budget given → status report, exit 0
# --------------------------------------------------------------------------- #


def test_no_budget_is_status_exit_zero(tmp_path: Path) -> None:
    pins = write_pins(tmp_path / "pins.json", [blocking_pin()])
    code, report = gate.run_check(pins_path=pins, repo_root=tmp_path)
    assert code == gate.EXIT_OK
    assert report["verdict"] == "status"
    assert report["rental_budget"] is None
    assert [b["run"] for b in report["blockers"]] == [RUN]


def test_no_budget_still_lists_pending_blockers(tmp_path: Path) -> None:
    """Without a budget there is nothing to block, but the status must be visible."""
    pins = write_pins(tmp_path / "pins.json", [blocking_pin()])
    _code, report = gate.run_check(pins_path=pins, repo_root=tmp_path)
    assert report["blocking_pins"] == 1
    assert report["blockers"][0]["reason"] == gate.REASON_NO_METRICS_FILE


def test_cli_no_budget_prints_status_and_exits_zero(tmp_path: Path) -> None:
    pins = write_pins(tmp_path / "pins.json", [blocking_pin()])
    result = run_cli("--pins", str(pins))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FAIL" not in result.stderr
    report = json.loads(result.stdout)
    assert report["verdict"] == "status"
    assert report["blockers"][0]["run"] == RUN


# --------------------------------------------------------------------------- #
# Scenario (г): no pin declares blocks_rental → nothing to block → PASS
# --------------------------------------------------------------------------- #


def test_no_blocking_pins_passes(tmp_path: Path) -> None:
    pins = write_pins(tmp_path / "pins.json", [
        {"run": RUN, "kpi_tok_s_baseline": BASELINE, "threshold": THRESHOLD},
    ])
    budget = make_budget(tmp_path / "rental.json")
    code, report = gate.run_check(
        pins_path=pins, rental_budget=budget, repo_root=tmp_path
    )
    assert code == gate.EXIT_OK
    assert report["verdict"] == "pass"
    assert report["blocking_pins"] == 0
    assert report["blockers"] == []


def test_blocks_rental_false_does_not_block(tmp_path: Path) -> None:
    pins = write_pins(tmp_path / "pins.json", [blocking_pin(blocks_rental=False)])
    budget = make_budget(tmp_path / "rental.json")
    code, report = gate.run_check(
        pins_path=pins, rental_budget=budget, repo_root=tmp_path
    )
    assert code == gate.EXIT_OK
    assert report["verdict"] == "pass"


def test_no_blocking_pins_without_budget_is_status(tmp_path: Path) -> None:
    pins = write_pins(tmp_path / "pins.json", [
        {"run": RUN, "threshold": THRESHOLD},
    ])
    code, report = gate.run_check(pins_path=pins, repo_root=tmp_path)
    assert code == gate.EXIT_OK
    assert report["verdict"] == "status"
    assert report["blockers"] == []


# --------------------------------------------------------------------------- #
# Scenario (д): determinism of the median
# --------------------------------------------------------------------------- #


def test_median_odd_count() -> None:
    assert roofline.median([300.0, 100.0, 200.0]) == 200.0


def test_median_even_count_is_mean_of_central_pair() -> None:
    assert roofline.median([400.0, 100.0, 300.0, 200.0]) == 250.0


def test_median_is_invariant_to_order(tmp_path: Path) -> None:
    pins = write_pins(tmp_path / "pins.json", [blocking_pin()])
    plain = write_metrics(
        tmp_path / "evidence" / "kda-wyut" / "metrics.jsonl",
        [100.0, 200.0, 300.0, 400.0],
    )
    first = gate.run_check(pins_path=pins, repo_root=tmp_path)
    shuffled = tmp_path / "shuffled.jsonl"
    write_metrics(shuffled, [200.0, 400.0, 100.0, 300.0])
    second = gate.run_check(
        pins_path=write_pins(tmp_path / "shuffle-pins.json",
                             [blocking_pin(metrics_path=str(shuffled))]),
        repo_root=tmp_path,
    )
    assert plain.exists()
    assert first[1]["blockers"][0]["tok_s_median"] == second[1]["blockers"][0]["tok_s_median"]


def test_repeated_runs_are_identical(tmp_path: Path) -> None:
    pins = write_pins(tmp_path / "pins.json", [blocking_pin()])
    budget = make_budget(tmp_path / "rental.json")
    write_metrics(tmp_path / "evidence" / "kda-wyut" / "metrics.jsonl", [850.0, 810.0])
    first = gate.run_check(pins_path=pins, rental_budget=budget, repo_root=tmp_path)
    second = gate.run_check(pins_path=pins, rental_budget=budget, repo_root=tmp_path)
    assert first == second


def test_window_is_floored_to_ten_records(tmp_path: Path) -> None:
    pins = write_pins(tmp_path / "pins.json", [blocking_pin()])
    budget = make_budget(tmp_path / "rental.json")
    write_metrics(
        tmp_path / "evidence" / "kda-wyut" / "metrics.jsonl",
        [9000.0] * 5 + [86.0] * 15,
    )
    code, report = gate.run_check(
        pins_path=pins, rental_budget=budget, repo_root=tmp_path, window=3
    )
    # A sub-10 window is clamped: the last 10 records (all slow) still block.
    assert code == gate.EXIT_FAIL
    assert report["blockers"][0]["samples"] == roofline.MIN_WINDOW


# --------------------------------------------------------------------------- #
# Scenario (е): a present-but-empty metrics file is unreached (fail-closed)
# --------------------------------------------------------------------------- #


def test_empty_metrics_is_a_blocker(tmp_path: Path) -> None:
    pins = write_pins(tmp_path / "pins.json", [blocking_pin()])
    budget = make_budget(tmp_path / "rental.json")
    empty = tmp_path / "evidence" / "kda-wyut" / "metrics.jsonl"
    empty.parent.mkdir(parents=True, exist_ok=True)
    empty.write_text("\n", encoding="utf-8")
    code, report = gate.run_check(
        pins_path=pins, rental_budget=budget, repo_root=tmp_path
    )
    assert code == gate.EXIT_FAIL
    assert report["verdict"] == "blocked"
    assert "fail-closed" in report["blockers"][0]["reason"]


def test_garbage_metrics_line_is_a_blocker(tmp_path: Path) -> None:
    pins = write_pins(tmp_path / "pins.json", [blocking_pin()])
    broken = tmp_path / "evidence" / "kda-wyut" / "metrics.jsonl"
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_text("не json\n", encoding="utf-8")
    _code, report = gate.run_check(
        pins_path=pins, rental_budget=make_budget(tmp_path / "b.json"), repo_root=tmp_path
    )
    assert "fail-closed" in report["blockers"][0]["reason"]


def test_absent_metrics_blocks_where_roofline_is_neutral(tmp_path: Path) -> None:
    """The two guards intentionally disagree on an absent metrics file.

    C-046 is neutral (the run has not started); C-048 blocks (efficiency unproven).
    """
    pins = write_pins(tmp_path / "pins.json", [blocking_pin()])
    perf_code, perf_report = roofline.run_check(
        RUN, tmp_path / "evidence" / "kda-wyut" / "metrics.jsonl", pins_path=pins
    )
    gate_code, _gate_report = gate.run_check(
        pins_path=pins, rental_budget=make_budget(tmp_path / "b.json"), repo_root=tmp_path
    )
    assert perf_code == roofline.EXIT_OK and perf_report["verdict"] == "neutral"
    assert gate_code == gate.EXIT_FAIL


# --------------------------------------------------------------------------- #
# Metrics-path resolution: explicit pin field vs. inference convention
# --------------------------------------------------------------------------- #


def test_infers_metrics_path_from_delta_run(tmp_path: Path) -> None:
    """``kda-wyut-delta`` profiles the ``kda-wyut`` leg: its metrics live there."""
    assert gate.infer_metrics_family(RUN) == "kda-wyut"
    pin = gate._coerce_rental_pin(tmp_path / "p", 0, blocking_pin())
    assert gate.resolve_metrics_path(pin, tmp_path) == (
        tmp_path / "evidence" / "kda-wyut" / "metrics.jsonl"
    )


def test_inference_family_without_delta_suffix_is_the_run_itself(tmp_path: Path) -> None:
    assert gate.infer_metrics_family("pretrain-pilot") == "pretrain-pilot"
    assert gate.infer_metrics_family("dense124m") == "dense124m"


def test_explicit_metrics_path_overrides_inference(tmp_path: Path) -> None:
    custom = write_metrics(tmp_path / "custom" / "m.jsonl", [9000.0] * 12)
    pins = write_pins(tmp_path / "pins.json", [
        blocking_pin(metrics_path=str(custom)),
    ])
    code, report = gate.run_check(
        pins_path=pins, rental_budget=make_budget(tmp_path / "b.json"), repo_root=tmp_path
    )
    assert code == gate.EXIT_OK
    assert report["verdict"] == "pass"


def test_relative_metrics_path_resolves_against_repo_root(tmp_path: Path) -> None:
    write_metrics(tmp_path / "rel" / "m.jsonl", [9000.0] * 12)
    pins = write_pins(tmp_path / "pins.json", [
        blocking_pin(metrics_path="rel/m.jsonl"),
    ])
    code, _report = gate.run_check(
        pins_path=pins, rental_budget=make_budget(tmp_path / "b.json"), repo_root=tmp_path
    )
    assert code == gate.EXIT_OK


def test_metrics_alias_field_is_honoured(tmp_path: Path) -> None:
    write_metrics(tmp_path / "alt" / "m.jsonl", [9000.0] * 12)
    pins = write_pins(tmp_path / "pins.json", [
        blocking_pin(metrics="alt/m.jsonl"),
    ])
    code, _report = gate.run_check(
        pins_path=pins, rental_budget=make_budget(tmp_path / "b.json"), repo_root=tmp_path
    )
    assert code == gate.EXIT_OK


def test_blocker_reports_the_inferred_metrics_path(tmp_path: Path) -> None:
    pins = write_pins(tmp_path / "pins.json", [blocking_pin()])
    _code, report = gate.run_check(
        pins_path=pins, rental_budget=make_budget(tmp_path / "b.json"), repo_root=tmp_path
    )
    assert report["blockers"][0]["metrics_file"] == str(
        tmp_path / "evidence" / "kda-wyut" / "metrics.jsonl"
    )


# --------------------------------------------------------------------------- #
# Budget path boundary: passed but absent → exit 0 with a warning
# --------------------------------------------------------------------------- #


def test_budget_path_absent_is_exit_zero(tmp_path: Path) -> None:
    pins = write_pins(tmp_path / "pins.json", [blocking_pin()])
    code, report = gate.run_check(
        pins_path=pins, rental_budget=tmp_path / "nope.json", repo_root=tmp_path
    )
    assert code == gate.EXIT_OK
    assert report["verdict"] == "budget-absent"
    assert report["budget_exists"] is False
    # The blocker is still surfaced even though nothing is blocked yet.
    assert report["blockers"][0]["run"] == RUN


def test_budget_directory_is_not_a_budget_file(tmp_path: Path) -> None:
    pins = write_pins(tmp_path / "pins.json", [blocking_pin()])
    (tmp_path / "budgetdir").mkdir()
    code, report = gate.run_check(
        pins_path=pins, rental_budget=tmp_path / "budgetdir", repo_root=tmp_path
    )
    assert code == gate.EXIT_OK
    assert report["verdict"] == "budget-absent"


def test_cli_budget_absent_exits_zero(tmp_path: Path) -> None:
    pins = write_pins(tmp_path / "pins.json", [blocking_pin()])
    result = run_cli("--pins", str(pins), "--rental-budget", str(tmp_path / "no.json"))
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["verdict"] == "budget-absent"


# --------------------------------------------------------------------------- #
# Fail-closed: a broken pins file must not silently unlock the budget
# --------------------------------------------------------------------------- #


def test_missing_pins_file_fails_closed(tmp_path: Path) -> None:
    budget = make_budget(tmp_path / "rental.json")
    code, report = gate.run_check(
        pins_path=tmp_path / "no-pins.json", rental_budget=budget, repo_root=tmp_path
    )
    assert code == gate.EXIT_FAIL
    assert report["verdict"] == "no-data"


def test_malformed_pins_json_fails_closed(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{не json", encoding="utf-8")
    code, report = gate.run_check(
        pins_path=bad, rental_budget=make_budget(tmp_path / "b.json"), repo_root=tmp_path
    )
    assert code == gate.EXIT_FAIL
    assert report["verdict"] == "no-data"


def test_blocking_pin_without_threshold_fails_closed(tmp_path: Path) -> None:
    pins = write_pins(tmp_path / "pins.json", [{"run": RUN, "blocks_rental": True}])
    code, report = gate.run_check(
        pins_path=pins, rental_budget=make_budget(tmp_path / "b.json"), repo_root=tmp_path
    )
    assert code == gate.EXIT_FAIL
    assert report["verdict"] == "no-data"
    assert "threshold" in report["message"]


def test_non_bool_blocks_rental_fails_closed(tmp_path: Path) -> None:
    pins = write_pins(tmp_path / "pins.json", [
        {"run": RUN, "threshold": THRESHOLD, "blocks_rental": "yes"},
    ])
    code, report = gate.run_check(
        pins_path=pins, rental_budget=make_budget(tmp_path / "b.json"), repo_root=tmp_path
    )
    assert code == gate.EXIT_FAIL
    assert report["verdict"] == "no-data"


def test_pin_without_run_fails_closed(tmp_path: Path) -> None:
    pins = write_pins(tmp_path / "pins.json", [
        {"threshold": THRESHOLD, "blocks_rental": True},
    ])
    code, _report = gate.run_check(
        pins_path=pins, rental_budget=make_budget(tmp_path / "b.json"), repo_root=tmp_path
    )
    assert code == gate.EXIT_FAIL


def test_missing_pins_fails_closed_even_without_budget(tmp_path: Path) -> None:
    code, report = gate.run_check(
        pins_path=tmp_path / "no-pins.json", repo_root=tmp_path
    )
    assert code == gate.EXIT_FAIL
    assert report["verdict"] == "no-data"


def test_duplicate_run_blocks_once(tmp_path: Path) -> None:
    pins = write_pins(tmp_path / "pins.json", [blocking_pin(), blocking_pin()])
    _code, report = gate.run_check(
        pins_path=pins, rental_budget=make_budget(tmp_path / "b.json"), repo_root=tmp_path
    )
    assert [b["run"] for b in report["blockers"]] == [RUN]
    assert report["blocking_pins"] == 2


# --------------------------------------------------------------------------- #
# Shipped default pins (ADR-032 / ADR-034)
# --------------------------------------------------------------------------- #


def test_shipped_default_pins_block_rental() -> None:
    assert DEFAULT_PINS.exists(), f"нет дефолтного пин-файла: {DEFAULT_PINS}"
    pins = gate.load_rental_pins(DEFAULT_PINS)
    armed = next((p for p in pins if p.get("blocks_rental")), None)
    assert armed is not None
    assert armed["run"] == RUN
    assert armed["threshold"] == THRESHOLD


def test_default_pins_path_is_repo_anchored() -> None:
    assert gate.DEFAULT_PINS == CASE_DIR / "evidence" / "kpi-pins.json"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_cli_selftest_is_green() -> None:
    result = run_cli("--selftest")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FAIL" not in result.stdout
    assert "PASS" in result.stdout


def test_cli_blocked_exits_one_and_prints_to_stderr(tmp_path: Path) -> None:
    pins = write_pins(tmp_path / "pins.json", [blocking_pin()])
    budget = make_budget(tmp_path / "rental.json")
    result = run_cli("--pins", str(pins), "--rental-budget", str(budget))
    assert result.returncode == 1
    assert "FAIL" in result.stderr
    assert "арендная смета заблокирована" in result.stderr
    report = json.loads(result.stdout)
    assert report["verdict"] == "blocked"


def test_cli_writes_report(tmp_path: Path) -> None:
    budget = make_budget(tmp_path / "rental.json")
    metrics = write_metrics(
        tmp_path / "evidence" / "kda-wyut" / "metrics.jsonl", [9000.0] * 12
    )
    report_path = tmp_path / "gate.json"
    # The CLI infers metrics from the repo root it is invoked from; point it at
    # the synthetic tree with the pin's explicit metrics path instead.
    pins = write_pins(tmp_path / "pins.json", [
        blocking_pin(metrics_path=str(metrics)),
    ])
    result = run_cli(
        "--pins", str(pins), "--rental-budget", str(budget),
        "--json", str(report_path), "--quiet",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["verdict"] == "pass"
    assert report["exit_code"] == 0
    assert report["schema"] == gate.REPORT_SCHEMA
