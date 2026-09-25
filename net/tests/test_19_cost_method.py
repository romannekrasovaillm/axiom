"""Guards for the ADR-015 measurement procedure itself (``cost_method``).

ADR-015 turns the measurement method into part of the acceptance contour, and
says so explicitly: "методика сама становится предметом ревью: её изменение
потребует нового решения".  These tests pin the *method* — not the outcome — so
that a silent change of the procedure cannot move the criterion 16 verdict:

* the thresholds and the metric definition are the criterion's own numbers;
* the round order is reversed every round and the first round is dropped;
* the verdict is the **median** with the spread reported;
* a spread above 5% of the threshold is **not** a verdict: it is "не определён";
* an uncontrolled / idle / drifting clock state is also "не определён";
* the strict reading of ADR-015 p.1 (no hard pin => no verdict) is available and
  behaves differently from the implemented one, so the lever is visible.

None of these touch the model, only the procedure.
"""

from __future__ import annotations

import statistics
import time

import cost_method
from cost_method import UNDETERMINED, ClockControl, Ratios, verdict


def _ratios(values: list[float], leg: str = "after", ref: str = "dense") -> Ratios:
    return Ratios(
        leg=leg, ref=ref, n=len(values),
        median=statistics.median(values), lo=min(values), hi=max(values),
        spread=max(values) - min(values), leg_median=0.0, ref_median=0.0,
        per_round=tuple(values),
    )


def _clock(*, hard_pin: bool, soft: str = "", samples: tuple[int, ...] = ()) -> ClockControl:
    """A ClockControl with a hand-set report (no subprocess, no GPU needed)."""
    control = ClockControl()
    control.report.backend = "gpu"
    control.report.max_clock = 3105
    control.report.hard_pin_attempt = "lgc 3105,3105"
    control.report.hard_pin_ok = hard_pin
    control.report.hard_pin_detail = "locked" if hard_pin else "permission denied (rc=4)"
    control.report.soft_control = soft
    control.report.warmup_samples = samples
    return control


# --- the criterion's numbers are not the procedure's to change --------------


def test_criterion_numbers_are_pinned():
    """ADR-015 p.6: the method changes, the threshold and metric do not."""
    assert cost_method.THRESHOLD_8K == 1.05
    assert cost_method.THRESHOLD_STRICTLY_BETTER == 1.0
    assert cost_method.DISPERSION_FRACTION_OF_THRESHOLD == 0.05
    assert cost_method.MIN_COUNTED_ROUNDS == 5
    assert cost_method.DEFAULT_ROUNDS >= cost_method.MIN_COUNTED_ROUNDS + 1


# --- the measurement mechanics ----------------------------------------------


def test_round_order_alternates_and_first_round_is_dropped():
    calls: list[str] = []
    legs = {
        "dense": lambda: calls.append("dense"),
        "after": lambda: calls.append("after"),
    }
    m = cost_method.measure(
        legs, label="unit", rounds=3, inner_repeats=1,
        warmup_passes=1, warmup_min_seconds=0.0, warmup_max_seconds=0.0,
    )
    assert m.dropped == 1
    assert len(m.rounds) == 3
    assert len(m.counted) == 2
    # One warm-up pass (2 calls) then three rounds of two calls.
    assert calls[-6:] == ["dense", "after", "after", "dense", "dense", "after"]
    assert m.rounds[0].order == ("dense", "after")
    assert m.rounds[1].order == ("after", "dense")
    assert m.rounds[2].order == ("dense", "after")


def test_round_value_is_the_min_over_inner_repeats():
    """The estimator is the minimum over the round's repeats (host-jitter filter)."""
    durations = iter([0.004, 0.0001])
    best, worst = cost_method._time_leg(lambda: time.sleep(next(durations, 0.0001)), 2)
    assert best < 0.002 < worst  # 4 ms vs 0.1 ms is far above scheduling noise


# --- the verdict rules ------------------------------------------------------


def test_verdict_uses_the_median_not_the_best_or_worst_round():
    clock = _clock(hard_pin=True)
    # The best round (1.04) is under the threshold, the median (1.06) is over it:
    # the verdict follows the median, not the best or the worst round.
    ratios = _ratios([1.04, 1.05, 1.06, 1.07, 1.08])  # spread 0.04 <= 0.0525
    v = verdict(ratios, cost_method.THRESHOLD_8K, clock=clock)
    assert v.status == "fail"
    assert v.ratios.lo < cost_method.THRESHOLD_8K < v.ratios.median == 1.06
    assert "1.0600" in v.line("x")


def test_verdict_passes_at_the_threshold_but_strict_better_does_not():
    clock = _clock(hard_pin=True)
    at_threshold = _ratios([1.05, 1.05, 1.05, 1.05, 1.05])
    assert verdict(at_threshold, cost_method.THRESHOLD_8K, clock=clock).status == "pass"
    assert verdict(at_threshold, cost_method.THRESHOLD_STRICTLY_BETTER,
                   clock=clock, strictly_better=True).status == "fail"


def test_spread_above_five_percent_of_the_threshold_is_not_a_verdict():
    """ADR-015 p.5: noise is neither PASS nor FAIL — it is "не определён"."""
    clock = _clock(hard_pin=True)
    median_under_threshold = _ratios([1.00, 1.02, 1.04, 1.08, 1.06])  # spread 0.08 > 0.0525
    v = verdict(median_under_threshold, cost_method.THRESHOLD_8K, clock=clock)
    assert v.status == UNDETERMINED
    assert v.ratios.median <= cost_method.THRESHOLD_8K  # would have been green
    assert any("разброс" in r for r in v.reasons)

    median_over_threshold = _ratios([1.07, 1.09, 1.20, 1.10, 1.12])  # spread 0.13
    v2 = verdict(median_over_threshold, cost_method.THRESHOLD_8K, clock=clock)
    assert v2.status == UNDETERMINED
    assert v2.ratios.median > cost_method.THRESHOLD_8K  # would have been red
    assert any("разброс" in r for r in v2.reasons)


def test_spread_below_the_limit_keeps_the_verdict():
    clock = _clock(hard_pin=True)
    v = verdict(_ratios([1.03, 1.04, 1.05, 1.06, 1.07]), cost_method.THRESHOLD_8K, clock=clock)
    assert v.status == "pass"  # spread 0.04 <= 0.0525; median 1.05 <= 1.05 is the boundary
    assert v.ratios.median == 1.05


def test_fewer_than_five_counted_rounds_is_not_a_verdict():
    clock = _clock(hard_pin=True)
    v = verdict(_ratios([1.02, 1.03, 1.04]), cost_method.THRESHOLD_8K, clock=clock)
    assert v.status == UNDETERMINED
    assert any("раундов" in r for r in v.reasons)


def test_uncontrolled_clocks_withhold_the_verdict():
    """ADR-015 p.1: no clock control => the numbers are not a verdict."""
    clock = _clock(hard_pin=False)  # no hard pin and no soft fallback
    v = verdict(_ratios([1.03, 1.03, 1.03, 1.03, 1.03]), cost_method.THRESHOLD_8K, clock=clock)
    assert v.status == UNDETERMINED
    assert any("не зафиксированы" in r for r in v.reasons)


def test_idle_clocks_withhold_the_verdict():
    """The idle floor is exactly the state ADR-015 p.1 refuses to judge."""
    clock = _clock(hard_pin=True, samples=(210, 210, 210))
    v = verdict(_ratios([1.03, 1.03, 1.03, 1.03, 1.03]), cost_method.THRESHOLD_8K, clock=clock)
    assert v.status == UNDETERMINED
    assert any("холостых" in r for r in v.reasons)


def test_strict_reading_withholds_more_than_the_implemented_one():
    """The task's stronger reading is available and is a visible lever."""
    clock = _clock(hard_pin=False, soft="GPUPowerMizerMode=2->1", samples=(2685, 2685, 2685))
    ratios = _ratios([1.03, 1.03, 1.03, 1.03, 1.03])
    implemented = verdict(ratios, cost_method.THRESHOLD_8K, clock=clock)
    strict = verdict(ratios, cost_method.THRESHOLD_8K, clock=clock, strict=True)
    assert implemented.status == "pass"
    assert strict.status == UNDETERMINED
    assert any("строгое чтение" in r for r in strict.reasons)


def test_cpu_run_has_no_clock_gate():
    """On CPU there are no GPU clocks to control; the noise gate still applies."""
    clock = ClockControl()  # backend stays "cpu"
    v = verdict(_ratios([1.03, 1.03, 1.03, 1.03, 1.03]), cost_method.THRESHOLD_8K, clock=clock)
    assert v.status == "pass"
