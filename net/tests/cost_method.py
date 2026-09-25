"""ADR-015 cost-measurement procedure — the method behind criterion 16.

Why this module exists
----------------------

Criterion 16 (spec §6 p.16) says: "the forward at 8K is no worse than 5% against
the dense path in time/memory; at 64K strictly better".  The old gate measured
the dense leg *first* and the sparse leg right after it, once per process, and
took the minimum over three repeats.  That made the 8K ratio land anywhere in
**1.02…1.20 across processes** — an 18% spread for a 3% effect and a 5%
threshold — so the verdict was a lottery, not a measurement (ADR-015 Context).
The machine's GPU also idles at 210 MHz of 3105, so a leg measured before the
clock ramp and a leg measured after it are not measured at the same frequency.

The procedure below is ADR-015's Decision, implemented literally:

1. **Clocks are pinned before the measurement** (``nvidia-smi -lgc``).  Where the
   hard lock is not permitted, a user-level fallback is used —
   ``nvidia-settings`` ``GPUPowerMizerMode=1`` ("prefer maximum performance"),
   which removes the idle downclock floor — and the **actual sampled clocks are
   printed either way**.  A verdict is withheld unless the clock state was
   controlled *and* verified stable during the measured rounds (ADR-015 p.1).
   On a non-GPU run there are no clocks to control and the gate is N/A.
2. **The order of the legs is reversed every round**, so no leg systematically
   runs on the warmed-up state; **the first round is dropped** (ADR-015 p.2).
3. **Both legs are warmed up until the sampled clock has plateaued**, then a
   fixed number of rounds is run (ADR-015 p.3).
4. **The verdict is the median** of the per-round ratios; the report always
   carries the **spread (min…max)**, the round count and the clock state.  A
   single number without a spread is not a verdict (ADR-015 p.4).
5. **No verdict on noise:** if the spread within the process exceeds 5% of the
   measurement threshold, the verdict is **"не определён" / undetermined** —
   neither PASS nor FAIL — with the reason spelled out (ADR-015 p.5).
6. **The thresholds and the metric are untouched** (ADR-015 p.6): this module
   changes how an already-declared criterion is measured, never what it is.

Two readings of p.1, and which one is implemented
-------------------------------------------------

ADR-015 p.1 withholds the verdict "при работе на «холостых» частотах" — *while
running at idle clocks*.  The task statement restates it more strictly ("no hard
pin ⇒ no verdict").  This module implements the **p.1 letter**: the hard pin is
attempted and reported verbatim; when it is unavailable the clock state must
still be *under control and verified stable* (samples printed) for a verdict to
be issued.  The strict reading is reported explicitly in the output as
``strict (p.3 of the task): ...`` so the architect can see both.  Under either
reading the thresholds, the metric and the noise gate are identical, so a red
verdict stays red.

Parameters (fixed in code on purpose — ADR-015 Consequences: "методика сама
становится предметом ревью", so the verdict-affecting knobs are not settable
from the environment):

* ``DEFAULT_ROUNDS = 10`` — the ADR requires ≥5 counted rounds; the spread is a
  max−min statistic, which is biased low on a small sample, so a starved sample
  would under-report the very quantity the gate tests.  10 rounds → 9 counted.
* ``DEFAULT_INNER_REPEATS = 4`` — each round's per-leg value is the minimum over
  four calls, i.e. the same estimator family as the previous procedure (min over
  repeats): the minimum removes host-side jitter without removing the state
  drift that the between-round spread is meant to expose.
* ``WARMUP_*`` — warm-up ends on a clock plateau, not on a fixed count.
"""

from __future__ import annotations

import dataclasses
import shutil
import statistics
import subprocess
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import jax

# --- the criterion's own numbers: not tunable (ADR-015 p.6) -----------------
THRESHOLD_8K = 1.05  # 8K: sparse time/memory <= 1.05 x dense
THRESHOLD_STRICTLY_BETTER = 1.0  # 64K: sparse time/memory < dense
DISPERSION_FRACTION_OF_THRESHOLD = 0.05  # ADR-015 p.5: 5% of the threshold

# --- procedure parameters (fixed in code, see the module docstring) ---------
MIN_COUNTED_ROUNDS = 5
DEFAULT_ROUNDS = 10
DEFAULT_INNER_REPEATS = 4

WARMUP_MIN_PASSES = 3
WARMUP_MIN_SECONDS = 15.0
WARMUP_MAX_SECONDS = 180.0
WARMUP_STABLE_SAMPLES = 3
WARMUP_PLATEAU_HOLD_SECONDS = 3.0  # the plateau must *persist*, not just appear
CLOCK_PLATEAU_BAND = 0.02  # warm-up ends when the clock is within +-2%
CLOCK_MEASUREMENT_BAND = 0.02  # measured rounds must hold the clock within +-2%
CLOCK_IDLE_FRACTION = 0.5  # below half the max clock is the idle floor

UNDETERMINED = "undetermined"


# ---------------------------------------------------------------------------
# GPU clock control
# ---------------------------------------------------------------------------


def _backend() -> str:
    """``"gpu"``/``"cpu"`` — whether there are GPU clocks to control at all."""
    try:
        return jax.devices()[0].platform
    except Exception:  # pragma: no cover - no device at all
        return "cpu"


def _sm_clocks() -> tuple[int, ...] | None:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        proc = subprocess.run(
            [exe, "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return None
    if proc.returncode != 0:
        return None
    values = tuple(int(v) for v in (s.strip() for s in proc.stdout.splitlines()) if v.isdigit())
    return values or None


def _max_sm_clock() -> int | None:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        proc = subprocess.run(
            [exe, "--query-gpu=clocks.max.sm", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return None
    if proc.returncode != 0:
        return None
    vals = [int(v) for v in (s.strip() for s in proc.stdout.splitlines()) if v.isdigit()]
    return max(vals) if vals else None


def band(samples: Sequence[int | None]) -> tuple[float, float, float] | None:
    """(median, min, max) of the non-empty samples, or None if there are <2."""
    vals = [s for s in samples if s]
    if len(vals) < 2:
        return None
    return float(statistics.median(vals)), float(min(vals)), float(max(vals))


def _run(cmd: Sequence[str]) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        return 127, f"{type(exc).__name__}: {exc}"
    out = (proc.stdout + proc.stderr).strip().replace("\n", " ")
    return proc.returncode, out


@dataclasses.dataclass
class ClockReport:
    """What was done to the clocks, and what they actually did."""

    backend: str = "cpu"
    hard_pin_attempt: str = ""
    hard_pin_ok: bool = False
    hard_pin_detail: str = ""
    soft_control: str = ""
    soft_detail: str = ""
    max_clock: int | None = None
    warmup_samples: tuple[int, ...] = ()

    @property
    def on_gpu(self) -> bool:
        return self.backend == "gpu"

    @property
    def controlled(self) -> bool:
        """Some clock control is in effect: a hard lock or the soft fallback."""
        return self.hard_pin_ok or bool(self.soft_control)

    def warmup_band(self) -> tuple[float, float, float] | None:
        """(median MHz, min MHz, max MHz) over the warm-up samples."""
        return band(self.warmup_samples)

    def clock_line(self) -> str:
        bits = [f"backend={self.backend}"]
        if self.max_clock:
            bits.append(f"max={self.max_clock}MHz")
        if self.hard_pin_attempt:
            bits.append(f"hard pin `{self.hard_pin_attempt}` -> "
                        f"{'OK' if self.hard_pin_ok else 'FAILED'}: {self.hard_pin_detail}")
        if self.soft_control:
            bits.append(f"soft control {self.soft_control} ({self.soft_detail})")
        if self.warmup_samples:
            band_ = self.warmup_band()
            bits.append(f"warm-up clocks {band_[1]}…{band_[2]} MHz (median {band_[0]:.0f}, "
                        f"n={len(self.warmup_samples)})")
        return "; ".join(bits)


class ClockControl:
    """Context manager: put the GPU clocks under control, restore on exit.

    Hard pin first (``nvidia-smi -lgc`` — the mechanism ADR-015 names); if the
    user is not permitted to lock clocks, fall back to the user-level
    ``nvidia-settings`` ``GPUPowerMizerMode=1``, which removes the idle
    downclock floor.  Either way the *actual* clocks are sampled and reported.
    """

    def __init__(self) -> None:
        self.report = ClockReport()
        self._restore_sm = False
        self._prev_powermizer: str | None = None

    # -- lifecycle ----------------------------------------------------------
    def __enter__(self) -> "ClockControl":
        self.report.backend = _backend()
        if not self.report.on_gpu:
            return self
        if shutil.which("nvidia-smi") is None:
            self.report.hard_pin_detail = "nvidia-smi not found"
            self._try_soft_control()
            return self
        self.report.max_clock = _max_sm_clock()
        self._try_hard_pin()
        if not self.report.hard_pin_ok:
            self._try_soft_control()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._restore_sm:
            _run(["nvidia-smi", "-rgc"])
        if self._prev_powermizer is not None:
            _run(["nvidia-settings", "-a",
                  f"[gpu:0]/GPUPowerMizerMode={self._prev_powermizer}"])
        return None

    def _try_hard_pin(self) -> None:
        target = self.report.max_clock
        if not target:
            return
        cmd = ["nvidia-smi", "-lgc", f"{target},{target}"]
        self.report.hard_pin_attempt = " ".join(cmd[1:])
        rc, out = _run(cmd)
        self.report.hard_pin_ok = rc == 0
        self.report.hard_pin_detail = out or f"rc={rc}"
        self._restore_sm = rc == 0

    def _try_soft_control(self) -> None:
        if shutil.which("nvidia-settings") is None:
            self.report.soft_detail = "nvidia-settings not found"
            return
        rc, out = _run(["nvidia-settings", "-q", "GPUPowerMizerMode"])
        if rc != 0:
            self.report.soft_detail = out or f"query rc={rc}"
            return
        current = ""
        for token in out.replace(".", " ").split():
            if token.isdigit():
                current = token
        self._prev_powermizer = current or None
        rc, out = _run(["nvidia-settings", "-a", "[gpu:0]/GPUPowerMizerMode=1"])
        if rc != 0:
            self.report.soft_detail = out or f"set rc={rc}"
            self._prev_powermizer = None
            return
        rc, out = _run(["nvidia-settings", "-q", "GPUPowerMizerMode"])
        if rc != 0 or " 1." not in out:
            self.report.soft_detail = f"readback failed: {out or f'rc={rc}'}"
            return
        self.report.soft_control = f"GPUPowerMizerMode={current or '?'}->1"
        self.report.soft_detail = "readback 1 (prefer maximum performance)"

    # -- sampling -----------------------------------------------------------
    def read(self) -> int | None:
        """The current SM clock, without recording it."""
        if not self.report.on_gpu:
            return None
        clocks = _sm_clocks()
        return clocks[0] if clocks else None

    def sample(self) -> int | None:
        """Read the current SM clock and record it as a warm-up sample."""
        value = self.read()
        if value is not None:
            self.report.warmup_samples = self.report.warmup_samples + (value,)
        return value


# ---------------------------------------------------------------------------
# Warm-up and rounds
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class WarmupReport:
    passes: int
    seconds: float
    stable: bool
    samples: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class Round:
    index: int
    order: tuple[str, ...]
    times: Mapping[str, float]  # leg -> min over the round's inner repeats
    spreads: Mapping[str, float]  # leg -> (max-min)/min over the inner repeats
    clock: int | None


@dataclasses.dataclass
class Measurement:
    label: str
    legs: tuple[str, ...]
    rounds: list[Round]
    dropped: int
    inner_repeats: int
    warmup: WarmupReport

    @property
    def counted(self) -> list[Round]:
        return self.rounds[self.dropped:]

    def leg_times(self, leg: str) -> list[float]:
        return [r.times[leg] for r in self.counted]

    def ratios(self, leg: str, ref: str) -> list[float]:
        return [r.times[leg] / r.times[ref] for r in self.counted]

    def stats(self, leg: str, ref: str) -> "Ratios":
        ratio = self.ratios(leg, ref)
        leg_times = self.leg_times(leg)
        ref_times = self.leg_times(ref)
        return Ratios(
            leg=leg, ref=ref, n=len(ratio),
            median=statistics.median(ratio), lo=min(ratio), hi=max(ratio),
            spread=max(ratio) - min(ratio),
            leg_median=statistics.median(leg_times),
            ref_median=statistics.median(ref_times),
            per_round=tuple(ratio),
        )

    def inner_spread(self, leg: str) -> float:
        vals = [r.spreads[leg] for r in self.counted]
        return statistics.median(vals) if vals else 0.0

    def clock_samples(self) -> tuple[int, ...]:
        """Clock samples of the *counted* rounds (the dropped one is not evidence)."""
        return tuple(r.clock for r in self.counted if r.clock)

    def clock_band(self) -> tuple[float, float, float] | None:
        return band(self.clock_samples())

    def clock_stability(self) -> tuple[bool | None, str]:
        """Was the clock steady across the counted rounds?  (ADR-015 p.1/p.3)"""
        vals = self.clock_samples()
        if len(vals) < 2:
            return None, f"частоты не засэмплированы в учтённых раундах (n={len(vals)})"
        med, lo, hi = self.clock_band()
        if (hi - lo) > CLOCK_MEASUREMENT_BAND * med:
            return False, (f"частоты гуляли в учтённых раундах: {lo:.0f}…{hi:.0f} МГц "
                           f"(медиана {med:.0f}, n={len(vals)}) при допуске ±{CLOCK_MEASUREMENT_BAND:.0%}")
        return True, f"частоты в учтённых раундах {lo:.0f}…{hi:.0f} МГц (медиана {med:.0f}, n={len(vals)})"


@dataclasses.dataclass(frozen=True)
class Ratios:
    leg: str
    ref: str
    n: int
    median: float
    lo: float
    hi: float
    spread: float
    leg_median: float
    ref_median: float
    per_round: tuple[float, ...]


def _plateau(samples: Sequence[int | None], band: float) -> bool:
    recent = [s for s in samples[-WARMUP_STABLE_SAMPLES:] if s]
    if len(recent) < WARMUP_STABLE_SAMPLES:
        return False
    median = statistics.median(recent)
    return all(abs(s - median) <= band * median for s in recent)


def warm_up(
    legs: Sequence[Callable[[], Any]],
    clock: ClockControl | None,
    label: str,
    *,
    min_passes: int = WARMUP_MIN_PASSES,
    min_seconds: float = WARMUP_MIN_SECONDS,
    max_seconds: float = WARMUP_MAX_SECONDS,
) -> WarmupReport:
    """Run every leg until the sampled clock has plateaued (ADR-015 p.3)."""
    print(f"[{label}] warm-up: all legs until the clock plateaus "
          f"(min {min_passes} passes / {min_seconds:.0f}s, plateau held "
          f"{WARMUP_PLATEAU_HOLD_SECONDS:.0f}s, budget {max_seconds:.0f}s)", flush=True)
    start = time.perf_counter()
    passes = 0
    samples: list[int | None] = []
    plateau_since: float | None = None
    stable = False
    while True:
        for fn in legs:
            jax.block_until_ready(fn())
        passes += 1
        samples.append(clock.sample() if clock is not None else None)
        elapsed = time.perf_counter() - start
        sampled = [s for s in samples if s]
        if clock is None or not sampled:
            # No clocks to watch (CPU run): the warm-up is the fixed floor.
            ready = passes >= min_passes and elapsed >= min_seconds
        else:
            # The driver raises the boost only after a sustained load (seconds),
            # so the plateau has to *persist* before the measurement starts.
            if _plateau(samples, CLOCK_PLATEAU_BAND):
                plateau_since = elapsed if plateau_since is None else plateau_since
            else:
                plateau_since = None
            held = 0.0 if plateau_since is None else elapsed - plateau_since
            ready = (passes >= min_passes and elapsed >= min_seconds
                     and held >= WARMUP_PLATEAU_HOLD_SECONDS)
        if ready:
            stable = True
            break
        if elapsed >= max_seconds:
            break
    report = WarmupReport(passes=passes, seconds=time.perf_counter() - start,
                          stable=stable, samples=tuple(s for s in samples if s))
    print(f"[{label}] warm-up done: {report.passes} passes, {report.seconds:.1f}s, "
          f"clock plateau={'yes' if report.stable else 'NO'}, samples={report.samples}", flush=True)
    return report


def _time_leg(fn: Callable[[], Any], repeats: int) -> tuple[float, float]:
    best = float("inf")
    worst = 0.0
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        dt = time.perf_counter() - t0
        best = min(best, dt)
        worst = max(worst, dt)
    return best, worst


def measure(
    legs: Mapping[str, Callable[[], Any]],
    *,
    label: str,
    clock: ClockControl | None = None,
    rounds: int = DEFAULT_ROUNDS,
    inner_repeats: int = DEFAULT_INNER_REPEATS,
    drop_first: bool = True,
    warmup_passes: int = WARMUP_MIN_PASSES,
    warmup_min_seconds: float = WARMUP_MIN_SECONDS,
    warmup_max_seconds: float = WARMUP_MAX_SECONDS,
) -> Measurement:
    """Warm both legs up, then run alternating-order rounds (ADR-015 p.2/p.3).

    Each round runs every leg once with ``inner_repeats`` calls; the round's
    value for a leg is the minimum of those calls.  The leg order is reversed
    every round, so no leg systematically runs on the warmed-up state, and the
    first round is dropped from the statistics.

    The gate uses the module defaults; the probes may pass different counts, and
    the procedure's own test drives the warm-up down to one pass.
    """
    names = tuple(legs)
    warmup = warm_up([legs[n] for n in names], clock, label,
                     min_passes=warmup_passes, min_seconds=warmup_min_seconds,
                     max_seconds=warmup_max_seconds)

    print(f"[{label}] rounds: {rounds} x {inner_repeats} inner repeats, "
          f"order alternated dense-first/sparse-first, round 0 dropped", flush=True)
    out: list[Round] = []
    for i in range(rounds):
        order = names if i % 2 == 0 else names[::-1]
        times: dict[str, float] = {}
        spreads: dict[str, float] = {}
        for name in order:
            best, worst = _time_leg(legs[name], inner_repeats)
            times[name] = best
            spreads[name] = (worst - best) / best if best else 0.0
        out.append(Round(index=i, order=order, times=times, spreads=spreads,
                         clock=clock.read() if clock is not None else None))
    dropped = 1 if drop_first else 0
    return Measurement(label=label, legs=names, rounds=out, dropped=dropped,
                       inner_repeats=inner_repeats, warmup=warmup)


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Verdict:
    status: str  # "pass" | "fail" | "undetermined"
    threshold: float
    dispersion_limit: float
    ratios: Ratios | None
    reasons: tuple[str, ...]

    @property
    def determined(self) -> bool:
        return self.status != UNDETERMINED

    def line(self, label: str) -> str:
        if self.ratios is None:
            return f"[{label}] VERDICT: НЕ ОПРЕДЕЛЁН — {'; '.join(self.reasons)}"
        r = self.ratios
        head = {
            "pass": "PASS",
            "fail": "FAIL",
            "undetermined": "НЕ ОПРЕДЕЛЁН (undetermined — это не PASS и не FAIL)",
        }[self.status]
        return (
            f"[{label}] VERDICT: {head} — {r.leg}/{r.ref} median {r.median:.4f} "
            f"(min {r.lo:.4f}…max {r.hi:.4f}, разброс {r.spread:.4f}, "
            f"порог {self.threshold:.2f}, лимит разброса {self.dispersion_limit:.4f}, "
            f"раундов учтено {r.n}); {'; '.join(self.reasons)}"
        )


def clock_reasons(
    clock: ClockControl | None,
    measurement: Measurement | None,
    *,
    strict: bool = False,
) -> list[str]:
    """Everything that disqualifies the clock state for a verdict (ADR-015 p.1)."""
    if clock is None:
        return []
    report = clock.report
    if not report.on_gpu:
        return []
    reasons: list[str] = []
    if strict and not report.hard_pin_ok:
        reasons.append(
            "строгое чтение п.1: жёсткий пиннинг частот недоступен "
            f"({report.hard_pin_detail}) — вердикт не выдаётся"
        )
    if not report.controlled:
        reasons.append(
            "частоты GPU не зафиксированы: жёсткий пиннинг недоступен "
            f"({report.hard_pin_detail or 'нет nvidia-smi -lgc'}) и мягкий контроль "
            f"не применился ({report.soft_detail or 'нет nvidia-settings'})"
        )
        return reasons
    measured = measurement.clock_band() if measurement is not None else None
    observed = measured or report.warmup_band()
    if observed is not None and report.max_clock and observed[0] < CLOCK_IDLE_FRACTION * report.max_clock:
        reasons.append(
            f"карта работает на «холостых» частотах ({observed[0]:.0f} МГц из "
            f"{report.max_clock}) — вердикт по ADR-015 п.1 не выдаётся"
        )
    if measurement is not None:
        stable, why = measurement.clock_stability()
        if stable is not True:
            reasons.append(why)
    return reasons


def verdict(
    ratios: Ratios | None,
    threshold: float,
    *,
    clock: ClockControl | None = None,
    measurement: Measurement | None = None,
    strictly_better: bool = False,
    strict: bool = False,
    label: str = "",
) -> Verdict:
    """Turn the measured ratios into a criterion-16 verdict (ADR-015 p.4/p.5).

    ``strict=False`` (the implemented reading of ADR-015 p.1) withholds the
    verdict only when the clock state is uncontrolled, at the idle floor or
    unstable during the counted rounds.  ``strict=True`` additionally withholds
    it whenever the hard pin itself was unavailable — the task statement's
    stronger reading; both are printed so the architect sees the lever.
    """
    limit = DISPERSION_FRACTION_OF_THRESHOLD * threshold
    if ratios is None:
        return Verdict(UNDETERMINED, threshold, limit, None,
                       ("замер не состоялся",))
    reasons: list[str] = []
    if ratios.n < MIN_COUNTED_ROUNDS:
        reasons.append(f"учтено {ratios.n} раундов < {MIN_COUNTED_ROUNDS} (ADR-015 п.3)")
    reasons.extend(clock_reasons(clock, measurement, strict=strict))
    if ratios.spread > limit:
        reasons.append(
            f"разброс {ratios.spread:.4f} > 5% порога ({limit:.4f}) — вердикт по шуму запрещён"
        )
    if reasons:
        return Verdict(UNDETERMINED, threshold, limit, ratios, tuple(reasons))
    if strictly_better:
        passed = ratios.median < threshold
    else:
        passed = ratios.median <= threshold
    return Verdict("pass" if passed else "fail", threshold, limit, ratios, ())


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_clock(clock: ClockControl | None, label: str) -> None:
    if clock is None:
        print(f"[{label}] clock: not controlled (no ClockControl in this run)")
        return
    rep = clock.report
    print(f"[{label}] clock: {rep.clock_line()}")
    if rep.on_gpu and not rep.hard_pin_ok:
        print(f"[{label}] strict (task p.3): жёсткий пиннинг недоступен -> по строгому чтению "
              f"вердикт не выдаётся; реализовано чтение п.1 ADR-015 "
              f"(контроль + проверенная стабильность частот)")


def print_measurement(m: Measurement, *, ref: str, threshold: float, legs: Iterable[str] | None = None,
                      unit_scale: float = 1e3, unit: str = "ms") -> None:
    """Print every leg's median/spread and the ratio statistics (ADR-015 p.4)."""
    print(f"[{m.label}] method: ADR-015 — {len(m.rounds)} rounds "
          f"({m.dropped} dropped, {len(m.counted)} counted) x {m.inner_repeats} inner repeats "
          f"(round value = min); warm-up {m.warmup.passes} passes, "
          f"{m.warmup.seconds:.1f}s, clock plateau {'yes' if m.warmup.stable else 'NO'}; "
          f"threshold {threshold:.2f}, dispersion limit "
          f"{DISPERSION_FRACTION_OF_THRESHOLD * threshold:.4f}")
    stable, why = m.clock_stability()
    print(f"[{m.label}] clock during counted rounds: "
          f"{'stable' if stable else ('UNSTABLE' if stable is False else 'not sampled')} — {why}")
    names = tuple(legs) if legs is not None else m.legs
    for name in names:
        times = m.leg_times(name)
        med = statistics.median(times)
        spread = max(times) - min(times)
        tag = " (ref)" if name == ref else ""
        print(f"[{m.label}]   {name:<8}{tag:<6} median {med * unit_scale:9.3f} {unit}  "
              f"min {min(times) * unit_scale:9.3f}  max {max(times) * unit_scale:9.3f}  "
              f"spread {spread * unit_scale:7.3f} {unit}  n={len(times)}  "
              f"inner spread {m.inner_spread(name):.1%}")
    for name in names:
        if name == ref:
            continue
        r = m.stats(name, ref)
        print(f"[{m.label}]   ratio {name}/{ref}: median {r.median:.4f}  "
              f"min {r.lo:.4f}  max {r.hi:.4f}  разброс {r.spread:.4f}  n={r.n}  "
              f"per-round {[round(x, 3) for x in r.per_round]}")


def print_verdict(v: Verdict, label: str) -> None:
    print(v.line(label), flush=True)
