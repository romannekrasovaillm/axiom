"""BF16-кампания фаза 1 — приборы замера (``tools/mfu_bf16_protocol.py``,
``tools/loss_parity_bf16.py``).

The task ships the MFU protocol and the loss-parity leg as *scripts* whose GPU
runs are the architect's (EMPTY-PENDING while no stand is up), so what has to be
mechanically true here is: the arithmetic of the verdict, the shape of the plan,
the empty-run contract (no invented numbers, status ``EMPTY-PENDING``), that the
denominator comes from the pinned carrier and nowhere else, and that the leg
really drives ``net/train_loop.train`` and writes the unchanged journal schema.

``test_run_leg_drives_the_real_training_leg`` is the one that would catch a
script that only *looks* runnable: it executes a leg for real (on a proto config
shrunk to CPU smoke sizes, the module constants monkeypatched) and asserts a
``pretrain-metrics/v1`` journal came out with one row per step.
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

import loss_parity_bf16 as parity  # noqa: E402
import mfu_bf16_protocol as mfu  # noqa: E402

PROTO_CONFIG = CASE_DIR / "net" / "config-proto-micro.json"


def run_cli(script: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        cwd=str(CASE_DIR),
    )


# --------------------------------------------------------------------------- #
# MFU protocol — selftest, plan, denominator, empty-run contract
# --------------------------------------------------------------------------- #


def test_mfu_selftest_is_green() -> None:
    result = run_cli(TOOLS_DIR / "mfu_bf16_protocol.py", "--selftest")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FAIL" not in result.stdout
    assert "PASS" in result.stdout


def test_mfu_plan_is_configs_times_modes() -> None:
    cells = mfu.plan()
    assert len(cells) == len(mfu.CONFIGS) * len(mfu.MODES)
    for name, _cfg, batch, seq in mfu.CONFIGS:
        modes = {c["mode"] for c in cells if c["name"] == name}
        assert modes == {m for m, _ in mfu.MODES}
        assert all(c["seq"] == seq and c["batch"] == batch for c in cells if c["name"] == name)


def test_mfu_configs_are_the_three_declared_shapes() -> None:
    shapes = {(c["config"], c["batch"], c["seq"]) for c in mfu.plan()}
    assert shapes == {
        ("net/config-dense124m.json", 1, 8192),
        ("net/config-dense124m.json", 4, 8192),
        ("net/config.json", 1, 8192),
    }


def test_mfu_denominator_comes_from_the_pinned_carrier() -> None:
    """The MFU denominator is the certified Gemm-peak pin, not a local constant."""
    peaks = mfu.load_peak_tflops()
    assert peaks["bf16"] == 98.2
    assert peaks["fp32"] == 45.2
    assert peaks["source"] == "evidence/gemm_peak_0810.log — git-объект в коммите 42b7408 (сертификация: двойное чтение объекта идентично; fetch верифицирует SHA каждого объекта)"


def test_mfu_missing_pin_leaves_the_denominator_empty(tmp_path: Path) -> None:
    """No pin → no number (never a guessed peak, never a fabricated MFU)."""
    peaks = mfu.load_peak_tflops(tmp_path / "absent.json")
    assert peaks["bf16"] is None and peaks["fp32"] is None
    assert mfu.peak_for_mode("bf16", peaks) is None


def test_mfu_tail_median_drops_the_jit_warmup() -> None:
    assert mfu.tail_median([1.0, 2.0, 3.0, 10.0, 12.0, 11.0]) == 11.0
    assert mfu.tail_median([1.0, 2.0, 3.0]) is None  # all warmup → no verdict
    assert mfu.warmup_median([1.0, 2.0, 3.0, 9.0]) == 2.0


def test_mfu_bf16_modes_use_the_bf16_peak() -> None:
    peaks = {"bf16": 98.2, "fp32": 45.2}
    assert mfu.peak_for_mode("bf16", peaks) == 98.2
    assert mfu.peak_for_mode("bf16+flash", peaks) == 98.2
    assert mfu.peak_for_mode("fp32", peaks) == 45.2


def test_mfu_empty_report_is_pending_without_numbers() -> None:
    report = mfu.build_report([], "EMPTY-PENDING", {"bf16": 98.2, "fp32": 45.2})
    assert report["status"] == "EMPTY-PENDING"
    assert report["cells"] == []
    assert len(report["comparisons"]) == len(mfu.CONFIGS)
    for row in report["comparisons"]:
        for entry in row["modes"].values():
            assert entry["tok_s"] is None and entry["mfu"] is None
    assert report["protocol"]["journal_schema"] == "pretrain-metrics/v1 (не изменяется)"


def test_mfu_cli_writes_pending_report_without_a_gpu(tmp_path: Path) -> None:
    if mfu.gpu_available():
        pytest.skip("на машине есть GPU — контракт EMPTY-PENDING проверяется без него")
    out = tmp_path / "mfu-report.json"
    result = run_cli(TOOLS_DIR / "mfu_bf16_protocol.py", "--out", str(out))
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["schema"] == mfu.REPORT_SCHEMA
    assert report["status"] == "EMPTY-PENDING"
    assert len(report["protocol"]["configs"]) == 3
    assert len(report["protocol"]["modes"]) == 3


def test_mfu_cell_drives_the_real_training_leg(tmp_path: Path, monkeypatch) -> None:
    """The cell runner executes ``train_loop.train`` and writes its journal."""
    monkeypatch.setattr(mfu, "STEPS", 3)
    cell = mfu.run_cell(
        str(PROTO_CONFIG.relative_to(CASE_DIR)), 1, 64, "fp32", tmp_path, steps=3
    )
    assert cell["journal_schema"] == "pretrain-metrics/v1"
    assert cell["steps_done"] == 3
    assert cell["peak_tflops"] == 45.2  # fp32 cell takes the fp32 pin
    rows = [
        json.loads(line)
        for line in (CASE_DIR / cell["journal"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 3
    assert all("tokens_per_sec" in r for r in rows)
    # The tail median exists for a 3-step run only from step `warmup`; with
    # steps == warmup there is no tail, and the protocol says so instead of
    # reporting a warmup number as a result.
    assert cell["tok_s_median_tail"] is None or cell["tok_s_median_tail"] > 0


# --------------------------------------------------------------------------- #
# Loss parity leg — selftest, threshold semantics, empty-run contract
# --------------------------------------------------------------------------- #


def test_parity_selftest_is_green() -> None:
    result = run_cli(TOOLS_DIR / "loss_parity_bf16.py", "--selftest")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FAIL" not in result.stdout


def test_parity_leg_budget_is_50m_tokens_dense124m() -> None:
    assert parity.LEG_CONFIG == "net/config-dense124m.json"
    assert parity.LEG_TOKENS == 50_000_000
    assert parity.leg_steps() == -(-50_000_000 // (parity.LEG_BATCH * parity.LEG_SEQ))


def test_parity_tolerance_is_the_task_threshold() -> None:
    assert parity.BPB_TOLERANCE_DELTA == 0.05


def _cells(fp32_loss: float, bf16_loss: float) -> dict:
    return {
        "fp32": {"loss_median_window": fp32_loss},
        "bf16": {"loss_median_window": bf16_loss},
    }


def test_parity_equal_curves_pass() -> None:
    v = parity.verdict(_cells(3.0, 3.0), 0.25)
    assert v["status"] == "pass" and v["delta_bpb"] == 0.0


def test_parity_at_the_tolerance_passes() -> None:
    """`<=`: the bar itself is a pass (same convention as the roofline gate)."""
    delta_nats = parity.BPB_TOLERANCE_DELTA * 0.6931471805599453 / 0.25
    v = parity.verdict(_cells(3.0, 3.0 + delta_nats), 0.25)
    assert v["status"] == "pass"


def test_parity_beyond_the_tolerance_fails() -> None:
    v = parity.verdict(_cells(3.0, 3.2), 0.25)
    assert v["status"] == "fail"
    assert v["delta_bpb"] > parity.BPB_TOLERANCE_DELTA


def test_parity_better_curve_passes() -> None:
    assert parity.verdict(_cells(3.2, 3.0), 0.25)["status"] == "pass"


def test_parity_without_the_coefficient_is_an_input_error() -> None:
    assert parity.verdict(_cells(3.0, 3.0), None)["status"] == "input-error"


def test_parity_without_a_leg_is_an_input_error() -> None:
    assert parity.verdict({"fp32": {"loss_median_window": 3.0}}, 0.25)["status"] == "input-error"


def test_parity_cli_requires_a_coefficient() -> None:
    result = run_cli(TOOLS_DIR / "loss_parity_bf16.py")
    assert result.returncode == 2
    assert "tokens-per-byte" in result.stderr


def test_parity_cli_writes_pending_report_without_a_gpu(tmp_path: Path) -> None:
    if parity.gpu_available():
        pytest.skip("на машине есть GPU — контракт EMPTY-PENDING проверяется без него")
    out = tmp_path / "loss-parity.json"
    result = run_cli(
        TOOLS_DIR / "loss_parity_bf16.py", "--tokens-per-byte", "0.25", "--out", str(out)
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["schema"] == parity.REPORT_SCHEMA
    assert report["status"] == "EMPTY-PENDING"
    assert report["verdict"] is None
    assert report["leg"]["tokens"] == 50_000_000
    assert set(report["cells"]) == {"fp32", "bf16"}


def test_parity_run_leg_drives_the_real_training_leg(tmp_path: Path, monkeypatch) -> None:
    """A leg really trains and journals — the script is not just a plan."""
    monkeypatch.setattr(parity, "LEG_CONFIG", str(PROTO_CONFIG.relative_to(CASE_DIR)))
    monkeypatch.setattr(parity, "LEG_SEQ", 64)
    monkeypatch.setattr(parity, "LEG_BATCH", 1)
    leg = parity.run_leg("fp32", tmp_path, steps=3)
    assert leg["steps"] == 3 and leg["steps_done"] == 3
    assert leg["tokens"] == 3 * 64
    assert leg["loss_median_window"] is not None
    rows = [
        json.loads(line)
        for line in (CASE_DIR / leg["journal"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 3
    assert all(r.get("schema") == "pretrain-metrics/v1" for r in rows)
