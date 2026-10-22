"""S-023/S-024 inproc — тесты встраиваемой библиотеки (CPU)."""

import time
from pathlib import Path

from tools.sensors.inproc import PHASES, DeviceMemory, PhaseTimer
from tools.sensors.subject import build_subject


def _subject(tmp_path: Path) -> dict:
    return build_subject(repo_root=tmp_path, git_sha="a" * 40, dirty=False, device="cpu")


def test_phases_correct_and_sampled(tmp_path):
    subject = _subject(tmp_path)
    timer = PhaseTimer(every=2, out_dir=tmp_path, subject=subject)
    # Шаг 0 — сэмпл.
    timer.begin_step(0)
    with timer.phase("forward", block=lambda: None):
        time.sleep(0.01)
    record = timer.commit(0)
    assert record is not None
    assert "forward" in record["phase_seconds"]
    assert record["phase_seconds"]["forward"] > 0
    # Шаг 1 — не сэмпл (every=2).
    timer.begin_step(1)
    with timer.phase("forward"):
        time.sleep(0.001)
    assert timer.commit(1) is None


def test_timer_writes_contract_records(tmp_path):
    subject = _subject(tmp_path)
    timer = PhaseTimer(every=1, out_dir=tmp_path, subject=subject)
    timer.begin_step(0)
    with timer.phase("data"):
        pass
    timer.commit(0, overhead_pct_value=1.5)
    lines = (tmp_path / "S-023.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2  # step_phase_seconds + timer_overhead_pct
    assert PHASES[0] == "data"


def test_device_memory_unverified_branch(tmp_path):
    class _NoStats:
        def memory_stats(self):
            return None

    mem = DeviceMemory(out_dir=tmp_path, subject=_subject(tmp_path))
    result = mem.sample(device=_NoStats())
    assert result["device_peak_bytes"]["status"] == "unverified"


def test_device_memory_writes_values(tmp_path):
    class _Stats:
        def memory_stats(self):
            return {"peak_bytes_in_use": 100, "bytes_in_use": 50}

    mem = DeviceMemory(out_dir=tmp_path, subject=_subject(tmp_path))
    result = mem.sample(device=_Stats())
    assert result["device_peak_bytes"]["value"] == 100
    assert result["device_bytes_in_use"]["value"] == 50


def test_overhead_pct():
    assert abs(PhaseTimer.overhead_pct(2.0, 2.2) - 10.0) < 1e-9
    assert PhaseTimer.overhead_pct(0.0, 1.0) is None
