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

import numpy as np
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


def _write_packed_set(tokens_root: Path, name: str, *, records: int, seq_len: int) -> None:
    """Синтетический packed-набор (``.bin`` + манифест контракта ``tokens/``).

    Строит ровно тот контракт, что читает ``PackedTokenLoader`` (та же раскладка
    записи и спец-токены), чтобы тест пары ног шёл по настоящему загрузчику, а не
    по его подделке.
    """
    from net import train_loop as tl

    out_dir = tokens_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = np.zeros((records, seq_len), dtype=np.uint32)
    rows[:, 0] = tl.BOS_ID
    rows[:, 1:] = np.arange(3, 3 + seq_len - 1, dtype=np.uint32)
    path = out_dir / f"{name}-00000.bin"
    path.write_bytes(rows.tobytes(order="C"))
    manifest = {
        "version": tl.PACKED_MANIFEST_SCHEMA,
        "shard": name,
        "seq_len": seq_len,
        "dtype": "uint32",
        "record_layout": tl.PACKED_RECORD_LAYOUT,
        "bos_id": tl.BOS_ID,
        "eos_id": tl.EOS_ID,
        "pad_id": tl.PAD_ID,
        "tokenizer_hash": "0" * 64,
        "tokenizer": {"file": "tokenizer.model", "vocab_size": 160000},
        "shards": [
            {
                "file": path.name,
                "source": f"{name}-00000.jsonl.zst",
                "source_sha256": "0" * 64,
                "records": records,
                "tokens": records * seq_len,
                "stream_tokens": records * (seq_len - 1),
                "pad_tokens": 0,
                "bytes": path.stat().st_size,
                "sha256": "0" * 64,
            }
        ],
    }
    (out_dir / f"manifest-{name.lower()}.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
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
    pytest.importorskip("jax", reason="прогон ноги требует jax; на этой машине его нет")
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


def _fake_jax(tmp_path: Path, device_repr: str, backend: str) -> Path:
    """A minimal stand-in for ``jax`` so the probe can be exercised without JAX.

    Real JAX is not installed on this machine, and the probe runs in a
    subprocess; putting this module on ``PYTHONPATH`` (which precedes
    site-packages for ``python -c``) lets us drive the *actual* probe code,
    including on a stand that does have JAX.
    """
    pkg = tmp_path / "fakejax"
    pkg.mkdir()
    (pkg / "jax.py").write_text(
        "class _Device:\n"
        f"    def __repr__(self):\n        return {device_repr!r}\n\n"
        "def devices():\n    return [_Device()]\n\n"
        f"def default_backend():\n    return {backend!r}\n",
        encoding="utf-8",
    )
    return pkg


@pytest.mark.parametrize(
    ("device_repr", "backend", "expected"),
    [
        # GB10 stand: device named CudaDevice(id=0) — no 'gpu' substring in repr.
        ("CudaDevice(id=0)", "gpu", True),
        # Repr spelling alone is enough even if the backend string is odd.
        ("GpuDevice(id=0)", "cpu", True),
        # The backend JAX actually selected wins over a repr that says otherwise.
        ("CpuDevice(id=0)", "gpu", True),
        # A genuinely CPU interpreter stays a false negative.
        ("CpuDevice(id=0)", "cpu", False),
    ],
)
def test_mfu_gpu_probe_accepts_cuda_device_reprs(
    tmp_path: Path, monkeypatch, device_repr: str, backend: str, expected: bool
) -> None:
    monkeypatch.setenv("PYTHONPATH", str(_fake_jax(tmp_path, device_repr, backend)))
    assert mfu.gpu_available() is expected


# --------------------------------------------------------------------------- #
# Loss parity leg — selftest, threshold semantics, empty-run contract
# --------------------------------------------------------------------------- #


def test_parity_selftest_is_green() -> None:
    result = run_cli(TOOLS_DIR / "loss_parity_bf16.py", "--selftest")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FAIL" not in result.stdout


def test_parity_defaults_are_the_dense124m_50m_leg() -> None:
    """Дефолты CLI: dense-124m, 50M токенов, поток W — но не захардкоженный объём."""
    assert parity.LEG_CONFIG == "net/config-dense124m.json"
    assert parity.LEG_TOKENS == 50_000_000
    assert parity.DEFAULT_STREAMS == "W"
    assert parity.leg_steps() == -(-50_000_000 // (parity.LEG_BATCH * parity.LEG_SEQ))
    assert parity.leg_steps() == 6104


def test_parity_steps_are_computed_for_the_owner_leg() -> None:
    """Решение владельца: l3full (net/config.json), 5M токенов, batch 1, seq 8192.

    ``ceil(5e6 / (1 * 8192)) = 611`` — число выводится из объёма, а не берётся
    константой: захардкоженные 50M/6104 описывали другую ногу.
    """
    assert parity.leg_steps(5_000_000, 1, 8192) == 611
    plan = parity.build_plan(
        config="net/config.json",
        batch=1,
        seq=8192,
        total_tokens=5_000_000,
        streams=("W", "C"),
        corpus="datasets/axiom-pretrain-l3/tokens-v2",
    )
    assert plan["config"] == "net/config.json"
    assert plan["total_tokens"] == 5_000_000
    assert plan["steps"] == 611
    assert plan["data"] == "corpus"
    assert [cell["mode"] for cell in plan["cells"]] == ["fp32", "bf16"]


def test_parity_plan_is_deterministic_and_the_cells_are_paired() -> None:
    """План клеток детерминирован, а клетки отличаются ТОЛЬКО dtype-гейтом."""
    kwargs = dict(
        config="net/config.json",
        batch=1,
        seq=8192,
        total_tokens=5_000_000,
        streams=("W", "C"),
        corpus="datasets/axiom-pretrain-l3/tokens-v2",
    )
    first = parity.build_plan(**kwargs)
    second = parity.build_plan(**kwargs)
    assert first == second
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    envs = [cell["env"] for cell in first["cells"]]
    assert {key for env in envs for key in env} == {"AXIOM_COMPUTE_DTYPE"}
    assert envs[0] != envs[1]


def test_parity_synthetic_data_never_passes() -> None:
    """Fail-closed: синтетика не несёт вердикта паритета — никогда ``pass``."""
    v = parity.verdict(_cells(3.0, 3.0), 0.25)  # корпус не объявлен → синтетика
    assert v["status"] == "input-error"
    assert "синтетическ" in v["reason"]
    # тот же вход с объявленным корпусом — обычная арифметика допуска
    declared = parity.verdict(_cells(3.0, 3.0), 0.25, real_corpus=True)
    assert declared["status"] == "pass"


def test_parity_plan_from_configs_not_volume(tmp_path) -> None:
    """``--config`` меняет ногу целиком: l3full и dense дают разные планы."""
    l3 = parity.build_plan(
        config="net/config.json", batch=1, seq=8192,
        total_tokens=5_000_000, streams=("W", "C"), corpus="corpus",
    )
    dense = parity.build_plan(
        config="net/config-dense124m.json", batch=1, seq=8192,
        total_tokens=50_000_000, streams=("W", "C"), corpus="corpus",
    )
    assert l3["config"] != dense["config"]
    assert (l3["steps"], dense["steps"]) == (611, 6104)


def test_parity_corpus_legs_see_the_same_batch_sequence(tmp_path, monkeypatch) -> None:
    """Парный дизайн: обе ноги читают ОДНУ последовательность батчей корпуса.

    Упакованный лоадер (``net.train_loop.PackedTokenLoader`` — тот же, что у
    ``tools/pretrain_run.py``) не шаффлит записи, поэтому две сборки на одном
    ``tokens_root`` дают побайтово одинаковый поток: расхождение кривых тогда
    принадлежит dtype-гейту, а не данным.

    ``NET_JAX_BACKEND`` снимается: соседние тест-модули выставляют его в
    окружение сессии, а он заставляет ``net/train_loop`` тянуть jax уже на
    импорте (пиннинг бэкенда, ADR-010). Нога паритета бэкенд не объявляет —
    лоадер обязан собираться без jax.
    """
    monkeypatch.delenv("NET_JAX_BACKEND", raising=False)
    tokens_root = tmp_path / "tokens-v2"
    _write_packed_set(tokens_root, "W", records=8, seq_len=16)
    _write_packed_set(tokens_root, "C", records=8, seq_len=16)
    fp32_leg = parity.corpus_batches(tokens_root, ("W", "C"), 16, 1)
    bf16_leg = parity.corpus_batches(tokens_root, ("W", "C"), 16, 1)
    for _ in range(6):
        assert np.array_equal(next(fp32_leg), next(bf16_leg))


def test_parity_cli_synthetic_run_is_fail_closed(tmp_path, monkeypatch) -> None:
    """CLI без флагов корпуса: 50-шаговый смоук — и вердикт ``input-error``.

    Смоук исполняет механику (по 50 шагов на клетку) и в отчёт не идёт;
    вердикта паритета на синтетике нет — статус ``input-error``, не ``pass``.
    """
    monkeypatch.setattr(parity, "gpu_available", lambda: True)
    calls: list[tuple[str, int]] = []

    def fake_spawn(mode: str, out_dir: Path, steps: int, **kwargs) -> dict:
        calls.append((mode, steps))
        return {"mode": mode, "steps": steps, "loss_median_window": 3.0}

    monkeypatch.setattr(parity, "spawn_leg", fake_spawn)
    out = tmp_path / "loss-parity.json"
    rc = parity.main(["--tokens-per-byte", "0.25", "--out", str(out)])
    report = json.loads(out.read_text(encoding="utf-8"))
    assert calls == [("fp32", parity.SMOKE_STEPS), ("bf16", parity.SMOKE_STEPS)]
    assert parity.SMOKE_STEPS == 50
    assert report["status"] == "input-error"
    assert report["verdict"]["status"] == "input-error"
    assert "синтетическ" in report["verdict"]["reason"]
    assert "cells_measured" not in report  # смоук в отчёт не идёт
    assert rc == 1


def test_parity_tolerance_is_the_task_threshold() -> None:
    assert parity.BPB_TOLERANCE_DELTA == 0.05


def _cells(fp32_loss: float, bf16_loss: float) -> dict:
    return {
        "fp32": {"loss_median_window": fp32_loss},
        "bf16": {"loss_median_window": bf16_loss},
    }


def test_parity_equal_curves_pass() -> None:
    v = parity.verdict(_cells(3.0, 3.0), 0.25, real_corpus=True)
    assert v["status"] == "pass" and v["delta_bpb"] == 0.0


def test_parity_at_the_tolerance_passes() -> None:
    """`<=`: the bar itself is a pass (same convention as the roofline gate)."""
    delta_nats = parity.BPB_TOLERANCE_DELTA * 0.6931471805599453 / 0.25
    v = parity.verdict(_cells(3.0, 3.0 + delta_nats), 0.25, real_corpus=True)
    assert v["status"] == "pass"


def test_parity_beyond_the_tolerance_fails() -> None:
    v = parity.verdict(_cells(3.0, 3.2), 0.25, real_corpus=True)
    assert v["status"] == "fail"
    assert v["delta_bpb"] > parity.BPB_TOLERANCE_DELTA


def test_parity_better_curve_passes() -> None:
    assert parity.verdict(_cells(3.2, 3.0), 0.25, real_corpus=True)["status"] == "pass"


def test_parity_without_the_coefficient_is_an_input_error() -> None:
    assert parity.verdict(_cells(3.0, 3.0), None, real_corpus=True)["status"] == "input-error"


def test_parity_without_a_leg_is_an_input_error() -> None:
    v = parity.verdict({"fp32": {"loss_median_window": 3.0}}, 0.25, real_corpus=True)
    assert v["status"] == "input-error"


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


def test_parity_gpu_probe_accepts_cuda_device_reprs(
    tmp_path: Path, monkeypatch
) -> None:
    """Parity probe parses ``CudaDevice(id=0)`` as a GPU (GB10 stand, 39b9fe1)."""
    monkeypatch.setenv("PYTHONPATH", str(_fake_jax(tmp_path, "CudaDevice(id=0)", "gpu")))
    assert parity.gpu_available() is True


def test_parity_gpu_probe_still_reports_a_cpu_interpreter(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("PYTHONPATH", str(_fake_jax(tmp_path, "CpuDevice(id=0)", "cpu")))
    assert parity.gpu_available() is False


def test_parity_run_leg_drives_the_real_training_leg(tmp_path: Path, monkeypatch) -> None:
    """A leg really trains and journals — the script is not just a plan."""
    pytest.importorskip("jax", reason="прогон ноги требует jax; на этой машине его нет")
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
