"""Tests of ``tools/bpb_report.py``.

The instrument is the dense verification leg's measuring device
(``docs/specs/VERIFICATION-LEG.ru.md``, criterion 1): it converts a pretrain loss
curve into tokenizer-independent BPB and compares it against a known GPT-2 124M
replication within a 10 % tolerance.  The tests pin the three things a measuring
device must get right:

* the **conversion** — BPB is *exactly* ``loss × (tokens/bytes) / ln 2`` (checked
  against a hand calculation);
* the **ratio** — deterministic on a sample, and obtainable both by measuring a
  tokenizer manifest and by the explicit ``--tokens-per-byte`` escape hatch that
  keeps the instrument working without the ``tokenizers`` package;
* the **verdict** — PASS/FAIL on planted curves, boundary at exactly the
  tolerance, and fail-closed (input error ⇒ exit 2, never a silent pass).
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1]
CASE_DIR = TOOLS_DIR.parent
for _p in (str(TOOLS_DIR), str(CASE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bpb_report as bpb  # noqa: E402

TOOL = TOOLS_DIR / "bpb_report.py"

LN2 = math.log(2.0)


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


def write_metrics(path: Path, losses: list[float], *, step_tokens: int = 8192) -> Path:
    rows = [
        {
            "schema": "pretrain-metrics/v1",
            "step": i + 1,
            "loss": loss,
            "tokens": step_tokens,
            "tokens_seen": step_tokens * (i + 1),
        }
        for i, loss in enumerate(losses)
    ]
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    return path


def write_sample(path: Path, text: str = "The quick brown fox. " * 10) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        capture_output=True,
        text=True,
        cwd=str(CASE_DIR),
    )


class FakeTokenizer:
    """Duck-typed ``tokenizers.Tokenizer``: one id per whitespace-separated word."""

    def encode(self, text: str, add_special_tokens: bool = False):  # noqa: ARG002
        ids = list(range(len(text.split())))
        return type("Enc", (), {"ids": ids})()


# --------------------------------------------------------------------------- #
# (а) Контрольная конверсия — ручной расчёт
# --------------------------------------------------------------------------- #


def test_bpb_is_loss_times_ratio_over_ln2() -> None:
    # loss = 8·ln2, ratio = 0.25  →  BPB = 8 · 0.25 = 2.0
    assert bpb.bpb_from_loss(8.0 * LN2, 0.25) == pytest.approx(2.0, abs=1e-12)
    # ratio = 1 and loss = ln2 → BPB = 1 bit per byte (the unit definition).
    assert bpb.bpb_from_loss(LN2, 1.0) == pytest.approx(1.0, abs=1e-12)


def test_bpb_is_linear_in_ratio() -> None:
    assert bpb.bpb_from_loss(LN2, 0.5) == pytest.approx(0.5, abs=1e-12)
    assert bpb.bpb_from_loss(3.28, 0.25) == pytest.approx(3.28 * 0.25 / LN2, rel=1e-12)


def test_report_values_match_hand_calculation(tmp_path: Path) -> None:
    metrics = write_metrics(tmp_path / "m.jsonl", [LN2] * 12)
    sample = write_sample(tmp_path / "s.txt")
    rep = bpb.build_report(metrics, sample, tokens_per_byte=0.25)
    assert rep["bpb"]["median"] == pytest.approx(0.25, rel=1e-12)
    assert rep["curve"][0]["bpb"] == pytest.approx(0.25, rel=1e-12)


def test_reference_constants_and_translation() -> None:
    # The two replications named by the spec are declared, with sources.
    assert bpb.REFERENCES["llm_c_fineweb"]["val_loss_nats_per_token"] == 3.28
    assert bpb.REFERENCES["nanogpt_owt"]["val_loss_nats_per_token"] == 2.85
    for ref in bpb.REFERENCES.values():
        assert ref["source"].startswith("http")
    # The reference is translated by the same formula as our curve.
    ref = bpb.reference_bpb("llm_c_fineweb")
    assert ref["bpb"] == pytest.approx(3.28 * bpb.REFERENCE_TOKENS_PER_BYTE / LN2, rel=1e-12)


# --------------------------------------------------------------------------- #
# (б) Детерминизм коэффициента
# --------------------------------------------------------------------------- #


def test_ratio_from_tokenizer_is_deterministic() -> None:
    texts = ["one two three four", "five six"]
    first = bpb.tokens_per_byte_from_tokenizer(FakeTokenizer(), texts)
    second = bpb.tokens_per_byte_from_tokenizer(FakeTokenizer(), texts)
    assert first == second
    # 6 whitespace tokens over the UTF-8 byte length.
    n_bytes = sum(len(t.encode("utf-8")) for t in texts)
    assert first["tokens"] == 6
    assert first["bytes"] == n_bytes
    assert first["tokens_per_byte"] == pytest.approx(6 / n_bytes, rel=1e-12)


def test_ratio_is_order_invariant() -> None:
    texts = ["a b c", "d e f g h"]
    forward = bpb.tokens_per_byte_from_tokenizer(FakeTokenizer(), texts)
    backward = bpb.tokens_per_byte_from_tokenizer(FakeTokenizer(), list(reversed(texts)))
    assert forward == backward


def test_report_is_deterministic(tmp_path: Path) -> None:
    metrics = write_metrics(tmp_path / "m.jsonl", [3.0] * 12)
    sample = write_sample(tmp_path / "s.txt")
    a = bpb.build_report(metrics, sample, tokens_per_byte=0.25)
    b = bpb.build_report(metrics, sample, tokens_per_byte=0.25)
    a.pop("generated_utc")
    b.pop("generated_utc")
    assert a == b


# --------------------------------------------------------------------------- #
# (в) PASS/FAIL на подложенных кривых
# --------------------------------------------------------------------------- #


def test_curve_matching_reference_passes(tmp_path: Path) -> None:
    ref_loss = bpb.REFERENCES["llm_c_fineweb"]["val_loss_nats_per_token"]
    metrics = write_metrics(tmp_path / "m.jsonl", [ref_loss] * 15)
    sample = write_sample(tmp_path / "s.txt")
    code, rep = bpb.run_report(metrics, sample, tokens_per_byte=bpb.REFERENCE_TOKENS_PER_BYTE)
    assert code == bpb.EXIT_OK
    assert rep["verdict"] == "pass"
    assert rep["delta"]["relative"] == pytest.approx(0.0, abs=1e-9)


def test_curve_far_from_reference_fails(tmp_path: Path) -> None:
    ref_loss = bpb.REFERENCES["llm_c_fineweb"]["val_loss_nats_per_token"]
    metrics = write_metrics(tmp_path / "m.jsonl", [ref_loss * 1.5] * 15)
    sample = write_sample(tmp_path / "s.txt")
    code, rep = bpb.run_report(metrics, sample, tokens_per_byte=bpb.REFERENCE_TOKENS_PER_BYTE)
    assert code == bpb.EXIT_FAIL
    assert rep["verdict"] == "fail"
    assert rep["delta"]["relative"] == pytest.approx(0.5, abs=1e-9)
    assert "FAIL" in rep["message"]


def test_tolerance_boundary_is_inclusive(tmp_path: Path) -> None:
    ref_loss = bpb.REFERENCES["llm_c_fineweb"]["val_loss_nats_per_token"]
    sample = write_sample(tmp_path / "s.txt")
    at = write_metrics(tmp_path / "at.jsonl", [ref_loss * 1.10] * 15)
    over = write_metrics(tmp_path / "over.jsonl", [ref_loss * 1.1001] * 15)
    assert bpb.run_report(at, sample, tokens_per_byte=0.25)[0] == bpb.EXIT_OK
    assert bpb.run_report(over, sample, tokens_per_byte=0.25)[0] == bpb.EXIT_FAIL


def test_window_takes_the_tail(tmp_path: Path) -> None:
    """A late degradation must be caught; a late recovery must pass (last window)."""
    ref_loss = bpb.REFERENCES["llm_c_fineweb"]["val_loss_nats_per_token"]
    sample = write_sample(tmp_path / "s.txt")
    late_bad = write_metrics(tmp_path / "lb.jsonl", [ref_loss] * 20 + [ref_loss * 2] * 20)
    late_good = write_metrics(tmp_path / "lg.jsonl", [ref_loss * 2] * 20 + [ref_loss] * 20)
    assert bpb.run_report(late_bad, sample, tokens_per_byte=0.25)[0] == bpb.EXIT_FAIL
    assert bpb.run_report(late_good, sample, tokens_per_byte=0.25)[0] == bpb.EXIT_OK


def test_at_tokens_restricts_to_equal_corpus_budget(tmp_path: Path) -> None:
    ref_loss = bpb.REFERENCES["llm_c_fineweb"]["val_loss_nats_per_token"]
    sample = write_sample(tmp_path / "s.txt")
    # bad early phase, healthy tail; --at-tokens cuts off before the healthy tail.
    metrics = write_metrics(tmp_path / "m.jsonl", [ref_loss * 2] * 20 + [ref_loss] * 20)
    _, restricted = bpb.run_report(
        metrics, sample, tokens_per_byte=0.25, at_tokens=8192 * 20
    )
    assert restricted["bpb"]["n_rows_selected"] == 20
    assert restricted["verdict"] == "fail"


# --------------------------------------------------------------------------- #
# (г) Границы ошибок: fail-closed
# --------------------------------------------------------------------------- #


def test_missing_inputs_fail_closed(tmp_path: Path) -> None:
    sample = write_sample(tmp_path / "s.txt")
    metrics = write_metrics(tmp_path / "m.jsonl", [3.0] * 12)
    assert bpb.run_report(tmp_path / "nope.jsonl", sample, tokens_per_byte=0.25)[0] == bpb.EXIT_INPUT
    assert bpb.run_report(metrics, tmp_path / "no-sample.txt", tokens_per_byte=0.25)[0] == bpb.EXIT_INPUT


def test_empty_and_corrupt_jsonl_fail_closed(tmp_path: Path) -> None:
    sample = write_sample(tmp_path / "s.txt")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")
    broken = tmp_path / "broken.jsonl"
    broken.write_text("not json\n", encoding="utf-8")
    no_loss = tmp_path / "no-loss.jsonl"
    no_loss.write_text(json.dumps({"step": 1}) + "\n", encoding="utf-8")
    for path in (empty, broken, no_loss):
        code, rep = bpb.run_report(path, sample, tokens_per_byte=0.25)
        assert code == bpb.EXIT_INPUT, path
        assert rep["verdict"] == "input-error"


def test_ratio_source_is_exclusive_and_required(tmp_path: Path) -> None:
    sample = write_sample(tmp_path / "s.txt")
    metrics = write_metrics(tmp_path / "m.jsonl", [3.0] * 12)
    manifest = tmp_path / "tokenizer-manifest.json"
    manifest.write_text(json.dumps({"tokenizer": {"file": "tokenizer.model"}}), encoding="utf-8")
    # both
    assert bpb.run_report(
        metrics, sample, tokens_per_byte=0.25, tokenizer_manifest=manifest
    )[0] == bpb.EXIT_INPUT
    # neither
    assert bpb.run_report(metrics, sample)[0] == bpb.EXIT_INPUT
    # non-positive explicit ratio
    assert bpb.run_report(metrics, sample, tokens_per_byte=0.0)[0] == bpb.EXIT_INPUT


def test_window_below_minimum_fails_closed(tmp_path: Path) -> None:
    sample = write_sample(tmp_path / "s.txt")
    metrics = write_metrics(tmp_path / "m.jsonl", [3.0] * 12)
    assert bpb.run_report(
        metrics, sample, tokens_per_byte=0.25, window=bpb.MIN_WINDOW - 1
    )[0] == bpb.EXIT_INPUT


def test_unknown_reference_fails_closed(tmp_path: Path) -> None:
    sample = write_sample(tmp_path / "s.txt")
    metrics = write_metrics(tmp_path / "m.jsonl", [3.0] * 12)
    assert bpb.run_report(
        metrics, sample, tokens_per_byte=0.25, reference="nope"
    )[0] == bpb.EXIT_INPUT


# --------------------------------------------------------------------------- #
# (д) Измерение коэффициента по манифесту токенизатора
# --------------------------------------------------------------------------- #


def _make_manifest(tmp_path: Path):
    """A real ``tokenizers`` artifact + its manifest, or ``None`` if unavailable."""
    try:
        from tokenizers import Tokenizer, models, pre_tokenizers
    except ImportError:  # pragma: no cover — environment without the package
        return None
    vocab = {"[UNK]": 0, "the": 1, "quick": 2, "brown": 3, "fox": 4}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    artifact = tmp_path / "tokenizer.model"
    tok.save(str(artifact))
    digest = bpb._file_sha256(artifact)
    manifest = tmp_path / "tokenizer-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": "axiom-pretrain-tokenizer/1",
                "tokenizer": {"file": "tokenizer.model", "format": "tokenizers-json/v1"},
                "tokenizer_hash": digest,
                "vocab_size": len(vocab),
            }
        ),
        encoding="utf-8",
    )
    return manifest, artifact


def test_manifest_measures_ratio_and_is_deterministic(tmp_path: Path) -> None:
    made = _make_manifest(tmp_path)
    if made is None:
        pytest.skip("tokenizers package unavailable")
    manifest, _ = made
    sample = write_sample(tmp_path / "s.txt", "the quick brown fox the quick fox")
    text = sample.read_text(encoding="utf-8")
    a = bpb.build_report(
        write_metrics(tmp_path / "m.jsonl", [3.0] * 12), sample,
        tokenizer_manifest=manifest,
    )
    b = bpb.build_report(
        write_metrics(tmp_path / "m.jsonl", [3.0] * 12), sample,
        tokenizer_manifest=manifest,
    )
    a.pop("generated_utc")
    b.pop("generated_utc")
    assert a == b
    cal = a["calibration"]
    assert cal["source"] == "tokenizer-manifest"
    # WordLevel: 7 known whitespace tokens over the text's bytes.
    assert cal["tokens"] == 7
    assert cal["bytes"] == len(text.encode("utf-8"))
    assert cal["tokens_per_byte"] == pytest.approx(cal["tokens"] / cal["bytes"], rel=1e-12)


def test_manifest_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    made = _make_manifest(tmp_path)
    if made is None:
        pytest.skip("tokenizers package unavailable")
    manifest, artifact = made
    artifact.write_bytes(artifact.read_bytes() + b"x")  # tamper after packing
    sample = write_sample(tmp_path / "s.txt")
    metrics = write_metrics(tmp_path / "m.jsonl", [3.0] * 12)
    code, rep = bpb.run_report(metrics, sample, tokenizer_manifest=manifest)
    assert code == bpb.EXIT_INPUT
    assert "подменён" in rep["error"]


# --------------------------------------------------------------------------- #
# (е) CLI-контракт
# --------------------------------------------------------------------------- #


def test_cli_writes_report_and_exit_codes(tmp_path: Path) -> None:
    ref_loss = bpb.REFERENCES["llm_c_fineweb"]["val_loss_nats_per_token"]
    sample = write_sample(tmp_path / "s.txt")
    good = write_metrics(tmp_path / "good.jsonl", [ref_loss] * 15)
    bad = write_metrics(tmp_path / "bad.jsonl", [ref_loss * 2] * 15)

    out_ok = tmp_path / "ok.json"
    proc = run_cli(
        "--loss-jsonl", str(good), "--sample-texts", str(sample),
        "--tokens-per-byte", "0.25", "--out", str(out_ok), "--quiet",
    )
    assert proc.returncode == bpb.EXIT_OK, proc.stderr
    report = json.loads(out_ok.read_text(encoding="utf-8"))
    assert report["schema"] == bpb.REPORT_SCHEMA
    assert report["verdict"] == "pass"
    assert report["exit_code"] == bpb.EXIT_OK

    out_bad = tmp_path / "bad.json"
    proc = run_cli(
        "--loss-jsonl", str(bad), "--sample-texts", str(sample),
        "--tokens-per-byte", "0.25", "--out", str(out_bad), "--quiet",
    )
    assert proc.returncode == bpb.EXIT_FAIL

    proc = run_cli(
        "--loss-jsonl", str(tmp_path / "nope.jsonl"), "--sample-texts", str(sample),
        "--tokens-per-byte", "0.25", "--out", str(tmp_path / "x.json"), "--quiet",
    )
    assert proc.returncode == bpb.EXIT_INPUT


def test_cli_requires_core_flags() -> None:
    proc = run_cli("--tokens-per-byte", "0.25")
    assert proc.returncode == 2  # argparse error
    assert "--loss-jsonl" in proc.stderr


def test_cli_selftest_green() -> None:
    proc = run_cli("--selftest")
    assert proc.returncode == bpb.EXIT_OK, proc.stdout + proc.stderr
    assert "FAIL:" not in proc.stdout
