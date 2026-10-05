"""Tests of ``tools/check_sft_structure.py`` (SFT-STAGE.delta §8.1, C-044).

The auditor gates the SFT stage before it runs: three structural defect classes
on the assistant trajectory (unclosed ``<think>``, ``<tool_call>`` inside
``<think>``, no answer), a budget-truncated tail that must **not** read as a
defect, saturation by truncation share, fail-closed inputs, and a normalization
that writes a new file with a reversible journal while byte-copying everything
it did not change.
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

import check_sft_structure as sft  # noqa: E402

TOOL = TOOLS_DIR / "check_sft_structure.py"

# A clean trajectory in the real dataset's shape (see module docstring of the
# tool): one assistant message, each step's `<think>` closed before its tool call.
CLEAN = (
    "<think>\nкраткое рассуждение\n</think>\n"
    '<tool_call>{"name": "search_concepts", "query": "x"}</tool_call>\n'
    "<tool_response>\nslug: x\ntype: t\n</tool_response>\n"
    "Финальный ответ по концептам."
)
UNCLOSED = (
    "<think>\nпервое рассуждение без закрытия\n"
    "<think>\nвторое рассуждение\n</think>\n"
    "Финальный ответ."
)
TOOL_IN_THINK = (
    "<think>\nрассуждение\n"
    '<tool_call>{"name": "search_concepts", "query": "y"}</tool_call>\n'
    "продолжение рассуждения\n</think>\n"
    "<tool_response>\nslug: y\n</tool_response>\n"
    "Финальный ответ."
)
NO_ANSWER = (
    "<think>\nрассуждение\n</think>\n"
    '<tool_call>{"name": "search_concepts", "query": "z"}</tool_call>\n'
    "<tool_response>\nslug: z\n</tool_response>"
)
TRUNCATED = "<think>\nрассуждение\n</think>\nначало ответа\n<think>\nоборвано бюджетом"


def write_records(path: Path, assistants: list[str]) -> None:
    lines = []
    for text in assistants:
        record = {
            "messages": [
                {"role": "system", "content": "инструкция с <tool_call> в прозе"},
                {"role": "user", "content": "вопрос"},
                {"role": "assistant", "content": text},
            ],
            "task_type": "explain_relation",
        }
        lines.append(json.dumps(record, ensure_ascii=False) + "\n")
    path.write_text("".join(lines), encoding="utf-8")


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        capture_output=True,
        text=True,
        cwd=str(CASE_DIR),
    )


# --------------------------------------------------------------------------- #
# Clean set / each class / truncation
# --------------------------------------------------------------------------- #


def test_clean_set_is_admissible(tmp_path: Path) -> None:
    path = tmp_path / "clean.jsonl"
    write_records(path, [CLEAN] * 5)
    code, report = sft.run_check([path], strict=True)
    assert code == sft.EXIT_OK
    assert report["verdict"] == "admissible"
    assert report["defects_found"] is False
    assert report["completed_turns"] == 5
    assert report["truncated_turns"] == 0


def test_unclosed_think_is_flagged_alone(tmp_path: Path) -> None:
    path = tmp_path / "unclosed.jsonl"
    write_records(path, [CLEAN] * 4 + [UNCLOSED])
    code, report = sft.run_check([path], strict=True)
    assert code == sft.EXIT_DEFECT
    assert report["classes"]["unclosed_think"]["count"] == 1
    assert report["classes"]["tool_call_in_think"]["count"] == 0
    assert report["classes"]["no_answer"]["count"] == 0


def test_tool_call_in_think_is_flagged_alone(tmp_path: Path) -> None:
    path = tmp_path / "in_think.jsonl"
    write_records(path, [CLEAN] * 4 + [TOOL_IN_THINK])
    code, report = sft.run_check([path], strict=True)
    assert code == sft.EXIT_DEFECT
    assert report["classes"]["tool_call_in_think"]["count"] == 1
    assert report["classes"]["unclosed_think"]["count"] == 0


def test_no_answer_is_flagged_alone(tmp_path: Path) -> None:
    path = tmp_path / "no_answer.jsonl"
    write_records(path, [CLEAN] * 4 + [NO_ANSWER])
    code, report = sft.run_check([path], strict=True)
    assert code == sft.EXIT_DEFECT
    assert report["classes"]["no_answer"]["count"] == 1
    assert report["classes"]["unclosed_think"]["count"] == 0


def test_budget_truncated_tail_is_not_a_defect(tmp_path: Path) -> None:
    path = tmp_path / "truncated.jsonl"
    write_records(path, [CLEAN] * 10 + [TRUNCATED])
    code, report = sft.run_check([path], strict=True)
    assert code == sft.EXIT_OK
    assert report["classes"]["unclosed_think"]["count"] == 0
    assert report["truncated_turns"] == 1
    assert report["completed_turns"] == 10


def test_quoted_tags_in_prose_are_not_defects() -> None:
    """A tool call quoted inline in prose is not a structural defect."""
    text = (
        "<think>\nинструкция говорит: выведи <tool_call> в прозе, и всё\n</think>\n"
        "<tool_response>\nobs\n</tool_response>\nОтвет."
    )
    verdict = sft.analyze_content(text)
    assert verdict["tool_call_in_think"] == 0
    assert verdict["unclosed_think"] == 0


def test_truncation_share_saturates_to_cannot_check(tmp_path: Path) -> None:
    path = tmp_path / "saturated.jsonl"
    write_records(path, [CLEAN] + [TRUNCATED] * 3)
    code, report = sft.run_check([path], strict=True)
    assert code == sft.EXIT_CANNOT
    assert report["saturated"] is True
    assert report["verdict"] == "saturated"


# --------------------------------------------------------------------------- #
# Fail-closed edges
# --------------------------------------------------------------------------- #


def test_no_input_is_cannot_check() -> None:
    assert sft.run_check([])[0] == sft.EXIT_CANNOT


def test_missing_path_is_cannot_check(tmp_path: Path) -> None:
    assert sft.run_check([tmp_path / "нет-такого.jsonl"])[0] == sft.EXIT_CANNOT


def test_unreadable_input_is_cannot_check(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_bytes(b"\xff\xfe not json\n")
    assert sft.run_check([path])[0] == sft.EXIT_CANNOT


def test_defects_without_strict_are_exit_zero(tmp_path: Path) -> None:
    path = tmp_path / "unclosed.jsonl"
    write_records(path, [UNCLOSED])
    code, report = sft.run_check([path], strict=False)
    assert code == sft.EXIT_OK
    assert report["verdict"] == "defects"


# --------------------------------------------------------------------------- #
# Report contents
# --------------------------------------------------------------------------- #


def test_report_carries_counts_shares_and_examples(tmp_path: Path) -> None:
    path = tmp_path / "unclosed.jsonl"
    write_records(path, [CLEAN] * 2 + [UNCLOSED])
    _code, report = sft.run_check([path])
    entry = report["classes"]["unclosed_think"]
    assert entry["count"] == 1
    assert entry["affected_turns"] == 1
    assert entry["share"] == pytest.approx(1 / 3, abs=1e-6)
    example = report["examples"]["unclosed_think"][0]
    assert example["line"] == 3
    assert "рассуждение" in example["snippet"]


# --------------------------------------------------------------------------- #
# Normalization: new file, byte-identical passthrough, reversible journal
# --------------------------------------------------------------------------- #


def _replay(text: str, edits: list[dict]) -> str:
    for edit in reversed(edits):
        start, new = edit["start"], edit["new"]
        assert text[start:start + len(new)] == new
        text = text[:start] + edit["old"] + text[start + len(new):]
    return text


def test_normalize_writes_new_file_and_keeps_source_bytes(tmp_path: Path) -> None:
    src = tmp_path / "src.jsonl"
    write_records(src, [CLEAN, UNCLOSED, TOOL_IN_THINK, NO_ANSWER])
    before = src.read_bytes()
    out = tmp_path / "out.jsonl"
    code, journal = sft.run_normalize(src, out, tmp_path / "out.journal.json")
    assert code == sft.EXIT_OK
    assert src.read_bytes() == before, "исходник обязан остаться байт-в-байт"
    assert journal["source_unchanged"] is True
    assert out.exists()
    src_lines = before.splitlines(keepends=True)
    out_lines = out.read_bytes().splitlines(keepends=True)
    # The clean record (line 1) is copied byte for byte.
    assert out_lines[0] == src_lines[0]


def test_normalize_changes_only_defective_records(tmp_path: Path) -> None:
    src = tmp_path / "src.jsonl"
    write_records(src, [CLEAN, UNCLOSED, TOOL_IN_THINK, NO_ANSWER])
    _code, journal = sft.run_normalize(src, tmp_path / "out.jsonl", tmp_path / "j.json")
    assert journal["records_changed"] == 2  # unclosed + tool-in-think; no_answer unfixable
    out_lines = (tmp_path / "out.jsonl").read_bytes().splitlines(keepends=True)
    src_lines = src.read_bytes().splitlines(keepends=True)
    assert out_lines[0] == src_lines[0]
    assert out_lines[3] == src_lines[3]  # no_answer copied byte for byte


def test_normalize_journal_is_reversible(tmp_path: Path) -> None:
    src = tmp_path / "src.jsonl"
    write_records(src, [UNCLOSED, TOOL_IN_THINK])
    out = tmp_path / "out.jsonl"
    _code, journal = sft.run_normalize(src, out, tmp_path / "j.json")
    out_lines = out.read_bytes().splitlines(keepends=True)
    src_lines = src.read_bytes().splitlines(keepends=True)
    for transformation in journal["transformations"]:
        line = transformation["line"]
        original = json.loads(src_lines[line - 1])
        normalized = json.loads(out_lines[line - 1])
        for message in transformation["messages"]:
            index = message["message_index"]
            assert (
                _replay(normalized["messages"][index]["content"], message["edits"])
                == original["messages"][index]["content"]
            )


def test_normalize_removes_structural_defects(tmp_path: Path) -> None:
    src = tmp_path / "src.jsonl"
    write_records(src, [UNCLOSED, TOOL_IN_THINK])
    out = tmp_path / "out.jsonl"
    sft.run_normalize(src, out, tmp_path / "j.json")
    _code, report = sft.run_check([out], strict=True)
    assert report["classes"]["unclosed_think"]["count"] == 0
    assert report["classes"]["tool_call_in_think"]["count"] == 0


def test_normalize_marks_no_answer_unfixable(tmp_path: Path) -> None:
    src = tmp_path / "src.jsonl"
    write_records(src, [NO_ANSWER])
    _code, journal = sft.run_normalize(src, tmp_path / "out.jsonl", tmp_path / "j.json")
    assert journal["records_changed"] == 0
    assert journal["transformations"][0]["unfixable"].get("no_answer") == 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_cli_selftest_is_green() -> None:
    result = run_cli("--selftest")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FAIL" not in result.stdout
    assert "PASS" in result.stdout


@pytest.mark.skipif(
    not Path("/home/roman/gb10-shared/datasets/sft_train_v12.jsonl").exists(),
    reason="real SFT set is a read-only external mount (AD-6)",
)
def test_real_dataset_probe_reads_the_declared_format() -> None:
    """The declared format has no turn-end sentinels: the probe must not saturate."""
    result = run_cli(
        "--input", "/home/roman/gb10-shared/datasets/sft_train_v12.jsonl",
        "--limit", "200", "--json", "/tmp/axiom-sft-real-probe.json", "--quiet",
    )
    assert result.returncode in (sft.EXIT_OK, sft.EXIT_DEFECT), result.stdout + result.stderr
    report = json.loads(Path("/tmp/axiom-sft-real-probe.json").read_text())
    assert report["saturated"] is False
    assert report["assistant_messages"] == 200
    assert report["records"] == 200
