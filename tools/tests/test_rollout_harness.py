"""Тесты агентного harness-лупа роллаутов (ENVIRONMENT-V1 §13, E-1).

Покрытие: детерминизм журнала (AD-4/AD-11), инструменты по отдельности
(кэп read_file, ошибка соответствия edit_file, JSON-вердикт run_gates), лимиты
(attempts / budget / tokens), маскирование assistant_mask, игнор правок после
``finish``, shaping-компоненты Лагуны, MockLM-интеграционный эпизод end-to-end
на задаче ``corruption-l0-00`` (edit_file → run_gates → finish → вердикт).

Все тесты — CPU, без GPU и сети (инструкция E-1); ``run_gates`` в E2E идёт по
существующему пути верификации ``env`` (arch-ml), а при его отсутствии
подменяется детерминированной заглушкой, чтобы тест оставался исполнимым.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1]
CASE_DIR = TOOLS_DIR.parent
for _p in (str(TOOLS_DIR), str(CASE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rollout_harness as rh  # noqa: E402


# ── Фикстуры и помощники ───────────────────────────────────────────────────


class StubVerifier:
    """Детерминированный верификатор для тестов: вердикт = маркер в NOTES.md."""

    def __init__(self, marker: str = "MARKER-EDITED") -> None:
        self.marker = marker

    def _passed(self, workspace: Path) -> bool:
        notes = Path(workspace) / "NOTES.md"
        return notes.is_file() and self.marker in notes.read_text(encoding="utf-8")

    def run_gates(self, workspace: Path) -> dict:
        passed = self._passed(workspace)
        return {
            "passed": passed,
            "violations": [] if passed else [{"rule": "stub", "file": "NOTES.md"}],
            "tests_passed": passed,
            "gates": {"fitness": passed},
        }

    def evaluate(self, workspace: Path, spent_tokens: int) -> rh.EvalResult:
        passed = self._passed(workspace)
        total = 1.0 if passed else 0.0
        return rh.EvalResult(
            passed=passed,
            reward_total=total,
            reward_parts={"total": total},
            violations=[] if passed else [["stub", "NOTES.md"]],
            tests_passed=passed,
        )


def make_workspace(root: Path) -> Path:
    ws = root / "ws"
    (ws / "sub").mkdir(parents=True)
    (ws / "a.txt").write_text("hello world\nsecond line\n", encoding="utf-8")
    (ws / "sub" / "b.txt").write_text("nested\n", encoding="utf-8")
    (ws / "NOTES.md").write_text("MARKER-ORIGINAL\n", encoding="utf-8")
    return ws


def make_spec(**overrides) -> dict:
    spec = {
        "id": "corruption-l0-00",
        "source": "corruption",
        "prompt": "Восстанови гейты кейса.",
        "attempts": 1,
        "budget_seconds": 1800,
        "max_tokens": 131072,
        "seed": 1000,
    }
    spec.update(overrides)
    return spec


def tool_turn(name: str, args: dict) -> str:
    return f'<tool_call>{json.dumps({"name": name, "args": args})}</tool_call>'


FINISH_TURN = tool_turn("finish", {})
EDIT_TURN = tool_turn("edit_file", {"path": "NOTES.md", "old": "MARKER-ORIGINAL", "new": "MARKER-EDITED"})


def run(
    ws: Path, lm: rh.MockLM, workdir: Path, *, spec=None, config=None, **kwargs
) -> rh.EpisodeJournal:
    return rh.run_episode(
        spec or make_spec(),
        ws,
        lm,
        verifier=StubVerifier(),
        config=config,
        workdir=workdir,
        **kwargs,
    )


# ── Инструменты ────────────────────────────────────────────────────────────


def test_list_files_deterministic_tree(tmp_path: Path) -> None:
    ws = make_workspace(tmp_path)
    tools = rh.WorkspaceTools(ws, StubVerifier())
    first = tools.list_files()
    second = tools.list_files()
    assert first == second
    assert "a.txt" in first and "sub/b.txt" in first
    # отсортированность путей — часть детерминизма наблюдения
    lines = [ln.split("\t")[0] for ln in first.splitlines()]
    assert lines == sorted(lines)


def test_read_file_caps_length(tmp_path: Path) -> None:
    ws = make_workspace(tmp_path)
    big = "x" * (rh.READ_CAP_CHARS * 2)
    (ws / "big.txt").write_text(big, encoding="utf-8")
    tools = rh.WorkspaceTools(ws, StubVerifier())
    out = tools.read_file("big.txt")
    assert len(out) < len(big)
    assert "обрезано" in out
    # кэп не превышен (с запасом на пометку обрезки)
    assert len(out) <= rh.READ_CAP_CHARS + 100


def test_edit_file_requires_single_match(tmp_path: Path) -> None:
    ws = make_workspace(tmp_path)
    tools = rh.WorkspaceTools(ws, StubVerifier())
    assert tools.edit_file("a.txt", "hello", "hi") == "ok"
    assert (ws / "a.txt").read_text(encoding="utf-8").startswith("hi world")
    # ноль совпадений
    assert tools.edit_file("a.txt", "нет-такого", "x").startswith("error")
    # два совпадения
    (ws / "dup.txt").write_text("aa\n", encoding="utf-8")
    assert tools.edit_file("dup.txt", "a", "b").startswith("error")
    assert (ws / "dup.txt").read_text(encoding="utf-8") == "aa\n"


def test_edit_file_blocks_escape(tmp_path: Path) -> None:
    ws = make_workspace(tmp_path)
    tools = rh.WorkspaceTools(ws, StubVerifier())
    with pytest.raises(rh.WorkspaceEscape):
        tools.edit_file("../outside.txt", "a", "b")
    with pytest.raises(rh.WorkspaceEscape):
        tools.read_file("/etc/hostname")
    # диспетчер превращает выход за workspace в наблюдение-ошибку, не в сбой
    outcome = tools.dispatch(rh.ToolCall("read_file", {"path": "../x"}, '{"path":"../x"}'))
    assert outcome.body.startswith("error")


def test_parse_turn_handles_braces_in_args() -> None:
    text = (
        "<think>правлю функцию</think>"
        '<tool_call>{"name": "edit_file", "args": {"path": "m.py", "old": "def f() {", "new": "def g() {"}}</tool_call>'
    )
    parsed = rh.parse_assistant_turn(text)
    assert not parsed.parse_error
    assert parsed.think == "правлю функцию"
    assert parsed.call is not None
    assert parsed.call.args["new"] == "def g() {"


def test_parse_turn_reports_errors() -> None:
    assert rh.parse_assistant_turn("просто текст").parse_error
    assert rh.parse_assistant_turn('<tool_call>{"name": "x"</tool_call>').parse_error
    assert rh.parse_assistant_turn('<tool_call>{"n": 1}</tool_call>').parse_error
    assert rh.parse_assistant_turn('<tool_call>{"name": "x", "args": []}</tool_call>').parse_error


def test_run_gates_returns_json_verdict(tmp_path: Path) -> None:
    ws = make_workspace(tmp_path)
    tools = rh.WorkspaceTools(ws, StubVerifier())
    (ws / "NOTES.md").write_text("MARKER-EDITED\n", encoding="utf-8")
    payload = json.loads(tools.run_gates())
    assert payload["passed"] is True
    assert payload["tests_passed"] is True
    assert isinstance(payload["violations"], list)


# ── Детерминизм ────────────────────────────────────────────────────────────


def test_two_runs_same_seed_identical_journal(tmp_path: Path) -> None:
    ws = make_workspace(tmp_path)
    script = [
        tool_turn("list_files", {}),
        tool_turn("read_file", {"path": "a.txt"}),
        EDIT_TURN,
        tool_turn("run_gates", {}),
        FINISH_TURN,
    ]
    j1 = run(ws, rh.MockLM(script), tmp_path / "r1")
    j2 = run(ws, rh.MockLM(script), tmp_path / "r2")
    assert j1.to_json() == j2.to_json()
    assert j1.to_json().encode("utf-8") == j2.to_json().encode("utf-8")
    assert j1.reward == j2.reward


# ── Лимиты ─────────────────────────────────────────────────────────────────


def test_attempts_limit(tmp_path: Path) -> None:
    ws = make_workspace(tmp_path)
    lm = rh.MockLM()  # никогда не завершается (default = list_files)
    prompt_tokens = len(lm.encode(rh.SYSTEM_PROMPT)) + len(lm.encode(make_spec()["prompt"]))
    cfg = rh.EpisodeConfig(max_rollout_tokens=prompt_tokens + 20)
    j = run(ws, lm, tmp_path / "wd", spec=make_spec(attempts=2), config=cfg)
    assert len(j.attempts) == 2
    assert j.attempts_used == 2
    assert j.termination == "tokens"


def test_budget_limit_zero_reward(tmp_path: Path) -> None:
    ws = make_workspace(tmp_path)
    clock_calls = {"n": 0}

    def clock() -> float:
        clock_calls["n"] += 1
        return 0.0 if clock_calls["n"] == 1 else 100.0

    j = rh.run_episode(
        make_spec(budget_seconds=10), ws, rh.MockLM([FINISH_TURN]),
        verifier=StubVerifier(), workdir=tmp_path / "wd", clock=clock,
    )
    assert j.termination == "budget"
    assert j.reward == 0.0
    assert j.attempts[0].turns == 0


def test_token_limit(tmp_path: Path) -> None:
    ws = make_workspace(tmp_path)
    lm = rh.MockLM()
    spec = make_spec()
    limit = len(lm.encode(rh.SYSTEM_PROMPT)) + len(lm.encode(spec["prompt"])) + 5
    j = run(ws, lm, tmp_path / "wd", spec=spec, config=rh.EpisodeConfig(max_rollout_tokens=limit))
    assert j.termination == "tokens"
    a = j.attempts[0]
    assert a.tokens_used >= limit
    assert a.turns >= 1


# ── Маскирование и finish ──────────────────────────────────────────────────


def test_assistant_mask_alignment(tmp_path: Path) -> None:
    ws = make_workspace(tmp_path)
    j = run(ws, rh.MockLM([tool_turn("read_file", {"path": "a.txt"}), FINISH_TURN]), tmp_path / "wd")
    a = j.attempts[0]
    assert len(a.to_dict()["token_ids"]) == len(a.to_dict()["assistant_mask"])
    for seg in a.segments:
        if seg.kind == "assistant":
            assert set(seg.assistant_mask) == {1}
            assert len(seg.assistant_mask) == len(seg.token_ids)
        else:
            assert set(seg.assistant_mask) == {0}
    # промпт и наблюдение не участвуют в лоссе
    assert a.to_dict()["assistant_mask"].count(1) == sum(
        len(s.token_ids) for s in a.segments if s.kind == "assistant"
    )


def test_finish_ignores_later_edits(tmp_path: Path) -> None:
    ws = make_workspace(tmp_path)
    after_finish = tool_turn(
        "edit_file", {"path": "NOTES.md", "old": "MARKER-EDITED", "new": "MARKER-LATE"}
    )
    j = run(ws, rh.MockLM([EDIT_TURN, FINISH_TURN, after_finish]), tmp_path / "wd")
    a = j.attempts[0]
    assert a.finished and a.termination == "finish"
    assert a.turns == 2  # ход после finish не генерировался
    text = (tmp_path / "wd" / "attempt-0" / "NOTES.md").read_text(encoding="utf-8")
    assert "MARKER-EDITED" in text and "MARKER-LATE" not in text
    # оригинал не мутирован (инструменты — по копии)
    assert "MARKER-ORIGINAL" in (ws / "NOTES.md").read_text(encoding="utf-8")


# ── Shaping (arm laguna, ADR-027) ──────────────────────────────────────────


def test_shaping_parse_error_penalty() -> None:
    assert rh.prefix_penalty("laguna", parse_errors=1, tool_calls=5, n_min=2) == pytest.approx(-0.1)
    assert rh.prefix_penalty("laguna", parse_errors=0, tool_calls=5, n_min=2) == 0.0


def test_shaping_give_up_penalty() -> None:
    assert rh.prefix_penalty("laguna", parse_errors=0, tool_calls=1, n_min=2) == pytest.approx(-0.1)
    assert rh.prefix_penalty("laguna", parse_errors=0, tool_calls=2, n_min=2) == 0.0


def test_shaping_both_penalties_stack() -> None:
    assert rh.prefix_penalty("laguna", parse_errors=1, tool_calls=0, n_min=2) == pytest.approx(-0.2)


def test_shaping_frognano_has_no_prefix() -> None:
    assert rh.prefix_penalty("frognano", parse_errors=3, tool_calls=0, n_min=2) == 0.0
    assert rh.episode_reward(
        0.5, arm="frognano", parse_errors=1, tool_calls=0, termination="budget", n_min=2
    ) == 0.5


def test_shaping_budget_and_timeout_zero() -> None:
    for term in ("budget", "timeout", "tokens"):
        assert rh.episode_reward(
            1.5, arm="laguna", parse_errors=0, tool_calls=5, termination=term, n_min=2
        ) == 0.0


def test_shaping_binary_verdict_survives() -> None:
    assert rh.episode_reward(
        1.0, arm="laguna", parse_errors=0, tool_calls=3, termination="finish", n_min=2
    ) == pytest.approx(1.0)
    assert rh.episode_reward(
        0.0, arm="laguna", parse_errors=0, tool_calls=3, termination="finish", n_min=2
    ) == pytest.approx(0.0)


def test_parse_error_and_unknown_tool_in_episode(tmp_path: Path) -> None:
    ws = make_workspace(tmp_path)
    script = [
        "болтовня без вызова",          # parse error
        tool_turn("no_such_tool", {}),  # неизвестный инструмент — не parse error
        tool_turn("list_files", {}),
        tool_turn("read_file", {"path": "a.txt"}),
        FINISH_TURN,
    ]
    j = run(ws, rh.MockLM(script), tmp_path / "wd")
    assert j.attempts[0].parse_errors == 1
    assert j.attempts[0].tool_calls >= 2  # неизвестный инструмент не считается вызовом
    # laguna: parse −0.1, вызовов >= n_min → без give-up
    assert j.reward == pytest.approx(0.0 - 0.1)


def test_jax_adapter_is_interface_only() -> None:
    lm = rh.JaxLM()
    with pytest.raises(NotImplementedError):
        lm.generate([{"role": "user", "content": "x"}], seed=0, max_tokens=8)
    with pytest.raises(NotImplementedError):
        lm.encode("x")


# ── E2E на corruption-l0-00 ────────────────────────────────────────────────


def _build_minimal_case(dst: Path) -> Path:
    """Минимальный кейс (подмножество контура), годный для мутатора/arch-ml."""
    case = dst / "case"
    case.mkdir(parents=True)
    for item in ("CONSTRAINTS.yaml", "ARCHITECTURE-SPINE.md", "AGENTS.md", "README.md"):
        src = CASE_DIR / item
        if src.exists():
            shutil.copy2(src, case / item)
    shutil.copytree(CASE_DIR / "docs" / "adr", case / "docs" / "adr")
    shutil.copytree(CASE_DIR / "model", case / "model")
    (case / "NOTES.md").write_text("MARKER-ORIGINAL\n", encoding="utf-8")
    return case


@pytest.fixture(scope="session")
def corruption_task(tmp_path_factory) -> dict:
    """Задача ``corruption-l0-00`` из детерминированного генератора (seed=0)."""
    from env import generate

    root = tmp_path_factory.mktemp("e1-e2e")
    case = _build_minimal_case(root)
    out = root / "tasks"
    generate.generate(case, out, seed=0, grid=(("corruption", "L0", 1),), holdout_count=0)
    spec = json.loads((out / "public" / "corruption-l0-00.json").read_text(encoding="utf-8"))
    return {"spec": spec, "workspace": out / "public" / "corruption-l0-00"}


def _verifier_for(spec: dict, workspace: Path):
    from env.verifier import arch_ml_available

    if arch_ml_available():
        return rh.EnvVerifier(spec, workspace)
    return StubVerifier()


def test_e2e_mocklm_episode_on_corruption_l0(corruption_task, tmp_path: Path) -> None:
    spec = corruption_task["spec"]
    workspace = corruption_task["workspace"]
    assert spec["id"] == "corruption-l0-00"

    script = [
        tool_turn("list_files", {}),
        tool_turn("read_file", {"path": "NOTES.md"}),
        EDIT_TURN,
        tool_turn("run_gates", {}),
        FINISH_TURN,
    ]
    lm = rh.MockLM(script)
    journal = rh.run_episode(
        spec, workspace, lm,
        verifier=_verifier_for(spec, workspace),
        workdir=tmp_path / "wd",
    )
    a = journal.attempts[0]
    assert a.finished and a.termination == "finish"
    assert a.tool_calls == 4  # list_files, read_file, edit_file, run_gates
    assert journal.verdict_passed in (True, False)
    assert isinstance(journal.reward, float)
    # наблюдение run_gates — JSON-вердикт
    gates_seg = [s for s in a.segments if "run_gates" in s.text or '"passed"' in s.text]
    assert gates_seg
    # правка попала в копию, оригинал не мутирован
    assert "MARKER-EDITED" in (tmp_path / "wd" / "attempt-0" / "NOTES.md").read_text(encoding="utf-8")
    assert "MARKER-ORIGINAL" in (workspace / "NOTES.md").read_text(encoding="utf-8")
    # адресация: после finish ходов нет
    assert a.turns == 5
