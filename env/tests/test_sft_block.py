"""Тесты генератора env-блока SFT-микса E-2 (SFT-STAGE.delta §8.5).

Покрытие (deliverable 3): парсинг каждого типа хода пиннутым парсером §13;
детерминизм (два прогона одного seed → байт-идентичный файл); доли S1/S2 и
«≥2 действий ≥50%» на мини-выпуске; страж-интеграция (валидатор ловит
подложенный ``unclosed_think``); кап/снапшот воркспейса не нарушены.

Тесты быстрые: инструменты идут через детерминированный стаб-верификатор (без
arch-ml). Реальная интеграция с arch-ml — отдельный тест, пропускается при
отсутствии бинаря.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

_BLOCK_RE = re.compile(
    r"<think>.*?</think>|<tool_call>.*?</tool_call>|<tool_response>.*?</tool_response>",
    re.S,
)

CASE_DIR = Path(__file__).resolve().parents[2]
if str(CASE_DIR) not in sys.path:
    sys.path.insert(0, str(CASE_DIR))

from env import sft_block as sb  # noqa: E402
from env.util import WORKSPACE_CAP_BYTES, dir_total_bytes  # noqa: E402

import rollout_harness as rh  # noqa: E402


class StubVerifier:
    """Детерминированный верификатор: фиксированный вердикт, без arch-ml."""

    def __init__(self, spec: dict, ws: Path, *_a, **_k) -> None:
        self.spec = spec

    def run_gates(self, workspace: Path) -> dict:
        return {
            "passed": False,
            "violations": [{"rule": "C-003", "file": "docs/adr/x.md"}],
            "tests_passed": True,
            "gates": {"fitness": False, "spine": True, "trace": True},
        }


def _gen(tmp_path: Path, n_s1: int, n_s2: int, seed: int = 1, level: str = "L0"):
    workdir = tmp_path / "wd"
    return sb.generate_block(
        n_s1=n_s1, n_s2=n_s2, seed=seed, s1_level=level,
        verifier_factory=lambda spec, ws: StubVerifier(spec, ws),
        workdir=workdir,
    )


def _arch_ml_available() -> bool:
    from env.verifier import arch_ml_available

    return arch_ml_available()


# ── Формат: разбор каждого типа хода ───────────────────────────────────────


def test_parse_assistant_and_tool_turns(tmp_path: Path):
    records, _ = _gen(tmp_path, n_s1=1, n_s2=1)
    for rec in records:
        roles = [m["role"] for m in rec["messages"]]
        assert roles[0] == "system" and roles[1] == "user"
        # роли чередуются assistant → tool → … → assistant(finish)
        for i, m in enumerate(rec["messages"]):
            if m["role"] == "assistant":
                assert m["assistant_mask"] == 1
                parsed = rh.parse_assistant_turn(m["content"])
                assert not parsed.parse_error, parsed.error
                assert parsed.think, "у каждого хода есть think"
                assert parsed.call is not None
                if i < len(rec["messages"]) - 1:
                    assert parsed.call.name in rh.TOOL_NAMES
                    nxt = rec["messages"][i + 1]
                    assert nxt["role"] == "tool"
                    assert nxt["assistant_mask"] == 0
                    assert nxt["content"].startswith("<tool_response>")
                    assert nxt["content"].rstrip().endswith("</tool_response>")
                else:
                    # последний ход — finish + содержательный ответ (иначе страж
                    # C-044 видит no_answer: текст после снятия блоков пуст)
                    assert parsed.call.name == rh.FINISH_NAME
                    body = m["content"]
                    for block in _BLOCK_RE.finditer(body):
                        body = body.replace(block.group(0), "")
                    assert body.strip(), "финальный ход несёт содержательный ответ"
            else:
                assert m["assistant_mask"] == 0


def test_s1_has_real_edit_and_gates_observation(tmp_path: Path):
    records, _ = _gen(tmp_path, n_s1=1, n_s2=0)
    rec = records[0]
    assert rec["task_type"] == "env_repair_l0"
    edits = [
        json.loads(rh.parse_assistant_turn(m["content"]).call.raw)
        for m in rec["messages"]
        if m["role"] == "assistant"
        and rh.parse_assistant_turn(m["content"]).call
        and rh.parse_assistant_turn(m["content"]).call.name == "edit_file"
    ]
    assert edits, "S1 несёт реальный edit_file"
    # наблюдение правки — реальный результат инструмента
    obs = [m["content"] for m in rec["messages"] if m["role"] == "tool"]
    assert any("ok" in o for o in obs)
    # run_gates наблюдение — реальный JSON-вердикт
    gates = [o for o in obs if '"passed"' in o]
    assert gates and json.loads(
        gates[0].removeprefix("<tool_response>\n").removesuffix("\n</tool_response>")
    )["passed"] in (True, False)


def test_s2_is_clean_and_has_one_gates_call(tmp_path: Path):
    records, _ = _gen(tmp_path, n_s1=0, n_s2=2)
    for rec in records:
        assert rec["task_type"] == "env_verify"
        assert rec["n_tool_calls"] == 1
        calls = [
            rh.parse_assistant_turn(m["content"]).call.name
            for m in rec["messages"]
            if m["role"] == "assistant"
        ]
        assert calls == ["run_gates", rh.FINISH_NAME]
        assert "corruption" not in rec


# ── Детерминизм ────────────────────────────────────────────────────────────


def test_same_seed_byte_identical(tmp_path: Path):
    a, _ = _gen(tmp_path, n_s1=2, n_s2=1, seed=99)
    b, _ = _gen(tmp_path, n_s1=2, n_s2=1, seed=99)
    f1, f2 = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    sb.write_block(a, f1)
    sb.write_block(b, f2)
    assert f1.read_bytes() == f2.read_bytes()


def test_different_seed_differs(tmp_path: Path):
    a, _ = _gen(tmp_path, n_s1=2, n_s2=0, seed=1)
    b, _ = _gen(tmp_path, n_s1=2, n_s2=0, seed=2)
    assert sb.write_block(a, tmp_path / "a.jsonl") != sb.write_block(b, tmp_path / "b.jsonl")


# ── Доли S1/S2 и ≥2 действий ───────────────────────────────────────────────


def test_shares_and_multi_action(tmp_path: Path):
    records, summary = _gen(tmp_path, n_s1=7, n_s2=3, seed=5)
    assert summary["s1"] == 7 and summary["s2"] == 3
    assert summary["s1_share"] == pytest.approx(0.7)
    assert summary["multi_action_share"] >= 0.5
    # S1 несёт ≥2 действий (list/read/edit/gates), S2 — 1 (run_gates)
    assert all(r["n_tool_calls"] >= 4 for r in records if r["scenario"] == "S1")
    assert all(r["n_tool_calls"] == 1 for r in records if r["scenario"] == "S2")


# ── Кап/снапшот ────────────────────────────────────────────────────────────


def test_workspace_snapshot_cap_and_exclusions(tmp_path: Path):
    workdir = tmp_path / "wd"
    sb.generate_block(
        n_s1=1, n_s2=1, seed=3,
        verifier_factory=lambda spec, ws: StubVerifier(spec, ws),
        workdir=workdir,
    )
    ws_dirs = sorted(p for p in workdir.iterdir() if p.is_dir())
    assert ws_dirs
    for ws in ws_dirs:
        assert dir_total_bytes(ws) <= WORKSPACE_CAP_BYTES
        assert not (ws / "env").exists()
        assert not (ws / "data").exists()
        assert not (ws / ".arch-handoff").exists()
        assert not (ws / "evidence").exists()


# ── Страж-интеграция ───────────────────────────────────────────────────────


def test_validator_catches_planted_unclosed_think(tmp_path: Path):
    # корректный блок — зелёный
    records, _ = _gen(tmp_path, n_s1=1, n_s2=1, seed=11)
    good = tmp_path / "good.jsonl"
    sb.write_block(records, good)
    ok, summary = sb.validate_block(good)
    assert ok, summary
    assert summary["unclosed_think"] == 0 and summary["tool_call_in_think"] == 0

    # подложенный незакрытый think (паттерн стража: два открытия, одно закрытие)
    planted = {
        "messages": [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "<think>\nfirst\n<think>\nsecond\n</think>\nFinal answer.",
            },
        ],
        "task_type": "env_verify",
        "source": sb.SOURCE,
    }
    bad = tmp_path / "bad.jsonl"
    sb.write_block([planted], bad)
    ok_bad, summary_bad = sb.validate_block(bad)
    assert not ok_bad
    assert summary_bad["unclosed_think"] >= 1


def test_write_and_read_roundtrip(tmp_path: Path):
    records, summary = _gen(tmp_path, n_s1=1, n_s2=0)
    out = tmp_path / "block.jsonl"
    sha = sb.write_block(records, out)
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(records) == summary["records"]
    assert json.loads(lines[0])["source"] == sb.SOURCE
    assert len(sha) == 64


# ── Реальная интеграция с arch-ml (пиннутый путь верификации) ───────────────


@pytest.mark.skipif(not _arch_ml_available(), reason="arch-ml недоступен (stand gate)")
def test_real_gates_integration_and_determinism(tmp_path: Path):
    kw = dict(n_s1=1, n_s2=1, seed=20261005, bin=None)
    rec_a, sum_a = sb.generate_block(**kw)
    rec_b, _ = sb.generate_block(**kw)
    f1, f2 = tmp_path / "r1.jsonl", tmp_path / "r2.jsonl"
    sb.write_block(rec_a, f1)
    sb.write_block(rec_b, f2)
    assert f1.read_bytes() == f2.read_bytes()
    # наблюдение run_gates — нормализованный путь (нет абсолютного tmp-пути)
    raw = f1.read_text(encoding="utf-8")
    assert str(tmp_path) not in raw
    ok, guard = sb.validate_block(f1)
    assert ok, guard
    assert sum_a["tool_calls"] >= 5


# ── E-2.7: параллелизация генератора (--workers) ───────────────────────────
#
# Воркеры — процессы (:class:`multiprocessing.Pool`), фабрика верификатора
# передаётся в пул через pickle, поэтому здесь она — модульная функция, а не
# lambda. Содержимое каждой траектории зависит только от ``(seed, scenario,
# index)``; шард — лишь исполнитель, порядок записей восстанавливается по
# глобальному индексу, поэтому файл инвариантен по ``workers``.


def _stub_factory(spec: dict, ws: Path) -> StubVerifier:
    """Модульная (picklable) фабрика стаба — для воркеров пула."""
    return StubVerifier(spec, ws)


def _timeout_factory(spec: dict, ws: Path):
    """Стаб с поддельным таймаутом на траектории ``s2-00003``.

    Исключение из воркера (в т.ч. TimeoutError) не должно ронять пул:
    траектория исключается, счётчик ``fails`` растёт, строка — в stderr.
    """
    if Path(ws).name == "s2-00003":
        raise TimeoutError("fake worker timeout")
    return StubVerifier(spec, ws)


def _gen_w(tmp_path: Path, n_s1: int, n_s2: int, seed: int = 1, level: str = "L0",
           workers: int = 1, factory=_stub_factory):
    return sb.generate_block(
        n_s1=n_s1, n_s2=n_s2, seed=seed, s1_level=level, workers=workers,
        verifier_factory=factory, workdir=tmp_path / f"wd-w{workers}",
    )


def test_workers_repeatable_byte_identical(tmp_path: Path):
    """(а) одинаковые seed+workers → байт-идентичные файлы (повторяемость)."""
    a, _ = _gen_w(tmp_path, n_s1=3, n_s2=2, seed=99, workers=2)
    b, _ = _gen_w(tmp_path, n_s1=3, n_s2=2, seed=99, workers=2)
    f1, f2 = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    sb.write_block(a, f1)
    sb.write_block(b, f2)
    assert f1.read_bytes() == f2.read_bytes()


def test_workers_output_invariant_across_n(tmp_path: Path):
    """Разбиение на шарды не меняет содержимое: N=1 и N=2 дают один файл.

    Строгий инвариант E-2.7: содержимое траектории не зависит от ``workers``,
    записи собираются в порядке глобальных индексов.
    """
    r1, _ = _gen_w(tmp_path, n_s1=6, n_s2=2, seed=20261005, workers=1)
    r2, _ = _gen_w(tmp_path, n_s1=6, n_s2=2, seed=20261005, workers=2)
    f1, f2 = tmp_path / "w1.jsonl", tmp_path / "w2.jsonl"
    sb.write_block(r1, f1)
    sb.write_block(r2, f2)
    assert f1.read_bytes() == f2.read_bytes()


def test_workers_partitions_valid(tmp_path: Path):
    """(б) мини-выпуск workers=1 vs workers=2: обе партиции валидны стражем."""
    for workers in (1, 2):
        recs, _ = _gen_w(tmp_path, n_s1=6, n_s2=2, seed=20261005, workers=workers)
        assert len(recs) == 8
        out = tmp_path / f"mini-w{workers}.jsonl"
        sb.write_block(recs, out)
        ok, guard = sb.validate_block(out)
        assert ok, (workers, guard)


def test_record_count_independent_of_workers(tmp_path: Path):
    """(в) число записей = n_s1+n_s2 при любом workers."""
    for workers in (1, 2, 3):
        recs, summary = _gen_w(tmp_path, n_s1=5, n_s2=3, seed=7, workers=workers)
        assert len(recs) == 8
        assert summary["records"] == 8
        assert summary["s1"] == 5 and summary["s2"] == 3
        assert summary["fails"] == 0


def test_worker_exception_does_not_crash_pool(tmp_path: Path, capsys):
    """(г) падение воркера не роняет пул: траектория исключена, fails, stderr-лог."""
    recs, summary = _gen_w(
        tmp_path, n_s1=4, n_s2=4, seed=5, workers=2, factory=_timeout_factory
    )
    assert summary["requested"] == 8
    assert summary["fails"] == 1
    assert len(recs) == 7 and summary["records"] == 7
    err = capsys.readouterr().err
    assert "fail" in err
    assert "TimeoutError" in err
    assert "index=7" in err  # s2-00003 — глобальный индекс 7 (после 4 S1)
    # порядок уцелевших записей сохранён (нет дыры в порядке)
    assert [r["scenario"] for r in recs] == ["S1"] * 4 + ["S2"] * 3
