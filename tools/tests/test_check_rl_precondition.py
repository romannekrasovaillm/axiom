"""Стражи предусловий RL-прогона (tools/check_rl_precondition.py, ADR-058).

Две части амендмента B проверяются раздельно: ``env`` (C-056, «среда
откалибрована», блокирует RL-расход) и ``k3`` (C-057, «K3-baseline», блокирует
заявление о превосходстве).  Годность отчёта и обязательность ссылки из
precondition проверяются фикстурами во временных каталогах.  Отдельно
фиксируется known-red текущего репозитория: ни среды, ни K3-baseline нет, обе
части обязаны быть красными с разными классификациями.  Сеть и jax не нужны.
"""

from __future__ import annotations

import pytest

from tools import check_rl_precondition as crp


def test_selftest_green():
    assert crp.run_selftest() == 0


@pytest.mark.parametrize("part", [crp.PART_ENV, crp.PART_K3])
def test_empty_evidence_is_red(tmp_path, part):
    ok, notes = crp.evaluate(tmp_path / "evidence", tmp_path / "missing.json", part=part)
    assert ok is False
    assert notes


@pytest.mark.parametrize("part", [crp.PART_ENV, crp.PART_K3])
def test_linked_valid_report_is_green(tmp_path, part):
    evidence, precondition = crp._write_case(tmp_path, part=part)
    ok, notes = crp.evaluate(evidence, precondition, part=part)
    assert ok is True, notes


@pytest.mark.parametrize("part", [crp.PART_ENV, crp.PART_K3])
def test_main_green_with_linked_report(tmp_path, part):
    evidence, precondition = crp._write_case(tmp_path, part=part)
    assert crp.main([
        "--part", part,
        "--evidence-dir", str(evidence),
        "--precondition", str(precondition),
    ]) == crp.EXIT_PASS


@pytest.mark.parametrize(
    ("part", "known_red"),
    [
        (crp.PART_ENV, crp.KNOWN_RED_ENV),
        (crp.PART_K3, crp.KNOWN_RED_K3),
    ],
)
def test_repo_is_known_red(capsys, part, known_red):
    """Сейчас нет ни отчёта среды, ни K3-baseline: обе части красные."""
    assert crp.main(["--part", part]) == crp.EXIT_FAIL
    out = capsys.readouterr().out
    assert known_red in out


def test_parts_known_red_messages_differ(capsys):
    """known-red частей — разные: недостижимый и достижимый долги."""
    assert crp.KNOWN_RED_ENV != crp.KNOWN_RED_K3
    assert crp.main(["--part", crp.PART_ENV]) == crp.EXIT_FAIL
    env_out = capsys.readouterr().out
    assert crp.main(["--part", crp.PART_K3]) == crp.EXIT_FAIL
    k3_out = capsys.readouterr().out
    assert crp.KNOWN_RED_ENV in env_out
    assert crp.KNOWN_RED_K3 in k3_out
    assert crp.KNOWN_RED_K3 not in env_out
    assert crp.KNOWN_RED_ENV not in k3_out


def test_default_part_is_env(tmp_path):
    """Вызов без --part ведёт себя как env (обратная совместимость)."""
    evidence, precondition = crp._write_case(tmp_path, part=crp.PART_ENV, model="own-checkpoint-7b")
    assert crp.main([
        "--evidence-dir", str(evidence),
        "--precondition", str(precondition),
    ]) == crp.EXIT_PASS


def test_env_rejects_k3_report(tmp_path):
    """env не принимает K3-отчёт: чужой baseline не калибрует среду."""
    evidence, precondition = crp._write_case(tmp_path, part=crp.PART_ENV, model="kimi-k3")
    ok, notes = crp.evaluate(evidence, precondition, part=crp.PART_ENV)
    assert ok is False
    assert any("K3" in note for note in notes)


def test_k3_rejects_non_k3_report(tmp_path):
    evidence, precondition = crp._write_case(tmp_path, part=crp.PART_K3, model="own-checkpoint-7b")
    ok, notes = crp.evaluate(evidence, precondition, part=crp.PART_K3)
    assert ok is False
    assert any("k3|kimi" in note for note in notes)


def test_stub_is_red_for_both_parts(tmp_path):
    for part in (crp.PART_ENV, crp.PART_K3):
        evidence, precondition = crp._write_case(tmp_path / part, part=part, model="stub")
        assert crp.evaluate(evidence, precondition, part=part)[0] is False


def test_k3_default_precondition_name(tmp_path):
    """У части k3 свой precondition: evidence/k3-baseline.json."""
    evidence, precondition = crp._write_case(tmp_path, part=crp.PART_K3)
    assert precondition.name == "k3-baseline.json"
    assert "evidence/k3-baseline.json" == crp.PARTS[crp.PART_K3].precondition
    assert "evidence/rl-precondition.json" == crp.PARTS[crp.PART_ENV].precondition
