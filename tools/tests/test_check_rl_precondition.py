"""Страж предусловия RL-прогона (tools/check_rl_precondition.py, C-056, ADR-058).

Годность отчёта и обязательность ссылки из precondition проверяются
фикстурами во временных каталогах. Отдельно фиксируется known-red текущего
репозитория: K3-калибровки нет, страж обязан быть красным, а не зелёным.
Сеть и jax не нужны.
"""

from __future__ import annotations

from tools import check_rl_precondition as crp


def test_selftest_green():
    assert crp.run_selftest() == 0


def test_empty_evidence_is_red(tmp_path):
    ok, notes = crp.evaluate(tmp_path / "evidence", tmp_path / "rl-precondition.json")
    assert ok is False
    assert notes


def test_linked_valid_report_is_green(tmp_path):
    evidence, precondition = crp._write_case(tmp_path)
    ok, notes = crp.evaluate(evidence, precondition)
    assert ok is True, notes


def test_repo_is_known_red(capsys):
    """Сейчас K3-калибровки нет: обычный прогон красный с классификацией."""
    assert crp.main([]) == crp.EXIT_FAIL
    out = capsys.readouterr().out
    assert crp.KNOWN_RED in out


def test_main_green_with_linked_report(tmp_path):
    evidence, precondition = crp._write_case(tmp_path)
    assert crp.main([
        "--evidence-dir", str(evidence),
        "--precondition", str(precondition),
    ]) == crp.EXIT_PASS
