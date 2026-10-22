"""Тесты ``tools/remat_policy_ab.py`` — воспроизводимый A/B политик (ADR-049).

Проверяется контракт раннера, а не прогон на железе:

* **флаг** — ноги отличаются ровно аргументом ``--remat-policy``, база общая
  (это и есть равный бюджет на уровне плана); база, уже задающая политику, —
  отказ;
* **fail-closed** — неизвестное имя и имя, объявленное, но отсутствующее в
  установленной версии JAX (``hasattr``), отвергаются до старта;
* **первый шаг вне KPI** — KPI-медиана считается без первого (компиляционного)
  шага, и это видно в журнале (``first_step_excluded``);
* **равный бюджет** — свод отказывает в вердикте (``budget-mismatch``), если у
  ног разошлись шаги или суммарные токены;
* **журнал** — единая схема ``axiom/remat-policy-ab/1`` у плана и свода.

Прогоны на GPU тут не запускаются: ``compare`` работает на покадровых журналах.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1]
CASE_DIR = TOOLS_DIR.parent

# Пиннинг бэкенда ДО первого импорта jax (ADR-010): тест остаётся файловым.
os.environ.setdefault("NET_JAX_BACKEND", "cpu")

for _path in (str(CASE_DIR), str(TOOLS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import remat_policy_ab as ab  # noqa: E402

BASE = ["python3", "-m", "tools.pretrain_run", "--run-ref", "remat-ab", "--steps", "20"]


def _metrics(steps: int, *, tokens: int = 8192, tps=(10.0, 20.0, 30.0), loss0: float = 9.0) -> list[dict]:
    """Покадровый журнал: первый шаг намеренно медленный (компиляция)."""
    rows = []
    for i in range(steps):
        speed = tps[i] if i < len(tps) else tps[-1]
        rows.append(
            {
                "step": i + 1,
                "tokens": tokens,
                "tokens_seen": tokens * (i + 1),
                "step_seconds": tokens / speed,
                "tokens_per_sec": speed,
                "loss": loss0 - i * 0.1,
            }
        )
    return rows


def _write(path: Path, rows) -> Path:
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    return path


# ---------------------------------------------------------------------------
# план: две ноги, одна переменная
# ---------------------------------------------------------------------------


def test_plan_legs_differ_only_by_the_policy_flag():
    artifact = ab.plan(BASE, "none", "dots_with_no_batch_dims_saveable")
    baseline = artifact["legs"]["none"]
    candidate = artifact["legs"]["dots_with_no_batch_dims_saveable"]
    assert baseline[: len(BASE)] == candidate[: len(BASE)] == BASE
    assert baseline[-2:] == ["--remat-policy", "none"]
    assert candidate[-2:] == ["--remat-policy", "dots_with_no_batch_dims_saveable"]
    # единственная переменная — ровно хвост из двух токенов
    assert len(baseline) == len(candidate) == len(BASE) + 2


def test_plan_schema_and_single_variable():
    artifact = ab.plan(BASE, "none", "everything_saveable")
    assert artifact["schema"] == ab.SCHEMA == "axiom/remat-policy-ab/1"
    assert artifact["single_variable"] == "--remat-policy"
    assert artifact["equal_budget"] is True


def test_plan_rejects_base_that_already_sets_the_policy():
    with pytest.raises(ValueError, match="уже задаёт --remat-policy"):
        ab.plan([*BASE, "--remat-policy", "none"], "none", "dots_saveable")


def test_plan_rejects_equal_legs():
    with pytest.raises(ValueError, match="сравнивать нечего"):
        ab.plan(BASE, "none", "none")


def test_plan_rejects_unknown_policy():
    with pytest.raises(ValueError, match="неизвестная политика"):
        ab.plan(BASE, "none", "save_everything_please")


def test_plan_rejects_unavailable_declared_policy(monkeypatch):
    import jax

    monkeypatch.delattr(jax.checkpoint_policies, "dots_saveable", raising=False)
    with pytest.raises(ValueError, match="отсутствует в установленной версии JAX"):
        ab.plan(BASE, "none", "dots_saveable")


# ---------------------------------------------------------------------------
# KPI: первый шаг вне KPI
# ---------------------------------------------------------------------------


def test_kpi_excludes_the_first_step():
    rows = _metrics(3, tps=(10.0, 20.0, 30.0))
    kpi = ab.leg_kpi(rows)
    assert kpi["first_step_excluded"] is True
    assert kpi["kpi_rows"] == 2
    assert kpi["kpi_tokens_per_sec"] == 25.0  # медиана (20, 30), не 20 (с первым)


def test_kpi_single_row_cannot_exclude_anything():
    kpi = ab.leg_kpi(_metrics(1, tps=(10.0,)))
    assert kpi["first_step_excluded"] is False
    assert kpi["kpi_tokens_per_sec"] == 10.0


def test_kpi_can_be_taken_raw_for_cross_check():
    rows = _metrics(3, tps=(10.0, 20.0, 30.0))
    assert ab.leg_kpi(rows, exclude_first=False)["kpi_tokens_per_sec"] == 20.0


def test_empty_metrics_file_is_a_refusal(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="пуст"):
        ab.load_metrics(path)


# ---------------------------------------------------------------------------
# свод: равный бюджет, дельта, вердикт
# ---------------------------------------------------------------------------


def test_compare_equal_budget_and_faster_candidate(tmp_path):
    baseline = _write(tmp_path / "b.jsonl", _metrics(3, tps=(10.0, 20.0, 30.0)))
    candidate = _write(tmp_path / "c.jsonl", _metrics(3, tps=(10.0, 40.0, 60.0)))
    artifact = ab.compare(
        baseline_policy="none",
        candidate_policy="dots_with_no_batch_dims_saveable",
        baseline_rows=ab.load_metrics(baseline),
        candidate_rows=ab.load_metrics(candidate),
        baseline_metrics=str(baseline),
        candidate_metrics=str(candidate),
    )
    assert artifact["schema"] == ab.SCHEMA
    assert artifact["equal_budget"] is True
    assert artifact["verdict"] == "faster"
    assert artifact["baseline"]["kpi_tokens_per_sec"] == 25.0
    assert artifact["candidate"]["kpi_tokens_per_sec"] == 50.0
    assert artifact["delta"]["tokens_per_sec_ratio"] == 2.0


def test_compare_flags_budget_mismatch():
    artifact = ab.compare(
        baseline_policy="none",
        candidate_policy="dots_saveable",
        baseline_rows=_metrics(3),
        candidate_rows=_metrics(4),  # лишний шаг — бюджет не тот
    )
    assert artifact["equal_budget"] is False
    assert artifact["verdict"] == "budget-mismatch"


def test_compare_flags_token_budget_mismatch():
    artifact = ab.compare(
        baseline_policy="none",
        candidate_policy="dots_saveable",
        baseline_rows=_metrics(3, tokens=8192),
        candidate_rows=_metrics(3, tokens=4096),
    )
    assert artifact["equal_budget"] is False
    assert artifact["verdict"] == "budget-mismatch"


# ---------------------------------------------------------------------------
# CLI: план и свод через main()
# ---------------------------------------------------------------------------


def test_cli_plan_prints_and_writes_journal(tmp_path, capsys):
    out = tmp_path / "plan.json"
    code = ab.main(
        [
            "plan",
            "--baseline", "none",
            "--candidate", "dots_with_no_batch_dims_saveable",
            "--out", str(out),
            "--", *BASE,
        ]
    )
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["mode"] == "plan"
    assert set(payload["legs"]) == {"none", "dots_with_no_batch_dims_saveable"}
    assert "--remat-policy" in capsys.readouterr().out


def test_cli_compare_end_to_end(tmp_path, capsys):
    baseline = _write(tmp_path / "b.jsonl", _metrics(3, tps=(10.0, 20.0, 30.0)))
    candidate = _write(tmp_path / "c.jsonl", _metrics(3, tps=(10.0, 40.0, 60.0)))
    out = tmp_path / "ab.json"
    code = ab.main(
        [
            "compare",
            "--baseline", "none",
            "--candidate", "dots_with_no_batch_dims_saveable",
            "--baseline-metrics", str(baseline),
            "--candidate-metrics", str(candidate),
            "--out", str(out),
        ]
    )
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["verdict"] == "faster"
    assert payload["equal_budget"] is True
    assert payload["baseline"]["first_step_excluded"] is True


def test_cli_refuses_unknown_policy_with_code_2(capsys):
    code = ab.main(["plan", "--baseline", "none", "--candidate", "nope", "--", *BASE])
    assert code == 2
    assert "ОТКАЗ" in capsys.readouterr().err


def test_cli_run_without_execute_only_prints(capsys):
    code = ab.main(
        ["run", "--baseline", "none", "--candidate", "dots_saveable", "--", *BASE]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "[none]" in out and "[dots_saveable]" in out
    assert "--remat-policy none" in out
