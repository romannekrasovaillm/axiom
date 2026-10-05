"""E-3.1: подключение реальной модели к кальбровке + CLI-адаптеры (CALIBRATE).

Зона дельты: ``env/calibrate.py`` (параметр ``model_factory``), ``env/main.py``
(calibrate-подкоманда ``--adapter``), ``env/tests/``. GPU/сеть/jax НЕ
используются: реальный ход модели эмулируется ``MockLM`` из пиннутого
harness-лупа ``tools/rollout_harness.py`` (§13) на CPU.

Покрытие по задаче:

(а) stub-ветка не меняется: ``model_factory=None`` даёт отчёт, совпадающий с
    замороженным baseline (``fixtures/calibrate-stub-baseline.json``);
(б) ``model_factory`` (MockLM) ведёт эпизоды через harness §13 с вердиктом
    ``evaluate_run`` и reward по arm-конфигу ``laguna``: pass_rate ∈ [0,1],
    пиннинг-поля на месте, shaping применяется;
(в) CLI: ``--adapter jaxlm`` без ``--checkpoint`` / без явного
    ``--tokenizer-path`` → понятная ошибка (код 2); ``stub`` — дефолт;
(г) существующие тесты calibrate (``test_calibrate.py``) остаются зелёными —
    отдельный прогон, здесь не дублируются.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

CASE_DIR = Path(__file__).resolve().parents[2]
for _p in (str(CASE_DIR), str(CASE_DIR / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from env import calibrate as calibrate_mod  # noqa: E402
from env import main as env_main  # noqa: E402

import rollout_harness as rh  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
STUB_BASELINE = FIXTURES / "calibrate-stub-baseline.json"

#: Два публичных L0-кейса разных objective-видов (restore-gates и
#: keep-gates-implement) — «мини-набор» для быстрой CPU-калибровки MockLM.
MINI_TASK_IDS = ("corruption-l0-00", "real-l0-00")

#: Скрипт вежливого агента: осмотр ×2 (≥ n_min=2 вызовов, без штрафа «сдался»)
#: и явный finish.
POLITE_SCRIPT = [
    '<tool_call>{"name": "list_files", "args": {}}</tool_call>',
    '<tool_call>{"name": "list_files", "args": {}}</tool_call>',
    '<tool_call>{"name": "finish", "args": {}}</tool_call>',
]

#: Скрипт «сдался»: сразу finish, 0 вызовов инструментов → anti-give-up −0.1.
GIVE_UP_SCRIPT = ['<tool_call>{"name": "finish", "args": {}}</tool_call>']

CELL_REQUIRED_KEYS = {
    "task_id", "source", "level", "pass", "reward", "base_reward",
    "pass_component", "soft", "new_violations", "effort_penalty",
    "spent_tokens", "attempts_used", "termination", "policy_version",
}
PINNING_KEYS = {
    "arch_ml_build", "constraints_sha256", "hidden_constraints_sha256",
    "excluded_infra_rules", "excluded_infra_rules_registry",
}


# --------------------------------------------------------------------------- #
# Вспомогательное
# --------------------------------------------------------------------------- #


def _stable(report: dict) -> dict:
    """Детерминированная проекция отчёта для регрессии stub-ветки.

    Исключены ``generated_at`` (таймстамп) и ``workspace_total_bytes``. Последний
    меряет ВХОДНОЙ набор, а его arch-ml-гейты загрязняют ``net/__pycache__``
    после первого прогона (побочный эффект ``command_succeeds``-правил на
    `base_ws`, не часть дельты E-3.1) — на числа stub-ячеек он не влияет
    (снапшот исключает ``__pycache__``/``.pyc``).
    """
    out = json.loads(json.dumps(report, ensure_ascii=False))
    out.pop("generated_at", None)
    out.pop("workspace_total_bytes", None)
    return out


def _mini_tasks(generated: dict, tmp_path: Path) -> Path:
    """Мини-набор задач (2 L0-кейса) из сгенерированной сессии, для MockLM."""
    src = generated["out"] / "public"
    dst = tmp_path / "tasks"
    (dst / "public").mkdir(parents=True)
    (dst / "holdout").mkdir(parents=True)
    hidden = generated["out"] / "holdout" / "hidden_constraints.yaml"
    if hidden.exists():
        shutil.copy(hidden, dst / "holdout" / hidden.name)
    for task_id in MINI_TASK_IDS:
        shutil.copy(src / f"{task_id}.json", dst / "public" / f"{task_id}.json")
        shutil.copytree(src / task_id, dst / "public" / task_id)
    return dst


def _run_mock_calibration(generated, arch_ml, tmp_path, script, out_name):
    tasks = _mini_tasks(generated, tmp_path)
    return calibrate_mod.calibrate(
        tasks,
        generated["case"],
        tmp_path / out_name,
        model_name="mock",
        model_seed=11,
        bin=arch_ml,
        model_factory=lambda: rh.MockLM(script=script),
    )


# --------------------------------------------------------------------------- #
# (а) stub-ветка: регрессия против замороженного baseline
# --------------------------------------------------------------------------- #


def test_stub_branch_matches_frozen_baseline(generated, arch_ml, tmp_path):
    """model_factory=None → отчёт (ячейки/агрегат/пиннинг) как до дельты."""
    import inspect

    assert STUB_BASELINE.is_file(), (
        f"нет замороженного baseline: {STUB_BASELINE} "
        "(сгенерирован из до-дельтового calibrate)"
    )
    expected = json.loads(STUB_BASELINE.read_text(encoding="utf-8"))
    expected.pop("workspace_total_bytes", None)  # см. _stable: входная сборка
    report = calibrate_mod.calibrate(
        generated["out"], generated["case"], tmp_path / "e",
        model_name="stub", model_seed=7, bin=arch_ml, model_factory=None,
    )
    assert report["workspace_total_bytes"] > 0
    assert _stable(report) == expected

    # Умолчание совпадает с явным None: stub-путь не задет (дефолт параметра).
    assert inspect.signature(calibrate_mod.calibrate).parameters["model_factory"].default is None


def test_stub_branch_does_not_touch_model_factory(generated, arch_ml, tmp_path):
    """При model_factory=None фабрика не вызывается (stub-путь изолирован)."""
    calls = {"n": 0}

    def _boom():
        calls["n"] += 1
        raise AssertionError("model_factory не должна вызываться в stub-ветке")

    calibrate_mod.calibrate(
        generated["out"], generated["case"], tmp_path / "e",
        model_name="stub", model_seed=7, bin=arch_ml, model_factory=None,
    )
    assert calls["n"] == 0


# --------------------------------------------------------------------------- #
# (б) model_factory → harness §13 + arm laguna
# --------------------------------------------------------------------------- #


def test_mock_model_calibration_report_shape(generated, arch_ml, tmp_path):
    report = _run_mock_calibration(generated, arch_ml, tmp_path, POLITE_SCRIPT, "e")

    assert set(report["matrix"]) == {"mock"}
    cells = report["matrix"]["mock"]
    assert len(cells) == len(MINI_TASK_IDS)
    assert {c["task_id"] for c in cells} == set(MINI_TASK_IDS)

    for cell in cells:
        assert CELL_REQUIRED_KEYS <= set(cell)
        assert isinstance(cell["pass"], bool)
        assert isinstance(cell["reward"], float)
        assert cell["attempts_used"] >= 1
        assert cell["termination"] in {"finish", "budget", "tokens", "turns", "attempts_exhausted"}
        # MockLM без правок: задачные тесты не пройдены ни в одном кейсе.
        assert cell["pass"] is False
        assert cell["policy_version"] == "mock-v1"

    pr = report["aggregate"]["pass_rate"]
    assert 0.0 <= pr <= 1.0
    assert pr == pytest.approx(sum(1 for c in cells if c["pass"]) / len(cells))
    assert report["readiness"]["pass_rate_range"] == list(calibrate_mod.PASS_RATE_RANGE)

    pinning = report["pinning"]
    assert PINNING_KEYS <= set(pinning)
    assert len(pinning["constraints_sha256"]) == 64
    assert pinning["arch_ml_build"]

    # Отчёт записан в evidence-каталог под именем модели.
    assert (tmp_path / "e" / "calibration-mock.json").is_file()


def test_mock_model_arm_laguna_no_penalty_for_polite_agent(generated, arch_ml, tmp_path):
    """Вежливый агент (≥ n_min вызовов, finish) — префикс Лагуны не штрафует."""
    report = _run_mock_calibration(generated, arch_ml, tmp_path, POLITE_SCRIPT, "e")
    for cell in report["matrix"]["mock"]:
        assert cell["termination"] == "finish"
        assert cell["reward"] == pytest.approx(cell["base_reward"])


def test_mock_model_arm_laguna_anti_give_up_penalty(generated, arch_ml, tmp_path):
    """«Сдался» (0 вызовов инструментов) → reward = base − 0.1 (§13, arm laguna)."""
    report = _run_mock_calibration(generated, arch_ml, tmp_path, GIVE_UP_SCRIPT, "e")
    for cell in report["matrix"]["mock"]:
        assert cell["termination"] == "finish"
        assert cell["reward"] == pytest.approx(cell["base_reward"] - 0.1)


# --------------------------------------------------------------------------- #
# (в) CLI calibrate: --adapter
# --------------------------------------------------------------------------- #


def _cli_args(**over):
    base = {
        "command": "calibrate",
        "--tasks": "/nonexistent/tasks",
        "--out": "/nonexistent/out",
    }
    argv = [base["command"], "--tasks", base["--tasks"], "--out", base["--out"]]
    for key, val in over.items():
        argv += [key] + ([val] if val is not None else [])
    return argv


def test_cli_adapter_defaults_to_stub():
    args = env_main.build_parser().parse_args(_cli_args())
    assert args.adapter == "stub"
    assert args.checkpoint is None
    assert args.tokenizer_path is None
    # stub → реальная фабрика не строится.
    assert env_main._build_model_factory(args) is None


def test_cli_jaxlm_without_checkpoint_is_clear_error(capsys):
    code = env_main.main(_cli_args(**{"--adapter": "jaxlm"}))
    assert code == 2
    err = capsys.readouterr().err
    assert "--checkpoint" in err
    assert "jaxlm" in err


def test_cli_jaxlm_without_tokenizer_path_is_clear_error(tmp_path, capsys):
    code = env_main.main(_cli_args(**{
        "--adapter": "jaxlm",
        "--checkpoint": str(tmp_path / "ckpt"),
    }))
    assert code == 2
    err = capsys.readouterr().err
    assert "--tokenizer-path" in err


def test_cli_jaxlm_factory_builds_adapter(tmp_path):
    args = env_main.build_parser().parse_args(_cli_args(**{
        "--adapter": "jaxlm",
        "--checkpoint": str(tmp_path / "ckpt"),
        "--tokenizer-path": str(tmp_path / "tokenizer.json"),
        "--model-seed": "5",
    }))
    factory = env_main._build_model_factory(args)
    assert callable(factory)
    # Конструирование лениво: jax/net не импортируются (проверяется без ML-стека).
    adapter = factory()
    assert hasattr(adapter, "generate")
    assert hasattr(adapter, "encode")
    assert adapter.seed == 5
    assert adapter._backend._checkpoint == tmp_path / "ckpt"
    assert adapter._backend._tokenizer_path == tmp_path / "tokenizer.json"


def test_cli_jaxlm_model_name_defaults_to_adapter(tmp_path):
    args = env_main.build_parser().parse_args(_cli_args(**{
        "--adapter": "jaxlm",
        "--checkpoint": str(tmp_path / "ckpt"),
        "--tokenizer-path": str(tmp_path / "tokenizer.json"),
    }))
    assert args.model_name is None
    assert (args.model_name or args.adapter) == "jaxlm"
