"""Тесты поведенческого стража AD-2 — C-039 (``tools/check_reward_isolation.py``).

Страж решает, попадёт ли LLM-судья в контур награды RL. Ошибка в нём в одну
сторону опаснее: ложный PASS оставляет reward hacking незамеченным. Поэтому
закреплены оба полюса — чистый контур (PASS) и три разные формы нарушения
(прямой импорт судьи, LLM-клиент, внешний API), — а также запрет ложного PASS
при ненайденном контуре («НЕ ПРОВЕРЕНО»).

Обязательные сценарии (из постановки дельты):
  (i)   чистая фикстура (награда без LLM)                        -> PASS;
  (ii)  фикстура с импортом LLM/судьи в контур награды           -> FAIL;
  (iii) отсутствие контура награды                               -> FAIL «НЕ ПРОВЕРЕНО».
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import check_reward_isolation as guard  # noqa: E402  (путь добавляется выше)

CASE_DIR = Path(__file__).resolve().parents[2]

CLEAN_REWARD = "def compute(passed):\n    return 1.0 if passed else 0.0\n"


# --- построение фикстурного кейса -------------------------------------------


def _write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _clean_case(root: Path, extra: dict[str, str] | None = None) -> Path:
    """Кейс с детерминированной наградой в пакете ``env/``."""
    _write(root, "env/__init__.py", '"""Среда."""\n')
    _write(root, "env/reward.py", CLEAN_REWARD)
    for rel, text in (extra or {}).items():
        _write(root, rel, text)
    return root


def _run(root: Path, capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    code = guard.main(["--root", str(root)])
    return code, capsys.readouterr().out


def _codes(root: Path) -> set[str]:
    return {finding.code for finding in guard.build_report(root).findings}


# --- (i) чистая награда без LLM ---------------------------------------------


def test_clean_reward_circuit_passes(tmp_path: Path, capsys) -> None:
    """(i) награда без LLM-вызовов — зелёный вердикт, контур напечатан."""
    case = _clean_case(tmp_path)

    report = guard.build_report(case)
    code, out = _run(case, capsys)

    assert report.ok, [finding.message for finding in report.findings]
    assert report.exit_code == guard.EXIT_PASS
    assert code == 0
    assert [item.rel for item in report.core] == ["env/__init__.py", "env/reward.py"]
    assert "env/reward.py" in out
    assert "Итог: PASS" in out


def test_marker_basis_is_printed(tmp_path: Path, capsys) -> None:
    """(c) основание включения в контур печатается, а не подразумевается."""
    case = _clean_case(tmp_path)

    _, out = _run(case, capsys)

    assert "Сиды контура" in out
    assert "маркер имени" in out
    assert "Файлы контура награды" in out


# --- (ii) судья и LLM-клиенты в контуре награды ------------------------------


def test_judge_import_in_reward_path_fails(tmp_path: Path, capsys) -> None:
    """(ii) награда импортирует судейский модуль — красный вердикт."""
    case = _clean_case(
        tmp_path,
        {
            "env/reward.py": "from .judge import score\n\n\n" + CLEAN_REWARD,
            "env/judge.py": '"""LLM-судья: калиброванная метрика."""\n\n\n'
            "def score(x):\n    return 0.5\n",
        },
    )

    report = guard.build_report(case)
    code, out = _run(case, capsys)

    assert not report.ok
    assert "REWARD-PATH-JUDGE-IMPORT" in _codes(case)
    assert report.exit_code == guard.EXIT_VIOLATION
    assert code == guard.EXIT_VIOLATION
    assert "Итог: FAIL" in out


def test_llm_client_import_in_reward_path_fails(tmp_path: Path) -> None:
    """(ii) награда импортирует LLM-клиент — красный вердикт."""
    case = _clean_case(
        tmp_path, {"env/reward.py": "import openai\n\n\n" + CLEAN_REWARD}
    )

    assert "REWARD-PATH-LLM-IMPORT" in _codes(case)
    assert guard.build_report(case).exit_code == guard.EXIT_VIOLATION


def test_external_http_client_in_reward_path_fails(tmp_path: Path) -> None:
    """(ii) награда ходит во внешний API по HTTP — красный вердикт."""
    case = _clean_case(
        tmp_path, {"env/reward.py": "import requests\n\n\n" + CLEAN_REWARD}
    )

    assert "REWARD-PATH-EXTERNAL-API" in _codes(case)
    assert guard.build_report(case).exit_code == guard.EXIT_VIOLATION


def test_provider_credential_in_reward_path_fails(tmp_path: Path) -> None:
    """(ii) ключ провайдера LLM в коде награды — красный вердикт."""
    case = _clean_case(
        tmp_path,
        {
            "env/reward.py": "import os\n\n"
            'KEY = os.environ["OPENAI_API_KEY"]\n\n\n' + CLEAN_REWARD
        },
    )

    assert "REWARD-PATH-PROVIDER-CREDENTIAL" in _codes(case)


def test_indirect_llm_import_through_dependency_is_caught(tmp_path: Path) -> None:
    """(a) судейский вызов за помощником из другого пакета не прячется."""
    case = _clean_case(
        tmp_path,
        {
            "env/reward.py": "from helpers.scoring import compute_score\n\n\n"
            "def compute(x):\n    return compute_score(x)\n",
            "helpers/__init__.py": "",
            "helpers/scoring.py": "import anthropic\n\n\n"
            "def compute_score(x):\n    return anthropic.messages(x)\n",
        },
    )

    report = guard.build_report(case)

    assert "REWARD-PATH-LLM-IMPORT" in {f.code for f in report.findings}
    assert "helpers/scoring.py" in {item.rel for item in report.dependencies}
    # нарушение найдено в зависимости контура, а не в самом контуре
    assert {f.rel for f in report.findings} == {"helpers/scoring.py"}


def test_network_cli_call_in_reward_path_fails(tmp_path: Path) -> None:
    """(a) внешний API через запуск процесса (curl) — красный вердикт."""
    case = _clean_case(
        tmp_path,
        {
            "env/reward.py": "import subprocess\n\n\n"
            "def compute(x):\n"
            '    subprocess.run(["curl", "https://api.example.com/score"])\n'
            "    return 0.0\n"
        },
    )

    assert "REWARD-PATH-NETWORK-CALL" in _codes(case)


def test_dynamic_import_of_llm_client_fails(tmp_path: Path) -> None:
    """(a) динамический импорт LLM-клиента — красный вердикт."""
    case = _clean_case(
        tmp_path,
        {
            "env/reward.py": "import importlib\n\n\n"
            "def compute(x):\n"
            '    importlib.import_module("openai")\n'
            "    return 0.0\n"
        },
    )

    assert "REWARD-PATH-DYNAMIC-IMPORT" in _codes(case)


def test_judge_module_outside_circuit_is_allowed(tmp_path: Path) -> None:
    """ADR-002: судья как отдельная метрика вне пути награды допустим."""
    case = _clean_case(
        tmp_path,
        {
            "judges/__init__.py": "",
            "judges/llm_judge.py": '"""LLM-судья: отдельная калиброванная метрика."""\n',
        },
    )

    report = guard.build_report(case)

    assert report.ok, [finding.message for finding in report.findings]
    assert [rel for rel, _ in report.judge_modules] == ["judges/llm_judge.py"]
    assert report.exit_code == guard.EXIT_PASS


# --- (iii) контура награды нет ----------------------------------------------


def test_missing_reward_circuit_is_not_verified(tmp_path: Path, capsys) -> None:
    """(iii) нет кода награды — «НЕ ПРОВЕРЕНО», а не зелёный по умолчанию."""
    _write(tmp_path, "net/model.py", "def forward(x):\n    return x\n")

    report = guard.build_report(tmp_path)
    code, out = _run(tmp_path, capsys)

    assert not report.verified
    assert report.exit_code == guard.EXIT_NOT_VERIFIED
    assert code == guard.EXIT_NOT_VERIFIED
    assert guard.NOT_VERIFIED_MESSAGE in out
    assert "ложный PASS запрещён" in out


def test_empty_circuit_is_not_verified(tmp_path: Path) -> None:
    """Пустой репозиторий — тоже «НЕ ПРОВЕРЕНО», не PASS."""
    assert guard.build_report(tmp_path).exit_code == guard.EXIT_NOT_VERIFIED


# --- регрессия на рабочий кейс ----------------------------------------------


def test_case_reward_circuit_passes_and_lists_files(capsys) -> None:
    """Рабочий кейс: контур награды найден, чист, инструмент себя не сканирует."""
    report = guard.build_report(CASE_DIR)
    code, out = _run(CASE_DIR, capsys)
    rels = {item.rel for item in report.core}

    assert report.ok, [finding.message for finding in report.findings]
    assert code == guard.EXIT_PASS
    assert {"env/reward.py", "env/verifier.py", "env/run.py"} <= rels
    assert "tools/check_reward_isolation.py" not in rels
    assert "env/reward.py" in out and "Итог: PASS" in out


def test_c039_is_behavioural_guard_of_ad2() -> None:
    """Правило C-039 введено, классифицировано и привязано к AD-2."""
    constraints = yaml.safe_load((CASE_DIR / "CONSTRAINTS.yaml").read_text("utf-8"))
    rules = {rule["id"]: rule for rule in constraints["constraints"]}
    rule = rules["C-039"]

    assert rule["type"] == "command_succeeds"
    assert rule["command"] == "python3 tools/check_reward_isolation.py"
    assert rule["timeout_secs"] == 60
    assert rule["severity"] == "critical"
    assert rule["kind"] == "behavioural"
    assert "env/" in rule["evidence"]

    card = (CASE_DIR / "model" / "AD-2-mehanicheskiy-verdikt.md").read_text("utf-8")
    frontmatter = yaml.safe_load(card.split("---", 2)[1])
    assert "C-039" in frontmatter["verified_by"]


def test_ad2_has_no_pending_evidence_marker() -> None:
    """AD-2 переведён в состояние behavioural-стража, а не «ждёт доказательств»."""
    spine = (CASE_DIR / "ARCHITECTURE-SPINE.md").read_text("utf-8")
    block = spine.split("## AD-2:", 1)[1].split("## AD-3:", 1)[0]

    assert "PENDING-EVIDENCE" not in block
    assert "C-039" in block
