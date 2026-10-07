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

Отдельный полюс дельты E-5.2 — область сканирования: страж читает **только
git-tracked файлы** (``git ls-files --cached``). Untracked-мусор прогонов (тысячи
``*.py`` под ``env/``) не входит в кодовую базу: он не должен ни давать находок,
ни замедлять прогон до таймаута правила C-039 (60 с). Без git — честный откат на
файловую систему с предупреждением ``SCAN-FALLBACK-FILESYSTEM``.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import check_reward_isolation as guard  # noqa: E402  (путь добавляется выше)

CASE_DIR = Path(__file__).resolve().parents[2]

CLEAN_REWARD = "def compute(passed):\n    return 1.0 if passed else 0.0\n"

requires_git = pytest.mark.skipif(
    shutil.which("git") is None,
    reason="git недоступен — сценарии tracked-only пропущены",
)


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


def _git_repo(root: Path) -> Path:
    """Фикстура становится git-репозиторием: уже записанное — tracked (индекс).

    Коммит не нужен: ``git ls-files --cached`` читает индекс, а всё записанное
    после ``git add -A`` остаётся untracked — ровно то, что моделирует мусор прогонов.
    """
    subprocess.run(["git", "init", "-q", str(root)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(root), "add", "-A"], check=True, capture_output=True
    )
    return root


def _untracked_junk(index: int) -> str:
    """Мусор прогона: LLM-клиент + маркер судьи — под старым сканом это находка."""
    noise = "".join(
        f"def noise_{index}_{n}(x):\n    return x + {n}  # llm-судья\n\n"
        for n in range(40)
    )
    return "import httpx\n\n\n" + noise


# --- E-5.2: область сканирования — только git-tracked файлы -------------------


@requires_git
def test_untracked_module_with_httpx_is_not_a_finding(tmp_path: Path) -> None:
    """(а) untracked py с httpx в env/ — мусор прогона, не находка и не код контура."""
    _clean_case(
        tmp_path,
        {"env/reward.py": "from .helper import compute\n\n\n" + CLEAN_REWARD},
    )
    _git_repo(tmp_path)
    # Записано ПОСЛЕ git add: untracked, в индекс не входит.
    _write(
        tmp_path, "env/helper.py", "import httpx\n\n\ndef compute(x):\n    return x\n"
    )

    report = guard.build_report(tmp_path)
    circuit = {item.rel for item in report.core} | {
        item.rel for item in report.dependencies
    }

    assert "env/helper.py" not in circuit
    assert report.ok, [finding.message for finding in report.findings]
    assert report.exit_code == guard.EXIT_PASS


@requires_git
def test_tracked_module_with_httpx_is_still_a_finding(tmp_path: Path) -> None:
    """(а) контроль: тот же модуль, но tracked — находка (граница не сместилась)."""
    _clean_case(
        tmp_path,
        {
            "env/reward.py": "from .helper import compute\n\n\n" + CLEAN_REWARD,
            "env/helper.py": "import httpx\n\n\ndef compute(x):\n    return x\n",
        },
    )
    _git_repo(tmp_path)

    report = guard.build_report(tmp_path)

    assert report.exit_code == guard.EXIT_VIOLATION
    assert "env/helper.py" in {finding.rel for finding in report.findings}
    assert "REWARD-PATH-EXTERNAL-API" in {finding.code for finding in report.findings}


@requires_git
def test_untracked_junk_does_not_slow_the_guard(tmp_path: Path) -> None:
    """(б) 1000 untracked py под env/ — страж не читает их: < 5 с, находок нет."""
    _clean_case(tmp_path)
    _git_repo(tmp_path)
    for index in range(1000):
        _write(tmp_path, f"env/junk_{index:04d}.py", _untracked_junk(index))

    started = time.perf_counter()
    report = guard.build_report(tmp_path)
    elapsed = time.perf_counter() - started

    assert elapsed < 5.0, f"страж читал untracked-мусор: {elapsed:.1f} с на 1000 файлов"
    assert report.exit_code == guard.EXIT_PASS
    # Мусор не сканировался: ни судей, ни находок из него.
    assert report.judge_modules == []
    assert report.findings == []
    # Механизм, а не только секундомер: в кандидаты попали ровно tracked-файлы.
    candidates, note = guard.scan_plan(tmp_path)
    assert note is None
    assert {path.relative_to(tmp_path).as_posix() for path in candidates} == {
        "env/__init__.py",
        "env/reward.py",
    }


def test_git_unavailable_falls_back_to_filesystem_with_warning(
    tmp_path: Path, capsys
) -> None:
    """Без git (не репозиторий) — прежний обход диска, но с предупреждением."""
    _clean_case(
        tmp_path,
        {
            "env/reward.py": "from helpers.scoring import compute_score\n\n\n"
            "def compute(x):\n    return compute_score(x)\n",
            "helpers/__init__.py": "",
            "helpers/scoring.py": "import anthropic\n\n\n"
            "def compute_score(x):\n    return x\n",
        },
    )

    report = guard.build_report(tmp_path)
    code, out = _run(tmp_path, capsys)

    assert any(w.code == guard.SCAN_FALLBACK_CODE for w in report.warnings)
    # Откат не отключает проверку: находка в untracked-помощнике находится.
    assert "REWARD-PATH-LLM-IMPORT" in {finding.code for finding in report.findings}
    assert code == guard.EXIT_VIOLATION
    assert guard.SCAN_FALLBACK_CODE in out
    assert "режим отката без git" in out


@requires_git
def test_case_dir_scan_is_tracked_only_and_fast() -> None:
    """Рабочий кейс — git-репозиторий: скан tracked-only, без отката, ~секунда."""
    started = time.perf_counter()
    report = guard.build_report(CASE_DIR)
    elapsed = time.perf_counter() - started

    assert not any(w.code == guard.SCAN_FALLBACK_CODE for w in report.warnings)
    assert report.exit_code == guard.EXIT_PASS
    assert elapsed < 5.0, f"страж на рабочем дереве шёл {elapsed:.1f} с"


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


def test_declared_root_basis_is_printed(tmp_path: Path, capsys) -> None:
    """(c) основание включения в контур печатается, а не подразумевается."""
    case = _clean_case(tmp_path)

    report = guard.build_report(case)
    _, out = _run(case, capsys)

    assert [item.rel for item in report.roots] == ["env/reward.py"]
    assert "Корни пути награды" in out
    assert "объявленный корень" in out
    assert "Файлы контура награды" in out


# --- сужение контура: import-замыкание, а не весь пакет env/ -----------------


def test_whole_env_package_is_not_the_circuit(tmp_path: Path) -> None:
    """Прибор в env/ вне import-замыкания награды не проверяется (сужение C-039)."""
    case = _clean_case(
        tmp_path,
        {
            # «Прибор»: импортирует HTTP-клиент, но награда (корень) его не зовёт.
            "env/eval_probe.py": "import httpx\n\n\ndef probe():\n    return httpx\n",
        },
    )

    report = guard.build_report(case)
    rels = {item.rel for item in report.core} | {
        item.rel for item in report.dependencies
    }

    assert "env/eval_probe.py" not in rels
    assert report.ok, [finding.message for finding in report.findings]
    assert report.exit_code == guard.EXIT_PASS


def test_clients_package_is_outside_reward_path(tmp_path: Path) -> None:
    """clients/ (httpx-механизм) — отдельный домен, стражем награды не проверяется."""
    case = _clean_case(
        tmp_path,
        {
            "clients/__init__.py": "",
            "clients/openai_http.py": (
                '"""httpx-транспорт."""\n\nimport httpx\n\n\n'
                "def build():\n    return httpx.Client()\n"
            ),
        },
    )

    report = guard.build_report(case)
    rels = {item.rel for item in report.core} | {
        item.rel for item in report.dependencies
    }

    assert not any(rel.startswith("clients/") for rel in rels)
    assert report.ok, [finding.message for finding in report.findings]
    assert report.exit_code == guard.EXIT_PASS


def test_openai_port_in_env_is_not_a_finding(tmp_path: Path) -> None:
    """env/openai_adapter.py (порт без httpx) — не корень и не находка."""
    case = _clean_case(
        tmp_path,
        {
            "env/openai_adapter.py": (
                '"""Порт OpenAI-совместимого адаптера: без HTTP-библиотеки."""\n\n'
                'DEFAULT_BASE_URL = "http://127.0.0.1:8080"\n\n\n'
                "class OpenAIModelAdapter:\n"
                "    def generate(self, messages):\n"
                "        raise NotImplementedError\n"
            ),
        },
    )

    report = guard.build_report(case)
    assert {item.rel for item in report.core} == {"env/__init__.py", "env/reward.py"}
    assert report.ok, [finding.message for finding in report.findings]


def test_httpx_import_in_reward_root_is_a_finding(tmp_path: Path) -> None:
    """Мутант: httpx в корне награды — находка (граница не сместилась)."""
    case = _clean_case(tmp_path, {"env/reward.py": "import httpx\n\n\n" + CLEAN_REWARD})

    report = guard.build_report(case)

    assert report.exit_code == guard.EXIT_VIOLATION
    assert {f.rel for f in report.findings} == {"env/reward.py"}
    assert "REWARD-PATH-EXTERNAL-API" in {f.code for f in report.findings}


def test_httpx_import_in_declared_verifier_root_is_a_finding(tmp_path: Path) -> None:
    """Объявленный корень env/verifier.py тоже проверяется (не только reward.py)."""
    case = _clean_case(tmp_path, {"env/verifier.py": "import httpx\n"})

    report = guard.build_report(case)

    assert report.exit_code == guard.EXIT_VIOLATION
    assert {f.rel for f in report.findings} == {"env/verifier.py"}


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
    checked = rels | {item.rel for item in report.dependencies}

    assert report.ok, [finding.message for finding in report.findings]
    assert code == guard.EXIT_PASS
    assert {"env/reward.py", "env/verifier.py", "env/run.py"} <= rels
    assert "tools/check_reward_isolation.py" not in rels
    # Приборы env/ и пакет clients/ — вне import-замыкания награды.
    assert "env/openai_adapter.py" not in checked
    assert not any(rel.startswith("clients/") for rel in checked)
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
