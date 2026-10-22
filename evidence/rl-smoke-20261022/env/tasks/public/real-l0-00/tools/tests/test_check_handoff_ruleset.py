"""Тесты стража C-037 (tools/check_handoff_ruleset.py).

Страж проверяет канал гейта (ADR-011, п. 5): пакет обязан нести рабочий
ruleset кейса. Обязательные сценарии из постановки дельты:

* совпадение рабочего и пакетного файлов (число правил + sha256) -> PASS;
* расхождение -> FAIL;
* отсутствие пакета -> NOT-VERIFIED с ненулевым кодом (ложный PASS запрещён);
* отсутствие рабочего файла -> NOT-VERIFIED (сверять не с чем);
* равное число правил при разном содержимом -> FAIL (вердикт держится на
  sha256, а не на счётчике).

Скрипт stdlib-only: тесты запускают его как отдельный процесс, как это
делает `control check` (bash -c, cwd = каталог кейса).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1]
SCRIPT = TOOLS_DIR / "check_handoff_ruleset.py"

RULE_ONE = (
    "constraints:\n"
    "  - id: C-01\n"
    "    name: первое\n"
    "    type: file_exists\n"
    "    path: README.md\n"
    "    severity: warn\n"
)
RULE_TWO = RULE_ONE + (
    "  - id: C-02\n"
    "    name: второе\n"
    "    type: file_exists\n"
    "    path: Cargo.toml\n"
    "    severity: warn\n"
)
#: Два правила, но другое содержимое — число совпадает, sha256 нет.
RULE_TWO_OTHER = (
    "constraints:\n"
    "  - id: C-01\n"
    "    name: первое\n"
    "    type: file_exists\n"
    "    path: README.md\n"
    "    severity: warn\n"
    "  - id: C-02\n"
    "    name: другое\n"
    "    type: file_exists\n"
    "    path: OTHER.md\n"
    "    severity: warn\n"
)


def _run(case: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--case", str(case)],
        capture_output=True,
        text=True,
        timeout=30,
    )


def _case(tmp_path: Path, working: str | None, packet: str | None) -> Path:
    case = tmp_path / "case"
    case.mkdir(parents=True, exist_ok=True)
    if working is not None:
        (case / "CONSTRAINTS.yaml").write_text(working, encoding="utf-8")
    if packet is not None:
        handoff = case / ".arch-handoff"
        handoff.mkdir(parents=True, exist_ok=True)
        (handoff / "CONSTRAINTS.yaml").write_text(packet, encoding="utf-8")
    return case


def test_matching_ruleset_passes(tmp_path: Path) -> None:
    case = _case(tmp_path, RULE_TWO, RULE_TWO)
    result = _run(case)
    assert result.returncode == 0, result.stderr
    assert "PASS" in result.stderr
    assert "правил: рабочий 2, пакетный 2" in result.stdout


def test_divergent_ruleset_fails(tmp_path: Path) -> None:
    case = _case(tmp_path, RULE_TWO, RULE_ONE)
    result = _run(case)
    assert result.returncode == 1, result.stderr
    assert "FAIL" in result.stderr
    assert "число правил" in result.stderr


def test_same_count_different_content_fails_on_sha256(tmp_path: Path) -> None:
    case = _case(tmp_path, RULE_TWO, RULE_TWO_OTHER)
    result = _run(case)
    assert result.returncode == 1, result.stderr
    assert "FAIL" in result.stderr
    assert "sha256 не совпал" in result.stderr


def test_missing_packet_is_not_verified(tmp_path: Path) -> None:
    case = _case(tmp_path, RULE_TWO, None)
    result = _run(case)
    assert result.returncode != 0, "отсутствие пакета не может быть PASS"
    assert result.returncode == 2
    assert "NOT-VERIFIED" in result.stderr
    assert "отсутствует" in result.stderr
    assert "PASS:" not in result.stderr, "NOT-VERIFIED не должен выглядеть как PASS"


def test_missing_working_ruleset_is_not_verified(tmp_path: Path) -> None:
    case = _case(tmp_path, None, RULE_ONE)
    result = _run(case)
    assert result.returncode == 2
    assert "NOT-VERIFIED" in result.stderr
    assert "рабочий ruleset" in result.stderr
    assert "PASS:" not in result.stderr, "NOT-VERIFIED не должен выглядеть как PASS"


def test_default_case_is_cwd(tmp_path: Path) -> None:
    case = _case(tmp_path, RULE_TWO, RULE_TWO)
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(case),
    )
    assert result.returncode == 0, result.stderr
    assert "PASS" in result.stderr
