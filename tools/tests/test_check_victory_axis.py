"""Страж оси победы (tools/check_victory_axis.py, C-013, ADR-058).

Фикстуры-мутанты — в selftest стража; здесь проверяются коды и негативные
каталоги. Сеть и jax не нужны. Репозиторий не мутируется: все негативные
кейсы собираются во временных каталогах.
"""

from __future__ import annotations

import json
from pathlib import Path

from tools import check_victory_axis as cva

ROOT = Path(__file__).resolve().parents[2]


def test_selftest_green():
    assert cva.run_selftest() == 0


def test_repo_victory_axis_green():
    assert cva.evaluate(ROOT) == []


def test_missing_package_red(tmp_path):
    errors = cva.evaluate(tmp_path)
    assert any(cva.PACKAGE_REL in e for e in errors)


def test_top_level_axis_red(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "model").mkdir()
    (tmp_path / cva.PACKAGE_REL).write_text(
        json.dumps({"choice": {"victory_axis": "ось"}, "victory_axis": "дубль"}),
        encoding="utf-8",
    )
    for rel in (cva.SPINE_REL, cva.AD1_REL):
        (tmp_path / rel).write_text(f"ось: {cva.AXIS_REF}\n", encoding="utf-8")
    errors = cva.evaluate(tmp_path)
    assert any("верхнем уровне" in e for e in errors)


def test_cli_returns_nonzero_on_broken_root(tmp_path):
    assert cva.main(["--root", str(tmp_path)]) == cva.EXIT_FAIL
