"""Страж Run Manifest v2 (tools/check_manifest_v2.py, C-055, ADR-058).

Мутанты (нет оси / нет baseline / заглушка / чужая ось / v1 без allow_legacy)
прогоняются в selftest самого стража; здесь — код selftest и контракт
``--check`` на временных файлах. Сеть и jax не нужны.
"""

from __future__ import annotations

import json

from env import axis, manifest as manifest_mod
from tools import check_manifest_v2 as cm


def test_selftest_green():
    assert cm.run_selftest() == 0


def test_check_valid_v2_against_real_package(tmp_path):
    axis_value = axis.load_axis(manifest_mod.DEFAULT_PACKAGE_PATH)
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(cm._fixture_manifest(axis_value), ensure_ascii=False), encoding="utf-8"
    )
    assert cm.check_file(path, manifest_mod.DEFAULT_PACKAGE_PATH) == []


def test_check_v1_manifest_red(tmp_path):
    legacy = cm._legacy_v1(cm._fixture_manifest(cm.SELFTEST_AXIS))
    path = tmp_path / "manifest-v1.json"
    path.write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")
    errors = cm.check_file(path, manifest_mod.DEFAULT_PACKAGE_PATH)
    assert errors


def test_cli_check_returns_nonzero_on_v1(tmp_path):
    legacy = cm._legacy_v1(cm._fixture_manifest(cm.SELFTEST_AXIS))
    path = tmp_path / "manifest-v1.json"
    path.write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")
    assert cm.main(["--check", str(path)]) == cm.EXIT_FAIL
