"""§11(4): загрузчик манифестов отклоняет прогон без любого поля пиннинга."""

from __future__ import annotations

import json

import pytest

from env.manifest import ManifestError, dump_manifest, full_validate, load_manifest
from env.schemas import PINNING_PATHS

from .test_schemas import valid_manifest, valid_manifest_v1


def test_load_valid_manifest(tmp_path):
    p = tmp_path / "m.json"
    dump_manifest(valid_manifest(), p)
    assert load_manifest(p)["task_id"] == "corruption-l0-00"


def test_load_rejects_missing_each_pinning_field(tmp_path):
    for path_tuple in PINNING_PATHS:
        m = valid_manifest()
        # удаляем поле по пути (например, ("model", "snapshot_sha256"))
        cur = m
        for i, key in enumerate(path_tuple):
            if i == len(path_tuple) - 1:
                del cur[key]
            else:
                cur = cur[key]
        p = tmp_path / f"missing-{'-'.join(path_tuple)}.json"
        p.write_text(json.dumps(m), encoding="utf-8")
        with pytest.raises(ManifestError, match="пиннинг"):
            load_manifest(p)


def test_load_rejects_empty_pinning_value(tmp_path):
    m = valid_manifest()
    m["workspace_sha256"] = ""
    p = tmp_path / "empty.json"
    p.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(ManifestError, match="пиннинг"):
        load_manifest(p)


def test_full_validate_requires_pinning():
    m = valid_manifest()
    del m["model"]["snapshot_sha256"]
    errs = full_validate(m)
    assert any("model.snapshot_sha256" in e for e in errs)


def test_dump_rejects_incomplete_pinning(tmp_path):
    m = valid_manifest()
    del m["gates_version"]["arch_ml_build"]
    with pytest.raises(ManifestError):
        dump_manifest(m, tmp_path / "bad.json")


# ── v2 (ADR-058): заглушка, ось и исторические манифесты ───────────────────
def test_dump_rejects_stub_manifest(tmp_path):
    m = valid_manifest()
    m["model"]["is_stub"] = True
    with pytest.raises(ManifestError, match="model.is_stub"):
        dump_manifest(m, tmp_path / "stub.json")
    assert not (tmp_path / "stub.json").exists()


def test_dump_rejects_axis_mismatch(tmp_path):
    package = tmp_path / "DECISION-PACKAGE.json"
    package.write_text(
        '{"choice": {"victory_axis": "другая ось"}}\n', encoding="utf-8"
    )
    with pytest.raises(ManifestError, match="не совпадает с пакетом решения"):
        dump_manifest(valid_manifest(), tmp_path / "m.json", package_path=package)


def test_dump_accepts_axis_matching_package(tmp_path):
    package = tmp_path / "DECISION-PACKAGE.json"
    package.write_text(
        json.dumps(
            {"choice": {"victory_axis": valid_manifest()["victory_axis"]}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    dump_manifest(valid_manifest(), tmp_path / "m.json", package_path=package)
    assert (tmp_path / "m.json").is_file()


def test_load_rejects_legacy_without_allow_legacy(tmp_path):
    p = tmp_path / "v1.json"
    p.write_text(json.dumps(valid_manifest_v1()), encoding="utf-8")
    with pytest.raises(ManifestError):
        load_manifest(p)


def test_full_validate_legacy_only_with_flag():
    v1 = valid_manifest_v1()
    assert full_validate(v1) != []
    assert full_validate(v1, allow_legacy=True) == []
