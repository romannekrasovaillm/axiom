"""§11(4): загрузчик манифестов отклоняет прогон без любого поля пиннинга."""

from __future__ import annotations

import json

import pytest

from env.manifest import ManifestError, dump_manifest, full_validate, load_manifest
from env.schemas import PINNING_PATHS

from .test_schemas import valid_manifest


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
