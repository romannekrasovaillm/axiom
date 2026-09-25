"""Генерация набора: ≥20 публичных задач + holdout-пул ≥50, вне workspace.

Покрывает §9 (лесенка), NFR-001 (holdout вне публичных задач) и C-032 (§11(7):
в workspace нет реальных файлов весов).
"""

from __future__ import annotations

from collections import Counter

from env import schemas
from env.util import find_weight_files, read_json


def test_public_and_holdout_counts(generated):
    assert generated["summary"]["public_count"] >= 20
    assert generated["summary"]["holdout_count"] >= 50


def test_grid_coverage(generated):
    public = generated["out"] / "public"
    specs = [read_json(p) for p in public.glob("*.json")]
    cells = Counter((s["source"], s["difficulty"]["level"]) for s in specs)
    for src in ("real", "corruption"):
        for lvl in ("L0", "L1"):
            assert cells[(src, lvl)] >= 5, f"{src}×{lvl} < 5"


def test_generated_specs_schema_valid(generated):
    out = generated["out"]
    for p in list((out / "public").glob("*.json")) + list((out / "holdout").glob("*.json")):
        s = read_json(p)
        assert schemas.validate_task_spec(s) == [], f"{p.name}: {schemas.validate_task_spec(s)}"


def test_no_weight_files_in_workspaces(generated):
    for ws in (generated["out"] / "public").iterdir():
        if ws.is_dir():
            assert find_weight_files(ws) == [], f"веса в workspace {ws.name}"


def test_holdout_outside_public_workspace(generated):
    out = generated["out"]
    assert not list((out / "public").glob("holdout-*.json"))
    assert len(list((out / "holdout").glob("holdout-*.json"))) >= 50
    assert (out / "holdout" / "hidden_constraints.yaml").exists()


def test_public_tasks_never_holdout(generated):
    for p in (generated["out"] / "public").glob("*.json"):
        assert read_json(p)["source"] != "holdout"


def test_corruption_restore_real_keep(generated):
    for p in (generated["out"] / "public").glob("*.json"):
        s = read_json(p)
        if s["source"] == "corruption":
            assert s["objective"]["kind"] == "restore-gates"
        else:
            assert s["objective"]["kind"] == "keep-gates-implement"
