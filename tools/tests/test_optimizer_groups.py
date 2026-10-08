"""Тесты прибора ``tools/optimizer_groups.py`` (отчёт классификации, ADR-048).

Предмет:

* **дисциплина ADR-041** — маркер ``ensure_mem_fraction()`` в файле и его вызов
  на прогонном пути ДО импорта jax (импорт jax внутри функций, не в модуле);
* **отчёт по решению ADR-048** на реальной геометрии ``l3-full``: embedding
  (связанный LM head) в группе AdamW с полным числом параметров
  ``vocab x hidden``, неклассифицированных листьев нет;
* **режим ``--legacy``** возвращает embedding в Muon — пара «до/после» на одной
  ревизии кода (тем же предикатом, что читают ``init_state``/``make_step``);
* **запись артефактов** — JSON + markdown, с суффиксом ``-legacy`` в прежнем
  режиме, имена групп попадают в таблицу.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1]
CASE_DIR = TOOLS_DIR.parent
for _path in (str(TOOLS_DIR), str(CASE_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import optimizer_groups as og  # noqa: E402

L3_VOCAB, L3_HIDDEN = 160000, 1536


def test_preflight_marker_present_and_called_before_report_import():
    """ADR-041: лимит памяти XLA ставится до импорта jax — маркер и вызов на пути."""
    source = (TOOLS_DIR / "optimizer_groups.py").read_text(encoding="utf-8")
    assert "jax_preflight.ensure_mem_fraction()" in source
    body = source[source.index("def main(") :]
    assert body.index("_preflight_memory()") < body.index("build_artifact(")


def test_jax_is_not_imported_at_module_level():
    """Прибор импортируется сьютом без jax: jax тянется только внутри функций."""
    source = (TOOLS_DIR / "optimizer_groups.py").read_text(encoding="utf-8")
    module_level = source[: source.index("def ")]
    assert "import jax" not in module_level


def test_report_matches_adr048_decision():
    report = og.build_artifact(og.DEFAULT_CONFIG, legacy=False)
    assert report["schema"] == "axiom/optimizer-param-groups/1"
    assert report["adr"] == "ADR-048"
    assert report["source"] == "net.optimizer.classify_leaf"
    assert report["unclassified"] == []
    assert report["total_leaves"] == 723

    groups = report["groups"]
    embed = groups["adamw_embed"]
    assert embed["leaves"] == 1
    assert embed["names"] == ["embedding"]
    assert embed["params"] == L3_VOCAB * L3_HIDDEN
    assert groups["muon_matrix"]["params"] > embed["params"]
    assert "embedding" not in groups["muon_matrix"]["names"]


def test_legacy_report_restores_muon_for_embedding():
    current = og.build_artifact(og.DEFAULT_CONFIG, legacy=False)
    legacy = og.build_artifact(og.DEFAULT_CONFIG, legacy=True)
    assert legacy["legacy_muon_all_2d"] is True
    assert legacy["groups"]["adamw_embed"]["leaves"] == 0
    assert "embedding" in legacy["groups"]["muon_matrix"]["names"]
    assert (
        legacy["groups"]["muon_matrix"]["params"]
        == current["groups"]["muon_matrix"]["params"] + current["groups"]["adamw_embed"]["params"]
    )
    assert legacy["groups"]["muon_per_head"] == current["groups"]["muon_per_head"]
    assert legacy["groups"]["muon_batched"] == current["groups"]["muon_batched"]


def test_write_artifact_writes_json_and_markdown(tmp_path):
    report = og.build_artifact(og.DEFAULT_CONFIG, legacy=False)
    json_path, md_path = og.write_artifact(report, tmp_path)

    assert json_path.name == "optimizer-param-groups.json"
    assert md_path.name == "optimizer-param-groups.md"
    assert json.loads(json_path.read_text(encoding="utf-8"))["groups"]["adamw_embed"]["leaves"] == 1
    table = md_path.read_text(encoding="utf-8")
    assert "adamw_embed" in table and "embedding" in table
    assert "legacy_muon_all_2d=False" in table


def test_write_artifact_uses_legacy_suffix(tmp_path):
    report = og.build_artifact(og.DEFAULT_CONFIG, legacy=True)
    json_path, md_path = og.write_artifact(report, tmp_path)
    assert json_path.name == "optimizer-param-groups-legacy.json"
    assert md_path.name == "optimizer-param-groups-legacy.md"
    assert "все 2-D листья Muon" in md_path.read_text(encoding="utf-8") or "прежняя" in md_path.read_text(encoding="utf-8")


def test_write_artifact_warns_on_unclassified(tmp_path):
    """Пропуск в классификации не прячется: отчёт пишется, но с предупреждением."""
    import jax.numpy as jnp

    from net import optimizer

    report = optimizer.classification_report({"W_mystery": jnp.zeros((4, 4))})
    with pytest.warns(UserWarning, match="W_mystery"):
        og.write_artifact(report, tmp_path)
    assert report["unclassified"] == ["W_mystery"]
