"""Canonical SFT checkpoint path is bound to the run/out directory.

Before the fix ``save_stage_checkpoint`` wrote to a fixed
``shared/checkpoints/sft-smoke`` regardless of ``--out``/``--ckpt-dir``: the
second smoke (l3-full -> small) overwrote the first one's weights while the
first one's journal stayed behind.  Pinned here:

* ``canonical_checkpoint_dir`` derives the directory from ``ckpt_dir`` — two
  different run/out directories give two different canonical paths;
* the slug is deterministic and consists only of ``[A-Za-z0-9._-]``; an empty
  or suspicious run name falls back to a short sha256 of the path;
* ``save_stage_checkpoint`` writes to exactly this canonical directory and
  leaves a symlink in the working tree (C-032), never a copy;
* an existing plain directory is still rejected (C-032).
"""

from __future__ import annotations

import hashlib
import re
import sys
import types
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1]
CASE_DIR = TOOLS_DIR.parent
for _p in (str(TOOLS_DIR), str(CASE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_sft_smoke as runner  # noqa: E402

SLUG_RE = re.compile(r"[A-Za-z0-9._-]+")


def test_two_outs_get_different_canonical_dirs(tmp_path) -> None:
    """The collision fix: distinct ``--out`` -> distinct canonical paths."""
    shared = tmp_path / "shared"
    small = runner.canonical_checkpoint_dir(
        shared, Path("evidence/sft-small-20261022/checkpoint")
    )
    smoke = runner.canonical_checkpoint_dir(
        shared, Path("evidence/sft-smoke-20261022/checkpoint")
    )

    assert small != smoke
    assert small.parent == shared / "checkpoints"
    assert smoke.parent == shared / "checkpoints"
    assert small.name.endswith("sft-small-20261022")
    assert smoke.name.endswith("sft-smoke-20261022")


def test_slug_is_sanitized_and_stable(tmp_path) -> None:
    """Only ``[A-Za-z0-9._-]`` survives; repeated calls agree."""
    shared = tmp_path / "shared"
    ckpt_dir = Path("evidence") / "sft small:20261022*run" / "checkpoint"

    first = runner.canonical_checkpoint_dir(shared, ckpt_dir)
    second = runner.canonical_checkpoint_dir(shared, ckpt_dir)

    assert first == second
    assert SLUG_RE.fullmatch(first.name)
    for forbidden in (" ", "/", ":", "*"):
        assert forbidden not in first.name
    assert first.name.endswith("sft-small-20261022-run")


def test_empty_or_suspicious_run_name_uses_path_hash_fallback(tmp_path) -> None:
    """An unusable run name is replaced by a short sha256 of the path."""
    shared = tmp_path / "shared"
    empty = runner.canonical_checkpoint_dir(shared, Path("/checkpoint"))
    empty_again = runner.canonical_checkpoint_dir(shared, Path("/checkpoint"))
    other = runner.canonical_checkpoint_dir(shared, Path("/other/checkpoint"))
    weird = runner.canonical_checkpoint_dir(
        shared, Path("evidence/???/checkpoint")
    )

    digest = hashlib.sha256(str(Path("/checkpoint")).encode("utf-8")).hexdigest()
    assert empty.name == "sft-" + digest[:12]
    assert empty == empty_again  # stable between calls
    assert empty != other  # different paths -> different fallbacks
    for path in (empty, weird):
        assert SLUG_RE.fullmatch(path.name)
        assert re.search(r"[0-9a-f]{12}", path.name)


@pytest.fixture
def fake_checkpoint(monkeypatch):
    """Replace ``net.checkpoint`` without importing jax/orbax (CPU-only test)."""
    import net

    calls: dict[str, list] = {"saved": [], "loaded": []}
    module = types.ModuleType("net.checkpoint")

    def save_checkpoint(params, path):
        calls["saved"].append(Path(path))
        (Path(path) / "payload.txt").write_text("weights", encoding="utf-8")
        return "digest-1"

    def load_checkpoint(path, target=None):
        calls["loaded"].append(Path(path))
        return target

    def tree_hash(tree):
        return "digest-1"

    module.save_checkpoint = save_checkpoint
    module.load_checkpoint = load_checkpoint
    module.tree_hash = tree_hash
    monkeypatch.setitem(sys.modules, "net.checkpoint", module)
    monkeypatch.setattr(net, "checkpoint", module, raising=False)
    return calls


def test_save_stage_checkpoint_writes_to_computed_canonical(
    tmp_path, fake_checkpoint
) -> None:
    """The writer uses the helper's path; the repo-side entry is a symlink."""
    shared = tmp_path / "shared"
    ckpt_dir = tmp_path / "evidence" / "sft-small-20261022" / "checkpoint"
    expected = runner.canonical_checkpoint_dir(shared, ckpt_dir)

    report = runner.save_stage_checkpoint({"w": 1}, ckpt_dir, shared, "digest-1")

    assert fake_checkpoint["saved"] == [expected]
    assert fake_checkpoint["loaded"] == [expected]
    assert ckpt_dir.is_symlink()
    assert ckpt_dir.resolve() == expected.resolve()
    assert (expected / "payload.txt").read_text(encoding="utf-8") == "weights"
    assert report["canonical"].endswith(expected.name)
    assert report["symlink"] is True
    assert report["roundtrip_ok"] is True
    assert report["tree_hash"] == "digest-1"


def test_rerun_keeps_the_same_canonical(tmp_path, fake_checkpoint) -> None:
    """A second run with an already mounted symlink writes to the same place."""
    shared = tmp_path / "shared"
    ckpt_dir = tmp_path / "evidence" / "sft-small-20261022" / "checkpoint"
    expected = runner.canonical_checkpoint_dir(shared, ckpt_dir)

    runner.save_stage_checkpoint({"w": 1}, ckpt_dir, shared, "digest-1")
    runner.save_stage_checkpoint({"w": 1}, ckpt_dir, shared, "digest-1")

    assert fake_checkpoint["saved"] == [expected, expected]
    assert ckpt_dir.resolve() == expected.resolve()


def test_stale_legacy_symlink_is_repointed_without_touching_old_weights(
    tmp_path, fake_checkpoint
) -> None:
    """A pre-fix mount (old fixed canonical) is re-pointed; old weights stay."""
    shared = tmp_path / "shared"
    legacy = shared / "checkpoints" / "sft-smoke"
    legacy.mkdir(parents=True)
    (legacy / "payload.txt").write_text("legacy-weights", encoding="utf-8")
    ckpt_dir = tmp_path / "evidence" / "a4-run-wire" / "sft" / "checkpoint"
    ckpt_dir.parent.mkdir(parents=True)
    ckpt_dir.symlink_to(legacy)
    expected = runner.canonical_checkpoint_dir(shared, ckpt_dir)

    runner.save_stage_checkpoint({"w": 1}, ckpt_dir, shared, "digest-1")

    assert ckpt_dir.resolve() == expected.resolve()
    assert (legacy / "payload.txt").read_text(encoding="utf-8") == "legacy-weights"
    assert fake_checkpoint["saved"] == [expected]


def test_existing_plain_directory_is_rejected(tmp_path, fake_checkpoint) -> None:
    """C-032: a copy in the working tree is an error, not an accepted target."""
    shared = tmp_path / "shared"
    ckpt_dir = tmp_path / "run" / "checkpoint"
    ckpt_dir.mkdir(parents=True)

    with pytest.raises(runner.StageError, match="C-032"):
        runner.save_stage_checkpoint({"w": 1}, ckpt_dir, shared, "digest-1")

    assert fake_checkpoint["saved"] == []
