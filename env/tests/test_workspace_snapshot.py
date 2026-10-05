"""§7 «Снапшот воркспейса» + §11(8): минимальность снапшота, кап объёма, детерминизм.

Дефект E-1.5: снапшот копировал ``data/`` (симлинки-датасеты раскрывались в
1,8 ГБ×20 = 35 ГБ). Эти тесты фиксируют: тяжёлые/нерантайм-пути исключены,
верхнеуровневый ``data/`` — не копия, объём воркспейса ≤ 64 МБ, хеш дерева
воспроизводим.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from env.util import (
    WORKSPACE_CAP_BYTES,
    copy_case_snapshot,
    dir_total_bytes,
    tree_sha256,
    workspace_size_cap,
)

# zstd-магия (0x28 B5 2F FD) + тело — валидное расширение для проверки исключения.
ZST_BYTES = b"\x28\xb5\x2f\xfd" + b"\x00" * 32


def _make_mini_case(root: Path) -> Path:
    """Эталонный мини-кейс: архитектурные файлы + тяжёлые/рантайм-каталоги."""
    (root / "model").mkdir(parents=True)
    (root / "model" / "AD-001.md").write_text("affects: C-001\n", encoding="utf-8")
    (root / "model" / "NFR-001.md").write_text("measure: p99\n", encoding="utf-8")
    (root / "docs" / "adr").mkdir(parents=True)
    (root / "docs" / "adr" / "ADR-001.md").write_text("# ADR-001\n", encoding="utf-8")
    (root / "tools").mkdir()
    (root / "tools" / "tool.py").write_text("print('x')\n", encoding="utf-8")
    (root / "net").mkdir()
    (root / "net" / "net.py").write_text("x = 1\n", encoding="utf-8")
    for name in ("CONSTRAINTS.yaml", "ARCHITECTURE-SPINE.md", "README.md"):
        (root / name).write_text(f"# {name}\n", encoding="utf-8")

    # Тяжёлое/рантайм — обязано быть исключено.
    (root / "data" / "datasets").mkdir(parents=True)
    (root / "data" / "README.md").write_text("data\n", encoding="utf-8")
    (root / "data" / "datasets" / "sft.jsonl").write_bytes(b"y" * (2 * 1024 * 1024))
    (root / "evidence" / "run").mkdir(parents=True)
    (root / "evidence" / "run" / "log.txt").write_text("evidence\n", encoding="utf-8")
    (root / "benchmarks").mkdir()
    (root / "benchmarks" / "b.md").write_text("bench\n", encoding="utf-8")
    (root / "runs-20260101").mkdir()
    (root / "runs-20260101" / "out.bin").write_bytes(b"r" * 4096)
    (root / "env" / "data").mkdir(parents=True)
    (root / "env" / "data" / "x.bin").write_bytes(b"e" * 4096)
    (root / "env" / "tasks").mkdir()
    (root / "env" / "tasks" / "t.json").write_text("{}", encoding="utf-8")
    (root / ".git").mkdir()
    (root / ".git" / "HEAD").write_text("ref\n", encoding="utf-8")
    (root / ".arch-handoff").mkdir()
    (root / ".arch-handoff" / "TASK.md").write_text("task\n", encoding="utf-8")

    # Бинарные расширения — исключаются на любом уровне.
    (root / "model" / "weights.safetensors").write_bytes(b"w" * 1024)
    (root / "model" / "pack.zst").write_bytes(ZST_BYTES)
    (root / "net" / "data.parquet").write_bytes(b"p" * 1024)

    # Вложенный data/ (не верхнеуровневый) — сохраняется (§7: исключается data/ кейса).
    (root / "tools" / "data").mkdir()
    (root / "tools" / "data" / "keep.md").write_text("keep\n", encoding="utf-8")
    return root


def _files(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def test_snapshot_excludes_heavy_and_runtime_paths(tmp_path):
    case = _make_mini_case(tmp_path / "case")
    dst = tmp_path / "snap"
    copy_case_snapshot(case, dst)
    rel = _files(dst)

    # Исключено: data/ кейса, evidence/, benchmarks/, runs-*/, env/, .git, .arch-handoff.
    assert not any(r.startswith("data/") for r in rel), sorted(rel)
    assert not any(r.startswith("evidence/") for r in rel)
    assert not any(r.startswith("benchmarks/") for r in rel)
    assert not any(r.startswith("runs-") for r in rel)
    assert not any(r.startswith("env/") for r in rel)
    assert not any(r.startswith(".git/") for r in rel)
    assert not any(r.startswith(".arch-handoff/") for r in rel)
    # Исключено по расширению на любом уровне.
    assert not any(r.endswith(".zst") for r in rel)
    assert not any(r.endswith((".safetensors", ".parquet", ".gguf", ".pt", ".ckpt")) for r in rel)
    # Верхнеуровневый data/ не скопирован.
    assert not (dst / "data").exists()

    # Сохранено: архитектура и код задачи.
    for kept in (
        "CONSTRAINTS.yaml",
        "ARCHITECTURE-SPINE.md",
        "README.md",
        "model/AD-001.md",
        "model/NFR-001.md",
        "docs/adr/ADR-001.md",
        "tools/tool.py",
        "net/net.py",
    ):
        assert kept in rel, f"должен копироваться: {kept}"
    # Вложенный tools/data/ остаётся (исключается только верхнеуровневый data/).
    assert "tools/data/keep.md" in rel


def test_cap_raises_with_top_paths_on_bloated_case(tmp_path):
    ws = tmp_path / "bloated"
    (ws / "docs").mkdir(parents=True)
    (ws / "docs" / "small.md").write_text("x", encoding="utf-8")
    (ws / "data" / "datasets").mkdir(parents=True)
    (ws / "data" / "datasets" / "sft_train.jsonl").write_bytes(b"a" * (2 * 1024 * 1024))
    (ws / "runs-1").mkdir()
    (ws / "runs-1" / "blob.bin").write_bytes(b"b" * (3 * 1024 * 1024))

    with pytest.raises(ValueError) as ei:
        workspace_size_cap(ws, cap_bytes=1 * 1024 * 1024)
    msg = str(ei.value)
    assert "bloated" in msg, msg  # задача
    assert "5.0 MiB" in msg, msg  # объём (3 МиБ + 2 МиБ)
    assert "топ-5" in msg, msg
    assert "runs-1" in msg and "data" in msg, msg  # крупнейшие пути

    # В пределах капа (штатный 64 МБ) — молча проходит.
    workspace_size_cap(ws, cap_bytes=WORKSPACE_CAP_BYTES)


def test_snapshot_deterministic(tmp_path):
    case = _make_mini_case(tmp_path / "case")
    d1, d2 = tmp_path / "s1", tmp_path / "s2"
    copy_case_snapshot(case, d1)
    copy_case_snapshot(case, d2)
    assert _files(d1) == _files(d2)
    assert tree_sha256(d1) == tree_sha256(d2)


def test_real_case_snapshot_under_cap(case_dir, tmp_path):
    """Реальный кейс axiom после исключений укладывается в кап (§11(8))."""
    dst = tmp_path / "snap"
    copy_case_snapshot(case_dir, dst)
    workspace_size_cap(dst)  # не должно бросить
    assert dir_total_bytes(dst) <= WORKSPACE_CAP_BYTES
    assert not (dst / "data").exists()


def test_snapshot_constraints_is_full_case_ruleset(tmp_path):
    """R-1' (§7): CONSTRAINTS.yaml снапшота == полный кейсовый ruleset.

    Ревизия R-1 (E-2.5): подмена на workspace-ruleset отклонена — ломала trace
    (12× ad-not-verified); машинонезависимость даёт фильтр в env.verifier.
    """
    case = _make_mini_case(tmp_path / "case")
    dst = tmp_path / "snap"
    copy_case_snapshot(case, dst)
    # Байт-в-байт равенство исходному кейсовому CONSTRAINTS.yaml (не редукция).
    assert (dst / "CONSTRAINTS.yaml").read_bytes() == (case / "CONSTRAINTS.yaml").read_bytes()


def test_generated_workspaces_under_cap(generated):
    """Каждый сгенерированный воркспейс ≤ 64 МБ и без data/datasets (§7, §11(8))."""
    for ws in (generated["out"] / "public").iterdir():
        if not ws.is_dir():
            continue
        assert dir_total_bytes(ws) <= WORKSPACE_CAP_BYTES, ws.name
        assert not (ws / "data").exists(), ws.name
