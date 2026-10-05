"""§11(3): мутатор порчи воспроизводим — одинаковый seed → байт-в-байт одинаковый кейс."""

from __future__ import annotations

from pathlib import Path

from env import corruption
from env.util import tree_sha256


def _files_equal(a: Path, b: Path) -> bool:
    fa = sorted(p.relative_to(a).as_posix() for p in a.rglob("*") if p.is_file())
    fb = sorted(p.relative_to(b).as_posix() for p in b.rglob("*") if p.is_file())
    if fa != fb:
        return False
    for rel in fa:
        if (a / rel).read_bytes() != (b / rel).read_bytes():
            return False
    return True


def test_same_seed_byte_identical(tmp_path, case_dir):
    out1 = tmp_path / "a"
    out2 = tmp_path / "b"
    corruption.corrupt(case_dir, out1, seed=42, level="L1")
    corruption.corrupt(case_dir, out2, seed=42, level="L1")
    assert _files_equal(out1, out2)
    assert tree_sha256(out1) == tree_sha256(out2)


def test_different_seed_differs(tmp_path, case_dir):
    out1 = tmp_path / "a"
    out2 = tmp_path / "b"
    corruption.corrupt(case_dir, out1, seed=42, level="L1")
    corruption.corrupt(case_dir, out2, seed=43, level="L1")
    assert tree_sha256(out1) != tree_sha256(out2)


def test_l0_single_atom_l1_three_atoms(tmp_path, case_dir):
    d0 = corruption.corrupt(case_dir, tmp_path / "l0", seed=7, level="L0")
    d1 = corruption.corrupt(case_dir, tmp_path / "l1", seed=7, level="L1")
    assert len(d0) == 1
    assert len(d1) == 3


def test_revert_restores_clean(tmp_path, case_dir):
    out = tmp_path / "corr"
    damages = corruption.corrupt(case_dir, out, seed=42, level="L1")
    for d in damages:
        corruption.revert_damage(out, case_dir, d)
    # после отката всех повреждений дерево совпадает с чистым снапшотом
    from env.util import copy_case_snapshot

    clean = tmp_path / "clean"
    copy_case_snapshot(case_dir, clean)
    assert _files_equal(out, clean)


def test_damage_detectable_by_gates(tmp_path, case_dir, arch_ml):
    from env.verifier import collect_violations

    out = tmp_path / "corr"
    corruption.corrupt(case_dir, out, seed=42, level="L1")
    spec = {
        "id": "probe",
        "source": "corruption",
        "objective": {"kind": "restore-gates", "tests_cmd": "true"},
        "verifier": {"constraints": "CONSTRAINTS.yaml", "spine": True, "trace": True,
                     "hidden_constraints_sha256": "0" * 64},
        "max_tokens": 131072,
    }
    violations = collect_violations(out, spec, bin=arch_ml)
    assert violations, "повреждённый кейс должен давать error-нарушения"


def test_l1_atoms_recoverable_by_edit_file(tmp_path, case_dir):
    """R-3 (§9): атомы L1 восстановимы 4 инструментами §13 (без создания файлов).

    ``break_ad_link`` (удаление файла) исключён из L1: восстановление требовало бы
    создания отсутствующего файла, которого интерфейс агента v1 не несёт.
    Оставшиеся атомы — чистые вставки текста (``edit_file`` old→new), проверяем
    конструктивно: восстановление каждого возвращает файл байт-в-байт к чистому.
    """
    assert "break_ad_link" not in corruption.LEVEL_ATOMS["L1"]
    assert len(corruption.LEVEL_ATOMS["L1"]) == 3

    out = tmp_path / "corr"
    damages = corruption.corrupt(case_dir, out, seed=42, level="L1")
    for d in damages:
        assert d.kind != "break_ad_link"
        # файл существует, повреждение — вставка текста, а не удаление файла
        target = out / d.file
        assert target.is_file(), d.file
        clean_text = (case_dir / d.file).read_text(encoding="utf-8")
        damaged_text = target.read_text(encoding="utf-8")
        assert clean_text != damaged_text
        # восстановление edit_file-вставкой: clean == damaged со вставленным блоком
        p = 0
        m = min(len(clean_text), len(damaged_text))
        while p < m and clean_text[p] == damaged_text[p]:
            p += 1
        assert len(clean_text) > len(damaged_text), f"{d.kind}: не вставка"
        assert clean_text[:p] == damaged_text[:p]
        assert clean_text[p + (len(clean_text) - len(damaged_text)):] == damaged_text[p:]
        # откат возвращает байт-в-байт чистое состояние
        corruption.revert_damage(out, case_dir, d)
        assert target.read_bytes() == (case_dir / d.file).read_bytes()
