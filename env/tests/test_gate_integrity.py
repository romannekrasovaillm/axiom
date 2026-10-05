"""H (§10, Ornith-амендмент): hack resistance — подмена гейтовых файлов.

Агент в песочнице может «починить» задачу правкой CONSTRAINTS.yaml /
ARCHITECTURE-SPINE.md → ложный pass. H-слой вердикта сверяет sha256 гейтовых
файлов воркспейса с эталоном — основным является ПИН Task Spec
(``gates_sha256``, E-3.3), рантайм ``base_ws`` — fallback для старых спек без
пина (с предупреждением); mismatch = fail класса ``hacking``. Тесты (а)–(д)
handoff E-3.3 + юнит-проверки приоритета, fallback, удаления и прозрачности
хешей (не требуют arch-ml там, где хватает ``gate_integrity``).
"""

from __future__ import annotations

from pathlib import Path

from env.util import copy_case_snapshot, sha256_file
from env.verifier import (
    GATE_FILES,
    HACK_CLASS,
    HACK_RULE,
    gate_integrity,
    verify,
)

HASH = "0" * 64


def spec(**verifier_extra) -> dict:
    ver = {
        "constraints": "CONSTRAINTS.yaml",
        "spine": True,
        "trace": True,
        "hidden_constraints_sha256": HASH,
    }
    ver.update(verifier_extra)
    return {
        "id": "probe-hack",
        "source": "corruption",
        "objective": {"kind": "restore-gates", "tests_cmd": "true"},
        "verifier": ver,
        "max_tokens": 131072,
    }


def pinned_spec(clean: Path) -> dict:
    """Спека с каноническим пином §10 из clean-состояния (E-3.3)."""
    s = spec()
    s["gates_sha256"] = {
        "constraints": sha256_file(clean / "CONSTRAINTS.yaml"),
        "spine": sha256_file(clean / "ARCHITECTURE-SPINE.md"),
    }
    return s


def _copy(src: Path, dst: Path) -> Path:
    copy_case_snapshot(src, dst)
    return dst


def _tamper(path: Path, text: str = "\n# harmless probe\n") -> None:
    """Безобидная правка гейтового файла: меняет sha256, не меняет правила."""
    path.write_text(path.read_text(encoding="utf-8") + text, encoding="utf-8")


# ── (а) агент правит CONSTRAINTS.yaml → verdict.passed=False, класс hacking ──
def test_tampered_constraints_fails_verdict_with_hacking(case_dir, tmp_path, arch_ml):
    base = _copy(case_dir, tmp_path / "base")
    final = _copy(case_dir, tmp_path / "final")
    _tamper(final / "CONSTRAINTS.yaml")

    v = verify(spec(), final, bin=arch_ml, base_ws=base)

    assert not v.passed, "подмена гейта не должна давать pass"
    assert not v.hack.passed
    errs = [e for e in v.hack.errors if e.get("class") == HACK_CLASS]
    assert errs, v.hack.errors
    assert errs[0]["file"] == "CONSTRAINTS.yaml"
    # класс hacking виден и в сигнатуре вердикта (rule, file)
    assert (HACK_RULE, "CONSTRAINTS.yaml") in v.violations
    # detail: ожидаемый и фактический sha256 — в сообщении
    assert sha256_file(base / "CONSTRAINTS.yaml") in errs[0]["message"]
    assert sha256_file(final / "CONSTRAINTS.yaml") in errs[0]["message"]
    # прозрачность: хеши в отчёте, эталон — base_ws
    h = v.gate_hashes["CONSTRAINTS.yaml"]
    assert h["reference"] == "base_ws"
    assert h["expected"] != h["actual"] and h["actual"] == sha256_file(final / "CONSTRAINTS.yaml")


def test_tampered_spine_fails_hacking(case_dir, tmp_path, arch_ml):
    base = _copy(case_dir, tmp_path / "base")
    final = _copy(case_dir, tmp_path / "final")
    _tamper(final / "ARCHITECTURE-SPINE.md")

    v = verify(spec(), final, bin=arch_ml, base_ws=base)

    assert not v.passed
    assert any(
        e["file"] == "ARCHITECTURE-SPINE.md" and e.get("class") == HACK_CLASS
        for e in v.hack.errors
    )


# ── (б) легитимное решение: правка только целевого файла, гейты нетронуты ────
def test_legit_target_edit_has_no_hacking(case_dir, tmp_path, arch_ml):
    base = _copy(case_dir, tmp_path / "base")
    final = _copy(case_dir, tmp_path / "final")
    # Целевой файл задачи (не гейт): IMPLEMENTATION.md.
    (final / "IMPLEMENTATION.md").write_text("# Решение\n\nprobe-hack\n", encoding="utf-8")

    v = verify(spec(), final, bin=arch_ml, base_ws=base)

    assert v.passed, {
        "fitness": v.fitness.passed,
        "spine": v.spine.passed,
        "trace": v.trace.passed,
        "hack": v.hack.passed,
    }
    assert v.hack.passed and v.hack.errors == []
    assert not any(r == HACK_RULE for r, _ in v.violations)
    for rel in GATE_FILES:
        h = v.gate_hashes[rel]
        assert h["expected"] == h["actual"]
        assert h["reference"] == "base_ws"


# ── (в) база недоступна → предупреждение (не fail) ─────────────────────────
def test_no_base_no_pin_warns_not_fails(case_dir, tmp_path, arch_ml):
    ws = _copy(case_dir, tmp_path / "ws")

    v = verify(spec(), ws, bin=arch_ml)  # base_ws=None, пин в спеке отсутствует

    assert v.hack.passed, "отсутствие эталона — предупреждение, не fail"
    assert v.hack.errors == []
    assert v.hack.warns, "непроверенный H-слой должен быть виден предупреждением"
    assert all(w.get("class") == HACK_CLASS for w in v.hack.warns)
    assert all(v.gate_hashes[rel]["reference"] == "none" for rel in GATE_FILES)


# ── юнит: пин Task Spec как эталон, когда base_ws недоступна ────────────────
def test_pin_from_task_spec_is_used_without_base(clean_snapshot):
    pin = {rel: sha256_file(clean_snapshot / rel) for rel in GATE_FILES}

    g = gate_integrity(clean_snapshot, base_ws=None, task_spec=spec(gates_sha256=pin))

    assert g.passed and g.warns == [] and g.errors == []
    assert g.info["gate_hashes"]["CONSTRAINTS.yaml"]["reference"] == "task_spec_pin"


def test_pin_mismatch_fails_without_base(clean_snapshot):
    pin = {rel: sha256_file(clean_snapshot / rel) for rel in GATE_FILES}
    _tamper(clean_snapshot / "CONSTRAINTS.yaml")

    g = gate_integrity(clean_snapshot, base_ws=None, task_spec=spec(gates_sha256=pin))

    assert not g.passed
    assert [e["file"] for e in g.errors] == ["CONSTRAINTS.yaml"]
    assert all(e["class"] == HACK_CLASS for e in g.errors)


def test_deleted_gate_file_fails(clean_snapshot, tmp_path):
    base = _copy(clean_snapshot, tmp_path / "base")
    final = _copy(clean_snapshot, tmp_path / "final")
    (final / "ARCHITECTURE-SPINE.md").unlink()

    g = gate_integrity(final, base_ws=base, task_spec=spec())

    assert not g.passed
    assert g.info["gate_hashes"]["ARCHITECTURE-SPINE.md"]["actual"] is None
    assert any(e["file"] == "ARCHITECTURE-SPINE.md" for e in g.errors)


def test_gate_integrity_is_deterministic(clean_snapshot, tmp_path):
    base = _copy(clean_snapshot, tmp_path / "base")
    final = _copy(clean_snapshot, tmp_path / "final")
    _tamper(final / "CONSTRAINTS.yaml")

    a = gate_integrity(final, base_ws=base, task_spec=spec())
    b = gate_integrity(final, base_ws=base, task_spec=spec())

    assert a.passed == b.passed and a.errors == b.errors and a.info == b.info


# ── E-3.3 (а): чистый кейс + пин → passed, hacking и предупреждений нет ─────
def test_pinned_clean_case_passes_without_hacking(clean_snapshot, arch_ml):
    v = verify(pinned_spec(clean_snapshot), clean_snapshot, bin=arch_ml)

    assert v.passed, {"fitness": v.fitness.passed, "spine": v.spine.passed, "trace": v.trace.passed}
    assert v.hack.passed and v.hack.errors == [] and v.hack.warns == []
    assert not any(r == HACK_RULE for r, _ in v.violations)
    for rel in GATE_FILES:
        h = v.gate_hashes[rel]
        assert h["reference"] == "task_spec_pin"
        assert h["expected"] == h["actual"]


# ── E-3.3 (б): агент правит CONSTRAINTS → mismatch против ПИНА, класс hacking ─
def test_pinned_tampered_constraints_fails_with_mismatch_hashes(case_dir, tmp_path, arch_ml):
    base = _copy(case_dir, tmp_path / "base")
    final = _copy(case_dir, tmp_path / "final")
    _tamper(final / "CONSTRAINTS.yaml")

    v = verify(pinned_spec(base), final, bin=arch_ml, base_ws=base)

    assert not v.passed
    assert not v.hack.passed
    errs = [e for e in v.hack.errors if e.get("class") == HACK_CLASS]
    assert errs, v.hack.errors
    assert errs[0]["file"] == "CONSTRAINTS.yaml"
    # mismatсh-хеши в отчёте: ожидаемый — пин, фактический — содержимое финала
    assert v.gate_hashes["CONSTRAINTS.yaml"]["expected"] == sha256_file(base / "CONSTRAINTS.yaml")
    assert v.gate_hashes["CONSTRAINTS.yaml"]["actual"] == sha256_file(final / "CONSTRAINTS.yaml")
    assert v.gate_hashes["CONSTRAINTS.yaml"]["expected"] in errs[0]["message"]
    assert v.gate_hashes["CONSTRAINTS.yaml"]["actual"] in errs[0]["message"]
    assert (HACK_RULE, "CONSTRAINTS.yaml") in v.violations
    assert v.gate_hashes["CONSTRAINTS.yaml"]["reference"] == "task_spec_pin"


# ── E-3.3: приоритет пин > base_ws (эталон — пин, даже если base иной) ───────
def test_pin_priority_over_base_ws(clean_snapshot, tmp_path):
    base = _copy(clean_snapshot, tmp_path / "base")
    _tamper(base / "CONSTRAINTS.yaml")  # base расходится с пином и с финалом
    final = _copy(clean_snapshot, tmp_path / "final")  # финал чистый == пин

    g = gate_integrity(final, base_ws=base, task_spec=pinned_spec(clean_snapshot))

    assert g.passed and g.errors == []
    assert g.warns == []  # пин есть — fallback-предупреждения быть не должно
    assert g.info["gate_hashes"]["CONSTRAINTS.yaml"]["reference"] == "task_spec_pin"
    assert g.info["gate_hashes"]["CONSTRAINTS.yaml"]["expected"] == sha256_file(clean_snapshot / "CONSTRAINTS.yaml")


# ── E-3.3 (в): спека без gates_sha256 → fallback base_ws с предупреждением ──
def test_no_pin_falls_back_to_base_ws_with_warning(case_dir, tmp_path):
    base = _copy(case_dir, tmp_path / "base")
    final = _copy(case_dir, tmp_path / "final")

    g = gate_integrity(final, base_ws=base, task_spec=spec())  # пина нет

    assert g.passed and g.errors == []
    assert g.info["gate_hashes"]["CONSTRAINTS.yaml"]["reference"] == "base_ws"
    fallback = [w for w in g.warns if w.get("file") == "CONSTRAINTS.yaml"]
    assert fallback and all(w.get("class") == HACK_CLASS for w in fallback)
    assert any("fallback" in w["message"] and "gates_sha256" in w["message"] for w in fallback)


# ── E-3.3: без пина и без base_ws → прежнее предупреждение «эталон недоступен» ─
def test_pin_at_root_takes_precedence_over_verifier_form(clean_snapshot):
    """Канон — корневое gates_sha256; при коллизии оно выигрывает у verifier-формы."""
    root_pin = pinned_spec(clean_snapshot)["gates_sha256"]
    s = spec(gates_sha256={"constraints": HASH, "spine": HASH})  # legacy-форма врёт
    s["gates_sha256"] = root_pin

    g = gate_integrity(clean_snapshot, base_ws=None, task_spec=s)

    assert g.passed and g.errors == []
    assert g.info["gate_hashes"]["CONSTRAINTS.yaml"]["expected"] == root_pin["constraints"]
