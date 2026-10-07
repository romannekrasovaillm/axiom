"""§9 «Смысловая порча»: смысловые атомы мутатора и их связка со стражем C-047.

Проверяется договор амендмента 06.10 (ENVIRONMENT-V1 §9):

* **детерминизм** — один seed даёт байт-в-байт один и тот же повреждённый кейс;
* **валидность** — каждый атом, применённый в одиночку, либо ловится смысловым
  стражем (``numeric_drift`` / ``term_swap`` / ``config_drift`` → C-047,
  ``adr-config-mismatch``), либо НЕ ловится структурными правилами
  (``claim_inversion`` — класс модель-детекции, задокументировано);
* **обратимость** — откат возвращает файл байт-в-байт, причём смысловые атомы
  восстанавливаются ОДНИМ ``Damage.meta`` (без доступа к чистому кейсу): это и
  есть «восстановим 4 инструментами §13»;
* **состав уровней** — L2 = структурные + numeric_drift + term_swap;
  L3 = L2 + claim_inversion + config_drift; L0/L1 не изменились.

Структурная невидимость claim_inversion проверяется дифференциально: набор
исходов всех content-правил кейсового ``CONSTRAINTS.yaml`` до и после порчи
обязан совпасть (правило, которому всё равно, — не страж смысла). Полный прогон
вердикта (fitness ∧ spine ∧ trace бинарём ``arch-ml``) для структурных атомов
покрыт ``test_corruption.py::test_damage_detectable_by_gates``; здесь он не
дублируется — смысловые атомы структурные гейты не задевают по построению.
"""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path

import pytest
import yaml

from env import corruption
from env.util import copy_case_snapshot

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import check_adr_config_consistency as guard  # noqa: E402  (путь добавляется выше)

#: Проверяемые seed'ы: разные ветвления выбора цели (файл/секция/поле/фраза).
SEEDS = (0, 3, 7, 42)

#: Атомы, детектируемые стражем C-047 (смысловая сверка ADR↔config).
C047_ATOMS = ("numeric_drift", "term_swap", "config_drift")

#: Content-правила CONSTRAINTS.yaml — то, что видит структурный гейт.
_CONTENT_RULE_TYPES = ("must_contain", "must_not_contain", "each_file_must_contain")


def _files_equal(a: Path, b: Path) -> bool:
    fa = sorted(p.relative_to(a).as_posix() for p in a.rglob("*") if p.is_file())
    fb = sorted(p.relative_to(b).as_posix() for p in b.rglob("*") if p.is_file())
    if fa != fb:
        return False
    return all((a / rel).read_bytes() == (b / rel).read_bytes() for rel in fa)


def _snapshot(case_dir: Path, dst: Path) -> Path:
    copy_case_snapshot(case_dir, dst)
    return dst


def _only(case_dir: Path, seed: int, level: str, kind: str) -> corruption.Damage:
    """Единственный запланированный атом вида ``kind`` (остальные не применяются)."""
    damages = [d for d in corruption.plan_damages(case_dir, seed, level) if d.kind == kind]
    assert len(damages) == 1, f"{kind}: ожидался ровно один атом на уровне {level}, получено {len(damages)}"
    return damages[0]


def _content_rule_outcomes(root: Path) -> dict[str, bool]:
    """Исходы content-правил кейсового ruleset над деревом ``root``.

    Зеркалит ту часть фитнес-гейта, которая читает ТЕКСТ правил (наличие фразы),
    а не поведение: именно её и обходит смысловая порча. Сравнение идёт «до/после»
    одним и тем же кодом, поэтому значим только дифференциал, а не абсолют.
    """
    spec = yaml.safe_load((root / "CONSTRAINTS.yaml").read_text(encoding="utf-8"))
    outcomes: dict[str, bool] = {}
    for rule in spec["constraints"]:
        kind = rule.get("type")
        if kind not in _CONTENT_RULE_TYPES:
            continue
        rx = re.compile(rule["pattern"])
        files = [p for p in root.glob(rule["glob"]) if p.is_file()]
        hits = [bool(rx.search(p.read_text(encoding="utf-8"))) for p in files]
        if kind == "must_contain":
            outcomes[rule["id"]] = any(hits)
        elif kind == "each_file_must_contain":
            outcomes[rule["id"]] = all(hits)
        else:
            outcomes[rule["id"]] = not any(hits)
    return outcomes


# ── состав уровней ───────────────────────────────────────────────────────────

def test_level_atoms_composition():
    """L0/L1 не изменились; L2/L3 = структурные + смысловые по амендменту §9."""
    assert corruption.LEVEL_ATOMS["L0"] == ["remove_adr_section"]
    assert corruption.LEVEL_ATOMS["L1"] == [
        "remove_adr_section", "break_affects", "break_verified_by",
    ]

    structural = list(corruption.STRUCTURAL_ATOMS)
    assert corruption.LEVEL_ATOMS["L2"] == [*structural, "numeric_drift", "term_swap"]
    assert corruption.LEVEL_ATOMS["L3"] == [
        *structural, "numeric_drift", "term_swap", "claim_inversion", "config_drift",
    ]
    assert set(corruption.LEVEL_ATOMS["L2"]) <= set(corruption.LEVEL_ATOMS["L3"])


# ── детерминизм ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("seed", SEEDS)
def test_semantic_plan_deterministic_by_seed(case_dir, seed):
    """Один seed → один план (даже там, где выбирается поле/фраза/секция)."""
    first = corruption.plan_damages(case_dir, seed, "L3")
    second = corruption.plan_damages(case_dir, seed, "L3")
    assert first == second
    assert [d.kind for d in first] == corruption.LEVEL_ATOMS["L3"]


def test_semantic_plan_varies_with_seed(case_dir):
    """Разные seed'ы дают разные цели смысловых атомов (в противном случае
    детерминизм выродился бы в константу и лесенка не масштабировалась бы)."""
    plans = {
        seed: tuple(
            (d.kind, str(d.meta.get("field") or d.meta.get("old") or d.meta.get("swap")))
            for d in corruption.plan_damages(case_dir, seed, "L3")
            if d.kind in corruption.SEMANTIC_ATOMS
        )
        for seed in range(12)
    }
    assert len(set(plans.values())) > 1, f"все seed'ы дали один план: {plans}"


def test_semantic_targets_come_from_guard_map(case_dir):
    """Цель смысловых атомов берётся из карты C-047, а не из зашитых констант."""
    import adr_config_map as acm

    spec = acm.load(TOOLS_DIR / "adr_config_map.yaml")
    known = {m["id"] for m in acm.get_mappings(spec)}
    swap_ids = {swap["id"] for swap in acm.get_swaps(spec)}

    for seed in SEEDS:
        for damage in corruption.plan_damages(case_dir, seed, "L3"):
            meta = damage.meta or {}
            if damage.kind in ("numeric_drift", "config_drift"):
                assert meta["mapping"] in known, damage
            elif damage.kind == "term_swap":
                assert meta["swap"] in swap_ids, damage
                assert {f["field"] for f in meta["fields"]} <= {
                    m["config_field"] for m in acm.get_mappings(spec)
                }


# ── связка: порча → страж краснеет, откат → зелёный ──────────────────────────

@pytest.mark.parametrize("kind", C047_ATOMS)
@pytest.mark.parametrize("seed", SEEDS)
def test_c047_detects_semantic_drift(case_dir, tmp_path, kind, seed):
    """numeric_drift/term_swap/config_drift ловятся стражем; откат снимает находку."""
    ws = _snapshot(case_dir, tmp_path / f"ws-{kind}-{seed}")
    damage = _only(case_dir, seed, "L3", kind)
    corruption.apply_damage(ws, damage)

    code, report = guard.evaluate(ws)
    assert code == 1, f"{kind}: страж обязан краснеть, отчёт: {report}"
    assert {f["class"] for f in report["findings"]} == {"adr-config-mismatch"}
    touched = {f["field"] for f in damage.meta.get("fields", [])} or {damage.meta["field"]}
    assert touched <= {f["config_field"] for f in report["findings"]}

    corruption.revert_damage(ws, case_dir, damage)
    code_after, report_after = guard.evaluate(ws)
    assert code_after == 0, f"{kind}: после отката страж обязан позеленеть, отчёт: {report_after}"


def test_c047_detects_config_side_drift(case_dir, tmp_path):
    """config_drift — вторая сторона рассинхрона: страж видит именно config-число."""
    ws = _snapshot(case_dir, tmp_path / "ws-config")
    damage = _only(case_dir, 0, "L3", "config_drift")
    corruption.apply_damage(ws, damage)
    code, report = guard.evaluate(ws)
    assert code == 1
    finding = next(f for f in report["findings"] if f["config_field"] == damage.meta["field"])
    assert finding["config_value"] == damage.meta["new"]
    assert finding["adr_value"] == damage.meta["old"]


def test_term_swap_mismatches_both_fields(case_dir, tmp_path):
    """Своп роняет согласованность ОБОИХ полей карты — обмен виден только сверкой."""
    ws = _snapshot(case_dir, tmp_path / "ws-swap")
    damage = _only(case_dir, 0, "L3", "term_swap")
    corruption.apply_damage(ws, damage)
    code, report = guard.evaluate(ws)
    assert code == 1
    mismatched = {f["config_field"] for f in report["findings"]}
    assert {f["field"] for f in damage.meta["fields"]} <= mismatched


def test_claim_inversion_invisible_to_structural_gates(case_dir, tmp_path):
    """claim_inversion невидима структурным правилам И стражу C-047.

    Проверяется дифференциально: набор исходов content-правил кейсового
    ``CONSTRAINTS.yaml`` не меняется, а числа ADR остаются на месте (страж молчит).
    Это и есть объявленный класс детекции — модель-понимание, а не grep (§9).
    """
    clean = _snapshot(case_dir, tmp_path / "clean")
    ws = _snapshot(case_dir, tmp_path / "ws-claim")
    damage = _only(case_dir, 0, "L3", "claim_inversion")
    corruption.apply_damage(ws, damage)

    before = _content_rule_outcomes(clean)
    after = _content_rule_outcomes(ws)
    assert before == after, (
        "структурные content-правила заметили claim_inversion: "
        f"{[k for k in before if before[k] != after[k]]}"
    )

    code, report = guard.evaluate(ws)
    assert code == 0, f"числа не трогались, страж обязан молчать: {report}"

    # Инверсия действительно произошла: якорь «до» исчез, якорь «после» появился
    # (для отрицания старое слово остаётся внутри нового — сравнение по якорю).
    assert damage.meta["old"] != damage.meta["new"]
    text = (ws / damage.file).read_text(encoding="utf-8")
    old_anchor = damage.meta["prefix"] + damage.meta["old"] + damage.meta["suffix"]
    new_anchor = damage.meta["prefix"] + damage.meta["new"] + damage.meta["suffix"]
    assert new_anchor in text and old_anchor not in text

    corruption.revert_damage(ws, case_dir, damage)
    assert (ws / damage.file).read_bytes() == (clean / damage.file).read_bytes()


def test_claim_inversion_pair_mode(case_dir, tmp_path):
    """Режим словаря пар («обязательно» → «запрещено»).

    В самом ADR-009 ветвление ``pair`` кандидатов не имеет (там только
    глаголы-требования), поэтому пара проверяется на синтетическом ADR под тот
    же glob карты: механизм обязан работать, даже если текст решения его пока
    не задействует.
    """
    case = tmp_path / "synthetic"
    (case / "docs" / "adr").mkdir(parents=True)
    (case / "tools").mkdir()
    (case / "tools" / "adr_config_map.yaml").write_bytes(
        (TOOLS_DIR / "adr_config_map.yaml").read_bytes()
    )
    adr = case / "docs" / "adr" / "ADR-009-synthetic.md"
    clean_text = "# ADR-009. Синтетика\n\n## Decision\n\nРазрешение обязательно.\n"
    adr.write_text(clean_text, encoding="utf-8")

    planner = corruption._SemanticPlanner(case, random.Random(0))
    damage = planner.claim_inversion()
    assert damage.meta["mode"] == "pair"
    assert damage.meta["old"] == "обязательно" and damage.meta["new"] == "запрещено"
    assert damage.section == "## Decision"

    corruption.apply_damage(case, damage)
    assert "Разрешение запрещено." in adr.read_text(encoding="utf-8")
    corruption.revert_damage(case, Path("/nonexistent"), damage)
    assert adr.read_text(encoding="utf-8") == clean_text


# ── обратимость ──────────────────────────────────────────────────────────────

def test_semantic_atoms_recoverable_by_edit_file_alone(case_dir, tmp_path):
    """Все четыре смысловых атома восстанавливаются ОДНИМ ``Damage.meta``.

    ``revert_damage`` вызывается с заведомо несуществующим ``clean_dir``: если
    атом подглядывает в чистый кейс, тест падает. Так проверяется обещание §9
    «все четыре восстановимы 4 инструментами (edit_file обратной заменой)» и
    сохраняется граница ``break_ad_link`` (создание файла агенту v1 недоступно).
    """
    clean = _snapshot(case_dir, tmp_path / "clean")
    ws = _snapshot(case_dir, tmp_path / "ws")
    damages = [
        d for d in corruption.plan_damages(case_dir, 0, "L3") if d.kind in corruption.SEMANTIC_ATOMS
    ]
    assert {d.kind for d in damages} == set(corruption.SEMANTIC_ATOMS)
    for damage in damages:
        assert damage.meta, f"{damage.kind}: Damage.meta обязателен для восстановления"
        corruption.apply_damage(ws, damage)
    assert not _files_equal(ws, clean)

    for damage in damages:
        corruption.revert_damage(ws, Path("/nonexistent-clean"), damage)
    assert _files_equal(ws, clean)


@pytest.mark.parametrize("level", ("L0", "L1", "L2", "L3"))
def test_full_level_revert_is_byte_identical(case_dir, tmp_path, level):
    """L3 держит на одном файле (ADR-009) три атома — откат обязан быть пофайловым."""
    clean = _snapshot(case_dir, tmp_path / "clean")
    for seed in SEEDS:
        ws = _snapshot(case_dir, tmp_path / f"ws-{level}-{seed}")
        damages = corruption.corrupt(case_dir, ws, seed=seed, level=level)
        assert damages
        for damage in damages:
            corruption.revert_damage(ws, case_dir, damage)
        assert _files_equal(ws, clean), f"{level}/{seed}: откат не вернул чистое состояние"


@pytest.mark.parametrize("level", ("L2", "L3"))
def test_shared_adr_keeps_all_semantic_signatures(case_dir, tmp_path, level):
    """Три атома делят один ADR-009 — все три подписи обязаны дожить до конца порчи.

    Иначе смысловая порча сама себя съедала бы: ``remove_adr_section`` унёс бы
    секцию с числом ``numeric_drift``, а порча стала бы недетектируемой.
    """
    ws = _snapshot(case_dir, tmp_path / f"ws-{level}")
    damages = corruption.corrupt(case_dir, ws, seed=7, level=level)
    semantic = [d for d in damages if d.kind in corruption.SEMANTIC_ATOMS]
    assert semantic, "L2/L3 обязаны нести смысловые атомы"
    assert len({d.file for d in semantic if d.file.endswith(".md")}) == 1, (
        "смысловые атомы ADR обязаны делить один файл решения"
    )
    text = (ws / semantic[0].file).read_text(encoding="utf-8") if semantic[0].file.endswith(".md") else ""
    for damage in semantic:
        if damage.kind == "numeric_drift":
            match = re.search(damage.meta["pattern"], text)
            assert match and int(match.group(1)) == damage.meta["new"]
        elif damage.kind == "term_swap":
            for field in damage.meta["fields"]:
                match = re.search(field["pattern"], text)
                assert match and int(match.group(1)) == field["new"]
        elif damage.kind == "claim_inversion":
            assert damage.meta["new"] in text


def test_config_drift_never_touches_tokenizer_hash(case_dir, tmp_path):
    """``tokenizer_hash`` — пин генератора: смысловая порча config его не трогает."""
    import json

    pin = json.loads((case_dir / "net" / "config.json").read_text(encoding="utf-8"))["tokenizer_hash"]
    planned: list[str] = []
    for seed in range(30):
        ws = _snapshot(case_dir, tmp_path / f"ws-{seed}")
        damages = corruption.corrupt(case_dir, ws, seed=seed, level="L3")
        for damage in damages:
            if damage.kind != "config_drift":
                continue
            planned.append(damage.meta["field"])
            assert damage.meta["field"] != "tokenizer_hash"
        after = json.loads((ws / "net" / "config.json").read_text(encoding="utf-8"))
        assert after["tokenizer_hash"] == pin, f"seed={seed}: tokenizer_hash сдвинут"

    import adr_config_map as acm

    spec = acm.load(TOOLS_DIR / "adr_config_map.yaml")
    assert "tokenizer_hash" in acm.frozen_config_fields(spec)
    assert set(planned) & {"tokenizer_hash"} == set()
    assert planned, "ни один seed не дал config_drift — проверка вырождена"
