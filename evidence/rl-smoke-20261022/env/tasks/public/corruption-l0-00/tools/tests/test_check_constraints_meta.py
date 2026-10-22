"""Тесты мета-валидатора C-036 (ADR-011, AD-10).

Валидатор — точка отказа всего контура контроля: ошибка в нём даёт ложный
PASS. Поэтому проверки (a)-(c) закреплены фикстурами, а не только прогоном
на рабочем наборе правил кейса.

Обязательные сценарии (из постановки дельты):
  (i)   documentary-страж на AD без [PENDING-EVIDENCE] -> FAIL;
  (ii)  AD с behavioural-стражем -> PASS;
  (iii) AD с [PENDING-EVIDENCE] и documentary-стражем -> PASS;
  (iv)  правило без kind/evidence -> FAIL.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import check_constraints_meta as meta  # noqa: E402  (путь добавляется выше)


#: Заголовки инвариантов спайна: «## AD-11: ...». Источник истины для
#: ожидаемого набора AD — спайн, а не литерал: иначе тест ломается на каждом
#: новом инварианте, хотя мета-валидатор отработал верно.
_AD_HEADING_RE = re.compile(r"^##\s+AD-(\d+)\b", re.MULTILINE)


# --- построение фикстурного кейса -------------------------------------------


def _write_case(
    root: Path,
    constraints: str,
    ads: dict[str, str],
    spine_pending: dict[str, str] | None = None,
) -> Path:
    """Собирает минимальный кейс: правила, карточки AD-*.md, спайн."""
    (root / "model").mkdir(parents=True, exist_ok=True)
    (root / "CONSTRAINTS.yaml").write_text(constraints, encoding="utf-8")

    for ad_id, verified_by in ads.items():
        number = ad_id.removeprefix("AD-")
        (root / "model" / f"{ad_id}-fixtura.md").write_text(
            "---\n"
            f"id: {ad_id}\n"
            "type: ad\n"
            f'title: "Фикстурный инвариант {ad_id}"\n'
            'status: "PROPOSED"\n'
            f"verified_by: [{verified_by}]\n"
            "---\n\n"
            f"- **Rule**: инвариант {ad_id} ради теста AD-{number}.\n",
            encoding="utf-8",
        )

    blocks = []
    for ad_id in ads:
        note = (spine_pending or {}).get(ad_id, "")
        blocks.append(
            f"## {ad_id}: Фикстурный инвариант\n\n"
            "- **Binds**: тест ↔ валидатор\n"
            "- **Prevents**: ложный PASS\n"
            f"- **Rule**: правило теста. Страж: {ads[ad_id]}.\n"
            + (f"- **Pending evidence**: [PENDING-EVIDENCE: {note}]\n" if note else "")
        )
    (root / "ARCHITECTURE-SPINE.md").write_text(
        "# SPINE (фикстура)\n\n" + "\n".join(blocks), encoding="utf-8"
    )
    return root


def _behavioural_rule(rule_id: str = "C-100", severity: str = "critical") -> str:
    return (
        f"  - id: {rule_id}\n"
        f'    name: "фикстура: предметная проверка"\n'
        "    type: command_succeeds\n"
        "    command: 'true'\n"
        "    timeout_secs: 10\n"
        f"    severity: {severity}\n"
        "    kind: behavioural\n"
        '    evidence: "command: true — проверяет исполнение"\n'
    )


def _documentary_rule(rule_id: str = "C-101", severity: str = "critical") -> str:
    return (
        f"  - id: {rule_id}\n"
        f'    name: "фикстура: декоративная проверка надписи"\n'
        "    type: must_contain\n"
        '    glob: "ARCHITECTURE-SPINE.md"\n'
        '    pattern: "инвариант"\n'
        f"    severity: {severity}\n"
        "    kind: documentary\n"
        '    evidence: "ARCHITECTURE-SPINE.md — строка «инвариант»"\n'
    )


def _run(root: Path, *extra: str) -> int:
    return meta.main(["--case-dir", str(root), *extra])


# --- (i) декоративный страж инварианта --------------------------------------


def test_documentary_guard_on_ad_without_pending_fails(tmp_path: Path) -> None:
    """(i) must_contain на AD без пометки — красный гейт, код META-C."""
    case = _write_case(
        tmp_path,
        "constraints:\n" + _documentary_rule(),
        {"AD-1": "C-101"},
    )

    report = meta.evaluate_case(case)
    codes = {finding.code for finding in report.findings}

    assert not report.ok
    assert "META-C-DOCUMENTARY-ON-AD" in codes
    assert "META-B-NO-BEHAVIOURAL" in codes
    assert _run(case) == 1
    # --report печатает то же, но не красит гейт
    assert _run(case, "--report") == 0


# --- (ii) инвариант с поведенческим стражем ---------------------------------


def test_ad_with_behavioural_guard_passes(tmp_path: Path) -> None:
    """(ii) behavioural-страж закрывает инвариант без пометки."""
    case = _write_case(
        tmp_path,
        "constraints:\n" + _behavioural_rule(),
        {"AD-1": "C-100"},
    )

    report = meta.evaluate_case(case)

    assert report.ok, [finding.message for finding in report.findings]
    assert report.behavioural_ads == ["AD-1"]
    assert _run(case) == 0


# --- (iii) пометка снимает запрет documentary -------------------------------


def test_pending_evidence_allows_documentary_guard(tmp_path: Path) -> None:
    """(iii) AD с [PENDING-EVIDENCE] и documentary-стражем — зелёный."""
    case = _write_case(
        tmp_path,
        "constraints:\n" + _documentary_rule(),
        {"AD-1": "C-101"},
        spine_pending={"AD-1": "поведенческий страж появится до гейта A5"},
    )

    report = meta.evaluate_case(case)

    assert report.ok, [finding.message for finding in report.findings]
    assert report.pending_ads == ["AD-1"]
    assert report.behavioural_ads == []
    assert _run(case) == 0


def test_pending_marker_in_inline_code_is_not_a_declaration(tmp_path: Path) -> None:
    """Пометка, упомянутая в прозе, не считается декларацией."""
    case = _write_case(
        tmp_path,
        "constraints:\n" + _documentary_rule(),
        {"AD-1": "C-101"},
    )
    spine = case / "ARCHITECTURE-SPINE.md"
    spine.write_text(
        spine.read_text(encoding="utf-8")
        + "- **Rule**: механизм описывается как `[PENDING-EVIDENCE: <что и когда>]`.\n",
        encoding="utf-8",
    )

    report = meta.evaluate_case(case)

    assert not report.ok
    assert report.pending_ads == []
    assert {finding.code for finding in report.findings} >= {"META-C-DOCUMENTARY-ON-AD"}


def test_pending_marker_outside_ad_block_is_ignored(tmp_path: Path) -> None:
    """Пометка формата в шапке спайна — легенда, а не декларация инварианта."""
    case = _write_case(
        tmp_path,
        "constraints:\n" + _documentary_rule(),
        {"AD-1": "C-101"},
    )
    spine = case / "ARCHITECTURE-SPINE.md"
    spine.write_text(
        "# SPINE (фикстура)\n\n"
        "> Легенда: [PENDING-EVIDENCE: <что и когда появится>] — честная декларация.\n\n"
        + spine.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    report = meta.evaluate_case(case)

    assert report.pending_ads == []
    assert {finding.code for finding in report.findings} >= {"META-C-DOCUMENTARY-ON-AD"}


# --- (iv) правило без kind/evidence -----------------------------------------


def test_rule_without_kind_and_evidence_fails(tmp_path: Path) -> None:
    """(iv) блокирующее правило без классификации — ошибка конфигурации."""
    case = _write_case(
        tmp_path,
        "constraints:\n"
        "  - id: C-100\n"
        '    name: "фикстура: правило без метаданных"\n'
        "    type: must_contain\n"
        '    glob: "ARCHITECTURE-SPINE.md"\n'
        '    pattern: "инвариант"\n'
        "    severity: critical\n",
        {"AD-1": "C-100"},
    )

    report = meta.evaluate_case(case)
    codes = {finding.code for finding in report.findings}

    assert not report.ok
    assert "META-A-KIND-MISSING" in codes
    assert "META-A-EVIDENCE-MISSING" in codes
    assert _run(case) == 1


def test_invalid_kind_is_rejected(tmp_path: Path) -> None:
    """Классификация вне перечня ADR-011 — ошибка, а не «своё слово»."""
    case = _write_case(
        tmp_path,
        "constraints:\n"
        "  - id: C-103\n"
        '    name: "фикстура: выдуманная классификация"\n'
        "    type: must_contain\n"
        '    glob: "ARCHITECTURE-SPINE.md"\n'
        '    pattern: "инвариант"\n'
        "    severity: high\n"
        "    kind: ornamental\n"
        '    evidence: "ARCHITECTURE-SPINE.md"\n'
        + _behavioural_rule("C-104"),
        {"AD-1": "C-104"},
    )

    report = meta.evaluate_case(case)
    codes = {finding.code for finding in report.findings}

    assert "META-A-KIND-INVALID" in codes
    assert not report.ok


def test_low_severity_rule_needs_no_metadata(tmp_path: Path) -> None:
    """Метаданные обязательны только для high|critical — иначе шум в гейте."""
    case = _write_case(
        tmp_path,
        "constraints:\n" + _behavioural_rule("C-105")
        + "  - id: C-106\n"
        '    name: "фикстура: advisory-правило"\n'
        "    type: must_contain\n"
        '    glob: "ARCHITECTURE-SPINE.md"\n'
        '    pattern: "инвариант"\n'
        "    severity: medium\n",
        {"AD-1": "C-105"},
    )

    report = meta.evaluate_case(case)

    assert report.ok, [finding.message for finding in report.findings]


# --- ссылка на несуществующее правило ---------------------------------------


def test_verified_by_unknown_rule_is_warning_and_blocks_ad(tmp_path: Path) -> None:
    """verified_by на отсутствующее правило: страж не существует -> FAIL + warn."""
    case = _write_case(
        tmp_path,
        "constraints:\n" + _behavioural_rule("C-107"),
        {"AD-1": "C-999"},
    )

    report = meta.evaluate_case(case)

    assert not report.ok
    assert {finding.code for finding in report.findings} == {"META-B-NO-BEHAVIOURAL"}
    assert {warning.code for warning in report.warnings} == {"META-REF-UNKNOWN-RULE"}
    # предупреждение само по себе гейт не красит
    assert _run(case, "--report") == 0


# --- рабочий набор правил кейса ---------------------------------------------


def test_case_ruleset_is_fully_classified_and_prints_all_ads() -> None:
    """Регрессия на кейс: все AD спайна в таблице, все правила имеют kind и evidence."""
    case_dir = Path(__file__).resolve().parents[2]

    spine = (case_dir / "ARCHITECTURE-SPINE.md").read_text(encoding="utf-8")
    declared = {f"AD-{number}" for number in _AD_HEADING_RE.findall(spine)}
    assert declared, "спайн кейса не объявляет ни одного инварианта"

    report = meta.evaluate_case(case_dir)
    rendered = meta.render(report)

    # ровно объявленный набор: пропущенный или лишний AD — красный тест
    assert {row.ad_id for row in report.rows} == declared
    assert not [
        rule for rule in report.rules if rule.enforced and rule.kind is None
    ]
    assert not [
        rule for rule in report.rules if rule.enforced and not (rule.evidence or "").strip()
    ]
    assert not [ad_id for ad_id in declared if ad_id not in rendered]

    classified = {rule.rule_id for rule in report.rules if rule.kind == "behavioural"}
    unguarded = {
        ad.ad_id
        for ad in report.ads
        if not ad.pending_evidence
        and not classified.intersection(ad.verified_by)
    }
    # AD-4 — единственный незакрытый инвариант этой дельты: его поведенческий
    # страж вводится в параллельной ветке a4-manifest (C-038, манифест A4),
    # здесь AD-4 запрещено трогать. Все прочие AD закрыты пометкой или стражем.
    assert unguarded <= {"AD-4"}, unguarded
