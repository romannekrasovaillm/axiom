#!/usr/bin/env python3
"""C-036 — мета-валидатор контура стражей кейса (ADR-011, AD-10).

Проверяет не поведение системы, а конфигурацию контроля: что у каждого
критичного правила есть классификация (`kind`) и доказательство (`evidence`),
что у каждого AD-инварианта есть поведенческий страж либо честная пометка
`[PENDING-EVIDENCE: ...]` в спайне, и что документарный страж не выдаётся
за доказательство инварианта.

Входы (относительно каталога кейса):
  * ``CONSTRAINTS.yaml``        — правила с полями ``id``/``name``/``type``/
                                  ``severity``/``kind``/``evidence``;
  * ``model/AD-*.md``           — карточки инвариантов, поле ``verified_by``;
  * ``ARCHITECTURE-SPINE.md``   — блоки ``## AD-n`` с пометками
                                  ``[PENDING-EVIDENCE: <что и когда>]``.

Запуск::

    python3 tools/check_constraints_meta.py [--report]

Коды возврата: ``0`` — нарушений нет; ``1`` — есть. ``--report`` печатает
таблицу и находки, но всегда возвращает ``0`` (режим отчёта, не гейта).
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover — среда без PyYAML: валидатор бесполезен
    print(
        "check_constraints_meta: нужен PyYAML (pip install pyyaml)",
        file=sys.stderr,
    )
    raise SystemExit(1)


#: Допустимые классификации стража (ADR-011, п. 1).
KINDS = ("behavioural", "structural", "documentary")

#: Шкала severity кейса; метаданные обязательны только для блокирующих правил.
ENFORCED_SEVERITIES = ("high", "critical")

_AD_HEADING_RE = re.compile(r"^##\s+(AD-\d+)\b.*$", re.MULTILINE)
_PENDING_RE = re.compile(r"\[PENDING-EVIDENCE:\s*(.+?)\]", re.DOTALL)
#: Упоминание пометки в прозе (в инлайн-коде ``[PENDING-EVIDENCE: ...]``) —
#: не декларация: спайн AD-10 описывает сам механизм пометки.
_INLINE_CODE_RE = re.compile(r"`[^`]*`", re.DOTALL)
_FENCED_CODE_RE = re.compile(r"^```.*?^```", re.DOTALL | re.MULTILINE)


class MetaError(Exception):
    """Вход не читается или не является ожидаемым артефактом."""


@dataclass(frozen=True)
class Rule:
    """Одно правило ``CONSTRAINTS.yaml`` в срезе, нужном мета-валидатору."""

    rule_id: str
    name: str
    type: str
    severity: str
    kind: str | None
    evidence: str | None

    @property
    def enforced(self) -> bool:
        """Правило блокирующее — значит обязано нести ``kind`` и ``evidence``."""
        return self.severity.strip().lower() in ENFORCED_SEVERITIES


@dataclass
class Ad:
    """AD-инвариант: карточка модели + пометка из спайна."""

    ad_id: str
    path: Path
    verified_by: list[str] = field(default_factory=list)
    pending_evidence: str | None = None


@dataclass(frozen=True)
class Finding:
    """Нарушение (a)-(c). Каждое — красный гейт."""

    code: str
    message: str


@dataclass(frozen=True)
class Warning:
    """Замечание, не красящее гейт: конфигурация исполнима, но подозрительна."""

    code: str
    message: str


@dataclass(frozen=True)
class Row:
    """Строка таблицы «AD → страж → kind → evidence → статус»."""

    ad_id: str
    guard: str
    kind: str
    evidence: str
    status: str


@dataclass
class Report:
    ads: list[Ad]
    rules: list[Rule]
    rows: list[Row]
    findings: list[Finding]
    warnings: list[Warning]

    @property
    def ok(self) -> bool:
        return not self.findings

    @property
    def behavioural_ads(self) -> list[str]:
        return sorted(
            {
                row.ad_id
                for row in self.rows
                if row.kind == "behavioural" and row.status.startswith("OK")
            },
            key=_ad_sort_key,
        )

    @property
    def pending_ads(self) -> list[str]:
        return sorted(
            {ad.ad_id for ad in self.ads if ad.pending_evidence},
            key=_ad_sort_key,
        )


def _ad_sort_key(ad_id: str) -> int:
    match = re.search(r"(\d+)", ad_id)
    return int(match.group(1)) if match else 0


def _strip_code(text: str) -> str:
    """Убирает код-спаны: пометка в примере — не декларация инварианта."""
    return _INLINE_CODE_RE.sub(" ", _FENCED_CODE_RE.sub(" ", text))


def load_rules(constraints_path: Path) -> list[Rule]:
    """Читает правила из ``constraints:``/``rules:`` (оба корня — схема харнесса)."""
    try:
        raw = yaml.safe_load(constraints_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise MetaError(f"нет файла правил: {constraints_path}") from exc
    except yaml.YAMLError as exc:
        raise MetaError(f"{constraints_path}: невалидный YAML — {exc}") from exc
    if not isinstance(raw, dict):
        raise MetaError(f"{constraints_path}: ожидался словарь верхнего уровня")

    entries: list[dict] = []
    for key in ("rules", "constraints"):
        value = raw.get(key)
        if value is None:
            continue
        if not isinstance(value, list):
            raise MetaError(f"{constraints_path}: секция '{key}' должна быть списком")
        entries.extend(entry for entry in value if isinstance(entry, dict))

    rules: list[Rule] = []
    for entry in entries:
        rule_id = str(entry.get("id") or entry.get("name") or "").strip()
        if not rule_id:
            raise MetaError(f"{constraints_path}: правило без id/name")
        rules.append(
            Rule(
                rule_id=rule_id,
                name=str(entry.get("name") or ""),
                type=str(entry.get("type") or ""),
                severity=str(entry.get("severity") or "error"),
                kind=(str(entry["kind"]).strip() if entry.get("kind") is not None else None),
                evidence=(
                    str(entry["evidence"]) if entry.get("evidence") is not None else None
                ),
            )
        )
    return rules


def _split_frontmatter(text: str, path: Path) -> dict:
    if not text.startswith("---"):
        raise MetaError(f"{path}: нет YAML-frontmatter")
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise MetaError(f"{path}: frontmatter не закрыт")
    data = yaml.safe_load(parts[1])
    if not isinstance(data, dict):
        raise MetaError(f"{path}: frontmatter — не словарь")
    return data


def load_ads(model_dir: Path) -> list[Ad]:
    """Читает карточки ``model/AD-*.md`` (ADR-файлы под glob не попадают)."""
    ads: list[Ad] = []
    for path in sorted(model_dir.glob("AD-*.md")):
        data = _split_frontmatter(path.read_text(encoding="utf-8"), path)
        if str(data.get("type") or "ad") != "ad":
            continue
        ad_id = str(data.get("id") or path.stem).strip()
        verified_by = data.get("verified_by") or []
        if isinstance(verified_by, str):
            verified_by = [verified_by]
        ads.append(
            Ad(
                ad_id=ad_id,
                path=path,
                verified_by=[str(ref).strip() for ref in verified_by],
            )
        )
    if not ads:
        raise MetaError(f"{model_dir}: не найдено ни одной карточки AD-*.md")
    return sorted(ads, key=lambda ad: _ad_sort_key(ad.ad_id))


def load_pending(spine_path: Path) -> dict[str, str]:
    """Собирает пометки ``[PENDING-EVIDENCE: ...]`` по блокам ``## AD-n``."""
    try:
        text = spine_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise MetaError(f"нет спайна: {spine_path}") from exc

    matches = list(_AD_HEADING_RE.finditer(text))
    pending: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = _strip_code(text[match.end() : end])
        note = _PENDING_RE.search(block)
        if note and note.group(1).strip():
            pending[match.group(1)] = " ".join(note.group(1).split())
    return pending


def _check_rule_metadata(rules: list[Rule]) -> list[Finding]:
    """(a) У каждого блокирующего правила есть kind и непустой evidence."""
    findings: list[Finding] = []
    for rule in rules:
        if not rule.enforced:
            continue
        if rule.kind is None:
            findings.append(
                Finding(
                    "META-A-KIND-MISSING",
                    f"{rule.rule_id}: severity '{rule.severity}' без поля kind "
                    f"(ожидается одно из {', '.join(KINDS)})",
                )
            )
        elif rule.kind not in KINDS:
            findings.append(
                Finding(
                    "META-A-KIND-INVALID",
                    f"{rule.rule_id}: kind '{rule.kind}' вне "
                    f"{'/'.join(KINDS)} — правило без классификации",
                )
            )
        if not (rule.evidence or "").strip():
            findings.append(
                Finding(
                    "META-A-EVIDENCE-MISSING",
                    f"{rule.rule_id}: поле evidence пусто — страж без доказательства",
                )
            )
    return findings


def _check_ads(ads: list[Ad], rules: list[Rule]) -> tuple[list[Finding], list[Warning], list[Row]]:
    """(b)-(c) Инвариант защищён поведением или честно помечен."""
    by_id = {rule.rule_id: rule for rule in rules}
    findings: list[Finding] = []
    warnings: list[Warning] = []
    rows: list[Row] = []

    for ad in ads:
        guards = [by_id[ref] for ref in ad.verified_by if ref in by_id]
        unknown = [ref for ref in ad.verified_by if ref not in by_id]
        for ref in unknown:
            warnings.append(
                Warning(
                    "META-REF-UNKNOWN-RULE",
                    f"{ad.ad_id}: verified_by ссылается на '{ref}', "
                    "которого нет в CONSTRAINTS.yaml — страж не существует",
                )
            )

        behavioural = [g for g in guards if g.kind == "behavioural"]
        documentary = [g for g in guards if g.kind == "documentary"]
        pending = ad.pending_evidence
        # (b) инвариант без поведенческого стража — только с пометкой.
        no_behavioural = not behavioural and not pending
        # (c) documentary-страж инварианта допустим только под пометкой.
        decorative = bool(documentary) and not pending

        if no_behavioural:
            findings.append(
                Finding(
                    "META-B-NO-BEHAVIOURAL",
                    f"{ad.ad_id}: нет behavioural-стража и нет "
                    "[PENDING-EVIDENCE] в ARCHITECTURE-SPINE.md — "
                    "декоративный контроль инварианта",
                )
            )
        if decorative:
            findings.append(
                Finding(
                    "META-C-DOCUMENTARY-ON-AD",
                    f"{ad.ad_id}: documentary-страж "
                    f"({', '.join(g.rule_id for g in documentary)}) "
                    "у инварианта без [PENDING-EVIDENCE] — "
                    "проверяет надпись, а не выполнение",
                )
            )

        reasons: list[str] = []
        if no_behavioural:
            reasons.append("нет behavioural-стража и нет [PENDING-EVIDENCE]")
        if decorative:
            reasons.append(
                f"documentary-страж без пометки ({', '.join(g.rule_id for g in documentary)})"
            )
        if reasons:
            status = "FAIL: " + "; ".join(reasons)
        elif behavioural:
            status = "OK: behavioural-страж"
        else:
            status = "OK: [PENDING-EVIDENCE]"

        if not guards:
            rows.append(
                Row(
                    ad_id=ad.ad_id,
                    guard="—",
                    kind="—",
                    evidence=(
                        pending if pending else "нет привязки в verified_by"
                    ),
                    status=status,
                )
            )
            continue
        for guard in guards:
            rows.append(
                Row(
                    ad_id=ad.ad_id,
                    guard=guard.rule_id,
                    kind=guard.kind or "—",
                    evidence=(guard.evidence or "—").strip() or "—",
                    status=status,
                )
            )
    return findings, warnings, rows


def evaluate(
    rules: list[Rule], ads: list[Ad], pending: dict[str, str]
) -> Report:
    """Прогоняет проверки (a)-(c) и собирает таблицу."""
    for ad in ads:
        ad.pending_evidence = pending.get(ad.ad_id)

    findings = _check_rule_metadata(rules)
    ad_findings, warnings, rows = _check_ads(ads, rules)
    findings.extend(ad_findings)
    rows.sort(key=lambda row: (_ad_sort_key(row.ad_id), row.guard))
    return Report(ads=ads, rules=rules, rows=rows, findings=findings, warnings=warnings)


def evaluate_case(
    case_dir: Path,
    constraints: Path | None = None,
    spine: Path | None = None,
    model_dir: Path | None = None,
) -> Report:
    """Прогон по каталогу кейса с путями по умолчанию."""
    constraints = constraints or case_dir / "CONSTRAINTS.yaml"
    spine = spine or case_dir / "ARCHITECTURE-SPINE.md"
    model_dir = model_dir or case_dir / "model"
    return evaluate(load_rules(constraints), load_ads(model_dir), load_pending(spine))


def _clip(text: str, width: int) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= width else text[: width - 1] + "…"


def render(report: Report, evidence_width: int = 58, status_width: int = 46) -> str:
    """Таблица «AD → страж → kind → evidence → статус» + находки."""
    header = ("AD", "Страж", "kind", "evidence", "Статус")
    body = [
        (
            row.ad_id,
            row.guard,
            row.kind,
            _clip(row.evidence, evidence_width),
            _clip(row.status, status_width),
        )
        for row in report.rows
    ]
    widths = [
        max(len(header[i]), *(len(line[i]) for line in body)) if body else len(header[i])
        for i in range(len(header))
    ]

    def line(cells: tuple[str, ...]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)).rstrip()

    out = [line(header), "  ".join("-" * width for width in widths)]
    out.extend(line(cells) for cells in body)

    out.append("")
    out.append(
        "Сводка: AD — {total}; behavioural-страж — {beh}; "
        "[PENDING-EVIDENCE] — {pend}; нарушений — {bad}".format(
            total=len(report.ads),
            beh=len(report.behavioural_ads),
            pend=len(report.pending_ads),
            bad=len(report.findings),
        )
    )
    if report.behavioural_ads:
        out.append(f"  behavioural: {', '.join(report.behavioural_ads)}")
    if report.pending_ads:
        out.append(f"  pending-evidence: {', '.join(report.pending_ads)}")

    if report.findings:
        out.append("")
        out.append(f"Нарушения ({len(report.findings)}):")
        for finding in report.findings:
            out.append(f"  [{finding.code}] {finding.message}")
    if report.warnings:
        out.append("")
        out.append(f"Замечания ({len(report.warnings)}) — гейт не красят:")
        for warning in report.warnings:
            out.append(f"  [{warning.code}] {warning.message}")

    out.append("")
    out.append("Итог: " + ("PASS" if report.ok else "FAIL"))
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="C-036: мета-валидатор стражей кейса (ADR-011, AD-10)"
    )
    parser.add_argument("--case-dir", default=".", help="каталог кейса (по умолчанию '.')")
    parser.add_argument("--constraints", help="путь к CONSTRAINTS.yaml")
    parser.add_argument("--spine", help="путь к ARCHITECTURE-SPINE.md")
    parser.add_argument("--model-dir", help="каталог model/ с карточками AD-*.md")
    parser.add_argument(
        "--report",
        action="store_true",
        help="печатать таблицу и находки, но не падать (exit 0)",
    )
    args = parser.parse_args(argv)

    case_dir = Path(args.case_dir)
    try:
        report = evaluate_case(
            case_dir,
            constraints=Path(args.constraints) if args.constraints else None,
            spine=Path(args.spine) if args.spine else None,
            model_dir=Path(args.model_dir) if args.model_dir else None,
        )
    except MetaError as exc:
        print(f"check_constraints_meta: конфигурация контроля не читается — {exc}")
        if args.report:
            return 0
        return 1

    print(render(report))
    if args.report:
        return 0
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
