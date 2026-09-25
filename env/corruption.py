"""Детерминированный мутатор чистого кейса (приём §3).

Повреждения (по seed → фиксированный набор), все детектируются гейтами:
- ``remove_adr_section`` — удаление обязательной секции ADR (C-001/C-002/C-003);
- ``break_affects`` — разрыв ``affects:`` в model/AD-*.md (C-010);
- ``break_verified_by`` — разрыв ``verified_by:`` в model/AD-*.md (C-005 + trace);
- ``break_ad_link`` — удаление model/AD-*.md (trace ``spine-ad-missing-in-model``).

Воспроизводимость: одинаковый seed → байт-в-байт одинаковый повреждённый кейс.
"""

from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

ADR_SECTIONS = ("## Alternatives Considered", "### Negative", "## Reversibility")

# Состав повреждений по уровню лесенки (§9): L0 — одна порча, L1 — три.
LEVEL_ATOMS: dict[str, list[str]] = {
    "L0": ["remove_adr_section"],
    "L1": ["remove_adr_section", "break_verified_by", "break_ad_link"],
    "L2": ["remove_adr_section", "break_affects", "break_verified_by", "break_ad_link"],
    "L3": ["remove_adr_section", "break_affects", "break_verified_by", "break_ad_link"],
}


@dataclass(frozen=True)
class Damage:
    """Одно применённое повреждение. ``file`` — путь относительно корня кейса."""

    kind: str
    file: str
    section: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"kind": self.kind, "file": self.file}
        if self.section is not None:
            d["section"] = self.section
        return d


def _adr_files(clean_dir: Path) -> list[str]:
    return sorted(p.relative_to(clean_dir).as_posix() for p in (clean_dir / "docs" / "adr").glob("*.md"))


def _ad_files(clean_dir: Path) -> list[str]:
    return sorted(p.relative_to(clean_dir).as_posix() for p in (clean_dir / "model").glob("AD-*.md"))


def _remove_section(text: str, header: str) -> str:
    """Удаляет заголовок секции ``header`` и её тело до следующего заголовка
    того же или более высокого уровня."""
    lines = text.splitlines(keepends=True)
    idx = None
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith(header):
            idx = i
            break
    if idx is None:
        raise ValueError(f"секция {header!r} не найдена")
    level = len(header) - len(header.lstrip("#"))
    end = len(lines)
    for j in range(idx + 1, len(lines)):
        stripped = lines[j].lstrip()
        if stripped.startswith("#"):
            hlevel = len(stripped) - len(stripped.lstrip("#"))
            if hlevel <= level:
                end = j
                break
    return "".join(lines[:idx] + lines[end:])


def _strip_yaml_field(text: str, field: str) -> str:
    """Удаляет строку frontmatter ``field: ...`` (если есть)."""
    lines = text.splitlines(keepends=True)
    kept = [ln for ln in lines if not ln.lstrip().startswith(field + ":")]
    return "".join(kept)


def plan_damages(clean_dir: Path, seed: int, level: str) -> list[Damage]:
    """Планирует повреждения детерминированно по seed и уровню (§9)."""
    kinds = LEVEL_ATOMS.get(level, LEVEL_ATOMS["L0"])
    rng = random.Random(seed)
    adr = list(_adr_files(clean_dir))
    ad = list(_ad_files(clean_dir))
    rng.shuffle(adr)
    rng.shuffle(ad)

    damages: list[Damage] = []
    used_adr: set[str] = set()
    used_ad: set[str] = set()
    for kind in kinds:
        if kind == "remove_adr_section":
            if not adr:
                continue
            file = next((f for f in adr if f not in used_adr), adr[0])
            used_adr.add(file)
            section = rng.choice(ADR_SECTIONS)
            damages.append(Damage("remove_adr_section", file, section))
        elif kind in ("break_affects", "break_verified_by"):
            if not ad:
                continue
            file = next((f for f in ad if f not in used_ad), ad[0])
            used_ad.add(file)
            damages.append(Damage(kind, file))
        elif kind == "break_ad_link":
            if not ad:
                continue
            file = next((f for f in ad if f not in used_ad), ad[0])
            used_ad.add(file)
            damages.append(Damage("break_ad_link", file))
    return damages


def apply_damage(ws_dir: Path, damage: Damage) -> None:
    """Применяет одно повреждение к рабочему каталогу (in place)."""
    target = ws_dir / damage.file
    if damage.kind == "remove_adr_section":
        text = target.read_text(encoding="utf-8")
        target.write_text(_remove_section(text, damage.section), encoding="utf-8")
    elif damage.kind in ("break_affects", "break_verified_by"):
        field = "affects" if damage.kind == "break_affects" else "verified_by"
        text = target.read_text(encoding="utf-8")
        target.write_text(_strip_yaml_field(text, field), encoding="utf-8")
    elif damage.kind == "break_ad_link":
        target.unlink(missing_ok=True)
    else:
        raise ValueError(f"неизвестный вид повреждения: {damage.kind}")


def revert_damage(ws_dir: Path, clean_dir: Path, damage: Damage) -> None:
    """Откатывает повреждение копированием исходного файла из чистого кейса."""
    src = clean_dir / damage.file
    dst = ws_dir / damage.file
    if damage.kind == "break_ad_link":
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def corrupt(clean_dir: Path, out_dir: Path, seed: int, level: str) -> list[Damage]:
    """Создаёт повреждённую копию чистого кейса ``out_dir`` из ``clean_dir``.

    ``clean_dir`` — каталог чистого кейса (может содержать env/: он исключается
    при копировании). Возвращает список применённых повреждений (для метаданных
    и отката).
    """
    from .util import copy_case_snapshot

    copy_case_snapshot(clean_dir, out_dir)
    damages = plan_damages(clean_dir, seed, level)
    for d in damages:
        apply_damage(out_dir, d)
    return damages
