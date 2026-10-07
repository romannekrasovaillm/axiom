"""Детерминированный мутатор чистого кейса (приём §3).

Повреждения (по seed → фиксированный набор), все детектируются гейтами:
- ``remove_adr_section`` — удаление обязательной секции ADR (C-001/C-002/C-003 + trace);
- ``break_affects`` — разрыв ``affects:`` в model/AD-*.md (C-010 + trace);
- ``break_verified_by`` — разрыв ``verified_by:`` в model/AD-*.md (C-005 + trace);
- ``break_ad_link`` — удаление model/AD-*.md (trace ``spine-ad-missing-in-model``).

**R-3 (§9, 05.10.2026):** набор атомов v1 — только восстановимые 4 инструментами
§13. ``break_ad_link`` исключён из L1-набора: восстановление требует создания
отсутствующего файла, а интерфейс агента v1 не несёт инструмента создания файла
(создание файлов = новый инструмент = новый ADR). L1 сохраняет три атома
(``remove_adr_section``, ``break_affects``, ``break_verified_by``) — все
восстановимы ``edit_file``-вставкой. L2/L3 сохраняют ``break_ad_link`` как
заявленный (пока не восстановимый) атом объёма порчи.

Воспроизводимость: одинаковый seed → байт-в-байт одинаковый повреждённый кейс.
"""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

ADR_SECTIONS = ("## Alternatives Considered", "### Negative", "## Reversibility")

# Состав повреждений по уровню лесенки (§9): L0 — одна порча, L1 — три.
# L1 — только восстановимые edit_file-атомы (R-3): три атома без break_ad_link.
LEVEL_ATOMS: dict[str, list[str]] = {
    "L0": ["remove_adr_section"],
    "L1": ["remove_adr_section", "break_affects", "break_verified_by"],
    "L2": ["remove_adr_section", "break_affects", "break_verified_by", "break_ad_link"],
    "L3": ["remove_adr_section", "break_affects", "break_verified_by", "break_ad_link"],
}

#: Версия атомов v2 (ADR-037, дельта D5): v1 + ``drift_config_value`` на L1–L3
#: (на L0 нет). Задачи/калибровки ``atoms_version: v1`` воспроизводятся
#: побайтово — словарь v1 не меняется.
LEVEL_ATOMS_V2: dict[str, list[str]] = {
    "L0": list(LEVEL_ATOMS["L0"]),
    "L1": [*LEVEL_ATOMS["L1"], "drift_config_value"],
    "L2": [*LEVEL_ATOMS["L2"], "drift_config_value"],
    "L3": [*LEVEL_ATOMS["L3"], "drift_config_value"],
}

ATOM_VERSIONS = ("v1", "v2")


def level_atoms(atoms_version: str = "v1") -> dict[str, list[str]]:
    return LEVEL_ATOMS_V2 if atoms_version == "v2" else LEVEL_ATOMS


@dataclass(frozen=True)
class Damage:
    """Одно применённое повреждение. ``file`` — путь относительно корня кейса.

    ``path`` — ключ конфига (для ``drift_config_value``): правка значения в
    ``net/config.json`` по этому ключу.
    """

    kind: str
    file: str
    section: Optional[str] = None
    path: Optional[str] = None
    new_value: Any = None

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"kind": self.kind, "file": self.file}
        if self.section is not None:
            d["section"] = self.section
        if self.path is not None:
            d["path"] = self.path
        return d


#: Файл конфигурации, значения которого сверяет C-049 (config_binding).
CONFIG_FILE = "net/config.json"


def _miniyaml():
    """Загружает stdlib-парсер подмножества YAML из кода репозитория (без зависимостей)."""
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "tools" / "miniyaml.py"
    spec = importlib.util.spec_from_file_location("_sensors_miniyaml", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config_bindings(clean_dir: Path) -> list[tuple[str, str, Any]]:
    """Привязки ``(file, path, value)`` из ``model/claims.yaml`` (kind config_binding)."""
    claims_path = clean_dir / "model" / "claims.yaml"
    if not claims_path.is_file():
        return []
    try:
        data = _miniyaml().load_file(claims_path)
    except Exception:  # noqa: BLE001 — нет реестра = нет атома
        return []
    out: list[tuple[str, str, Any]] = []
    for claim in data if isinstance(data, list) else []:
        if not isinstance(claim, dict) or claim.get("kind") != "config_binding":
            continue
        binding = claim.get("binding")
        if isinstance(binding, dict) and binding.get("file") and binding.get("path") is not None:
            out.append((str(binding["file"]), str(binding["path"]), binding.get("value")))
    return sorted(out)


def _mutate_value(value: Any, rng: random.Random) -> Optional[Any]:
    """Правдоподобное другое значение: int ±1, bool инверсия, hex — один символ."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + rng.choice((-1, 1))
    if isinstance(value, float):
        return value + rng.choice((-0.1, 0.1))
    if isinstance(value, str) and len(value) >= 2 and all(
        ch in "0123456789abcdefABCDEF" for ch in value
    ):
        idx = rng.randrange(len(value))
        choices = [ch for ch in "0123456789abcdef" if ch.lower() != value[idx].lower()]
        return value[:idx] + rng.choice(choices) + value[idx + 1 :]
    return None


def _set_path(data: dict, dotted: str, value: Any) -> bool:
    parts = dotted.split(".")
    cur: Any = data
    for part in parts[:-1]:
        if not isinstance(cur, dict) or part not in cur:
            return False
        cur = cur[part]
    if not isinstance(cur, dict) or parts[-1] not in cur:
        return False
    cur[parts[-1]] = value
    return True


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


def plan_damages(clean_dir: Path, seed: int, level: str, atoms_version: str = "v1") -> list[Damage]:
    """Планирует повреждения детерминированно по seed, уровню и версии атомов (§9)."""
    kinds = level_atoms(atoms_version).get(level, level_atoms(atoms_version)["L0"])
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
        elif kind == "drift_config_value":
            bindings = _config_bindings(clean_dir)
            if not bindings:
                continue
            file, path, value = bindings[rng.randrange(len(bindings))]
            mutated = _mutate_value(value, rng)
            if mutated is None or not (clean_dir / file).is_file():
                continue
            damages.append(Damage("drift_config_value", file, path=path, new_value=mutated))
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
    elif damage.kind == "drift_config_value":
        if damage.path is None:
            raise ValueError("drift_config_value без path")
        data = json.loads(target.read_text(encoding="utf-8"))
        if not _set_path(data, damage.path, damage.new_value):
            raise ValueError(f"drift_config_value: ключ не найден: {damage.path}")
        target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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


def corrupt(clean_dir: Path, out_dir: Path, seed: int, level: str, atoms_version: str = "v1") -> list[Damage]:
    """Создаёт повреждённую копию чистого кейса ``out_dir`` из ``clean_dir``.

    ``clean_dir`` — каталог чистого кейса (может содержать env/: он исключается
    при копировании). Возвращает список применённых повреждений (для метаданных
    и отката). ``atoms_version`` — версия набора атомов (``v1`` по умолчанию;
    ``v2`` добавляет ``drift_config_value``, ADR-037 дельта D5).
    """
    from .util import copy_case_snapshot, workspace_size_cap

    copy_case_snapshot(clean_dir, out_dir)
    damages = plan_damages(clean_dir, seed, level, atoms_version=atoms_version)
    for d in damages:
        apply_damage(out_dir, d)
    # Гейт объёма снапшота после порчи (§7, §11(8)): дефект генерации — сразу.
    workspace_size_cap(out_dir)
    return damages
