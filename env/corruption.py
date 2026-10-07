"""Детерминированный мутатор чистого кейса (приём §3).

Повреждения (по seed → фиксированный набор), все детектируются гейтами:

**Структурные атомы** — порча формы, восстановимы вставкой текста (``edit_file``):
- ``remove_adr_section`` — удаление обязательной секции ADR (C-001/C-002/C-003 + trace);
- ``break_affects`` — разрыв ``affects:`` в model/AD-*.md (C-010 + trace);
- ``break_verified_by`` — разрыв ``verified_by:`` в model/AD-*.md (C-005 + trace);
- ``break_ad_link`` — удаление model/AD-*.md (trace ``spine-ad-missing-in-model``).

**Смысловые атомы** (§9, амендмент 06.10 — «Смысловая порча»): искажение
СОДЕРЖАНИЯ, а не формы. Глазами и ``must_contain`` такое повреждение невидимо —
видно только сверкой смысла (страж C-047) или пониманием (модель):

- ``numeric_drift`` — число в ADR-009 из КАРТЫ СТРАЖА заменяется детерминированно
  (N → N×2 либо N ± ранг, по seed): решение говорит одно, ``net/config.json`` —
  другое. Детект: C-047 (``adr-config-mismatch``).
- ``claim_inversion`` — инверсия утверждения в секции ADR по словарю пар
  («обязательно» ↔ «запрещено») либо вставкой отрицания перед глаголом-требованием
  («допускается» → «не допускается»). Структурно невидима, детектор — модель-понимание
  (целевой класс L2+).
- ``term_swap`` — подмена значений двух полей карты местами (латент MLA 512 ↔
  размерность indexer'а 128) в тексте ADR: обмен числами одного класса. Детект: C-047.
- ``config_drift`` — числовая правка ``net/config.json`` (config-сторона рассинхрона).
  Детект: C-047. Поле ``tokenizer_hash`` не трогается никогда (пин генератора).

Источник соответствий «число ADR ↔ поле config» — ДЕКЛАРАТИВНАЯ карта
``tools/adr_config_map.yaml`` (та же, что читает страж ``tools/check_adr_config_consistency.py``):
мутатор выбирает цель из карты, а не из зашитых констант, поэтому «порча,
детектируемая стражем» остаётся проверяемым утверждением, а не совпадением.

**R-3 (§9, 05.10.2026):** набор атомов v1 — только восстановимые 4 инструментами
§13. ``break_ad_link`` исключён из L1-набора: восстановление требует создания
отсутствующего файла, а интерфейс агента v1 не несёт инструмента создания файла
(создание файлов = новый инструмент = новый ADR). L1 сохраняет три атома
(``remove_adr_section``, ``break_affects``, ``break_verified_by``) — все
восстановимы ``edit_file``-вставкой. L2/L3 сохраняют ``break_ad_link`` как
заявленный (пока не восстановимый) атом объёма порчи. Смысловые атомы
восстановимы ``edit_file`` обратной заменой (``Damage.meta`` несёт «было → стало»).

Воспроизводимость: одинаковый seed → байт-в-байт одинаковый повреждённый кейс.
Обратимость: ``revert_damage`` возвращает файл байт-в-байт к чистому состоянию,
в том числе когда смысловой и структурный атомы делят один файл (смысловые
откатываются обратной подменой, а не копированием из чистого кейса).
"""

from __future__ import annotations

import json
import random
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

ADR_SECTIONS = ("## Alternatives Considered", "### Negative", "## Reversibility")

#: Структурные атомы: порча формы файлов кейса.
STRUCTURAL_ATOMS = ("remove_adr_section", "break_affects", "break_verified_by", "break_ad_link")

#: Смысловые атомы (§9, амендмент 06.10): порча содержания, невидимая grep-правилам.
SEMANTIC_ATOMS = ("numeric_drift", "claim_inversion", "term_swap", "config_drift")

# Состав повреждений по уровню лесенки (§9): L0 — одна порча, L1 — три.
# L1 — только восстановимые edit_file-атомы (R-3): три атома без break_ad_link.
# L2 — структурные + смысловые numeric_drift/term_swap (комбинированная порча).
# L3 — L2 + claim_inversion (модель-детекция) + config_drift (config-сторона).
LEVEL_ATOMS: dict[str, list[str]] = {
    "L0": ["remove_adr_section"],
    "L1": ["remove_adr_section", "break_affects", "break_verified_by"],
    "L2": [*STRUCTURAL_ATOMS, "numeric_drift", "term_swap"],
    "L3": [*STRUCTURAL_ATOMS, "numeric_drift", "term_swap", "claim_inversion", "config_drift"],
}

# ── смысловые атомы E-7: словари инверсии и карта C-047 ──────────────────────

#: Словарь пар инверсии утверждения (claim_inversion, режим ``pair``).
CLAIM_PAIRS: dict[str, str] = {"обязательно": "запрещено", "запрещено": "обязательно"}

#: Глаголы/предикативы требования: отрицание вставляется перед словом (режим ``negation``).
CLAIM_VERBS = (
    "обязателен",
    "обязательна",
    "обязательны",
    "обязательный",
    "допускается",
    "разрешается",
    "разрешено",
    "требуется",
)

#: Слова-цели claim_inversion, длинные впереди (иначе «обязательны» съест «обязательный»).
_CLAIM_WORDS = tuple(sorted((*CLAIM_PAIRS, *CLAIM_VERBS), key=len, reverse=True))
_CLAIM_RX = re.compile(r"(?<![\w-])(" + "|".join(_CLAIM_WORDS) + r")(?![\w-])")

#: Ширина контекстного якоря claim_inversion (символов слева/справа).
_CLAIM_ANCHOR = 30

#: Признак уже-отрицательного контекста: «не » перед словом-целью.
_NEGATION_PREFIX = "не "

#: Каталог ``tools/`` репозитория — карта C-047 и её загрузчик живут там.
_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
_MAP_REL = Path("tools") / "adr_config_map.yaml"

# ── версии набора атомов (ADR-037, дельта D5) ────────────────────────────────

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

    ``meta`` несёт данные обратной подмены (``field``/``old``/``new`` у числовых
    атомов, ``prefix``/``suffix`` у claim_inversion): агент может восстановить
    повреждение ``edit_file``-ом по записи «было → стало», не читая исходников.

    ``path``/``new_value`` — config-сторона ``drift_config_value`` (ADR-037):
    правка значения конфига по ключу ``path``. Поля обеих механик сосуществуют,
    ``None`` по умолчанию; какой набор заполнен — определяет ``kind``.
    """

    kind: str
    file: str
    section: Optional[str] = None
    meta: Optional[dict[str, Any]] = field(default=None)
    path: Optional[str] = None
    new_value: Any = None

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"kind": self.kind, "file": self.file}
        if self.section is not None:
            d["section"] = self.section
        if self.meta is not None:
            d["meta"] = dict(self.meta)
        if self.path is not None:
            d["path"] = self.path
        if self.new_value is not None:
            d["new_value"] = self.new_value
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
    """Записывает ``value`` по точечному пути; ``False``, если родитель/ключ отсутствует.

    Единая запись пути для обеих config-механик: ``drift_config_value`` (``path``) и
    ``config_drift`` (``meta['field']``, найденный через ``_lookup``). Возвращаемый
    ``bool`` проверяет ``drift_config_value``; ``config_drift`` вызывает после ``_lookup``.
    """
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


def _has_section(path: Path, header: str) -> bool:
    """Есть ли в файле секция ``header`` (та же сверка, что у ``_remove_section``).

    Не каждый ADR кейса несёт канонические секции (пост-baseline ADR-036 не несёт
    ни одной): выбор цели ``remove_adr_section`` обязан это учитывать, иначе
    ``apply_damage`` падает на нечётном seed вместо того, чтобы спланировать порчу.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    return any(ln.lstrip().startswith(header) for ln in text.splitlines())


# ── смысловая порча: словари, якоря, загрузка карты C-047 ─────────────────────

def _load_map_module():
    """Модуль загрузчика карты C-047 (``tools/adr_config_map.py``).

    Импорт отложенный: L0/L1 порче карта не нужна, а тянуть её ``sys.path``
    на импорте модуля мутатора незачем.
    """
    if str(_TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(_TOOLS_DIR))
    import adr_config_map  # noqa: PLC0415  (путь добавляется выше)

    return adr_config_map


def _lookup(obj: Any, dotted: str) -> Any:
    """Значение по точечному пути; ``None``, если пути нет."""
    cur = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _section_of(text: str, offset: int) -> Optional[str]:
    """Ближайший предшествующий заголовок (``## …``/``### …``) для позиции."""
    header: Optional[str] = None
    for line in text[:offset].splitlines():
        if line.startswith("#"):
            header = line.strip()
    return header


def _drift_value(rng: random.Random, old: int) -> int:
    """Детерминированное искажение числа: N×2 либо N ± ранг (1..4), по seed."""
    if rng.random() < 0.5:
        return old * 2
    rank = 1 + rng.randrange(4)
    new = old - rank if rng.random() < 0.5 else old + rank
    if new <= 0 or new == old:
        new = old + rank
    return new


def _claim_candidates(text: str) -> list[tuple["re.Match[str]", str]]:
    """Утверждения-цели claim_inversion в порядке текста.

    Пропускаются заголовки (правка оглавления — уже структурная порча) и уже
    отрицательные вхождения («не требуется»): второй раз отрицать нечего.
    """
    found: list[tuple["re.Match[str]", str]] = []
    for match in _CLAIM_RX.finditer(text):
        line_start = text.rfind("\n", 0, match.start()) + 1
        if text[line_start:match.start()].lstrip().startswith("#"):
            continue
        if text[max(0, match.start() - len(_NEGATION_PREFIX)):match.start()] == _NEGATION_PREFIX:
            continue
        found.append((match, "pair" if match.group(0) in CLAIM_PAIRS else "negation"))
    return found


class _SemanticPlanner:
    """Планировщик смысловых атомов по карте C-047 (детерминирован по ``rng``).

    Карта читается из КЕЙСА (``<clean_dir>/tools/adr_config_map.yaml``): порча
    строится по тому же соответствию, что проверяет страж, и карта едет вместе
    со снапшотом. Карты нет или она дефектна → мутатор отказывается работать:
    молча пропустить смысловой атом значило бы выдать L2/L3 без порчи.
    """

    def __init__(self, clean_dir: Path, rng: random.Random) -> None:
        self.clean_dir = clean_dir
        self.rng = rng
        self.map_path = clean_dir / _MAP_REL
        self.module = _load_map_module()
        if not self.map_path.is_file():
            raise ValueError(
                f"смысловая порча требует карту C-047: {self.map_path.as_posix()} не найдена"
            )
        self.spec = self.module.load(self.map_path)
        problems = self.module.validate(self.spec)
        if problems:
            raise ValueError("карта C-047 структурно дефектна: " + "; ".join(problems))
        self.adr_rel = self.module.resolve_adr_strict(clean_dir, self.spec)
        self.config_rel = self.module.config_rel(self.spec)
        self.by_id = self.module.mapping_by_id(self.spec)

    # ── вспомогательное ─────────────────────────────────────────────────────
    def _adr_text(self) -> str:
        return (self.clean_dir / self.adr_rel).read_text(encoding="utf-8")

    def _drift_entries(self) -> list[dict[str, Any]]:
        """Соответствия, пригодные для числовой подмены (``drift: true``)."""
        return [m for m in self.module.get_mappings(self.spec) if m.get("drift")]

    def _match(self, text: str, pattern: str, what: str) -> "re.Match[str]":
        match = re.search(pattern, text)
        if match is None:
            raise ValueError(
                f"{what}: паттерн {pattern!r} не найден в {self.adr_rel} — "
                "карта C-047 и текст ADR разошлись"
            )
        return match

    # ── атомы ───────────────────────────────────────────────────────────────
    def numeric_drift(self) -> Damage:
        """Число ADR из карты → N×2 либо N ± ранг (по seed)."""
        entries = self._drift_entries()
        if not entries:
            raise ValueError("карта C-047: нет соответствий с drift: true — числовую подмену строить не из чего")
        entry = self.rng.choice(entries)
        text = self._adr_text()
        match = self._match(text, entry["adr_pattern"], "numeric_drift")
        old = int(match.group(1))
        new = _drift_value(self.rng, old)
        return Damage(
            "numeric_drift",
            self.adr_rel,
            _section_of(text, match.start()),
            meta={
                "field": entry["config_field"],
                "old": old,
                "new": new,
                "mapping": entry["id"],
                "pattern": entry["adr_pattern"],
            },
        )

    def term_swap(self) -> Damage:
        """Обмен значениями двух полей карты в тексте ADR (латент ↔ indexer)."""
        swaps = self.module.get_swaps(self.spec)
        if not swaps:
            raise ValueError("карта C-047: свопы не объявлены — term_swap строить не из чего")
        swap = self.rng.choice(swaps)
        left_id, right_id = swap["between"]
        left, right = self.by_id[left_id], self.by_id[right_id]
        text = self._adr_text()
        left_match = self._match(text, left["adr_pattern"], "term_swap")
        right_match = self._match(text, right["adr_pattern"], "term_swap")
        old_left, old_right = int(left_match.group(1)), int(right_match.group(1))
        if old_left == old_right:
            raise ValueError(
                f"term_swap: значения полей {left_id!r}/{right_id!r} совпадают ({old_left}) — "
                "обмен неотличим от покоя"
            )
        return Damage(
            "term_swap",
            self.adr_rel,
            _section_of(text, min(left_match.start(), right_match.start())),
            meta={
                "swap": swap["id"],
                "fields": [
                    {"field": left["config_field"], "old": old_left, "new": old_right,
                     "pattern": left["adr_pattern"]},
                    {"field": right["config_field"], "old": old_right, "new": old_left,
                     "pattern": right["adr_pattern"]},
                ],
            },
        )

    def claim_inversion(self) -> Damage:
        """Инверсия утверждения в секции ADR (словарь пар либо отрицание)."""
        text = self._adr_text()
        candidates = _claim_candidates(text)
        if not candidates:
            raise ValueError(
                f"claim_inversion: в {self.adr_rel} нет утверждений из словаря пар/глаголов требования — "
                "инвертировать нечего"
            )
        match, mode = candidates[self.rng.randrange(len(candidates))]
        old = match.group(0)
        new = CLAIM_PAIRS[old] if mode == "pair" else _NEGATION_PREFIX + old
        return Damage(
            "claim_inversion",
            self.adr_rel,
            _section_of(text, match.start()),
            meta={
                "mode": mode,
                "old": old,
                "new": new,
                # Контекстный якорь: числа соседних атомов могут сдвинуть смещения,
                # поэтому место ищется по содержимому, а не по позиции.
                "prefix": text[max(0, match.start() - _CLAIM_ANCHOR):match.start()],
                "suffix": text[match.end():match.end() + _CLAIM_ANCHOR],
            },
        )

    def config_drift(self) -> Damage:
        """Числовая правка net/config.json (config-сторона рассинхрона)."""
        frozen = self.module.frozen_config_fields(self.spec)
        entries = [
            m for m in self._drift_entries()
            if m["config_field"] not in frozen and "tokenizer_hash" not in m["config_field"]
        ]
        if not entries:
            raise ValueError("карта C-047: нет пригодных полей config для config_drift")
        entry = self.rng.choice(entries)
        config = json.loads((self.clean_dir / self.config_rel).read_text(encoding="utf-8"))
        old = _lookup(config, entry["config_field"])
        if isinstance(old, bool) or not isinstance(old, int):
            raise ValueError(
                f"config_drift: config.{entry['config_field']} не целое ({old!r}) — числовая подмена неприменима"
            )
        return Damage(
            "config_drift",
            self.config_rel,
            None,
            meta={
                "field": entry["config_field"],
                "old": old,
                "new": _drift_value(self.rng, old),
                "mapping": entry["id"],
            },
        )


def plan_damages(clean_dir: Path, seed: int, level: str, atoms_version: str = "v1") -> list[Damage]:
    """Планирует повреждения детерминированно по seed, уровню и версии атомов (§9).

    Файл ADR из карты C-047 закреплён за смысловыми атомами: структурные берут
    другие ADR, поэтому ``remove_adr_section`` не уносит секцию, в которой
    смысловой атом оставил свою подпись. ``atoms_version`` переключает набор
    атомов (``level_atoms``); ``drift_config_value`` (v2) выбирает цель из
    ``model/claims.yaml`` и не берёт поле, уже искажённое ``config_drift``.
    """
    kinds = level_atoms(atoms_version).get(level, level_atoms(atoms_version)["L0"])
    rng = random.Random(seed)
    adr = list(_adr_files(clean_dir))
    ad = list(_ad_files(clean_dir))
    rng.shuffle(adr)
    rng.shuffle(ad)

    planner: Optional[_SemanticPlanner] = None
    reserved_adr: Optional[str] = None
    if any(kind in SEMANTIC_ATOMS for kind in kinds):
        planner = _SemanticPlanner(clean_dir, rng)
        reserved_adr = planner.adr_rel

    damages: list[Damage] = []
    used_adr: set[str] = set()
    used_ad: set[str] = set()
    used_config_fields: set[str] = set()
    for kind in kinds:
        if kind == "remove_adr_section":
            if not adr:
                continue
            free = [f for f in adr if f not in used_adr and f != reserved_adr] or [
                f for f in adr if f not in used_adr
            ]
            if not free:
                # Все свободные кончились: расширяем пул, но ADR карты C-047 не
                # трогаем — его секции нужны смысловым атомам как место подписи.
                free = [f for f in adr if f != reserved_adr] or list(adr)
            # Секция выбирается по seed, файл — первый из свободных, который её
            # НЕСЁТ: иначе удалять нечего и порча падала бы на нечётном seed.
            section = rng.choice(ADR_SECTIONS)
            file = next((f for f in free if _has_section(clean_dir / f, section)), None)
            if file is None:
                pairs = [
                    (f, s) for f in free for s in ADR_SECTIONS if _has_section(clean_dir / f, s)
                ]
                if not pairs:
                    continue  # во всём кейсе нет ни одной канонической секции
                file, section = pairs[0]
            used_adr.add(file)
            damages.append(Damage("remove_adr_section", file, section))
        elif kind in ("break_affects", "break_verified_by", "break_ad_link"):
            if not ad:
                continue
            file = next((f for f in ad if f not in used_ad), ad[0])
            used_ad.add(file)
            damages.append(Damage(kind, file))
        elif kind == "numeric_drift":
            assert planner is not None
            damages.append(planner.numeric_drift())
        elif kind == "term_swap":
            assert planner is not None
            damages.append(planner.term_swap())
        elif kind == "claim_inversion":
            assert planner is not None
            damages.append(planner.claim_inversion())
        elif kind == "config_drift":
            assert planner is not None
            config_damage = planner.config_drift()
            # Поле, искажённое config_drift, исключается из целей drift_config_value:
            # две правки одного ключа конфига откатывались бы неоднозначно.
            used_config_fields.add(config_damage.meta["field"])
            damages.append(config_damage)
        elif kind == "drift_config_value":
            bindings = [
                b for b in _config_bindings(clean_dir) if b[1] not in used_config_fields
            ]
            if not bindings:
                continue
            file, path, value = bindings[rng.randrange(len(bindings))]
            mutated = _mutate_value(value, rng)
            if mutated is None or not (clean_dir / file).is_file():
                continue
            used_config_fields.add(path)
            damages.append(Damage("drift_config_value", file, path=path, new_value=mutated))
    return damages


# ── применение ───────────────────────────────────────────────────────────────

def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _splice(text: str, start: int, end: int, value: str) -> str:
    return text[:start] + value + text[end:]


def _substitute_mapped(text: str, entries: list[dict[str, Any]], expected: str, target: str,
                       what: str) -> str:
    """Меняет числа по паттернам ``entries`` (``expected`` → ``target``), сверяя «было».

    Замены применяются от конца к началу, поэтому смещения не сдвигаются.
    """
    spans: list[tuple[int, int, str]] = []
    for entry in entries:
        match = re.search(entry["pattern"], text)
        if match is None:
            raise ValueError(f"{what}: паттерн {entry['pattern']!r} не найден при подмене")
        found = int(match.group(1))
        if found != entry[expected]:
            raise ValueError(
                f"{what}: ожидалось {entry[expected]} в {entry['field']}, найдено {found} — "
                "текст ADR изменился не этим повреждением"
            )
        spans.append((*match.span(1), str(entry[target])))
    for start, end, value in sorted(spans, reverse=True):
        text = _splice(text, start, end, value)
    return text


def _claim_anchor(damage: Damage, token: str) -> str:
    meta = damage.meta or {}
    return f"{meta['prefix']}{meta[token]}{meta['suffix']}"


def _dump_json_like(path: Path, payload: Any, like: str) -> None:
    """Пишет JSON в разметке образца ``like`` (indent, хвостовой ``\\n``).

    Кейсовый ``net/config.json`` размечен ``json.dump(indent=2, ensure_ascii=False)``
    без хвостового перевода строки — на такой разметке запись байт-в-байт равна
    исходной (проверяется тестом). Иная разметка сохраняет ЗНАЧЕНИЕ, но не пробелы:
    контракт отката — значение поля, а не форматирование файла.
    """
    indent_match = re.search(r"\n( +)\S", like)
    indent = len(indent_match.group(1)) if indent_match else 2
    text = json.dumps(payload, indent=indent, ensure_ascii=False)
    if like.endswith("\n"):
        text += "\n"
    path.write_text(text, encoding="utf-8")


def apply_damage(ws_dir: Path, damage: Damage) -> None:
    """Применяет одно повреждение к рабочему каталогу (in place)."""
    target = ws_dir / damage.file
    if damage.kind == "remove_adr_section":
        text = _read(target)
        target.write_text(_remove_section(text, damage.section), encoding="utf-8")
    elif damage.kind in ("break_affects", "break_verified_by"):
        text = _read(target)
        field_name = "affects" if damage.kind == "break_affects" else "verified_by"
        target.write_text(_strip_yaml_field(text, field_name), encoding="utf-8")
    elif damage.kind == "break_ad_link":
        target.unlink(missing_ok=True)
    elif damage.kind == "numeric_drift":
        meta = damage.meta or {}
        text = _read(target)
        match = re.search(meta["pattern"], text)
        if match is None:
            raise ValueError(f"numeric_drift: паттерн {meta['pattern']!r} не найден при применении")
        found = int(match.group(1))
        if found != meta["old"]:
            raise ValueError(
                f"numeric_drift: ожидалось {meta['old']} в {meta['field']}, найдено {found}"
            )
        start, end = match.span(1)
        target.write_text(_splice(text, start, end, str(meta["new"])), encoding="utf-8")
    elif damage.kind == "term_swap":
        meta = damage.meta or {}
        target.write_text(
            _substitute_mapped(_read(target), meta["fields"], "old", "new", "term_swap"),
            encoding="utf-8",
        )
    elif damage.kind == "claim_inversion":
        meta = damage.meta or {}
        text = _read(target)
        anchor_old = _claim_anchor(damage, "old")
        if text.count(anchor_old) != 1:
            raise ValueError(
                f"claim_inversion: якорь фразы {meta['old']!r} встречается "
                f"{text.count(anchor_old)} раз — место не однозначно"
            )
        target.write_text(text.replace(anchor_old, _claim_anchor(damage, "new")), encoding="utf-8")
    elif damage.kind == "config_drift":
        meta = damage.meta or {}
        like = _read(target)
        config = json.loads(like)
        found = _lookup(config, meta["field"])
        if found != meta["old"]:
            raise ValueError(
                f"config_drift: ожидалось config.{meta['field']} = {meta['old']}, найдено {found!r}"
            )
        _set_path(config, meta["field"], meta["new"])
        _dump_json_like(target, config, like)
    elif damage.kind == "drift_config_value":
        if damage.path is None:
            raise ValueError("drift_config_value без path")
        like = _read(target)
        data = json.loads(like)
        if not _set_path(data, damage.path, damage.new_value):
            raise ValueError(f"drift_config_value: ключ не найден: {damage.path}")
        _dump_json_like(target, data, like)
    else:
        raise ValueError(f"неизвестный вид повреждения: {damage.kind}")


def revert_damage(ws_dir: Path, clean_dir: Path, damage: Damage) -> None:
    """Откатывает повреждение.

    Структурные атомы восстанавливаются копией файла из чистого кейса; смысловые —
    ОБРАТНОЙ подменой по ``Damage.meta``: два атома могут делить один файл (число
    и утверждение в ADR-009), и копия из чистого кейса снесла бы соседнее
    повреждение вместе со своим.
    """
    src = clean_dir / damage.file
    dst = ws_dir / damage.file
    if damage.kind == "numeric_drift":
        meta = damage.meta or {}
        text = _read(dst)
        match = re.search(meta["pattern"], text)
        if match is None:
            raise ValueError(f"numeric_drift: паттерн {meta['pattern']!r} не найден при откате")
        start, end = match.span(1)
        dst.write_text(_splice(text, start, end, str(meta["old"])), encoding="utf-8")
    elif damage.kind == "term_swap":
        meta = damage.meta or {}
        dst.write_text(
            _substitute_mapped(_read(dst), meta["fields"], "new", "old", "term_swap"),
            encoding="utf-8",
        )
    elif damage.kind == "claim_inversion":
        text = _read(dst)
        anchor_new = _claim_anchor(damage, "new")
        if text.count(anchor_new) != 1:
            raise ValueError(
                f"claim_inversion: якорь {anchor_new!r} встречается {text.count(anchor_new)} раз — "
                "откат не однозначен"
            )
        dst.write_text(text.replace(anchor_new, _claim_anchor(damage, "old")), encoding="utf-8")
    elif damage.kind == "config_drift":
        meta = damage.meta or {}
        like = _read(dst)
        config = json.loads(like)
        found = _lookup(config, meta["field"])
        if found != meta["new"]:
            raise ValueError(
                f"config_drift: ожидалось config.{meta['field']} = {meta['new']}, найдено {found!r}"
            )
        # Каноническое значение — то, что объявляет карта C-047 (meta['old']:
        # снято из чистого config и сверено стражем); чистый кейс не нужен.
        _set_path(config, meta["field"], meta["old"])
        _dump_json_like(dst, config, like)
    elif damage.kind == "drift_config_value":
        # ``Damage`` несёт только «стало» (``path``/``new_value``): каноническое
        # значение живёт в чистом кейсе. Правится ОДИН ключ — соседняя config-правка
        # (``config_drift``) не сносится, разметка файла сохраняется.
        if damage.path is None:
            raise ValueError("drift_config_value без path")
        like = _read(dst)
        data = json.loads(like)
        canonical = _lookup(json.loads(_read(src)), damage.path)
        if canonical is None:
            raise ValueError(
                f"drift_config_value: канонический ключ не найден в чистом кейсе: {damage.path}"
            )
        if not _set_path(data, damage.path, canonical):
            raise ValueError(f"drift_config_value: ключ не найден при откате: {damage.path}")
        _dump_json_like(dst, data, like)
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
