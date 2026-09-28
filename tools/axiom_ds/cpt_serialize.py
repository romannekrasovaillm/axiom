"""Сериализатор K/D/S → доменный CPT-корпус ``axiom-domain-ds-v1`` (ADR-020, дельта-2).

Фаза CPT (amendment ADR-020 от 27.09.2026) учится на **прозаическом** доменном
знании: концепты из статей (K), дистилляты статей (D), процедурные скиллы (S).
Ступени этой дельты::

    фильтр источника → скраб (ступень 1) → <HOME> → дедуп (общий пул K∪D∪S)
        → контейнмент D→K → шард jsonl

Ключевые контракты:

* **Одна запись — один документ.** Записи не склеиваются: упаковка в
  последовательности — задача CPT-лупа, а не сериализатора.
* **Скраб — первой ступенью, до записи** (``scrub.scrub_text_stats``): ни один
  секрет не доживает до jsonl; отчёт получает только счётчики правил.
* **Дедуп межкомпонентный**: один ``Deduper`` на K∪D∪S — пересечения D↔K и S↔D
  ожидаемы (дистиллят и карточка нередко выросли из одной статьи).
* **Контейнмент D→K — ступень после дедупа** (``dedup.containment_dedup``):
  содержание карточки K может лежать внутри дистиллята D, не будучи дублем
  (документный Jaccard такой пары ≈0.3). Снимается K, D остаётся — он богаче
  контекстом; ступень отключается ``containment=False`` (``--no-containment``).
* **Буфер записей**: K→D→S читаются и дедуплицируются потоково, но пишутся
  после обхода — судьба карточки K зависит от дистиллятов D, которые идут позже.
* **Приватность**: содержимое библиотек в репозиторий и в отчёт не попадает —
  в отчёте числа и три примера id без содержимого (AD-6, C-032/C-033).

Фильтры источников (механические, зафиксированы в карточке датасета):

* **K** — ``~/library/concepts/<тип>/``: типы из :data:`K_TYPES` (уровни α/β/γ —
  все; прочие типы — ``behavioral_*``, ``attack_strategy`` и пр. — вне);
* **D** — ``~/library/distillate/``: подкаталоги :data:`D_SUBDIRS`
  (``1_методология`` — вне: это про процесс дистилляции, не про домен);
* **S** — ``~/experiments/agents/0710-ariadna/plugins/<плагин>/…/SKILL.md``:
  плагин — первый компонент пути под корнем, фильтр — :data:`PLUGIN_ALLOWLIST`.

Примеры::

    # проба: по 300 файлов на компоненту, каталог /tmp/axiom-cpt-probe
    python -m axiom_ds.cpt_serialize build-cpt --probe --limit-files 300

    # боевой прогон (шарды zstd на gb10-shared, C-033)
    python -m axiom_ds.cpt_serialize build-cpt \\
        --out ~/gb10-shared/datasets/axiom-domain-ds-v1/cpt-kds-v0.1.jsonl.zst \\
        --report ~/gb10-shared/datasets/axiom-domain-ds-v1/cpt-kds-v0.1-report.json
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

if __package__ in (None, ""):  # запуск файлом: python tools/axiom_ds/cpt_serialize.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from axiom_ds import dedup as dedup_mod
    from axiom_ds import scrub as scrub_mod
    from prep_pretrain import common as pp_common
else:
    from . import dedup as dedup_mod
    from . import scrub as scrub_mod
    from prep_pretrain import common as pp_common

#: Версия пайплайна сборки. /2 — дельта-2: ступень контейнмента D→K и
#: консистентная выборка шинглов (путь корпуса при этом не меняется).
PIPELINE_VERSION = "axiom-cpt-kds/2"

#: Порядок компонентов в корпусе и в общем пуле дедупа. Он же — приоритет при
#: коллизии: дубль между компонентами оставляет **первый** (K → D → S).
COMPONENT_ORDER = ("K", "D", "S")

#: Компонент → ранг «старейшинства» в общем пуле дедупа.
_COMPONENT_RANK = {name: index + 1 for index, name in enumerate(COMPONENT_ORDER)}

# --------------------------------------------------------------------------- #
# Фильтры источников (карточка ADR-020)
# --------------------------------------------------------------------------- #

#: K: объявленные типы карточек. Прочие типы (behavioral_*, attack_strategy,
#: interpretive_*, boundary_*, theorem, procedural_* и пр.) — вне корпуса.
K_TYPES = frozenset(
    {
        "algorithmic",
        "algorithmic_primitive",
        "architectural_component",
        "design_proposition",
        "hypothesis",
        "task",
        "benchmark",
        "dataset",
    }
)

#: K: уровни доменных карточек. Фильтром НЕ являются (α/β/γ — все), но попадают
#: в отчёт: видно, что именно легло в корпус.
K_LEVELS = ("α", "β", "γ")

#: D: подкаталоги дистиллятов. ``1_методология`` — вне (про процесс дистилляции).
D_SUBDIRS = ("2_статьи", "3_блоги")

#: S: плагины домена axiom. Скиллы остальных плагинов — вне фильтра.
PLUGIN_ALLOWLIST = frozenset(
    {
        "laguna",
        "agentic-rl",
        "data-curator",
        "verification",
        "effort",
        "frontier-lab",
        "frontier-intelligence",
        "document-tools",
        "misc-tools",
        "agent-infra",
        "agent-harnesses",
        "agent-harness",
        "cpt",
        "pretrain",
        "patterns-resilience",
        "patterns-integration",
        "aws-builders",
        "arch-core",
        "dka",
        "kimi",
        "tui-agent-skills",
        "arch-distilled",
    }
)

#: Имя файла скилла: скиллом считается только он.
SKILL_FILENAME = "SKILL.md"

# --------------------------------------------------------------------------- #
# Пути и пороги
# --------------------------------------------------------------------------- #

DEFAULT_K_ROOT = "~/library/concepts"
DEFAULT_D_ROOT = "~/library/distillate"
DEFAULT_S_ROOT = "~/experiments/agents/0710-ariadna/plugins"

#: Канонический корень датасета (C-032/C-033: данные на gb10-shared).
DATASET_ROOT = "~/gb10-shared/datasets/axiom-domain-ds-v1"
DATASET_NAME = "cpt-kds-v0.1"
MANIFEST_NAME = "manifest-cpt.json"

#: Каталог пробы (--probe): черновики — в /tmp, не на сетевом диске.
PROBE_ROOT = "/tmp/axiom-cpt-probe"
PROBE_REPORT_NAME = "report-cpt.json"

#: Минимальная длина текста записи: карточка из одного frontmatter (или обрывок)
#: не учит ничему и в корпус не идёт.
MIN_TEXT_CHARS = 32

#: Замена абсолютных путей владельца в тексте и в источнике записи.
HOME_TOKEN = "<HOME>"


def default_out(probe: bool, prefix: str = DATASET_NAME, codec: str = "zstd") -> str:
    """Путь шард-файла по умолчанию: проба — в /tmp, бой — в корень датасета."""
    extension = pp_common.CODEC_EXTENSIONS.get(codec, ".jsonl")
    root = PROBE_ROOT if probe else os.path.expanduser(DATASET_ROOT)
    return str(Path(root) / f"{prefix}{extension}")


# --------------------------------------------------------------------------- #
# Обход источников
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SourceFile:
    """Файл-кандидат корпуса: компонент, путь на диске, путь для записи (без HOME)."""

    component: str
    path: Path
    source_path: str
    plugin: str = ""

    @property
    def dir_type(self) -> str:
        """Каталог-«тип» карточки K (тип определяется каталогом, не frontmatter)."""
        return self.path.parent.name


def source_path_for(path: Path) -> str:
    """Путь записи: относительно домашнего каталога, без абсолютного ``/home/…``.

    Файлы вне домашнего каталога (фикстуры тестов, временные деревья) остаются
    абсолютными, но с заменой ``/home/<владелец>`` → ``<HOME>``: приватные пути
    не попадают в датасет ни в относительном, ни в абсолютном виде.
    """
    home = Path(os.path.expanduser("~")).resolve()
    try:
        return Path(path).resolve().relative_to(home).as_posix()
    except (ValueError, OSError):
        return replace_home(str(path))[0]


def _home_prefixes() -> tuple[str, ...]:
    roots = {os.path.expanduser("~"), os.environ.get("HOME", "")}
    cleaned = [root.rstrip("/") for root in roots if root and root != "/"]
    return tuple(sorted(set(cleaned), key=len, reverse=True))


def replace_home(text: str) -> tuple[str, int]:
    """Заменить абсолютные пути ``/home/<владелец>`` на ``<HOME>``; вернуть счётчик."""
    count = 0
    for prefix in _home_prefixes():
        if prefix in text:
            count += text.count(prefix)
            text = text.replace(prefix, HOME_TOKEN)
    return text, count


def _count_files(directory: Path, cap: int = 100_000) -> int:
    total = 0
    for _root, _dirs, files in os.walk(directory):
        total += len(files)
        if total >= cap:
            return cap
    return total


def _note_denied(denial: Counter | None, files: int = 0, dirs: int = 0) -> None:
    if denial is None:
        return
    denial["files"] += files
    denial["dirs"] += dirs


def _iter_md(base: Path, denial: Counter | None = None) -> Iterator[Path]:
    """Поток ``*.md`` под каталогом; запретные пути не читаются (deny-list).

    Запретные каталоги не обходятся вовсе (обрезка ``dirnames``): их содержимое
    не открывается даже для «почистить», а объём пропущенного считается по именам.
    """
    for current, dirnames, filenames in os.walk(base):
        keep: list[str] = []
        for name in dirnames:
            candidate = Path(current) / name
            if scrub_mod.is_denied_path(candidate):
                _note_denied(denial, files=_count_files(candidate), dirs=1)
                continue
            keep.append(name)
        dirnames[:] = keep
        for name in filenames:
            if not name.endswith(".md"):
                continue
            candidate = Path(current) / name
            if scrub_mod.is_denied_path(candidate):
                _note_denied(denial, files=1)
                continue
            yield candidate


def _apply_limit(files: list[SourceFile], limit: int | None) -> list[SourceFile]:
    if limit is None:
        return files
    return files[: max(0, limit)]


def discover_k(root: str | os.PathLike[str], limit: int | None = None,
               denial: Counter | None = None) -> list[SourceFile]:
    """Карточки концептов: только объявленные типы каталогов (уровни — все)."""
    base_root = Path(os.path.expanduser(os.fspath(root)))
    files: list[SourceFile] = []
    for ctype in sorted(K_TYPES):
        base = base_root / ctype
        if not base.is_dir():
            continue
        for path in _iter_md(base, denial):
            files.append(SourceFile("K", path, source_path_for(path)))
    files.sort(key=lambda item: item.source_path)
    return _apply_limit(files, limit)


def discover_d(root: str | os.PathLike[str], limit: int | None = None,
               denial: Counter | None = None) -> list[SourceFile]:
    """Дистилляты статей: только ``2_статьи`` и ``3_блоги``."""
    base_root = Path(os.path.expanduser(os.fspath(root)))
    files: list[SourceFile] = []
    for subdir in D_SUBDIRS:
        base = base_root / subdir
        if not base.is_dir():
            continue
        for path in _iter_md(base, denial):
            files.append(SourceFile("D", path, source_path_for(path)))
    files.sort(key=lambda item: item.source_path)
    return _apply_limit(files, limit)


def plugin_of(relative: str | os.PathLike[str]) -> str:
    """Плагин скилла — первый компонент пути под корнем ``plugins/``."""
    parts = Path(os.fspath(relative)).parts
    return parts[0] if parts else ""


def plugin_inventory(root: str | os.PathLike[str]) -> dict[str, int]:
    """Сколько ``SKILL.md`` в каждом плагине источника (по факту содержимого)."""
    base_root = Path(os.path.expanduser(os.fspath(root)))
    inventory: Counter = Counter()
    if not base_root.is_dir():
        return {}
    for current, dirnames, filenames in os.walk(base_root):
        dirnames[:] = [
            name for name in dirnames if not scrub_mod.is_denied_path(Path(current) / name)
        ]
        if SKILL_FILENAME not in filenames:
            continue
        plugin = plugin_of(Path(current).relative_to(base_root) / SKILL_FILENAME)
        if plugin:
            inventory[plugin] += 1
    return dict(inventory)


def discover_s(root: str | os.PathLike[str], limit: int | None = None,
               denial: Counter | None = None,
               inventory: Counter | None = None) -> list[SourceFile]:
    """Скиллы домена axiom: ``SKILL.md`` под плагинами из ``PLUGIN_ALLOWLIST``.

    Скиллом считается только файл ``SKILL.md``; плагин — первый компонент пути,
    поэтому вложенные уровни (``<плагин>/skills/<группа>/<скилл>/SKILL.md``)
    поддержаны, а архивные деревья (``_archive``, ``_inbox_*``) отсекаются как
    отдельные «плагины»: их имён в allowlist нет.
    """
    base_root = Path(os.path.expanduser(os.fspath(root)))
    files: list[SourceFile] = []
    if not base_root.is_dir():
        return []
    for current, dirnames, filenames in os.walk(base_root):
        keep: list[str] = []
        for name in dirnames:
            candidate = Path(current) / name
            if scrub_mod.is_denied_path(candidate):
                _note_denied(denial, files=_count_files(candidate), dirs=1)
                continue
            keep.append(name)
        dirnames[:] = keep
        if SKILL_FILENAME not in filenames:
            continue
        relative = Path(current).relative_to(base_root) / SKILL_FILENAME
        plugin = plugin_of(relative)
        if inventory is not None:
            inventory[plugin] += 1
        if plugin not in PLUGIN_ALLOWLIST:
            continue
        path = Path(current) / SKILL_FILENAME
        if scrub_mod.is_denied_path(path):
            _note_denied(denial, files=1)
            continue
        files.append(SourceFile("S", path, source_path_for(path), plugin=plugin))
    files.sort(key=lambda item: item.source_path)
    return _apply_limit(files, limit)


# --------------------------------------------------------------------------- #
# Скраб и разбор документа
# --------------------------------------------------------------------------- #

_FRONTMATTER = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n?", re.DOTALL)
_TITLE = re.compile(r"^title:[ \t]*(.+?)[ \t]*$", re.MULTILINE)
_TYPE = re.compile(r"^type:[ \t]*(.+?)[ \t]*$", re.MULTILINE)
_LEVEL = re.compile(r"^level:[ \t]*(.+?)[ \t]*$", re.MULTILINE)


def split_frontmatter(text: str) -> tuple[str, str]:
    """``(тело, frontmatter)``: frontmatter без разделителей, тело — что после него."""
    body = text.lstrip("﻿")
    match = _FRONTMATTER.match(body)
    if match is None:
        return body, ""
    return body[match.end():], match.group(1)


def frontmatter_field(front: str, pattern: re.Pattern[str]) -> str:
    """Значение поля frontmatter (пусто, если поля нет или это YAML-блок)."""
    match = pattern.search(front or "")
    if not match:
        return ""
    value = match.group(1).strip()
    if value in {"|", ">", "|-", ">-", "|+", ">+"}:
        return ""
    return value.strip("\"'")


#: Границы корзин распределения длин записей (символов) — свидетельство о составе.
LENGTH_BUCKETS = ((200, "<200"), (500, "200-500"), (1000, "500-1000"),
                  (3000, "1000-3000"), (None, "3000+"))


def length_bucket(chars: int) -> str:
    """Корзина распределения длин: тонкие записи видно по числам, не по ощущению."""
    for bound, label in LENGTH_BUCKETS:
        if bound is None or chars < bound:
            return label
    return LENGTH_BUCKETS[-1][1]


def read_text(path: Path) -> tuple[str, bool]:
    """Текст файла; второй элемент — True, если UTF-8 пришлось заменить."""
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8"), False
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace"), True


# --------------------------------------------------------------------------- #
# Счётчики
# --------------------------------------------------------------------------- #


@dataclass
class ComponentCounters:
    """Числовые счётчики компоненты: ни содержимого, ни значений секретов."""

    files_total: int = 0
    files_read: int = 0
    records_written: int = 0
    dropped_dedup: int = 0
    dropped_containment: int = 0
    chars: int = 0
    home_replacements: int = 0
    encoding_replacements: int = 0
    type_mismatch: int = 0
    denied_files: int = 0
    denied_dirs: int = 0
    errors: int = 0
    skipped: Counter = field(default_factory=Counter)
    by_type: Counter = field(default_factory=Counter)
    by_level: Counter = field(default_factory=Counter)
    redactions: scrub_mod.ScrubStats = field(default_factory=scrub_mod.ScrubStats)
    plugins: Counter = field(default_factory=Counter)
    length_buckets: Counter = field(default_factory=Counter)
    sample_ids: list[str] = field(default_factory=list)

    def undo_written(self, chars: int, bucket: str, plugin: str = "") -> None:
        """Снять запись со счёта «написано»: контейнмент решил её судьбу иначе.

        Счётчики компоненты — про состав корпуса, а не про промежуточный результат
        ступеней: снятая контейнментом запись не должна ни считаться записанной,
        ни добавлять символы в объём датасета.
        """
        self.records_written -= 1
        self.chars -= chars
        self.dropped_containment += 1
        self.length_buckets[bucket] -= 1
        if plugin:
            self.plugins[plugin] -= 1

    def to_dict(self) -> dict:
        return {
            "files_total": self.files_total,
            "files_read": self.files_read,
            "records_written": self.records_written,
            "dropped_dedup": self.dropped_dedup,
            "dropped_containment": self.dropped_containment,
            "chars": self.chars,
            "home_replacements": self.home_replacements,
            "encoding_replacements": self.encoding_replacements,
            "type_mismatch": self.type_mismatch,
            "denied_files": self.denied_files,
            "denied_dirs": self.denied_dirs,
            "errors": self.errors,
            "skipped": dict(sorted(self.skipped.items())),
            "by_type": dict(sorted(self.by_type.items())),
            "by_level": dict(sorted(self.by_level.items())),
            "redactions": self.redactions.to_dict(),
            "plugins": dict(sorted(self.plugins.items())),
            "length_buckets": {label: int(self.length_buckets.get(label, 0))
                               for _bound, label in LENGTH_BUCKETS},
        }

    def compact(self) -> dict:
        """Проекция для манифеста шарда и отчёта (``by_component``)."""
        return {
            "files_total": self.files_total,
            "files_read": self.files_read,
            "records": self.records_written,
            "dropped_dedup": self.dropped_dedup,
            "dropped_containment": self.dropped_containment,
            "chars": self.chars,
        }


# --------------------------------------------------------------------------- #
# Запись
# --------------------------------------------------------------------------- #


def build_record_id(component: str, source_path: str) -> str:
    """Ид записи: компонент + sha256 пути (уникален в общем пуле дедупа)."""
    digest = hashlib.sha256(source_path.encode("utf-8")).hexdigest()[:16]
    return f"{component.lower()}-{digest}"


def _dedup_record(record_id: str, component: str, text: str) -> dict:
    """Адаптер к ``Deduper``: он работает с записями эпизодов (``turns``).

    Блок дедупа у CPT-записи один — весь документ; ``started_at`` кодирует
    порядок компонентов (K → D → S), поэтому при коллизии остаётся запись
    старшего компонента независимо от порядка обхода файлов.
    """
    return {
        "id": record_id,
        "started_at": f"{_COMPONENT_RANK.get(component, 99):02d}-{component}",
        "turns": [{"content": text}],
    }


#: Состав записи корпуса. Всё остальное в возврате ``serialize_one`` — служебное
#: (``block`` нужен ступеням дедупа и контейнмента) и в jsonl не пишется.
OUTPUT_FIELDS = ("id", "component", "source_path", "text")


def output_record(record: dict) -> dict:
    """Проекция записи на состав корпуса: служебные поля не доживают до jsonl."""
    return {key: record[key] for key in OUTPUT_FIELDS}


def serialize_one(
    source: SourceFile,
    counters: ComponentCounters,
    deduper: dedup_mod.Deduper,
    min_chars: int = MIN_TEXT_CHARS,
    title_prefix: bool = True,
) -> dict | None:
    """Файл → запись CPT (или None, если запись не состоялась).

    Порядок: чтение → разбор frontmatter (K) → **скраб** → ``<HOME>`` → порог
    длины → дедуп. Ни один секрет не доживает до возврата записи; счётчики
    скраба копятся в ``counters.redactions`` (значения — только числа).

    Записываемый текст и блок дедупа различаются: дедуп сравнивает **блок
    источника** (тело карточки / полный текст D и S), а заголовок ``# title``
    (K) добавляется только в запись. Иначе один и тот же материал в K и D
    перестал бы быть точным дублем из-за синтезированной строки. Тот же блок
    отдаётся и контейменту (``block``) — синтезированный заголовок не должен
    решать судьбу карточки.
    """
    try:
        raw, replaced_encoding = read_text(source.path)
    except OSError:
        counters.errors += 1
        counters.skipped["read_error"] += 1
        return None
    counters.files_read += 1
    if replaced_encoding:
        counters.encoding_replacements += 1

    title = ""
    if source.component == "K":
        body, front = split_frontmatter(raw)
        if not front and body.lstrip().startswith("---"):
            # Обрезанная карточка-заготовка: весь файл — YAML-строки без
            # закрывающего разделителя. Для CPT это мусор, а не проза.
            counters.skipped["unterminated_frontmatter"] += 1
            return None
        card_type = frontmatter_field(front, _TYPE)
        counters.by_type[card_type or "unknown"] += 1
        counters.by_level[frontmatter_field(front, _LEVEL) or "unknown"] += 1
        if card_type and card_type != source.dir_type:
            counters.type_mismatch += 1
        if not body.strip():
            counters.skipped["empty_body"] += 1
            return None
        text = body
        title = frontmatter_field(front, _TITLE)
    else:
        text = raw

    scrubbed = scrub_mod.scrub_text_stats(text, counters.redactions)
    scrubbed, replaced_home = replace_home(scrubbed)
    counters.home_replacements += replaced_home

    block = scrubbed.strip()
    if len(block) < min_chars:
        counters.skipped["too_short"] += 1
        return None

    record_id = build_record_id(source.component, source.source_path)
    if not deduper.add(_dedup_record(record_id, source.component, block)):
        counters.dropped_dedup += 1
        return None

    # Заголовок карточки — в запись; дедуп сравнивает блок источника (см. docstring).
    final = f"# {title}\n\n{block}" if title and title_prefix and not block.startswith("# ") \
        else block
    counters.records_written += 1
    counters.chars += len(final)
    counters.length_buckets[length_bucket(len(final))] += 1
    if not counters.sample_ids:
        counters.sample_ids.append(record_id)
    if source.component == "S":
        counters.plugins[source.plugin] += 1
    return {
        "id": record_id,
        "component": source.component,
        "source_path": source.source_path,
        "text": final,
        "block": block,
    }


#: Размеры контрольных пар калибровки MinHash, слов в документе.
NEAR_DUP_CALIBRATION_SIZES = (40, 120, 300, 600, 1200)


def near_dup_calibration(sizes: Sequence[int] = NEAR_DUP_CALIBRATION_SIZES,
                         share: float = 0.95) -> dict:
    """Калибровка near-dup ``Deduper``: оценка против ИСТИННОГО Jaccard.

    Синтетика, без приватного текста: пара документов, у которых ``share``
    содержания общее. Таблица — свидетельство чувствительности дедупа: оценка
    обязана следовать за истинным Jaccard и ВЫШЕ ``MAX_SHINGLES`` (иначе выборка
    шинглов обрушила бы оценку и near-дубли реальных объёмов — карточка ≈2,5 КБ,
    дистиллят ≈9 КБ — снимались бы единицами). После дельты-2 выборка шинглов
    консистентная (bottom-k по значению хеша), поэтому столбец ``estimated``
    сходится с ``true`` на всех размерах, а ``sampled`` показывает, где выборка
    вообще включалась. Провал этой таблицы — сигнал деградации дедупа.
    """
    rows: list[dict] = []
    for size in sizes:
        base = [f"w{index}" for index in range(size)]
        cut = max(1, int(size * share))
        left = " ".join(base)
        right = " ".join(base[:cut] + [f"x{index}" for index in range(size - cut)])
        left_shingles = set(dedup_mod.shingle_hashes(left, max_shingles=10**9))
        right_shingles = set(dedup_mod.shingle_hashes(right, max_shingles=10**9))
        union = left_shingles | right_shingles
        true_jaccard = len(left_shingles & right_shingles) / len(union) if union else 0.0
        estimate = dedup_mod.jaccard_estimate(
            dedup_mod.minhash_signature(left), dedup_mod.minhash_signature(right)
        )
        rows.append(
            {
                "words": size,
                "chars": len(left),
                "unique_shingles": len(left_shingles),
                "sampled": len(left_shingles) > dedup_mod.MAX_SHINGLES,
                "true_jaccard": round(true_jaccard, 4),
                "estimated_jaccard": round(estimate, 4),
                "detected": estimate >= dedup_mod.JACCARD_THRESHOLD,
            }
        )
    return {
        "share": share,
        "threshold": dedup_mod.JACCARD_THRESHOLD,
        "max_shingles": dedup_mod.MAX_SHINGLES,
        "pairs": rows,
    }


def plugin_table(inventory: dict[str, int], plugins_written: dict[str, int]) -> list[dict]:
    """Таблица «плагин → в/вне → почему»: по факту содержимого источника."""
    rows: list[dict] = []
    for name in sorted(set(inventory) | set(PLUGIN_ALLOWLIST)):
        inside = name in PLUGIN_ALLOWLIST
        skills = int(inventory.get(name, 0))
        if inside and skills == 0:
            reason = "в allowlist карточки ADR-020, в источнике скиллов нет (0)"
        elif inside:
            reason = "allowlist домена axiom (ADR-020): ML/агентный контур"
        elif skills == 0:
            reason = "вне allowlist карточки ADR-020"
        elif name.startswith("_"):
            reason = "архив/инбокс источника: копии вне доменного контура"
        else:
            reason = "вне allowlist карточки ADR-020 (чужой домен контура)"
        rows.append(
            {
                "plugin": name,
                "in": inside,
                "skills": skills,
                "records": int(plugins_written.get(name, 0)),
                "reason": reason,
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Прогон
# --------------------------------------------------------------------------- #


def _shard_prefix(name: str) -> str:
    """``cpt-kds-v0.1.jsonl.zst`` → ``cpt-kds-v0.1`` (префикс шард-файлов)."""
    stem = name
    for suffix in (".zst", ".gz"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    if stem.endswith(".jsonl"):
        stem = stem[: -len(".jsonl")]
    return stem or DATASET_NAME


def _drop_shards(out_dir: Path, prefix: str) -> None:
    """--restart: снять прежние шарды и хвост ``.part``, чтобы не смешать состав."""
    for stale in list(out_dir.glob(f"{prefix}-*")):
        if stale.is_file():
            stale.unlink(missing_ok=True)


def _load_manifest(path: Path, roots: dict[str, str], codec: str,
                   restart: bool) -> pp_common.Manifest:
    defaults = {
        "version": PIPELINE_VERSION,
        "codec": codec,
        "cursor_by_component": {},
        "roots": roots,
    }
    if restart:
        return pp_common.Manifest(path, "cpt-kds", **defaults)
    return pp_common.Manifest.load(path, "cpt-kds", **defaults)


def run_build_cpt(
    *,
    out: str | os.PathLike[str],
    report: str | os.PathLike[str] | None = None,
    manifest: str | os.PathLike[str] | None = None,
    codec: str = "zstd",
    level: int = 3,
    shard_bytes: int = pp_common.DEFAULT_SHARD_BYTES,
    k_root: str | os.PathLike[str] = DEFAULT_K_ROOT,
    d_root: str | os.PathLike[str] = DEFAULT_D_ROOT,
    s_root: str | os.PathLike[str] = DEFAULT_S_ROOT,
    limit_files: int | None = None,
    min_chars: int = MIN_TEXT_CHARS,
    title_prefix: bool = True,
    containment: bool = True,
    containment_threshold: float = dedup_mod.CONTAINMENT_THRESHOLD,
    restart: bool = False,
    progress: bool = False,
    probe: bool = False,
    allow_any_out: bool = False,
) -> dict:
    """Собрать CPT-корпус K/D/S; вернуть числовой отчёт (и записать его).

    Порядок ступеней: скраб → блок источника → точный дедуп → **контейнмент**
    (``containment``, порог ``containment_threshold``) → запись. Записи держатся
    в буфере до конца обхода: контейнмент решает судьбу карточки K только после
    того, как прочитаны все дистилляты D, а «старшинство» компонентов в дедупе
    (K → D → S) задаётся порядком обхода.

    Возобновляемый прогон: курсор хранится **по компоненте** (``limit-files``
    режет компоненту, поэтому один глобальный сдвиг источника неточен). Пул
    дедупа живёт в памяти и на resume НЕ восстанавливается: дубль через границу
    прогонов снят не будет — боевая сборка идёт одним прогоном.
    """
    started = time.time()
    rss_at_start = pp_common.max_rss_mb()

    out_path = Path(os.path.expanduser(os.fspath(out)))
    out_dir = pp_common.ensure_output_allowed(out_path.parent, allow_any=allow_any_out)
    prefix = _shard_prefix(out_path.name)
    manifest_path = pp_common.ensure_output_allowed(
        os.path.expanduser(os.fspath(manifest)) if manifest else out_dir / MANIFEST_NAME,
        allow_any=allow_any_out,
    )
    report_path = pp_common.ensure_output_allowed(
        os.path.expanduser(os.fspath(report))
        if report
        else (Path(os.path.expanduser(PROBE_ROOT)) / PROBE_REPORT_NAME)
        if probe
        else out_dir / f"{prefix}-report.json",
        allow_any=allow_any_out,
    )

    roots = {
        "K": replace_home(str(Path(os.path.expanduser(os.fspath(k_root)))))[0],
        "D": replace_home(str(Path(os.path.expanduser(os.fspath(d_root)))))[0],
        "S": replace_home(str(Path(os.path.expanduser(os.fspath(s_root)))))[0],
    }

    counters = {name: ComponentCounters() for name in COMPONENT_ORDER}
    denial = {name: Counter() for name in COMPONENT_ORDER}
    inventory: Counter = Counter()
    discovered = {
        "K": discover_k(k_root, None, denial["K"]),
        "D": discover_d(d_root, None, denial["D"]),
        "S": discover_s(s_root, None, denial["S"], inventory=inventory),
    }
    for name in COMPONENT_ORDER:
        counters[name].files_total = len(discovered[name])
        counters[name].denied_files = denial[name]["files"]
        counters[name].denied_dirs = denial[name]["dirs"]

    manifest_obj = _load_manifest(manifest_path, roots, codec, restart)
    cursor = manifest_obj.data.get("cursor_by_component") or {}
    resume_cursor = {name: int(cursor.get(name, 0)) for name in COMPONENT_ORDER}
    # ``--limit-files`` задаёт ОКНО выборки компоненты (первые N файлов), а курсор
    # resume — точку внутри окна: повторный прогон с бо́льшим N лишь дописывает
    # хвост окна, состав уже собранного не меняется.
    pending = {
        name: _apply_limit(discovered[name], limit_files)[resume_cursor[name]:]
        for name in COMPONENT_ORDER
    }

    deduper = dedup_mod.Deduper()
    scrub_stats = scrub_mod.ScrubStats()
    shard_components: Counter = Counter()
    consumed = dict(resume_cursor)
    written_total = 0
    #: Скраб → блок → дедуп дают записи в буфер: контейнмент решает судьбу K
    #: только после того, как прочитаны ВСЕ записи-контейнеры (D идут после K),
    #: а «старшинство» компонентов в дедупе сохраняется порядком K → D → S.
    records: list[dict] = []

    def consumed_total() -> int:
        return sum(consumed.values())

    def flush_shard(entry: dict | None) -> None:
        if entry is None:
            return
        entry["by_component"] = dict(sorted(shard_components.items()))
        shard_components.clear()
        manifest_obj.add_shard(entry, consumed_total())

    for name in COMPONENT_ORDER:
        for source in pending[name]:
            consumed[name] += 1
            if progress and consumed_total() % 1000 == 0:
                print(
                    f"[axiom-cpt] файлов {consumed_total()}, записей в буфере {len(records)}",
                    file=sys.stderr,
                    flush=True,
                )
            record = serialize_one(
                source,
                counters[name],
                deduper,
                min_chars=min_chars,
                title_prefix=title_prefix,
            )
            if record is None:
                continue
            records.append(record)

    # Ступень контейнмента: после точного дедупа и до записи (ADR-020, дельта-2).
    containment_started = time.time()
    containment_stats = dedup_mod.ContainmentStats(threshold=containment_threshold)
    if containment:
        before = records
        # Сравнивается блок источника (без синтезированного заголовка K) — тот же
        # материал, что видел дедуп: заголовок не должен решать судьбу карточки.
        wires = [
            {"id": record["id"], "component": record["component"], "text": record["block"]}
            for record in before
        ]
        kept_wires, containment_stats = dedup_mod.containment_dedup(
            wires, threshold=containment_threshold
        )
        kept_ids = {wire["id"] for wire in kept_wires}
        records = [record for record in before if record["id"] in kept_ids]
        for record in before:
            if record["id"] in kept_ids:
                continue
            counters[record["component"]].undo_written(
                len(record["text"]), length_bucket(len(record["text"]))
            )
    containment_seconds = pp_common.elapsed(containment_started)

    writer = pp_common.ShardWriter(
        out_dir,
        prefix=prefix,
        shard_bytes=shard_bytes,
        codec=codec,
        level=level,
        start_index=len(manifest_obj.shards),
        existing_shards=manifest_obj.shards,
    )
    if restart:
        _drop_shards(out_dir, prefix)
    else:
        writer.drop_partial()

    try:
        for record in records:
            written_total += 1
            shard_components[record["component"]] += 1
            flush_shard(
                writer.add(output_record(record), pp_common.approx_tokens(record["text"]))
            )
    finally:
        flush_shard(writer.close())

    for name in COMPONENT_ORDER:
        scrub_stats.merge(counters[name].redactions)

    filters = {
        "k_types": sorted(K_TYPES),
        "k_levels": list(K_LEVELS),
        "d_subdirs": list(D_SUBDIRS),
        "s_plugins": sorted(PLUGIN_ALLOWLIST),
        "skill_filename": SKILL_FILENAME,
        "min_text_chars": min_chars,
        "k_title_prefix": title_prefix,
        "component_order": list(COMPONENT_ORDER),
    }
    by_component = {name: counters[name].compact() for name in COMPONENT_ORDER}
    containment_block = {
        "enabled": bool(containment),
        **containment_stats.to_dict(),
        "dropped_K": int(containment_stats.dropped_by_component.get("K", 0)),
        "seconds": round(containment_seconds, 3),
    }
    filters["containment"] = containment
    filters["containment_threshold"] = containment_threshold
    manifest_obj.data["filters"] = filters
    manifest_obj.data["roots"] = roots
    manifest_obj.data["dedup"] = {
        **deduper.stats.to_dict(),
        "calibration": near_dup_calibration(),
    }
    manifest_obj.data["containment"] = containment_block
    manifest_obj.data["redactions"] = scrub_stats.to_dict()
    manifest_obj.data["by_component"] = by_component
    manifest_obj.data["plugin_table"] = plugin_table(dict(inventory), counters["S"].plugins)
    manifest_obj.data["cursor_by_component"] = dict(sorted(consumed.items()))
    manifest_obj.data["resume_cursor"] = dict(sorted(resume_cursor.items()))
    manifest_obj.set_cursor(
        consumed_total(),
        {
            "by_component": by_component,
            "files_read": sum(item["files_read"] for item in by_component.values()),
            "records": sum(item["records"] for item in by_component.values()),
        },
    )
    manifest_obj.totals
    manifest_obj.save()

    seconds = pp_common.elapsed(started)
    totals = manifest_obj.totals
    chars = sum(item["chars"] for item in by_component.values())
    sample_ids = [counters[name].sample_ids[0] for name in COMPONENT_ORDER
                  if counters[name].sample_ids]
    payload = {
        "status": "ok",
        "pipeline": PIPELINE_VERSION,
        "probe": bool(probe),
        "out": str(out_path),
        "out_dir": str(out_dir),
        "shard_prefix": prefix,
        "manifest": str(manifest_path),
        "report": str(report_path),
        "codec": codec,
        "limit_files": limit_files,
        "stop_reason": "limit_files" if limit_files is not None else "completed",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "filters": filters,
        "roots": roots,
        "components": {name: counters[name].to_dict() for name in COMPONENT_ORDER},
        "by_component": by_component,
        "dedup": deduper.stats.to_dict(),
        "dedup_calibration": near_dup_calibration(),
        "containment": containment_block,
        "redactions": scrub_stats.to_dict(),
        "plugin_table": manifest_obj.data["plugin_table"],
        "sample_ids": sample_ids,
        "resume": {
            "cursor_by_component": resume_cursor,
            "cursor_after": dict(consumed),
            "source_records_before": manifest_obj.source_records,
        },
        "shards": manifest_obj.shards,
        "totals": totals,
        "chars": chars,
        "seconds": round(seconds, 3),
        "files_per_s": pp_common.rate_per_second(consumed_total(), seconds),
        "chars_per_s": pp_common.rate_per_second(chars, seconds),
        "output_mb_per_s": pp_common.rate_per_second(totals["bytes"] / (1024 * 1024), seconds),
        "max_rss_mb": pp_common.max_rss_mb(),
        "rss_at_start_mb": rss_at_start,
        "rss_growth_mb": round(pp_common.max_rss_mb() - rss_at_start, 1),
    }
    pp_common.write_report(report_path, payload)
    return payload


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="axiom_ds.cpt_serialize",
        description="Сериализация K/D/S → доменный CPT-корпус (ADR-020, дельта-2)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser(
        "build-cpt",
        help="собрать CPT-корпус из K (концепты), D (дистилляты), S (скиллы)",
    )
    build.add_argument("--out", default=None,
                       help=f"шард-файл корпуса (по умолчанию {DATASET_ROOT}/{DATASET_NAME}.jsonl.zst)")
    build.add_argument("--report", default=None, help="json с числовым отчётом")
    build.add_argument("--manifest", default=None,
                       help=f"манифест (по умолчанию <out>/{MANIFEST_NAME})")
    build.add_argument("--k-root", default=DEFAULT_K_ROOT, help="корень библиотеки концептов")
    build.add_argument("--d-root", default=DEFAULT_D_ROOT, help="корень дистиллятов")
    build.add_argument("--s-root", default=DEFAULT_S_ROOT, help="корень плагинов со скиллами")
    build.add_argument("--limit-files", type=int, default=None,
                       help="предел файлов НА КОМПОНЕНТУ (проба покрывает все три источника)")
    build.add_argument("--min-chars", type=int, default=MIN_TEXT_CHARS,
                       help=f"минимальная длина текста записи (по умолчанию {MIN_TEXT_CHARS})")
    build.add_argument("--no-title-prefix", action="store_true",
                       help="не добавлять заголовок карточки (K) заголовком записи")
    build.add_argument("--no-containment", action="store_true",
                       help="отключить ступень контейнмента D→K (отладка и сверка)")
    build.add_argument("--probe", action="store_true",
                       help=f"проба: по умолчанию писать в {PROBE_ROOT}")
    build.add_argument("--codec", choices=sorted(pp_common.CODEC_EXTENSIONS),
                       default=pp_common.DEFAULT_CODEC, help="кодек шардов")
    build.add_argument("--level", type=int, default=3, help="уровень сжатия")
    build.add_argument("--shard-mb", type=float,
                       default=pp_common.DEFAULT_SHARD_BYTES / (1024 * 1024),
                       help="целевой размер сжатого шард-файла, МБ")
    build.add_argument("--restart", action="store_true",
                       help="начать заново, игнорируя курсор манифеста")
    build.add_argument("--progress", action="store_true", help="печатать прогресс в stderr")
    build.add_argument("--allow-any-out", action="store_true",
                       help="разрешить выход вне gb10-shared и /tmp (нарушает C-033)")
    build.set_defaults(func=cmd_build_cpt)
    return parser


def cmd_build_cpt(args: argparse.Namespace) -> int:
    out = args.out or default_out(probe=args.probe, codec=args.codec)
    try:
        payload = run_build_cpt(
            out=out,
            report=args.report,
            manifest=args.manifest,
            codec=args.codec,
            level=args.level,
            shard_bytes=int(args.shard_mb * 1024 * 1024),
            k_root=args.k_root,
            d_root=args.d_root,
            s_root=args.s_root,
            limit_files=args.limit_files,
            min_chars=args.min_chars,
            title_prefix=not args.no_title_prefix,
            containment=not args.no_containment,
            restart=args.restart,
            progress=args.progress,
            probe=args.probe,
            allow_any_out=args.allow_any_out,
        )
    except ValueError as error:
        print(f"[axiom-cpt] отказ: {error}", file=sys.stderr)
        return 2
    print(
        "[axiom-cpt] записей {records}, шардов {shards}, дублей снято {dups}, "
        "контейнментом снято {contained} (пар {pairs}), "
        "redactions {red}, отчёт {report}".format(
            records=payload["totals"]["records"],
            shards=payload["totals"]["shards"],
            dups=payload["dedup"]["exact"] + payload["dedup"]["near"],
            contained=payload["containment"]["dropped"],
            pairs=payload["containment"]["checked_pairs"],
            red=payload["redactions"]["total"],
            report=payload["report"],
        ),
        file=sys.stderr,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
