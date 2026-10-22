"""Детерминированный билдер доменного корпуса токенизатора (ADR-042, ADR-043, ADR-044).

Контекст.  ``tools/bpe_train_ru.py`` обучает byte-level BPE 160K, добавляя поток
**R** — русский доменный корпус, документ = абзац (разделитель ``"\\n\\n"``).
Корпус v10.1 (25.08.2026, ``299 141 131`` байт) отстал от домена на 165 МБ, а
билдер, которым он собирался, не сохранился.  Этот скрипт собирает корпус
заново — **тем же контрактом формата**, что читает тренер, и детерминированно:
повторный прогон на том же входе байт-идентичен (обход сортируется по
posix-относительному пути, порядок строк внутри файла — как есть, писать
нечего, что зависело бы от времени/случайности).

Три потока (доли D из обследования ADR-043):

* **R** — доменные концепты (``~/library/concepts``, ``distillate``,
  ``cpt_corpus/clean``) и любые явные ``--source``;
* **T** — трейсы Spine (ADR-043): сессионные jsonl (Claude Code, сессии
  ``~/.arch-ml/sessions``, траектории openclaw, чаты qwen), текстовые отчёты,
  json-отчёты прогонов (``~/.arch-ml/reports/**/*.json``; структурные json/yaml
  в spine-каталогах — нет) и экспорты сессий. Строгий отбор — по разбору строк
  (см. :mod:`axiom_ds.trace_skills`), SQLite-сессии отложены отдельной дельтой и
  фиксируются в манифесте как ``EMPTY-PENDING``;
* **S** — методики библиотеки плагинов (ADR-044): ``SKILL.md`` и прочие ``*.md``;
  YAML-фронтматтер срезается, ``description`` (триггеры и синонимы) сохраняется.

Ступени обработки документа строго в этом порядке: **скраб → дедуп → запись**.

* **Скраб обязателен** (ADR-043 п.3, ADR-020): каждый документ проходит через
  ``tools/axiom_ds/scrub.py`` до записи; недоступность скраба — отказ сборки
  (fail-closed, ненулевой код). Режима «без скраба» нет.
* **Дедуп** — по абзацу (sha256 после скраба), остаётся первое вхождение;
  область — потоки S и T, поток R не трогается (его состав уже отобран доменно).
  S обрабатывается раньше T, поэтому методический блок при цитировании в трейсе
  остаётся копией из S (ADR-044 п.7).
* **Доли** потоков ограничены параметрами ``--traces-share`` (дефолт 0.15) и
  ``--skills-share`` (дефолт 0.10) — от объёма базового потока R
  (``SHARE_BASIS = domain_r``), а не от итогового корпуса: домен задаёт масштаб,
  T/S — добавки к нему (ADR-043 Amendment п.1). Потолок жёсткий: доля выше
  0.15/0.10 — отказ с ненулевым кодом; обход — только ``--allow-share-override``,
  и тогда в манифесте стоит пометка о намеренном превышении (Amend. п.2).
  Фактическая доля от R, доля от корпуса, база и ``share_cap`` попадают в манифест.
* **Отбор потока T** — стратифицированный и сидированный (Amend. п.3): страта —
  корень источника и его подкаталог первого уровня, бюджет делится пропорционально
  объёму, порядок — сидированная перестановка, страты чередуются; при одном
  ``--seed`` результат побитово воспроизводим. Авторитетная граница — док-кап в
  символах (``share_cap × chars(R)``).

Формат документа — ровно как в ``iter_ru_docs``: файл читается как UTF-8
(нечитаемое заменяется), режется по ``"\\n\\n"``, каждый блок обрезается по
краям от ``"\\n"``; пустые блоки отбрасываются (``--min-chars`` добавляет
порог, по умолчанию 1 — то есть поведение совпадает с тренером).  Билдер
**аддитивен** к тренеру: ``bpe_train.py`` / ``bpe_train_ru.py`` не трогаются.

CLI::

    python tools/build_domain_corpus.py --out ~/gb10-shared/datasets/cpt_corpus_v13.txt
    python tools/build_domain_corpus.py --traces --skills --out .../cpt_corpus_v13.txt
    python tools/build_domain_corpus.py --source ~/library/concepts --out ... --dry-run

Рядом с корпусом пишутся манифест (``<stem>.manifest.json``: sha256, bytes,
docs, chars, поимённые источники, потоки, скраб, дедуп, ``EMPTY-PENDING``, seed,
``generated_at``) и карточка (``<stem>.card.json``) — структура по образцу
``docs/datasets/axiom-domain-ds-v1-card.json``.  Приватные тексты в git не
попадают (C-032/C-033, AD-6): корпус внутрь дерева репозитория писать
запрещено, обход — только явным ``--allow-any-out``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Iterator, Protocol, Sequence

TOOLS_DIR = Path(__file__).resolve().parent
CASE_DIR = TOOLS_DIR.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

#: Версия схемы манифеста и карточки (меняется при смене состава полей).
SCHEMA_VERSION = "domain-corpus-manifest-v2"
CARD_VERSION = "domain-corpus-card-v2"

#: Контракт документа: стык с ``bpe_train_ru.iter_ru_docs`` (не менять молча).
DOC_SEPARATOR = "\n\n"

#: Дефолт совместим с тренером: отбрасываются только пустые абзацы.
DEFAULT_MIN_CHARS = 1
DEFAULT_SEED = 0

#: Метки потоков и жёсткие потолки долей (ADR-043 п.4, ADR-044 п.4, Amendment п.2).
STREAM_R = "r"
STREAM_S = "s"
STREAM_T = "t"
TRACES_SHARE_CAP = 0.15
SKILLS_SHARE_CAP = 0.10
#: Дефолт доли совпадает с потолком: выше — отказ без явного --allow-share-override.
DEFAULT_TRACES_SHARE = TRACES_SHARE_CAP
DEFAULT_SKILLS_SHARE = SKILLS_SHARE_CAP

#: База расчёта доли — объём домена R, а не итоговый корпус (ADR-043 Amendment п.1).
#: Константа пишется в манифест: домен задаёт масштаб, T/S — добавки к нему.
SHARE_BASIS = "domain_r"
#: Формула потолка доли — фиксируется в манифесте дословно.
SHARE_FORMULA = "cap_chars = share_cap * chars(R)"
#: Пометка манифеста при намеренном превышении потолка (--allow-share-override).
SHARE_OVERRIDE_NOTE = (
    "намеренное превышение потолка доли: --allow-share-override "
    "(ADR-043 Amendment п.2)"
)

#: Правило дедупа — в одном месте: формулировка попадает в манифест.
DEDUP_RULE = "sha256 абзаца после скраба; остаётся первое вхождение (в порядке обхода)"
DEDUP_PRIORITY = "методические блоки остаются из потока s: s обрабатывается раньше t"
SCRUB_STAGES = "deny-list → скраб → дедуп → запись (ADR-020; ADR-043 п.3)"

#: Стратифицированный отбор потока T (ADR-043 Amendment п.3): страта — корень
#: источника и его подкаталог первого уровня; «(root)» — файлы прямо в корне.
STRATUM_ROOT = "(root)"
#: Прокси для бюджета отбора: UTF-8 байты ≥ символы, поэтому запас ×2 даёт
#: материал на весь потолок (авторитетная граница — док-кап в символах).
SELECTION_BYTE_HEADROOM = 2.0

#: Решения по составу потоков, обязанные быть зафиксированными в манифесте:
#: по ним видно, что именно попало в корпус, не читая приватных текстов.
STREAM_NOTES: dict[str, tuple[str, ...]] = {
    STREAM_R: (
        "домен как есть; дедуп к R не применяется — его состав уже отобран доменно",
    ),
    STREAM_S: (
        "YAML-фронтматтер срезан: структурные ключи (name, allowed-tools, …) в корпус "
        "не входят, description сохранён как доменный текст (ADR-044 п.3)",
        "берутся только *.md методики; код, json-конфиги, архивы и изображения исключены",
        "дедуп с потоком T: методический блок остаётся копией из S (ADR-044 п.7)",
    ),
    STREAM_T: (
        "только сессионные тексты: jsonl-события (Claude Code, сессии Spine, openclaw, "
        "qwen) и текстовые отчёты (md/txt/log/rst/csv)",
        "kind=system/usage/event/tool и блоки tool_use/thinking исключены: системный "
        "промпт и память контура — повтор в каждой сессии и приватные детали (ADR-043 п.2)",
        "json-отчёты прогонов (~/.arch-ml/reports/**/*.json) включены: это содержательный "
        "текст, а не конфиг (ADR-043 Amendment п.4)",
        "структурные *.json/*.yaml/*.yml не читаются: служебные конфиги и метрики, "
        "а не трейсы; внутри spine-каталогов (spine-*/**) — исключены всегда "
        "(ADR-043 Amendment п.4)",
        "отбор — стратифицированный и сидированный (корень источника × подкаталог "
        "первого уровня, пропорционально объёму, детерминизм от --seed): не префикс "
        "обхода (ADR-043 Amendment п.3)",
        "SQLite-сессии openclaw/hermes — в pending как EMPTY-PENDING (ADR-043 п.7)",
    ),
}

#: Источники домена по умолчанию: (корень, glob-маски).  Порядок объявления —
#: часть контракта детерминизма, он же попадает в манифест как порядок обхода.
DEFAULT_SOURCE_PATTERNS = ("*.md", "*.txt")
DEFAULT_SOURCES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("~/library/concepts", ("*.md",)),
    ("~/library/distillate", ("*.md", "*.txt")),
    ("~/library/cpt_corpus/clean", ("*",)),
)

MANIFEST_SUFFIX = ".manifest.json"
CARD_SUFFIX = ".card.json"


class CorpusBuildError(Exception):
    """Отказ сборки: пустой вход, битый источник, запрещённый путь вывода, нет скраба."""


class ScrubberLike(Protocol):
    """Минимум, который билдер требует от ступени скраба (``axiom_ds.scrub``)."""

    def scrub_text_stats(self, text: str, stats: Any = ...) -> str: ...


def load_scrubber() -> Any:
    """Ступень скраба или отказ сборки — fail-closed (ADR-043 п.3).

    Скраб не «желателен»: корпус, собранный без него, содержал бы секреты из
    трейсов (в E-компоненте того же источника — 1906 редактирований).  Поэтому
    недоступность модуля скраба — не деградация, а отказ: ненулевой код возврата,
    корпус не пишется.  Режима «без скраба» в контракте нет.
    """
    try:
        from axiom_ds import scrub as scrub_module
    except Exception as exc:  # noqa: BLE001 - сообщение важнее типа исключения
        raise CorpusBuildError(
            f"скраб недоступен ({exc}): tools/axiom_ds/scrub.py не импортируется, "
            "а сборка без скраба запрещена (ADR-043 п.3, fail-closed)"
        ) from exc
    if not hasattr(scrub_module, "scrub_text_stats"):
        raise CorpusBuildError(
            "скраб недоступен: в tools/axiom_ds/scrub.py нет scrub_text_stats "
            "(ADR-043 п.3, fail-closed)"
        )
    return scrub_module


def paragraph_hash(text: str) -> int:
    """Хеш абзаца для дедупа: blake2b-128 как целое (память важнее криптостойкости).

    Хранится только хеш — дедуп не держит тексты в памяти (поток T ≈ 1.6 ГБ сырья).
    """
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest(), "big")


# --------------------------------------------------------------------------- #
# Источники и файлы
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SourceSpec:
    """Источник домена: корень, маски файлов и его метка для манифеста/карточки."""

    root: str
    patterns: tuple[str, ...]
    label: str

    @property
    def path(self) -> Path:
        return Path(self.root)


@dataclass
class SourceStats:
    """Счётчики одного источника: сколько файлов прочитано и что из них вышло."""

    root: str
    label: str
    patterns: list[str]
    stream: str = STREAM_R
    present: bool = True
    files: int = 0
    files_skipped: int = 0
    docs: int = 0
    chars: int = 0
    bytes: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "label": self.label,
            "stream": self.stream,
            "present": self.present,
            "patterns": list(self.patterns),
            "files": self.files,
            "files_skipped": self.files_skipped,
            "docs": self.docs,
            "chars": self.chars,
            "bytes": self.bytes,
        }


def parse_source(raw: str) -> SourceSpec:
    """``ROOT[::GLOB[,GLOB...]]`` → :class:`SourceSpec`.

    Без явных масок источник читает ``*.md`` и ``*.txt``; дефолтные источники
    объявлены со своими масками отдельно (у ``cpt_corpus/clean`` — любые файлы).
    """
    root, _, patterns_raw = raw.partition("::")
    root = os.path.expanduser(root.strip())
    if not root:
        raise CorpusBuildError(f"пустой источник: {raw!r}")
    patterns = tuple(p.strip() for p in patterns_raw.split(",") if p.strip())
    if not patterns:
        patterns = DEFAULT_SOURCE_PATTERNS
    return SourceSpec(root=root, patterns=patterns, label=Path(root).name)


def default_sources() -> list[SourceSpec]:
    return [
        SourceSpec(root=os.path.expanduser(root), patterns=patterns, label=Path(root).name)
        for root, patterns in DEFAULT_SOURCES
    ]


def source_files(spec: SourceSpec) -> list[Path]:
    """Файлы источника: объединение по маскам, сортировка по posix-относительному пути.

    Сортировка — не косметика: от неё зависит воспроизводимость корпуса
    (``readdir`` не даёт порядка), поэтому ключ — относительный posix-путь.
    """
    root = spec.path
    found: dict[Path, None] = {}
    for pattern in spec.patterns:
        for path in root.rglob(pattern):
            if path.is_file():
                found.setdefault(path, None)
    return sorted(found, key=lambda path: path.relative_to(root).as_posix())


# --------------------------------------------------------------------------- #
# Документы
# --------------------------------------------------------------------------- #


def split_documents(text: str, min_chars: int = DEFAULT_MIN_CHARS) -> Iterator[str]:
    """Документы текста: абзацы, разделённые пустой строкой (контракт ``iter_ru_docs``).

    Единственная реализация контракта «документ = абзац» — её же использует
    прибор замера (``measure_tokenizer``), чтобы определение не разошлось.
    """
    for block in text.split(DOC_SEPARATOR):
        block = block.strip("\n")
        if block and len(block) >= min_chars:
            yield block


def read_text(path: Path) -> str:
    """Чтение источника: UTF-8, нечитаемое заменяется (как ``iter_ru_docs``)."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:  # файл исчез между обходом и чтением — не роняем сборку
        return ""


# --------------------------------------------------------------------------- #
# Потоки
# --------------------------------------------------------------------------- #


@dataclass
class StreamStats:
    """Счётчики потока: файлы, документы, символы, дедуп, усечение, доля, база."""

    key: str
    enabled: bool = False
    files: int = 0
    files_skipped: int = 0
    docs: int = 0
    chars: int = 0
    dropped: int = 0
    truncated: bool = False
    share_cap: float | None = None
    share_cap_chars: int | None = None
    share_of_corpus: float = 0.0
    #: База расчёта доли и её объём: доля потока ≤ share_cap × chars(R).
    share_basis: str = SHARE_BASIS
    share_basis_chars: int = 0
    share_of_basis: float = 0.0
    #: Намеренное превышение потолка долей (--allow-share-override).
    share_override: bool = False
    roots: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "roots": list(self.roots),
            "notes": list(self.notes),
            "files": self.files,
            "files_skipped": self.files_skipped,
            "docs": self.docs,
            "chars": self.chars,
            "dropped_docs": self.dropped,
            "truncated": self.truncated,
            "share_cap": self.share_cap,
            "share_cap_chars": self.share_cap_chars,
            "share_basis": self.share_basis,
            "share_basis_chars": self.share_basis_chars,
            "share_of_basis": round(self.share_of_basis, 6),
            "share_of_corpus": round(self.share_of_corpus, 6),
            "share_override": self.share_override,
        }


@dataclass
class BuildPlan:
    """Состав сборки: источники трёх потоков, потолки долей T/S и их обход."""

    r_specs: list[SourceSpec]
    s_specs: list[Any]
    t_specs: list[Any]
    traces_share: float = DEFAULT_TRACES_SHARE
    skills_share: float = DEFAULT_SKILLS_SHARE
    traces_enabled: bool = False
    skills_enabled: bool = False
    #: Явный обход жёсткого потолка долей (--allow-share-override).
    share_override: bool = False


@dataclass
class BuildResult:
    """Результат сборки: счётчики, отчёты ступеней и поимённые источники."""

    stats: list[SourceStats]
    streams: dict[str, StreamStats]
    docs: int
    chars: int
    scrub: dict[str, Any]
    dedup: dict[str, Any]
    pending: list[dict[str, Any]]


class _Writer:
    """Запись документов разделителем ``"\\n\\n"`` (как читает тренер)."""

    def __init__(self, handle: IO[str] | None) -> None:
        self.handle = handle
        self.first = True
        self.docs = 0
        self.chars = 0

    def write(self, doc: str) -> None:
        if self.handle is not None:
            self.handle.write(("" if self.first else DOC_SEPARATOR) + doc)
        self.first = False
        self.docs += 1
        self.chars += len(doc)

    def finish(self) -> None:
        if self.handle is not None and self.docs:
            self.handle.write("\n")


def _stream_files(spec: Any, mode: str) -> list[Path]:
    if mode == STREAM_R:
        return source_files(spec)
    return trace_skills().stream_files(spec)


def _source_units(path: Path, mode: str) -> Iterator[str]:
    """Текстовые единицы файла: для R — файл целиком (делится на абзацы общим циклом)."""
    if mode == STREAM_R:
        yield read_text(path)
        return
    yield from trace_skills().iter_texts(path, mode=mode)


def trace_skills() -> Any:
    """Модуль извлечения потоков T/S — импорт ленив, чтобы R работал и без него."""
    try:
        from axiom_ds import trace_skills as module
    except Exception as exc:  # noqa: BLE001
        raise CorpusBuildError(
            f"потоки T/S недоступны ({exc}): нет tools/axiom_ds/trace_skills.py"
        ) from exc
    return module


def _stratum_of(root: Path, path: Path) -> str:
    """Страта файла: подкаталог первого уровня или ``(root)`` для файлов в корне."""
    rel = path.relative_to(root).as_posix()
    head, sep, _ = rel.partition("/")
    return head if sep else STRATUM_ROOT


def _stratum_seed(seed: int, key: tuple[str, str]) -> int:
    """Сиид страты: ``seed`` + корень + подкаталог → целое (детерминизм явный).

    ``hashlib``, а не ``hash()``: значение не должно зависеть от ``PYTHONHASHSEED``
    процесса — иначе корпус перестал бы воспроизводиться между запусками.
    """
    raw = f"{seed}\x00{key[0]}\x00{key[1]}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def stratified_selection(
    grouped: Sequence[tuple[Any, list[Path]]],
    *,
    seed: int,
    char_budget: int | None,
) -> list[tuple[Any, Path]]:
    """Детерминированный стратифицированный отбор файлов потока T (Amend. п.3).

    Страта — (корень источника, подкаталог первого уровня). Между стратами
    бюджет отбора делится **пропорционально объёму** (чтобы крупный проект не
    занял весь срез), внутри страты порядок — сидированная перестановка. Отбор
    страт чередуется (round-robin) и подаётся в обработку так, чтобы док-кап
    (авторитетная граница в символах) срезал не первую по алфавиту страту, а
    представительную выборку по всем стратам.

    Первый элемент каждой страты берётся всегда, даже если бюджет страты мал:
    иначе крошечная страта была бы не представлена вовсе. При ``seed`` и том же
    входе результат побитово воспроизводим.
    """
    strata: dict[tuple[str, str], list[tuple[Any, Path, int]]] = {}
    for spec, files in grouped:
        root = spec.path
        for path in files:
            try:
                size = path.stat().st_size
            except OSError:  # файл исчез между обходом и отбором
                size = 0
            key = (spec.root, _stratum_of(root, path))
            strata.setdefault(key, []).append((spec, path, size))
    if not strata:
        return []

    total_bytes = sum(size for items in strata.values() for _, _, size in items)
    selected: dict[tuple[str, str], list[tuple[Any, Path]]] = {}
    for key in sorted(strata):
        items = list(strata[key])
        random.Random(_stratum_seed(seed, key)).shuffle(items)
        if char_budget is None or total_bytes <= 0:
            chosen = [(spec, path) for spec, path, _ in items]
        else:
            stratum_bytes = sum(size for _, _, size in items)
            budget = char_budget * SELECTION_BYTE_HEADROOM * stratum_bytes / total_bytes
            chosen, used = [], 0
            for spec, path, size in items:
                chosen.append((spec, path))
                used += size
                if used >= budget:
                    break
        selected[key] = chosen

    queues = [selected[key] for key in sorted(selected)]
    out: list[tuple[Any, Path]] = []
    index = 0
    while True:
        advanced = False
        for queue in queues:  # чередование страт — срез не уходит в первую
            if index < len(queue):
                out.append(queue[index])
                advanced = True
        if not advanced:
            break
        index += 1
    return out


def _process_file(
    path: Path,
    entry: SourceStats,
    stream: StreamStats,
    *,
    mode: str,
    writer: _Writer,
    scrubber: Any,
    scrub_stats: Any,
    min_chars: int,
    seen: set[int] | None,
    cap_chars: int | None,
) -> bool:
    """Обработать один файл: скраб/дедуп каждого абзаца, запись, учёт.

    Возвращает ``True``, если поток обязан остановиться (док-кап превышен).
    """
    try:
        entry.bytes += path.stat().st_size
    except OSError:  # pragma: no cover - гонка с чужой очисткой
        pass
    produced = 0
    for unit in _source_units(path, mode):
        for doc in split_documents(unit, min_chars):
            doc = scrubber.scrub_text_stats(doc, scrub_stats)
            if not doc:
                continue
            if seen is not None:
                digest = paragraph_hash(doc)
                if digest in seen:
                    stream.dropped += 1
                    continue
                seen.add(digest)
            if cap_chars is not None and stream.chars + len(doc) > cap_chars:
                stream.truncated = True
                if produced:
                    entry.files += 1
                    stream.files += 1
                else:
                    entry.files_skipped += 1
                    stream.files_skipped += 1
                return True
            writer.write(doc)
            produced += 1
            entry.docs += 1
            entry.chars += len(doc)
            stream.docs += 1
            stream.chars += len(doc)
    # R считает файлы источника (как раньше: пустышка — тоже файл), T/S —
    # только те, из которых вышел текст: в трейсах 90% обхода —
    # служебные форматы, и «файлов прочитано» должно значить прочитано.
    if produced or mode == STREAM_R:
        entry.files += 1
        stream.files += 1
    else:
        entry.files_skipped += 1
        stream.files_skipped += 1
    return False


def _process_stream(
    specs: Sequence[Any],
    *,
    mode: str,
    writer: _Writer,
    scrubber: Any,
    scrub_stats: Any,
    min_chars: int,
    files_budget: list[int] | None,
    seen: set[int] | None,
    cap_chars: int | None,
    stream: StreamStats,
    entries: list[SourceStats],
    units: Sequence[tuple[Any, Path]] | None = None,
) -> None:
    """Обработать один поток: скраб каждого абзаца, дедуп, запись, учёт.

    Порядок файлов задаёт вызывающий. Для потока T он передаётся готовым
    (``units`` — стратифицированный сидированный отбор, ADR-043 Amendment п.3);
    для R/S файлы берутся детерминированным обходом (posix-сортировка). Усечение
    по доле авторитетно и в символах: поток останавливается на документе, который
    превысил бы потолок.
    """
    if units is not None:
        entry_by_spec: dict[int, SourceStats] = {}
        for spec in specs:
            entry = SourceStats(
                root=spec.root,
                label=spec.label,
                patterns=list(spec.patterns),
                stream=mode,
                present=spec.path.is_dir(),
            )
            entries.append(entry)
            entry_by_spec[id(spec)] = entry
        selected = list(units)
        if files_budget is not None:
            take = max(0, files_budget[0])
            selected = selected[:take]
            files_budget[0] -= len(selected)
        for spec, path in selected:
            entry = entry_by_spec[id(spec)]
            if not entry.present:
                continue
            if _process_file(
                path, entry, stream, mode=mode, writer=writer, scrubber=scrubber,
                scrub_stats=scrub_stats, min_chars=min_chars, seen=seen, cap_chars=cap_chars,
            ):
                return
        return

    for spec in specs:
        files = _stream_files(spec, mode)
        present = spec.path.is_dir()
        if files_budget is not None:
            take = max(0, files_budget[0])
            files = files[:take]
            files_budget[0] -= len(files)
        entry = SourceStats(
            root=spec.root,
            label=spec.label,
            patterns=list(spec.patterns),
            stream=mode,
            present=present,
        )
        entries.append(entry)
        if not present:
            continue
        for path in files:
            if _process_file(
                path, entry, stream, mode=mode, writer=writer, scrubber=scrubber,
                scrub_stats=scrub_stats, min_chars=min_chars, seen=seen, cap_chars=cap_chars,
            ):
                return


def stream_corpus(
    plan: BuildPlan,
    *,
    min_chars: int,
    max_files: int | None,
    handle: IO[str] | None,
    scrubber: Any,
    seed: int = DEFAULT_SEED,
) -> BuildResult:
    """Собрать корпус потоком.  ``handle=None`` — сухой прогон (ничего не пишем).

    Порядок потоков фиксирован и содержателен: **R → S → T**.  R пишется первым,
    потому что от его объёма считаются потолки долей T/S (база ``domain_r``);
    S идёт раньше T, потому что методический блок при цитате в трейсе должен
    остаться копией из S (ADR-044 п.7).  Сеть, GPU и время в результат не входят:
    корпус воспроизводится байт-в-байт.  ``seed`` задаёт стратифицированный отбор
    потока T (ADR-043 Amendment п.3).
    """
    from axiom_ds.scrub import ScrubStats  # локальный импорт: тот же модуль, что у скраббера

    writer = _Writer(handle)
    scrub_stats = ScrubStats()
    streams = {
        STREAM_R: StreamStats(key=STREAM_R, enabled=True, notes=list(STREAM_NOTES[STREAM_R])),
        STREAM_S: StreamStats(
            key=STREAM_S, enabled=plan.skills_enabled, notes=list(STREAM_NOTES[STREAM_S])
        ),
        STREAM_T: StreamStats(
            key=STREAM_T, enabled=plan.traces_enabled, notes=list(STREAM_NOTES[STREAM_T])
        ),
    }
    entries: list[SourceStats] = []
    budget: list[int] | None = [max_files] if max_files is not None else None

    for key, specs in ((STREAM_R, plan.r_specs), (STREAM_S, plan.s_specs), (STREAM_T, plan.t_specs)):
        streams[key].roots = [spec.root for spec in specs]

    _process_stream(
        plan.r_specs,
        mode=STREAM_R,
        writer=writer,
        scrubber=scrubber,
        scrub_stats=scrub_stats,
        min_chars=min_chars,
        files_budget=budget,
        seen=None,
        cap_chars=None,
        stream=streams[STREAM_R],
        entries=entries,
    )

    base_chars = streams[STREAM_R].chars
    if (plan.skills_enabled or plan.traces_enabled) and base_chars == 0:
        raise CorpusBuildError(
            "поток R пуст, а доли T/S считаются от его объёма: соберите базовый "
            "домен (--source) или отключите --traces/--skills — молчаливый ноль "
            "доли выдал бы пустой корпус за собранный (ADR-043 п.4)"
        )

    seen: set[int] = set()  # дедуп разделён между S и T, R не трогается
    caps = {STREAM_S: SKILLS_SHARE_CAP, STREAM_T: TRACES_SHARE_CAP}
    for key, specs, share in (
        (STREAM_S, plan.s_specs, plan.skills_share),
        (STREAM_T, plan.t_specs, plan.traces_share),
    ):
        stream = streams[key]
        if not stream.enabled:
            continue
        cap_chars = int(share * base_chars)
        stream.share_cap = share
        stream.share_cap_chars = cap_chars
        stream.share_basis = SHARE_BASIS
        stream.share_basis_chars = base_chars
        if share > caps[key]:
            stream.share_override = True
            stream.notes.append(SHARE_OVERRIDE_NOTE)
        units = None
        if key == STREAM_T:  # стратифицированный сидированный отбор (Amend. п.3)
            grouped = [(spec, _stream_files(spec, STREAM_T)) for spec in specs]
            units = stratified_selection(grouped, seed=seed, char_budget=cap_chars)
        _process_stream(
            specs,
            mode=key,
            writer=writer,
            scrubber=scrubber,
            scrub_stats=scrub_stats,
            min_chars=min_chars,
            files_budget=budget,
            seen=seen,
            cap_chars=cap_chars,
            stream=stream,
            entries=entries,
            units=units,
        )

    writer.finish()
    total = writer.chars
    for stream in streams.values():
        stream.share_basis_chars = base_chars
        stream.share_of_corpus = (stream.chars / total) if total else 0.0
        stream.share_of_basis = (stream.chars / base_chars) if base_chars else 0.0

    pending = trace_skills().pending_entries() if plan.traces_enabled else []
    scrub_report = {
        "tool": "tools/axiom_ds/scrub.py",
        "applied": True,
        "stages": SCRUB_STAGES,
        "total": scrub_stats.total,
        "by_pattern": {key: int(value) for key, value in sorted(scrub_stats.by_pattern.items())},
    }
    dedup_report = {
        "scope": "s+t",
        "rule": DEDUP_RULE,
        "priority": DEDUP_PRIORITY,
        "dropped_docs": streams[STREAM_S].dropped + streams[STREAM_T].dropped,
    }
    return BuildResult(
        stats=entries,
        streams=streams,
        docs=writer.docs,
        chars=writer.chars,
        scrub=scrub_report,
        dedup=dedup_report,
        pending=pending,
    )


# --------------------------------------------------------------------------- #
# Запись артефактов
# --------------------------------------------------------------------------- #


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def resolve_output(raw: str, *, allow_any: bool) -> Path:
    """Разрешить путь вывода, запретив запись внутрь дерева репозитория.

    Приватный корпус не должен попадать в рабочий каталог (C-032/C-033, AD-6):
    он тут же оказался бы кандидатом на коммит.  Аварийный обход — явный флаг.
    """
    resolved = Path(os.path.expanduser(raw)).resolve()
    if allow_any:
        return resolved
    repo = CASE_DIR.resolve()
    if resolved == repo or repo in resolved.parents:
        raise CorpusBuildError(
            f"вывод внутри дерева репозитория {repo}: {resolved} — приватный корпус "
            f"в рабочий каталог не пишется (C-032/C-033, AD-6); "
            f"укажите путь вне репозитория или --allow-any-out"
        )
    return resolved


def _write_atomic(path: Path, text: str) -> None:
    """Запись через ``.part`` + rename: оборванный прогон не оставит «готовый» файл."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    with open(part, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(part, path)


def build_manifest(
    *,
    out: Path,
    sha256: str,
    bytes_: int,
    docs: int,
    chars: int,
    min_chars: int,
    seed: int,
    max_files: int | None,
    plan: BuildPlan,
    result: BuildResult,
    manifest_name: str,
    card_name: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": "tools/build_domain_corpus.py",
        "generated_at": utc_now(),
        "document_separator": DOC_SEPARATOR,
        "corpus": {
            "file": out.name,
            "path": str(out),
            "sha256": sha256,
            "bytes": bytes_,
            "docs": docs,
            "chars": chars,
        },
        "min_chars": min_chars,
        "seed": seed,
        "max_files": max_files,
        "walk_order": {
            "mode": "per-source sorted by posix-relative path; sources in declared order",
            "source_order": [spec.root for spec in plan.r_specs],
            "stream_order": [STREAM_R, STREAM_S, STREAM_T],
            "traces_selection": (
                "стратифицированный сидированный отбор: страта = корень источника × "
                "подкаталог первого уровня, бюджет пропорционален объёму, порядок "
                "детерминирован от seed (ADR-043 Amendment п.3)"
            ),
        },
        "share_basis": {
            "basis": SHARE_BASIS,
            "chars_r": result.streams[STREAM_R].chars,
            "formula": SHARE_FORMULA,
            "caps": {"traces": TRACES_SHARE_CAP, "skills": SKILLS_SHARE_CAP},
            # Флаг обхода честен только при фактическом превышении (Amend. п.2).
            "override": any(stream.share_override for stream in result.streams.values()),
        },
        "streams": {
            **{key: stream.as_dict() for key, stream in result.streams.items()},
            "share_basis": SHARE_BASIS,
            "share_formula": SHARE_FORMULA,
        },
        "scrub": result.scrub,
        "dedup": result.dedup,
        "pending": result.pending,
        "sources": [entry.as_dict() for entry in result.stats],
        "card": card_name,
        "manifest": manifest_name,
    }


def build_card(
    *,
    out: Path,
    sha256: str,
    bytes_: int,
    docs: int,
    chars: int,
    result: BuildResult,
    manifest_name: str,
) -> dict[str, Any]:
    """Карточка корпуса — структура по образцу ``axiom-domain-ds-v1-card.json``."""
    components: dict[str, Any] = {}
    for index, entry in enumerate(result.stats):
        label = entry.label or f"source-{index + 1}"
        while label in components:  # два источника с одним basename не теряются
            label = f"{label}-{index + 1}"
        components[label] = {
            "root": entry.root,
            "stream": entry.stream,
            "patterns": list(entry.patterns),
            "files": entry.files,
            "docs": entry.docs,
            "chars": entry.chars,
            "bytes": entry.bytes,
            "share": round(entry.chars / chars, 4) if chars else 0.0,
        }
    streams = {
        key: {
            "enabled": stream.enabled,
            "files": stream.files,
            "docs": stream.docs,
            "chars": stream.chars,
            "share": round(stream.share_of_corpus, 4),
            "share_cap": stream.share_cap,
            "share_basis": stream.share_basis,
            "share_of_basis": round(stream.share_of_basis, 4),
            "share_override": stream.share_override,
            "truncated": stream.truncated,
        }
        for key, stream in result.streams.items()
    }
    return {
        "card": out.stem,
        "card_version": CARD_VERSION,
        "status": "candidate",
        "generated_at": utc_now(),
        "dataset_root": str(out.parent),
        "decision": (
            "ADR-042 — актуализация RU-доменного корпуса токенизатора (кандидат v3); "
            "ADR-043 — трейсы Spine (поток T); ADR-044 — методики библиотеки "
            "плагинов (поток S); ADR-004 — карточка и хеш артефакта; ADR-021 — оценка токенов"
        ),
        "corpus": {
            "file": out.name,
            "sha256": sha256,
            "bytes": bytes_,
            "docs": docs,
            "chars": chars,
            "approx_tokens": max(1, chars // 4),
            "approx_tokens_basis": "chars // 4 (ADR-021)",
        },
        "manifest": manifest_name,
        "streams": streams,
        "components": components,
        "notes": [
            "приватные тексты в git не попадают: в репозиторий — только билдер, "
            "тесты и шаблоны карточки/манифеста (C-032/C-033, AD-6)",
            "документ = абзац, разделитель \\n\\n — контракт чтения "
            "tools/bpe_train_ru.iter_ru_docs сохранён",
            "канон токенизатора не переключается этой сборкой: v3 — кандидат "
            "в стороне (ADR-042, п.4)",
            "поток T (ADR-043): только сессионные jsonl и текстовые отчёты; "
            "системные промпты (kind=system) и служебные записи исключены, "
            "скраб секретов обязателен и fail-closed, доля ограничена",
            "поток S (ADR-044): методики плагинов, YAML-фронтматтер срезан, "
            "description сохранён как доменный текст; дедуп с T — приоритет S",
            "доводка (ADR-043 Amendment): доля T/S считается от объёма R "
            "(SHARE_BASIS=domain_r) и ограничена жёстким потолком 0.15/0.10 "
            "(обход — --allow-share-override с пометкой); отбор T — "
            "стратифицированный и сидированный; json-отчёты прогонов включены, "
            "структурный spine-json — нет",
            "SQLite-сессии openclaw/hermes не реализованы: в манифесте "
            "EMPTY-PENDING до отдельной дельты (ADR-043 п.7)",
        ],
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Детерминированный билдер доменного корпуса токенизатора "
                    "(ADR-042; потоки T — ADR-043, S — ADR-044)"
    )
    parser.add_argument(
        "--source", action="append", default=None, metavar="ROOT[::GLOB,...]",
        help="источник потока R (повторяемый); без флага — домен по умолчанию "
             "(concepts/*.md, distillate/*.md|*.txt, cpt_corpus/clean/*)",
    )
    parser.add_argument(
        "--traces", action="store_true",
        help=f"включить поток T (трейсы Spine, ADR-043); доля — --traces-share "
             f"(дефолт {DEFAULT_TRACES_SHARE})",
    )
    parser.add_argument(
        "--traces-source", action="append", default=None, metavar="ROOT[::GLOB,...]",
        help="источник потока T (повторяемый) — переопределяет пути по умолчанию; "
             "сам факт флага включает поток",
    )
    parser.add_argument(
        "--traces-share", type=float, default=DEFAULT_TRACES_SHARE,
        help=f"потолок доли потока T от объёма R (дефолт {DEFAULT_TRACES_SHARE}); "
             f"значение выше {TRACES_SHARE_CAP} — отказ без --allow-share-override",
    )
    parser.add_argument(
        "--skills", action="store_true",
        help=f"включить поток S (методики библиотеки плагинов, ADR-044); "
             f"доля — --skills-share (дефолт {DEFAULT_SKILLS_SHARE})",
    )
    parser.add_argument(
        "--skills-source", action="append", default=None, metavar="ROOT[::GLOB,...]",
        help="источник потока S (повторяемый) — переопределяет пути по умолчанию",
    )
    parser.add_argument(
        "--skills-share", type=float, default=DEFAULT_SKILLS_SHARE,
        help=f"потолок доли потока S от объёма R (дефолт {DEFAULT_SKILLS_SHARE}); "
             f"значение выше {SKILLS_SHARE_CAP} — отказ без --allow-share-override",
    )
    parser.add_argument(
        "--allow-share-override", action="store_true",
        help="явно разрешить долю T/S выше жёсткого потолка (0.15/0.10); "
             "в манифесте помечается как намеренное превышение (ADR-043 Amendment п.2)",
    )
    parser.add_argument("--out", default=None, help="путь корпуса (обязателен вне --dry-run)")
    parser.add_argument("--manifest", default=None, help="путь манифеста (по умолчанию рядом с корпусом)")
    parser.add_argument("--card", default=None, help="путь карточки (по умолчанию рядом с корпусом)")
    parser.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS,
                        help="минимальная длина документа в символах (дефолт 1 — как у тренера)")
    parser.add_argument("--max-files", type=int, default=None,
                        help="ограничить общее число файлов, обрезая порядок обхода "
                             "(потоки в порядке R → S → T; для тестов/проб)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="сид стратифицированного отбора потока T (ADR-043 Amendment п.3): "
                             "при одном значение отбор детерминирован, при разных — выборка "
                             "иная; попадает в манифест")
    parser.add_argument("--dry-run", action="store_true", help="только статистика, ничего не писать")
    parser.add_argument("--allow-any-out", action="store_true",
                        help="разрешить вывод внутрь дерева репозитория (C-032/C-033)")
    return parser


def _empty_input_message(plan: BuildPlan, min_chars: int) -> str:
    return (
        "вход пуст: ни одного документа не собрано "
        f"(источников R {len(plan.r_specs)}, S {len(plan.s_specs)}, T {len(plan.t_specs)}, "
        f"min_chars {min_chars}) — проверьте --source/--traces-source/--skills-source "
        "и содержимое источников"
    )


def _build_plan(args: argparse.Namespace) -> BuildPlan:
    """План сборки из аргументов: источники потоков и проверка долей."""
    module = trace_skills()
    r_specs = [parse_source(raw) for raw in args.source] if args.source else default_sources()

    if args.traces_source:
        t_specs = [module.parse_stream_source(raw, mode=STREAM_T) for raw in args.traces_source]
        traces_enabled = True
    else:
        t_specs = module.default_sources(STREAM_T)
        traces_enabled = bool(args.traces)

    if args.skills_source:
        s_specs = [module.parse_stream_source(raw, mode=STREAM_S) for raw in args.skills_source]
        skills_enabled = True
    else:
        s_specs = module.default_sources(STREAM_S)
        skills_enabled = bool(args.skills)

    for name, share, cap in (
        ("--traces-share", args.traces_share, TRACES_SHARE_CAP),
        ("--skills-share", args.skills_share, SKILLS_SHARE_CAP),
    ):
        if not 0 < share <= 1.0:
            raise CorpusBuildError(
                f"{name} должен быть в (0, 1], получено {share}: доля потока "
                "ограничена сверху (ADR-043 п.4, ADR-044 п.4)"
            )
        if share > cap and not args.allow_share_override:
            raise CorpusBuildError(
                f"{name}={share} выше жёсткого потолка {cap}: доля потока T/S "
                "ограничена, чтобы добавка не перевесила домен R (ADR-043 Amendment п.2). "
                "Укажите долю ≤ потолка или разрешите превышение явно флагом "
                "--allow-share-override (тогда оно будет помечено в манифесте как "
                "намеренное)"
            )

    return BuildPlan(
        r_specs=r_specs,
        s_specs=s_specs,
        t_specs=t_specs,
        traces_share=args.traces_share,
        skills_share=args.skills_share,
        traces_enabled=traces_enabled,
        skills_enabled=skills_enabled,
        share_override=bool(args.allow_share_override),
    )


def _echo_streams(echo: Any, result: BuildResult) -> None:
    for key in (STREAM_R, STREAM_S, STREAM_T):
        stream = result.streams[key]
        if not stream.enabled:
            continue
        truncation = ", обрезан по доле" if stream.truncated else ""
        override = ", НАМЕРЕННОЕ ПРЕВЫШЕНИЕ потолка" if stream.share_override else ""
        echo(
            f"[corpus]   поток {key}: файлов {stream.files} "
            f"(пропущено {stream.files_skipped}), документов {stream.docs}, "
            f"символов {stream.chars}, доля от R {stream.share_of_basis:.4f}, "
            f"от корпуса {stream.share_of_corpus:.4f}{truncation}{override}"
        )


def cmd_build(args: argparse.Namespace) -> int:
    echo = lambda message: print(message, flush=True)  # noqa: E731
    plan = _build_plan(args)
    if args.min_chars < 1:
        raise CorpusBuildError(f"--min-chars должен быть ≥ 1, получено {args.min_chars}")
    if args.max_files is not None and args.max_files < 0:
        raise CorpusBuildError(f"--max-files не может быть отрицательным: {args.max_files}")

    for spec in plan.r_specs:
        if not spec.path.is_dir():
            raise CorpusBuildError(f"источник не найден: {spec.path}")

    scrubber = load_scrubber()  # fail-closed: нет скраба — нет сборки

    out: Path | None = None
    manifest_path: Path | None = None
    card_path: Path | None = None
    if not args.dry_run:
        if not args.out:
            raise CorpusBuildError("--out обязателен (вне --dry-run)")
        out = resolve_output(args.out, allow_any=args.allow_any_out)
        manifest_path = (
            resolve_output(args.manifest, allow_any=args.allow_any_out)
            if args.manifest
            else out.with_suffix(MANIFEST_SUFFIX)
        )
        card_path = (
            resolve_output(args.card, allow_any=args.allow_any_out)
            if args.card
            else out.with_suffix(CARD_SUFFIX)
        )

    if args.dry_run:
        result = stream_corpus(
            plan, min_chars=args.min_chars, max_files=args.max_files, handle=None,
            scrubber=scrubber, seed=args.seed,
        )
        if result.docs == 0 or result.chars == 0:
            raise CorpusBuildError(_empty_input_message(plan, args.min_chars))
    else:
        assert out is not None and manifest_path is not None and card_path is not None
        out.parent.mkdir(parents=True, exist_ok=True)
        # Пишем в .part и подменяем только непустым результатом: отказ не должен
        # стирать ранее собранный корпус (fail-closed ≠ destroy).
        part = out.with_name(out.name + ".part")
        try:
            with open(part, "w", encoding="utf-8", newline="") as handle:
                result = stream_corpus(
                    plan, min_chars=args.min_chars, max_files=args.max_files,
                    handle=handle, scrubber=scrubber, seed=args.seed,
                )
                handle.flush()
                os.fsync(handle.fileno())
        except CorpusBuildError:
            part.unlink(missing_ok=True)
            raise
        if result.docs == 0 or result.chars == 0:
            part.unlink(missing_ok=True)
            raise CorpusBuildError(_empty_input_message(plan, args.min_chars))
        os.replace(part, out)

    files_total = sum(entry.files for entry in result.stats)
    echo(
        f"[corpus] источников {len(result.stats)}, файлов {files_total}, "
        f"документов {result.docs}, символов {result.chars} "
        f"({result.chars / (1 << 20):.1f} МиБ)"
    )
    _echo_streams(echo, result)
    echo(
        f"[corpus] скраб: применён ({result.scrub['tool']}), замен {result.scrub['total']}, "
        f"правила {result.scrub['by_pattern']}"
    )
    echo(f"[corpus] дедуп S+T: отброшено документов {result.dedup['dropped_docs']}")
    for entry in result.pending:
        echo(f"[corpus]   {entry['id']}: {entry['status']} — {entry['reason']}")
    if args.dry_run:
        echo("[corpus] сухой прогон: ничего не записано")
        return 0

    assert out is not None and manifest_path is not None and card_path is not None
    sha256 = file_sha256(out)
    bytes_ = out.stat().st_size
    manifest_name = manifest_path.name
    card_name = card_path.name
    manifest = build_manifest(
        out=out, sha256=sha256, bytes_=bytes_, docs=result.docs, chars=result.chars,
        min_chars=args.min_chars, seed=args.seed, max_files=args.max_files,
        plan=plan, result=result, manifest_name=manifest_name, card_name=card_name,
    )
    card = build_card(
        out=out, sha256=sha256, bytes_=bytes_, docs=result.docs, chars=result.chars,
        result=result, manifest_name=manifest_name,
    )
    _write_atomic(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    _write_atomic(card_path, json.dumps(card, ensure_ascii=False, indent=1) + "\n")
    echo(f"[corpus] sha256 {sha256}")
    echo(f"[corpus] корпус {out} ({bytes_ / (1 << 20):.1f} МиБ)")
    echo(f"[corpus] манифест {manifest_path}")
    echo(f"[corpus] карточка {card_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return cmd_build(args)
    except CorpusBuildError as exc:
        print(f"[corpus] ОТКАЗ: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
