"""Шард C: код (ADR-021, ~3B токенов) — The Stack, лицензионный фильтр.

Языки: python, rust, go, javascript, shell.
Лицензии: mit, apache-2.0, bsd-3-clause, bsd-2-clause, isc, 0bsd.
Длина файла: 256 Б ≤ len ≤ 100 КБ (микробные выбрасываются, длинные усекаются).

Реестр источников (``SOURCES``) отражает реальность, а не пожелание:

* ``stack-v2-dedup`` — приоритет ADR-021 (``bigcode/the-stack-v2-dedup``).
  Конфиги по языкам ровно те, что нужны (``Python``/``Rust``/``Go``/
  ``JavaScript``/``Shell``) — стриминг по языку структурно осуществим;
* ``stack-dedup-v1`` — фолбэк (``bigcode/the-stack-dedup``): контенты файлов
  и поля ``lang``/``license``/``size`` лежат прямо в строке;
* ``common-pile-stackv2`` и ``codeparrot-clean`` — публичные НЕ-gated
  источники: нужны, когда доступ к bigcode не выдан (см. отчёт пробы).

Оба bigcode-датасета закрыты гейтом (``gated: auto``): без выданного доступа
итератор падает с внятной ошибкой до первого документа — пайплайн не
«деградирует молча», источник обязан быть назван явно (``--source``).
"""

from __future__ import annotations

import ast
import os
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

from . import common

SHARD = "C"

LANGUAGES: tuple[str, ...] = ("python", "rust", "go", "javascript", "shell")

ALLOWED_LICENSES: frozenset[str] = frozenset(
    {"mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "isc", "0bsd"}
)

MIN_FILE_BYTES = 256
MAX_FILE_BYTES = 100 * 1024

DEFAULT_TARGET_TOKENS = 3_000_000_000


# --------------------------------------------------------------------------- #
# Реестр источников
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SourceSpec:
    """Описание источника кода: где лежат текст, лицензия и язык записи."""

    name: str
    repo: str
    note: str
    gated: bool
    text_fields: tuple[str, ...]
    license_fields: tuple[str, ...]
    lang_fields: tuple[str, ...]
    path_fields: tuple[str, ...] = ()
    repo_fields: tuple[str, ...] = ()
    configs: dict[str, str] = field(default_factory=dict)
    #: True — по конфигу на язык; False — один поток, язык фильтруется полем.
    per_language_configs: bool = True
    #: Шаблон файла на язык для источников без parquet-конфигов
    #: (``{lang}`` подставляется строчным именем языка).
    data_files_template: str | None = None
    #: Язык источника, у которого в записи НЕТ поля языка, но корпус одноязычный
    #: по построению (codeparrot-clean — только python). Это свойство источника,
    #: а не догадка о записи; в мете записи оно фиксируется как факт источника.
    assumed_language: str | None = None

    def config_for(self, language: str) -> str | None:
        if not self.per_language_configs:
            return self.configs.get("*")
        return self.configs.get(language, language)

    def spec(self, languages: Sequence[str] = LANGUAGES) -> dict:
        return {"kind": "stack", "name": self.name, "languages": list(languages)}


SOURCES: dict[str, SourceSpec] = {
    "stack-v2-dedup": SourceSpec(
        name="stack-v2-dedup",
        repo="bigcode/the-stack-v2-dedup",
        note=(
            "приоритет ADR-021; дедуплицированный Stack v2, конфиг на язык "
            "(Python/Rust/Go/JavaScript/Shell). gated:auto — нужен выданный доступ"
        ),
        gated=True,
        text_fields=("content", "text"),
        license_fields=("detected_licenses", "license", "licenses"),
        lang_fields=("language", "lang"),
        path_fields=("path",),
        repo_fields=("repo_name", "repository_name"),
        configs={
            "python": "Python",
            "rust": "Rust",
            "go": "Go",
            "javascript": "JavaScript",
            "shell": "Shell",
        },
    ),
    "stack-dedup-v1": SourceSpec(
        name="stack-dedup-v1",
        repo="bigcode/the-stack-dedup",
        note=(
            "фолбэк: The Stack v1 dedup, конфиги-каталоги data/<язык> строчными "
            "именами; поля content/lang/license/size в строке. gated:auto"
        ),
        gated=True,
        text_fields=("content", "text"),
        license_fields=("license", "licenses", "detected_licenses"),
        lang_fields=("lang", "language"),
        path_fields=("path",),
        repo_fields=("repo_name",),
        configs={
            "python": "python",
            "rust": "rust",
            "go": "go",
            "javascript": "javascript",
            "shell": "shell",
        },
    ),
    "stack-smol-xl": SourceSpec(
        name="stack-smol-xl",
        repo="bigcode/the-stack-smol-xl",
        note=(
            "публичный НЕ-gated срез The Stack v1 ПО ЯЗЫКАМ (data/<язык>/data.json): "
            "те же поля, что у the-stack-dedup — lang и список лицензий на файл; "
            "объём мал (~0.6 ГБ на 5 языков) — годится для пробы и приёмки, не для 3B"
        ),
        gated=False,
        text_fields=("content",),
        license_fields=("max_stars_repo_licenses", "license", "licenses", "detected_licenses"),
        lang_fields=("lang", "language"),
        path_fields=("max_stars_repo_path", "path"),
        repo_fields=("max_stars_repo_name", "repo_name"),
        configs={
            "python": "python",
            "rust": "rust",
            "go": "go",
            "javascript": "javascript",
            "shell": "shell",
        },
        data_files_template="hf://datasets/bigcode/the-stack-smol-xl/data/{lang}/data.json",
    ),
    "common-pile-stackv2": SourceSpec(
        name="common-pile-stackv2",
        repo="common-pile/stackv2",
        note=(
            "публичный НЕ-gated фолбэк с контентом Stack v2 (один поток, "
            "лицензия/язык в metadata.{license,language}). Замер: language — "
            "детектированный формат (CSV/Futhark/…), файлов объявленных языков в "
            "потоке почти нет, поэтому для шарда C практически не даёт записей"
        ),
        gated=False,
        text_fields=("text", "content"),
        license_fields=("license", "detected_licenses"),
        lang_fields=("language", "lang"),
        path_fields=("path",),
        repo_fields=("repo_name",),
        configs={"*": None},
        per_language_configs=False,
    ),
    "codeparrot-clean": SourceSpec(
        name="codeparrot-clean",
        repo="codeparrot/codeparrot-clean",
        note=(
            "публичный НЕ-gated фолбэк (только python): поля content/license/size, "
            "поля языка в записи НЕТ — корпус одноязычный по построению "
            "(assumed_language=python); объём мал для 3B, годится для проверки"
        ),
        gated=False,
        text_fields=("content", "text"),
        license_fields=("license", "licenses"),
        lang_fields=("lang", "language"),
        path_fields=("path",),
        repo_fields=("repo_name",),
        configs={"*": None},
        per_language_configs=False,
        assumed_language="python",
    ),
}

DEFAULT_SOURCE_NAME = "stack-dedup-v1"

#: Порядок перебора для ``--source auto``: приоритет ADR-021, затем фолбэк,
#: затем публичные источники (нужны, пока доступ к bigcode не выдан).
AUTO_ORDER: tuple[str, ...] = (
    "stack-v2-dedup",
    "stack-dedup-v1",
    "stack-smol-xl",
    "codeparrot-clean",
    "common-pile-stackv2",
)


# --------------------------------------------------------------------------- #
# Лицензии и длина
# --------------------------------------------------------------------------- #


def normalize_licenses(raw: Any) -> tuple[str, ...]:
    """Привести поле лицензии к кортежу нормализованных имён (нижний регистр).

    Поддерживаются формы: ``"MIT"``, ``["MIT"]``, ``"['MIT', 'Apache-2.0']"``
    (строка с repr списка — так отдаёт the-stack-smol), ``None``.
    """
    if raw is None:
        return ()
    values: list[Any]
    if isinstance(raw, (list, tuple, set)):
        values = list(raw)
    elif isinstance(raw, str):
        text = raw.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                parsed = None
            values = list(parsed) if isinstance(parsed, (list, tuple)) else [text]
        elif text:
            values = [text]
        else:
            return ()
    else:
        values = [raw]
    return tuple(
        str(value).strip().lower() for value in values if str(value).strip()
    )


def licenses_allowed(raw: Any, allowed: frozenset[str] = ALLOWED_LICENSES) -> bool:
    """Лицензия проходит, если она непуста и ВСЯ входит в белый список.

    Консервативно: файл с набором ``['MIT', 'GPL-3.0']`` отбрасывается — смесь
    не доказывает, что применима разрешённая лицензия.
    """
    found = normalize_licenses(raw)
    return bool(found) and all(value in allowed for value in found)


def truncate_bytes(text: str, limit: int = MAX_FILE_BYTES) -> str:
    """Усечь текст до ``limit`` байт UTF-8, не разрывая символ."""
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text
    return encoded[:limit].decode("utf-8", "ignore")


# --------------------------------------------------------------------------- #
# Поток записей
# --------------------------------------------------------------------------- #


class _Tagged:
    """Итератор языка: помечает записи целевым языком и подсказкой об источнике."""

    def __init__(self, iterator: Iterator[dict], language: str, spec: SourceSpec) -> None:
        self.iterator = iterator
        self.language = language
        self.spec = spec

    def __iter__(self) -> "_Tagged":
        return self

    def __next__(self) -> dict:
        record = dict(next(self.iterator))
        record.setdefault("_target_lang", self.language)
        record.setdefault("_source_repo", self.spec.repo)
        return record


def _round_robin(iterators: Sequence[Iterator[dict]]) -> Iterator[dict]:
    """Чередование потоков языков: микс в шарде, а не «сначала весь Python»."""
    active = list(iterators)
    while active:
        for iterator in list(active):
            try:
                yield next(iterator)
            except StopIteration:
                active.remove(iterator)


def iter_stack_documents(
    name: str = DEFAULT_SOURCE_NAME,
    languages: Sequence[str] | None = None,
    source_spec: dict | None = None,
) -> Iterator[dict]:
    """Поток записей источника кода по языкам (``datasets`` streaming).

    При ``per_language_configs`` каждому языку соответствует свой конфиг, потоки
    чередуются. Гейтед-источник без выданного доступа падает здесь же с
    подсказкой, как получить доступ, — до первого записанного документа.
    """
    if source_spec is not None and source_spec.get("kind") == "local":
        func, kwargs = common.source_iterator(source_spec)
        return common.iter_source(func, **kwargs)

    if name not in SOURCES:
        raise ValueError(f"неизвестный источник кода: {name!r} (есть: {sorted(SOURCES)})")
    spec = SOURCES[name]
    langs = tuple(languages or LANGUAGES)
    common.sanitize_proxy_env()

    def single(language: str | None) -> Iterator[dict]:
        config = spec.config_for(language) if language else spec.config_for("*")
        data_files = None
        if spec.data_files_template and language:
            data_files = spec.data_files_template.format(lang=config or language)
        try:
            return common.iter_hf_stream(
                spec.repo, config, split="train", data_files=data_files
            )
        except Exception as exc:  # pragma: no cover - сетевой путь, проверяется пробой
            raise RuntimeError(
                f"источник {spec.repo} (конфиг {config!r}) недоступен: {exc}. "
                f"{'Датасет закрыт гейтом — нужен выданный доступ к репозиторию.' if spec.gated else ''}"
            ) from exc

    if spec.per_language_configs:
        return _round_robin([_Tagged(single(lang), lang, spec) for lang in langs])
    return _Tagged(single(None), "*", spec)


# --------------------------------------------------------------------------- #
# Нормализация записи
# --------------------------------------------------------------------------- #


class StackFiles:
    """Нормализация записи кода → ``(text, meta)``; ``None`` — отброшена.

    Порядок фильтров: язык → длина (микробные отбрасываются, длинные усекаются)
    → лицензия. Причины отбраковки копятся в ``stats`` и идут в отчёт.
    """

    def __init__(
        self,
        languages: Sequence[str] = LANGUAGES,
        allowed_licenses: frozenset[str] = ALLOWED_LICENSES,
        min_bytes: int = MIN_FILE_BYTES,
        max_bytes: int = MAX_FILE_BYTES,
        source_name: str = DEFAULT_SOURCE_NAME,
    ) -> None:
        self.languages = tuple(languages)
        self.allowed_licenses = allowed_licenses
        self.min_bytes = min_bytes
        self.max_bytes = max_bytes
        self.source_name = source_name
        self.stats: dict[str, int] = {
            "dropped_language": 0,
            "dropped_too_small": 0,
            "dropped_license": 0,
            "dropped_empty_text": 0,
            "truncated_oversized": 0,
        }
        self.by_language: dict[str, int] = {}

    def __call__(self, record: dict) -> tuple[str, dict] | None:
        spec = SOURCES.get(self.source_name)
        text_fields = spec.text_fields if spec else ("content", "text")
        license_fields = spec.license_fields if spec else ("license", "licenses")
        lang_fields = spec.lang_fields if spec else ("lang", "language")
        path_fields = spec.path_fields if spec else ("path",)
        repo_fields = spec.repo_fields if spec else ("repo_name",)

        text = common.hm_get(record, *text_fields)
        if not isinstance(text, str) or not text.strip():
            self.stats["dropped_empty_text"] += 1
            return None

        target_lang = record.get("_target_lang")
        if target_lang == "*":  # один поток на все языки — целевого языка нет
            target_lang = None
        assumed = spec.assumed_language if spec else None
        language = common.hm_get(record, *lang_fields, default=target_lang or assumed)
        language = str(language).strip().lower() if language else ""
        aliases = {"py": "python", "js": "javascript", "sh": "shell", "bash": "shell", "golang": "go"}
        language = aliases.get(language, language)
        if self.languages and language not in self.languages:
            self.stats["dropped_language"] += 1
            return None

        size = len(text.encode("utf-8"))
        if size < self.min_bytes:
            self.stats["dropped_too_small"] += 1
            return None
        if size > self.max_bytes:
            self.stats["truncated_oversized"] += 1
            text = truncate_bytes(text, self.max_bytes)

        raw_license = common.hm_get(record, *license_fields)
        if not licenses_allowed(raw_license, self.allowed_licenses):
            self.stats["dropped_license"] += 1
            return None

        licenses = normalize_licenses(raw_license)
        meta = {
            "source": record.get("_source_repo") or self.source_name,
            "lang": language,
            "license": ",".join(licenses),
            "path": common.hm_get(record, *path_fields),
            "repo": common.hm_get(record, *repo_fields),
        }
        self.by_language[language] = self.by_language.get(language, 0) + 1
        return text, {key: value for key, value in meta.items() if value is not None}


# --------------------------------------------------------------------------- #
# Прогон
# --------------------------------------------------------------------------- #

#: Итератор ``stack`` для общего прогона: имя источника и языки из спецификации.
STACK_ITERATORS: dict[str, tuple[Any, tuple[str, ...]]] = {
    **common.SOURCE_ITERATORS,
    "stack": (iter_stack_documents, ("name", "languages")),
}


def check_source(
    name: str,
    languages: Sequence[str] | None = None,
    sample: int = 2000,
) -> dict:
    """Доступность источника: читается ли поток и что из него проходит фильтры.

    Кроме факта чтения проверяется ПРИГОДНОСТЬ на пробе из ``sample`` записей:
    источник может открываться, но не давать ни одной записи нужных языков и
    лицензий (так ведёт себя ``common-pile/stackv2``). ``usable`` — признак,
    по которому ``--source auto`` выбирает рабочий источник.
    """
    spec = SOURCES.get(name)
    if spec is None:
        return {"source": name, "ok": False, "usable": False, "error": "нет в реестре"}
    started = time.time()
    outcome: dict = {
        "source": name,
        "repo": spec.repo,
        "gated": spec.gated,
        "note": spec.note,
    }
    try:
        langs = tuple(languages or LANGUAGES)
        normalizer = StackFiles(languages=langs, source_name=name)
        iterator = iter(iter_stack_documents(name, langs))
        languages_seen: dict[str, int] = {}
        licenses_seen: dict[str, int] = {}
        kept = 0
        read = 0
        for record in iterator:
            if read >= sample:
                break
            read += 1
            raw_lang = common.hm_get(record, *spec.lang_fields)
            if raw_lang:
                key = str(raw_lang).strip().lower()
                languages_seen[key] = languages_seen.get(key, 0) + 1
            for value in normalize_licenses(common.hm_get(record, *spec.license_fields)):
                licenses_seen[value] = licenses_seen.get(value, 0) + 1
            if normalizer(record) is not None:
                kept += 1
        outcome.update(
            {
                "ok": True,
                "usable": kept > 0,
                "sample": read,
                "kept": kept,
                "kept_share": round(kept / read, 4) if read else 0.0,
                "languages_seen": dict(sorted(languages_seen.items(), key=lambda kv: -kv[1])[:10]),
                "licenses_seen": dict(sorted(licenses_seen.items(), key=lambda kv: -kv[1])[:10]),
                "dropped_by_rule": dict(normalizer.stats),
            }
        )
    except Exception as exc:
        outcome.update(
            {
                "ok": False,
                "usable": False,
                "error": f"{type(exc).__name__}: {exc}"[:400],
            }
        )
    outcome["seconds"] = round(common.elapsed(started), 3)
    return outcome


def prepare_c(
    out_dir: str | os.PathLike[str] | None = None,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    source_name: str = DEFAULT_SOURCE_NAME,
    languages: Sequence[str] = LANGUAGES,
    manifest_path: str | os.PathLike[str] | None = None,
    report_path: str | os.PathLike[str] | None = None,
    source_spec: dict | None = None,
    **kwargs: Any,
) -> dict:
    """Подготовить шард C: поток по языкам → фильтры → дедуп → jsonl-шарды."""
    root = out_dir or os.path.join(common.DATASET_ROOT, SHARD)
    manifest = manifest_path or os.path.join(root, "manifest-c.json")
    report = report_path or os.path.join(root, "report-c.json")
    spec = dict(source_spec or SOURCES[source_name].spec(languages))
    normalizer = StackFiles(
        languages=languages,
        source_name=source_name if spec.get("kind") == "stack" else DEFAULT_SOURCE_NAME,
    )
    result = common.run_shard(
        shard=SHARD,
        out_dir=root,
        target_tokens=target_tokens,
        source_spec=spec,
        normalize=normalizer,
        manifest_path=manifest,
        report_path=report,
        source_iterators=STACK_ITERATORS,
        **kwargs,
    )
    result["rules"] = {
        "languages": list(languages),
        "allowed_licenses": sorted(ALLOWED_LICENSES),
        "min_file_bytes": MIN_FILE_BYTES,
        "max_file_bytes": MAX_FILE_BYTES,
        "kept_by_language": normalizer.by_language,
    }
    result["report"] = str(common.write_report(report, result))
    return result
