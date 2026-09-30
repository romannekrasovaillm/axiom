"""Шард C: код (ADR-021, ~3B токенов) — The Stack, лицензионный фильтр.

Языки: python, rust, go, javascript, shell (у ``codeparrot-clean`` — только
python, это свойство источника, см. ``assumed_language``).
Лицензии: mit, apache-2.0, bsd-3-clause, bsd-2-clause, isc, 0bsd.
Длина файла: 256 Б ≤ len ≤ 100 КБ (микробные выбрасываются, длинные усекаются).

Реестр источников (``SOURCES``) отражает реальность, а не пожелание:

* ``stack-v2-dedup`` — приоритет ADR-021 (``bigcode/the-stack-v2-dedup``).
  Конфиги по языкам ровно те, что нужны (``Python``/``Rust``/``Go``/
  ``JavaScript``/``Shell``) — стриминг по языку структурно осуществим;
  на практике контент лежит blob-ссылками (замер 28.09.2026: 150 118 722
  прочитанных записей — все ``dropped_empty_text``), раскрытие требует S3
  BigCode — отклонено владельцем;
* ``stack-dedup-v1`` — фолбэк (``bigcode/the-stack-dedup``): контенты файлов
  и поля ``lang``/``license``/``size`` лежат прямо в строке;
* ``stack-smol-xl`` — публичный срез Stack v1: объём мал (замер: 0,079B
  токенов при цели 3B), годится на приёмку пайплайна, не на боевую загрузку;
* ``common-pile-stackv2`` — публичный, но объявленных языков в потоке почти
  нет (язык — детектированный формат): нулевой выход замерен;
* ``codeparrot-clean`` — **план B ADR-021** (``codeparrot/codeparrot-clean``):
  публичный НЕ-gated, только python, 54 ``json.gz`` ≈ 12,8 ГБ сжатых
  (≈ 50 ГБ текста, 5 361 373 файла) — единственный публичный источник,
  которого хватает на цель 3B токенов. Поля: ``content`` (текст),
  ``license``/``path``/``repo_name`` (справка). Пометка в манифесте —
  ``python-only fallback (ADR-021 план B)``.

  **Лицензионный фильтр к нему НЕ применяется** (``license_preselected``) —
  так заявлено в задаче дельты. Замер 30.09.2026 её основание не подтвердил:
  карточка датасета фильтрации по лицензиям не заявляет (её шаги чистки —
  дедуп, длина строк, доля букв, отсев автогенерации), а поле ``license``
  (лицензия репозитория, унаследованная файлом) на 20 000 записей даёт
  apache-2.0 21,9 %, gpl-3.0 17,9 %, mit 17,4 %, bsd-3-clause 16,6 %,
  agpl-3.0 9,5 %, gpl-2.0 9,4 %, остальное ~7 % — то есть **copyleft ≈ 41 %**.

  Белый список на том же замере пропускает 57,8 % записей (57,6 % символов),
  и этого хватает с запасом: ≈ 28,8 ГБ текста → **7,19B токенов** при цели 3B.
  Политика выбрана архитектором; ниже — факт, а не рекомендация: с выключенным
  фильтром в шард C попадает ~41 % copyleft-кода, и это видно в отчёте
  (``rules.licenses_seen`` — распределение) и в предупреждении CLI. Если
  архитектор выберет белый список — снять ``license_preselected`` у источника
  и перезапустить прогон (шарды привязаны к политике: другой manifest).

Оба bigcode-датасета закрыты гейтом (``gated: auto``): без выданного доступа
итератор падает с внятной ошибкой до первого документа — пайплайн не
«деградирует молча», источник обязан быть назван явно (``--source``).
"""

from __future__ import annotations

import ast
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
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
    #: True — лицензионный фильтр пайплайна к источнику НЕ применяется
    #: (codeparrot-clean, политика дельты). Выключение фильтра — это изменение
    #: состава корпуса, поэтому оно именовано, видно в манифесте
    #: (``license_filter: preselected``) и в отчёте — с измеренным
    #: распределением лицензий (``rules.licenses_seen``), а не умалчивается.
    license_preselected: bool = False
    #: Оговорка источника для манифеста — чем он ограничен по факту.
    manifest_note: str | None = None

    def config_for(self, language: str) -> str | None:
        if not self.per_language_configs:
            return self.configs.get("*")
        return self.configs.get(language, language)

    def spec(self, languages: Sequence[str] = LANGUAGES) -> dict:
        """Спецификация источника для манифеста: имя, языки и оговорки.

        Ограничения источника фиксируются здесь — в самом манифесте, а не
        постфактум в отчёте: одноязычный корпус сужает список языков до своего
        (манифест не должен обещать языки, которых в источнике нет), политика
        лицензий и оговорка идут отдельными полями.
        """
        langs = list(languages)
        if self.assumed_language:
            langs = [lang for lang in langs if lang == self.assumed_language] or [
                self.assumed_language
            ]
        payload: dict = {"kind": "stack", "name": self.name, "languages": langs}
        if self.manifest_note:
            payload["note"] = self.manifest_note
        if self.license_preselected:
            payload["license_filter"] = "preselected"
        return payload


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
            "объём мал (замер: 50 000 записей → 0,079B токенов при цели 3B) — "
            "годится для пробы и приёмки пайплайна, не для боевой загрузки"
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
            "план B ADR-021: публичный НЕ-gated фолбэк (только python), 54 json.gz "
            "≈ 12,8 ГБ сжатых ≈ 50 ГБ текста — на цель 3B хватает; поле content, "
            "поля языка в записи НЕТ — корпус одноязычный по построению "
            "(assumed_language=python); лицензионный фильтр НЕ применяется "
            "(license_preselected) — политика дельты; распределение лицензий "
            "измеряется в отчёте (замер 30.09.2026: copyleft ≈ 41 %)"
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
        license_preselected=True,
        manifest_note="python-only fallback (ADR-021 план B)",
    ),
}

DEFAULT_SOURCE_NAME = "stack-dedup-v1"

#: Порядок перебора для ``--source auto``: приоритет ADR-021, затем фолбэк,
#: затем публичные источники (нужны, пока доступ к bigcode не выдан).
#: ``codeparrot-clean`` стоит ВПЕРЕД ``stack-smol-xl``: оба публичные, но
#: у среза Stack v1 измеренный потолок 0,079B при цели 3B — авто-выбор не
#: должен приводить к источнику, которого на цель заведомо не хватает.
AUTO_ORDER: tuple[str, ...] = (
    "stack-v2-dedup",
    "stack-dedup-v1",
    "codeparrot-clean",
    "stack-smol-xl",
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

    Лицензионный шаг выключается свойством источника
    (``license_preselected``, codeparrot-clean): датасет уже отфильтрован по
    лицензиям на стороне источника, и повторный белый список отбраковал бы
    годный текст по полю-справке. Для остальных источников белый список
    действует без изменений.
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
        source = SOURCES.get(source_name)
        self.license_preselected = bool(source and source.license_preselected)
        #: Политика лицензий — свойство нормализатора: шаг выключен, если
        #: у источника фильтр снят политикой. Отчёт обязан нести это фактом, а
        #: не умалчивать: ``rules["license_filter"]`` + распределение лицензий.
        self.licenses_seen: dict[str, int] = {}
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
        if not self.license_preselected and not licenses_allowed(
            raw_license, self.allowed_licenses
        ):
            self.stats["dropped_license"] += 1
            return None

        licenses = normalize_licenses(raw_license)
        if self.license_preselected:
            # Политика фильтра выключена — распределение лицензий всё равно
            # измеряется: иначе шард уходит в корпус «разрешённым по умолчанию».
            for value in licenses or ("<нет поля>",):
                if value in self.licenses_seen or len(self.licenses_seen) < 64:
                    self.licenses_seen[value] = self.licenses_seen.get(value, 0) + 1
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
        # Политика лицензий — свойство источника: у codeparrot-clean фильтр
        # снят (license_preselected), у остальных — белый список.
        "license_filter": "preselected" if spec.license_preselected else "whitelist",
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


def _manifest_state(path: str | os.PathLike[str]) -> dict:
    """Что уже лежит в манифесте прошлого прогона (пусто — манифеста нет)."""
    try:
        data = json.loads(
            Path(os.path.expanduser(os.fspath(path))).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return {}
    source = data.get("source")
    return {
        "source": source.get("name") if isinstance(source, dict) else None,
        "shards": len(data.get("shards") or []),
        "source_records": int(data.get("source_records") or 0),
    }


def _source_change_guard(
    manifest: str | os.PathLike[str],
    requested: str | None,
    restart: bool,
) -> dict | None:
    """Курсор манифеста принадлежит прежнему источнику — чужой resume опасен.

    ``source_records`` — позиция в потоке КОНКРЕТНОГО источника. Если в том же
    манифесте запускается другой источник, проматывание чужого диапазона даёт
    либо тихий ``source_exhausted`` на живом датасете, либо пропуск куска
    корпуса. Пустой манифест (прошлый прогон не дал шардов) лечится сбросом
    курсора; непустой — смешивать шарды разных источников нельзя, нужен свой
    ``--out``/``--manifest`` или явный ``--restart``.
    """
    state = _manifest_state(manifest)
    previous = state.get("source")
    if previous is None or requested is None or previous == requested or restart:
        return None
    if state["shards"]:
        raise ValueError(
            f"манифест {manifest} собран источником {previous!r} "
            f"({state['shards']} шардов), а запрошен {requested!r}: шарды разных "
            f"источников в одном каталоге смешивать нельзя. Задайте отдельный "
            f"--out/--manifest либо --restart, если прежние шарды не нужны"
        )
    return {
        "previous_source": previous,
        "current_source": requested,
        "previous_source_records": state["source_records"],
        "previous_shards": state["shards"],
        "reason": "манифест прошлого прогона принадлежит другому источнику — курсор сброшен",
    }


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
    if source_spec is None:
        if source_name not in SOURCES:
            raise ValueError(
                f"неизвестный источник кода: {source_name!r} (есть: {sorted(SOURCES)})"
            )
        spec = dict(SOURCES[source_name].spec(languages))
        # Источник может сузить языки (одноязычный корпус): фильтры обязаны
        # судиться по тому же списку, что обещан манифестом.
        languages = tuple(spec["languages"])
    else:
        spec = dict(source_spec)

    source_reset = _source_change_guard(
        manifest, spec.get("name"), bool(kwargs.get("restart"))
    )
    if source_reset is not None:
        kwargs["restart"] = True

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
    if source_reset is not None:
        result["manifest_source_reset"] = source_reset
    result["rules"] = {
        "languages": list(languages),
        "allowed_licenses": sorted(ALLOWED_LICENSES),
        "license_filter": "preselected" if normalizer.license_preselected else "whitelist",
        "license_filter_note": (
            f"не применяется (license_preselected) у {spec.get('name')}: фильтр "
            f"снят политикой источника, состав шарда по лицензиям измерен в "
            f"licenses_seen (карточка codeparrot-clean фильтрации по лицензиям "
            f"не заявляет: её чистка — дедуп, длина строк, доля букв, отсев "
            f"автогенерации)"
            if normalizer.license_preselected
            else "белый список: файл проходит, если ВСЕ заявленные лицензии разрешены"
        ),
        "min_file_bytes": MIN_FILE_BYTES,
        "max_file_bytes": MAX_FILE_BYTES,
        "kept_by_language": normalizer.by_language,
    }
    if normalizer.license_preselected:
        # Фильтр выключен политикой источника: распределение лицензий в шарде
        # измерено и положено в отчёт (иначе «разрешённый корпус» — допущение).
        result["rules"]["licenses_seen"] = dict(
            sorted(normalizer.licenses_seen.items(), key=lambda kv: -kv[1])
        )
    result["report"] = str(common.write_report(report, result))
    return result
