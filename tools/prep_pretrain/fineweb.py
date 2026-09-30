"""Шарды W (веб, ~17B) и Q (decay/annealing, ~1B) — источники-веб ADR-021.

**Шард W.** Источник — ``HuggingFaceFW/fineweb-edu``, конфиг ``sample-100BT``
(100B токенов edu-выборки: цели 17B хватает с запасом, а множество parquet-файлов
на порядок меньше полного ``data/*/*``). Чтение — потоковое (``datasets``
streaming): датасет не материализуется, в памяти живёт одна запись и множество
хешей дедупа.

Фильтры: качество уже отфильтровано edu-классификатором на стороне датасета —
дополнительный порог ``min_int_score`` по умолчанию ВЫКЛЮЧЕН (ручка оставлена
зарезервированной, её значение фиксируется в отчёте). Дедупликация документов —
точная, по sha256 текста, с ограниченным окном (страховка: FineWeb уже
дедуплицирован).

Запись: ``{"text": …, "meta": {"source": …, "id": …, "url": …, "dump": …}}``
в jsonl-шарды ``W-000NN.jsonl.zst`` по ~500 МБ.

**Шард Q** (decay-фаза ADR-021, ~1B токенов) — сужёный микс двух источников,
уже покрытых шардами W и C:

* веб-часть — тот же FineWeb-Edu, но строже по качеству и длине
  (:class:`QualityDocuments`: edu-порог ``int_score >= 4`` и окно длины,
  по умолчанию 800…50 000 символов);
* код-часть — кодовые примеры С ТЕСТАМИ из ``codeparrot-clean``
  (:class:`TestCodeDocuments`: фильтры шарда C + обязательный тест-маркер
  ``def test_``/``unittest``/``pytest``), целевая доля — 15 % микса Q
  (:class:`MixGovernor`);
* дедуп — против записей W/C опорным near-dup индексом
  (:class:`PriorNearDupIndex`, MinHash из ``axiom_ds.dedup``) плюс штатный
  точный дедуп ``common.run_shard``.

Запись Q: ``{"text": …, "meta": {"component": "web"|"code", …}}`` в шарды
``Q-000NN.jsonl.zst``. Поток Q воспроизводим при resume: решение о источнике
следующей записи зависит только от позиций в исходных потоках, а не от того, что
прошло фильтры (иначе проматывание курсора манифеста привело бы к потере или
дублю записей).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from . import common, stack

try:  # near-dup Deduper — один на репозиторий (ADR-020, дельта-2), не дублируем MinHash
    from axiom_ds import dedup as near_dup
except ImportError:  # pragma: no cover - tools/ не в sys.path: near-dup недоступен
    near_dup = None  # type: ignore[assignment]
    _NEAR_DUP_ERROR = (
        "модуль axiom_ds.dedup недоступен: положите tools/ в sys.path (запуск через "
        "`python -m prep_pretrain.build` из tools/ делает это сам) — near-dup дедуп Q "
        "без него не выполняется, подмены «точным дедупом» нет"
    )

SHARD = "W"

#: Канонический репозиторий; HuggingFaceH4/fineweb-edu — тот же корпус-зеркало.
DEFAULT_REPO = "HuggingFaceFW/fineweb-edu"
DEFAULT_CONFIG = "sample-100BT"
DEFAULT_SPLIT = "train"

#: Боевой источник по умолчанию (спецификация JSON-сериализуема и идёт в отчёт).
DEFAULT_SOURCE: dict[str, Any] = {
    "kind": "hf",
    "repo": DEFAULT_REPO,
    "config": DEFAULT_CONFIG,
    "split": DEFAULT_SPLIT,
}

DEFAULT_TARGET_TOKENS = 17_000_000_000


class FinewebDocuments:
    """Нормализация записи FineWeb-Edu → ``(text, meta)``; ``None`` — отброшена.

    Объект копит причины отбраковки в ``stats`` — они попадают в отчёт.
    """

    def __init__(
        self,
        min_int_score: int | None = None,
        max_chars: int | None = None,
        source_name: str = DEFAULT_REPO,
    ) -> None:
        self.min_int_score = min_int_score
        self.max_chars = max_chars
        self.source_name = source_name
        self.stats: dict[str, int] = {
            "dropped_empty_text": 0,
            "dropped_low_int_score": 0,
            "dropped_oversized": 0,
        }

    def __call__(self, record: dict) -> tuple[str, dict] | None:
        text = record.get("text") or ""
        if not isinstance(text, str) or not text.strip():
            self.stats["dropped_empty_text"] += 1
            return None
        if self.min_int_score is not None:
            score = record.get("int_score")
            if score is None or int(score) < self.min_int_score:
                self.stats["dropped_low_int_score"] += 1
                return None
        if self.max_chars is not None and len(text) > self.max_chars:
            self.stats["dropped_oversized"] += 1
            return None
        meta = {
            "source": record.get("source") or self.source_name,
            "id": record.get("id"),
            "dump": record.get("dump"),
            "url": record.get("url"),
            "language": record.get("language"),
            "int_score": record.get("int_score"),
        }
        return text, {key: value for key, value in meta.items() if value is not None}


def iter_documents(source_spec: dict | None = None) -> Iterator[dict]:
    """Поток сырых записей источника W (для проб и отладки)."""
    spec = source_spec or DEFAULT_SOURCE
    func, kwargs = common.source_iterator(spec)
    return common.iter_source(func, **kwargs)


def prepare_w(
    out_dir: str | os.PathLike[str] | None = None,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    source_spec: dict | None = None,
    manifest_path: str | os.PathLike[str] | None = None,
    report_path: str | os.PathLike[str] | None = None,
    min_int_score: int | None = None,
    **kwargs: Any,
) -> dict:
    """Подготовить шард W: поток → фильтр → дедуп → jsonl-шарды в ``out_dir``."""
    root = out_dir or os.path.join(common.DATASET_ROOT, SHARD)
    source = dict(source_spec or DEFAULT_SOURCE)
    source_name = source.get("repo") or source.get("glob") or DEFAULT_REPO
    manifest = manifest_path or os.path.join(root, "manifest-w.json")
    report = report_path or os.path.join(root, "report-w.json")
    return common.run_shard(
        shard=SHARD,
        out_dir=root,
        target_tokens=target_tokens,
        source_spec=source,
        normalize=FinewebDocuments(min_int_score=min_int_score, source_name=str(source_name)),
        manifest_path=manifest,
        report_path=report,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Шард Q: decay/annealing-микс (ADR-021, ~1B токенов)
# --------------------------------------------------------------------------- #

SHARD_Q = "Q"

#: Сужёный edu-порог Q. FineWeb-Edu отдаёт ``int_score`` 0…5; в W порог выключен
#: (берётся всё, что пропустил классификатор датасета), в Q — только верхняя
#: полка. Замер 30.09.2026 по шарду W: score 3 — 85,7 % записей, 4 — 14,3 %,
#: 5 — 0,08 %; порог 4 сужает веб-часть до ~14 % исходного потока.
DEFAULT_Q_MIN_INT_SCORE = 4

#: Окно длины документа Q (символы). Короткие не дают содержательного контекста,
#: очень длинные раздувают один документ в ущерб разнообразию decay-выборки.
#: Границы — стартовая гипотеза, фиксируются в отчёте и манифесте.
DEFAULT_Q_MIN_CHARS = 800
DEFAULT_Q_MAX_CHARS = 50_000

#: Целевая доля кодовых примеров в Q по approx-токенам (ADR-021: стартовая
#: гипотеза 85/15, корректируется только до запуска претрейна).
DEFAULT_CODE_SHARE = 0.15

#: Ожидаемая «урожайность» источника — approx-токенов на символ ИСХОДНОГО текста
#: (замер 30.09.2026: строгий порог int_score ≥ 4 пропускает 14,3 % записей
#: FineWeb-Edu со средней длиной записи ≈ 0,9 КБ/4 символа на токен; тест-маркеры
#: — 19,5 % записей codeparrot-clean при средней длине ≈ 11,2 КБ/4). Урожайность
#: нужна, чтобы цель «15 % по записанным токенам» переводилась в решение о
#: потоке-источнике на входе (см. :class:`MixGovernor`); фактические значения
#: прогона измеряются и попадают в отчёт (``mix.measured_yield``).
DEFAULT_WEB_YIELD = 0.032
DEFAULT_CODE_YIELD = 0.049

#: Кодовые примеры «с тестами»: маркеры тестов в тексте файла (задача дельты).
#: Имена маркеров человекочитаемы (они идут в мету и отчёт), шаблоны — рядом;
#: ``unittest``/``pytest`` ищутся как слова, чтобы не ловить подстроки имён.
TEST_MARKERS: tuple[str, ...] = ("def test_", "unittest", "pytest")
TEST_MARKER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("def test_", re.compile(r"def test_", re.IGNORECASE)),
    ("unittest", re.compile(r"\bunittest\b", re.IGNORECASE)),
    ("pytest", re.compile(r"\bpytest\b", re.IGNORECASE)),
)

DEFAULT_TARGET_TOKENS_Q = 1_000_000_000

#: Опорный near-dup индекс Q строится из первых ``max_records`` записей W/C:
#: полный W (17B токенов) в память подписей не влезает, поэтому опора — окно,
#: и это окно, а не «весь W», попадает в отчёт числом.
DEFAULT_PRIOR_MAX_RECORDS = 50_000

#: Расширения шард-файлов, читаемых как опора (``common.CODEC_EXTENSIONS``).
_SHARD_SUFFIXES = (".jsonl.zst", ".jsonl.gz", ".jsonl")

CODE_COMPONENT = "code"
WEB_COMPONENT = "web"


def test_markers(text: str) -> list[str]:
    """Маркеры тестов, найденные в тексте (пусто — файл без тестов)."""
    return [marker for marker, pattern in TEST_MARKER_PATTERNS if pattern.search(text)]


# --------------------------------------------------------------------------- #
# Нормализаторы Q
# --------------------------------------------------------------------------- #


class QualityDocuments(FinewebDocuments):
    """Сужёный фильтр Q поверх источника W: строгий edu-порог + окно длины.

    Наследует фильтр W (пустой текст, edu-порог, верхняя граница длины) и
    добавляет нижнюю границу длины; счётчики причин отбоя — свои плюс
    унаследованные.
    """

    def __init__(
        self,
        min_int_score: int | None = DEFAULT_Q_MIN_INT_SCORE,
        min_chars: int | None = DEFAULT_Q_MIN_CHARS,
        max_chars: int | None = DEFAULT_Q_MAX_CHARS,
        source_name: str = DEFAULT_REPO,
    ) -> None:
        super().__init__(min_int_score=min_int_score, max_chars=max_chars, source_name=source_name)
        self.min_chars = min_chars
        self.stats["dropped_short"] = 0

    def __call__(self, record: dict) -> tuple[str, dict] | None:
        text = record.get("text") or ""
        if (
            self.min_chars is not None
            and isinstance(text, str)
            and text.strip()
            and len(text) < self.min_chars
        ):
            self.stats["dropped_short"] += 1
            return None
        return super().__call__(record)


class TestCodeDocuments(stack.StackFiles):
    """Кодовые примеры с тестами Q: фильтры шарда C + обязательный тест-маркер.

    Порядок ступеней: тест-маркер (по сырому тексту — счётчик ``dropped_no_tests``
    показывает селективность самого маркера, а не остатка после остальных
    фильтров) → язык → длина → лицензия (``stack.StackFiles`` целиком, включая
    политику ``license_preselected`` источника). В мете записи фиксируются
    найденные маркеры — свидетельство того, почему файл попал в decay-выборку.
    """

    def __init__(
        self,
        languages: Sequence[str] = ("python",),
        source_name: str = "codeparrot-clean",
        markers: Sequence[str] = TEST_MARKERS,
        **kwargs: Any,
    ) -> None:
        super().__init__(languages=languages, source_name=source_name, **kwargs)
        self.markers = tuple(markers)
        if self.markers == TEST_MARKERS:
            self._patterns = TEST_MARKER_PATTERNS
        else:  # свой набор маркеров (тесты): имена экранируются как подстроки
            self._patterns = tuple(
                (marker, re.compile(re.escape(marker), re.IGNORECASE)) for marker in self.markers
            )
        self.stats["dropped_no_tests"] = 0

    def found_markers(self, text: str) -> list[str]:
        return [marker for marker, pattern in self._patterns if pattern.search(text)]

    def __call__(self, record: dict) -> tuple[str, dict] | None:
        spec = stack.SOURCES.get(self.source_name)
        fields = spec.text_fields if spec else ("content", "text")
        raw = common.hm_get(record, *fields)
        if isinstance(raw, str) and raw.strip():
            found = self.found_markers(raw)
            if not found:
                self.stats["dropped_no_tests"] += 1
                return None
        parsed = super().__call__(record)
        if parsed is None:
            return None
        text, meta = parsed
        meta["tests"] = ",".join(self.found_markers(text))
        return text, meta


# --------------------------------------------------------------------------- #
# Опорный near-dup индекс (дедуп Q против W/C)
# --------------------------------------------------------------------------- #


def _require_near_dup():
    if near_dup is None:  # pragma: no cover - зависит от sys.path
        raise ImportError(_NEAR_DUP_ERROR)
    return near_dup


class PriorNearDupIndex(_require_near_dup().Deduper if near_dup is not None else object):
    """Near-dup индекс опорных записей (W/C): чем проверяется Q.

    Переиспользуется ``axiom_ds.dedup.Deduper`` (ADR-020, дельта-2): точный дубль —
    совпал sha256 нормализованного текста, near-dup — оценка Jaccard по бандам
    MinHash ≥ ``threshold``. Отличие от базового класса одно и принципиальное:
    **проверяемый документ в индекс не добавляется** (``duplicate`` вместо
    ``add``). Опорой служит заранее ограниченное окно записей W/C, а не сам
    прогон: иначе память индекса росла бы со всем шардом Q (миллионы записей ×
    16 бандов), а задача — дедуп ПРОТИВ W/C. Окно и его размер честно попадают в
    отчёт (``prior.reference_records``), а не выдаются за «весь W».
    """

    def __init__(
        self,
        threshold: float | None = None,
        max_records: int = DEFAULT_PRIOR_MAX_RECORDS,
    ) -> None:
        module = _require_near_dup()
        super().__init__(threshold=module.JACCARD_THRESHOLD if threshold is None else threshold)
        if max_records <= 0:
            raise ValueError("max_records должен быть > 0")
        self.max_records = max_records
        self.reference_records = 0
        self.reference_duplicates = 0
        self.checked = 0
        self.candidate_pairs = 0
        self.dropped_exact = 0
        self.dropped_near = 0
        #: Секунды на проверки — MinHash считается в Python, и цена шага должна
        #: быть видна в отчёте, а не обнаруживаться по «прогон стал вдвое дольше».
        self.check_seconds = 0.0

    # -- опора -------------------------------------------------------------- #

    def index(self, text: str) -> bool:
        """Добавить опорный документ; False — он дубль уже проиндексированного."""
        if self.reference_records >= self.max_records:
            raise ValueError(f"опорный индекс полон: {self.max_records} записей")
        self.reference_records += 1
        kept = self.add(_as_document(text))
        if not kept:
            self.reference_duplicates += 1
            self.reference_records -= 1
        return kept

    # -- проверка ----------------------------------------------------------- #

    def duplicate(self, text: str) -> bool:
        """True — документ дубль опорного корпуса (точный или near-dup)."""
        module = _require_near_dup()
        started = time.time()
        self.checked += 1
        try:
            fingerprint = hashlib.sha256(
                module.normalize_text(text).encode("utf-8")
            ).hexdigest()
            if fingerprint in self._by_fingerprint:
                self.dropped_exact += 1
                return True
            signature = module.minhash_signature(text)
            for candidate_id in self._near_candidates(signature):
                other = self._signatures.get(candidate_id)
                if other is None:
                    continue
                self.candidate_pairs += 1
                if module.jaccard_estimate(signature, other) >= self.threshold:
                    self.dropped_near += 1
                    return True
            return False
        finally:
            self.check_seconds += time.time() - started

    @property
    def capacity_used(self) -> int:
        return self.reference_records

    def state(self) -> dict:
        return {
            "threshold": self.threshold,
            "max_records": self.max_records,
            "reference_records": self.reference_records,
            "reference_duplicates": self.reference_duplicates,
            "checked": self.checked,
            "candidate_pairs": self.candidate_pairs,
            "dropped_exact": self.dropped_exact,
            "dropped_near": self.dropped_near,
            "check_seconds": round(self.check_seconds, 3),
            "check_ms_per_record": (
                round(self.check_seconds * 1000 / self.checked, 2) if self.checked else None
            ),
        }


def _as_document(text: str) -> dict:
    """Документ в форме эпизода: ``Deduper`` работает с ходами, текст — один ход."""
    return {"id": "", "turns": [{"content": text}]}


def iter_shard_records(path: str | os.PathLike[str], chunk: int = 1 << 20) -> Iterator[dict]:
    """Поток записей шард-файла (``.jsonl.zst`` / ``.jsonl.gz`` / ``.jsonl``).

    Читается построчно и потоково: опорный индекс берёт из шарда первые N
    записей, поднимать 500-МБ шард в память нельзя.
    """
    target = Path(os.path.expanduser(os.fspath(path)))
    name = target.name
    if name.endswith(".jsonl.zst"):
        yield from _iter_zstd_lines(target, chunk)
    elif name.endswith(".jsonl.gz"):
        with gzip.open(target, "rt", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)
    elif name.endswith(".jsonl"):
        with target.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)
    else:
        raise ValueError(f"не шард-файл: {name} (ждали {_SHARD_SUFFIXES})")


def _iter_zstd_lines(path: Path, chunk: int) -> Iterator[dict]:
    import zstandard

    decompressor = zstandard.ZstdDecompressor().decompressobj()
    buffer = ""
    with path.open("rb") as handle:
        while True:
            piece = handle.read(chunk)
            if not piece:
                break
            buffer += decompressor.decompress(piece).decode("utf-8", "ignore")
            while True:  # noqa: SIM113 - построчная выдача из буфера
                newline = buffer.find("\n")
                if newline < 0:
                    break
                line, buffer = buffer[:newline], buffer[newline + 1 :]
                line = line.strip()
                if line:
                    yield json.loads(line)


def discover_prior_shards(
    dirs: Sequence[str | os.PathLike[str]] | None = None,
    shards: Sequence[str] = ("W", "C"),
) -> list[Path]:
    """Шард-файлы опорных корпусов: ``<dir>/<шара>/<шара>-*.jsonl*``, по порядку."""
    roots = [Path(os.path.expanduser(os.fspath(item))) for item in (dirs or [common.DATASET_ROOT])]
    found: list[Path] = []
    for root in roots:
        for shard in shards:
            directory = root / shard if (root / shard).is_dir() else root
            for suffix in _SHARD_SUFFIXES:
                found.extend(sorted(directory.glob(f"{shard}-*{suffix}")))
    seen: list[Path] = []
    for path in found:
        if path not in seen and not path.name.endswith(".part"):
            seen.append(path)
    return seen


def load_prior_reference(
    index: PriorNearDupIndex,
    paths: Sequence[str | os.PathLike[str]],
    progress: bool = False,
) -> dict:
    """Наполнить опорный индекс первыми ``index.max_records`` записями шардов."""
    started = time.time()
    files: list[dict] = []
    chars = 0
    records = 0
    for path in paths:
        target = Path(os.path.expanduser(os.fspath(path)))
        per_file = 0
        for record in iter_shard_records(target):
            if index.reference_records >= index.max_records:
                break
            text = record.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            index.index(text)
            per_file += 1
            records += 1
            chars += len(text)
        files.append({"file": str(target), "records": per_file})
        if progress:
            print(f"  [Q] опора {target.name}: {per_file} записей (всего {records})", flush=True)
        if index.reference_records >= index.max_records:
            break
    truncated = index.reference_records >= index.max_records
    return {
        "files": files,
        "files_read": len(files),
        "records": records,
        "chars": chars,
        "approx_tokens": max(0, chars // 4),
        "max_records": index.max_records,
        "truncated": truncated,
        "seconds": round(common.elapsed(started), 3),
    }


# --------------------------------------------------------------------------- #
# Поток Q: два источника, доля кода
# --------------------------------------------------------------------------- #


class MixGovernor:
    """Из какого источника тянуть следующую запись Q — по ожидаемым токенам.

    Ожидаемые (после фильтров) токены источника = потянутые символы ×
    урожайность (``web_yield``/``code_yield``). Решение зависит ТОЛЬКО от позиций
    в исходных потоках, а не от того, что прошло фильтры и дедуп: иначе поток
    перестал бы воспроизводиться при resume (курсор манифеста проматывается
    повторной генерацией потока — см. ``common.iter_source``), и повторный
    запуск терял бы или дублировал бы записи. Фактическая доля кода в шарде
    измеряется на выходе и попадает в отчёт: цель по входу — ручка, результат —
    свидетельство.
    """

    def __init__(
        self,
        code_share: float = DEFAULT_CODE_SHARE,
        web_yield: float = DEFAULT_WEB_YIELD,
        code_yield: float = DEFAULT_CODE_YIELD,
    ) -> None:
        if not 0.0 < code_share < 1.0:
            raise ValueError(f"доля кода должна быть в (0, 1): {code_share}")
        if web_yield <= 0 or code_yield <= 0:
            raise ValueError("урожайность источников должна быть > 0")
        self.code_share = float(code_share)
        self.web_yield = float(web_yield)
        self.code_yield = float(code_yield)
        self.web_chars = 0
        self.code_chars = 0
        self.pulls = {WEB_COMPONENT: 0, CODE_COMPONENT: 0}

    def note(self, component: str, chars: int) -> None:
        """Учесть потянутый из источника текст (символы исходной записи)."""
        self.pulls[component] = self.pulls.get(component, 0) + 1
        if component == CODE_COMPONENT:
            self.code_chars += chars
        else:
            self.web_chars += chars

    @property
    def expected_web_tokens(self) -> float:
        return self.web_chars * self.web_yield

    @property
    def expected_code_tokens(self) -> float:
        return self.code_chars * self.code_yield

    def pull_code(self) -> bool:
        """True — следующая запись берётся из кодового потока."""
        expected_total = self.expected_web_tokens + self.expected_code_tokens
        return self.expected_code_tokens <= self.code_share * expected_total

    def state(self) -> dict:
        return {
            "code_share": self.code_share,
            "web_yield": self.web_yield,
            "code_yield": self.code_yield,
            "web_chars": self.web_chars,
            "code_chars": self.code_chars,
            "expected_web_tokens": round(self.expected_web_tokens, 1),
            "expected_code_tokens": round(self.expected_code_tokens, 1),
            "pulls": dict(self.pulls),
        }


def _stream_from_spec(spec: dict, languages: Sequence[str]) -> Iterator[dict]:
    """Поток записей по спецификации: реестр кода (``stack``) либо common-источник."""
    if spec.get("kind") == "stack":
        return stack.iter_stack_documents(spec["name"], spec.get("languages") or languages)
    func, kwargs = common.source_iterator(spec)
    return common.iter_source(func, **kwargs)


def _pulled_text(record: dict, spec: dict) -> str:
    """Текст записи для учёта потянутых символов (поля источника, не нормализатора)."""
    if spec.get("kind") == "stack":
        source = stack.SOURCES.get(spec.get("name"))
        fields = source.text_fields if source else ("content", "text")
    else:
        fields = ("text", "content")
    value = common.hm_get(record, *fields)
    return value if isinstance(value, str) else ""


def iter_q_mix(
    web: dict | None = None,
    code: dict | None = None,
    code_share: float = DEFAULT_CODE_SHARE,
    web_yield: float = DEFAULT_WEB_YIELD,
    code_yield: float = DEFAULT_CODE_YIELD,
    languages: Sequence[str] = ("python",),
    state: dict | None = None,
    **_: Any,
) -> Iterator[dict]:
    """Смешанный поток Q: веб (сужёный W) + код с тестами, доля кода от governor.

    Каждая запись помечается ``_component`` (``web``/``code``) и ``_source_repo``;
    нормализатор Q затем применяет к ней фильтр своего компонента. Источник,
    исчерпавшийся раньше, перестаёт участвовать — оставшийся добирает цель
    (исчерпание и доли попадают в ``state``, оттуда — в отчёт).

    ``state`` — необязательный словарь наблюдений прогона (не влияет на решения
    потока): потянутые записи по компонентам и признак исчерпания источника.
    """
    web_spec = dict(web or DEFAULT_SOURCE)
    code_spec = dict(code or {"kind": "stack", "name": "codeparrot-clean"})
    governor = MixGovernor(code_share=code_share, web_yield=web_yield, code_yield=code_yield)
    streams = {
        WEB_COMPONENT: iter(_stream_from_spec(web_spec, languages)),
        CODE_COMPONENT: iter(_stream_from_spec(code_spec, languages)),
    }
    specs = {WEB_COMPONENT: web_spec, CODE_COMPONENT: code_spec}
    exhausted = {WEB_COMPONENT: False, CODE_COMPONENT: False}
    while not (exhausted[WEB_COMPONENT] and exhausted[CODE_COMPONENT]):
        component = CODE_COMPONENT if governor.pull_code() else WEB_COMPONENT
        if exhausted[component]:
            component = WEB_COMPONENT if component == CODE_COMPONENT else CODE_COMPONENT
            if exhausted[component]:
                break
        try:
            record = next(streams[component])
        except StopIteration:
            exhausted[component] = True
            if state is not None:
                state["exhausted"] = dict(exhausted)
            continue
        governor.note(component, len(_pulled_text(record, specs[component])))
        record = dict(record)
        record["_component"] = component
        record.setdefault(
            "_source_repo",
            specs[component].get("repo") or specs[component].get("name") or "",
        )
        if state is not None:
            # Наблюдения обновляются ДО yield: потребитель может оборвать
            # генератор (достигнута цель), и код после цикла не выполнится.
            state["exhausted"] = dict(exhausted)
            state["governor"] = governor.state()
        yield record


# --------------------------------------------------------------------------- #
# Нормализатор микса и прогон
# --------------------------------------------------------------------------- #


class QMixDocuments:
    """Нормализатор Q: маршрутизация по компоненту + дедуп против W/C.

    Счётчики разделены по компонентам (``web``/``code``) — из них собирается
    фактическая доля кода в шарде. ``kept`` считает записи, дошедшие до
    ``common.run_shard``, то есть после фильтров компонента и опорного near-dup;
    штатный точный дедуп шарда (окно LRU) стоит следующим шагом и виден в отчёте
    как ``counters.dropped_dedup``.
    """

    def __init__(
        self,
        web: QualityDocuments | None = None,
        code: TestCodeDocuments | None = None,
        prior: PriorNearDupIndex | None = None,
    ) -> None:
        self.web = web or QualityDocuments()
        self.code = code or TestCodeDocuments()
        self.prior = prior
        self.filtered = {WEB_COMPONENT: 0, CODE_COMPONENT: 0}
        self.dropped_prior = {WEB_COMPONENT: 0, CODE_COMPONENT: 0}
        self.kept = {
            WEB_COMPONENT: {"records": 0, "tokens": 0, "chars": 0},
            CODE_COMPONENT: {"records": 0, "tokens": 0, "chars": 0},
        }

    def inner(self, component: str) -> Any:
        return self.code if component == CODE_COMPONENT else self.web

    def __call__(self, record: dict) -> tuple[str, dict] | None:
        component = record.get("_component") or WEB_COMPONENT
        parsed = self.inner(component)(record)
        if parsed is None:
            return None
        text, meta = parsed
        self.filtered[component] = self.filtered.get(component, 0) + 1
        if self.prior is not None and self.prior.duplicate(text):
            self.dropped_prior[component] = self.dropped_prior.get(component, 0) + 1
            return None
        tokens = common.approx_tokens(text)
        bucket = self.kept[component]
        bucket["records"] += 1
        bucket["tokens"] += tokens
        bucket["chars"] += len(text)
        meta["component"] = component
        return text, meta

    @property
    def stats(self) -> dict[str, int]:
        """Причины отбоя обоих компонентов в одном словаре (с префиксом)."""
        merged: dict[str, int] = {}
        for component, inner in ((WEB_COMPONENT, self.web), (CODE_COMPONENT, self.code)):
            for key, value in inner.stats.items():
                merged[f"{component}:{key}"] = value
            merged[f"{component}:dropped_prior_duplicate"] = self.dropped_prior.get(component, 0)
        return merged

    def mix_state(self) -> dict:
        """Фактический микс: записи/токены по компонентам и доля кода."""
        web, code = self.kept[WEB_COMPONENT], self.kept[CODE_COMPONENT]
        total = web["tokens"] + code["tokens"]
        share = (code["tokens"] / total) if total else 0.0
        measured = {
            component: round(bucket["tokens"] / bucket["chars"], 6) if bucket["chars"] else 0.0
            for component, bucket in self.kept.items()
        }
        return {
            "web": dict(web),
            "code": dict(code),
            "web_records_filtered": self.filtered[WEB_COMPONENT],
            "code_records_filtered": self.filtered[CODE_COMPONENT],
            "approx_tokens": total,
            "code_share": round(share, 4),
            "measured_yield": measured,
            "basis": (
                "records/tokens — после фильтров компонента и опорного near-dup, "
                "до точного дедупа common.run_shard (counters.dropped_dedup)"
            ),
        }


def _q_source_signature(spec: dict) -> dict:
    """Существенные для воспроизводимости потока поля спецификации Q."""
    keys = ("web", "code", "code_share", "web_yield", "code_yield", "languages")
    return {key: spec.get(key) for key in keys}


def _assert_same_stream(
    manifest: common.Manifest, source_spec: dict, restart: bool = False
) -> None:
    """Смена правил/источников Q в том же манифесте — отказ, а не тихая мешанина.

    Курсор манифеста — позиция в конкретном смешанном потоке: другой источник,
    доля или урожайность дают другой поток, и проматывание чужого диапазона
    потеряло бы кусок корпуса. ``--restart`` — явный отказ от прежних шардов,
    поэтому проверка снимается.
    """
    previous = manifest.data.get("source")
    if restart or not isinstance(previous, dict) or not manifest.shards:
        return
    if previous == source_spec or _q_source_signature(previous) == _q_source_signature(source_spec):
        return
    raise ValueError(
        f"манифест {manifest.path} собран другим потоком Q "
        f"({_q_source_signature(previous)}), а запрошен {_q_source_signature(source_spec)}: "
        f"шарды и курсор принадлежат прежнему потоку. Задайте отдельный --out/--manifest "
        f"либо --restart, если прежние шарды не нужны"
    )


def prepare_q(
    out_dir: str | os.PathLike[str] | None = None,
    target_tokens: int = DEFAULT_TARGET_TOKENS_Q,
    web_spec: dict | None = None,
    code_spec: dict | None = None,
    code_share: float = DEFAULT_CODE_SHARE,
    web_yield: float = DEFAULT_WEB_YIELD,
    code_yield: float = DEFAULT_CODE_YIELD,
    min_int_score: int | None = DEFAULT_Q_MIN_INT_SCORE,
    min_chars: int | None = DEFAULT_Q_MIN_CHARS,
    max_chars: int | None = DEFAULT_Q_MAX_CHARS,
    languages: Sequence[str] = ("python",),
    prior_dirs: Sequence[str | os.PathLike[str]] | None = None,
    prior_max_records: int = DEFAULT_PRIOR_MAX_RECORDS,
    prior_threshold: float | None = None,
    prior_enabled: bool = True,
    manifest_path: str | os.PathLike[str] | None = None,
    report_path: str | os.PathLike[str] | None = None,
    progress: bool = False,
    **kwargs: Any,
) -> dict:
    """Собрать шард Q (decay): сужёный W + код с тестами, дедуп против W/C.

    Опорный near-dup индекс обязателен по умолчанию (``prior_enabled``): без
    опоры дедуп против W/C не выполняется, и это не маскируется подменой —
    отсутствие шардов W/C останавливает прогон (``FileNotFoundError``). Явное
    ``prior_enabled=False`` (CLI ``--no-prior-dedup``) отключает шаг и видно в
    отчёте как ``prior.enabled: false``.
    """
    root = out_dir or os.path.join(common.DATASET_ROOT, SHARD_Q)
    manifest = manifest_path or os.path.join(root, "manifest-q.json")
    report = report_path or os.path.join(root, "report-q.json")
    web = dict(web_spec or DEFAULT_SOURCE)
    code = dict(code_spec or {"kind": "stack", "name": "codeparrot-clean"})
    if code.get("kind") == "stack" and code.get("name") not in stack.SOURCES:
        raise ValueError(f"неизвестный источник кода: {code.get('name')!r} (есть: {sorted(stack.SOURCES)})")

    source_spec: dict[str, Any] = {
        "kind": "q-mix",
        "web": web,
        "code": code,
        "code_share": code_share,
        "web_yield": web_yield,
        "code_yield": code_yield,
        "languages": list(languages),
        "target_tokens": target_tokens,
    }

    prior_index: PriorNearDupIndex | None = None
    prior_state: dict[str, Any] = {"enabled": False}
    if prior_enabled:
        prior_index = PriorNearDupIndex(threshold=prior_threshold, max_records=prior_max_records)
        shards = discover_prior_shards(prior_dirs)
        if not shards:
            raise FileNotFoundError(
                f"опорные шарды W/C не найдены в {list(prior_dirs or [common.DATASET_ROOT])}: "
                f"дедуп Q против W/C без них не выполняется. Укажите --prior-dir либо "
                f"отключите шаг явно (--no-prior-dedup)"
            )
        prior_state = load_prior_reference(prior_index, shards, progress=progress)
        prior_state["enabled"] = True
        prior_state["dirs"] = [str(item) for item in (prior_dirs or [common.DATASET_ROOT])]

    source_name = code.get("name") if code.get("kind") == "stack" else "codeparrot-clean"
    normalize = QMixDocuments(
        web=QualityDocuments(
            min_int_score=min_int_score,
            min_chars=min_chars,
            max_chars=max_chars,
            source_name=str(web.get("repo") or web.get("glob") or DEFAULT_REPO),
        ),
        code=TestCodeDocuments(languages=tuple(languages), source_name=str(source_name)),
        prior=prior_index,
    )

    loaded = common.Manifest.load(manifest, SHARD_Q)
    _assert_same_stream(loaded, source_spec, restart=bool(kwargs.get("restart")))

    run_state: dict[str, Any] = {}
    rules = _q_rules(normalize, min_int_score, min_chars, max_chars, languages, prior_state)
    registry = {
        **common.SOURCE_ITERATORS,
        **stack.STACK_ITERATORS,
        "q-mix": (
            lambda **names: iter_q_mix(state=run_state, **names),
            ("web", "code", "code_share", "web_yield", "code_yield", "languages"),
        ),
    }
    result = common.run_shard(
        shard=SHARD_Q,
        out_dir=root,
        target_tokens=target_tokens,
        source_spec=source_spec,
        normalize=normalize,
        manifest_path=manifest,
        report_path=report,
        source_iterators=registry,
        manifest_extra={
            "mix": {
                "code_share_target": code_share,
                "web_yield": web_yield,
                "code_yield": code_yield,
                "target_tokens": target_tokens,
            },
            "rules": rules,
        },
        progress=progress,
        **kwargs,
    )
    result["rules"] = rules
    mix = normalize.mix_state()
    mix["code_share_target"] = code_share
    mix["code_share_deviation_pp"] = round((mix["code_share"] - code_share) * 100, 2)
    mix["source_exhausted"] = dict(run_state.get("exhausted") or {})
    mix["pulls"] = dict((run_state.get("governor") or {}).get("pulls") or {})
    mix["source_chars"] = {
        key: (run_state.get("governor") or {}).get(f"{key}_chars", 0)
        for key in (WEB_COMPONENT, CODE_COMPONENT)
    }
    # Рекомендация — только когда микс решался потоком, а не исчерпанием
    # источника: исчерпание кода делает любую «настройку урожайности» ложной.
    exhausted = mix["source_exhausted"]
    mix["recommended_code_yield"] = (
        None if exhausted.get(CODE_COMPONENT) else _recommended_code_yield(mix, web_yield)
    )
    mix["recommendation_note"] = (
        "источник кода исчерпан раньше цели: доля кода ниже целевой не из-за "
        "урожайности — рекомендация не выдаётся"
        if exhausted.get(CODE_COMPONENT)
        else "при отклонении доли от цели боевой прогон запускается с этой code_yield"
    )
    result["mix"] = mix
    result["prior"] = dict(prior_state)
    if prior_index is not None:
        result["prior"].update(prior_index.state())
    result["report"] = str(common.write_report(report, result))
    return result


def _recommended_code_yield(mix: dict, web_yield: float) -> float | None:
    """``code_yield``, при которой доля кода вышла бы на цель.

    Governor выравнивает долю по ОЖИДАЕМЫМ токенам (потянутые символы ×
    объявленная урожайность), а факт даёт фактические урожайности. Отсюда
    условие точной доли: отношение объявленных урожайностей равно отношению
    фактических — рекомендация держит ``web_yield`` и подстраивает ``code_yield``.
    """
    measured = mix.get("measured_yield") or {}
    web_real = measured.get(WEB_COMPONENT, 0.0)
    code_real = measured.get(CODE_COMPONENT, 0.0)
    if web_real <= 0 or code_real <= 0:
        return None
    return round(web_yield * code_real / web_real, 6)


def _q_rules(
    normalize: QMixDocuments,
    min_int_score: int | None,
    min_chars: int | None,
    max_chars: int | None,
    languages: Sequence[str],
    prior_state: dict,
) -> dict:
    """Правила отбора Q — в отчёт и манифест, как ``rules`` у шарда C."""
    return {
        "components": {
            WEB_COMPONENT: {
                "source": "FineWeb-Edu (тот же источник, что шард W)",
                "min_int_score": min_int_score,
                "min_chars": min_chars,
                "max_chars": max_chars,
            },
            CODE_COMPONENT: {
                "source": stack.SOURCES.get(normalize.code.source_name).repo
                if normalize.code.source_name in stack.SOURCES
                else normalize.code.source_name,
                "languages": list(languages),
                "test_markers": list(TEST_MARKERS),
                "min_file_bytes": normalize.code.min_bytes,
                "max_file_bytes": normalize.code.max_bytes,
                "license_filter": "preselected"
                if normalize.code.license_preselected
                else "whitelist",
            },
        },
        "prior_dedup": {
            "enabled": bool(prior_state.get("enabled")),
            "threshold": prior_state.get("threshold"),
            "reference_records": prior_state.get("reference_records", 0),
            "max_records": prior_state.get("max_records"),
        },
    }
