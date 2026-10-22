#!/usr/bin/env python3
"""C-043 / AD-12 — страж чистоты eval-набора: нулевое пересечение с обучающими источниками.

Урок Лагуны (LAG-ADR-025/044): ``general_eval_v2`` оказался 200/200 в обучающем
источнике, «PPL ×0.99» был артефактом утечки.  Правило axiom: до первого
вердикта по eval-числу набор обязан иметь нулевое пересечение с каждым
кандидатным миксом и обучающим набором.  Проверки:

* **(а) точные документы** — нормализованная подстрока (lowercase, схлопывание
  пробелов): eval-документ встречается в тексте источника целиком;
* **(б) 12-граммовые окна** по словам (текстовые источники) или по токенным
  id (``tokens/*.bin``): доля окон eval-документа, встречающихся в источнике,
  > 0 — пересечение; в отчёт попадают примеры.

**Источники.** Текстовые — ``.jsonl`` / ``.json`` / ``.md`` / ``.txt``
(``.jsonl.zst`` тоже); токенные — каталог или манифест
``manifest-<s>.json`` схемы ``axiom-pretrain-tokens/1`` с ``.bin`` (uint32 LE).
Для токенного источника eval-текст кодируется токенизатором, объявленным в
манифесте (поле ``tokenizer.path``); недоступный токенизатор — отказ
(fail-closed), а не пропуск.

**Коды возврата** (контракт дельты):

* ``0`` — чисто: все проверки выполнены, пересечений нет;
* ``1`` — пересечение найдено **или** вход существует, но нечитаем/непригоден
  (fail-closed: «не смог прочитать» не превращается в «чисто»);
* ``2`` — проверить нельзя: не переданы eval- или source-файлы, либо
  объявленный путь отсутствует («нет файлов»).  Это не пропуск: гейт на
  ``command_succeeds`` краснеет и на нём.

Фильтр — против **всего источника** (не против обученного префикса), как
требует AD-12.  Прибор ничего не чинит: отчёт ``per-source {docs_overlap,
ngram_overlap, examples}`` + ``--json`` для карточки набора.  Реальные
eval-наборы в репозитории пока отсутствуют — поведенческий прогон по ним
остаётся PENDING-EVIDENCE; ``--selftest`` проверяет сам прибор на синтетике
в ``tmp`` (мутанты краснеют, чистая пара зелёная).

Запуск::

    python3 tools/check_eval_leak.py --selftest
    python3 tools/check_eval_leak.py --eval eval_holdout.jsonl \
        --source tokens/W --source sft_train.jsonl --json leak.json
"""

from __future__ import annotations

import argparse
import array
import io
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

#: Схема отчёта прибора.
REPORT_SCHEMA = "axiom-eval-leak/1"
#: Раскладка ``tokens/*.bin`` (tools/pretokenize.py): uint32 little-endian.
TOKENS_SCHEMA = "axiom-pretrain-tokens/1"
BIN_DTYPE_BYTES = 4
#: 12-граммовые окна — правило AD-12 (LAG-ADR-025/044).
DEFAULT_NGRAM = 12
#: Потолок примеров на источник, чтобы отчёт оставался читаемым.
DEFAULT_EXAMPLES = 5
#: Потолок длины eval-документа (символов) для «подстроки с переносом границы».
MAX_BOUNDARY_CHARS = 1 << 20

EXIT_CLEAN = 0
EXIT_LEAK = 1
EXIT_CANNOT = 2

#: Текстовые расширения источников (jsonl.zst — как суффикс имени).
TEXT_SUFFIXES = (".jsonl", ".jsonl.zst", ".json", ".md", ".txt")
#: Ключи, из которых текст берётся у текстового источника (порядок приоритета).
_TEXT_KEYS = ("text", "content", "document", "raw", "body", "doc")
#: Разделитель — не слово: окна не «протекают» через границу документов.
_SENTINEL = "\u0000"


class InputError(Exception):
    """Вход существует, но нечитаем/непригоден: fail-closed (exit 1)."""


# --------------------------------------------------------------------------- #
# Оценка (нормализация, окна)
# --------------------------------------------------------------------------- #


def normalize_text(text: Any) -> str:
    """Lowercase + схлопывание пробелов — база подстрочной проверки (AD-12)."""
    return " ".join(str(text).lower().split())


def word_ngrams(words: list[str], n: int) -> Iterator[tuple[str, ...]]:
    """Окна длины ``n`` по списку слов (по порядку)."""
    if n <= 0 or len(words) < n:
        return
    for start in range(len(words) - n + 1):
        yield tuple(words[start : start + n])


# --------------------------------------------------------------------------- #
# Загрузка текстовых источников
# --------------------------------------------------------------------------- #


def _docs_from_json_value(value: Any, path: Path) -> list[str]:
    """Документы из разобранного JSON (строка, список, словарь с текстом)."""
    docs: list[str] = []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        for item in value:
            docs.extend(_docs_from_json_value(item, path))
        return docs
    if isinstance(value, dict):
        for key in _TEXT_KEYS:
            if isinstance(value.get(key), str):
                return [value[key]]
        messages = value.get("messages")
        if isinstance(messages, list):
            parts = [
                str(message.get("content") or "")
                for message in messages
                if isinstance(message, dict)
            ]
            if any(parts):
                return ["\n".join(parts)]
        # Плоский словарь строк — как документ «ключ: значение».
        if value and all(isinstance(item, str) for item in value.values()):
            return ["\n".join(f"{key}: {item}" for key, item in value.items())]
    raise InputError(f"{path}: не найден текст документа (нет поля text/content/…)")


def _read_maybe_zst(path: Path) -> str:
    """Прочитать файл, прозрачно распаковав zstd (если расширение .zst)."""
    if path.name.endswith(".zst"):
        try:
            import zstandard as zstd
        except ImportError as exc:  # pragma: no cover — окружение без zstandard
            raise InputError(f"{path}: нужен zstandard для .zst: {exc}") from exc
        try:
            with path.open("rb") as handle:
                reader = zstd.ZstdDecompressor().stream_reader(handle)
                with io.TextIOWrapper(reader, encoding="utf-8") as text:
                    return text.read()
        except Exception as exc:  # noqa: BLE001 — любой сбой чтения = fail-closed
            raise InputError(f"{path}: .zst не читается ({type(exc).__name__}): {exc}") from exc
    try:
        return path.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeDecodeError) as exc:
        raise InputError(f"{path}: текст не читается ({type(exc).__name__}): {exc}") from exc


def _iter_text_lines(path: Path) -> Iterator[str]:
    """Строки текстового файла, потоково; ``.zst`` распаковывается на лету."""
    if path.name.endswith(".zst"):
        try:
            import zstandard as zstd
        except ImportError as exc:  # pragma: no cover — окружение без zstandard
            raise InputError(f"{path}: нужен zstandard для .zst: {exc}") from exc
        try:
            with path.open("rb") as handle:
                reader = zstd.ZstdDecompressor().stream_reader(handle)
                with io.TextIOWrapper(reader, encoding="utf-8") as text:
                    yield from text
        except Exception as exc:  # noqa: BLE001 — любой сбой чтения = fail-closed
            raise InputError(f"{path}: .zst не читается ({type(exc).__name__}): {exc}") from exc
        return
    try:
        with path.open("r", encoding="utf-8") as handle:
            yield from handle
    except (OSError, UnicodeDecodeError) as exc:
        raise InputError(f"{path}: текст не читается ({type(exc).__name__}): {exc}") from exc


def iter_text_docs(path: Path) -> Iterator[str]:
    """Документы текстового источника (jsonl/json/md/txt), потоково для jsonl."""
    name = path.name.lower()
    if name.endswith(".jsonl") or name.endswith(".jsonl.zst"):
        for lineno, line in enumerate(_iter_text_lines(path), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InputError(f"{path}:{lineno}: строка не разбирается как JSON: {exc}") from exc
            yield from _docs_from_json_value(record, path)
        return
    if path.suffix.lower() == ".json":
        try:
            value = json.loads(_read_maybe_zst(path))
        except json.JSONDecodeError as exc:
            raise InputError(f"{path}: JSON не разбирается: {exc}") from exc
        yield from _docs_from_json_value(value, path)
        return
    # md/txt — весь файл один документ.
    yield _read_maybe_zst(path)


def load_eval_docs(path: Path) -> list[str]:
    """Eval-документы: jsonl/json с обязательным полем ``text`` (спека AD-12)."""
    docs: list[str] = []
    name = path.name.lower()
    if name.endswith(".jsonl") or name.endswith(".jsonl.zst"):
        for lineno, line in enumerate(_iter_text_lines(path), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InputError(f"{path}:{lineno}: строка не разбирается как JSON: {exc}") from exc
            if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                raise InputError(f"{path}:{lineno}: нет строкового поля text (контракт eval)")
            if record["text"].strip():
                docs.append(record["text"])
        return docs
    if path.suffix.lower() == ".json":
        try:
            value = json.loads(_read_maybe_zst(path))
        except json.JSONDecodeError as exc:
            raise InputError(f"{path}: JSON не разбирается: {exc}") from exc
        records = value if isinstance(value, list) else [value]
        if not isinstance(value, list) and not isinstance(value, dict):
            raise InputError(f"{path}: ожидался список записей с полем text")
        if isinstance(value, dict) and "text" in value:
            records = [value]
        for index, record in enumerate(records):
            if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                raise InputError(f"{path}: запись {index}: нет строкового поля text")
            if record["text"].strip():
                docs.append(record["text"])
        return docs
    raise InputError(f"{path}: eval-набор ожидается .jsonl/.json с полем text")


# --------------------------------------------------------------------------- #
# Загрузка токенного источника (tokens/*.bin)
# --------------------------------------------------------------------------- #


@dataclass
class TokenSource:
    """Токенный источник: манифест tokens/, его ``.bin`` и токенизатор."""

    manifest_path: Path
    manifest: dict[str, Any]
    bin_paths: list[Path]
    tokenizer: Any
    seq_len: int
    pad_id: int | None


def _load_tokenizer(manifest: dict[str, Any], manifest_path: Path) -> Any:
    """Токенизатор из поля ``tokenizer.path`` манифеста; нет — fail-closed."""
    info = manifest.get("tokenizer") or {}
    raw = info.get("path") if isinstance(info, dict) else None
    if not raw:
        raise InputError(
            f"{manifest_path}: манифест tokens не объявляет tokenizer.path — "
            "eval-текст нечем закодировать, проверка невозможна (fail-closed)"
        )
    candidates = [Path(str(raw))]
    if not candidates[0].is_absolute():
        candidates.append(manifest_path.parent / candidates[0])
    for path in candidates:
        if path.is_file():
            try:
                from tokenizers import Tokenizer
            except ImportError as exc:  # pragma: no cover — окружение без библиотеки
                raise InputError(
                    f"{manifest_path}: нужен tokenizers для токенного источника: {exc}"
                ) from exc
            try:
                return Tokenizer.from_file(str(path))
            except Exception as exc:  # noqa: BLE001 — нечитаемый артефакт = fail-closed
                raise InputError(
                    f"{path}: токенизатор не читается ({type(exc).__name__}): {exc}"
                ) from exc
    raise InputError(
        f"{manifest_path}: токенизатор {raw} не найден — проверка невозможна (fail-closed)"
    )


def load_token_source(path: Path) -> TokenSource:
    """Токенный источник: манифест + ``.bin`` + токенизатор манифеста."""
    manifest_path = path
    if path.is_dir():
        candidates = sorted(path.glob("manifest-*.json"))
        if not candidates:
            raise InputError(f"{path}: в каталоге нет манифеста tokens (manifest-*.json)")
        manifest_path = candidates[0]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputError(f"{manifest_path}: манифест tokens не читается: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("version") != TOKENS_SCHEMA:
        raise InputError(
            f"{manifest_path}: схема манифеста {manifest.get('version')!r} != {TOKENS_SCHEMA!r}"
        )
    seq_len = int(manifest.get("seq_len") or 0)
    shards = manifest.get("shards") or []
    if not shards:
        raise InputError(f"{manifest_path}: манифест без шардов .bin")
    bin_paths: list[Path] = []
    for entry in shards:
        candidate = manifest_path.parent / str(entry.get("file") or "")
        if not candidate.is_file():
            raise InputError(f"{manifest_path}: шард отсутствует: {candidate}")
        bin_paths.append(candidate)
    return TokenSource(
        manifest_path=manifest_path,
        manifest=manifest,
        bin_paths=bin_paths,
        tokenizer=_load_tokenizer(manifest, manifest_path),
        seq_len=seq_len,
        pad_id=(int(manifest["pad_id"]) if isinstance(manifest.get("pad_id"), int) else None),
    )


def iter_bin_ids(path: Path) -> Iterator[int]:
    """Токенные id из ``.bin`` (uint32 LE), потоково — без загрузки в память."""
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1 << 22)
                if not chunk:
                    return
                if len(chunk) % BIN_DTYPE_BYTES:
                    raise InputError(f"{path}: размер не кратен uint32 ({len(chunk)} байт в хвосте)")
                values = array.array("I")
                values.frombytes(chunk)
                if sys.byteorder == "big":
                    values.byteswap()
                yield from values
    except OSError as exc:
        raise InputError(f"{path}: .bin не читается: {exc}") from exc


# --------------------------------------------------------------------------- #
# Отчёт
# --------------------------------------------------------------------------- #


@dataclass
class SourceReport:
    """Пересечение eval-набора с одним источником."""

    path: str
    kind: str
    docs_overlap: bool | None
    ngram_overlap: float
    examples: list[dict[str, Any]] = field(default_factory=list)
    records: int = 0

    def as_json(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind,
            "docs_overlap": self.docs_overlap,
            "ngram_overlap": self.ngram_overlap,
            "examples": self.examples,
            "records": self.records,
        }

    @property
    def leaked(self) -> bool:
        return bool(self.docs_overlap) or self.ngram_overlap > 0.0


# --------------------------------------------------------------------------- #
# Ядро проверки
# --------------------------------------------------------------------------- #


def _classify_source(path: Path) -> str:
    """Тип источника: ``tokens`` | ``text``.  Каталог — токенный (манифест)."""
    if path.is_dir():
        return "tokens"
    name = path.name.lower()
    if name.endswith(TEXT_SUFFIXES):
        if path.suffix.lower() == ".json":
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise InputError(f"{path}: не читается как источник: {exc}") from exc
            if isinstance(value, dict) and value.get("version") == TOKENS_SCHEMA:
                return "tokens"
        return "text"
    raise InputError(
        f"{path}: расширение источника не поддержано (text: {', '.join(TEXT_SUFFIXES)}; "
        "tokens: каталог или манифест tokens)"
    )


def _check_text_source(
    path: Path,
    eval_norm: list[str],
    eval_word_grams: dict[tuple[str, ...], set[int]],
    ngram: int,
    examples_limit: int,
) -> SourceReport:
    """Стемминг текстового источника: подстрока + 12-граммовые окна."""
    report = SourceReport(path=str(path), kind="text", docs_overlap=False, ngram_overlap=0.0)
    max_eval_chars = min(
        MAX_BOUNDARY_CHARS, max((len(doc) for doc in eval_norm), default=0)
    )
    tail = ""
    matched: set[tuple[str, ...]] = set()
    window: list[str] = []
    doc_index = -1
    docs_seen = 0
    for raw in iter_text_docs(path):
        doc_index += 1
        docs_seen += 1
        norm = normalize_text(raw)
        # (а) точная подстрока: внутри документа и на стыке с предыдущим хвостом.
        boundary = (tail + " " + norm) if tail else norm
        for index, doc in enumerate(eval_norm):
            if not doc:
                continue
            if doc in boundary:
                report.docs_overlap = True
                if len(report.examples) < examples_limit:
                    report.examples.append(
                        {
                            "eval_doc": index,
                            "check": "substring",
                            "detail": "eval-документ найден в источнике дословно (нормализованно)",
                        }
                    )
        if max_eval_chars:
            keep = max_eval_chars - 1
            tail = boundary[-keep:] if keep > 0 else ""
        # (б) 12-граммовые окна по словам, с разрывом на границе документа.
        for word in (norm.split() if norm else []):
            window.append(word)
            if len(window) > ngram:
                window.pop(0)
            if len(window) == ngram:
                gram = tuple(window)
                if gram in eval_word_grams:
                    if gram not in matched and len(report.examples) < examples_limit:
                        report.examples.append(
                            {
                                "eval_doc": sorted(eval_word_grams[gram])[0],
                                "check": "ngram",
                                "detail": "окно слов: " + " ".join(gram)[:200],
                            }
                        )
                    matched.add(gram)
        window.clear()  # граница документа рвёт окно
    report.records = docs_seen
    if docs_seen == 0:
        raise InputError(f"{path}: источник не содержит документов")
    # Доля окон каждого eval-документа, встретившихся в источнике (максимум по документам).
    fractions = []
    for doc in eval_norm:
        grams = list(word_ngrams(doc.split(), ngram))
        if not grams:
            continue
        fractions.append(sum(1 for gram in grams if gram in matched) / len(grams))
    report.ngram_overlap = max(fractions, default=0.0)
    return report


def _check_token_source(
    path: Path,
    eval_texts: list[str],
    ngram: int,
    examples_limit: int,
) -> SourceReport:
    """Токенный источник: 12-граммовые окна по id (eval кодируется токенизатором)."""
    source = load_token_source(path)
    eval_grams: dict[tuple[int, ...], set[int]] = {}
    eval_windows: list[list[tuple[int, ...]]] = []
    for index, text in enumerate(eval_texts):
        ids = [int(token) for token in source.tokenizer.encode(text).ids]
        grams = [
            tuple(ids[start : start + ngram])
            for start in range(max(0, len(ids) - ngram + 1))
        ]
        eval_windows.append(grams)
        for gram in grams:
            eval_grams.setdefault(gram, set()).add(index)
    matched: set[tuple[int, ...]] = set()
    examples: list[dict[str, Any]] = []
    tokens_seen = 0
    for bin_path in source.bin_paths:
        window: list[int] = []
        for token in iter_bin_ids(bin_path):
            tokens_seen += 1
            window.append(int(token))
            if len(window) > ngram:
                window.pop(0)
            if len(window) == ngram:
                gram = tuple(window)
                if gram in eval_grams and gram not in matched:
                    matched.add(gram)
                    if len(examples) < examples_limit:
                        examples.append(
                            {
                                "eval_doc": sorted(eval_grams[gram])[0],
                                "check": "ngram",
                                "detail": "окно токенных id: " + " ".join(map(str, gram))[:200],
                            }
                        )
        window.clear()  # граница шарда .bin рвёт окно
    if tokens_seen == 0:
        raise InputError(f"{path}: токенный источник пуст")
    fractions = []
    for grams in eval_windows:
        if grams:
            fractions.append(sum(1 for gram in grams if gram in matched) / len(grams))
    # Подстрочная проверка документов в токенном источнике неприменима без
    # обратной детокенизации; полное совпадение документа ловится окнами (доля 1.0).
    return SourceReport(
        path=str(path),
        kind="tokens",
        docs_overlap=None,
        ngram_overlap=max(fractions, default=0.0),
        examples=examples,
        records=tokens_seen,
    )


def run_check(
    eval_paths: Iterable[str | Path],
    source_paths: Iterable[str | Path],
    *,
    ngram: int = DEFAULT_NGRAM,
    examples: int = DEFAULT_EXAMPLES,
) -> tuple[int, dict[str, Any]]:
    """Проверить пересечение eval-набора с источниками.  Возвращает (код, отчёт)."""
    eval_list = [Path(p) for p in eval_paths]
    source_list = [Path(p) for p in source_paths]

    missing = [str(p) for p in eval_list + source_list if not p.exists()]
    if not eval_list or not source_list:
        return EXIT_CANNOT, {
            "schema": REPORT_SCHEMA,
            "verdict": "cannot-check",
            "reason": "не переданы eval- и/или source-файлы — проверять нечем",
            "eval_files": [str(p) for p in eval_list],
            "sources": [],
            "ngram": int(ngram),
        }
    if missing:
        return EXIT_CANNOT, {
            "schema": REPORT_SCHEMA,
            "verdict": "cannot-check",
            "reason": "объявленные пути отсутствуют: " + ", ".join(missing),
            "eval_files": [str(p) for p in eval_list],
            "sources": [],
            "ngram": int(ngram),
        }

    try:
        eval_texts: list[str] = []
        for path in eval_list:
            eval_texts.extend(load_eval_docs(path))
        if not eval_texts:
            raise InputError("eval-набор пуст — проверять нечего (fail-closed)")
        eval_norm = [normalize_text(text) for text in eval_texts]
        eval_word_grams: dict[tuple[str, ...], set[int]] = {}
        for index, doc in enumerate(eval_norm):
            for gram in word_ngrams(doc.split(), ngram):
                eval_word_grams.setdefault(gram, set()).add(index)
        reports: list[SourceReport] = []
        for path in source_list:
            kind = _classify_source(path)
            if kind == "text":
                reports.append(
                    _check_text_source(path, eval_norm, eval_word_grams, ngram, examples)
                )
            else:
                reports.append(_check_token_source(path, eval_texts, ngram, examples))
    except InputError as exc:
        return EXIT_LEAK, {
            "schema": REPORT_SCHEMA,
            "verdict": "cannot-verify",
            "reason": f"fail-closed: {exc}",
            "eval_files": [str(p) for p in eval_list],
            "sources": [],
            "ngram": int(ngram),
        }

    leaked = any(report.leaked for report in reports)
    return (EXIT_LEAK if leaked else EXIT_CLEAN), {
        "schema": REPORT_SCHEMA,
        "verdict": "leak" if leaked else "clean",
        "eval_files": [str(p) for p in eval_list],
        "eval_docs": len(eval_texts),
        "ngram": int(ngram),
        "sources": [report.as_json() for report in reports],
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _emit(report: dict[str, Any], json_path: str | None, quiet: bool) -> None:
    if json_path:
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(json_path).write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if quiet:
        return
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C-043/AD-12: чистота eval-набора")
    parser.add_argument("--eval", dest="eval_paths", action="append", default=[],
                        help="eval-файл (jsonl/json с полем text); повторяется")
    parser.add_argument("--source", dest="source_paths", action="append", default=[],
                        help="обучающий источник (text jsonl/json/md/txt или tokens-манифест); повторяется")
    parser.add_argument("--ngram", type=int, default=DEFAULT_NGRAM, help="длина окна (AD-12: 12)")
    parser.add_argument("--examples", type=int, default=DEFAULT_EXAMPLES, help="примеров на источник")
    parser.add_argument("--json", dest="json_path", default=None, help="куда записать отчёт")
    parser.add_argument("--quiet", action="store_true", help="не печатать отчёт в stdout")
    parser.add_argument("--selftest", action="store_true",
                        help="синтетический selftest с мутантами (tmp; реальные данные не трогаются)")
    args = parser.parse_args(argv)

    if args.selftest:
        return run_selftest()
    exit_code, report = run_check(
        args.eval_paths, args.source_paths, ngram=args.ngram, examples=args.examples
    )
    report["exit_code"] = exit_code
    _emit(report, args.json_path, args.quiet)
    return exit_code


# --------------------------------------------------------------------------- #
# Selftest (C-043: мутанты краснеют, чистая пара зелёная)
# --------------------------------------------------------------------------- #


def _write_jsonl(path: Path, texts: list[str]) -> None:
    path.write_text(
        "".join(json.dumps({"text": text}, ensure_ascii=False) + "\n" for text in texts),
        encoding="utf-8",
    )


def _words(prefix: str, count: int) -> list[str]:
    return [f"{prefix}{index}" for index in range(count)]


def run_selftest() -> int:
    """Синтетика в ``tmp``: чистая пара зелёная, мутанты и fail-closed краснеют."""
    import tempfile

    checks: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory(prefix="eval-leak-selftest-") as tmp:
        root = Path(tmp)

        disjoint_source = root / "source_clean.jsonl"
        _write_jsonl(disjoint_source, [" ".join(_words("src", 200))])
        clean_eval = root / "eval_clean.jsonl"
        _write_jsonl(clean_eval, [" ".join(_words("eval", 200))])

        code_clean, report_clean = run_check([clean_eval], [disjoint_source])
        checks.append(("чистая пара → exit 0", code_clean == EXIT_CLEAN))
        checks.append(
            (
                "чистая пара → docs_overlap=false, ngram_overlap=0.0",
                report_clean["sources"][0]["docs_overlap"] is False
                and report_clean["sources"][0]["ngram_overlap"] == 0.0,
            )
        )

        # Мутант (а): eval-документ = кусок источника (подстрока).
        leak_sub = root / "eval_substring.jsonl"
        _write_jsonl(leak_sub, [" ".join(_words("src", 60))])
        code_sub, report_sub = run_check([leak_sub], [disjoint_source])
        checks.append(("мутант «подстрока» → exit 1", code_sub == EXIT_LEAK))
        checks.append(
            ("мутант «подстрока» → docs_overlap=true", report_sub["sources"][0]["docs_overlap"] is True)
        )

        # Мутант (б): общее 12-граммовое окно, но не подстрока целиком.
        shared = " ".join(_words("shared", 12))
        leak_ngram = root / "eval_ngram.jsonl"
        _write_jsonl(leak_ngram, [" ".join(_words("uniq", 20)) + " " + shared])
        ngram_source = root / "source_ngram.jsonl"
        _write_jsonl(ngram_source, [" ".join(_words("other", 30)) + " " + shared])
        code_ngram, report_ngram = run_check([leak_ngram], [ngram_source])
        checks.append(("мутант «12-граммовое окно» → exit 1", code_ngram == EXIT_LEAK))
        checks.append(
            (
                "мутант «12-граммовое окно» → docs_overlap=false, ngram_overlap>0",
                report_ngram["sources"][0]["docs_overlap"] is False
                and report_ngram["sources"][0]["ngram_overlap"] > 0.0,
            )
        )

        # fail-closed: битый вход существует, но нечитаем.
        broken = root / "broken.jsonl"
        broken.write_bytes(b"\xff\xfe not json at all\n")
        code_broken, _ = run_check([broken], [disjoint_source])
        checks.append(("нечитаемый вход → exit 1 (fail-closed)", code_broken == EXIT_LEAK))

        # нет источников → exit 2 (не могу проверить).
        code_none, _ = run_check([clean_eval], [])
        checks.append(("нет источников → exit 2", code_none == EXIT_CANNOT))

    ok = all(passed for _, passed in checks)
    for label, passed in checks:
        print(f"[selftest] {'PASS' if passed else 'FAIL'}: {label}")
    print(f"[selftest] {'PASS' if ok else 'FAIL'}: прибор краснеет на утечке и не краснеет на чистой паре")
    return EXIT_CLEAN if ok else EXIT_LEAK


if __name__ == "__main__":
    raise SystemExit(main())
