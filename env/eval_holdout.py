"""Сборщик holdout-набора для PPL претрейн-чекпойнта (SFT-STAGE §1.1 п.2).

Holdout — детерминированный **срез микса**, не входящий в претрейн/SFT-источники
(правило AD-12/EVAL-HYGIENE §2: до первого вердикта по eval-числу набор обязан
иметь нулевое пересечение с обучающими источниками — урок Лагуны «PPL ×0.99 —
артефакт утечки»).

Прибор:

* ``--sources`` — текстовые источники (файлы/каталоги: jsonl/json/md/txt);
  точный состав пиннит прогон после уточнения run-манифеста;
* ``--share``/``--min-texts``/``--seed`` — детерминированная выборка:
  ``n = min(total, max(min_texts, ceil(share·total)))`` от стабильно
  упорядоченного списка документов, затем ``random.Random(seed)``;
* ``--exclude`` — источники, с которыми фильтруется 12-граммовое пересечение
  (текстовые — по словам, токенные ``tokens/*.bin`` — по id через токенизатор
  манифеста).  Механика — та же, что у стража ``tools/check_eval_leak.py``
  (импорт оттуда): пересечение хотя бы одного 12-граммового окна отбрасывает
  документ;
* выход — jsonl ``{text, source_path}``, стабильно отсортированный.

Пиннинг в stdout: ``{n_texts, sources, seed, sha256}`` — sha256 записанного
файла (предмет замера).  Детерминизм: один seed → байт-идентичный файл.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Iterator, Sequence

#: Корень репозитория — ``tools.check_eval_leak`` импортируется по пути репо-корня
#: (та же механика 12-граммовых окон, что у стража C-043/AD-12).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

#: 12-граммовые окна — правило AD-12.
DEFAULT_NGRAM = 12
#: Дефолтная доля среза микса.
DEFAULT_SHARE = 0.02
#: Дефолтный пол выборки (текстов).
DEFAULT_MIN_TEXTS = 200
#: Дефолтный выход.
DEFAULT_OUT = "eval_holdout.jsonl"

TEXT_SUFFIXES = (".jsonl", ".jsonl.zst", ".json", ".md", ".txt")


class HoldoutError(ValueError):
    """Вход непригоден: fail-closed, а не молчаливый пропуск."""


def _load_leak_helpers():
    """Хелперы стража AD-12 (импорт оттуда — механика окон, не копия)."""
    from tools.check_eval_leak import (  # type: ignore import-not-found
        InputError,
        iter_bin_ids,
        iter_text_docs,
        load_token_source,
        normalize_text,
        word_ngrams,
    )
    from tools import check_eval_leak as leak

    return {
        "InputError": InputError,
        "iter_bin_ids": iter_bin_ids,
        "iter_text_docs": iter_text_docs,
        "load_token_source": load_token_source,
        "normalize_text": normalize_text,
        "word_ngrams": word_ngrams,
        "TOKENS_SCHEMA": leak.TOKENS_SCHEMA,
    }


class _TextSource:
    def __init__(self, path: Path, grams: set[tuple[str, ...]]) -> None:
        self.path = path
        self.grams = grams


class _TokenSource:
    def __init__(self, path: Path, source: Any, ngram: int) -> None:
        self.path = path
        self.source = source
        self.ngram = ngram


def _is_token_manifest(path: Path) -> bool:
    if path.is_dir():
        return True
    if path.name.lower().endswith(".json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return isinstance(value, dict) and value.get("version") == "axiom-pretrain-tokens/1"
    return False


def iter_source_docs(path: Path, helpers: dict) -> Iterator[tuple[str, str]]:
    """Документы источника (файл или каталог), потоково и стабильно.

    Каталог: рекурсивный отсортированный обход текстовых файлов.  Возвращает
    ``(source_path, text)`` с путём от корня репозитория, если применимо.
    """
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.is_file())
    elif path.is_file():
        files = [path]
    else:
        raise HoldoutError(f"источник не существует: {path}")
    for file in files:
        if not file.name.lower().endswith(TEXT_SUFFIXES):
            continue
        for text in helpers["iter_text_docs"](file):
            if str(text).strip():
                yield (str(file), text)
    return


def collect_candidates(
    sources: Sequence[Path], helpers: dict
) -> list[tuple[str, str]]:
    """Все документы источников в стабильном порядке (без сэмплинга)."""
    items: list[tuple[str, str]] = []
    for source in sources:
        if _is_token_manifest(source):
            raise HoldoutError(
                f"{source}: токенный источник не годится как кандидатная база "
                "holdout — нужны текстовые источники (jsonl/json/md/txt)"
            )
        before = len(items)
        for item in iter_source_docs(source, helpers):
            items.append(item)
        if len(items) == before:
            raise HoldoutError(f"{source}: текстовых документов не найдено")
    return items


def sample_candidates(
    items: Sequence[tuple[str, str]],
    *,
    share: float,
    min_texts: int,
    seed: int,
) -> list[tuple[str, str]]:
    """Детерминированная выборка: seed → одинаковый набор (порядок задан входом)."""
    total = len(items)
    if total == 0:
        raise HoldoutError("нет документов для сэмплинга")
    target = min(total, max(int(min_texts), int(math.ceil(float(share) * total))))
    if target >= total:
        return list(items)
    rng = random.Random(int(seed))
    chosen = rng.sample(range(total), target)
    return [items[i] for i in sorted(chosen)]


def _build_text_exclude(path: Path, helpers: dict, ngram: int) -> _TextSource:
    grams: set[tuple[str, ...]] = set()
    for text in helpers["iter_text_docs"](path):
        norm = helpers["normalize_text"](text)
        grams.update(helpers["word_ngrams"](norm.split(), ngram))
    return _TextSource(path, grams)


def _build_token_exclude(path: Path, helpers: dict, ngram: int) -> _TokenSource:
    return _TokenSource(path, helpers["load_token_source"](path), ngram)


def load_excludes(paths: Sequence[Path], helpers: dict, ngram: int):
    """Разобрать exclude-источники: текстовые и токенные (fail-closed)."""
    text_sources: list[_TextSource] = []
    token_sources: list[_TokenSource] = []
    for path in paths:
        if not path.exists():
            raise HoldoutError(f"exclude-источник не существует: {path}")
        if _is_token_manifest(path):
            token_sources.append(_build_token_exclude(path, helpers, ngram))
        else:
            if not (path.is_dir() or path.name.lower().endswith(TEXT_SUFFIXES)):
                raise HoldoutError(f"exclude-источник не поддержан: {path}")
            if path.is_dir():
                for file in sorted(p for p in path.rglob("*") if p.is_file()):
                    if file.name.lower().endswith(TEXT_SUFFIXES):
                        text_sources.append(_build_text_exclude(file, helpers, ngram))
            else:
                text_sources.append(_build_text_exclude(path, helpers, ngram))
    return text_sources, token_sources


def filter_leaked(
    items: Sequence[tuple[str, str]],
    text_sources: Sequence[_TextSource],
    token_sources: Sequence[_TokenSource],
    helpers: dict,
    ngram: int,
) -> list[tuple[str, str]]:
    """Отбросить документы с 12-граммовым пересечением с exclude-источниками."""
    if not text_sources and not token_sources:
        return list(items)
    kept: list[tuple[str, str]] = []
    for source_path, text in items:
        norm = helpers["normalize_text"](text)
        grams = list(helpers["word_ngrams"](norm.split(), ngram))
        if any(gram in src.grams for gram in grams for src in text_sources):
            continue
        kept.append((source_path, text))

    # Токенные exclude: инверсия стража — набор кандидатных id-окон, затем один
    # проход по .bin помечает совпавшие кандидаты (память ~ документы, не корпус).
    for token_source in token_sources:
        reachable = list(kept)
        cand_grams: dict[tuple[int, ...], set[int]] = {}
        for index, (_path, text) in enumerate(reachable):
            ids = [int(t) for t in token_source.source.tokenizer.encode(text).ids]
            for start in range(max(0, len(ids) - ngram + 1)):
                gram = tuple(ids[start : start + ngram])
                cand_grams.setdefault(gram, set()).add(index)
        matched: set[int] = set()
        window: list[int] = []
        for bin_path in token_source.source.bin_paths:
            window = []
            for token in helpers["iter_bin_ids"](bin_path):
                window.append(int(token))
                if len(window) > ngram:
                    window.pop(0)
                if len(window) == ngram:
                    gram = tuple(window)
                    if gram in cand_grams:
                        matched.update(cand_grams[gram])
            window = []
        kept = [item for i, item in enumerate(reachable) if i not in matched]
    return kept


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_holdout(items: Sequence[tuple[str, str]], out: Path) -> str:
    """Записать jsonl стабильно отсортированным; вернуть sha256 файла."""
    ordered = sorted(items, key=lambda it: (it[0], _digest(it[1])))
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps({"text": text, "source_path": source}, ensure_ascii=False) + "\n"
        for source, text in ordered
    )
    out.write_text(payload, encoding="utf-8")
    return hashlib.sha256(out.read_bytes()).hexdigest()


def build_holdout(
    sources: Sequence[str | Path],
    out: str | Path,
    *,
    share: float = DEFAULT_SHARE,
    min_texts: int = DEFAULT_MIN_TEXTS,
    seed: int = 0,
    exclude: Sequence[str | Path] = (),
    ngram: int = DEFAULT_NGRAM,
) -> dict[str, Any]:
    """Собрать holdout и вернуть пиннинг-отчёт (stdout-JSON прибора)."""
    helpers = _load_leak_helpers()
    source_paths = [Path(s) for s in sources]
    if not source_paths:
        raise HoldoutError("--sources обязателен (хотя бы один источник)")
    candidates = collect_candidates(source_paths, helpers)
    sampled = sample_candidates(
        candidates, share=share, min_texts=min_texts, seed=int(seed)
    )
    text_sources, token_sources = load_excludes(
        [Path(e) for e in exclude], helpers, ngram
    )
    kept = filter_leaked(sampled, text_sources, token_sources, helpers, ngram)
    digest = write_holdout(kept, Path(out))
    return {
        "n_texts": len(kept),
        "sources": [str(p) for p in source_paths],
        "seed": int(seed),
        "sha256": digest,
        "out": str(out),
        "share": float(share),
        "min_texts": int(min_texts),
        "ngram": int(ngram),
        "n_candidates": len(candidates),
        "n_sampled": len(sampled),
        "n_dropped_leak": len(sampled) - len(kept),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Сборщик holdout-набора для PPL (SFT-STAGE §1.1)"
    )
    parser.add_argument(
        "--sources", required=True,
        help="текстовые источники через запятую (файлы/каталоги)",
    )
    parser.add_argument("--out", default=DEFAULT_OUT, help="выходной jsonl")
    parser.add_argument("--share", type=float, default=DEFAULT_SHARE, help="доля среза")
    parser.add_argument(
        "--min-texts", type=int, default=DEFAULT_MIN_TEXTS, help="пол выборки"
    )
    parser.add_argument("--seed", type=int, default=0, help="seed выборки")
    parser.add_argument(
        "--exclude", default="",
        help="источники утечки через запятую (текст или tokens-манифест)",
    )
    parser.add_argument("--ngram", type=int, default=DEFAULT_NGRAM, help="окно (AD-12: 12)")
    args = parser.parse_args(argv)

    sources = [s for s in (p.strip() for p in args.sources.split(",")) if s]
    exclude = [s for s in (p.strip() for p in args.exclude.split(",")) if s]
    try:
        report = build_holdout(
            sources,
            args.out,
            share=args.share,
            min_texts=args.min_texts,
            seed=args.seed,
            exclude=exclude,
            ngram=args.ngram,
        )
    except (HoldoutError, FileNotFoundError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
