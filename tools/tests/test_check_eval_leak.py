"""C-043 / AD-12 — страж чистоты eval-набора (``tools/check_eval_leak.py``).

Сценарии (спека ``docs/specs/EVAL-HYGIENE.delta.md`` §5.2):

* чистая синтетика (дизъюнктные тексты) — зелёная;
* мутант «пересечение подстрокой» — красный;
* мутант «пересечение 12-граммовым окном» — красный;
* ``fail-closed`` — нечитаемый вход даёт FAIL (exit 1), не пропуск;
* нет файлов/источников — exit 2 «не могу проверить»;
* токенный источник (``tokens/*.bin``) — окна по id, eval кодируется токенизатором.

Фикстуры синтетические (``tmp_path``); реальные ``~/gb10-shared`` и eval-наборы
не трогаются — их ещё нет в репозитории (PENDING-EVIDENCE в AD-12).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1]
CASE_DIR = TOOLS_DIR.parent
for _path in (str(CASE_DIR), str(TOOLS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import check_eval_leak as leak  # noqa: E402


def _write_jsonl(path: Path, texts: list[str]) -> None:
    path.write_text(
        "".join(json.dumps({"text": text}, ensure_ascii=False) + "\n" for text in texts),
        encoding="utf-8",
    )


def _words(prefix: str, count: int) -> list[str]:
    return [f"{prefix}{index}" for index in range(count)]


# --------------------------------------------------------------------------- #
# Чисто / мутанты (текстовые источники)
# --------------------------------------------------------------------------- #


def test_clean_disjoint_is_green(tmp_path: Path):
    """Дизъюнктные тексты: exit 0, отчёт с нулевым пересечением по каждому источнику."""
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, [" ".join(_words("src", 300))])
    eval_file = tmp_path / "eval.jsonl"
    _write_jsonl(eval_file, [" ".join(_words("eval", 300)), " ".join(_words("held", 50))])

    code, report = leak.run_check([eval_file], [source])
    assert code == leak.EXIT_CLEAN
    assert report["verdict"] == "clean"
    assert report["sources"][0]["docs_overlap"] is False
    assert report["sources"][0]["ngram_overlap"] == 0.0


def test_substring_leak_is_red(tmp_path: Path):
    """Мутант: eval-документ = кусок источника — красный по подстроке."""
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, [" ".join(_words("src", 120))])
    eval_file = tmp_path / "eval.jsonl"
    _write_jsonl(eval_file, [" ".join(_words("src", 60))])

    code, report = leak.run_check([eval_file], [source])
    assert code == leak.EXIT_LEAK
    assert report["verdict"] == "leak"
    assert report["sources"][0]["docs_overlap"] is True
    assert report["sources"][0]["examples"], "отчёт обязан перечислять примеры"


def test_ngram_window_leak_is_red(tmp_path: Path):
    """Мутант: общее 12-граммовое окно, но eval-документ — не подстрока источника."""
    shared = " ".join(_words("shared", 12))
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, [" ".join(_words("other", 30)) + " " + shared])
    eval_file = tmp_path / "eval.jsonl"
    _write_jsonl(eval_file, [" ".join(_words("uniq", 20)) + " " + shared])

    code, report = leak.run_check([eval_file], [source])
    assert code == leak.EXIT_LEAK
    assert report["sources"][0]["docs_overlap"] is False, "подстроки нет — окно не подстрока"
    assert report["sources"][0]["ngram_overlap"] > 0.0


def test_report_shape_has_per_source_fields(tmp_path: Path):
    """Поля отчёта: per-source {docs_overlap, ngram_overlap, examples}."""
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, [" ".join(_words("src", 40))])
    eval_file = tmp_path / "eval.jsonl"
    _write_jsonl(eval_file, [" ".join(_words("eval", 40))])

    _, report = leak.run_check([eval_file], [source])
    source_report = report["sources"][0]
    for field in ("docs_overlap", "ngram_overlap", "examples"):
        assert field in source_report, f"нет поля {field}"


# --------------------------------------------------------------------------- #
# Границы ошибок
# --------------------------------------------------------------------------- #


def test_unreadable_input_fails_closed(tmp_path: Path):
    """Нечитаемый вход (существует, но не разбирается) — FAIL, не пропуск."""
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, [" ".join(_words("src", 40))])
    broken = tmp_path / "broken.jsonl"
    broken.write_bytes(b"\xff\xfe not json\n")

    code, report = leak.run_check([broken], [source])
    assert code == leak.EXIT_LEAK
    assert report["verdict"] == "cannot-verify"
    assert "fail-closed" in report["reason"]


def test_eval_without_text_field_fails_closed(tmp_path: Path):
    """Eval-запись без поля text — контракт нарушен, FAIL (exit 1)."""
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, [" ".join(_words("src", 40))])
    bad_eval = tmp_path / "eval.jsonl"
    bad_eval.write_text(json.dumps({"prompt": "нет text"}) + "\n", encoding="utf-8")

    code, report = leak.run_check([bad_eval], [source])
    assert code == leak.EXIT_LEAK
    assert report["verdict"] == "cannot-verify"


def test_missing_path_is_cannot_check(tmp_path: Path):
    """Объявленный путь отсутствует — «нет файлов», exit 2 (не зелёный)."""
    eval_file = tmp_path / "eval.jsonl"
    _write_jsonl(eval_file, [" ".join(_words("eval", 40))])
    code, report = leak.run_check([eval_file], [tmp_path / "нет-такого.jsonl"])
    assert code == leak.EXIT_CANNOT
    assert report["verdict"] == "cannot-check"


def test_no_sources_is_cannot_check(tmp_path: Path):
    """Источник не передан — проверять нечем: exit 2."""
    eval_file = tmp_path / "eval.jsonl"
    _write_jsonl(eval_file, [" ".join(_words("eval", 40))])
    code, report = leak.run_check([eval_file], [])
    assert code == leak.EXIT_CANNOT
    assert report["verdict"] == "cannot-check"


# --------------------------------------------------------------------------- #
# Токенный источник (tokens/*.bin)
# --------------------------------------------------------------------------- #


def _build_token_source(root: Path, words: list[str]) -> Path:
    """Мини-источник tokens/: WordLevel-токенизатор + манифест + .bin (uint32 LE)."""
    import array

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    vocab = {"<unk>": 0}
    for word in words:
        if word not in vocab:
            vocab[word] = len(vocab)
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer_path = root / "tokenizer.model"
    tokenizer.save(str(tokenizer_path))

    stream = " ".join(words)
    ids = [int(token) for token in tokenizer.encode(stream).ids]
    bin_path = root / "stream.bin"
    buffer = array.array("I", ids)
    if sys.byteorder == "big":  # pragma: no cover — стенд little-endian
        buffer.byteswap()
    bin_path.write_bytes(buffer.tobytes())

    manifest = {
        "version": leak.TOKENS_SCHEMA,
        "kind": "packed-uint32",
        "shard": "W",
        "seq_len": len(ids),
        "dtype": "uint32",
        "tokenizer": {"path": str(tokenizer_path), "tokenizer_hash": "0" * 64},
        "shards": [{"file": bin_path.name, "records": 1, "tokens": len(ids)}],
    }
    manifest_path = root / "manifest-w.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_token_source_detects_window_leak(tmp_path: Path):
    """Токенный источник: eval, целиком лежащий в .bin, ловится окнами по id."""
    words = _words("w", 60)
    manifest = _build_token_source(tmp_path, words)
    eval_file = tmp_path / "eval.jsonl"
    _write_jsonl(eval_file, [" ".join(words[10:40])])

    code, report = leak.run_check([eval_file], [manifest])
    assert code == leak.EXIT_LEAK
    assert report["sources"][0]["kind"] == "tokens"
    assert report["sources"][0]["docs_overlap"] is None, "для токенов подстрока неприменима"
    assert report["sources"][0]["ngram_overlap"] > 0.0


def test_token_source_clean_is_green(tmp_path: Path):
    """Токенный источник: обратный порядок слов не даёт общих 12-граммовых окон."""
    words = _words("w", 60)
    manifest = _build_token_source(tmp_path, words)
    eval_file = tmp_path / "eval.jsonl"
    _write_jsonl(eval_file, [" ".join(reversed(words[10:60]))])

    code, report = leak.run_check([eval_file], [manifest])
    assert code == leak.EXIT_CLEAN
    assert report["sources"][0]["ngram_overlap"] == 0.0


# --------------------------------------------------------------------------- #
# CLI и selftest
# --------------------------------------------------------------------------- #


def test_selftest_is_green():
    """C-043: ``--selftest`` зелёный (мутанты краснеют, чистая пара — нет)."""
    assert leak.main(["--selftest"]) == leak.EXIT_CLEAN


def test_cli_writes_report_and_exit_codes(tmp_path: Path, capsys):
    """CLI: exit-код повторяет вердикт, отчёт пишется в ``--json``."""
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, [" ".join(_words("src", 60))])
    eval_file = tmp_path / "eval.jsonl"
    _write_jsonl(eval_file, [" ".join(_words("src", 60))])
    out = tmp_path / "leak.json"

    code = leak.main(
        ["--eval", str(eval_file), "--source", str(source), "--json", str(out), "--quiet"]
    )
    assert code == leak.EXIT_LEAK
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["exit_code"] == leak.EXIT_LEAK
    assert report["schema"] == leak.REPORT_SCHEMA


def test_ngram_length_is_configurable(tmp_path: Path):
    """Окно настраивается: при --ngram 5 общий 5-грамм ловится, при 12 — нет."""
    shared = " ".join(_words("shared", 5))
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, [" ".join(_words("other", 30)) + " " + shared])
    eval_file = tmp_path / "eval.jsonl"
    _write_jsonl(eval_file, [" ".join(_words("uniq", 20)) + " " + shared])

    assert leak.run_check([eval_file], [source], ngram=5)[0] == leak.EXIT_LEAK
    assert leak.run_check([eval_file], [source], ngram=12)[0] == leak.EXIT_CLEAN
