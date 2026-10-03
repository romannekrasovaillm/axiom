"""Корпусной BPE из ``<shard-root>/tokenizer`` для сырого пути претрейна (ADR-4).

Регресс на заблокированный пилот: ``--no-packed`` отказывал, потому что
``build_tokenizer_and_config`` собирал синтетическую заглушку ``net/tokenizer.py``
(хеш ``9f8d0309…``), а ``net/config.json`` пиннит корпусной BPE ``500f8023…``.
Фикс — тот же токенизатор, которым размечены ``tokens/*.bin``
(``tools/bpe_train.py`` + ``tokenizers``), со сверкой хеша файла с пином.

Сценарии:

* **T-ov1** — ``tokenizer.model`` найден: загружается корпусной BPE, кодирует
  тем же словарём, что ``tokenizers.Tokenizer.from_file``, и покрывает id;
* **T-ov2** — хеш не совпал с пином: **блокирующий** отказ (fail-closed), в
  журнале — причина; тихой подмены на заглушку нет;
* **T-ov3** — файла нет: прежнее поведение — скелетная заглушка;
* **T-ov4** — артефакт правлен после упаковки (манифест разошёлся): отказ.

Фикстуры синтетические (``tmp_path``), канонический диск ``~/gb10-shared`` не
трогается.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1]
CASE_DIR = TOOLS_DIR.parent

#: Пиннинг бэкенда ДО первого импорта jax (ADR-010) — тест остаётся файловым.
os.environ.setdefault("NET_JAX_BACKEND", "cpu")

for _path in (str(CASE_DIR), str(TOOLS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import bpe_train  # noqa: E402
import pretrain_run  # noqa: E402
import run_sft_smoke as sft_stage  # noqa: E402

#: Мини-словарь фикстуры: тренер BPE обязан дойти до него на синтетике.
SMALL_VOCAB = 512
#: Контракт раскладки корпусного BPE (``tools/bpe_train.py``): 4 спец + 256 байт.
N_SPECIALS, N_BYTES = 4, 256


# --------------------------------------------------------------------------- #
# Синтетический корпус (тот же контракт, что у ``tools/prep_pretrain``)
# --------------------------------------------------------------------------- #


#: Короткий повторяемый пул слов: у BPE есть частые пары для мерджей.
WORD_POOL = (
    "the model attention delta token context language network data train loss grad step "
    "batch scale layer head query key value cache window sparse dense pool block merge "
    "код данные сеть обучение пример тест функция возврат цикл условие класс модуль "
    "def return import class for while if else print range list dict set int str float"
).split()


def synthetic_docs(prefix: str, count: int, words: int, *, offset: int = 0) -> list[str]:
    """Документы из фиксированного пула слов (частые пары для мерджей)."""
    import random

    rng = random.Random(f"{prefix}:{offset}")
    return [" ".join(rng.choice(WORD_POOL) for _ in range(words)) for _ in range(count)]


def write_shard_set(root: Path, name: str, docs: list[str], *, shards: int = 1) -> Path:
    """Мини-шард-набор в контракте ``tools/prep_pretrain``."""
    import zstandard as zstd

    out_dir = root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk = max(1, (len(docs) + shards - 1) // shards)
    entries = []
    for index, start in enumerate(range(0, len(docs), chunk)):
        part = docs[start : start + chunk]
        payload = "".join(
            json.dumps({"text": text}, ensure_ascii=False) + "\n" for text in part
        ).encode("utf-8")
        path = out_dir / f"{name}-{index:05d}.jsonl.zst"
        path.write_bytes(zstd.ZstdCompressor(level=3).compress(payload))
        entries.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "approx_tokens": sum(max(1, len(text) // 4) for text in part),
                "records": len(part),
            }
        )
    manifest = {
        "version": "axiom-pretrain-l3/1",
        "shard": name,
        "shards": entries,
        "source_records": len(docs),
        "totals": {"shards": len(entries), "bytes": sum(e["bytes"] for e in entries)},
        "source": {"kind": "synthetic", "name": f"synthetic-{name}"},
        "codec": "zstd",
    }
    manifest_path = out_dir / f"manifest-{name.lower()}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return manifest_path


def build_mini_tokenizer(root: Path) -> Path:
    """Обучить мини-BPE в ``<root>/tokenizer`` (``tools/bpe_train.py``) и вернуть каталог."""
    out = root / pretrain_run.CORPUS_TOKENIZER_SUBDIR
    rc = bpe_train.main(
        [
            "--shard-root",
            str(root),
            "--out",
            str(out),
            "--vocab-size",
            str(SMALL_VOCAB),
            "--sample-mb",
            "1",
            "--round-trip-docs",
            "100",
        ]
    )
    assert rc == 0, "мини-BPE фикстуры не собрался"
    return out


@pytest.fixture(scope="module")
def corpus_with_tokenizer(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Мини-корпус W/C + корпусной BPE в его корне (собирается один раз)."""
    root = tmp_path_factory.mktemp("pretrain-tok-override") / "axiom-pretrain-l3"
    write_shard_set(root, "W", synthetic_docs("w", 400, 16), shards=2)
    write_shard_set(root, "C", synthetic_docs("c", 200, 16), shards=1)
    build_mini_tokenizer(root)
    return root


def _args(shard_root: Path) -> "pretrain_run.argparse.Namespace":
    return pretrain_run.parse_args(
        ["--shard-root", str(shard_root), "--run-ref", "tok-override-test"]
    )


def _fixture_hash(root: Path) -> str:
    return bpe_train.file_sha256(
        root / pretrain_run.CORPUS_TOKENIZER_SUBDIR / pretrain_run.CORPUS_TOKENIZER_FILE
    )


# --------------------------------------------------------------------------- #
# T-ov1: файл найден — грузится корпусной BPE
# --------------------------------------------------------------------------- #


def test_override_loads_corpus_tokenizer(corpus_with_tokenizer: Path, monkeypatch):
    """Найденный корпусной BPE грузится, сверяется с пином и кодирует тем же словарём."""
    expected = _fixture_hash(corpus_with_tokenizer)
    monkeypatch.setattr(sft_stage, "config_tokenizer_pin", lambda: expected)

    tokenizer, cfg, info = pretrain_run.build_tokenizer_and_config(
        _args(corpus_with_tokenizer)
    )

    assert info["hash"] == expected, "журнал обязан нести хеш файла-артефакта (AD-4)"
    assert tokenizer.vocab_hash() == expected
    assert info["matches_config_pin"] is True
    assert info["source"].startswith("corpus"), info["source"]
    assert "заглушка" not in info["source"]

    # Кодирование — тем же словарём, что у ``tokenizers`` (не заглушкой).
    from tokenizers import Tokenizer

    reference = Tokenizer.from_file(
        str(
            corpus_with_tokenizer
            / pretrain_run.CORPUS_TOKENIZER_SUBDIR
            / pretrain_run.CORPUS_TOKENIZER_FILE
        )
    )
    text = "the model delta context — данные и код"
    assert tokenizer.encode(text) == list(reference.encode(text).ids)

    # Словарь модели покрывает все фактические id токенизатора.
    assert int(cfg.vocab_size) == info["model_vocab_size"]
    assert int(cfg.vocab_size) >= info["vocab_size"]
    assert info["max_emitted_id"] < int(cfg.vocab_size)
    assert info["vocab_size"] == SMALL_VOCAB
    assert info["merges"] == SMALL_VOCAB - N_SPECIALS - N_BYTES


# --------------------------------------------------------------------------- #
# T-ov2: хеш не совпал с пином — блокирующий отказ
# --------------------------------------------------------------------------- #


def test_override_hash_mismatch_refuses(corpus_with_tokenizer: Path):
    """Несовпадение с пином ``net/config.json`` — отказ, а не тихая подмена."""
    # Пин в тесте реальный (500f8023…): хеш фикстурного BPE ему не равен.
    assert not sft_stage.tokenizer_hash_matches(
        _fixture_hash(corpus_with_tokenizer), sft_stage.config_tokenizer_pin()
    )
    with pytest.raises(pretrain_run.StageRefused) as excinfo:
        pretrain_run.build_tokenizer_and_config(_args(corpus_with_tokenizer))
    assert "пин" in str(excinfo.value)


def test_override_hash_mismatch_refuses_at_cli(tmp_path: Path, corpus_with_tokenizer: Path):
    """Отказ блокирует прогон: код 1 и причина в журнале (не «прогон поехал»)."""
    out_dir = tmp_path / "out"
    code = pretrain_run.main(
        [
            "--shard-root",
            str(corpus_with_tokenizer),
            "--run-ref",
            "tok-mismatch",
            "--out",
            str(out_dir),
            "--steps",
            "1",
        ]
    )
    assert code == 1
    journal = json.loads((out_dir / "journal.json").read_text(encoding="utf-8"))
    assert "токенизатор" in journal["refusal"]
    assert journal["status"] == "absent"


# --------------------------------------------------------------------------- #
# T-ov3: файла нет — прежнее поведение (заглушка)
# --------------------------------------------------------------------------- #


def test_absent_tokenizer_falls_back_to_stub(tmp_path: Path):
    """Нет ``<shard-root>/tokenizer`` — скелетная заглушка с собственным гейтом."""
    shard_root = tmp_path / "нет-токенизатора"
    shard_root.mkdir()
    tokenizer, cfg, info = pretrain_run.build_tokenizer_and_config(_args(shard_root))

    from net.tokenizer import BPETokenizer

    assert isinstance(tokenizer, BPETokenizer), "откат обязан дать заглушку скелета"
    assert "заглушка" in info["source"]
    assert info["hash"] == tokenizer.vocab_hash()
    # Пин описывает корпусной BPE, а не заглушку: расхождение фиксируется честно.
    assert info["matches_config_pin"] is False
    assert info["max_emitted_id"] < int(cfg.vocab_size)


# --------------------------------------------------------------------------- #
# T-ov4: артефакт правлен после упаковки — отказ
# --------------------------------------------------------------------------- #


def test_manifest_tamper_refuses(tmp_path: Path, corpus_with_tokenizer: Path):
    """Хеш файла ≠ ``tokenizer_hash`` манифеста — файл изменён после упаковки."""
    # Копируем каталог токенизатора, чтобы не портить module-scoped фикстуру.
    import shutil

    root = tmp_path / "tampered"
    root.mkdir()
    source = corpus_with_tokenizer / pretrain_run.CORPUS_TOKENIZER_SUBDIR
    target = root / pretrain_run.CORPUS_TOKENIZER_SUBDIR
    shutil.copytree(source, target)

    manifest_path = target / pretrain_run.CORPUS_TOKENIZER_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tokenizer_hash"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(pretrain_run.StageRefused) as excinfo:
        pretrain_run.build_tokenizer_and_config(_args(root))
    assert "манифест" in str(excinfo.value)
