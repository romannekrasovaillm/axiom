"""Канонический BPE 160K и претокенизация корпуса (ADR-004, CMP-001, ADR-021).

Проверяются свойства, от которых зависит претрейн на аренде: словарь и его
раскладка, round-trip по корпусу, детерминированность выборки, целостность
упаковки ``.bin`` и контракт чтения из ``net/train_loop.py``.

Сценарии:

* **T-tok1** — раскладка словаря: ``<pad>``=0, ``<bos>``=1, ``<eos>``=2,
  кодовый префикс=3, 256 байтовых токенов, словарь ровно объявленного размера;
* **T-tok2** — round-trip на корпусе (критерий 9 спеки скелета, ≥99,5 %);
* **T-tok3** — выборка детерминирована сидом и читает **окно**, а не весь шард
  (регресс на разрежённую выборку, которая требовала полного прохода корпуса);
* **T-tok4** — записи ``.bin`` ровно ``T=8192``, начинаются с BOS, PAD только в
  хвосте, размер файла кратен записи (оборванный шард не читается как готовый);
* **T-tok5** — EOS-стыки: между двумя EOS лежит ровно один документ, порядок
  документов сохраняется, кодовый префикс стоит только у потока C;
* **T-tok6** — per-shard счётчики токенов и throughput в манифесте сходятся с
  файлами на диске;
* **T-tok7** — читатель ``PackedTokenLoader``: ``(B, T) int32``, BOS в позиции 0,
  порядок записей, пропорция микса на гранулярности батча;
* **T-tok8** — отказы: чужая раскладка/схема/ids/BOS, укороченный ``.bin``,
  подменённый токенизатор, вывод вне разрешённых корней;
* **T-tok9** — resume: готовый шард не пересчитывается, а другой ``--limit-mb``
  или ``--restart`` заставляют пересчитать.

Все фикстуры синтетические (локальные ``.jsonl.zst`` мини-шарды), сеть не нужна,
канонический диск ``~/gb10-shared`` не трогается — вывод идёт в ``tmp_path``.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1]
CASE_DIR = TOOLS_DIR.parent
for _path in (str(CASE_DIR), str(TOOLS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import bpe_train  # noqa: E402
import pretokenize  # noqa: E402
from net import train_loop as tl  # noqa: E402
from prep_pretrain import common  # noqa: E402

SMALL_VOCAB = 512
SMALL_T = 64


# --------------------------------------------------------------------------- #
# Синтетический корпус
# --------------------------------------------------------------------------- #


#: Пул слов фикстуры: короткий и повторяемый, чтобы у BPE были частые пары.
#: Уникальные слова в каждом документе (первая версия фикстуры) дают пары с
#: частотой 1 — тренер доводит словарь до ~260 базовых токенов и останавливается.
WORD_POOL = (
    "the model attention delta token context language network data train loss grad step "
    "batch scale layer head query key value cache window sparse dense pool block merge "
    "код данные сеть обучение пример тест функция возврат цикл условие класс модуль "
    "def return import class for while if else print range list dict set int str float "
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi"
).split()


def synthetic_docs(prefix: str, count: int, words: int, *, offset: int = 0) -> list[str]:
    """Документы из фиксированного пула слов: у BPE есть частые пары для мерджей."""
    rng = random.Random(f"{prefix}:{offset}")
    return [" ".join(rng.choice(WORD_POOL) for _ in range(words)) for _ in range(count)]


def write_shard_set(root: Path, name: str, docs: list[str], *, shards: int = 1) -> Path:
    """Мини-шард-набор в контракте ``tools/prep_pretrain`` (тот же, что у лупа)."""
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


W_DOCS, C_DOCS, DOC_WORDS = 400, 200, 16


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """Мини-корпус: W из 2 шардов, C из 1 шарда (тексты пригодны для BPE)."""
    root = tmp_path / "axiom-pretrain-l3"
    write_shard_set(root, "W", synthetic_docs("w", W_DOCS, DOC_WORDS), shards=2)
    write_shard_set(root, "C", synthetic_docs("c", C_DOCS, DOC_WORDS), shards=1)
    return root


@pytest.fixture
def tokenizer_dir(corpus: Path, tmp_path: Path) -> Path:
    """Обученный мини-BPE + манифест (``tools/bpe_train.py`` на синтетике)."""
    out = tmp_path / "tokenizer"
    rc = bpe_train.main(
        [
            "--shard-root",
            str(corpus),
            "--out",
            str(out),
            "--vocab-size",
            str(SMALL_VOCAB),
            "--sample-mb",
            "1",
            "--round-trip-docs",
            "200",
        ]
    )
    assert rc == 0
    return out


def run_pretokenize(
    corpus: Path,
    tokenizer_dir: Path,
    out: Path,
    *extra: str,
    seq_len: int = SMALL_T,
    workers: int = 2,
) -> int:
    """Прогон претокенизации в ``out`` (тесты не трогают канонический диск)."""
    return pretokenize.main(
        [
            "--shard-root",
            str(corpus),
            "--out",
            str(out),
            "--tokenizer-dir",
            str(tokenizer_dir),
            "--seq-len",
            str(seq_len),
            "--workers",
            str(workers),
            *extra,
        ]
    )


def read_stream(path: Path, seq_len: int) -> list[int]:
    """Поток токенов из ``.bin``: записи сшиваются, BOS и хвостовой PAD отбрасываются.

    Именно так устроен обратный путь к документам: BOS — служебная граница
    записи, а не часть документа, поэтому в поток он не входит.
    """
    raw = np.fromfile(path, dtype=np.uint32).reshape(-1, seq_len)
    assert raw[0, 0] == bpe_train.BOS_ID, "запись обязана начинаться с BOS"
    stream: list[int] = []
    for index, row in enumerate(raw):
        body = row[1:]
        if index == len(raw) - 1:  # хвостовая запись: PAD — добивка, не данные
            body = body[body != bpe_train.PAD_ID]
        stream.extend(int(token) for token in body)
    return stream


def split_documents(stream: list[int]) -> list[list[int]]:
    """Разрезать поток по EOS: между двумя EOS — ровно один документ."""
    docs: list[list[int]] = []
    current: list[int] = []
    for token in stream:
        if token == bpe_train.EOS_ID:
            docs.append(current)
            current = []
        else:
            current.append(token)
    assert not current, "поток обязан заканчиваться EOS (документы закрыты)"
    return docs


# --------------------------------------------------------------------------- #
# T-tok1 / T-tok2: словарь и round-trip
# --------------------------------------------------------------------------- #


def test_specials_layout_and_manifest_hash(tokenizer_dir: Path) -> None:
    """T-tok1: раскладка словаря объявлена, а хеш манифеста — хеш файла (AD-4)."""
    manifest = json.loads((tokenizer_dir / "tokenizer-manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == bpe_train.TOKENIZER_SCHEMA
    assert manifest["specials"] == {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<|code|>": 3}
    assert manifest["code_prefix"] == bpe_train.CODE_PREFIX
    assert manifest["vocab_size"] == SMALL_VOCAB
    assert manifest["merges"] == SMALL_VOCAB - 4 - 256
    assert manifest["byte_tokens"] == [4, 259]
    # Хеш — sha256 файла, а не «что-то похожее»: файл правят руками, гейт молчит.
    artifact = tokenizer_dir / "tokenizer.model"
    actual = bpe_train.file_sha256(artifact)
    assert manifest["tokenizer_hash"] == actual
    assert bpe_train.verify_manifest(tokenizer_dir / "tokenizer-manifest.json")["tokenizer_hash"] == actual

    from tokenizers import Tokenizer

    vocab = Tokenizer.from_file(str(artifact)).get_vocab()
    assert vocab["<pad>"] == tl.PAD_ID and vocab["<bos>"] == tl.BOS_ID and vocab["<eos>"] == tl.EOS_ID
    assert vocab["<|code|>"] == 3
    assert len(vocab) == SMALL_VOCAB


def test_round_trip_on_corpus(tokenizer_dir: Path) -> None:
    """T-tok2: byte-level BPE лоссов — round-trip держится на неудобных текстах."""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.model"))
    texts = [
        "plain ascii text with punctuation, 100% and\nnew lines\n",
        "  leading and trailing spaces  ",
        "кириллица: проверка токенизации и обратно",
        "emoji 🚀 и юникод — 漢字, арабица: العربية",
        "code: def f(x: int) -> int:\n    return x + 1  # comment",
        "",
    ]
    for text in texts:
        ids = tokenizer.encode(text).ids
        assert tokenizer.decode(ids, skip_special_tokens=False) == text, text


def test_round_trip_rate_measured_on_sample(corpus: Path, tmp_path: Path) -> None:
    """T-tok2 (числом): rate ≥ 0,995 и записан в манифест, а не посчитан «на глаз»."""
    out = tmp_path / "tokenizer"
    assert (
        bpe_train.main(
            ["--shard-root", str(corpus), "--out", str(out), "--vocab-size", str(SMALL_VOCAB), "--sample-mb", "1"]
        )
        == 0
    )
    manifest = json.loads((out / "tokenizer-manifest.json").read_text(encoding="utf-8"))
    assert manifest["round_trip"]["rate"] >= 0.995
    assert manifest["round_trip"]["docs"] > 0


# --------------------------------------------------------------------------- #
# T-tok3: детерминированная выборка окном
# --------------------------------------------------------------------------- #


def test_sampling_deterministic_and_windowed(corpus: Path) -> None:
    """T-tok3: сид решает выборку; читается окно, а не весь шард.

    Регресс на первый вариант правила (каждый k-й документ): он требовал
    прочитать **весь** шард ради его доли, то есть 80 ГиБ на выборку в 2 ГиБ.
    """
    shards = bpe_train.load_corpus_shards(corpus)
    first, stats = bpe_train.sample_corpus(shards, sample_mb=0.005, seed=7)
    again = bpe_train.sample_corpus(shards, sample_mb=0.005, seed=7)[0]
    other = bpe_train.sample_corpus(shards, sample_mb=0.005, seed=8)[0]
    assert first == again, "тот же сид обязан дать ту же выборку"
    assert first and stats.chars > 0, "выборка непуста"
    assert bpe_train._shard_seed(7, "W", "W-00000.jsonl.zst") != bpe_train._shard_seed(
        8, "W", "W-00000.jsonl.zst"
    ), "сид обязан менять смещение окна шарда"
    assert first != other, "другой сид обязан дать другую выборку"

    total_records = sum(shard.records for shard in shards)
    assert stats.docs_seen < total_records // 2, (
        "выборка читает окно, а не весь корпус: "
        f"{stats.docs_seen} из {total_records} документов"
    )


def test_sampling_respects_stream_mix(corpus: Path) -> None:
    """T-tok3 (доли): объявленный микс выборки соблюдён с точностью до шарда."""
    shards = bpe_train.load_corpus_shards(corpus)
    _, stats = bpe_train.sample_corpus(shards, sample_mb=20, mix={"W": 0.5, "C": 0.5}, seed=3)
    assert set(stats.streams) == {"W", "C"}
    assert stats.streams["C"]["docs"] > 0 and stats.streams["W"]["docs"] > 0


# --------------------------------------------------------------------------- #
# T-tok4 / T-tok5 / T-tok6: упаковка .bin
# --------------------------------------------------------------------------- #


def test_packages_are_exactly_seq_len(corpus: Path, tokenizer_dir: Path, tmp_path: Path) -> None:
    """T-tok4: записи ровно ``T`` (канонический T=8192), PAD только в хвосте."""
    out = tmp_path / "tokens-8192"
    # Канонический T берётся из дефолта, а не подставляется тестом: контракт
    # «пакеты T=8192» проверяется тем же числом, что уйдёт в боевой прогон.
    assert run_pretokenize(corpus, tokenizer_dir, out, seq_len=pretokenize.DEFAULT_SEQ_LEN) == 0
    assert pretokenize.DEFAULT_SEQ_LEN == 8192

    manifest = json.loads((out / "W" / "manifest-w.json").read_text(encoding="utf-8"))
    assert manifest["seq_len"] == 8192 and manifest["dtype"] == "uint32"
    for entry in manifest["shards"]:
        path = out / "W" / entry["file"]
        raw = np.fromfile(path, dtype=np.uint32)
        assert raw.size % 8192 == 0, "размер файла обязан быть кратен записи"
        assert raw.size // 8192 == entry["records"]
        rows = raw.reshape(-1, 8192)
        assert np.all(rows[:, 0] == bpe_train.BOS_ID), "каждая запись начинается с BOS"
        if len(rows) > 1:
            assert not np.any(rows[:-1] == bpe_train.PAD_ID), "PAD допустим только в хвосте"


def test_eos_joints_and_code_prefix(corpus: Path, tokenizer_dir: Path, tmp_path: Path) -> None:
    """T-tok5: между двумя EOS — ровно один документ; префикс только у C."""
    out = tmp_path / "tokens-joints"
    assert run_pretokenize(corpus, tokenizer_dir, out) == 0

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.model"))
    for stream, docs, prefixed in (
        ("W", synthetic_docs("w", W_DOCS, DOC_WORDS), False),
        ("C", synthetic_docs("c", C_DOCS, DOC_WORDS), True),
    ):
        manifest = json.loads((out / stream / "manifest-{0}.json".format(stream.lower())).read_text(encoding="utf-8"))
        stream_tokens: list[int] = []
        for entry in manifest["shards"]:
            stream_tokens.extend(read_stream(out / stream / entry["file"], SMALL_T))
        recovered = split_documents(stream_tokens)
        assert len(recovered) == len(docs), "число EOS-сегментов равно числу документов"
        for index, (segment, text) in enumerate(zip(recovered, docs)):
            if prefixed:
                assert segment[0] == bpe_train.CODE_ID, f"документ C {index} без кодового префикса"
                segment = segment[1:]
            else:
                assert bpe_train.CODE_ID not in segment, "у потока W кодового префикса быть не должно"
            assert tokenizer.decode(segment, skip_special_tokens=False) == text, f"документ {index}"


def test_manifest_counts_and_throughput(corpus: Path, tokenizer_dir: Path, tmp_path: Path) -> None:
    """T-tok6: per-shard счётчики и throughput сходятся с файлами и с манифестом."""
    out = tmp_path / "tokens-counts"
    assert run_pretokenize(corpus, tokenizer_dir, out) == 0
    for stream in ("W", "C"):
        manifest = json.loads((out / stream / "manifest-{0}.json".format(stream.lower())).read_text(encoding="utf-8"))
        assert manifest["tokenizer_hash"] == bpe_train.file_sha256(tokenizer_dir / "tokenizer.model")
        totals = manifest["totals"]
        assert totals["shards"] == len(manifest["shards"]) > 0
        assert totals["records"] == sum(entry["records"] for entry in manifest["shards"])
        assert totals["tokens"] == sum(entry["tokens"] for entry in manifest["shards"])
        assert totals["bytes"] == sum(entry["bytes"] for entry in manifest["shards"])
        for entry in manifest["shards"]:
            assert entry["tokens"] == entry["records"] * manifest["seq_len"]
            assert (out / stream / entry["file"]).stat().st_size == entry["bytes"]
            assert entry["sha256"] == bpe_train.file_sha256(out / stream / entry["file"])
            assert entry["tokens_per_sec"] > 0, "throughput шарда обязан быть измерен"
            assert entry["eos_tokens"] == entry["documents"], "EOS по одному на документ"
            assert entry["stream_tokens"] == (
                entry["tokens"] - entry["records"] - entry["pad_tokens"]
            ), "слоты BOS и PAD не входят в токены потока"
        assert manifest["throughput"]["tokens_per_sec"] > 0


# --------------------------------------------------------------------------- #
# T-tok7: читатель net/train_loop.py
# --------------------------------------------------------------------------- #


def test_reader_adapter_batches(corpus: Path, tokenizer_dir: Path, tmp_path: Path) -> None:
    """T-tok7: адаптер отдаёт ``(B, T) int32`` в порядке записей и BOS в позиции 0."""
    out = tmp_path / "tokens-reader"
    assert run_pretokenize(corpus, tokenizer_dir, out) == 0
    loader = tl.PackedTokenLoader(
        tokens_root=out, streams=("W",), seq_len=SMALL_T, batch_size=4, mix={"W": 1.0}
    )
    batches = [batch for batch in loader]
    assert batches, "адаптер обязан отдать хотя бы один батч"
    for batch in batches:
        assert batch.shape == (4, SMALL_T)
        assert batch.dtype == np.int32
        assert np.all(batch[:, 0] == tl.BOS_ID)
    produced = np.stack(batches, axis=0).reshape(-1, SMALL_T)
    manifest = json.loads((out / "W" / "manifest-w.json").read_text(encoding="utf-8"))
    expected = np.concatenate(
        [
            np.fromfile(out / "W" / entry["file"], dtype=np.uint32).reshape(-1, SMALL_T)
            for entry in manifest["shards"]
        ],
        axis=0,
    )
    # Хвост потока меньше батча не выдаётся (у лупа фиксированная форма (B, T)):
    # теряются записи только в конце, и это видно счётчиком, а не молчанием.
    assert len(produced) == len(expected) - loader.stats()["dropped_tail_records"]["W"]
    assert np.array_equal(produced, expected[: len(produced)].astype(np.int32)), "порядок записей"
    stats = loader.stats()
    assert stats["streams"]["W"]["shard_index"] == len(manifest["shards"]) - 1
    assert stats["streams"]["W"]["tokenizer_hash"] == manifest["tokenizer_hash"]


def test_reader_adapter_keeps_mix(corpus: Path, tokenizer_dir: Path, tmp_path: Path) -> None:
    """T-tok7 (микс): доли потоков держатся на гранулярности батча."""
    out = tmp_path / "tokens-mix"
    assert run_pretokenize(corpus, tokenizer_dir, out) == 0
    loader = tl.PackedTokenLoader(
        tokens_root=out, streams=("W", "C"), seq_len=SMALL_T, batch_size=2, mix={"W": 0.5, "C": 0.5}
    )
    batches = list(loader)
    assert len(batches) >= 3, "нужно несколько батчей, чтобы доля была видна"
    stats = loader.stats()
    share = stats["token_share"]
    assert share["C"] > 0.25 and share["W"] > 0.25, share


# --------------------------------------------------------------------------- #
# T-tok8: отказы (негативные сценарии)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("version", "axiom-pretrain-tokens/0", "схема"),
        ("record_layout", "что-то другое", "раскладка"),
        ("bos_id", 7, "bos"),
        ("dtype", "int32", "dtype"),
        ("seq_len", 0, "seq_len"),
    ],
)
def test_reader_refuses_alien_manifest(
    corpus: Path, tokenizer_dir: Path, tmp_path: Path, field: str, value: object, reason: str
) -> None:
    """T-tok8: чужой манифест — отказ, а не «прочитаем как получится»."""
    out = tmp_path / f"tokens-bad-{field}"
    assert run_pretokenize(corpus, tokenizer_dir, out) == 0
    manifest_path = out / "W" / "manifest-w.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(tl.PretrainDataError):
        tl.load_packed_shard_set(manifest_path)
    assert reason  # параметр читается человеком в отчёте о провале


def test_reader_refuses_truncated_bin(corpus: Path, tokenizer_dir: Path, tmp_path: Path) -> None:
    """T-tok8 (обрыв): ``.bin`` не кратен записи — ошибка, а не короткий поток."""
    out = tmp_path / "tokens-truncated"
    assert run_pretokenize(corpus, tokenizer_dir, out) == 0
    manifest_path = out / "W" / "manifest-w.json"
    bin_path = out / "W" / json.loads(manifest_path.read_text(encoding="utf-8"))["shards"][0]["file"]
    with bin_path.open("r+b") as handle:
        handle.truncate(bin_path.stat().st_size - 7)
    with pytest.raises(tl.PretrainDataError):
        tl.PackedShardReader(tl.load_packed_shard_set(manifest_path).entries[0], SMALL_T)


def test_reader_refuses_alien_seq_len(corpus: Path, tokenizer_dir: Path, tmp_path: Path) -> None:
    """T-tok8 (T): заказ другого T, чем в манифесте, — отказ, а не пересборка."""
    out = tmp_path / "tokens-alien-t"
    assert run_pretokenize(corpus, tokenizer_dir, out) == 0
    with pytest.raises(tl.PretrainDataError):
        tl.PackedTokenLoader(tokens_root=out, streams=("W",), seq_len=SMALL_T * 2, batch_size=1)


def test_pretokenize_refuses_alien_tokenizer(corpus: Path, tokenizer_dir: Path, tmp_path: Path) -> None:
    """T-tok8 (подмена): хеш токенизатора сверяется с манифестом и с ожиданием."""
    pin = pretokenize.resolve_tokenizer_pin(tokenizer_dir=tokenizer_dir)
    assert pin.tokenizer_hash == bpe_train.file_sha256(tokenizer_dir / "tokenizer.model")
    with pytest.raises(pretokenize.PretokenizeError):
        pretokenize.resolve_tokenizer_pin(tokenizer_dir=tokenizer_dir, expect_hash="0" * 64)

    artifact = tokenizer_dir / "tokenizer.model"
    original = artifact.read_bytes()
    artifact.write_bytes(original + b"\n")  # подмена после упаковки
    try:
        with pytest.raises(pretokenize.PretokenizeError):
            pretokenize.resolve_tokenizer_pin(tokenizer_dir=tokenizer_dir)
    finally:
        artifact.write_bytes(original)


def test_output_root_guard(tmp_path: Path) -> None:
    """T-tok8 (C-032/C-033): вывод только в gb10-shared//tmp, иначе отказ."""
    with pytest.raises(ValueError):
        common.ensure_output_allowed(CASE_DIR / "tokens-out")
    inside = common.ensure_output_allowed(tmp_path / "anywhere")
    assert inside == (tmp_path / "anywhere").resolve()
    assert common.ensure_output_allowed(CASE_DIR / "tokens-out", allow_any=True)


# --------------------------------------------------------------------------- #
# T-tok9: resume
# --------------------------------------------------------------------------- #


def test_pretokenize_resume_skips_ready_shards(corpus: Path, tokenizer_dir: Path, tmp_path: Path) -> None:
    """T-tok9: готовый шард не пересчитывается, ``--restart`` — пересчитывает."""
    out = tmp_path / "tokens-resume"
    assert run_pretokenize(corpus, tokenizer_dir, out) == 0
    manifest_path = out / "W" / "manifest-w.json"
    before = json.loads(manifest_path.read_text(encoding="utf-8"))
    bin_path = out / "W" / before["shards"][0]["file"]
    stamp = bin_path.stat().st_mtime_ns

    assert run_pretokenize(corpus, tokenizer_dir, out) == 0  # resume
    after = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert after["shards"] == before["shards"], "готовый шард не переписывается"
    assert bin_path.stat().st_mtime_ns == stamp

    assert run_pretokenize(corpus, tokenizer_dir, out, "--restart") == 0
    assert bin_path.stat().st_mtime_ns != stamp, "--restart обязан пересчитать шард"


def test_pretokenize_limit_is_per_stream_budget(corpus: Path, tokenizer_dir: Path, tmp_path: Path) -> None:
    """T-tok9 (проба): ``--limit-mb`` ограничивает поток и не берёт лишние шарды."""
    out = tmp_path / "tokens-limit"
    assert run_pretokenize(corpus, tokenizer_dir, out, "--limit-mb", "0.01") == 0
    manifest = json.loads((out / "W" / "manifest-w.json").read_text(encoding="utf-8"))
    assert manifest["limit_mb"] == 0.01
    full = json.loads((corpus / "W" / "manifest-w.json").read_text(encoding="utf-8"))
    assert len(manifest["shards"]) < len(full["shards"]), "бюджет пробы не читает весь поток"
    manifest["shards"][0]["limit_bytes"] > 0
