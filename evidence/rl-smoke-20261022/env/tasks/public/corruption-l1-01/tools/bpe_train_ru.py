"""BPE v2 160K: выборка W/C v1 + русский доменный корпус (переприсэмпл ×N).

Аддитивная дельта к ``tools/bpe_train.py`` (сам тренер не трогается).  Контракт
обучения тот же — vocab 160000, раскладка ``<pad> <bos> <eos> <|code|>`` + 256
байтовых токенов, byte-level BPE, артефакт ``tokenizer.model`` + манифест с
``tokenizer_hash`` (AD-4/AD-11) — но в микс добавляется поток **R**: полный
русский доменный CPT-корпус агента, продублированный ``--ru-oversample`` раз.

Зачем.  В каноническом токенизаторе (``500f8023``) ноль кириллических токенов:
русский текст разбирается до почти отдельных байтов (замер зондом: 1,5–1,9
симв/токен против 5–7 у английского), поэтому доменная CPT-стадия платит
3–4-кратный налог по длине.  Переприсэмпл ×N поднимает доли мерджей на
кириллицу ровно настолько, чтобы словарь выучил русские слоги/слова.
Формат артефакта и манифеста совпадает с v1, поэтому ``pretokenize.py`` читает
v2 без правок.

CLI::

    python tools/bpe_train_ru.py \\
        --shard-root ~/gb10-shared/datasets/axiom-pretrain-l3 \\
        --ru-corpus ~/gb10-shared/datasets/cpt_corpus_v10.1.txt \\
        --ru-oversample 3 --sample-mb 2048 --seed 0 \\
        --out ~/gb10-shared/datasets/axiom-pretrain-l3/tokenizer-v2
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Sequence

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import bpe_train  # noqa: E402
from prep_pretrain import common  # noqa: E402

MB = bpe_train.MB

#: Полный русский доменный CPT-корпус агента (свежайшая ревизия по имени/дате;
#: она же — домен-источник миксов cpt_corpus_v12r/v12r50, см. их манифесты).
DEFAULT_RU_CORPUS = "~/gb10-shared/datasets/cpt_corpus_v10.1.txt"
DEFAULT_RU_OVERSAMPLE = 3
DEFAULT_RU_OUT = f"{common.DATASET_ROOT}/tokenizer-v2"


def iter_ru_docs(path: Path) -> Iterator[str]:
    """Документы русского корпуса: абзацы, разделённые пустой строкой.

    Корпус — markdown-подобные концепты (``# Концепт: …`` + секции), один
    документ = абзац.  Пустые абзацы отбрасываются; границы документов на
    частотный профиль мерджей не влияют, но честный счётчик ``records`` нужен
    манифесту.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    for block in text.split("\n\n"):
        block = block.strip("\n")
        if block:
            yield block


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BPE 160K v2: выборка W/C v1 + русский доменный корпус ×N"
    )
    parser.add_argument("--shard-root", default=bpe_train.DEFAULT_DATASET_ROOT,
                        help="корень датасета с {W,C}/manifest-*.json")
    parser.add_argument("--streams", nargs="+", default=["W", "C"], help="потоки корпуса W/C")
    parser.add_argument("--out", default=DEFAULT_RU_OUT, help="каталог артефакта и манифеста")
    parser.add_argument("--vocab-size", type=int, default=bpe_train.DEFAULT_VOCAB_SIZE)
    parser.add_argument("--sample-mb", type=float, default=bpe_train.DEFAULT_SAMPLE_MB,
                        help="объём выборки W/C в МиБ (как у v1)")
    parser.add_argument("--seed", type=int, default=0, help="сид выборки W/C (AD-11)")
    parser.add_argument("--mix", nargs="+", default=None, help="доли потоков W/C, напр. W=0.85 C=0.15")
    parser.add_argument("--text-keys", nargs="+", default=list(bpe_train.DEFAULT_TEXT_KEYS))
    parser.add_argument("--ru-corpus", default=DEFAULT_RU_CORPUS, help="путь к русскому доменному корпусу")
    parser.add_argument("--ru-oversample", type=int, default=DEFAULT_RU_OVERSAMPLE,
                        help="во сколько раз продублировать русский корпус в миксе (по умолчанию ×3)")
    parser.add_argument("--round-trip-docs", type=int, default=2000)
    parser.add_argument("--progress-every", type=int, default=200_000)
    parser.add_argument("--allow-any-out", action="store_true",
                        help="разрешить вывод вне gb10-shared//tmp (C-032/C-033)")
    return parser


def cmd_train(args: argparse.Namespace) -> int:
    echo = lambda message: print(message, flush=True)  # noqa: E731
    started = time.time()
    out_dir = common.ensure_output_allowed(args.out, allow_any=args.allow_any_out)
    manifest_path = out_dir / bpe_train.MANIFEST_FILE
    streams = [stream.upper() for stream in args.streams]
    mix = bpe_train.parse_mix(args.mix, streams)
    shards = bpe_train.load_corpus_shards(args.shard_root, streams)
    echo(
        f"[bpe2] шардов {len(shards)} "
        f"({', '.join(f'{s}: {sum(1 for x in shards if x.stream == s)}' for s in streams)}), "
        f"выборка W/C {args.sample_mb:.0f} МиБ, сид {args.seed}"
    )

    # -- W/C: та же логика выборки, что у v1 (окно, детерминированное сидом) --
    wc_texts, stats = bpe_train.sample_corpus(
        shards,
        sample_mb=args.sample_mb,
        mix=mix,
        seed=args.seed,
        text_keys=args.text_keys,
        progress_every=args.progress_every,
        echo=echo,
    )
    echo(
        f"[bpe2] выборка W/C: {stats.docs} документов, {stats.chars / MB:.1f} МиБ "
        f"за {time.time() - started:.1f} с"
    )

    # -- R: полный русский доменный корпус, переприсэмпленный ×N --------------
    ru_path = Path(os.path.expanduser(args.ru_corpus))
    if not ru_path.is_file():
        raise bpe_train.CorpusError(f"русский корпус не найден: {ru_path}")
    ru_docs = list(iter_ru_docs(ru_path))
    ru_chars = sum(len(doc) for doc in ru_docs)
    if not ru_docs:
        raise bpe_train.CorpusError(f"русский корпус пуст: {ru_path}")
    ru_texts = ru_docs * max(1, int(args.ru_oversample))
    ru_chars_eff = ru_chars * max(1, int(args.ru_oversample))
    echo(
        f"[bpe2] русский корпус: {len(ru_docs)} документов, {ru_chars / MB:.1f} МиБ "
        f"→ ×{args.ru_oversample} = {ru_chars_eff / MB:.1f} МиБ"
    )

    texts = wc_texts + ru_texts
    total_chars = stats.chars + ru_chars_eff
    ru_share = ru_chars_eff / total_chars if total_chars else 0.0
    echo(
        f"[bpe2] микс обучения: {len(texts)} документов, {total_chars / MB:.1f} МиБ, "
        f"доля R {ru_share * 100:.1f}% символов"
    )
    if not texts:
        raise bpe_train.CorpusError("микс пуст")

    tokenizer = bpe_train.train_bpe(texts, vocab_size=args.vocab_size, echo=echo)
    layout = bpe_train.verify_layout(
        tokenizer, vocab_size=args.vocab_size, specials=bpe_train.DEFAULT_SPECIALS
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = out_dir / bpe_train.TOKENIZER_FILE
    tokenizer.save(str(artifact))
    tokenizer_hash = bpe_train.file_sha256(artifact)
    round_trip = bpe_train.measure_round_trip(tokenizer, texts, limit=args.round_trip_docs)

    #: Счётчики микса: W/C из общей логики + поток R (наблюдаемые числа, не «~»).
    stats.docs += len(ru_texts)
    stats.chars += ru_chars_eff
    stats.streams["R"] = {
        "shards": 1,
        "docs": len(ru_texts),
        "chars": ru_chars_eff,
        "docs_skipped": 0,
        "chars_per_doc": round(ru_chars / len(ru_docs), 1),
        "source_docs": len(ru_docs),
        "oversample": int(args.ru_oversample),
        "source_file": ru_path.name,
    }
    ru_shard = bpe_train.CorpusShard(
        stream="R",
        file=ru_path.name,
        path=ru_path,
        sha256=bpe_train.file_sha256(ru_path),
        approx_tokens=max(1, ru_chars // 4),
        bytes=ru_path.stat().st_size,
        records=len(ru_docs),
    )
    elapsed = time.time() - started
    manifest = bpe_train.build_manifest(
        tokenizer_path=artifact,
        tokenizer_hash=tokenizer_hash,
        layout=layout,
        stats=stats,
        shards=list(shards) + [ru_shard],
        round_trip=round_trip,
        seed=args.seed,
        sample_mb=args.sample_mb,
        stream_scope=streams + ["R"],
        elapsed_s=elapsed,
    )
    manifest["ru_corpus"] = {
        "file": ru_path.name,
        "path": str(ru_path),
        "sha256": ru_shard.sha256,
        "bytes": ru_path.stat().st_size,
        "docs": len(ru_docs),
        "oversample": int(args.ru_oversample),
        "char_share": round(ru_share, 4),
        "chars_source": ru_chars,
        "chars_effective": ru_chars_eff,
    }
    bpe_train.write_manifest(manifest_path, manifest)
    echo(
        f"[bpe2] готово: vocab {layout['vocab_size']} / мерджей {layout['merges']}, "
        f"round-trip {round_trip['rate'] * 100:.3f}% на {round_trip['docs']} документах"
    )
    echo(f"[bpe2] tokenizer_hash {tokenizer_hash}")
    echo(f"[bpe2] артефакт {artifact} ({artifact.stat().st_size / MB:.1f} МиБ), манифест {manifest_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return cmd_train(args)
    except bpe_train.TokenizerError as exc:
        print(f"[bpe2] ОТКАЗ: {exc}", file=sys.stderr, flush=True)
        return 2
    except ValueError as exc:  # ensure_output_allowed
        print(f"[bpe2] ОТКАЗ вывода: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
