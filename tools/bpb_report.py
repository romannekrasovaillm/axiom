#!/usr/bin/env python3
"""BPB-отчёт dense-ноги верификации: токенизатор-независимая кривая против эталона.

Прибор ноги ``docs/specs/VERIFICATION-LEG.ru.md`` (критерий 1): тот же
претрейн-конвейер на dense-конфигурации GPT-2-класса должен воспроизводить
известную кривую.  Абсолютный лосс сравнивать нельзя: у нашего токенизатора
словарь 160K, у GPT-2 — 50K, и один и тот же текст распадается на разное число
токенов.  Поэтому метрика — **bits per byte**:

    BPB = loss_nats_per_token × (tokens / bytes) / ln(2)

``loss`` берётся из jsonl-метрик прогона (``PretrainMixLoader``/``train_loop``:
ключи ``step``/``loss``/``tokens_seen``), коэффициент ``tokens/bytes``
измеряется на сэмпле текстов тем же токенизатором, что размечены данные.
BPB — инвариант токенизатора: на одном тексте он сравнивает разные словари
честно.

Что делает CLI
--------------
1. Читает jsonl метрик (``--loss-jsonl``): все записи с числовым ``loss``.
2. Определяет ``tokens/bytes``:
   * ``--tokenizer-manifest`` — манифест корпусного BPE
     (``tools/bpe_train.py: tokenizer-manifest.json``): артефакт
     (``tokenizers-json/v1``, библиотека ``tokenizers``) читается, хеш
     сверяется с пином манифеста, сэмпл текстов кодируется, коэффициент —
     ``Σ токенов / Σ байт``;
   * ``--tokens-per-byte`` — прямое задание коэффициента: прибор работает без
     библиотеки ``tokenizers`` (токенизатор ещё не обучен, или прогон на
     вычислителе без пакета).
   Оба флага взаимоисключающи; отсутствие обоих — ошибка входа.
3. Считает BPB по каждой записи, агрегирует **медианой по окну** (детерминизм:
   последние ``--window`` записей, окно не меньше ``MIN_WINDOW``).
4. Сравнивает с эталонной репликацией (``--reference``, по умолчанию
   corpus-matched: llm.c/FineWeb) и выносит вердикт
   ``|ΔBPB| ≤ BPB_TOLERANCE`` (10 %, критерий 1 спеки).
5. Пишет JSON-отчёт в ``--out``.

Вердикты и коды возврата
------------------------
* ``pass``        — ``|ΔBPB| ≤ 10 %`` → exit 0;
* ``fail``        — отклонение больше допуска → exit 1;
* ``input-error`` — вход отсутствует/непригоден (нет файла, пустой jsonl,
  битая запись, нечитаемый токенизатор) → exit 2 (fail-closed: не оценённый
  прогон не считается прошедшим).

Сравнение ведётся на равном объёме токенов корпуса: ``--at-tokens`` ограничивает
хвост окна (по умолчанию — весь доступный хвост).  Отчёт честно несёт
``reference.tokens``: эталон llm.c измерен на ~10B токенов, наша нога по спеке —
~1B; сопоставление по BPB это не скрывает, а показывает.

Запуск::

    python3 tools/bpb_report.py --selftest
    python3 tools/bpb_report.py \\
        --loss-jsonl evidence/verification-leg-dense124m/metrics.jsonl \\
        --sample-texts evidence/verification-leg-dense124m/holdout.txt \\
        --tokenizer-manifest ~/gb10-shared/datasets/axiom-pretrain-l3/tokenizer/tokenizer-manifest.json \\
        --out evidence/verification-leg-dense124m/bpb-report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

#: Схема отчёта.
REPORT_SCHEMA = "axiom-bpb-report/1"

#: ln(2): перевод натурального лосса (наты/токен) в биты/токен.
LN2 = math.log(2.0)

#: Допуск вердикта (VERIFICATION-LEG.ru.md, критерий 1): |ΔBPB| ≤ 10 %.
BPB_TOLERANCE = 0.10

#: Машинная погрешность границы допуска: кривая, стоящая *ровно* на 10 %, при
#: делении в float даёт 0.10000000000000009 и без этой поправки падала бы на
#: собственном определении «≤».  Поправка на 12 порядков меньше самого допуска
#: и не меняет вердикт ни на одном реальном отклонении.
BPB_COMPARISON_EPS = 1e-12

#: Минимальный размер окна медианы (детерминизм: не меньше 10 записей).
MIN_WINDOW = 10
#: Размер окна медианы по умолчанию.
WINDOW = 10

#: Коэффициент ``tokens/bytes`` эталонного токенизатора GPT-2 для перевода
#: эталонного лосса в BPB.  GPT-2 использует byte-level BPE; на англоязычном
#: веб-тексте (FineWeb/OpenWebText) это ≈4 байта на токен, т.е. ≈0.25 токена на
#: байт.  Это единственное допущение сравнения: оно нужно, чтобы у эталона и
#: нашей кривой была одна и та же (токенизатор-независимая) метрика.  Число
#: вынесено константой с комментарием, а не спрятано в формуле.
REFERENCE_TOKENS_PER_BYTE = 0.25

#: Эталонные точки — репликации GPT-2 124M.  ``val_loss`` в натах на токен, как
#: опубликовано первоисточником; BPB получается умножением на коэффициент
#: токенизатора эталона (``REFERENCE_TOKENS_PER_BYTE``) и делением на ln 2.
#: Источники проверены 05.10.2026 (VERIFICATION-LEG.ru.md, «Метрика и эталон»).
REFERENCES: dict[str, dict[str, Any]] = {
    "llm_c_fineweb": {
        "id": "llm_c_fineweb",
        "label": "llm.c (karpathy), GPT-2 124M, FineWeb val",
        "corpus": "FineWeb",
        "val_loss_nats_per_token": 3.28,
        "tokens": 10_000_000_000,
        "source": (
            "https://github.com/karpathy/llm.c — репликация GPT-2 124M на "
            "FineWeb, val loss ≈3.28 на полном ~10B-токен прогоне "
            "(VERIFICATION-LEG.ru.md, 05.10.2026)"
        ),
    },
    "nanogpt_owt": {
        "id": "nanogpt_owt",
        "label": "nanoGPT (karpathy), GPT-2 124M, OpenWebText val",
        "corpus": "OpenWebText",
        "val_loss_nats_per_token": 2.85,
        "tokens": None,
        "source": (
            "https://github.com/karpathy/nanoGPT — репликация GPT-2 124M на "
            "OpenWebText, val loss ≈2.85 (VERIFICATION-LEG.ru.md, 05.10.2026)"
        ),
    },
}

#: Эталон по умолчанию — совпадающий по корпусу с нашей ногой (FineWeb).
DEFAULT_REFERENCE = "llm_c_fineweb"

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_INPUT = 2


class InputError(Exception):
    """Вход отсутствует или непригоден: fail-closed (вердикт ``input-error``)."""


# --------------------------------------------------------------------------- #
# Конверсия
# --------------------------------------------------------------------------- #


def bpb_from_loss(loss_nats_per_token: float, tokens_per_byte: float) -> float:
    """``BPB = loss × (tokens/bytes) / ln(2)`` — одна точка кривой."""
    return float(loss_nats_per_token) * float(tokens_per_byte) / LN2


def tokens_per_byte_from_counts(tokens: int, n_bytes: int) -> float:
    """Коэффициент ``Σ токенов / Σ байт``; ноль байт — ошибка входа."""
    if n_bytes <= 0:
        raise InputError("сэмпл текстов пуст: 0 байт — коэффициент не определён")
    if tokens < 0:
        raise InputError(f"число токенов отрицательно: {tokens}")
    return float(tokens) / float(n_bytes)


def tokens_per_byte_from_tokenizer(tokenizer: Any, texts: Iterable[str]) -> dict[str, Any]:
    """Измерить ``tokens/bytes`` на сэмпле тем же токенизатором, что размечены данные.

    Токенизатор — duck-typed объект ``tokenizers.Tokenizer``: ``encode(text,
    add_special_tokens=False).ids``.  Спецтокены не добавляются: считаем именно
    байты текста, а не разметку начала/конца.  Детерминизм: результат зависит
    только от (тексты, токенизатор), порядок записей не влияет на суммы.
    """
    total_tokens = 0
    total_bytes = 0
    n_texts = 0
    for text in texts:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        ids = getattr(encoded, "ids", encoded)
        total_tokens += len(ids)
        total_bytes += len(text.encode("utf-8"))
        n_texts += 1
    if n_texts == 0:
        raise InputError("сэмпл текстов не содержит текста — коэффициент не определён")
    ratio = tokens_per_byte_from_counts(total_tokens, total_bytes)
    return {
        "tokens_per_byte": ratio,
        "tokens": total_tokens,
        "bytes": total_bytes,
        "n_texts": n_texts,
    }


# --------------------------------------------------------------------------- #
# Чтение входов
# --------------------------------------------------------------------------- #


def load_loss_rows(path: str | Path) -> list[dict[str, Any]]:
    """Записи jsonl метрик с числовым ``loss``; битая строка — fail-closed.

    Пустой файл, отсутствие поля ``loss``, нечисловой ``loss`` и мусорная
    строка — ошибка входа: молчаливый пропуск записи исказил бы кривую, а
    «не оценён» не имеет права выглядеть прошедшим.
    """
    p = Path(path)
    try:
        raw = p.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise InputError(f"{p}: файл метрик не найден") from exc
    except OSError as exc:
        raise InputError(f"{p}: файл метрик не читается: {exc}") from exc

    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InputError(f"{p}:{lineno}: строка не разбирается как JSON: {exc}") from exc
        if not isinstance(obj, dict):
            raise InputError(f"{p}:{lineno}: запись jsonl не объект")
        loss = obj.get("loss")
        if not _is_number(loss):
            raise InputError(f"{p}:{lineno}: поле loss отсутствует или не число: {loss!r}")
        row: dict[str, Any] = {"loss": float(loss)}
        if "step" in obj:
            row["step"] = obj["step"]
        if "tokens_seen" in obj:
            if not _is_number(obj["tokens_seen"]) or obj["tokens_seen"] < 0:
                raise InputError(
                    f"{p}:{lineno}: tokens_seen не неотрицательное число: {obj['tokens_seen']!r}"
                )
            row["tokens_seen"] = obj["tokens_seen"]
        rows.append(row)

    if not rows:
        raise InputError(f"{p}: файл метрик пуст — кривой BPB нет")
    return rows


def read_sample_texts(path: str | Path) -> tuple[str, int]:
    """Сэмпл текстов и его размер в байтах (UTF-8); пустой — ошибка входа."""
    p = Path(path)
    try:
        data = p.read_bytes()
    except FileNotFoundError as exc:
        raise InputError(f"{p}: сэмпл текстов не найден") from exc
    except OSError as exc:
        raise InputError(f"{p}: сэмпл текстов не читается: {exc}") from exc
    if not data:
        raise InputError(f"{p}: сэмпл текстов пуст")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InputError(f"{p}: сэмпл текстов не UTF-8: {exc}") from exc
    return text, len(data)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_tokenizer_from_manifest(manifest_path: str | Path) -> tuple[Any, dict[str, Any]]:
    """Загрузить токенизатор манифеста ``tools/bpe_train.py`` и сверить пин.

    Манифест — контракт AD-4: ``tokenizer.file`` (артефакт ``tokenizers-json/v1``)
    и ``tokenizer_hash``.  Хеш файла пересчитывается и сверяется: подменённый
    после упаковки токенизатор исказил бы коэффициент молча.  Библиотека
    ``tokenizers`` отсутствует — ошибка входа с подсказкой про
    ``--tokens-per-byte`` (прибор обязан работать и без неё).
    """
    p = Path(manifest_path)
    try:
        manifest = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise InputError(f"{p}: манифест токенизатора не найден") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise InputError(f"{p}: манифест токенизатора нечитаем: {exc}") from exc
    if not isinstance(manifest, dict):
        raise InputError(f"{p}: манифест токенизатора не объект JSON")
    tokenizer_block = manifest.get("tokenizer")
    if not isinstance(tokenizer_block, dict) or not tokenizer_block.get("file"):
        raise InputError(f"{p}: в манифесте нет tokenizer.file")
    artifact = p.parent / str(tokenizer_block["file"])
    if not artifact.is_file():
        raise InputError(f"{artifact}: артефакт токенизатора отсутствует")
    pinned = manifest.get("tokenizer_hash")
    actual = _file_sha256(artifact)
    if pinned and pinned != actual:
        raise InputError(
            f"{artifact}: tokenizer_hash не совпал с пином манифеста "
            f"({actual[:16]}… != {str(pinned)[:16]}…) — файл подменён после упаковки"
        )
    try:
        from tokenizers import Tokenizer  # noqa: PLC0415 — ленивый импорт
    except ImportError as exc:
        raise InputError(
            "библиотека tokenizers недоступна; задайте коэффициент напрямую "
            "флагом --tokens-per-byte <ratio>"
        ) from exc
    try:
        tokenizer = Tokenizer.from_file(str(artifact))
    except Exception as exc:  # noqa: BLE001 — чужая библиотека: оборачиваем в InputError
        raise InputError(f"{artifact}: токенизатор не загружается: {exc}") from exc
    return tokenizer, {
        "manifest": str(p),
        "artifact": str(artifact),
        "tokenizer_hash": actual,
        "vocab_size": manifest.get("vocab_size"),
    }


# --------------------------------------------------------------------------- #
# Агрегация и вердикт
# --------------------------------------------------------------------------- #


def _is_number(value: Any) -> bool:
    """Число, но не ``bool`` (``bool`` — подкласс ``int``)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def median_over_window(values: list[float], window: int) -> float:
    """Медиана последних ``window`` значений (детерминизм окна)."""
    if not values:
        raise InputError("нет значений для медианы")
    tail = values[-window:] if window > 0 else values
    return float(statistics.median(tail))


def reference_bpb(reference: str) -> dict[str, Any]:
    """Эталонная точка, приведённая к BPB тем же переводом, что наша кривая."""
    if reference not in REFERENCES:
        raise InputError(
            f"неизвестный эталон {reference!r}; доступны: {', '.join(sorted(REFERENCES))}"
        )
    ref = dict(REFERENCES[reference])
    ref["tokens_per_byte"] = REFERENCE_TOKENS_PER_BYTE
    ref["bpb"] = bpb_from_loss(ref["val_loss_nats_per_token"], REFERENCE_TOKENS_PER_BYTE)
    return ref


def build_report(
    loss_jsonl: str | Path,
    sample_texts: str | Path,
    *,
    tokenizer_manifest: str | Path | None = None,
    tokens_per_byte: float | None = None,
    window: int = WINDOW,
    at_tokens: int | None = None,
    reference: str = DEFAULT_REFERENCE,
) -> dict[str, Any]:
    """Собрать отчёт (без записи на диск); сбой входа — :class:`InputError`."""
    if tokens_per_byte is not None and tokenizer_manifest is not None:
        raise InputError(
            "--tokens-per-byte и --tokenizer-manifest взаимоисключащи: "
            "коэффициент задаётся либо измерением, либо явно"
        )
    if tokens_per_byte is None and tokenizer_manifest is None:
        raise InputError(
            "нужен --tokenizer-manifest (измерение коэффициента) "
            "или --tokens-per-byte (прямое задание)"
        )
    if not _is_number(tokens_per_byte) and tokens_per_byte is not None:
        raise InputError(f"--tokens-per-byte должен быть числом: {tokens_per_byte!r}")
    if tokens_per_byte is not None and tokens_per_byte <= 0:
        raise InputError(f"--tokens-per-byte должен быть > 0: {tokens_per_byte!r}")
    if window < MIN_WINDOW:
        raise InputError(
            f"окно медианы {window} меньше минимума {MIN_WINDOW} "
            "(детерминизм агрегата)"
        )
    if at_tokens is not None and at_tokens <= 0:
        raise InputError(f"--at-tokens должен быть > 0: {at_tokens!r}")

    rows = load_loss_rows(loss_jsonl)
    text, sample_bytes = read_sample_texts(sample_texts)

    if tokenizer_manifest is not None:
        tokenizer, tok_meta = load_tokenizer_from_manifest(tokenizer_manifest)
        measured = tokens_per_byte_from_tokenizer(tokenizer, [text])
        ratio = measured["tokens_per_byte"]
        calibration = {
            "source": "tokenizer-manifest",
            "tokens_per_byte": ratio,
            "tokens": measured["tokens"],
            "bytes": measured["bytes"],
            "sample_bytes": sample_bytes,
            **tok_meta,
        }
    else:
        ratio = float(tokens_per_byte)  # type: ignore[arg-type]
        calibration = {
            "source": "explicit --tokens-per-byte",
            "tokens_per_byte": ratio,
            "tokens": None,
            "bytes": None,
            "sample_bytes": sample_bytes,
        }

    curve = []
    for row in rows:
        point: dict[str, Any] = {
            "loss": row["loss"],
            "bpb": bpb_from_loss(row["loss"], ratio),
        }
        if "step" in row:
            point["step"] = row["step"]
        if "tokens_seen" in row:
            point["tokens_seen"] = row["tokens_seen"]
        curve.append(point)

    # Хвост под вердикт: при --at-tokens — только записи в пределах бюджета
    # (сопоставление на равном объёме токенов корпуса).
    selected = curve
    if at_tokens is not None:
        selected = [p for p in curve if p.get("tokens_seen", 0) <= at_tokens]
        if not selected:
            raise InputError(
                f"--at-tokens={at_tokens}: ни одна запись метрик не укладывается "
                "в бюджет токенов"
            )
    our_bpb = median_over_window([p["bpb"] for p in selected], window)
    last = selected[-1]

    ref = reference_bpb(reference)
    delta_rel = (our_bpb - ref["bpb"]) / ref["bpb"]
    passed = abs(delta_rel) <= BPB_TOLERANCE + BPB_COMPARISON_EPS

    return {
        "schema": REPORT_SCHEMA,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "verdict": "pass" if passed else "fail",
        "inputs": {
            "loss_jsonl": str(loss_jsonl),
            "sample_texts": str(sample_texts),
            "tokenizer_manifest": str(tokenizer_manifest) if tokenizer_manifest else None,
            "at_tokens": at_tokens,
        },
        "calibration": calibration,
        "curve": curve,
        "bpb": {
            "window": window,
            "n_rows_total": len(curve),
            "n_rows_selected": len(selected),
            "tokens_seen": last.get("tokens_seen"),
            "median": our_bpb,
        },
        "reference": ref,
        "delta": {
            "relative": delta_rel,
            "abs_relative": abs(delta_rel),
            "tolerance": BPB_TOLERANCE,
            "passed": passed,
        },
        "message": (
            f"BPB={our_bpb:.4f} против эталона {ref['id']} {ref['bpb']:.4f} "
            f"(Δ={delta_rel:+.2%}, допуск ±{BPB_TOLERANCE:.0%}) — "
            f"{'PASS' if passed else 'FAIL'}"
        ),
    }


def run_report(
    loss_jsonl: str | Path,
    sample_texts: str | Path,
    *,
    tokenizer_manifest: str | Path | None = None,
    tokens_per_byte: float | None = None,
    out: str | Path | None = None,
    window: int = WINDOW,
    at_tokens: int | None = None,
    reference: str = DEFAULT_REFERENCE,
) -> tuple[int, dict[str, Any]]:
    """Отчёт + код возврата; сбой входа — ``(EXIT_INPUT, {verdict: input-error})``."""
    try:
        report = build_report(
            loss_jsonl,
            sample_texts,
            tokenizer_manifest=tokenizer_manifest,
            tokens_per_byte=tokens_per_byte,
            window=window,
            at_tokens=at_tokens,
            reference=reference,
        )
    except InputError as exc:
        return EXIT_INPUT, {
            "schema": REPORT_SCHEMA,
            "verdict": "input-error",
            "error": str(exc),
        }
    code = EXIT_OK if report["verdict"] == "pass" else EXIT_FAIL
    report["exit_code"] = code
    if out is not None:
        target = Path(out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return code, report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _emit(report: dict[str, Any], quiet: bool) -> None:
    if not quiet:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    if report.get("verdict") in {"fail", "input-error"}:
        detail = report.get("message") or report.get("error", "")
        print(f"[bpb-report] {report['verdict'].upper()}: {detail}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="BPB-отчёт dense-ноги: loss × (tokens/bytes) / ln(2) против эталона"
    )
    parser.add_argument("--loss-jsonl", required=False, default=None,
                        help="jsonl метрик прогона (ключи step/loss/tokens_seen)")
    parser.add_argument("--sample-texts", required=False, default=None,
                        help="сэмпл текстов holdout для коэффициента tokens/bytes")
    parser.add_argument("--out", required=False, default=None,
                        help="куда записать JSON-отчёт")
    parser.add_argument("--tokenizer-manifest", default=None,
                        help="манифест корпусного BPE (tools/bpe_train.py) — измерение")
    parser.add_argument("--tokens-per-byte", type=float, default=None,
                        help="прямое задание коэффициента (прибор без tokenizers)")
    parser.add_argument("--window", type=int, default=WINDOW,
                        help=f"размер окна медианы, не меньше {MIN_WINDOW} "
                             f"(по умолчанию {WINDOW})")
    parser.add_argument("--at-tokens", type=int, default=None,
                        help="бюджет токенов корпуса для вердикта (равный объём)")
    parser.add_argument("--reference", default=DEFAULT_REFERENCE,
                        choices=sorted(REFERENCES),
                        help=f"эталонная репликация (по умолчанию {DEFAULT_REFERENCE})")
    parser.add_argument("--quiet", action="store_true", help="не печатать отчёт в stdout")
    parser.add_argument("--selftest", action="store_true",
                        help="синтетический selftest в tmp (реальные данные не трогаются)")
    args = parser.parse_args(argv)

    if args.selftest:
        return run_selftest()

    missing = [
        flag
        for flag, value in (
            ("--loss-jsonl", args.loss_jsonl),
            ("--sample-texts", args.sample_texts),
            ("--out", args.out),
        )
        if not value
    ]
    if missing:
        parser.error(f"обязательные флаги: {', '.join(missing)}")

    code, report = run_report(
        args.loss_jsonl,
        args.sample_texts,
        tokenizer_manifest=args.tokenizer_manifest,
        tokens_per_byte=args.tokens_per_byte,
        out=args.out,
        window=args.window,
        at_tokens=args.at_tokens,
        reference=args.reference,
    )
    _emit(report, args.quiet)
    return code


# --------------------------------------------------------------------------- #
# Selftest: контрольная конверсия, детерминизм, PASS/FAIL, fail-closed
# --------------------------------------------------------------------------- #


def _write_metrics(path: Path, losses: list[float], *, total_tokens: int | None = None) -> None:
    step_tokens = 8192
    rows = []
    for i, loss in enumerate(losses):
        row = {"schema": "pretrain-metrics/v1", "step": i + 1, "loss": loss,
               "tokens": step_tokens, "tokens_seen": step_tokens * (i + 1)}
        rows.append(row)
    if total_tokens is not None:
        rows = rows[: max(1, total_tokens // step_tokens)]
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )


def run_selftest() -> int:
    """Синтетика в ``tmp``: конверсия, окно, детерминизм, PASS/FAIL, границы."""
    import tempfile

    checks: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory(prefix="bpb-report-selftest-") as tmp:
        root = Path(tmp)
        sample = root / "holdout.txt"
        sample.write_text("The quick brown fox jumps over the lazy dog. " * 20,
                          encoding="utf-8")

        # Контрольная конверсия: loss = 8·ln2, ratio 0.25 → BPB = 8·0.25 = 2.0.
        checks.append(("контрольная конверсия BPB=2.0",
                       abs(bpb_from_loss(8.0 * LN2, 0.25) - 2.0) < 1e-12))
        # BPB не зависит от токенизатора: смена ratio меняет BPB пропорционально.
        checks.append(("BPB линейна по tokens/byte",
                       abs(bpb_from_loss(LN2, 0.5) - 0.5) < 1e-12))

        ref = reference_bpb("llm_c_fineweb")
        # Эталон llm.c: 3.28 × 0.25 / ln2 ≈ 1.1830.
        checks.append(("эталон llm.c переведён в BPB",
                       abs(ref["bpb"] - 3.28 * 0.25 / LN2) < 1e-12))
        # Наша кривая с тем же коэффициентом и тем же лоссом → PASS (Δ=0).
        at_ref = root / "at-ref.jsonl"
        _write_metrics(at_ref, [3.28] * 15)
        code, rep = run_report(at_ref, sample, tokens_per_byte=0.25, out=root / "r1.json")
        checks.append(("совпадение с эталоном → exit 0", code == EXIT_OK))
        checks.append(("совпадение с эталоном → verdict pass", rep["verdict"] == "pass"))
        checks.append(("совпадение → |Δ|≈0", abs(rep["delta"]["relative"]) < 1e-9))

        # Отклонение +50 % (loss 1.5×) → FAIL (за порогом 10 %).
        far = root / "far.jsonl"
        _write_metrics(far, [3.28 * 1.5] * 15)
        code_f, rep_f = run_report(far, sample, tokens_per_byte=0.25, out=root / "r2.json")
        checks.append(("отклонение 50% → exit 1", code_f == EXIT_FAIL))
        checks.append(("отклонение 50% → verdict fail", rep_f["verdict"] == "fail"))
        checks.append(("Δ зафиксирована как +50%",
                       abs(rep_f["delta"]["relative"] - 0.5) < 1e-9))

        # Граница допуска: +10 % ровно → PASS; чуть выше → FAIL.
        edge = root / "edge.jsonl"
        _write_metrics(edge, [3.28 * 1.10] * 15)
        checks.append(("отклонение ровно 10% → PASS (≤)",
                       run_report(edge, sample, tokens_per_byte=0.25)[0] == EXIT_OK))
        over = root / "over.jsonl"
        _write_metrics(over, [3.28 * 1.1001] * 15)
        checks.append(("отклонение >10% → FAIL",
                       run_report(over, sample, tokens_per_byte=0.25)[0] == EXIT_FAIL))

        # Детерминизм: два прогона на тех же входах — байт-в-байт равный отчёт
        # (без поля времени) и равный коэффициент.
        r_a = build_report(at_ref, sample, tokens_per_byte=0.25)
        r_b = build_report(at_ref, sample, tokens_per_byte=0.25)
        r_a.pop("generated_utc"); r_b.pop("generated_utc")
        checks.append(("детерминизм отчёта", r_a == r_b))
        checks.append(("детерминизм коэффициента",
                       r_a["calibration"]["tokens_per_byte"] == 0.25))

        # Окно: ранняя плохая фаза, поздняя хорошая → берём последние (PASS).
        late_good = root / "late-good.jsonl"
        _write_metrics(late_good, [3.28 * 2.0] * 20 + [3.28] * 20)
        checks.append(("поздняя здоровая фаза ловится окном → PASS",
                       run_report(late_good, sample, tokens_per_byte=0.25)[0] == EXIT_OK))
        early_good = root / "early-good.jsonl"
        _write_metrics(early_good, [3.28] * 20 + [3.28 * 2.0] * 20)
        checks.append(("поздняя деградация → FAIL",
                       run_report(early_good, sample, tokens_per_byte=0.25)[0] == EXIT_FAIL))

        # --at-tokens: обрезает хвост по бюджету токенов корпуса.
        mixed = root / "mixed.jsonl"
        _write_metrics(mixed, [3.28 * 2.0] * 20 + [3.28] * 20)
        code_at, rep_at = run_report(
            mixed, sample, tokens_per_byte=0.25, at_tokens=8192 * 20
        )
        checks.append(("--at-tokens обрезает ранний хвост → FAIL",
                       code_at == EXIT_FAIL and rep_at["bpb"]["n_rows_selected"] == 20))

        # Границы ошибок входа (fail-closed, exit 2).
        empty = root / "empty.jsonl"
        empty.write_text("\n", encoding="utf-8")
        checks.append(("пустой jsonl → exit 2",
                       run_report(empty, sample, tokens_per_byte=0.25)[0] == EXIT_INPUT))
        broken = root / "broken.jsonl"
        broken.write_text("не json\n", encoding="utf-8")
        checks.append(("мусорная строка → exit 2",
                       run_report(broken, sample, tokens_per_byte=0.25)[0] == EXIT_INPUT))
        no_loss = root / "no-loss.jsonl"
        no_loss.write_text(json.dumps({"step": 1}) + "\n", encoding="utf-8")
        checks.append(("запись без loss → exit 2",
                       run_report(no_loss, sample, tokens_per_byte=0.25)[0] == EXIT_INPUT))
        checks.append(("нет файла метрик → exit 2",
                       run_report(root / "nope.jsonl", sample,
                                  tokens_per_byte=0.25)[0] == EXIT_INPUT))
        checks.append(("оба способа коэффициента → exit 2",
                       run_report(at_ref, sample, tokens_per_byte=0.25,
                                  tokenizer_manifest=root / "m.json")[0] == EXIT_INPUT))
        checks.append(("ни одного способа → exit 2",
                       run_report(at_ref, sample)[0] == EXIT_INPUT))
        checks.append(("окно меньше минимума → exit 2",
                       run_report(at_ref, sample, tokens_per_byte=0.25,
                                  window=MIN_WINDOW - 1)[0] == EXIT_INPUT))
        checks.append(("неизвестный эталон → exit 2",
                       run_report(at_ref, sample, tokens_per_byte=0.25,
                                  reference="нет-такого")[0] == EXIT_INPUT))

        # Чтение токенизатора манифеста: артефакт отсутствует → fail-closed.
        manifest = root / "tokenizer-manifest.json"
        manifest.write_text(json.dumps({"tokenizer": {"file": "tokenizer.model"},
                                        "tokenizer_hash": "0" * 64}), encoding="utf-8")
        checks.append(("манифест без артефакта → exit 2",
                       run_report(at_ref, sample,
                                  tokenizer_manifest=manifest)[0] == EXIT_INPUT))

    ok = all(passed for _, passed in checks)
    for label, passed in checks:
        print(f"[selftest] {'PASS' if passed else 'FAIL'}: {label}")
    print(
        f"[selftest] {'PASS' if ok else 'FAIL'}: конверсия и эталоны точны, "
        "окно детерминировано, вердикт |ΔBPB|≤10%, вход fail-closed"
    )
    return EXIT_OK if ok else EXIT_FAIL


if __name__ == "__main__":
    raise SystemExit(main())
