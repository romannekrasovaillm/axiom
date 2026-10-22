"""Прибор сравнения токенизаторов на holdout (ADR-042, AD-4).

Зачем.  Кандидат v3 имеет право на существование ровно настолько, насколько
измерен: RU-выигрыш к канону v1 (``500f8023``, 2.26 симв/токен на 400 000
символов домена) должен быть не меньше, чем у работающего v2 (``16e4a532``,
4.64 симв/токен, ×2.06), а английский — не просесть (ADR-042, п.3–4).  Прибор
мерит **все** переданные артефакты одним и тем же кодом на одном и том же
holdout, поэтому сравнение честное: разница в числах — разница словарей, а не
разница процедуры.

Метрики на каждый токенизатор и каждый holdout:

* ``tokens`` — ``len(encode(text).ids)`` по всему holdout одним вызовом;
* ``chars`` / ``bytes`` — размер holdout (UTF-8);
* ``chars_per_token`` / ``bytes_per_token`` — те самые «симв/токен» из ADR-042;
* ``round_trip`` — доля документов, где ``decode(encode(doc)) == doc``
  (документ = абзац, тот же контракт, что у билдера и тренера).

EN-holdout не задан → сторона помечается ``EMPTY-PENDING`` с причиной (не
падение и не выдуманное число): образец из потока W берётся отдельно.

Режим ``--verdict`` (ADR-042 п.4) — механическое прочтение замера по трём
порогам: RU-выигрыш к базовому словарю ``--baseline-label`` (≥ +15%), деградация
EN (≤ 2%), ``round_trip`` кандидата (== 1.0).  По каждому порогу печатается
PASS/FAIL (не измеренный порог — честный SKIP, а не «прошёл»), затем итог:
``PASS`` (все пороги пройдены), ``FAIL`` (есть провал), ``PENDING`` (порог не
подтверждён).  Коды возврата: 0 / 1 / 3.  Вердикт **не переключает канон** —
переключение остаётся отдельным решением владельца; пороги задаются параметрами
(``--min-ru-gain``, ``--max-en-degradation``, ``--min-round-trip``), а не
подгоняются под результат.

CLI::

    python tools/measure_tokenizer.py \\
        --tokenizer v1=~/gb10-shared/datasets/.../tokenizer.model \\
        --tokenizer v2=~/gb10-shared/datasets/.../tokenizer-v2/tokenizer.model \\
        --tokenizer v3=~/gb10-shared/datasets/.../tokenizer-v3/tokenizer.model \\
        --ru-holdout ~/gb10-shared/datasets/holdout.ru.txt

Порядок полей JSON стабилен — файл пригоден для диффа между прогонами.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Protocol, Sequence

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from build_domain_corpus import split_documents  # noqa: E402

SCHEMA_VERSION = "tokenizer-compare-v1"

#: Схема блока вердикта: он не часть замера, а его механическое прочтение (ADR-042 п.4).
VERDICT_SCHEMA_VERSION = "tokenizer-verdict-v1"

#: Пороги допуска кандидата к переключению канона — дефолты из ADR-042 п.4.
DEFAULT_MIN_RU_GAIN = 0.15
DEFAULT_MAX_EN_DEGRADATION = 0.02
DEFAULT_MIN_ROUND_TRIP = 1.0

#: Допуск сравнения порогов: +15% на округлённых метриках (4 знака) не должен
#: становиться FAIL из-за двоичной арифметики (0.6/4.0 = 0.14999999999999997).
VERDICT_EPS = 1e-9

VERDICT_PASS = "PASS"
VERDICT_FAIL = "FAIL"
VERDICT_SKIP = "SKIP"
VERDICT_PENDING = "PENDING"
#: Коды возврата вердикта: PASS — 0, FAIL — 1, PENDING (порог не подтверждён) — 3.
VERDICT_EXIT = {VERDICT_PASS: 0, VERDICT_FAIL: 1, VERDICT_PENDING: 3}

#: Куда по умолчанию кладётся отчёт: рядом с evidence кандидата v3 (ADR-042).
DEFAULT_OUT = "evidence/tokenizer-v3/tokenizer-compare.json"

EN_PENDING_REASON = (
    "--en-holdout не задан: англоязычная сторона (поток W) не измерена, "
    "образец подставляется отдельно"
)


class TokenizerLike(Protocol):
    """Минимум, который нужен прибору: ``encode(text).ids`` и ``decode(ids)``."""

    def encode(self, text: str) -> Any: ...
    def decode(self, ids: list[int]) -> str: ...


class MeasureError(Exception):
    """Отказ прибора: нет артефакта, пустой holdout, битая пара ``метка=путь``."""


# --------------------------------------------------------------------------- #
# Метрики
# --------------------------------------------------------------------------- #


def measure_holdout(
    tokenizer: TokenizerLike, text: str, docs: Sequence[str]
) -> dict[str, Any]:
    """Метрики одного токенизатора на одном holdout.

    ``tokens`` считается по всему тексту (как в замере ADR-042: 400 000 символов
    → одно число токенов), ``round_trip`` — по документам: доля восстановленных
    документов чувствительнее к битому decode, чем одна общая проверка.
    """
    tokens = len(tokenizer.encode(text).ids)
    chars = len(text)
    bytes_ = len(text.encode("utf-8"))
    matched = sum(1 for doc in docs if tokenizer.decode(tokenizer.encode(doc).ids) == doc)
    doc_count = len(docs)
    return {
        "tokens": tokens,
        "chars": chars,
        "bytes": bytes_,
        "chars_per_token": round(chars / tokens, 4) if tokens else None,
        "bytes_per_token": round(bytes_ / tokens, 4) if tokens else None,
        "round_trip": {
            "docs": doc_count,
            "matched": matched,
            "rate": round(matched / doc_count, 6) if doc_count else None,
        },
    }


def load_tokenizer(path: Path) -> Any:
    """Загрузить ``tokenizer.model`` (byte-level BPE / SentencePiece) через tokenizers."""
    if not path.is_file():
        raise MeasureError(f"файл токенизатора не найден: {path}")
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:  # pragma: no cover - окружение без tokenizers
        raise MeasureError(f"не установлена библиотека tokenizers: {exc}") from exc
    try:
        return Tokenizer.from_file(str(path))
    except Exception as exc:  # noqa: BLE001 - сообщение важнее типа
        raise MeasureError(f"не удалось прочитать токенизатор {path}: {exc}") from exc


def parse_tokenizer(raw: str) -> tuple[str, Path]:
    """``МЕТКА=ПУТЬ`` (или просто ``ПУТЬ`` → метка по имени каталога)."""
    label, sep, path_raw = raw.partition("=")
    if not sep:
        label, path_raw = "", raw
    path = Path(os.path.expanduser(path_raw.strip()))
    label = label.strip() or path.parent.name or path.stem
    if not str(path_raw).strip():
        raise MeasureError(f"пустой путь токенизатора: {raw!r}")
    return label, path


def read_holdout(path: Path, kind: str) -> str:
    if not path.is_file():
        raise MeasureError(f"{kind}-holdout не найден: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        raise MeasureError(f"{kind}-holdout пуст: {path}")
    return text


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# --------------------------------------------------------------------------- #
# Отчёт
# --------------------------------------------------------------------------- #


def build_report(
    pairs: Sequence[tuple[str, Path]],
    *,
    ru_path: Path,
    ru_text: str,
    en_path: Path | None,
    en_text: str | None,
) -> dict[str, Any]:
    ru_docs = list(split_documents(ru_text, 1))
    if not ru_docs:
        raise MeasureError(f"RU-holdout не содержит документов: {ru_path}")
    en_docs = list(split_documents(en_text, 1)) if en_text is not None else []

    holdouts: dict[str, Any] = {
        "ru": {
            "path": str(ru_path),
            "chars": len(ru_text),
            "bytes": len(ru_text.encode("utf-8")),
            "docs": len(ru_docs),
        }
    }
    holdouts["en"] = (
        {
            "path": str(en_path),
            "chars": len(en_text or ""),
            "bytes": len((en_text or "").encode("utf-8")),
            "docs": len(en_docs),
        }
        if en_text is not None
        else {"status": "EMPTY-PENDING", "reason": EN_PENDING_REASON}
    )

    tokenizers: list[dict[str, Any]] = []
    for label, path in pairs:
        tokenizer = load_tokenizer(path)
        entry: dict[str, Any] = {
            "label": label,
            "path": str(path),
            "sha256": file_sha256(path),
            "holdouts": {"ru": measure_holdout(tokenizer, ru_text, ru_docs)},
        }
        entry["holdouts"]["en"] = (
            measure_holdout(tokenizer, en_text, en_docs)
            if en_text is not None
            else {"status": "EMPTY-PENDING", "reason": EN_PENDING_REASON}
        )
        tokenizers.append(entry)

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "holdouts": holdouts,
        "tokenizers": tokenizers,
    }


# --------------------------------------------------------------------------- #
# Механический вердикт (--verdict)
# --------------------------------------------------------------------------- #


def _side(entry: dict[str, Any], side: str) -> dict[str, Any]:
    return entry["holdouts"].get(side) or {}


def _is_pending(metrics: dict[str, Any]) -> bool:
    return metrics.get("status") == "EMPTY-PENDING"


VERDICT_NOTE = (
    "вердикт механический (ADR-042 п.4): PASS/FAIL по порогам, канон им не "
    "переключается — переключение остаётся отдельным решением владельца"
)


def compute_verdict(
    report: dict[str, Any],
    *,
    baseline_label: str,
    candidate_label: str | None = None,
    min_ru_gain: float = DEFAULT_MIN_RU_GAIN,
    max_en_degradation: float = DEFAULT_MAX_EN_DEGRADATION,
    min_round_trip: float = DEFAULT_MIN_ROUND_TRIP,
) -> dict[str, Any]:
    """Механический вердикт по трём порогам ADR-042 п.4 — без судьи и без модели.

    * ``ru_gain`` — ``(cand − base) / base`` по ``chars_per_token`` на RU-holdout
      (PASS при ``≥ min_ru_gain``);
    * ``en_degradation`` — ``(base − cand) / base`` на EN-holdout (PASS при
      ``≤ max_en_degradation``); если EN не измерен — ``SKIP`` (честно: порог не
      подтверждён, а не «прошёл»);
    * ``round_trip`` — доля восстановленных документов кандидата (PASS при
      ``≥ min_round_trip``).

    Итог: ``FAIL``, если есть FAIL; ``PENDING``, если есть SKIP и нет FAIL;
    иначе ``PASS``.  Пороги — параметры: их смена обосновывается решением, а не
    подгонкой под результат.
    """
    entries = {entry["label"]: entry for entry in report["tokenizers"]}
    if baseline_label not in entries:
        raise MeasureError(
            f"базовый токенизатор {baseline_label!r} не передан прибору: "
            f"есть {sorted(entries)}"
        )
    if candidate_label is None:
        candidates = [entry["label"] for entry in report["tokenizers"] if entry["label"] != baseline_label]
        if not candidates:
            raise MeasureError("вердикту нужен кандидат: кроме базового словаря ничего не передано")
        candidate_label = candidates[-1]
    if candidate_label == baseline_label:
        raise MeasureError(
            f"кандидат и база — один словарь ({candidate_label!r}): сравнивать нечего"
        )
    if candidate_label not in entries:
        raise MeasureError(
            f"токенизатор кандидата {candidate_label!r} не передан прибору: есть {sorted(entries)}"
        )

    base = entries[baseline_label]
    candidate = entries[candidate_label]
    checks: list[dict[str, Any]] = []

    base_ru = _side(base, "ru").get("chars_per_token")
    cand_ru = _side(candidate, "ru").get("chars_per_token")
    if not base_ru or cand_ru is None:
        checks.append({
            "id": "ru_gain", "status": VERDICT_SKIP, "metric": "chars_per_token",
            "baseline": base_ru, "candidate": cand_ru, "value": None,
            "threshold": min_ru_gain, "reason": "RU-holdout не измерен",
        })
    else:
        gain = (cand_ru - base_ru) / base_ru
        checks.append({
            "id": "ru_gain", "status": VERDICT_PASS if gain + VERDICT_EPS >= min_ru_gain else VERDICT_FAIL,
            "metric": "chars_per_token", "baseline": base_ru, "candidate": cand_ru,
            "value": round(gain, 6), "threshold": min_ru_gain,
            "basis": "RU: (кандидат − база) / база",
        })

    base_en_metrics = _side(base, "en")
    cand_en_metrics = _side(candidate, "en")
    if _is_pending(base_en_metrics) or _is_pending(cand_en_metrics):
        checks.append({
            "id": "en_degradation", "status": VERDICT_SKIP, "metric": "chars_per_token",
            "baseline": None, "candidate": None, "value": None,
            "threshold": max_en_degradation,
            "reason": (base_en_metrics.get("reason") or cand_en_metrics.get("reason") or EN_PENDING_REASON),
        })
    else:
        base_en = base_en_metrics.get("chars_per_token")
        cand_en = cand_en_metrics.get("chars_per_token")
        if not base_en or cand_en is None:
            checks.append({
                "id": "en_degradation", "status": VERDICT_SKIP, "metric": "chars_per_token",
                "baseline": base_en, "candidate": cand_en, "value": None,
                "threshold": max_en_degradation, "reason": "EN-holdout не измерен",
            })
        else:
            degradation = (base_en - cand_en) / base_en
            checks.append({
                "id": "en_degradation",
                "status": VERDICT_PASS if degradation <= max_en_degradation + VERDICT_EPS else VERDICT_FAIL,
                "metric": "chars_per_token", "baseline": base_en, "candidate": cand_en,
                "value": round(degradation, 6), "threshold": max_en_degradation,
                "basis": "EN: (база − кандидат) / база",
            })

    round_trip = _side(candidate, "ru").get("round_trip") or {}
    rate = round_trip.get("rate")
    if rate is None:
        checks.append({
            "id": "round_trip", "status": VERDICT_SKIP, "metric": "round_trip.rate",
            "baseline": None, "candidate": None, "value": None,
            "threshold": min_round_trip, "reason": "round_trip кандидата не измерен",
        })
    else:
        checks.append({
            "id": "round_trip",
            "status": VERDICT_PASS if rate + VERDICT_EPS >= min_round_trip else VERDICT_FAIL,
            "metric": "round_trip.rate", "baseline": None, "candidate": rate,
            "value": round(rate, 6), "threshold": min_round_trip,
        })

    statuses = {check["status"] for check in checks}
    if VERDICT_FAIL in statuses:
        status = VERDICT_FAIL
    elif VERDICT_SKIP in statuses:
        status = VERDICT_PENDING
    else:
        status = VERDICT_PASS
    return {
        "schema_version": VERDICT_SCHEMA_VERSION,
        "baseline": baseline_label,
        "candidate": candidate_label,
        "thresholds": {
            "min_ru_gain": min_ru_gain,
            "max_en_degradation": max_en_degradation,
            "min_round_trip": min_round_trip,
        },
        "checks": checks,
        "status": status,
        "note": VERDICT_NOTE,
    }


def format_verdict(verdict: dict[str, Any]) -> str:
    """Человекочитаемые строки вердикта: по каждому порогу — статус и число."""
    lines = [f"[verdict] база {verdict['baseline']} → кандидат {verdict['candidate']}"]
    for check in verdict["checks"]:
        head = f"[verdict] {check['id']}: {check['status']}"
        if check["status"] == VERDICT_SKIP:
            lines.append(f"{head} — не проверено: {check.get('reason', '')}")
            continue
        value = check["value"]
        threshold = check["threshold"]
        if check["id"] == "ru_gain":
            lines.append(f"{head}  {value * 100:+.2f}% (порог ≥ {threshold * 100:+.2f}%, "
                         f"симв/ток {check['baseline']} → {check['candidate']})")
        elif check["id"] == "en_degradation":
            lines.append(f"{head}  {value * 100:+.2f}% (порог ≤ {threshold * 100:.2f}%, "
                         f"симв/ток {check['baseline']} → {check['candidate']})")
        else:
            lines.append(f"{head}  {value:.6f} (порог ≥ {threshold:.6f})")
    lines.append(f"[verdict] ИТОГ: {verdict['status']} — {verdict['note']}")
    return "\n".join(lines)


def format_table(report: dict[str, Any]) -> str:
    """Плоская таблица в stdout: метки в порядке CLI, дельта к первому столбцу."""
    header = f"{'метка':<16} {'RU токенов':>12} {'RU симв/ток':>12} {'RU round-trip':>14} {'EN':>14}"
    lines = [header, "-" * len(header)]
    for entry in report["tokenizers"]:
        ru = entry["holdouts"]["ru"]
        rt = ru["round_trip"]
        en = entry["holdouts"]["en"]
        en_cell = "EMPTY-PENDING" if en.get("status") == "EMPTY-PENDING" else f"{en['chars_per_token']:.4f}"
        lines.append(
            f"{entry['label']:<16} {ru['tokens']:>12} {ru['chars_per_token']:>12.4f} "
            f"{rt['rate'] * 100:>13.4f}% {en_cell:>14}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Сравнить токенизаторы на holdout: симв/токен, байт/токен, round-trip (ADR-042)"
    )
    parser.add_argument(
        "--tokenizer", action="append", default=None, metavar="МЕТКА=ПУТЬ",
        help="артефакт tokenizer.model (повторяемый; без метки — имя каталога)",
    )
    parser.add_argument("--ru-holdout", default=None, help="файл русского доменного текста")
    parser.add_argument("--en-holdout", default=None, help="англоязычный образец (необязателен)")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"путь JSON-отчёта (по умолчанию {DEFAULT_OUT})")
    parser.add_argument(
        "--verdict", action="store_true",
        help="механический вердикт по трём порогам ADR-042 п.4 (PASS/FAIL + итог); "
             "канон не переключается — решение остаётся за владельцем",
    )
    parser.add_argument("--baseline-label", default=None,
                        help="метка базового словаря для --verdict (например, работающий v2)")
    parser.add_argument("--candidate-label", default=None,
                        help="метка кандидата для --verdict (по умолчанию — последний, кроме базы)")
    parser.add_argument("--min-ru-gain", type=float, default=DEFAULT_MIN_RU_GAIN,
                        help=f"порог RU-выигрыша к базе (дефолт {DEFAULT_MIN_RU_GAIN} из ADR-042 п.4)")
    parser.add_argument("--max-en-degradation", type=float, default=DEFAULT_MAX_EN_DEGRADATION,
                        help=f"порог деградации EN (дефолт {DEFAULT_MAX_EN_DEGRADATION} из ADR-042 п.4)")
    parser.add_argument("--min-round-trip", type=float, default=DEFAULT_MIN_ROUND_TRIP,
                        help=f"порог доли round_trip (дефолт {DEFAULT_MIN_ROUND_TRIP} из ADR-042 п.4)")
    return parser


def cmd_measure(args: argparse.Namespace) -> int:
    if not args.tokenizer:
        raise MeasureError("не задан ни один --tokenizer МЕТКА=ПУТЬ")
    if not args.ru_holdout:
        raise MeasureError("--ru-holdout обязателен: сравнивать не на чем")

    pairs: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for raw in args.tokenizer:
        label, path = parse_tokenizer(raw)
        if label in seen:
            raise MeasureError(f"метка токенизатора повторяется: {label!r} — метки должны быть уникальны")
        seen.add(label)
        pairs.append((label, path))

    ru_path = Path(os.path.expanduser(args.ru_holdout))
    ru_text = read_holdout(ru_path, "RU")
    en_path = Path(os.path.expanduser(args.en_holdout)) if args.en_holdout else None
    en_text = read_holdout(en_path, "EN") if en_path is not None else None

    report = build_report(pairs, ru_path=ru_path, ru_text=ru_text, en_path=en_path, en_text=en_text)

    verdict: dict[str, Any] | None = None
    if args.verdict:
        if not args.baseline_label:
            raise MeasureError(
                "--verdict требует --baseline-label: выигрыш считается к базовому "
                "словарю (например, работающему v2), а не к пустоте (ADR-042 п.4)"
            )
        verdict = compute_verdict(
            report,
            baseline_label=args.baseline_label,
            candidate_label=args.candidate_label,
            min_ru_gain=args.min_ru_gain,
            max_en_degradation=args.max_en_degradation,
            min_round_trip=args.min_round_trip,
        )
        report["verdict"] = verdict  # вердикт — режим: без флага форма отчёта не меняется

    out = Path(os.path.expanduser(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    print(format_table(report), flush=True)
    print(f"[measure] RU-holdout {ru_path} ({report['holdouts']['ru']['chars']} символов, "
          f"{report['holdouts']['ru']['docs']} документов)", flush=True)
    if report["holdouts"]["en"].get("status") == "EMPTY-PENDING":
        print(f"[measure] EN: EMPTY-PENDING — {EN_PENDING_REASON}", flush=True)
    if verdict is not None:
        print(format_verdict(verdict), flush=True)
    print(f"[measure] отчёт {out}", flush=True)
    return VERDICT_EXIT.get(verdict["status"], 0) if verdict is not None else 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return cmd_measure(args)
    except MeasureError as exc:
        print(f"[measure] ОТКАЗ: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
