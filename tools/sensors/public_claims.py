"""S-030 — публичная карточка HF сходится с фактами (производный факт, дельта C6).

Сравнивает числа карточки HF (loss, токены, ток/с) с фактами S-012/S-005.
Источник карточки — staging-копия из ``evidence/hf-publications.md`` или путь
аргументом; недоступна → ``unverified``. Расхождения перечисляются списком.

Запуск::

    python3 -m tools.sensors.public_claims [--card evidence/hf-publications.md] [--out-dir DIR]
    python3 -m tools.sensors.public_claims --selftest
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Optional

from ._common import REPO_ROOT, config_subject, emit
from ._derive import fact_ref

METHOD = "сверка чисел карточки HF (loss, токены, ток/с) с фактами S-012/S-005"
DEFAULT_CARD = REPO_ROOT / "evidence" / "hf-publications.md"
_NUM_RE = re.compile(r"\d[\d._]*")


def _variants(value: Any) -> list[str]:
    variants: list[str] = []
    if isinstance(value, float):
        for precision in (0, 1, 2, 3):
            variants.append(f"{value:.{precision}f}")
        variants.append(f"{value:g}")
    elif isinstance(value, int):
        variants.append(str(value))
        variants.append(f"{value/1e9:.2f}B")
        variants.append(f"{value/1e9:.1f}B")
    else:
        variants.append(str(value))
    return [v for v in dict.fromkeys(variants) if v]


def measure(
    *,
    card_path: Optional[str | Path] = None,
    out_dir: Optional[str | Path] = None,
    device: Optional[str] = None,
) -> dict[str, Any]:
    card = Path(card_path) if card_path else DEFAULT_CARD
    facts = {
        "S-012:tok_s_median_window": fact_ref("S-012", "tok_s_median_window", out_dir=out_dir),
        "S-012:loss_median_window": fact_ref("S-012", "loss_median_window", out_dir=out_dir),
        "S-005:corpus_tokens_total": fact_ref("S-005", "corpus_tokens_total", out_dir=out_dir),
    }
    usable = {name: ref for name, ref in facts.items() if ref and ref.get("value") is not None}
    inputs = [ref for ref in facts.values() if ref is not None]

    matches: Optional[bool] = None
    mismatches: Optional[list[str]] = None
    note = ""
    if not card.is_file():
        note = f"карточка недоступна: {card}"
    elif not usable:
        note = "нет фактов с числами (S-012/S-005) — сверять не с чем"
    else:
        text = card.read_text(encoding="utf-8", errors="replace")
        mismatches = []
        for name, ref in usable.items():
            if not any(v in text for v in _variants(ref["value"])):
                mismatches.append(name)
        matches = not mismatches
        note = "" if matches else "расхождения: " + ", ".join(mismatches)

    subject = config_subject(dataset_ref=str(card) if card.is_file() else None, device=device)
    written: dict[str, Any] = {}
    written["public_card_matches_facts"] = emit(
        "S-030", "public_card_matches_facts", matches, unit="bool", quality="derived",
        method=METHOD, subject=subject, out_dir=out_dir, inputs=inputs,
        status="ok" if matches is not None else "unverified", note=note,
    )
    written["public_card_mismatches"] = emit(
        "S-030", "public_card_mismatches", mismatches, unit="list", quality="derived",
        method=METHOD, subject=subject, out_dir=out_dir, inputs=inputs,
        status="ok" if mismatches is not None else "unverified",
        note="" if mismatches is not None else note,
    )
    return written


def run_selftest() -> int:
    import tempfile

    from .fact import write_fact
    from .subject import build_subject

    checks: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory(prefix="s030-selftest-") as tmp:
        subject = build_subject(repo_root=Path(tmp), git_sha="a" * 40, dirty=False, device="cpu")
        write_fact("S-012", "tok_s_median_window", 800.0, unit="tok_s", quality="wrapped",
                   method="fixture", subject=subject, out_dir=tmp)
        card = Path(tmp) / "card.md"
        card.write_text("speed 800 tok/s", encoding="utf-8")
        written = measure(card_path=card, out_dir=tmp)
        checks.append(("число карточки найдено", written["public_card_matches_facts"]["value"] is True))
        card.write_text("speed 100 tok/s", encoding="utf-8")
        broken = measure(card_path=card, out_dir=tmp)
        checks.append(("расхождение перечислено",
                       broken["public_card_matches_facts"]["value"] is False
                       and "S-012:tok_s_median_window" in broken["public_card_mismatches"]["value"]))
        missing = measure(card_path=Path(tmp) / "none.md", out_dir=tmp)
        checks.append(("нет карточки → unverified",
                       missing["public_card_matches_facts"]["status"] == "unverified"))
    ok = all(p for _, p in checks)
    for label, passed in checks:
        print(f"[selftest] {'PASS' if passed else 'FAIL'}: {label}")
    print(f"[selftest] {'PASS' if ok else 'FAIL'}: public_claims")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="S-030: сверка карточки HF с фактами (ADR-036)")
    parser.add_argument("--card", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--selftest", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.selftest:
        return run_selftest()
    written = measure(card_path=args.card, out_dir=args.out_dir, device=args.device)
    for name, rec in written.items():
        print(f"S-030 {name} = {rec['value']} [{rec['status']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
