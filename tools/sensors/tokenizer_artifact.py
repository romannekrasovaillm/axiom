"""S-004 — артефакт токенизатора (дельта C2).

Что меряет: полный sha256 файла канонического токенизатора (ADR-004-амендмент,
`tokens-v2/tokenizer/tokenizer.model`), размер словаря из самого артефакта и
совпадение с пином `net/config.json:tokenizer_hash`. Файл недоступен →
`unverified` с причиной (никогда подстановка).

Запуск::

    python3 -m tools.sensors.tokenizer_artifact [--tokenizer PATH] [--out-dir DIR]
    python3 -m tools.sensors.tokenizer_artifact --selftest
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from ._common import (
    REPO_ROOT,
    TOKENIZER_ARTIFACT,
    config_subject,
    emit,
    load_config_json,
    safe_exists,
    sha256_file,
)

METHOD = "sha256 файла артефакта токенизатора + чтение словаря из JSON артефакта"


def _vocab_size(path: Path) -> Optional[int]:
    """Размер словаря из JSON артефакта (HF tokenizers): len(model.vocab)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    model = data.get("model") if isinstance(data, dict) else None
    if isinstance(model, dict):
        vocab = model.get("vocab")
        if isinstance(vocab, dict):
            return len(vocab)
        if isinstance(vocab, list):
            return len(vocab)
    if isinstance(data, dict) and isinstance(data.get("vocab_size"), int):
        return int(data["vocab_size"])
    return None


def measure(
    tokenizer_path: str | Path = TOKENIZER_ARTIFACT,
    *,
    config_path: str | Path | None = None,
    out_dir: Optional[str | Path] = None,
    device: Optional[str] = None,
) -> dict[str, Any]:
    config_path = config_path or (REPO_ROOT / "net" / "config.json")
    try:
        declared_hash = load_config_json(config_path).get("tokenizer_hash")
    except OSError:
        declared_hash = None
    subject = config_subject(
        config_path=config_path,
        tokenizer_path=None,  # хеш токенизатора пишется в факт, не в предмет
        device=device,
    )
    written: dict[str, Any] = {}

    if not safe_exists(tokenizer_path):
        reason = f"артефакт токенизатора недоступен: {tokenizer_path}"
        for name, unit in (
            ("tokenizer_sha256", "sha256"),
            ("tokenizer_vocab_size", "count"),
            ("matches_config_hash", "bool"),
        ):
            written[name] = emit(
                "S-004", name, None, unit=unit, quality="measured", method=METHOD,
                subject=subject, out_dir=out_dir, status="unverified", note=reason,
            )
        return written

    digest = sha256_file(tokenizer_path)
    vocab = _vocab_size(Path(tokenizer_path))
    matches = (digest == declared_hash) if (digest and declared_hash) else None

    written["tokenizer_sha256"] = emit(
        "S-004", "tokenizer_sha256", digest, unit="sha256", quality="measured",
        method=METHOD, subject=subject, out_dir=out_dir,
        status="ok" if digest else "unverified",
        note="" if digest else "файл не прочитался",
    )
    written["tokenizer_vocab_size"] = emit(
        "S-004", "tokenizer_vocab_size", vocab, unit="count", quality="measured",
        method=METHOD, subject=subject, out_dir=out_dir,
        status="ok" if vocab is not None else "unverified",
        note="" if vocab is not None else "словарь не прочитался из артефакта",
    )
    written["matches_config_hash"] = emit(
        "S-004", "matches_config_hash", matches, unit="bool", quality="measured",
        method=METHOD, subject=subject, out_dir=out_dir,
        status="ok" if matches is not None else "unverified",
        note=(
            "" if matches is not None
            else "нет пина tokenizer_hash в конфиге или артефакт не прочитан"
        ),
    )
    return written


def run_selftest() -> int:
    import tempfile

    checks: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory(prefix="s004-selftest-") as tmp:
        art = Path(tmp) / "tokenizer.model"
        art.write_text(
            json.dumps({"model": {"vocab": {"a": 0, "b": 1, "c": 2}}}), encoding="utf-8"
        )
        written = measure(art, config_path=Path(tmp) / "noconfig.json", out_dir=tmp)
        checks.append(("sha256 посчитан", written["tokenizer_sha256"]["value"] == sha256_file(art)))
        checks.append(("словарь прочитан", written["tokenizer_vocab_size"]["value"] == 3))
        checks.append(("matches_config_hash unknown без пина", written["matches_config_hash"]["status"] == "unverified"))
        missing = measure(Path(tmp) / "nope.model", out_dir=tmp)
        checks.append(("недоступный артефакт → unverified",
                       missing["tokenizer_sha256"]["status"] == "unverified"))
    ok = all(p for _, p in checks)
    for label, passed in checks:
        print(f"[selftest] {'PASS' if passed else 'FAIL'}: {label}")
    print(f"[selftest] {'PASS' if ok else 'FAIL'}: tokenizer_artifact")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="S-004: артефакт токенизатора (ADR-036)")
    parser.add_argument("--tokenizer", default=str(TOKENIZER_ARTIFACT))
    parser.add_argument("--config", default=str(REPO_ROOT / "net" / "config.json"))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--selftest", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.selftest:
        return run_selftest()
    written = measure(args.tokenizer, config_path=args.config, out_dir=args.out_dir, device=args.device)
    for name, rec in written.items():
        print(f"S-004 {name} = {rec['value']} [{rec['status']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
