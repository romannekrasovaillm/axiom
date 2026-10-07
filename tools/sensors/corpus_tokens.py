"""S-005 — токены корпуса претрейна по шардам (дельта C2).

Что меряет: суммы токенов из манифестов ``tokens/{W,C,Q}/manifest-*.json``
(``totals.stream_tokens``) и выборочную сверку sha256 N случайных бинов (сид
записи фиксируется в ``method``). Данные недоступны (сетевой каталог
``gb10-shared`` не смонтирован) → ``unverified`` с причиной.

Запуск::

    python3 -m tools.sensors.corpus_tokens [--tokens-root PATH] [--spotcheck N] [--out-dir DIR]
    python3 -m tools.sensors.corpus_tokens --selftest
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Optional

from ._common import (
    REPO_ROOT,
    TOKENS_V2_ROOT,
    config_subject,
    emit,
    safe_exists,
    sha256_file,
)

STREAMS = ("W", "C", "Q")
SPOTCHECK_SEED = 0


def _stream_dir(root: Path, stream: str) -> Path:
    d = root / stream
    return d if d.is_dir() else root


def _read_manifests(stream_dir: Path) -> list[dict[str, Any]]:
    manifests = []
    for path in sorted(stream_dir.glob("manifest-*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            data["_path"] = str(path)
            manifests.append(data)
    return manifests


def _stream_tokens(manifests: list[dict[str, Any]]) -> Optional[int]:
    total = 0
    seen = False
    for data in manifests:
        totals = data.get("totals")
        value = None
        if isinstance(totals, dict):
            value = totals.get("stream_tokens", totals.get("tokens"))
        if value is None:
            value = data.get("stream_tokens", data.get("tokens"))
        if isinstance(value, int):
            total += value
            seen = True
    return total if seen else None


def _spotcheck(stream_dir: Path, manifests: list[dict[str, Any]], n: int, seed: int) -> tuple[Optional[bool], int]:
    """Сверяет sha256 N случайных бинов с манифестом. (ok|None, checked)."""
    bins: list[tuple[Path, str]] = []
    for data in manifests:
        shards = data.get("shards")
        if not isinstance(shards, list):
            continue
        for entry in shards:
            if not isinstance(entry, dict):
                continue
            name = entry.get("file")
            expected = entry.get("sha256")
            if not isinstance(name, str) or not isinstance(expected, str):
                continue
            candidate = Path(name)
            if not candidate.is_absolute():
                candidate = stream_dir / name
            bins.append((candidate, expected))
    if not bins:
        return None, 0
    rng = random.Random(seed)
    sample = bins if len(bins) <= n else rng.sample(bins, n)
    ok = True
    for path, expected in sample:
        got = sha256_file(path)
        if got != expected:
            ok = False
    return ok, len(sample)


def measure(
    tokens_root: str | Path = TOKENS_V2_ROOT,
    *,
    spotcheck: int = 5,
    out_dir: Optional[str | Path] = None,
    device: Optional[str] = None,
) -> dict[str, Any]:
    root = Path(tokens_root)
    subject = config_subject(dataset_ref=str(tokens_root), device=device)
    written: dict[str, Any] = {}
    if not safe_exists(root):
        reason = f"данные tokens-v2 недоступны: {tokens_root} (сетевой каталог не смонтирован)"
        for name in ("corpus_tokens_total", "corpus_tokens_w", "corpus_tokens_c",
                     "corpus_tokens_q", "bins_spotcheck_ok"):
            written[name] = emit(
                "S-005", name, None, unit="count", quality="measured",
                method="суммы манифестов tokens/", subject=subject,
                out_dir=out_dir, status="unverified", note=reason,
            )
        return written

    per_stream: dict[str, Optional[int]] = {}
    spot_results: list[bool] = []
    for stream in STREAMS:
        sd = _stream_dir(root, stream)
        manifests = _read_manifests(sd)
        per_stream[stream] = _stream_tokens(manifests)
        ok, _checked = _spotcheck(sd, manifests, spotcheck, SPOTCHECK_SEED)
        if ok is not None:
            spot_results.append(ok)

    total = None
    if all(v is not None for v in per_stream.values()):
        total = sum(v for v in per_stream.values() if v is not None)
    spot = all(spot_results) if spot_results else None

    method = (
        f"суммы totals.stream_tokens из manifest-*.json tokens/{{W,C,Q}}; "
        f"выборочная сверка sha256 {spotcheck} бинов (сид {SPOTCHECK_SEED})"
    )
    written["corpus_tokens_total"] = emit(
        "S-005", "corpus_tokens_total", total, unit="count", quality="measured",
        method=method, subject=subject, out_dir=out_dir,
        status="ok" if total is not None else "unverified",
        note="" if total is not None else "не все шарды отдали totals",
    )
    for stream in STREAMS:
        written[f"corpus_tokens_{stream.lower()}"] = emit(
            "S-005", f"corpus_tokens_{stream.lower()}", per_stream[stream], unit="count",
            quality="measured", method=method, subject=subject, out_dir=out_dir,
            status="ok" if per_stream[stream] is not None else "unverified",
            note="" if per_stream[stream] is not None else f"шард {stream}: манифестов нет",
        )
    written["bins_spotcheck_ok"] = emit(
        "S-005", "bins_spotcheck_ok", spot, unit="bool", quality="measured",
        method=method, subject=subject, out_dir=out_dir,
        status="ok" if spot is not None else "unverified",
        note="" if spot is not None else "бинов с sha256 в манифестах не найдено",
    )
    return written


def run_selftest() -> int:
    import tempfile

    checks: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory(prefix="s005-selftest-") as tmp:
        root = Path(tmp)
        (root / "W").mkdir()
        (root / "C").mkdir()
        (root / "Q").mkdir()
        bin_path = root / "W" / "shard-000.bin"
        bin_path.write_bytes(b"hello-tokens")
        digest = sha256_file(bin_path)
        manifest = {
            "version": "tokens/v1",
            "totals": {"stream_tokens": 1234},
            "shards": [{"file": "shard-000.bin", "sha256": digest}],
        }
        (root / "W" / "manifest-W.json").write_text(json.dumps(manifest), encoding="utf-8")
        (root / "C" / "manifest-C.json").write_text(
            json.dumps({"totals": {"stream_tokens": 10}}), encoding="utf-8"
        )
        (root / "Q" / "manifest-Q.json").write_text(
            json.dumps({"totals": {"stream_tokens": 5}}), encoding="utf-8"
        )
        written = measure(root, spotcheck=3, out_dir=tmp)
        checks.append(("total = сумма шардов", written["corpus_tokens_total"]["value"] == 1249))
        checks.append(("spotcheck зелёный", written["bins_spotcheck_ok"]["value"] is True))
        # Порча бина → spotcheck красный.
        bin_path.write_bytes(b"corrupted")
        broken = measure(root, spotcheck=3, out_dir=tmp)
        checks.append(("порча бина → spotcheck red", broken["bins_spotcheck_ok"]["value"] is False))
        # Недоступный корень → unverified.
        missing = measure(root / "nope", spotcheck=1, out_dir=tmp)
        checks.append(("нет корня → unverified", missing["corpus_tokens_total"]["status"] == "unverified"))
    ok = all(p for _, p in checks)
    for label, passed in checks:
        print(f"[selftest] {'PASS' if passed else 'FAIL'}: {label}")
    print(f"[selftest] {'PASS' if ok else 'FAIL'}: corpus_tokens")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="S-005: токены корпуса (ADR-036)")
    parser.add_argument("--tokens-root", default=str(TOKENS_V2_ROOT))
    parser.add_argument("--spotcheck", type=int, default=5)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--selftest", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.selftest:
        return run_selftest()
    written = measure(args.tokens_root, spotcheck=args.spotcheck, out_dir=args.out_dir, device=args.device)
    for name, rec in written.items():
        print(f"S-005 {name} = {rec['value']} [{rec['status']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
