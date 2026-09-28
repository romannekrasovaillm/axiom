"""CLI сборки датасета агентных эпизодов (ADR-020, дельта-1, компонент E).

Порядок ступеней строгий и зашит в поток обработки::

    deny-list → скраб → эпизодизация → верификация → дедуп → запись jsonl

Каждая сессия читается потоково (``iter_episodes``), в память поднимается не
больше одной сессии, а из эпизодов — только подписи дедупа. Отчёт — числовой:
счётчики, правила и длительности; ни содержимого сессий, ни значений секретов.

Пример::

    python -m axiom_ds.build \\
        --source ~/.claude/projects \\
        --out ~/gb10-shared/datasets/axiom-domain-ds-v1/episodes-v1.jsonl \\
        --report ~/gb10-shared/datasets/axiom-domain-ds-v1/episodes-v1-report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):  # запуск файлом: python tools/axiom_ds/build.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from axiom_ds import dedup as dedup_mod
    from axiom_ds import episodes as ep_mod
    from axiom_ds import harness as harness_mod
    from axiom_ds import scrub as scrub_mod
    from axiom_ds import verify as verify_mod
    from prep_pretrain import common as pp_common
else:
    from . import dedup as dedup_mod
    from . import episodes as ep_mod
    from . import harness as harness_mod
    from . import scrub as scrub_mod
    from . import verify as verify_mod
    from prep_pretrain import common as pp_common

PIPELINE_VERSION = "axiom-ds-episodes/1"

DEFAULT_SOURCES = ("~/.claude/projects",)

#: Объявленный в задаче источник механической верификации. Отчёты привязываются
#: к эпизоду только по времени и только при однозначной паре (см. harness.py).
DEFAULT_RESULTS_GLOBS = ("~/.arch-ml/reports/harness/*/result.json",)

#: Куда запрещено писать выход датасета (приватные данные остаются в контуре).
ALLOWED_OUTPUT_ROOTS = ("~/gb10-shared", "/tmp", "/var/tmp")


@dataclass
class BuildCounters:
    sessions_total: int = 0
    sessions_processed: int = 0
    sessions_denied: int = 0
    sessions_failed: int = 0
    denied_dirs_pruned: int = 0
    events: int = 0
    parse_errors: int = 0
    oversized_lines: int = 0
    episodes_total: int = 0
    episodes_written: int = 0
    #: Объём записанного корпуса: байты jsonl, символы текста ходов и оценка
    #: токенов той же мерой, что CPT-компонент (``chars // 4``, ADR-021) —
    #: иначе доли компонент в карточке были бы несравнимы.
    bytes_written: int = 0
    chars: int = 0
    approx_tokens: int = 0
    by_class: dict = field(default_factory=dict)
    by_class_written: dict = field(default_factory=dict)
    by_evidence: dict = field(default_factory=dict)
    harness_matched: int = 0
    turns: dict = field(default_factory=dict)

    def bump(self, counter: str, amount: int = 1) -> None:
        setattr(self, counter, getattr(self, counter) + amount)

    def bump_map(self, counter: str, key: str, amount: int = 1) -> None:
        mapping = getattr(self, counter)
        mapping[key] = mapping.get(key, 0) + amount


# --------------------------------------------------------------------------- #
# Обход источников
# --------------------------------------------------------------------------- #


def discover_sessions(
    sources: Sequence[str], pattern: str = "*.jsonl", denied_counter: list | None = None
) -> list[Path]:
    """Файлы сессий под источниками, старейшие первыми; запретные — не читаются.

    Запретные каталоги не обходятся вовсе (``os.walk`` с обрезкой ``dirnames``);
    их содержимое только пересчитывается по именам, чтобы отчёт знал объём
    пропущенного, — ни один файл оттуда не открывается.

    ``denied_counter`` получает ``[файлов_пропущено_по_deny_list, каталогов_обрезано]``;
    в счёт попадают все файлы запретных каталогов и файлы-ключи (``*.pem``,
    ``*.key``, ``id_rsa``) независимо от маски сессий.
    """
    kept: list[Path] = []
    denied = 0
    pruned_dirs = 0

    for source in sources:
        root = Path(os.path.expanduser(source))
        if not root.exists():
            continue
        if scrub_mod.is_denied_path(root):
            denied += sum(1 for _ in root.rglob("*") if _.is_file()) if root.is_dir() else 1
            pruned_dirs += 1
            continue
        for current, dirnames, filenames in os.walk(root):
            keep_dirs = []
            for name in dirnames:
                candidate = Path(current) / name
                if scrub_mod.is_denied_path(candidate):
                    denied += _count_files(candidate)
                    pruned_dirs += 1
                    continue
                keep_dirs.append(name)
            dirnames[:] = keep_dirs

            for name in filenames:
                candidate = Path(current) / name
                if scrub_mod.is_denied_path(candidate):
                    denied += 1
                    continue
                if not candidate.name.endswith(pattern.lstrip("*")):
                    continue
                kept.append(candidate)

    kept.sort(key=lambda path: (_mtime(path), str(path)))
    if denied_counter is not None:
        denied_counter.append(denied)
        denied_counter.append(pruned_dirs)
    return kept


def _count_files(directory: Path, cap: int = 100_000) -> int:
    """Число файлов в каталоге (только имена, без открытия содержимого)."""
    total = 0
    for _root, _dirs, files in os.walk(directory):
        total += len(files)
        if total >= cap:
            return cap
    return total


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


# --------------------------------------------------------------------------- #
# Прогон
# --------------------------------------------------------------------------- #


def run_build(
    source: Sequence[str] | str,
    out: str | os.PathLike[str],
    report: str | os.PathLike[str] | None = None,
    limit: int | None = None,
    results_globs: Sequence[str] = DEFAULT_RESULTS_GLOBS,
    pattern: str = "*.jsonl",
    sft_out: str | os.PathLike[str] | None = None,
    negative_out: str | os.PathLike[str] | None = None,
    progress: bool = False,
) -> dict:
    """Собрать датасет эпизодов; вернуть числовой отчёт (и записать его)."""
    sources = [source] if isinstance(source, str) else list(source)
    started = time.time()
    counters = BuildCounters()

    denied_counter: list[int] = []
    session_paths = discover_sessions(sources, pattern=pattern, denied_counter=denied_counter)
    counters.sessions_total = len(session_paths)
    if denied_counter:
        counters.sessions_denied = denied_counter[0]
        counters.denied_dirs_pruned = denied_counter[1] if len(denied_counter) > 1 else 0

    if limit is not None:
        session_paths = session_paths[: max(0, limit)]

    harness_index = harness_mod.HarnessIndex.from_globs(results_globs)

    scrub_stats = scrub_mod.ScrubStats()
    deduper = dedup_mod.Deduper()

    out_path = Path(os.path.expanduser(os.fspath(out)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sft_handle = _open_optional(sft_out)
    negative_handle = _open_optional(negative_out)
    out_digest = hashlib.sha256()

    try:
        with out_path.open("w", encoding="utf-8") as out_handle:
            for position, session_path in enumerate(session_paths, start=1):
                if progress and (position % 100 == 0 or position == len(session_paths)):
                    print(
                        f"[axiom-ds] {position}/{len(session_paths)} сессий, "
                        f"{counters.episodes_total} эпизодов",
                        file=sys.stderr,
                        flush=True,
                    )
                try:
                    _process_session(
                        session_path,
                        counters,
                        scrub_stats,
                        deduper,
                        harness_index,
                        out_handle,
                        sft_handle,
                        negative_handle,
                        out_digest,
                    )
                    counters.sessions_processed += 1
                except Exception as error:  # noqa: BLE001 - сессия не роняет прогон
                    counters.sessions_failed += 1
                    print(
                        f"[axiom-ds] сессия пропущена ({type(error).__name__})",
                        file=sys.stderr,
                        flush=True,
                    )
    finally:
        for handle in (sft_handle, negative_handle):
            if handle is not None:
                handle.close()

    duration = time.time() - started
    report_payload = {
        "status": "ok",
        "pipeline": PIPELINE_VERSION,
        "sources": [os.path.expanduser(item) for item in sources],
        "out": str(out_path),
        "out_bytes": counters.bytes_written,
        "out_sha256": out_digest.hexdigest(),
        "limit": limit,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_sec": round(duration, 3),
        "sessions_total": counters.sessions_total,
        "sessions_processed": counters.sessions_processed,
        "sessions_denied": counters.sessions_denied,
        "sessions_failed": counters.sessions_failed,
        "denied_dirs_pruned": counters.denied_dirs_pruned,
        "events": counters.events,
        "parse_errors": counters.parse_errors,
        "oversized_lines": counters.oversized_lines,
        "episodes_total": counters.episodes_total,
        "episodes_written": counters.episodes_written,
        # Объём компоненты E — той же мерой, что K/D/S (ADR-021): иначе доли в
        # карточке датасета несравнимы.
        "chars": counters.chars,
        "approx_tokens": counters.approx_tokens,
        # by_class — все эпизоды до дедупа (состав корпуса); by_class_written —
        # то, что легло в jsonl (состав датасета); sft_ready — размер SFT-ядра
        # (только verified-complete), sft_partial_ready — допущенные с флагом.
        "by_class": dict(sorted(counters.by_class.items())),
        "by_class_written": dict(sorted(counters.by_class_written.items())),
        "sft_ready": counters.by_class_written.get(verify_mod.VERIFIED_COMPLETE, 0),
        "sft_partial_ready": counters.by_class_written.get(verify_mod.VERIFIED_PARTIAL, 0),
        "negative_ready": counters.by_class_written.get(verify_mod.VERIFIED_FAILED, 0),
        "by_evidence": dict(sorted(counters.by_evidence.items())),
        "turns": dict(sorted(counters.turns.items())),
        "dedup": deduper.stats.to_dict(),
        "redactions": scrub_stats.to_dict(),
        "harness": {**harness_index.to_dict(), "patterns": list(results_globs)},
    }

    if report is not None:
        report_path = Path(os.path.expanduser(os.fspath(report)))
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report_payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
    return report_payload


def _open_optional(path):
    if path is None:
        return None
    target = Path(os.path.expanduser(os.fspath(path)))
    target.parent.mkdir(parents=True, exist_ok=True)
    return target.open("w", encoding="utf-8")


def _process_session(
    session_path: Path,
    counters: BuildCounters,
    scrub_stats: scrub_mod.ScrubStats,
    deduper: dedup_mod.Deduper,
    harness_index: harness_mod.HarnessIndex,
    out_handle,
    sft_handle,
    negative_handle,
    out_digest=None,
) -> None:
    session_stats = ep_mod.SessionStats()
    for episode in ep_mod.iter_episodes(session_path, session_stats, scrub_stats):
        counters.episodes_total += 1
        harness_status = harness_index.match(episode.started_at, episode.ended_at)
        if harness_status is not None:
            counters.harness_matched += 1
        cls = verify_mod.apply_class(episode, harness_status)
        counters.bump_map("by_class", cls)
        counters.bump_map("by_evidence", episode.evidence)
        for turn in episode.turns:
            counters.bump_map("turns", turn.kind)

        if not deduper.add(episode):
            continue

        row = episode.to_dict()
        line = json.dumps(row, ensure_ascii=False) + "\n"
        payload = line.encode("utf-8")
        text = episode.text()
        counters.episodes_written += 1
        counters.bump_map("by_class_written", cls)
        counters.bytes_written += len(payload)
        counters.chars += len(text)
        counters.approx_tokens += pp_common.approx_tokens(text)
        out_handle.write(line)
        if out_digest is not None:
            out_digest.update(payload)
        # SFT-компонент: полные (verified-complete) и частичные с механическим
        # подтверждением (verified-partial, флаг verification=partial-green).
        if cls in verify_mod.SFT_CLASSES and sft_handle is not None:
            sft_handle.write(line)
        elif cls == verify_mod.VERIFIED_FAILED and negative_handle is not None:
            negative_handle.write(line)

    counters.events += session_stats.events
    counters.parse_errors += session_stats.parse_errors
    counters.oversized_lines += session_stats.oversized_lines


def check_output_path(path: str | os.PathLike[str]) -> str | None:
    """Выход датасета — только gb10-shared или /tmp (C-032/C-033, AD-6)."""
    resolved = str(Path(os.path.expanduser(os.fspath(path))).absolute())
    for root in ALLOWED_OUTPUT_ROOTS:
        expanded = os.path.expanduser(root)
        if resolved == expanded or resolved.startswith(expanded.rstrip("/") + "/"):
            return None
    return (
        f"выход {resolved} вне разрешённых корней {ALLOWED_OUTPUT_ROOTS}: "
        "датасет живёт на gb10-shared или /tmp (C-033, AD-6)"
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="axiom_ds.build",
        description="Сборка датасета агентных эпизодов axiom-domain-ds-v1 (ADR-020, дельта-1)",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="корень с сессиями (можно несколько раз); по умолчанию ~/.claude/projects",
    )
    parser.add_argument("--out", required=True, help="jsonl с эпизодами (по одному в строке)")
    parser.add_argument("--report", default=None, help="json с числовым отчётом")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="сколько сессий взять (проба); 0 — пустой валидный выход + отчёт",
    )
    parser.add_argument(
        "--results-glob",
        action="append",
        default=None,
        help="glob отчётов турникета (result.json / *.log); можно несколько раз",
    )
    parser.add_argument("--sft-out", default=None, help="только verified-complete (SFT-ядро)")
    parser.add_argument("--negative-out", default=None, help="только verified-failed (negative-пул)")
    parser.add_argument("--pattern", default="*.jsonl", help="маска файлов сессий")
    parser.add_argument("--progress", action="store_true", help="печатать прогресс в stderr")
    parser.add_argument(
        "--allow-any-out",
        action="store_true",
        help="разрешить выход вне gb10-shared//tmp (нарушает C-033/AD-6)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    for path in (args.out, args.sft_out, args.negative_out):
        if path is None or args.allow_any_out:
            continue
        problem = check_output_path(path)
        if problem:
            print(f"[axiom-ds] отказ: {problem}", file=sys.stderr)
            return 2

    sources = args.source or list(DEFAULT_SOURCES)
    results_globs = tuple(args.results_glob) if args.results_glob else DEFAULT_RESULTS_GLOBS

    report = run_build(
        source=sources,
        out=args.out,
        report=args.report,
        limit=args.limit,
        results_globs=results_globs,
        pattern=args.pattern,
        sft_out=args.sft_out,
        negative_out=args.negative_out,
        progress=args.progress,
    )
    if args.report:
        print(
            "[axiom-ds] сессий: {sessions_processed}/{sessions_total}, эпизодов: "
            "{episodes_total}, записано: {episodes_written}, дублей снято: {dups}, "
            "redactions: {red}".format(
                dups=report["dedup"]["exact"] + report["dedup"]["near"],
                red=report["redactions"]["total"],
                **report,
            ),
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
