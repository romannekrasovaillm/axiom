"""CLI подготовки претрейн-датасета L3 (ADR-021): шарды W (веб) и C (код).

    python -m prep_pretrain.build prepare-w --target-tokens 17e9
    python -m prep_pretrain.build prepare-c --target-tokens 3e9 --source stack-dedup-v1
    python -m prep_pretrain.build probe --limit-mb 200
    python -m prep_pretrain.build sources
    python -m prep_pretrain.build verify-manifest --manifest <путь>

Прогон потоковый и возобновляемый: манифест пишется по мере закрытия шардов,
повторный запуск продолжает с последнего целого шарда (``--restart`` начинает
заново). Отчёты числовые: счётчики, байты, хеши, скорости, пиковый RSS.

Полная загрузка (17B+3B) этим CLI не запускается автоматически — только проба
на малом объёме; боевой прогон запускается владельцем отдельно.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

if __package__ in (None, ""):  # запуск файлом: python tools/prep_pretrain/build.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from prep_pretrain import common, fineweb, stack
else:
    from . import common, fineweb, stack

PROBE_ROOT = "/tmp/axiom-pretrain-probe"

MB = 1024 * 1024


# --------------------------------------------------------------------------- #
# Общие аргументы
# --------------------------------------------------------------------------- #


def add_shard_args(parser: argparse.ArgumentParser, default_root: str) -> None:
    parser.add_argument(
        "--out",
        default=default_root,
        help=f"каталог шардов (по умолчанию {default_root}; C-032/C-033 — только gb10-shared и /tmp)",
    )
    parser.add_argument("--shard-mb", type=float, default=common.DEFAULT_SHARD_BYTES / MB,
                        help="целевой размер сжатого шард-файла, МБ (по умолчанию 500)")
    parser.add_argument("--codec", choices=sorted(common.CODEC_EXTENSIONS),
                        default=common.DEFAULT_CODEC, help="кодек шардов")
    parser.add_argument("--level", type=int, default=3, help="уровень сжатия")
    parser.add_argument("--dedup-window", type=int, default=common.DEFAULT_DEDUP_WINDOW,
                        help="окно точного дедупа, хешей (память ∝ окну)")
    parser.add_argument("--manifest", default=None, help="путь манифеста (по умолчанию <out>/manifest-*.json)")
    parser.add_argument("--report", default=None, help="путь отчёта (по умолчанию <out>/report-*.json)")
    parser.add_argument("--restart", action="store_true",
                        help="начать заново, игнорируя курсор манифеста")
    parser.add_argument("--progress", action="store_true", help="печатать прогресс каждые 10k записей")
    parser.add_argument("--allow-any-out", action="store_true",
                        help="разрешить вывод вне gb10-shared и /tmp (по умолчанию запрещено)")


def shard_kwargs(args: argparse.Namespace) -> dict:
    return {
        "shard_bytes": int(args.shard_mb * MB),
        "codec": args.codec,
        "level": args.level,
        "dedup_window": args.dedup_window,
        "skip_records": 0,
        "restart": args.restart,
        "progress": args.progress,
        "allow_any_out": args.allow_any_out,
    }


def parse_tokens(value: str) -> int:
    """``17e9`` / ``3.5e9`` / ``17000000000`` → целое число токенов."""
    try:
        tokens = int(float(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"не число токенов: {value!r}") from exc
    if tokens <= 0:
        raise argparse.ArgumentTypeError("число токенов должно быть > 0")
    return tokens


def echo_report(report: dict, title: str) -> None:
    totals = report["totals"]
    counters = report["counters"]
    print(f"\n== {title} ==")
    print(f"  источник      : {report['source']}")
    print(f"  шардов        : {totals['shards']}")
    print(f"  байт (сжатых) : {totals['bytes']} ({common.human_bytes(totals['bytes'])})")
    print(f"  approx_tokens : {totals['approx_tokens']} ({totals['approx_tokens'] / 1e9:.3f}B)")
    print(f"  записей       : {totals['records']}")
    print(f"  прочитано     : {counters['records_read']} (дедуп отбросил {counters['dropped_dedup']})")
    print(f"  стоп          : {report['stop_reason']}")
    print(f"  время, с      : {report['seconds']}")
    print(f"  скорость      : {report['output_mb_per_s']} МБ/с на выходе, "
          f"{report['tokens_per_s']} токенов/с, {report['source_text_mb_per_s']} МБ/с исходного текста")
    if report.get("eta_hours_to_target") is not None:
        print(f"  оценка цели   : {report['eta_hours_to_target']} ч на {report['target_tokens'] / 1e9:.0f}B токенов")
    print(f"  пик RSS, МиБ  : {report['max_rss_mb']}")


# --------------------------------------------------------------------------- #
# Команды
# --------------------------------------------------------------------------- #


def cmd_prepare_w(args: argparse.Namespace) -> int:
    source = common.parse_source_spec(args.source) if args.source else dict(fineweb.DEFAULT_SOURCE)
    if getattr(args, "min_int_score", None) is not None:
        print("примечание: дополнительный edu-порог включён вручную", file=sys.stderr)
    report = fineweb.prepare_w(
        out_dir=args.out,
        target_tokens=args.target_tokens,
        source_spec=source,
        manifest_path=args.manifest,
        report_path=args.report,
        min_int_score=getattr(args, "min_int_score", None),
        **shard_kwargs(args),
    )
    echo_report(report, "шард W (веб)")
    print(f"  манифест      : {report['manifest']}")
    print(f"  отчёт         : {report['report']}")
    return 0


def cmd_prepare_c(args: argparse.Namespace) -> int:
    source_name = args.source
    source_spec = None
    check: dict | None = None
    if source_name == "auto":
        order = list(stack.AUTO_ORDER)
        if args.prefer:
            order = [args.prefer] + [name for name in order if name != args.prefer]
        for candidate in order:
            result = stack.check_source(candidate, args.languages)
            print(f"  проверка источника {candidate}: "
                  f"{'доступен' if result['ok'] else 'НЕДОСТУПЕН'} ({result.get('error', 'ok')})")
            if result["ok"]:
                source_name = candidate
                check = result
                break
        if source_name == "auto":
            print("ни один источник кода не доступен — см. `build.py sources`", file=sys.stderr)
            return 3
    elif source_name not in stack.SOURCES:
        print(f"неизвестный источник {source_name!r}; доступные: {sorted(stack.SOURCES)}",
              file=sys.stderr)
        return 2

    report = stack.prepare_c(
        out_dir=args.out,
        target_tokens=args.target_tokens,
        source_name=source_name,
        languages=args.languages,
        manifest_path=args.manifest,
        report_path=args.report,
        source_spec=source_spec,
        **shard_kwargs(args),
    )
    if check is not None:
        report["source_check"] = check
    echo_report(report, "шард C (код)")
    rules = report["rules"]
    print(f"  языки         : {', '.join(rules['languages'])}")
    print(f"  лицензии      : {', '.join(rules['allowed_licenses'])}")
    print(f"  длина файла   : {rules['min_file_bytes']}..{rules['max_file_bytes']} Б")
    print(f"  по языкам     : {rules['kept_by_language']}")
    print(f"  причины отбоя : {report['counters'].get('dropped_by_rule', {})}")
    print(f"  манифест      : {report['manifest']}")
    print(f"  отчёт         : {report['report']}")
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    """Проба на малом объёме: числовая оценка канала и темпа подготовки."""
    root = common.ensure_output_allowed(args.out, allow_any=args.allow_any_out)
    root.mkdir(parents=True, exist_ok=True)
    limit_bytes = int(args.limit_mb * MB)
    shard_bytes = int(args.shard_mb * MB)
    payload: dict = {
        "pipeline": common.PIPELINE_VERSION,
        "probe_root": str(root),
        "limit_mb": args.limit_mb,
        "shard_mb": args.shard_mb,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    print(f"проба: {args.limit_mb} МБ на шард, шард-файл {args.shard_mb} МБ, каталог {root}")
    try:
        w_report = fineweb.prepare_w(
            out_dir=root / "W",
            target_tokens=0,
            source_spec=common.parse_source_spec(args.source_w),
            max_output_bytes=limit_bytes,
            shard_bytes=shard_bytes,
            codec=args.codec,
            level=args.level,
            dedup_window=args.dedup_window,
            allow_any_out=args.allow_any_out,
            progress=args.progress,
        )
        payload["W"] = w_report
        echo_report(w_report, "проба W (веб)")
    except Exception as exc:
        payload["W"] = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"проба W не удалась: {type(exc).__name__}: {exc}", file=sys.stderr)

    source_name = args.source_c
    checks: list[dict] = []
    if source_name == "auto":
        order = list(stack.AUTO_ORDER)
        if args.prefer:
            order = [args.prefer] + [name for name in order if name != args.prefer]
        source_name = ""
        for candidate in order:
            result = stack.check_source(candidate, args.languages)
            checks.append(result)
            state = "доступен" if result["ok"] else "НЕДОСТУПЕН"
            print(f"  проверка источника {candidate}: {state} ({result.get('error', 'ok')})")
            if result["ok"]:
                source_name = candidate
                break
    payload["source_checks_c"] = checks

    if source_name:
        try:
            c_report = stack.prepare_c(
                out_dir=root / "C",
                target_tokens=0,
                source_name=source_name,
                languages=args.languages,
                max_output_bytes=limit_bytes,
                shard_bytes=shard_bytes,
                codec=args.codec,
                level=args.level,
                dedup_window=args.dedup_window,
                allow_any_out=args.allow_any_out,
                progress=args.progress,
            )
            payload["C"] = c_report
            echo_report(c_report, f"проба C (код, источник {source_name})")
            print(f"  по языкам     : {c_report['rules']['kept_by_language']}")
            print(f"  причины отбоя : {c_report['counters'].get('dropped_by_rule', {})}")
        except Exception as exc:
            payload["C"] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"проба C не удалась: {type(exc).__name__}: {exc}", file=sys.stderr)
    else:
        payload["C"] = {"error": "нет доступного источника кода (см. source_checks_c)"}
        print("проба C не выполнена: ни один источник кода не доступен", file=sys.stderr)

    payload["summary"] = summarise_probe(payload)
    report_path = common.write_report(root / "probe-report.json", payload)
    print(f"\nотчёт пробы: {report_path}")
    for key, value in payload["summary"].items():
        if isinstance(value, dict):
            print(f"  {key}: {json.dumps(value, ensure_ascii=False)}")
        else:
            print(f"  {key}: {value}")
    return 0


def summarise_probe(payload: dict) -> dict:
    """Сводка пробы: скорости по шардам и оценка длительности полной загрузки."""
    summary: dict = {}
    for key, target in (("W", fineweb.DEFAULT_TARGET_TOKENS), ("C", stack.DEFAULT_TARGET_TOKENS)):
        report = payload.get(key) or {}
        if "error" in report:
            summary[key] = {"error": report["error"]}
            continue
        summary[key] = {
            "source": report["source"],
            "shards": report["totals"]["shards"],
            "bytes": report["totals"]["bytes"],
            "approx_tokens": report["totals"]["approx_tokens"],
            "records": report["totals"]["records"],
            "seconds": report["seconds"],
            "output_mb_per_s": report["output_mb_per_s"],
            "source_text_mb_per_s": report["source_text_mb_per_s"],
            "tokens_per_s": report["tokens_per_s"],
            "max_rss_mb": report["max_rss_mb"],
            "target_tokens": target,
            "eta_hours_to_target": report.get("eta_hours_to_target"),
            "stop_reason": report["stop_reason"],
        }
    return summary


def cmd_sources(args: argparse.Namespace) -> int:
    """Проверка доступности источников кода — свидетельство для решения v2/v1."""
    results = [stack.check_source(name, args.languages) for name in stack.SOURCES]
    for result in results:
        state = "доступен" if result["ok"] else "НЕДОСТУПЕН"
        print(f"{result['source']:<20} gated={str(result['gated']):<5} {state:<10} "
              f"{result.get('error', '')[:120]}")
        print(f"{'':<20} {result['note']}")
    payload = {
        "pipeline": common.PIPELINE_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "languages": args.languages,
        "sources": results,
    }
    if args.report:
        print(f"отчёт: {common.write_report(args.report, payload)}")
    return 0 if any(r["ok"] for r in results) else 3


def cmd_verify_manifest(args: argparse.Namespace) -> int:
    manifest = json.loads(Path(os.path.expanduser(args.manifest)).read_text(encoding="utf-8"))
    out_dir = args.out or str(Path(os.path.expanduser(args.manifest)).parent)
    result = common.verify_manifest(manifest, out_dir)
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0 if not result["bad"] else 1


# --------------------------------------------------------------------------- #
# Сборка парсера
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prep_pretrain.build",
        description="Подготовка претрейн-датасета L3 (ADR-021): шарды W и C",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    w = sub.add_parser("prepare-w", help="шард W: FineWeb-Edu, ~17B токенов")
    w.add_argument("--target-tokens", type=parse_tokens, default=fineweb.DEFAULT_TARGET_TOKENS)
    w.add_argument("--source", default=None, help="hf:<repo>[:<config>] (по умолчанию sample-100BT)")
    w.add_argument("--min-int-score", type=int, default=None,
                   help="зарезервировано: доп. порог edu-оценки (по умолчанию выключен)")
    add_shard_args(w, os.path.join(common.DATASET_ROOT, fineweb.SHARD))
    w.set_defaults(func=cmd_prepare_w)

    c = sub.add_parser("prepare-c", help="шард C: The Stack, ~3B токенов")
    c.add_argument("--target-tokens", type=parse_tokens, default=stack.DEFAULT_TARGET_TOKENS)
    c.add_argument("--source", default=stack.DEFAULT_SOURCE_NAME,
                   help=f"источник из реестра или auto; есть: {sorted(stack.SOURCES)}")
    c.add_argument("--prefer", default=None, help="источник, который пробовать первым при --source auto")
    c.add_argument("--languages", nargs="+", default=list(stack.LANGUAGES))
    add_shard_args(c, os.path.join(common.DATASET_ROOT, stack.SHARD))
    c.set_defaults(func=cmd_prepare_c)

    probe = sub.add_parser("probe", help="проба на малом объёме в /tmp (без боевой загрузки)")
    probe.add_argument("--limit-mb", type=float, default=200.0, help="бюджет выхода на шард, МБ")
    probe.add_argument("--out", default=PROBE_ROOT, help=f"каталог пробы (по умолчанию {PROBE_ROOT})")
    probe.add_argument("--source-w", default="hf:HuggingFaceFW/fineweb-edu:sample-100BT")
    probe.add_argument("--source-c", default="auto", help="источник кода или auto (перебор реестра)")
    probe.add_argument("--prefer", default=None, help="источник кода, который пробовать первым при auto")
    probe.add_argument("--languages", nargs="+", default=list(stack.LANGUAGES))
    probe.add_argument("--shard-mb", type=float, default=100.0)
    probe.add_argument("--codec", choices=sorted(common.CODEC_EXTENSIONS), default=common.DEFAULT_CODEC)
    probe.add_argument("--level", type=int, default=3)
    probe.add_argument("--dedup-window", type=int, default=common.DEFAULT_DEDUP_WINDOW)
    probe.add_argument("--progress", action="store_true")
    probe.add_argument("--allow-any-out", action="store_true")
    probe.set_defaults(func=cmd_probe)

    src = sub.add_parser("sources", help="доступность источников кода (свидетельство v2/v1)")
    src.add_argument("--languages", nargs="+", default=list(stack.LANGUAGES))
    src.add_argument("--report", default=None, help="куда записать JSON-отчёт проверки")
    src.set_defaults(func=cmd_sources)

    vm = sub.add_parser("verify-manifest", help="пересчитать sha256 шардов по манифесту")
    vm.add_argument("--manifest", required=True)
    vm.add_argument("--out", default=None, help="каталог шардов (по умолчанию — каталог манифеста)")
    vm.set_defaults(func=cmd_verify_manifest)
    return parser


def main(argv: list[str] | None = None) -> int:
    common.sanitize_proxy_env()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
