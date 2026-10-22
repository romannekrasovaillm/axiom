"""CLI подготовки претрейн-датасета L3 (ADR-021): шарды W (веб), C (код), Q (decay).

    python -m prep_pretrain.build prepare-w --target-tokens 17e9
    python -m prep_pretrain.build prepare-c --target-tokens 3e9 --source stack-dedup-v1
    python -m prep_pretrain.build prepare-q --target-tokens 1e9
    python -m prep_pretrain.build probe --limit-mb 200
    python -m prep_pretrain.build sources
    python -m prep_pretrain.build verify-manifest --manifest <путь>

Прогон потоковый и возобновляемый: манифест пишется по мере закрытия шардов,
повторный запуск продолжает с последнего целого шарда (``--restart`` начинает
заново). Отчёты числовые: счётчики, байты, хеши, скорости, пиковый RSS.

Полная загрузка (17B+3B+1B) этим CLI не запускается автоматически — только проба
на малом объёме (``prepare-q --limit-mb 50``); боевой прогон запускается владельцем
отдельно.
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


def probe_limits(args: argparse.Namespace) -> dict:
    """Предохранители пробы (``--limit-mb`` / ``--max-minutes``), если заданы."""
    limits: dict = {}
    if getattr(args, "limit_mb", None):
        limits["max_output_bytes"] = int(args.limit_mb * MB)
    if getattr(args, "max_minutes", None):
        limits["max_seconds"] = args.max_minutes * 60
    return limits


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
            result = stack.check_source(candidate, args.languages, sample=args.check_sample)
            print(f"  проверка источника {candidate}: {describe_source_check(result)}")
            if result["usable"]:
                source_name = candidate
                check = result
                break
        if source_name == "auto":
            print("ни один источник кода не пригоден — см. `build.py sources`", file=sys.stderr)
            return 3
    elif source_name not in stack.SOURCES:
        print(f"неизвестный источник {source_name!r}; доступные: {sorted(stack.SOURCES)}",
              file=sys.stderr)
        return 2

    spec = stack.SOURCES.get(source_name)
    if spec is not None and spec.license_preselected:
        # Выключенный лицензионный фильтр — не деталь реализации, а свойство
        # корпуса: предупреждение печатается всегда, чтобы прогон не выглядел
        # «разрешённым по умолчанию». Распределение лицензий — в отчёте.
        print(
            f"ВНИМАНИЕ: у источника {source_name} лицензионный фильтр выключен "
            f"(license_preselected) — в шард попадут файлы с любыми лицензиями; "
            f"распределение лицензий смотрите в отчёте (rules.licenses_seen)",
            file=sys.stderr,
        )

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


def cmd_prepare_q(args: argparse.Namespace) -> int:
    """Шард Q: сужёный W + код с тестами, дедуп против W/C (decay ADR-021)."""
    code_spec: dict
    if args.source_c == "auto":
        # Для Q годен только публичный источник: гейтед bigcode без выданного
        # доступа роняет поток до первого документа (см. README шарда C).
        public = next(
            (name for name in stack.AUTO_ORDER if not stack.SOURCES[name].gated),
            stack.DEFAULT_SOURCE_NAME,
        )
        code_spec = {"kind": "stack", "name": public}
    elif args.source_c in stack.SOURCES:
        code_spec = {"kind": "stack", "name": args.source_c}
    elif ":" in args.source_c:
        code_spec = common.parse_source_spec(args.source_c)
    else:
        print(f"неизвестный источник кода {args.source_c!r}; доступные: "
              f"{sorted(stack.SOURCES)} либо local:<glob>/hf:<repo>", file=sys.stderr)
        return 2

    prior_dirs = args.prior_dir or [common.DATASET_ROOT]
    if args.no_prior_dedup:
        print("ВНИМАНИЕ: дедуп Q против записей W/C отключён явно (--no-prior-dedup): "
              "отчёт фиксирует prior.enabled=false", file=sys.stderr)

    report = fineweb.prepare_q(
        out_dir=args.out,
        target_tokens=args.target_tokens,
        web_spec=common.parse_source_spec(args.source_w),
        code_spec=code_spec,
        code_share=args.code_share,
        web_yield=args.web_yield,
        code_yield=args.code_yield,
        min_int_score=args.min_int_score or None,
        min_chars=args.min_chars or None,
        max_chars=args.max_chars or None,
        languages=args.languages,
        prior_dirs=prior_dirs,
        prior_max_records=args.prior_max_records,
        prior_threshold=args.prior_threshold,
        prior_enabled=not args.no_prior_dedup,
        manifest_path=args.manifest,
        report_path=args.report,
        **shard_kwargs(args),
        **probe_limits(args),
    )
    echo_report(report, "шард Q (decay: сужёный веб + код с тестами)")
    rules = report["rules"]
    print(f"  веб-фильтр    : int_score >= {rules['components']['web']['min_int_score']}, "
          f"длина {rules['components']['web']['min_chars']}..{rules['components']['web']['max_chars']} символов")
    print(f"  код-фильтр    : {rules['components']['code']['source']}, "
          f"языки {', '.join(rules['components']['code']['languages'])}, "
          f"маркеры тестов {', '.join(rules['components']['code']['test_markers'])}")
    print(f"  причины отбоя : {report['counters'].get('dropped_by_rule', {})}")
    prior = report["prior"]
    if prior.get("enabled"):
        print(f"  опора W/C     : {prior['reference_records']} записей из "
              f"{prior['files_read']} файлов ({prior['chars'] / 1e6:.1f}M символов, "
              f"индекс {prior['seconds']} с), порог Jaccard {prior['threshold']}")
        print(f"  дедуп W/C     : проверено {prior['checked']} кандидатов, "
              f"дублей точных {prior['dropped_exact']} + near {prior['dropped_near']} "
              f"({prior['check_seconds']} с, {prior['check_ms_per_record']} мс/запись)")
    else:
        print("  опора W/C     : ВЫКЛЮЧЕНА (--no-prior-dedup)")
    mix = report["mix"]
    print(f"  микс          : цель кода {mix['code_share_target']:.1%}, "
          f"факт {mix['code_share']:.2%} ({mix['code_share_deviation_pp']:+.2f} п.п.)")
    print(f"                  веб {mix['web']['tokens']} токенов / {mix['web']['records']} записей, "
          f"код {mix['code']['tokens']} токенов / {mix['code']['records']} записей")
    print(f"                  источники исчерпаны: {mix['source_exhausted']}")
    if mix.get("recommended_code_yield"):
        print(f"                  рекомендация: --code-yield {mix['recommended_code_yield']}")
    print(f"  манифест      : {report['manifest']}")
    print(f"  отчёт         : {report['report']}")
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    """Проба на малом объёме: числовая оценка канала и темпа подготовки."""
    root = common.ensure_output_allowed(args.out, allow_any=args.allow_any_out)
    root.mkdir(parents=True, exist_ok=True)
    limit_bytes = int(args.limit_mb * MB)
    shard_bytes = int(args.shard_mb * MB)
    max_seconds = args.max_minutes * 60 if args.max_minutes else None
    payload: dict = {
        "pipeline": common.PIPELINE_VERSION,
        "probe_root": str(root),
        "limit_mb": args.limit_mb,
        "shard_mb": args.shard_mb,
        "max_minutes": args.max_minutes,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    print(f"проба: {args.limit_mb} МБ на шард, шард-файл {args.shard_mb} МБ, каталог {root}")
    try:
        w_report = fineweb.prepare_w(
            out_dir=root / "W",
            target_tokens=0,
            source_spec=common.parse_source_spec(args.source_w),
            max_output_bytes=limit_bytes,
            max_seconds=max_seconds,
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
            result = stack.check_source(candidate, args.languages, sample=args.check_sample)
            checks.append(result)
            print(f"  проверка источника {candidate}: {describe_source_check(result)}")
            if result["usable"]:
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
                max_seconds=max_seconds,
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
        chars_per_token = 4  # ADR-021: approx_tokens = len(text) // 4
        text_rate = report["source_text_mb_per_s"]
        eta = (
            round((target * chars_per_token) / (1024 * 1024) / text_rate / 3600, 3)
            if text_rate > 0
            else None
        )
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
            "eta_hours_to_target": eta,
            "stop_reason": report["stop_reason"],
            "rss_samples": report.get("rss_samples", []),
        }
    return summary


def describe_source_check(result: dict) -> str:
    """Короткая строка о результате проверки источника (для консоли и отчёта)."""
    if not result["ok"]:
        return f"НЕДОСТУПЕН ({result.get('error', '')[:160]})"
    if not result["usable"]:
        return (f"читается, но НЕПРИГОДЕН: {result['kept']}/{result['sample']} "
                f"записей прошли фильтры")
    return (f"пригоден: {result['kept']}/{result['sample']} записей прошли фильтры "
            f"({result['kept_share']:.1%}, {result['seconds']} с)")


def cmd_sources(args: argparse.Namespace) -> int:
    """Проверка доступности источников кода — свидетельство для решения v2/v1."""
    results = [
        stack.check_source(name, args.languages, sample=args.check_sample)
        for name in stack.SOURCES
    ]
    for result in results:
        print(f"{result['source']:<22} gated={str(result['gated']):<5} {describe_source_check(result)}")
        if result.get("kept_share") is not None:
            print(f"{'':<22} языки: {result['languages_seen']}")
            print(f"{'':<22} лицензии: {result['licenses_seen']}")
            print(f"{'':<22} отбой: {result['dropped_by_rule']}")
        print(f"{'':<22} {result['note']}")
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
    c.add_argument("--check-sample", type=int, default=2000,
                   help="сколько записей источника прогнать через фильтры при проверке")
    add_shard_args(c, os.path.join(common.DATASET_ROOT, stack.SHARD))
    c.set_defaults(func=cmd_prepare_c)

    q = sub.add_parser("prepare-q", help="шард Q: decay-микс (сужёный W + код с тестами), ~1B")
    q.add_argument("--target-tokens", type=parse_tokens, default=fineweb.DEFAULT_TARGET_TOKENS_Q)
    q.add_argument("--source-w", default="hf:HuggingFaceFW/fineweb-edu:sample-100BT",
                   help="источник веб-части (тот же, что у шарда W)")
    q.add_argument("--source-c", default="codeparrot-clean",
                   help=f"источник кода: имя из реестра, auto либо local:<glob>; есть: {sorted(stack.SOURCES)}")
    q.add_argument("--languages", nargs="+", default=["python"],
                   help="языки кодовой части (codeparrot-clean одноязычный — python)")
    q.add_argument("--code-share", type=float, default=fineweb.DEFAULT_CODE_SHARE,
                   help="целевая доля кодовых примеров в Q (по approx-токенам, 0.15)")
    q.add_argument("--web-yield", type=float, default=fineweb.DEFAULT_WEB_YIELD,
                   help="ожидаемых approx-токенов на символ веб-источника (калибровка микса)")
    q.add_argument("--code-yield", type=float, default=fineweb.DEFAULT_CODE_YIELD,
                   help="ожидаемых approx-токенов на символ кодового источника (калибровка микса)")
    q.add_argument("--min-int-score", type=int, default=fineweb.DEFAULT_Q_MIN_INT_SCORE,
                   help="сужёный edu-порог веб-части (по умолчанию 4; 0 — порог выключен)")
    q.add_argument("--min-chars", type=int, default=fineweb.DEFAULT_Q_MIN_CHARS,
                   help="нижняя граница длины веб-документа, символов")
    q.add_argument("--max-chars", type=int, default=fineweb.DEFAULT_Q_MAX_CHARS,
                   help="верхняя граница длины веб-документа, символов")
    q.add_argument("--prior-dir", action="append", default=None,
                   help=f"каталог(и) опорных шардов W/C (по умолчанию {common.DATASET_ROOT})")
    q.add_argument("--prior-max-records", type=int, default=fineweb.DEFAULT_PRIOR_MAX_RECORDS,
                   help="сколько записей W/C индексируется как опора near-dup (память ∝ числу)")
    q.add_argument("--prior-threshold", type=float, default=None,
                   help="порог Jaccard near-dup (по умолчанию — порог axiom_ds.dedup, 0.8)")
    q.add_argument("--no-prior-dedup", action="store_true",
                   help="явно отключить дедуп против W/C (в отчёте prior.enabled=false)")
    q.add_argument("--limit-mb", type=float, default=0.0,
                   help="бюджет выхода, МБ (проба: 50); 0 — без ограничения")
    q.add_argument("--max-minutes", type=float, default=0.0,
                   help="предохранитель по времени, минут (0 — без ограничения)")
    add_shard_args(q, os.path.join(common.DATASET_ROOT, fineweb.SHARD_Q))
    q.set_defaults(func=cmd_prepare_q)

    probe = sub.add_parser("probe", help="проба на малом объёме в /tmp (без боевой загрузки)")
    probe.add_argument("--limit-mb", type=float, default=200.0, help="бюджет выхода на шард, МБ")
    probe.add_argument("--out", default=PROBE_ROOT, help=f"каталог пробы (по умолчанию {PROBE_ROOT})")
    probe.add_argument("--source-w", default="hf:HuggingFaceFW/fineweb-edu:sample-100BT")
    probe.add_argument("--source-c", default="auto", help="источник кода или auto (перебор реестра)")
    probe.add_argument("--prefer", default=None, help="источник кода, который пробовать первым при auto")
    probe.add_argument("--languages", nargs="+", default=list(stack.LANGUAGES))
    probe.add_argument("--check-sample", type=int, default=2000,
                       help="сколько записей источника прогнать через фильтры при проверке")
    probe.add_argument("--shard-mb", type=float, default=100.0)
    probe.add_argument("--max-minutes", type=float, default=20.0,
                       help="предохранитель пробы: остановить шард по времени (0 — без ограничения)")
    probe.add_argument("--codec", choices=sorted(common.CODEC_EXTENSIONS), default=common.DEFAULT_CODEC)
    probe.add_argument("--level", type=int, default=3)
    probe.add_argument("--dedup-window", type=int, default=common.DEFAULT_DEDUP_WINDOW)
    probe.add_argument("--progress", action="store_true")
    probe.add_argument("--allow-any-out", action="store_true")
    probe.set_defaults(func=cmd_probe)

    src = sub.add_parser("sources", help="доступность источников кода (свидетельство v2/v1)")
    src.add_argument("--languages", nargs="+", default=list(stack.LANGUAGES))
    src.add_argument("--check-sample", type=int, default=2000,
                     help="сколько записей источника прогнать через фильтры при проверке")
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
