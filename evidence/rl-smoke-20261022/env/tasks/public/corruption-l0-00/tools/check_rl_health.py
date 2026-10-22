#!/usr/bin/env python3
"""C-045 / RL-STAGE.delta §8.1 — страж вырождения награды RL-прогона.

Урок Лагуны (LAG-ADR-017/008/011): вырожденную награду видно по первым шагам, и
учить по ней нечему — это не «медленный старт», а сигнал остановки.  Страж
читает ряд метрик прогона и классифицирует каждый по правилу, а не по прозе.

Формат входа
------------
jsonl, одна строка — один шаг::

    {"step": 0, "pass_rate": 0.39, "reward_mean": 0.378,
     "zero_reward_share": 0.51, "adv_nonzero_share": 0.9,
     "entropy_mean": 1.7, "ngram_repeat_max": 3, "clip_frac": 0.04}

Доли принимаются и как дроби (``0.9``), и как проценты (``90``) — страж
нормирует их к ``[0, 1]``.

Правила (§8.1)
--------------
Стоп (вырождение — остановка после ближайшего чекпойнта, не смена алгоритма):

* ``zero_pass_zero_reward`` — ``pass_rate = 0 ∧ reward_mean ≤ 0.05``;
* ``zero_reward_share``     — ``zero_reward_share ≥ 90 %``;
* ``no_group_variance``     — ``adv_nonzero_share = 0`` на всём окне;
* ``entropy_collapse``      — ``entropy_mean < 0.5`` два шага подряд.

Warn (не остановка, но сигнал):

* ``template_collapse`` — ``ngram_repeat_max ≥ 8``;
* ``clip_frac``         — ``clip_frac ≥ 0.2``.

Окно оценки — первые 50 шагов **и** текущее окно (последние 50 шагов): живая
строка лога ≠ свод окна, «не оценён» ≠ «чисто» (LAG-ADR-017).  Страж краснеет,
если сработало любое окно.

Коды возврата
-------------
* ``0`` — стопов нет (возможны warnings, они в отчёте);
* ``1`` — сработал хотя бы один стоп-класс;
* ``2`` — проверить нельзя: нет входных файлов, объявленный путь отсутствует,
  строка нечитаема/без обязательных полей, либо метрик нет вовсе (fail-closed).

Запуск::

    python3 tools/check_rl_health.py --selftest
    python3 tools/check_rl_health.py --input rl_metrics.jsonl --json health.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

#: Схема отчёта.
REPORT_SCHEMA = "axiom-rl-health/1"

EXIT_OK = 0
EXIT_DEGENERATE = 1
EXIT_CANNOT = 2

#: Размер окна оценки (первые 50 шагов и текущее окно).
WINDOW = 50
#: Пороги стоп-классов.
STOP_REWARD_MEAN = 0.05
STOP_ZERO_REWARD_SHARE = 0.90
STOP_ENTROPY = 0.5
STOP_ENTROPY_CONSECUTIVE = 2
#: Пороги warn-классов.
WARN_NGRAM_REPEAT = 8.0
WARN_CLIP_FRAC = 0.2

#: Обязательные числовые поля строки метрик (контракт §8.1).
FIELDS = (
    "step",
    "pass_rate",
    "reward_mean",
    "zero_reward_share",
    "adv_nonzero_share",
    "entropy_mean",
    "ngram_repeat_max",
    "clip_frac",
)
#: Поля-доли, нормируемые к [0, 1] (принимаем и проценты).
SHARE_FIELDS = ("pass_rate", "zero_reward_share", "adv_nonzero_share", "clip_frac")

STOP_CLASSES = (
    "zero_pass_zero_reward",
    "zero_reward_share",
    "no_group_variance",
    "entropy_collapse",
)
WARN_CLASSES = ("template_collapse", "clip_frac")


class InputError(Exception):
    """Вход существует, но нечитаем/непригоден: fail-closed (exit 2)."""


def _as_fraction(value: float) -> float:
    """Доля как дробь ``[0, 1]``; значение > 1 трактуется как процент."""
    return value / 100.0 if value > 1.0 else value


def _iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, float]]]:
    """Строки jsonl: ``(номер, метрики)``; сбой — fail-closed."""
    try:
        with path.open("rb") as handle:
            for lineno, raw in enumerate(handle, start=1):
                if not raw.strip():
                    continue
                try:
                    record = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise InputError(
                        f"{path}:{lineno}: строка не разбирается как JSON: {exc}"
                    ) from exc
                yield lineno, _coerce(path, lineno, record)
    except OSError as exc:
        raise InputError(
            f"{path}: файл не читается ({type(exc).__name__}): {exc}"
        ) from exc


def _coerce(path: Path, lineno: int, record: Any) -> dict[str, float]:
    """Привести строку метрик к контракту §8.1; отсутствие поля — fail-closed."""
    if not isinstance(record, dict):
        raise InputError(f"{path}:{lineno}: строка не является объектом JSON")
    metrics: dict[str, float] = {}
    for field in FIELDS:
        if field not in record or isinstance(record[field], bool):
            raise InputError(f"{path}:{lineno}: нет числового поля «{field}»")
        try:
            value = float(record[field])
        except (TypeError, ValueError) as exc:
            raise InputError(
                f"{path}:{lineno}: поле «{field}» не число: {record[field]!r}"
            ) from exc
        metrics[field] = _as_fraction(value) if field in SHARE_FIELDS else value
    return metrics


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_window(rows: list[dict[str, float]], label: str) -> dict[str, Any]:
    """Оценить окно рядом метрик: сработавшие стоп/warn-классы и числа."""
    pass_rate = _mean([row["pass_rate"] for row in rows])
    reward_mean = _mean([row["reward_mean"] for row in rows])
    zero_reward = _mean([row["zero_reward_share"] for row in rows])
    adv_nonzero = [row["adv_nonzero_share"] for row in rows]
    entropy = [row["entropy_mean"] for row in rows]
    ngram_max = max((row["ngram_repeat_max"] for row in rows), default=0.0)
    clip_frac = _mean([row["clip_frac"] for row in rows])

    entropy_collapse = any(
        entropy[i] < STOP_ENTROPY and entropy[i + 1] < STOP_ENTROPY
        for i in range(len(entropy) - 1)
    )
    stops = {
        "zero_pass_zero_reward": pass_rate <= 0.0 and reward_mean <= STOP_REWARD_MEAN,
        "zero_reward_share": zero_reward >= STOP_ZERO_REWARD_SHARE,
        "no_group_variance": bool(adv_nonzero) and all(v == 0.0 for v in adv_nonzero),
        "entropy_collapse": entropy_collapse,
    }
    warnings = {
        "template_collapse": ngram_max >= WARN_NGRAM_REPEAT,
        "clip_frac": clip_frac >= WARN_CLIP_FRAC,
    }
    return {
        "window": label,
        "steps": len(rows),
        "pass_rate_mean": round(pass_rate, 6),
        "reward_mean": round(reward_mean, 6),
        "zero_reward_share_mean": round(zero_reward, 6),
        "adv_nonzero_share_min": round(min(adv_nonzero), 6) if adv_nonzero else None,
        "entropy_mean_mean": round(_mean(entropy), 6),
        "entropy_min": round(min(entropy), 6) if entropy else None,
        "ngram_repeat_max": round(ngram_max, 6),
        "clip_frac_mean": round(clip_frac, 6),
        "stops": {name: bool(stops[name]) for name in STOP_CLASSES},
        "warnings": {name: bool(warnings[name]) for name in WARN_CLASSES},
    }


def run_check(
    input_paths: Iterable[str | Path],
    *,
    window: int = WINDOW,
) -> tuple[int, dict[str, Any]]:
    """Проверить здоровье ряда метрик RL.  Возвращает ``(код, отчёт)``."""
    paths = [Path(p) for p in input_paths]
    if not paths:
        return EXIT_CANNOT, _cannot("не переданы входные файлы — проверять нечего", paths)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        return EXIT_CANNOT, _cannot(
            "объявленные пути отсутствуют: " + ", ".join(missing), paths
        )

    rows: list[dict[str, float]] = []
    try:
        for path in paths:
            rows.extend(metrics for _lineno, metrics in _iter_jsonl(path))
    except InputError as exc:
        return EXIT_CANNOT, _cannot(f"fail-closed: {exc}", paths)

    if not rows:
        return EXIT_CANNOT, _cannot("во входе нет ни одной строки метрик", paths)

    step = max(window, 1)
    first = rows[:step]
    current = rows[-step:]
    windows = [evaluate_window(first, "first_50"), evaluate_window(current, "current")]

    fired_stops = sorted(
        {
            name
            for window_report in windows
            for name, fired in window_report["stops"].items()
            if fired
        }
    )
    fired_warnings = sorted(
        {
            name
            for window_report in windows
            for name, fired in window_report["warnings"].items()
            if fired
        }
    )

    if fired_stops:
        code, verdict = EXIT_DEGENERATE, "degenerate"
    else:
        code, verdict = EXIT_OK, "ok"

    return code, {
        "schema": REPORT_SCHEMA,
        "verdict": verdict,
        "input_files": [str(path) for path in paths],
        "steps": len(rows),
        "window": step,
        "stops": fired_stops,
        "warnings": fired_warnings,
        "windows": windows,
        "thresholds": {
            "reward_mean": STOP_REWARD_MEAN,
            "zero_reward_share": STOP_ZERO_REWARD_SHARE,
            "entropy": STOP_ENTROPY,
            "entropy_consecutive": STOP_ENTROPY_CONSECUTIVE,
            "ngram_repeat_max": WARN_NGRAM_REPEAT,
            "clip_frac": WARN_CLIP_FRAC,
        },
    }


def _cannot(reason: str, paths: list[Path]) -> dict[str, Any]:
    return {
        "schema": REPORT_SCHEMA,
        "verdict": "cannot-check",
        "reason": reason,
        "input_files": [str(path) for path in paths],
        "steps": 0,
        "stops": [],
        "warnings": [],
        "windows": [],
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _emit(report: dict[str, Any], json_path: str | None, quiet: bool) -> None:
    if json_path:
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(json_path).write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if not quiet:
        print(json.dumps(report, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="C-045/RL-STAGE §8.1: страж вырождения награды RL"
    )
    parser.add_argument("--input", dest="input_paths", action="append", default=[],
                        help="jsonl с метриками шагов; повторяется")
    parser.add_argument("--window", type=int, default=WINDOW,
                        help=f"размер окна оценки (по умолчанию {WINDOW})")
    parser.add_argument("--json", dest="json_path", default=None, help="куда записать отчёт")
    parser.add_argument("--quiet", action="store_true", help="не печатать отчёт в stdout")
    parser.add_argument("--selftest", action="store_true",
                        help="синтетический selftest с мутантами (tmp; реальные данные не трогаются)")
    args = parser.parse_args(argv)

    if args.selftest:
        return run_selftest()

    code, report = run_check(args.input_paths, window=args.window)
    report["exit_code"] = code
    _emit(report, args.json_path, args.quiet)
    return code


# --------------------------------------------------------------------------- #
# Selftest (C-045: мутанты краснеют, здоровый ряд зелёный, warn ≠ стоп)
# --------------------------------------------------------------------------- #


def _healthy_row(step: int, **overrides: float) -> dict[str, float]:
    row = {
        "step": step,
        "pass_rate": 0.39,
        "reward_mean": 0.38,
        "zero_reward_share": 0.51,
        "adv_nonzero_share": 0.90,
        "entropy_mean": 1.70,
        "ngram_repeat_max": 3.0,
        "clip_frac": 0.04,
    }
    row.update(overrides)
    return row


def _write_rows(path: Path, rows: list[dict[str, float]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def run_selftest() -> int:
    """Синтетика в ``tmp``: здоровый ряд зелёный, каждый стоп-класс краснеет."""
    import tempfile

    checks: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory(prefix="rl-health-selftest-") as tmp:
        root = Path(tmp)

        healthy = root / "healthy.jsonl"
        _write_rows(healthy, [_healthy_row(i) for i in range(60)])
        code_h, report_h = run_check([healthy])
        checks.append(("здоровый ряд → exit 0", code_h == EXIT_OK))
        checks.append(
            ("здоровый ряд → стопов и warn нет",
             not report_h["stops"] and not report_h["warnings"])
        )

        for label, overrides in (
            ("zero_pass_zero_reward", {"pass_rate": 0.0, "reward_mean": 0.02}),
            ("zero_reward_share", {"zero_reward_share": 0.95}),
            ("no_group_variance", {"adv_nonzero_share": 0.0}),
        ):
            path = root / f"{label}.jsonl"
            _write_rows(path, [_healthy_row(i, **overrides) for i in range(60)])
            code, report = run_check([path])
            checks.append((f"мутант «{label}» → exit 1", code == EXIT_DEGENERATE))
            checks.append(
                (f"мутант «{label}» → класс {label} сработал",
                 label in report["stops"])
            )

        # Энтропия < 0.5 два шага подряд.
        entropy_rows = [_healthy_row(i) for i in range(60)]
        entropy_rows[10]["entropy_mean"] = 0.30
        entropy_rows[11]["entropy_mean"] = 0.30
        entropy_path = root / "entropy.jsonl"
        _write_rows(entropy_path, entropy_rows)
        code_e, report_e = run_check([entropy_path])
        checks.append(("мутант «энтропия <0.5 ×2» → exit 1", code_e == EXIT_DEGENERATE))
        checks.append(
            ("мутант «энтропия <0.5 ×2» → класс entropy_collapse",
             "entropy_collapse" in report_e["stops"])
        )

        # Одинокая просадка энтропии (не два подряд) — не стоп.
        single = [_healthy_row(i) for i in range(60)]
        single[10]["entropy_mean"] = 0.30
        single_path = root / "entropy_single.jsonl"
        _write_rows(single_path, single)
        checks.append(
            ("одинокая просадка энтропии → не стоп", run_check([single_path])[0] == EXIT_OK)
        )

        # Только warn-классы: зелёный exit 0, но с warnings.
        warn_rows = [
            _healthy_row(i, ngram_repeat_max=9.0, clip_frac=0.25) for i in range(60)
        ]
        warn_path = root / "warn.jsonl"
        _write_rows(warn_path, warn_rows)
        code_w, report_w = run_check([warn_path])
        checks.append(("ряд только с warn → exit 0", code_w == EXIT_OK))
        checks.append(
            ("ряд только с warn → warnings не пусты, стопов нет",
             bool(report_w["warnings"]) and not report_w["stops"])
        )
        checks.append(
            ("ряд только с warn → оба warn-класса названы",
             set(report_w["warnings"]) == {"template_collapse", "clip_frac"})
        )

        # Проценты вместо долей принимаются.
        pct_path = root / "percent.jsonl"
        _write_rows(pct_path, [_healthy_row(i, zero_reward_share=51.0) for i in range(60)])
        checks.append(
            ("доля в процентах → не стоп (51 %% < 90 %%)", run_check([pct_path])[0] == EXIT_OK)
        )

        # Мусорный вход / нет входа / нет файла → exit 2.
        broken = root / "broken.jsonl"
        broken.write_text("не json\n", encoding="utf-8")
        checks.append(("мусорный вход → exit 2", run_check([broken])[0] == EXIT_CANNOT))
        checks.append(("нет входных файлов → exit 2", run_check([])[0] == EXIT_CANNOT))
        checks.append(
            ("отсутствующий путь → exit 2",
             run_check([root / "нет.jsonl"])[0] == EXIT_CANNOT)
        )
        empty = root / "empty.jsonl"
        empty.write_text("\n", encoding="utf-8")
        checks.append(("пустой вход → exit 2", run_check([empty])[0] == EXIT_CANNOT))
        missing_field = root / "missing.jsonl"
        _write_rows(missing_field, [{"step": 0, "pass_rate": 0.4}])
        checks.append(
            ("строка без обязательного поля → exit 2",
             run_check([missing_field])[0] == EXIT_CANNOT)
        )

    ok = all(passed for _, passed in checks)
    for label, passed in checks:
        print(f"[selftest] {'PASS' if passed else 'FAIL'}: {label}")
    print(
        f"[selftest] {'PASS' if ok else 'FAIL'}: вырожденный ряд краснеет по каждому "
        "стоп-классу, здоровый зелёный, warn ≠ стоп, мусор fail-closed"
    )
    return EXIT_OK if ok else EXIT_DEGENERATE


if __name__ == "__main__":
    raise SystemExit(main())
