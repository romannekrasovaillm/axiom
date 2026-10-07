"""S-012 — обёртка метрик претрейна ``tools/pretrain_run.py`` (дельта C3).

Читает jsonl-метрики шагов и журнал прогона и переводит в факты:
``tok_s_median_window``, ``step_seconds_median``, ``loss_median_window``,
``loss_slope_window`` и ``mfu_declared_peak`` — последний **явно** назван
объявленным пиком (``peak_declared_not_measured``), а не измеренным MFU.
Предмет — ``run_ref``/``config_sha``/чекпойнт из журнала. Файла метрик нет →
``unverified`` (числа не выдумываются).

Запуск::

    python3 -m tools.sensors.wrap_pretrain_metrics [--metrics PATH] [--journal PATH] [--out-dir DIR]
    python3 -m tools.sensors.wrap_pretrain_metrics --selftest
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

from ._common import REPO_ROOT, config_subject, emit
from ._wrap import median, read_json, read_jsonl

METHOD = "обёртка: чтение metrics.jsonl и run-journal.json (tools/pretrain_run.py)"
WINDOW = 20


def _slope(values: list[float]) -> Optional[float]:
    """Наклон линейной регрессии на окне (loss_slope_window)."""
    n = len(values)
    if n < 2:
        return None
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(values) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, values)) / denom


def measure(
    metrics_path: Optional[str | Path] = None,
    journal_path: Optional[str | Path] = None,
    *,
    window: int = WINDOW,
    out_dir: Optional[str | Path] = None,
    device: Optional[str] = None,
) -> dict[str, Any]:
    rows = read_jsonl(metrics_path) if metrics_path else []
    journal = read_json(journal_path) if journal_path else None
    run_ref = None
    config_sha = None
    if isinstance(journal, dict):
        run_ref = journal.get("run_ref")
        config_sha = (journal.get("config") or {}).get("sha256") if isinstance(journal.get("config"), dict) else None
    subject = config_subject(
        config_path=REPO_ROOT / "net" / "config.json" if not config_sha else None,
        run_ref=run_ref, dataset_ref=None, device=device,
    )
    written: dict[str, Any] = {}

    tail = rows[-window:] if rows else []
    tok_s = [float(r["tok_s"]) for r in tail if isinstance(r.get("tok_s"), (int, float))]
    step_s = [
        float(r[key]) for r in tail for key in ("step_seconds", "sec_per_step", "step_time_s")
        if isinstance(r.get(key), (int, float))
    ]
    losses = [float(r["loss"]) for r in rows] if rows else []
    loss_win = losses[-window:]

    def _emit(name: str, value: Any, unit: str) -> None:
        written[name] = emit(
            "S-012", name, value, unit=unit, quality="wrapped", method=METHOD,
            subject=subject, out_dir=out_dir,
            status="ok" if value is not None else "unverified",
            note="" if value is not None else "источник метрик недоступен или поле отсутствует",
        )

    _emit("tok_s_median_window", median(tok_s), "tok_s")
    _emit("step_seconds_median", median(step_s), "seconds")
    _emit("loss_median_window", median(loss_win), "loss")
    _emit("loss_slope_window", _slope(loss_win), "loss/step")

    mfu = None
    if isinstance(journal, dict):
        mfu = journal.get("mfu_median")
        if mfu is None:
            mfu_values = [float(r["mfu"]) for r in rows if isinstance(r.get("mfu"), (int, float))]
            mfu = median(mfu_values)
    written["mfu_declared_peak"] = emit(
        "S-012", "mfu_declared_peak", mfu, unit="fraction", quality="wrapped",
        method=METHOD + "; доля от ОБЪЯВЛЕННОГО пика (peak_declared_not_measured)",
        subject=subject, out_dir=out_dir,
        status="ok" if mfu is not None else "unverified",
        note=(
            "mfu от объявленного пика (peak_declared_not_measured) — НЕ измеренный MFU"
            if mfu is not None else "нет журнала/поля mfu_median"
        ),
    )
    return written


def run_selftest() -> int:
    import json
    import tempfile

    checks: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory(prefix="s012-selftest-") as tmp:
        metrics = Path(tmp) / "metrics.jsonl"
        rows = [{"step": i, "tok_s": 100.0 + i, "loss": 8.0 - 0.1 * i, "step_seconds": 2.0} for i in range(30)]
        metrics.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        journal = Path(tmp) / "journal.json"
        journal.write_text(json.dumps({"run_ref": "x", "mfu_median": 0.35}), encoding="utf-8")
        written = measure(metrics, journal, out_dir=tmp)
        checks.append(("tok_s медиана посчитана", written["tok_s_median_window"]["value"] is not None))
        checks.append(("loss slope отрицательный (loss падает)", written["loss_slope_window"]["value"] < 0))
        checks.append(("mfu_declared_peak от объявленного пика",
                       written["mfu_declared_peak"]["value"] == 0.35
                       and "объявленн" in written["mfu_declared_peak"]["note"]))
        missing = measure(None, None, out_dir=tmp)
        checks.append(("нет источника → unverified",
                       missing["tok_s_median_window"]["status"] == "unverified"))
    ok = all(p for _, p in checks)
    for label, passed in checks:
        print(f"[selftest] {'PASS' if passed else 'FAIL'}: {label}")
    print(f"[selftest] {'PASS' if ok else 'FAIL'}: wrap_pretrain_metrics")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="S-012: обёртка метрик претрейна (ADR-036)")
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--journal", default=None)
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--selftest", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.selftest:
        return run_selftest()
    written = measure(args.metrics, args.journal, window=args.window, out_dir=args.out_dir, device=args.device)
    for name, rec in written.items():
        print(f"S-012 {name} = {rec['value']} [{rec['status']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
