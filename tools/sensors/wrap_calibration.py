"""S-015 — обёртка калибровки Track-2 ``env/calibrate.py`` (дельта C3).

Читает отчёт калибровки (``evidence/track2-stage-a-*/calibration*.json`` или
``evidence/calibration-*.json``): ``pass_rate`` по уровням, sha256 чекпойнта (или
имя внешней модели) и список ``defects``. Отчёта нет → ``unverified``.

Запуск::

    python3 -m tools.sensors.wrap_calibration [--report PATH] [--out-dir DIR]
    python3 -m tools.sensors.wrap_calibration --selftest
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

from ._common import REPO_ROOT, config_subject, emit
from ._wrap import read_json

METHOD = "обёртка: отчёт env/calibrate.py (calibration*.json)"


def _candidates() -> list[Path]:
    """Отчёты калибровки в порядке приоритета (track-2 раньше стаба)."""
    track2 = sorted(REPO_ROOT.glob("evidence/track2-stage-a-*/calibration*.json"))
    if track2:
        return track2
    generic = [
        p
        for p in sorted(REPO_ROOT.glob("evidence/calibration-*.json"))
        if "stub" not in p.name
    ]
    return generic


def _pass_rate(data: dict[str, Any]) -> Any:
    metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
    value: dict[str, Any] = {}
    for key, label in (
        ("pass_rate_L0", "L0"),
        ("pass_rate_L1", "L1"),
        ("pass_rate_L2", "L2"),
        ("pass_rate_L3", "L3"),
        ("aggregate_pass_rate", "aggregate"),
    ):
        if key in metrics:
            value[label] = metrics[key]
    return value or None


def measure(
    report_path: Optional[str | Path] = None,
    *,
    out_dir: Optional[str | Path] = None,
    device: Optional[str] = None,
) -> dict[str, Any]:
    path = Path(report_path) if report_path else None
    if path is None:
        candidates = _candidates()
        path = candidates[-1] if candidates else None
    data = read_json(path) if path else None

    pass_rate = _pass_rate(data) if isinstance(data, dict) else None
    checkpoint_sha = None
    model_name = None
    defects: Any = None
    if isinstance(data, dict):
        subject = data.get("subject_of_measurement") if isinstance(data.get("subject_of_measurement"), dict) else {}
        checkpoint_sha = subject.get("checkpoint_sha256") or subject.get("gguf_sha256")
        model_name = subject.get("model_name_in_report") or subject.get("served_model_in_request")
        raw_defects = data.get("defects")
        if isinstance(raw_defects, list):
            defects = [d.get("id") if isinstance(d, dict) else str(d) for d in raw_defects]

    subject_pin = config_subject(
        checkpoint_path=None, device=device, run_ref=str(path) if path else None
    )
    written: dict[str, Any] = {}
    written["pass_rate"] = emit(
        "S-015", "pass_rate", pass_rate, unit="fraction", quality="wrapped",
        method=METHOD, subject=subject_pin, out_dir=out_dir,
        status="ok" if pass_rate is not None else "unverified",
        note="" if pass_rate is not None else "нет отчёта калибровки",
    )
    written["model_checkpoint_sha256"] = emit(
        "S-015", "model_checkpoint_sha256", checkpoint_sha, unit="sha256", quality="wrapped",
        method=METHOD + f"; модель={model_name!r}", subject=subject_pin, out_dir=out_dir,
        status="ok" if checkpoint_sha else "unverified",
        note="" if checkpoint_sha else f"чекпойнт не запинен; внешняя модель={model_name!r}",
    )
    written["defects"] = emit(
        "S-015", "defects", defects, unit="list", quality="wrapped",
        method=METHOD, subject=subject_pin, out_dir=out_dir,
        status="ok" if defects is not None else "unverified",
        note="" if defects is not None else "нет отчёта калибровки",
    )
    return written


def run_selftest() -> int:
    import json
    import tempfile

    checks: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory(prefix="s015-selftest-") as tmp:
        report = Path(tmp) / "calibration.json"
        report.write_text(
            json.dumps({
                "metrics": {"pass_rate_L0": 0.5, "aggregate_pass_rate": 0.5},
                "subject_of_measurement": {"gguf_sha256": "ab" * 32, "model_name_in_report": "ext"},
                "defects": [{"id": "DEF-1"}],
            }),
            encoding="utf-8",
        )
        written = measure(report, out_dir=tmp)
        checks.append(("pass_rate по уровням", written["pass_rate"]["value"]["L0"] == 0.5))
        checks.append(("чекпойнт запинен", written["model_checkpoint_sha256"]["value"] == "ab" * 32))
        checks.append(("дефекты перечислены", written["defects"]["value"] == ["DEF-1"]))
        missing = measure(Path(tmp) / "nope.json", out_dir=tmp)
        checks.append(("нет отчёта → unverified", missing["pass_rate"]["status"] == "unverified"))
    ok = all(p for _, p in checks)
    for label, passed in checks:
        print(f"[selftest] {'PASS' if passed else 'FAIL'}: {label}")
    print(f"[selftest] {'PASS' if ok else 'FAIL'}: wrap_calibration")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="S-015: обёртка калибровки (ADR-036)")
    parser.add_argument("--report", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--selftest", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.selftest:
        return run_selftest()
    written = measure(args.report, out_dir=args.out_dir, device=args.device)
    for name, rec in written.items():
        print(f"S-015 {name} = {rec['value']} [{rec['status']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
