"""S-042 — признак промаха кампании MFU-55: MFU шага и ядровая занятость (дельта mfu-55).

Два измеряемых факта признака промаха ADR-050:

* ``mfu_step`` — MFU полного шага по базовой конвенции числителя
  ``6·N_active·tokens`` (N_active = 502 399 236 ⇒ 3.014 GFLOP/токен), знаменатель —
  измеренный bf16-пик 98.2 TFLOPS (``evidence/kpi-pins.json:mfu_reference``).
  Лестница гейтов G1–G4: ≥3% (977 ток/с) / ≥10% (3 258) / ≥20% (6 515) /
  ≥55% (17 917).
* ``core_occupancy`` — ядровая занятость на шаге: доля времени шага, занятая
  исполнением ядер (против хост-блокировок), из nsys-профиля стационарной
  фазы. Цель G1 ≥40%; признак промаха — ниже 10%.

Исполняется **только в окне GB10** (AD-7): на исполнителе (CPU) пишется
``unverified`` с причиной; объявленные пороги в факт не подставляются (C-007).
Расчёт mfu_step сам по себе чистый (арифметика), измеренные tok/s и занятость
передаются из прогона кампании флагами ``--tok-s`` / ``--occupancy``.

Запуск::

    python3 -m tools.sensors.mfu_campaign_miss --allow-device --tok-s 532.8 --occupancy 0.051
    python3 -m tools.sensors.mfu_campaign_miss --selftest
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

from ._common import config_subject, emit

METHOD = (
    "mfu_step = 6·N_active·tok/s / 98.2e12 (базовая конвенция ADR-050, "
    "N_active=502399236); core_occupancy — ядра/шаг из nsys-профиля"
)
N_ACTIVE = 502_399_236  # evidence/facts/S-002.jsonl → 3.014 GFLOP/токен
DENOMINATOR_TFLOPS = 98.2  # измеренный bf16-пик (evidence/kpi-pins.json:mfu_reference)
#: Лестница гейтов кампании (ADR-050, evidence/kpi-pins.json:target_55pct.stages).
GATES: tuple[tuple[str, float, int], ...] = (
    ("G1", 0.03, 977),
    ("G2", 0.10, 3258),
    ("G3", 0.20, 6515),
    ("G4", 0.55, 17917),
)


def mfu_from_tok_s(tok_s: Optional[float]) -> Optional[float]:
    """MFU шага (fraction) из ток/с по базовой конвенции; ``None`` при пустом входе."""
    if tok_s is None or float(tok_s) <= 0:
        return None
    return 6.0 * N_ACTIVE * float(tok_s) / (DENOMINATOR_TFLOPS * 1e12)


def measure(
    *,
    allow_device: bool = False,
    tok_s: Optional[float] = None,
    occupancy: Optional[float] = None,
    out_dir: Optional[str | Path] = None,
    device: Optional[str] = None,
) -> dict[str, Any]:
    subject = config_subject(device=device)
    note = (
        "измерение только в окне GB10 (AD-7) — на исполнителе unverified; "
        "объявленные пороги не подставляются (C-007)"
    )
    mfu = None
    occ = None
    if allow_device:
        mfu = mfu_from_tok_s(tok_s)
        occ = float(occupancy) if occupancy is not None else None
        note = "" if (mfu is not None or occ is not None) else "прогон кампании не дал tok/s и занятости"
    return {
        "mfu_step": emit(
            "S-042", "mfu_step", mfu, unit="fraction", quality="measured",
            method=METHOD, subject=subject, out_dir=out_dir,
            status="ok" if mfu is not None else "unverified", note=note,
        ),
        "core_occupancy": emit(
            "S-042", "core_occupancy", occ, unit="fraction", quality="measured",
            method=METHOD, subject=subject, out_dir=out_dir,
            status="ok" if occ is not None else "unverified", note=note,
        ),
    }


def run_selftest() -> int:
    import tempfile

    checks: list[tuple[str, bool]] = [
        ("G1: 977 ток/с ≈ 3%", abs(mfu_from_tok_s(977.0) - 0.03) < 0.002),
        ("G2: 3258 ток/с ≈ 10%", abs(mfu_from_tok_s(3258.0) - 0.10) < 0.002),
        ("G3: 6515 ток/с ≈ 20%", abs(mfu_from_tok_s(6515.0) - 0.20) < 0.002),
        ("G4: 17917 ток/с ≈ 55%", abs(mfu_from_tok_s(17917.0) - 0.55) < 0.002),
        ("пустой вход → None", mfu_from_tok_s(None) is None),
        ("нулевой вход → None", mfu_from_tok_s(0.0) is None),
    ]
    with tempfile.TemporaryDirectory(prefix="s042-selftest-") as tmp:
        written = measure(allow_device=False, out_dir=tmp)
    checks.append(("без окна → unverified (mfu_step)", written["mfu_step"]["status"] == "unverified"))
    checks.append(("без окна → unverified (core_occupancy)", written["core_occupancy"]["status"] == "unverified"))
    checks.append(("объявленный порог не подставлен", written["mfu_step"]["value"] is None))
    with tempfile.TemporaryDirectory(prefix="s042-selftest2-") as tmp:
        measured = measure(allow_device=True, tok_s=977.0, occupancy=0.40, out_dir=tmp)
    checks.append(("в окне: mfu_step ≈ 3% (ok)", measured["mfu_step"]["status"] == "ok"
                   and abs(measured["mfu_step"]["value"] - 0.03) < 0.002))
    checks.append(("в окне: занятость записана (ok)", measured["core_occupancy"]["status"] == "ok"
                   and abs(measured["core_occupancy"]["value"] - 0.40) < 1e-9))
    ok = all(p for _, p in checks)
    for label, passed in checks:
        print(f"[selftest] {'PASS' if passed else 'FAIL'}: {label}")
    print(f"[selftest] {'PASS' if ok else 'FAIL'}: mfu_campaign_miss")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="S-042: признак промаха кампании MFU-55 (ADR-050)")
    parser.add_argument("--allow-device", action="store_true",
                        help="разрешить запись измеренных значений (только в окне GB10)")
    parser.add_argument("--tok-s", type=float, default=None,
                        help="измеренный tok/s полного шага (прогон кампании)")
    parser.add_argument("--occupancy", type=float, default=None,
                        help="измеренная ядровая занятость, fraction 0..1 (nsys-профиль)")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--selftest", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.selftest:
        return run_selftest()
    written = measure(allow_device=args.allow_device, tok_s=args.tok_s,
                      occupancy=args.occupancy, out_dir=args.out_dir, device=args.device)
    for name, rec in written.items():
        print(f"S-042 {name} = {rec['value']} [{rec['status']}] {rec['note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
