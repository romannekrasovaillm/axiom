"""S-002 — число параметров модели (дельта C2).

Что меряет: ``params_total`` (все параметры по формам) и ``params_active``
(активные на токен) через ``jax.eval_shape``; сверяет с объявленными
``actual_param_count`` / ``active_params_per_token`` из ``net/config.json`` и
пишет расхождение в ``note`` (декларация против факта — предмет этого датчика).

Запуск::

    python3 -m tools.sensors.param_count [--config net/config.json] [--out-dir DIR]
    python3 -m tools.sensors.param_count --selftest
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

from ._common import REPO_ROOT, config_subject, emit

METHOD = "jax.eval_shape(init_params) по net/config.json (формы без аллокации)"


def measure(config_path: str | Path, *, out_dir: Optional[str | Path] = None,
            device: Optional[str] = None) -> dict[str, Any]:
    from net import model as net_model
    from net.config import load_config

    from ._common import load_config_json

    cfg = load_config(config_path)
    declared = load_config_json(config_path)
    total = int(net_model.param_count(cfg))
    active = int(net_model.active_param_count(cfg))

    declared_total = declared.get("actual_param_count")
    declared_active = declared.get("active_params_per_token")
    notes = []
    if declared_total is not None and int(declared_total) != total:
        notes.append(f"params_total {total} ≠ declared actual_param_count {declared_total}")
    if declared_active is not None and int(declared_active) != active:
        notes.append(f"params_active {active} ≠ declared active_params_per_token {declared_active}")
    note = "; ".join(notes)

    subject = config_subject(config_path=config_path, device=device)
    written = {}
    written["params_total"] = emit(
        "S-002", "params_total", total, unit="count", quality="measured",
        method=METHOD, subject=subject, out_dir=out_dir, note=note,
    )
    written["params_active"] = emit(
        "S-002", "params_active", active, unit="count", quality="measured",
        method=METHOD, subject=subject, out_dir=out_dir, note=note,
    )
    return written


def run_selftest() -> int:
    import tempfile

    from ._common import load_config_json

    declared = load_config_json(REPO_ROOT / "net" / "config.json")
    with tempfile.TemporaryDirectory(prefix="s002-selftest-") as tmp:
        written = measure(REPO_ROOT / "net" / "config.json", out_dir=tmp)
    checks: list[tuple[str, bool]] = [
        ("params_total > 0", written["params_total"]["value"] > 0),
        ("params_active > 0", written["params_active"]["value"] > 0),
        ("params_active ≤ params_total", written["params_active"]["value"] <= written["params_total"]["value"]),
        ("params_total сходится с декларацией",
         written["params_total"]["value"] == declared.get("actual_param_count")),
        ("params_active сходится с декларацией",
         written["params_active"]["value"] == declared.get("active_params_per_token")),
    ]
    ok = all(p for _, p in checks)
    for label, passed in checks:
        print(f"[selftest] {'PASS' if passed else 'FAIL'}: {label}")
    print(f"[selftest] {'PASS' if ok else 'FAIL'}: param_count")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="S-002: число параметров модели (ADR-036)")
    parser.add_argument("--config", default=str(REPO_ROOT / "net" / "config.json"))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--selftest", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.selftest:
        return run_selftest()
    written = measure(args.config, out_dir=args.out_dir, device=args.device)
    for name, rec in written.items():
        print(f"S-002 {name} = {rec['value']} [{rec['status']}] note={rec['note']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
