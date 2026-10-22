"""S-003 — байты реплики: параметры и состояние оптимизатора (дельта C2).

Что меряет: точную арифметику по формам и dtype — байты параметров и байты
состояния оптимизатора (Per-Head Muon для матриц, AdamW ``m,v`` для векторов,
``net/optimizer.py:init_state``). Основание для утверждения ADR-034 «реплика
~16 ГБ ≪ 80 ГБ» — число берётся из измеренной структуры, а не из константы.

Запуск::

    python3 -m tools.sensors.replica_bytes [--config net/config.json] [--out-dir DIR]
    python3 -m tools.sensors.replica_bytes --selftest
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Optional

import jax

from ._common import REPO_ROOT, config_subject, emit

METHOD = (
    "точная арифметика по формам и dtype из jax.eval_shape(init_params) и "
    "jax.eval_shape(optimizer.init_state) (Muon: momentum матрицы; AdamW: m,v вектора)"
)


def _bytes(tree: Any) -> int:
    total = 0
    for leaf in jax.tree_util.tree_leaves(tree):
        shape = getattr(leaf, "shape", None)
        dtype = getattr(leaf, "dtype", None)
        if shape is None or dtype is None:
            continue
        total += math.prod(shape) * dtype.itemsize
    return int(total)


def measure(config_path: str | Path, *, out_dir: Optional[str | Path] = None,
            device: Optional[str] = None) -> dict[str, Any]:
    from net import model as net_model
    from net import optimizer as net_optim
    from net.config import load_config

    cfg = load_config(config_path)
    shapes = jax.eval_shape(
        lambda key: net_model.init_params(key, cfg), jax.random.PRNGKey(0)
    )
    params_bytes = _bytes(shapes)
    opt_state = jax.eval_shape(net_optim.init_state, shapes)
    opt_bytes = _bytes(opt_state)
    total = params_bytes + opt_bytes

    subject = config_subject(config_path=config_path, device=device)
    written = {}
    written["replica_bytes_params"] = emit(
        "S-003", "replica_bytes_params", params_bytes, unit="bytes",
        quality="measured", method=METHOD, subject=subject, out_dir=out_dir,
    )
    written["replica_bytes_opt_state"] = emit(
        "S-003", "replica_bytes_opt_state", opt_bytes, unit="bytes",
        quality="measured", method=METHOD, subject=subject, out_dir=out_dir,
    )
    written["replica_bytes_total"] = emit(
        "S-003", "replica_bytes_total", total, unit="bytes",
        quality="measured", method=METHOD, subject=subject, out_dir=out_dir,
    )
    return written


def run_selftest() -> int:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="s003-selftest-") as tmp:
        written = measure(REPO_ROOT / "net" / "config.json", out_dir=tmp)
    params = written["replica_bytes_params"]["value"]
    opt = written["replica_bytes_opt_state"]["value"]
    total = written["replica_bytes_total"]["value"]
    checks: list[tuple[str, bool]] = [
        ("параметры > 0", params > 0),
        ("состояние оптимизатора > 0", opt > 0),
        ("итог = параметры + состояние", total == params + opt),
        ("реплика (параметры+состояние) в коридоре 2–12 ГиБ",
         2 * 1024**3 <= total <= 12 * 1024**3),
    ]
    ok = all(p for _, p in checks)
    for label, passed in checks:
        print(f"[selftest] {'PASS' if passed else 'FAIL'}: {label}")
    print(f"[selftest] {'PASS' if ok else 'FAIL'}: replica_bytes")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="S-003: байты реплики (ADR-037)")
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
        print(f"S-003 {name} = {rec['value']} bytes [{rec['status']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
