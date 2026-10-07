"""S-001 — исполняемый конфиг модели (дельта C2).

Что меряет: фактическую структуру, которую строит ``net/`` по ``net/config.json``
— число слоёв KDA/MLA **по типам блоков параметров**, ``moe_top_k``, число
routed/shared экспертов, ``vocab_size`` по форме эмбеддинга и фактически
выбранный ``kda_impl``. Значения берутся из ``jax.eval_shape(init_params)``
(формы параметров и построенная структура), а не из JSON: JSON — декларация,
факт конфигурации даёт исполнение (ADR-037).

Запуск::

    python3 -m tools.sensors.runtime_config [--config net/config.json] [--out-dir DIR]
    python3 -m tools.sensors.runtime_config --selftest
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

import jax

from ._common import REPO_ROOT, config_subject, emit

METHOD = "jax.eval_shape(init_params) по net/config.json; типы блоков и формы параметров"


def _shapes(config_path: str | Path):
    from net import model as net_model
    from net.config import load_config

    cfg = load_config(config_path)
    shapes = jax.eval_shape(
        lambda key: net_model.init_params(key, cfg), jax.random.PRNGKey(0)
    )
    return cfg, shapes


def measure(config_path: str | Path, *, out_dir: Optional[str | Path] = None,
            device: Optional[str] = None) -> dict[str, Any]:
    cfg, shapes = _shapes(config_path)
    blocks = list(shapes.layers)
    n_kda = sum(1 for b in blocks if hasattr(b.attn, "W_beta"))
    n_mla = sum(1 for b in blocks if hasattr(b.attn, "W_c"))

    routed: Optional[int] = None
    shared: Optional[int] = None
    for block in blocks:
        if hasattr(block.mlp, "expert_g"):
            routed = int(block.mlp.expert_g.shape[0])
            shared = int(block.mlp.shared_g.shape[0])
            break

    vocab = int(shapes.embedding.shape[0])
    kda_impl = getattr(cfg, "kda_impl", None)
    subject = config_subject(config_path=config_path, device=device)

    facts: list[tuple[str, Any, str, str, str]] = [
        ("num_kda_layers", n_kda, "count", "ok", ""),
        ("num_mla_layers", n_mla, "count", "ok", ""),
        ("moe_top_k", int(cfg.moe_top_k), "count", "ok", ""),
        (
            "moe_routed", routed, "count",
            "ok" if routed is not None else "unverified",
            "" if routed is not None else "в построенных блоках нет LatentMoE (нет expert_g)",
        ),
        (
            "moe_shared", shared, "count",
            "ok" if shared is not None else "unverified",
            "" if shared is not None else "в построенных блоках нет LatentMoE (нет shared_g)",
        ),
        ("vocab_size", vocab, "count", "ok", ""),
        (
            "kda_impl_selected", kda_impl, "enum",
            "ok" if isinstance(kda_impl, str) else "unverified",
            "" if isinstance(kda_impl, str) else "поле kda_impl не строковое",
        ),
    ]
    written = {}
    for name, value, unit, status, note in facts:
        written[name] = emit(
            "S-001", name, value, unit=unit, quality="measured", method=METHOD,
            subject=subject, out_dir=out_dir, status=status, note=note,
        )
    return written


def run_selftest() -> int:
    from ._common import load_config_json

    checks: list[tuple[str, bool]] = []
    cfg, shapes = _shapes(REPO_ROOT / "net" / "config.json")
    declared = load_config_json(REPO_ROOT / "net" / "config.json")
    blocks = list(shapes.layers)
    n_kda = sum(1 for b in blocks if hasattr(b.attn, "W_beta"))
    n_mla = sum(1 for b in blocks if hasattr(b.attn, "W_c"))
    checks.append(("число блоков == num_layers", len(blocks) == cfg.num_layers))
    checks.append(("KDA-блоки сходятся с декларацией", n_kda == cfg.num_kda_layers))
    checks.append(("MLA-блоки сходятся с декларацией", n_mla == cfg.num_mla_layers))
    checks.append(("vocab_size == объявленный vocab_size",
                   int(shapes.embedding.shape[0]) == declared.get("vocab_size")))
    moe = next((b for b in blocks if hasattr(b.mlp, "expert_g")), None)
    checks.append(("routed-эксперты найдены", moe is not None and int(moe.mlp.expert_g.shape[0]) == cfg.moe_num_routed))
    checks.append(("shared-эксперты найдены", moe is not None and int(moe.mlp.shared_g.shape[0]) == cfg.moe_num_shared))
    ok = all(p for _, p in checks)
    for label, passed in checks:
        print(f"[selftest] {'PASS' if passed else 'FAIL'}: {label}")
    print(f"[selftest] {'PASS' if ok else 'FAIL'}: runtime_config")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="S-001: исполняемый конфиг модели (ADR-037)")
    parser.add_argument("--config", default=str(REPO_ROOT / "net" / "config.json"))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default=None, help="переопределить device_kind (фикстуры)")
    parser.add_argument("--selftest", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.selftest:
        return run_selftest()
    written = measure(args.config, out_dir=args.out_dir, device=args.device)
    for name, rec in written.items():
        print(f"S-001 {name} = {rec['value']} [{rec['status']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
