#!/usr/bin/env python3
"""ADR-048 Amendment — CPU-смоук: jit-шаг оптимизатора против eager.

Зачем
-----
После переписывания KDA (ADR-047) шаг оптимизатора стал доминантой шага
(``sec_backopt`` = 20.4 с из ~31 с).  Уточнение Amendment: это **хостовой**
overhead, а не арифметика — NS-работа всего шага порядка 3.6e13 FLOP (десятки
мс), а шаг исполнялся вне ``jax.jit`` и проходил по дереву дважды.  Дельта
«jit + один проход» — рычаг; этот прибор снимает её CPU-числа на **малой**
геометрии (GPU-время и loss-динамику даёт прогон на GB10, архитектор).

Как
---
Одна и та же малая модель (малый словарь намеренно: измеряется хостовой
overhead на листьях дерева, а не пропускная способность памяти), один сид.
Сравниваются четыре конфигурации шага, доступные из ``make_step``:

* ``eager_two_pass`` — пре-Amendment путь (два ``tree_map_with_path``);
* ``eager_one_pass`` — один проход, без jit;
* ``jit_one_pass``   — штатный путь раннера (дефолт ``make_step``);
* ``jit_two_pass``   — jit без выигрыша одного прохода (контроль).

Проверяется паритет числом (eager == jit, один проход == два прохода), а не
взглядом: несовпадение — ``passed: false`` в отчёте.

Использование::

    python3 tools/optimizer_jit_smoke.py                # пишет evidence/kda-rewrite/optimizer-jit-smoke.json
    python3 tools/optimizer_jit_smoke.py --print-only   # только stdout
    python3 tools/optimizer_jit_smoke.py --repeats 30 --config net/config-proto-micro.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "tools"))

DEFAULT_OUT = _REPO_ROOT / "evidence" / "kda-rewrite" / "optimizer-jit-smoke.json"
SCHEMA = "axiom/optimizer-jit-smoke/1"


def _preflight_memory() -> None:
    """ADR-041: лимит памяти XLA — ДО импорта jax (та же дисциплина, что у
    ``tools/optimizer_groups.py``: импорт jax создаёт клиента и резервирует
    память, без лимита совмещённый стенд уходит в global OOM)."""
    import jax_preflight

    jax_preflight.ensure_mem_fraction()


def _smoke_config(config: Path | None):
    """Малая геометрия смоука (4 слоя — минимальная целая [K,K,K,M]-раскладка)."""
    from net.config import ModelConfig, load_config

    if config is not None:
        return load_config(config)
    # Матрицы крошечные, словарь малый: стоимость шага здесь — хостовой обход
    # дерева и число листьев (138 — ровно та геометрия, что в смоуке Amendment),
    # а не арифметика NS.
    return replace(
        ModelConfig(),
        vocab_size=1024,
        hidden=64,
        num_layers=4,
        num_kda_layers=3,
        num_mla_layers=1,
        num_heads=4,
        head_dim=16,
        kda_dk=16,
        kda_dv=16,
        kda_decay_rank=16,
        mla_latent_dim=32,
        mla_head_dim=16,
        mlp_intermediate=128,
        moe_latent_dim=32,
        moe_num_routed=4,
        moe_num_shared=1,
        moe_top_k=2,
        moe_expert_intermediate=16,
        moe_shared_intermediate=32,
        vit_hidden=32,
        vit_depth=2,
        vit_heads=2,
        vit_mlp=64,
    )


def _inputs(cfg):
    """(params, grads, state) — один сид, детерминированные входы."""
    import jax
    import jax.random as jr

    from net import model, optimizer

    params = model.init_params(jr.PRNGKey(0), cfg)
    grads = jax.tree_util.tree_map(
        lambda leaf: jr.normal(jr.PRNGKey(int(leaf.size) + 1), leaf.shape), params
    )
    state = optimizer.init_state(params)
    return params, grads, state


def _time_step(step, params, grads, state, lr, repeats: int) -> dict:
    """Медиана и минимум времени шага; первая компиляция — отдельным числом."""
    t0 = time.perf_counter()
    step(params, grads, state, lr)
    compile_s = time.perf_counter() - t0
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        step(params, grads, state, lr)
        samples.append((time.perf_counter() - t0) * 1e3)
    return {
        "compile_or_warmup_ms": compile_s * 1e3,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "repeats": repeats,
    }


def _tree_equal(a, b, *, rtol: float = 0.0, atol: float = 0.0) -> bool:
    import jax
    import jax.numpy as jnp

    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    if len(la) != len(lb):
        return False
    return all(
        bool(jnp.allclose(x, y, rtol=rtol, atol=atol)) for x, y in zip(la, lb)
    )


def build_artifact(cfg, *, repeats: int) -> dict:
    import jax

    from net import optimizer

    params, grads, state = _inputs(cfg)
    lr = 1e-3
    leaf_count = len(jax.tree_util.tree_leaves(params))

    steps = {
        "eager_two_pass": optimizer.make_step(cfg, jit=False, single_pass=False),
        "eager_one_pass": optimizer.make_step(cfg, jit=False, single_pass=True),
        "jit_one_pass": optimizer.make_step(cfg, jit=True, single_pass=True),
        "jit_two_pass": optimizer.make_step(cfg, jit=True, single_pass=False),
    }
    timings = {
        name: _time_step(step, params, grads, state, lr, repeats)
        for name, step in steps.items()
    }

    # --- паритет: eager ↔ jit и один проход ↔ два прохода -------------------
    eager1 = steps["eager_one_pass"](params, grads, state, lr)
    eager2 = steps["eager_two_pass"](params, grads, state, lr)
    jit1 = steps["jit_one_pass"](params, grads, state, lr)
    parity = {
        "one_pass_vs_two_pass_eager_exact": _tree_equal(eager1, eager2),
        "one_pass_vs_two_pass_jit_exact": _tree_equal(jit1, steps["jit_two_pass"](params, grads, state, lr)),
        "jit_vs_eager_one_pass": _tree_equal(jit1, eager1, rtol=1e-6, atol=1e-7),
        "state_structure_preserved": jax.tree_util.tree_structure(jit1[1])
        == jax.tree_util.tree_structure(state),
    }

    # --- lr динамический: смена lr не перекомпилирует -----------------------
    jitted = steps["jit_one_pass"]
    jitted(params, grads, state, 1e-3)
    cache_after_first = jitted._cache_size()
    jitted(params, grads, state, 5e-3)  # другой lr, те же формы
    cache_after_lr_change = jitted._cache_size()

    ratios = {
        "eager_two_pass_over_jit_one_pass": timings["eager_two_pass"]["median_ms"]
        / timings["jit_one_pass"]["median_ms"],
        "eager_two_pass_over_eager_one_pass": timings["eager_two_pass"]["median_ms"]
        / timings["eager_one_pass"]["median_ms"],
    }
    passed = all(parity.values()) and cache_after_lr_change == cache_after_first
    return {
        "schema": SCHEMA,
        "adr": "ADR-048 Amendment (08.10.2026)",
        "passed": bool(passed),
        "backend": [str(d) for d in jax.devices()],
        "config_geometry": {
            "vocab_size": cfg.vocab_size,
            "hidden": cfg.hidden,
            "num_layers": cfg.num_layers,
            "num_heads": cfg.num_heads,
            "head_dim": cfg.head_dim,
        },
        "leaf_count": leaf_count,
        "lr": lr,
        "timings": timings,
        "ratios": ratios,
        "compilations": {
            "cache_after_first_call": cache_after_first,
            "cache_after_lr_change": cache_after_lr_change,
            "recompiled_on_lr_change": cache_after_lr_change != cache_after_first,
        },
        "parity": parity,
        "caveat": (
            "CPU-числа демонстрируют устранение хостового overhead (jit + один "
            "проход по дереву), а не время шага на GB10 и не loss-динамику: их "
            "дают прогоны на GB10 (режимы default vs --legacy-muon-all-2d)."
        ),
    }


def format_report(artifact: dict) -> str:
    t = artifact["timings"]
    lines = [
        "| конфигурация шага | медиана, мс | мин, мс | первая (компиляция), мс |",
        "|---|---:|---:|---:|",
    ]
    for name in ("eager_two_pass", "eager_one_pass", "jit_one_pass", "jit_two_pass"):
        row = t[name]
        lines.append(
            f"| {name} | {row['median_ms']:.3f} | {row['min_ms']:.3f} | "
            f"{row['compile_or_warmup_ms']:.0f} |"
        )
    lines.append("")
    lines.append(
        f"листьев в дереве: {artifact['leaf_count']}; "
        f"ускорение eager_two_pass / jit_one_pass = "
        f"×{artifact['ratios']['eager_two_pass_over_jit_one_pass']:.1f}"
    )
    lines.append(
        f"компиляций после смены lr: {artifact['compilations']['cache_after_lr_change']} "
        f"(перекомпиляция: {artifact['compilations']['recompiled_on_lr_change']})"
    )
    parity = ", ".join(f"{k}={v}" for k, v in artifact["parity"].items())
    lines.append(f"паритет: {parity}")
    lines.append(f"VERDICT: {'PASS' if artifact['passed'] else 'FAIL'}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    _preflight_memory()

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=None,
                        help="конфиг модели (по умолчанию — малая геометрия смоука)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="путь отчёта (по умолчанию evidence/kda-rewrite/optimizer-jit-smoke.json)")
    parser.add_argument("--repeats", type=int, default=20,
                        help="число замеров на конфигурацию (медиана)")
    parser.add_argument("--print-only", action="store_true",
                        help="только напечатать, ничего не писать на диск")
    args = parser.parse_args(argv)

    cfg = _smoke_config(args.config)
    artifact = build_artifact(cfg, repeats=max(1, int(args.repeats)))
    print(format_report(artifact))
    if not args.print_only:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\n[optimizer-jit-smoke] отчёт: {args.out}")
    return 0 if artifact["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
