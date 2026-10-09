#!/usr/bin/env python3
"""ADR-049 — CPU-смоук-матрица «политика рематериализации → время/память».

Зачем
-----
Раскладка шага l3-full показала 73% времени в backward при обязательном
grad-checkpointing (``--no-grad-checkpointing`` для l3-full запрещён: OOM 956 ГиБ
по активациям).  ADR-049 параметризует **что** сохраняется внутри remat-границы:
``none`` (текущее поведение — пересчитывается всё), ``dots_saveable`` (сохранять
выходы matmul, пересчитывать elementwise-хвост) и
``dots_with_no_batch_dims_saveable`` (то же, но только для matmul без
batch-измерений).

Как
---
Одна малая модель, один сид, три политики, каждая — свой ``jax.jit``
``value_and_grad`` шага.  На каждую снимается:

* медиана/минимум/максимум времени шага после прогрева;
* оценка памяти XLA (``memory_analysis()`` скомпилированного шага: ``peak`` и
  ``temp``) — детерминированная величина «сколько просит компилятор», аналог
  того активационного запроса, который на GB10 упирается в лимит ADR-041;
* число remat-границ и фактический объект политики в трассированном графе
  (механизм, а не объявление);
* паритет с ``none``: loss — побитово, градиенты — ``allclose`` (пересчёт может
  переставить сложение; это свойство XLA, а не ошибка).

Для ``none`` дополнительно сверяется **побитовое** совпадение с конструкцией до
дельты (``jax.checkpoint(fn)`` без аргумента ``policy``, подменённой в модулях-
потребителях) — прямое свидетельство, что дефолт не сдвинулся.

Границы
-------
Это **не** замена прогонам на GB10: числа на CPU говорят, что механика работает
и в какую сторону двигает, а не сколько секунд это даст на l3-full (там другая
геометрия, другой бэкенд и другой лимит памяти).  Матрицу «время/память» на
целевом профиле снимает архитектор тремя прогонами (none / dots_saveable /
dots_with_no_batch_dims_saveable) под префлайт-гейтом ADR-041.

Использование::

    python3 tools/remat_policy_smoke.py               # пишет evidence/kda-rewrite/remat-policy-smoke.json
    python3 tools/remat_policy_smoke.py --print-only  # только stdout
    python3 tools/remat_policy_smoke.py --repeats 30 --seq-len 128 --batch 2

Код возврата: ``0`` — все политики прошли и паритет держится; ``1`` — нет
(расхождение паритета — это сигнал отказа ADR-049, а не «шум»).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "tools")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

DEFAULT_OUT = _REPO_ROOT / "evidence" / "kda-rewrite" / "remat-policy-smoke.json"
SCHEMA = "axiom/remat-policy-smoke/1"


def _preflight_memory() -> None:
    """ADR-041: лимит памяти XLA — ДО импорта jax (та же дисциплина, что у
    ``tools/optimizer_jit_smoke.py``: импорт jax создаёт клиента и резервирует
    память, без лимита совмещённый стенд уходит в global OOM)."""
    import jax_preflight

    jax_preflight.ensure_mem_fraction()


def _smoke_config(config: Path | None):
    """Малая геометрия смоука с **включёнными** remat-границами.

    Переключатели, при которых границы существуют, задаются принудительно
    (``per_layer`` + ``chunked_cc`` + ``kda_chunked_backward``): матрица — про
    политику, а не про то, включён ли remat.  Принуждение записывается в отчёт
    (``switches_forced``), чтобы прогон нельзя было спутать с прогоном конфига.
    """
    from net.config import ModelConfig, load_config

    if config is not None:
        base = load_config(config)
    else:
        # 4 слоя — минимальная целая [K,K,K,M]-раскладка; словарь малый намеренно:
        # меряется remat-механика и её память, а не пропускная способность.
        base = replace(
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
            kda_wyut_chunk=16,
        )
    return replace(
        base,
        grad_ckpt_policy="per_layer",
        kda_impl="chunked_cc",
        kda_chunked_backward=True,
    )


def _inputs(cfg, *, batch: int, seq_len: int):
    """(params, ids) — один сид, детерминированные входы."""
    import jax.random as jr

    from net import model

    key = jr.PRNGKey(0)
    params = model.init_params(key, cfg)
    ids = jr.randint(key, (batch, seq_len), 0, cfg.vocab_size)
    return params, ids


def _digest(tree) -> str:
    import jax
    import jax.numpy as jnp

    h = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(tree):
        h.update(jnp.asarray(leaf).tobytes())
    return h.hexdigest()


def _remat_boundaries(cfg, params, ids) -> tuple[int, list[str]]:
    """(число remat-границ, объекты ``policy`` в графе) — механизм, не надпись."""
    import jax

    from net import model

    closed = jax.make_jaxpr(
        lambda p: model.compute_loss(p, cfg, ids, chunk_size=int(cfg.kda_wyut_chunk))
    )(params)
    policies = [
        repr(eqn.params.get("policy"))
        for eqn in closed.jaxpr.eqns
        if str(eqn.primitive.name).startswith("remat")
    ]
    return len(policies), sorted(set(policies))


def _memory_bytes(compiled) -> dict:
    """Оценка памяти XLA: ``peak`` (аргументы+temp) и ``temp``."""
    try:
        stats = compiled.memory_analysis()
    except Exception as exc:  # noqa: BLE001 — отчёт важнее отсутствия поля
        return {"unavailable": f"{type(exc).__name__}: {exc}"}
    peak = getattr(stats, "peak_memory_in_bytes", None)
    temp = getattr(stats, "temp_size_in_bytes", None)
    argument = getattr(stats, "argument_size_in_bytes", None)
    output = getattr(stats, "output_size_in_bytes", None)
    if peak is None and None not in (temp, argument, output):
        peak = temp + argument + output
    return {
        "peak_bytes": peak,
        "temp_bytes": temp,
        "argument_bytes": argument,
        "output_bytes": output,
    }


def _measure(cfg, params, ids, *, repeats: int) -> tuple[dict, object]:
    """Время (медиана после прогрева) + XLA-память + дайджесты loss/градиентов.

    Возвращает ``(строка отчёта, градиенты)``: градиенты нужны вызывающему для
    паритета между политиками и в JSON не пишутся (там только дайджест).
    """
    import jax
    import jax.numpy as jnp

    from net import model

    def loss_fn(p):
        return model.compute_loss(p, cfg, ids, chunk_size=int(cfg.kda_wyut_chunk))

    step = jax.jit(jax.value_and_grad(loss_fn))
    t0 = time.perf_counter()
    compiled = step.lower(params).compile()
    compile_s = time.perf_counter() - t0
    memory = _memory_bytes(compiled)

    samples: list[float] = []
    value = None
    grads = None
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        value, grads = step(params)
        samples.append((time.perf_counter() - t0) * 1e3)
    row = {
        "compile_ms": compile_s * 1e3,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "repeats": len(samples),
        "xla_memory": memory,
        "loss": float(value),
        "loss_bits": jnp.asarray(value).tobytes().hex(),
        "grad_sha256": _digest(grads),
    }
    return row, grads


def _pre_delta_checkpoint(fn, policy: str = "none"):
    """Конструкция до ADR-049: ``jax.checkpoint`` без аргумента ``policy``."""
    import jax

    del policy
    return jax.checkpoint(fn)


def build_artifact(cfg, *, repeats: int, batch: int, seq_len: int) -> dict:
    import jax

    from net.config import REMAT_POLICIES

    params, ids = _inputs(cfg, batch=batch, seq_len=seq_len)
    rows: dict[str, dict] = {}
    grads_by_policy: dict[str, object] = {}
    for policy in REMAT_POLICIES:
        cfg_p = replace(cfg, remat_policy=policy)
        row, grads = _measure(cfg_p, params, ids, repeats=repeats)
        count, policies = _remat_boundaries(cfg_p, params, ids)
        row["remat_boundaries"] = count
        row["remat_policy_in_graph"] = policies
        # Паритет — против ``none`` (первой в списке): политика не имеет права
        # менять математику, только место активаций.
        reference = rows.get("none")
        if reference is None:
            row["parity_vs_none"] = {
                "loss_bitwise": True,
                "grads_allclose": True,
                "grads_sha_equal": True,
            }
        else:
            ref_grads = grads_by_policy["none"]
            row["parity_vs_none"] = {
                "loss_bitwise": row["loss_bits"] == reference["loss_bits"],
                "grads_allclose": _tree_allclose(grads, ref_grads),
                "grads_sha_equal": row["grad_sha256"] == reference["grad_sha256"],
            }
        rows[policy] = row
        grads_by_policy[policy] = grads

    # --- «none» = прежний граф, побитово ------------------------------------
    pre_delta = _pre_delta_bitwise(cfg, params, ids, rows["none"])

    expected_boundaries = int(cfg.num_layers)
    # Паритет, которого требует ADR-049: loss — побитово (политика не трогает
    # forward), градиенты — численно (пересчёт может переставить сложение; это
    # свойство XLA, а не ошибка — см. «Consequences»: сверка до знака не
    # обещана).  Побитовое равенство градиентов остаётся информационным полем.
    parity_ok = all(
        row["parity_vs_none"]["loss_bitwise"]
        and row["parity_vs_none"]["grads_allclose"]
        for row in rows.values()
    )
    mechanism_ok = all(
        row["remat_boundaries"] == expected_boundaries for row in rows.values()
    )
    return {
        "schema": SCHEMA,
        "adr": "ADR-049 (09.10.2026)",
        "passed": bool(parity_ok and mechanism_ok and pre_delta["bitwise"]),
        "backend": [str(d) for d in jax.devices()],
        "geometry": {
            "vocab_size": cfg.vocab_size,
            "hidden": cfg.hidden,
            "num_layers": cfg.num_layers,
            "batch": batch,
            "seq_len": seq_len,
            "kda_impl": cfg.kda_impl,
            "kda_wyut_chunk": cfg.kda_wyut_chunk,
        },
        "switches_forced": {
            "grad_ckpt_policy": cfg.grad_ckpt_policy,
            "kda_impl": cfg.kda_impl,
            "kda_chunked_backward": cfg.kda_chunked_backward,
        },
        "expected_remat_boundaries": expected_boundaries,
        "policies": rows,
        "pre_delta_bitwise": pre_delta,
        "caveat": (
            "CPU-числа проверяют механику (политика доходит до remat-границ, "
            "память XLA меняется в ожидаемую сторону) и численный паритет, а не "
            "выигрыш на целевом профиле: матрицу «политика → время/память» для "
            "l3-full даёт архитектор тремя прогонами на GB10 (none / "
            "dots_saveable / dots_with_no_batch_dims_saveable) под лимитом "
            "памяти XLA (ADR-041)."
        ),
    }


def _tree_allclose(a, b, *, rtol: float = 2e-2, atol: float = 2e-3) -> bool:
    import jax
    import jax.numpy as jnp

    leaves_a = jax.tree_util.tree_leaves(a)
    leaves_b = jax.tree_util.tree_leaves(b)
    if len(leaves_a) != len(leaves_b):
        return False
    return all(
        bool(jnp.allclose(x, y, rtol=rtol, atol=atol, equal_nan=True))
        for x, y in zip(leaves_a, leaves_b)
    )


def _pre_delta_bitwise(cfg, params, ids, reference: dict) -> dict:
    """Побитовая сверка ``none`` с конструкцией до дельты (ADR-049 п. 3в)."""
    import jax

    from net import kda, model

    cfg_none = replace(cfg, remat_policy="none")
    patched = (model.remat_checkpoint, kda.remat_checkpoint)
    model.remat_checkpoint = _pre_delta_checkpoint  # type: ignore[assignment]
    kda.remat_checkpoint = _pre_delta_checkpoint  # type: ignore[assignment]
    try:
        row, _ = _measure(cfg_none, params, ids, repeats=1)
    finally:
        model.remat_checkpoint, kda.remat_checkpoint = patched  # type: ignore[assignment]
    return {
        "construction": "jax.checkpoint(fn) без аргумента policy (код до ADR-049)",
        "loss_bitwise": row["loss_bits"] == reference["loss_bits"],
        "grads_bitwise": row["grad_sha256"] == reference["grad_sha256"],
        "digest": {
            "loss_bits": row["loss_bits"],
            "grad_sha256": row["grad_sha256"],
        },
        "bitwise": bool(
            row["loss_bits"] == reference["loss_bits"]
            and row["grad_sha256"] == reference["grad_sha256"]
        ),
    }


def format_report(artifact: dict) -> str:
    lines = [
        "| политика | медиана, мс | мин, мс | память XLA: peak | temp | границ | "
        "паритет: loss побитово / град allclose / град побитово |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for policy, row in artifact["policies"].items():
        mem = row["xla_memory"]
        peak = mem.get("peak_bytes")
        temp = mem.get("temp_bytes")
        parity = row["parity_vs_none"]
        lines.append(
            f"| {policy} | {row['median_ms']:.2f} | {row['min_ms']:.2f} | "
            f"{_mib(peak)} | {_mib(temp)} | {row['remat_boundaries']} | "
            f"{parity['loss_bitwise']} / {parity['grads_allclose']} / "
            f"{parity['grads_sha_equal']} |"
        )
    lines.append("")
    lines.append(
        f"геометрия: layers={artifact['geometry']['num_layers']}, "
        f"B×T={artifact['geometry']['batch']}×{artifact['geometry']['seq_len']}, "
        f"impl={artifact['geometry']['kda_impl']}, "
        f"границ ожидается {artifact['expected_remat_boundaries']}"
    )
    pd = artifact["pre_delta_bitwise"]
    lines.append(
        f"none против кода до ADR-049: loss побитово={pd['loss_bitwise']}, "
        f"градиенты побитово={pd['grads_bitwise']}"
    )
    lines.append(f"VERDICT: {'PASS' if artifact['passed'] else 'FAIL'}")
    return "\n".join(lines)


def _mib(value) -> str:
    if not isinstance(value, int):
        return "—"
    return f"{value / 2**20:.1f} МиБ"


def main(argv: list[str] | None = None) -> int:
    _preflight_memory()

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=None,
                        help="конфиг модели (по умолчанию — малая геометрия смоука)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="путь отчёта (по умолчанию evidence/kda-rewrite/remat-policy-smoke.json)")
    parser.add_argument("--repeats", type=int, default=20,
                        help="число замеров на политику (медиана)")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--print-only", action="store_true",
                        help="только напечатать, ничего не писать на диск")
    args = parser.parse_args(argv)

    cfg = _smoke_config(args.config)
    artifact = build_artifact(
        cfg, repeats=max(1, int(args.repeats)),
        batch=max(1, int(args.batch)), seq_len=max(8, int(args.seq_len)),
    )
    print(format_report(artifact))
    if not args.print_only:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\n[remat-policy-smoke] отчёт: {args.out}")
    return 0 if artifact["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
