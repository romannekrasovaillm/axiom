#!/usr/bin/env python3
"""ADR-049 — воспроизводимый A/B-прогон политик рематериализации.

Зачем
-----
Матрица политик на GB10 (ADR-049 Amendment) — ``none`` 407.2 ток/с,
``dots_with_no_batch_dims_saveable`` **468.6** (лучшая рабочая точка),
``dots_saveable`` — OOM (аллокация 106.94 ГиБ).  Чтобы матрицу можно было
*повторить*, а не пересказать, нужен один воспроизводимый A/B: две ноги, у
которых **совпадает всё, кроме политики**, равный бюджет и журнал, по которому
сравнение читается машинально.

Что делает
----------
* ``plan`` — собирает две команды раннера (``tools/pretrain_run.py`` и любой
  другой, принимающий ``--remat-policy``) из одной базовой.  Ноги обязаны
  отличаться **ровно** одним аргументом ``--remat-policy``: шаги, токены, сид и
  данные — из одной базы, политика — единственная переменная.  Это и есть
  «равный бюджет» на уровне плана: база не дублируется, она общая.
* ``compare`` — сводит покадровые журналы двух ног (``metrics.jsonl``) в один
  A/B-журнал: KPI (медиана ток/с) **без первого шага** — первый шаг компилирует
  jit-граф и в KPI не входит (``first_step_excluded``, ADR-048 Amendment), —
  дельту кандидата к базовой ноге и признак равного бюджета (число шагов и
  суммарные токены обеих ног).
* ``run`` — печатает (при ``--execute`` — исполняет последовательно) обе
  спланированные ноги.  Прогон на GB10 делает архитектор; по умолчанию
  инструмент ничего не запускает.

Fail-closed
-----------
Политика проверяется **до** любого старта: имя вне ``net.config.REMAT_POLICIES``
и имя, объявленное, но отсутствующее в установленной версии JAX (``hasattr``),
— ошибка (код 2).  Молчаливая подмена похожей политикой вернула бы прогон к
полному пересчёту, и сравнение мерило бы не то, что объявлено (ADR-049,
Consequences).  ``plan`` дополнительно отказывает базе, в которой
``--remat-policy`` уже задан: иначе «единственная переменная» перестала бы ею
быть.

Границы
-------
Инструмент **не** меняет дефолт: базовая нога по умолчанию ``none``, политика
задаётся только явным ``--remat-policy`` у ноги, ``net/config.json`` остаётся
``none``.  GPU-прогоны здесь не запускаются — ``compare`` работает на уже снятых
журналах, ``run`` требует явного ``--execute``.

Использование::

    # что именно запускать (две команды для архитектора)
    python3 tools/remat_policy_ab.py plan \
        --baseline none --candidate dots_with_no_batch_dims_saveable \
        -- python3 -m tools.pretrain_run --run-ref remat-ab --steps 20 ...

    # свести снятые ноги в A/B-журнал (первый шаг вне KPI)
    python3 tools/remat_policy_ab.py compare \
        --baseline-metrics .../mfu-remat-none.jsonl \
        --candidate-metrics .../mfu-remat-dots.jsonl \
        --baseline none --candidate dots_with_no_batch_dims_saveable \
        --out evidence/kda-rewrite/remat-policy-ab.json

Коды возврата: ``0`` — план/свод собран; ``2`` — отказ конфигурации
(неизвестная/недоступная политика, не та база, неравный бюджет).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "tools")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

SCHEMA = "axiom/remat-policy-ab/1"


def _preflight_memory() -> None:
    """ADR-041: лимит памяти XLA — ДО первого импорта jax.

    Инструмент проверяющий (``plan``/``compare`` не запускают устройство), поэтому,
    как ``tools/remat_policy_smoke.py``, несёт только ``ensure_mem_fraction()`` —
    без ``preflight_gate``: контрольный контур не должен падать из-за чужой
    нагрузки.  ``resolve_policy`` импортирует ``net.remat`` (а тот — jax),
    поэтому вызов стоит первой строкой ``main``.
    """
    import jax_preflight

    jax_preflight.ensure_mem_fraction()

#: Аргумент, которым ноги отличаются ровно друг от друга (единственная переменная).
POLICY_FLAG = "--remat-policy"

#: Рабочая пара ADR-049: дефолт против лучшей измеренной точки.
DEFAULT_BASELINE_POLICY = "none"
DEFAULT_CANDIDATE_POLICY = "dots_with_no_batch_dims_saveable"


# ---------------------------------------------------------------------------
# Политика: fail-closed до старта
# ---------------------------------------------------------------------------


def resolve_policy(policy: str) -> str:
    """Проверить объявленность и доступность политики; вернуть её имя.

    Сама проверка живёт в ``net.remat`` (единственная точка «имя → callable»):
    неизвестное имя и имя, отсутствующее в установленной версии JAX, —
    :class:`ValueError`.  Здесь оно остаётся :class:`ValueError`, чтобы
    вызывающий решал про код возврата, а не ловил ``AttributeError``.
    """
    from net.remat import remat_policy_fn

    remat_policy_fn(policy)  # raises ValueError on unknown/unavailable
    return policy


def _split_remainder(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    """Отделить базовую команду (всё после ``--``) от аргументов инструмента."""
    argv = list(argv)
    if "--" in argv:
        index = argv.index("--")
        return argv[:index], argv[index + 1:]
    return argv, []


def _assert_base_is_single_variable(base: Sequence[str]) -> None:
    """База не должна сама задавать политику — иначе переменных две."""
    for token in base:
        if token == POLICY_FLAG or token.startswith(f"{POLICY_FLAG}="):
            raise ValueError(
                f"базовая команда уже задаёт {POLICY_FLAG}: политика обязана "
                f"быть единственной переменной A/B (уберите её из базы)"
            )


def build_legs(
    base: Sequence[str], baseline: str, candidate: str
) -> dict[str, list[str]]:
    """Две команды ног: база + политика.  Отличие — ровно ``POLICY_FLAG``."""
    baseline = resolve_policy(baseline)
    candidate = resolve_policy(candidate)
    if baseline == candidate:
        raise ValueError(
            f"ноги A/B совпадают ({baseline!r}): сравнивать нечего — нужны две "
            f"разные политики"
        )
    _assert_base_is_single_variable(base)
    base = list(base)
    return {
        baseline: [*base, POLICY_FLAG, baseline],
        candidate: [*base, POLICY_FLAG, candidate],
    }


def plan(base: Sequence[str], baseline: str, candidate: str) -> dict[str, Any]:
    """План A/B: две команды + заявка на равный бюджет (общая база)."""
    legs = build_legs(base, baseline, candidate)
    return {
        "schema": SCHEMA,
        "mode": "plan",
        "baseline_policy": baseline,
        "candidate_policy": candidate,
        "single_variable": POLICY_FLAG,
        "base_command": list(base),
        "legs": legs,
        "equal_budget": True,
        "budget_note": (
            "ноги отличаются ровно аргументом --remat-policy: шаги, токены, "
            "сид и данные берутся из одной базовой команды, поэтому бюджет "
            "обеих ног задаётся одним источником"
        ),
        "run_note": (
            "прогон делает архитектор на GB10: каждая нога пишет свой "
            "metrics.jsonl, затем их сводит `compare`"
        ),
    }


# ---------------------------------------------------------------------------
# Журналы ног: KPI без первого шага, проверка равного бюджета
# ---------------------------------------------------------------------------


def load_metrics(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Покадровый журнал ноги (``metrics.jsonl``) → список строк.

    Пустой/отсутствующий журнал — ошибка: сводить «ничего» нельзя (лучше
    отказ, чем зелёное сравнение без данных).
    """
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"журнал ноги пуст: {path}")
    return rows


def leg_kpi(rows: Sequence[dict[str, Any]], *, exclude_first: bool = True) -> dict[str, Any]:
    """KPI ноги: медиана ток/с **без первого шага** (ADR-048 Amendment).

    Первый шаг компилирует jit-граф и в KPI не входит; лоссы считаются по всем
    строкам.  ``exclude_first`` выключается только для сверки с сырым рядом.
    """
    if not rows:
        raise ValueError("нет строк для KPI")
    first_step_excluded = bool(exclude_first and len(rows) > 1)
    kpi_rows = rows[1:] if first_step_excluded else list(rows)
    tps = [float(r["tokens_per_sec"]) for r in kpi_rows if r.get("tokens_per_sec")]
    step_seconds = [float(r["step_seconds"]) for r in kpi_rows if r.get("step_seconds")]
    tokens_per_step = [
        int(r["tokens"]) for r in rows if r.get("tokens") is not None
    ]
    tokens_seen = [int(r["tokens_seen"]) for r in rows if r.get("tokens_seen") is not None]
    losses = [float(r["loss"]) for r in rows if r.get("loss") is not None]
    return {
        "steps": len(rows),
        "first_step_excluded": first_step_excluded,
        "kpi_rows": len(kpi_rows),
        "kpi_tokens_per_sec": statistics.median(tps) if tps else None,
        "step_seconds_median": statistics.median(step_seconds) if step_seconds else None,
        "tokens_per_step": (
            tokens_per_step[0] if tokens_per_step and len(set(tokens_per_step)) == 1
            else (statistics.median(tokens_per_step) if tokens_per_step else None)
        ),
        "tokens_total": tokens_seen[-1] if tokens_seen else None,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
    }


def _same_budget(baseline: dict[str, Any], candidate: dict[str, Any]) -> bool:
    """Равный бюджет: те же шаги и те же суммарные токены у обеих ног."""
    return (
        baseline["steps"] == candidate["steps"]
        and baseline["tokens_total"] == candidate["tokens_total"]
        and baseline["tokens_per_step"] == candidate["tokens_per_step"]
    )


def compare(
    *,
    baseline_policy: str,
    candidate_policy: str,
    baseline_rows: Sequence[dict[str, Any]],
    candidate_rows: Sequence[dict[str, Any]],
    baseline_metrics: str | None = None,
    candidate_metrics: str | None = None,
) -> dict[str, Any]:
    """A/B-журнал: KPI обеих ног (первый шаг вне KPI), дельта, равный бюджет."""
    resolve_policy(baseline_policy)
    resolve_policy(candidate_policy)
    baseline = leg_kpi(baseline_rows)
    candidate = leg_kpi(candidate_rows)
    equal_budget = _same_budget(baseline, candidate)

    base_tps = baseline["kpi_tokens_per_sec"]
    cand_tps = candidate["kpi_tokens_per_sec"]
    if not equal_budget:
        verdict = "budget-mismatch"
    elif base_tps and cand_tps and cand_tps > base_tps:
        verdict = "faster"
    elif base_tps and cand_tps and cand_tps < base_tps:
        verdict = "slower"
    else:
        verdict = "equal"

    return {
        "schema": SCHEMA,
        "mode": "compare",
        "adr": "ADR-049 (09.10.2026)",
        "baseline": {
            "policy": baseline_policy,
            "metrics": baseline_metrics,
            **baseline,
        },
        "candidate": {
            "policy": candidate_policy,
            "metrics": candidate_metrics,
            **candidate,
        },
        "equal_budget": equal_budget,
        "delta": {
            "tokens_per_sec": (
                (cand_tps - base_tps) if (base_tps and cand_tps) else None
            ),
            "tokens_per_sec_ratio": (
                (cand_tps / base_tps) if (base_tps and cand_tps) else None
            ),
            "step_seconds": (
                (candidate["step_seconds_median"] - baseline["step_seconds_median"])
                if (candidate["step_seconds_median"] and baseline["step_seconds_median"])
                else None
            ),
        },
        "verdict": verdict,
        "kpi_note": (
            "KPI — медиана ток/с без первого шага (компиляция jit-графа в KPI не "
            "входит); равный бюджет — совпадение числа шагов и суммарных токенов"
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("mode", choices=("plan", "compare", "run"))
    parser.add_argument("--baseline", default=DEFAULT_BASELINE_POLICY)
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE_POLICY)
    parser.add_argument("--baseline-metrics", default=None,
                        help="metrics.jsonl базовой ноги (для compare/run)")
    parser.add_argument("--candidate-metrics", default=None,
                        help="metrics.jsonl ноги-кандидата (для compare/run)")
    parser.add_argument("--out", default=None, help="путь журнала (JSON)")
    parser.add_argument("--execute", action="store_true",
                        help="run: реально исполнить ноги (иначе только печать)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _preflight_memory()
    head, base = _split_remainder(sys.argv[1:] if argv is None else argv)
    args = _build_parser().parse_args(head)

    try:
        if args.mode == "plan":
            if not base:
                raise ValueError("plan требует базовую команду после `--`")
            artifact = plan(base, args.baseline, args.candidate)
        elif args.mode == "compare":
            if not (args.baseline_metrics and args.candidate_metrics):
                raise ValueError("compare требует --baseline-metrics и --candidate-metrics")
            artifact = compare(
                baseline_policy=args.baseline,
                candidate_policy=args.candidate,
                baseline_rows=load_metrics(args.baseline_metrics),
                candidate_rows=load_metrics(args.candidate_metrics),
                baseline_metrics=args.baseline_metrics,
                candidate_metrics=args.candidate_metrics,
            )
        else:  # run
            if not base:
                raise ValueError("run требует базовую команду после `--`")
            artifact = plan(base, args.baseline, args.candidate)
            for policy, command in artifact["legs"].items():
                rendered = " ".join(command)
                print(f"[{policy}] {rendered}")
                if args.execute:
                    subprocess.run(command, check=True, cwd=str(_REPO_ROOT))
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"[remat-policy-ab] ОТКАЗ: {exc}", file=sys.stderr, flush=True)
        return 2

    print(json.dumps(artifact, ensure_ascii=False, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\n[remat-policy-ab] журнал: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
